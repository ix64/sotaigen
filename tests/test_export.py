"""The exported GLB and the runtime driver, with neither a browser nor Blender.

The GLB is read back by the glTF spec's rules and skinned here, so a mistake in the stored data
shows up without three.js; scripts/runtime_check.py does the same in a real browser. The driver is
JavaScript, and runs under the Node pinned in .node-version, with no npm package involved.
"""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gltf_support import (
    LIMIT_POSES,
    accessor,
    local_matrix,
    primitive,
    rotation_of,
    run_driver,
    skin_stored,
    trs,
)

import sotaigen.output.glb as glb_module
from sotaigen import pipeline, sources
from sotaigen.contact import groin as groin_contact
from sotaigen.contact import hang as hang_contact
from sotaigen.corrective import axilla, groin
from sotaigen.corrective import hang as hang_driver
from sotaigen.corrective.recipe import Correctives
from sotaigen.diagnostics import Diagnostic, as_json, rule
from sotaigen.form.bodies import load_specs
from sotaigen.mesh import skin_normals
from sotaigen.output import quality
from sotaigen.output.face import EXPRESSIONS, face_parts, head_normals
from sotaigen.output.glb import (
    ARRAY_BUFFER,
    COMPONENT,
    MORPH_EPSILON_MM,
    TO_GLTF,
    WEIGHT_STEPS,
    driver_table,
    export_body,
    export_with_face,
    glb_bytes,
    head_attachments,
    joints_and_weights,
    main,
    quaternion,
    read_glb,
    stored_normals,
    to_gltf,
    triangulate,
    vertex_normals,
)
from sotaigen.output.head_interface import eye_place
from sotaigen.output.metadata import file_info, package_version
from sotaigen.output.parts import Material, Part
from sotaigen.params.file import read_parameter_file
from sotaigen.pipeline import BUILD, rigged
from sotaigen.rig.poses import POSES
from sotaigen.rig.skinning import bone_local, euler, pose_verts, rest_matrix, sided
from sotaigen.uv import export_fields

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "web" / "runtime" / "driver.mjs"
STAND_IN = ROOT / "scripts" / "standin_parts.py"
NAMES = list(load_specs("typical"))
NAME = NAMES[0]
# What stored weights and 32-bit floats may cost; the runtime's limit is 1 mm.
STORED_LIMIT_MM = 0.1


@pytest.fixture
def keeps_the_export():
    """Puts NAME's body.glb back afterwards: a test's export isn't what the page should show."""
    path = BUILD / "bodies" / NAME / "body.glb"
    before = path.read_bytes() if path.exists() else None
    yield
    if before is None:
        path.unlink(missing_ok=True)
    else:
        path.write_bytes(before)


def glb_of(name):
    rig = rigged(name)
    return glb_bytes(rig.base, rig.verts, rig.bones, rig.weights, file_info(rig.spec))


def small_rig():
    """Three bones and a strip of quads, for the paths real bodies don't take yet."""
    z = np.array((0.0, -1.0, 0.0))
    bones = [
        {"name": "root", "head": np.zeros(3), "tail": np.array((0.0, 0.0, 100.0)), "parent": None},
        {
            "name": "hips",
            "head": np.array((0.0, 0.0, 100.0)),
            "tail": np.array((0.0, 0.0, 200.0)),
            "parent": "root",
        },
        {
            "name": "spine",
            "head": np.array((0.0, 0.0, 200.0)),
            "tail": np.array((0.0, 0.0, 300.0)),
            "parent": "hips",
        },
    ]
    for bone in bones:
        bone["align"] = z
    verts = np.array(
        [[x, 0.0, h] for h in (100.0, 200.0, 300.0) for x in (-10.0, 10.0)], dtype=float
    )
    weights = np.zeros((6, 3))
    weights[:2, 1] = 1.0
    weights[2:4, 1:] = 0.5
    weights[4:, 2] = 1.0
    base = {"verts": verts.copy(), "faces": [[0, 1, 3, 2], [2, 3, 5, 4]]}
    return base, verts, bones, weights


def test_glb_container_follows_the_spec():
    blob = glb_of(NAME)
    document, data = read_glb(blob)
    assert len(blob) % 4 == 0
    assert document["asset"]["version"] == "2.0"
    assert document["buffers"] == [{"byteLength": len(data)}]
    for view in document["bufferViews"]:
        assert view["byteOffset"] % 4 == 0
        assert view["byteOffset"] + view["byteLength"] <= len(data)


def test_read_glb_refuses_what_is_not_a_glb():
    with pytest.raises(ValueError):
        read_glb(b"\0" * 32)


def test_vertex_order_is_wrapped_order():
    rig = rigged(NAME)
    document, data = read_glb(glb_of(NAME))
    points = accessor(document, data, primitive(document)["attributes"]["POSITION"])
    assert np.array_equal(points, to_gltf(rig.verts).astype(np.float32))


def test_position_bounds_match_the_data():
    document, data = read_glb(glb_of(NAME))
    index = primitive(document)["attributes"]["POSITION"]
    points = accessor(document, data, index)
    assert document["accessors"][index]["min"] == points.min(axis=0).tolist()
    assert document["accessors"][index]["max"] == points.max(axis=0).tolist()


def test_gltf_axes_put_up_on_y_front_on_plus_z_and_the_left_on_plus_x():
    rig = rigged(NAME)
    moved = to_gltf(rig.verts)
    tall = rig.verts[:, 2].argmax()
    assert moved[tall, 1] == pytest.approx(rig.verts[tall, 2] / 1000)
    front = rig.verts[:, 1].argmin()
    assert moved[front, 2] == pytest.approx(-rig.verts[front, 1] / 1000)
    left = rig.verts[:, 0].argmax()
    assert moved[left, 0] == pytest.approx(rig.verts[left, 0] / 1000)


def test_every_preset_shares_one_triangle_list():
    lists = set()
    for name in NAMES:
        document, data = read_glb(glb_of(name))
        indices = accessor(document, data, primitive(document)["indices"])
        lists.add(indices.tobytes())
    assert len(lists) == 1


def test_triangles_cover_every_face_and_keep_its_winding():
    base, verts, _, _ = small_rig()
    triangles = triangulate(base["faces"], base["verts"])
    assert len(triangles) == 4
    normals = vertex_normals(verts, triangles)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)
    assert np.allclose(np.abs(normals[:, 1]), 1.0)
    # A triangle and a pentagon go by fans.
    many = triangulate([[0, 1, 2], [0, 1, 2, 3, 4]], np.zeros((5, 3)))
    assert many.tolist() == [[0, 1, 2], [0, 1, 2], [0, 2, 3], [0, 3, 4]]


def test_a_quad_splits_across_its_shorter_diagonal():
    corners = np.array([[0.0, 0, 0], [10, 0, 0], [11, 1, 0], [1, 1, 0]])
    assert triangulate([[0, 1, 2, 3]], corners).tolist() == [[1, 2, 3], [1, 3, 0]]


def test_normals_point_out_of_the_closed_skin():
    rig = rigged(NAME)
    document, data = read_glb(glb_of(NAME))
    attributes = primitive(document)["attributes"]
    points = accessor(document, data, attributes["POSITION"])
    normals = accessor(document, data, attributes["NORMAL"])
    triangles = accessor(document, data, primitive(document)["indices"]).reshape(-1, 3)
    corners = points[triangles].astype(float)
    crosses = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    # A closed surface with outward windings encloses a positive volume.
    assert np.einsum("ij,ij->", corners[:, 0], crosses) > 0
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
    # The top of the head faces up.
    top = rig.verts[:, 2].argmax()
    assert normals[top, 1] > 0.9


def test_stored_weights_sum_to_one_exactly_and_drop_unused_joints():
    rig = rigged(NAME)
    joints, steps = joints_and_weights(np.asarray(rig.weights))
    assert (steps.astype(int).sum(axis=1) == WEIGHT_STEPS).all()
    assert (joints[steps == 0] == 0).all()
    assert (np.diff(steps.astype(int), axis=1) <= 1).all()
    largest = np.take_along_axis(rig.weights, joints[:, :1].astype(int), axis=1)
    assert np.abs(largest[:, 0] - steps[:, 0] / WEIGHT_STEPS).max() < 2 / WEIGHT_STEPS


def test_stored_weights_refuse_a_fifth_influence():
    with pytest.raises(ValueError, match="four"):
        joints_and_weights(np.full((1, 5), 0.2))


def test_joint_order_is_the_weight_columns_and_parents_come_first():
    rig = rigged(NAME)
    document, _ = read_glb(glb_of(NAME))
    joints = document["skins"][0]["joints"]
    assert [document["nodes"][j]["name"] for j in joints] == [b["name"] for b in rig.bones]
    for i, node in enumerate(document["nodes"]):
        assert all(child > i for child in node.get("children", ()))


def test_the_skinned_mesh_node_has_no_transform_and_is_a_scene_root():
    document, _ = read_glb(glb_of(NAME))
    node = next(i for i, row in enumerate(document["nodes"]) if "mesh" in row)
    row = document["nodes"][node]
    assert not {"translation", "rotation", "scale", "matrix"} & row.keys()
    assert node in document["scenes"][0]["nodes"]


def test_a_node_keeps_the_rigs_local_axes():
    rig = rigged(NAME)
    document, _ = read_glb(glb_of(NAME))
    by_name = {b["name"]: b for b in rig.bones}
    for node in document["nodes"]:
        bone = by_name.get(node["name"])
        if bone is None or bone["parent"] is None:
            continue
        relative = np.linalg.inv(rest_matrix(by_name[bone["parent"]])) @ rest_matrix(bone)
        assert np.allclose(trs(node)[:3, :3], relative[:3, :3], atol=1e-12)


def test_the_root_node_is_identity():
    document, _ = read_glb(glb_of(NAME))
    assert np.allclose(trs(document["nodes"][0]), np.eye(4), atol=1e-12)


def test_quaternion_handles_every_branch():
    rng = np.random.default_rng(1)
    turns = [(180.0, 0.0, 0.0), (0.0, 180.0, 0.0), (0.0, 0.0, 180.0), (179.0, 2.0, 3.0)]
    turns += [tuple(row) for row in rng.uniform(-180, 180, (64, 3))]
    for turn in turns:
        q = quaternion(euler(turn)[:3, :3])
        assert q[3] >= 0
        assert np.allclose(rotation_of(q), euler(turn)[:3, :3], atol=1e-10)


@pytest.mark.slow
@pytest.mark.parametrize("name", NAMES)
def test_stored_skin_replays_numpy_in_every_check_pose(name):
    rig = rigged(name)
    document, data = read_glb(glb_of(name))
    for pose, rotations in POSES.items():
        wanted = to_gltf(pose_verts(rig.bones, rig.weights, rig.verts, rotations)) * 1000.0
        moved = skin_stored(document, data, rotations)
        worst = np.linalg.norm(moved - wanted, axis=1).max()
        assert worst < STORED_LIMIT_MM, f"{pose}: {worst} mm"


def random_poses():
    rng = np.random.default_rng(6)
    names = [
        "hips",
        "spine",
        "chest",
        "upperChest",
        "neck",
        "head",
        "shoulder",
        "upperArm",
        "lowerArm",
        "hand",
        "upperLeg",
        "lowerLeg",
        "foot",
        "toes",
        "indexProximal",
        "leftUpperArm",
        "rightUpperArm",
        "leftShoulder",
        "rightHand",
        "leftLowerLeg",
    ]
    poses = []
    for _ in range(200):
        chosen = rng.choice(names, size=int(rng.integers(1, 7)), replace=False)
        poses.append({str(n): rng.uniform(-170.0, 170.0, 3).tolist() for n in chosen})
    poses += [{"upperArm": [0.0, 0.0, 0.0]}, {"hand": [0.0, 90.0, 0.0]}, {}]
    return poses


@pytest.mark.node
def test_the_runtimes_own_unit_tests_pass():
    """web/runtime/driver.test.mjs: the runtime alone, without the page's packages; web-check runs it too."""
    done = subprocess.run(["node", "--test", str(DRIVER.with_name("driver.test.mjs"))], capture_output=True,
                          text=True, cwd=ROOT, check=False)
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.node
def test_the_driver_in_javascript_is_body_rig_sided():
    """If this fails after a change to sotaigen.rig.skeleton.sided, web/runtime/driver.mjs needs the same one."""
    rig = rigged(NAME)
    poses = [{k: list(v) for k, v in pose.items()} for pose in POSES.values()] + random_poses()
    table = driver_table(rig.bones)
    out = run_driver(
        "console.log(JSON.stringify(input.poses.map((p) => driver.poseQuaternions(p, input.table))));",
        {"poses": poses, "table": table},
    )
    names = {bone["name"] for bone in rig.bones}
    for pose, found in zip(poses, out, strict=True):
        expected = {
            name: value
            for name, value in sided({k: tuple(v) for k, v in pose.items()}).items()
            if name in names
        }
        assert {name for name in found if name in names} == expected.keys()
        for name, value in expected.items():
            assert np.allclose(rotation_of(found[name]), local_matrix(value)[:3, :3], atol=1e-9), (
                f"{name} in {pose}"
            )


@pytest.mark.node
def test_the_driver_in_javascript_slides_the_jaw_as_body_skinning_does():
    rig = rigged(NAME)
    jaw = next(bone for bone in rig.bones if bone["name"] == "jaw")
    table = driver_table(rig.bones)
    poses = [{}, {"jaw": [12.0, 0.0, 0.0]}, {"jaw": [-4.0, 1.0, 2.0]}, {"jaw": [27.0, 3.0, -5.0], "head": [9.0, 0.0, 0.0]}]
    out = run_driver(
        "console.log(JSON.stringify(input.poses.map((p) => driver.poseTranslations(p, input.table))));",
        {"poses": poses, "table": table},
    )
    for pose, found in zip(poses, out, strict=True):
        if "jaw" not in pose:
            assert found == {}
            continue
        expected = bone_local(jaw, tuple(pose["jaw"]), rig.bones.glides["jaw"])[:3, 3] / 1000
        assert np.allclose(found["jaw"], expected, atol=1e-12), pose


def test_the_driver_table_says_which_names_pass_through():
    rig = rigged(NAME)
    table = driver_table(rig.bones)
    assert "hips" in table["central"]
    assert "root" not in table["central"]
    assert table["rhythm"]["share"] > 0


def tiny(weights):
    """Weights that are neither 0 nor a weight anyone could see."""
    return {name: value for name, value in weights.items() if 0 < abs(value) < 1e-9}


@pytest.mark.node
def test_a_fade_turned_to_its_limit_drives_nothing_in_python_or_javascript():
    rig = rigged(NAME)
    base = {**rig.base, **export_fields(rig.base["faces"])}
    correctives = Correctives.production(base, rig.verts, rig.bones, rig.weights, field=pipeline.surface(NAME)[0])
    out = run_driver(
        "console.log(JSON.stringify(input.poses.map((p) =>"
        " driver.correctiveWeights(p, input.groups, input.central))));",
        {"poses": LIMIT_POSES, "groups": correctives.groups(), "central": driver_table(rig.bones)["central"]},
    )
    for pose, found in zip(LIMIT_POSES, out, strict=True):
        expected = correctives.pose_weights(pose)
        assert not tiny(expected) and not tiny(found), pose
        assert {n for n, v in found.items() if v == 0} == {n for n, v in expected.items() if v == 0}, pose
        np.testing.assert_allclose([found[n] for n in expected], list(expected.values()), atol=1e-12, rtol=0)


def test_python_javascript_and_blender_leave_a_fade_at_the_same_slack():
    from sotaigen.corrective.drive import LIMIT_SLACK

    script = (ROOT / "web" / "runtime" / "driver.mjs").read_text()
    blender = (ROOT / "blender" / "src" / "sotaigen_blender" / "rig.py").read_text()
    import re

    assert float(re.search(r"^export const LIMIT_SLACK = (\S+);$", script, re.MULTILINE)[1]) == LIMIT_SLACK
    assert float(re.search(r"^LIMIT_SLACK = (\S+)$", blender, re.MULTILINE)[1]) == LIMIT_SLACK


@pytest.mark.node
def test_the_runtime_refuses_a_file_it_does_not_know():
    info = read_glb(glb_of(NAME))[0]["extras"]["sotaigen"]
    out = run_driver(
        "console.log(JSON.stringify(input.map((info) => { try { driver.checkInfo(info); return null; }"
        " catch (error) { return error.message; } })));",
        [info, {**info, "version": 99}, {**info, "format": "x"}, None],
    )
    assert out[0] is None
    assert "99" in out[1] and '"x"' in out[2] and "not a sotaigen GLB" in out[3]


@pytest.mark.node
def test_euler_angles_survive_a_quaternion():
    rng = np.random.default_rng(3)
    angles = rng.uniform((-170, -85, -170), (170, 85, 170), (200, 3)).tolist()
    out = run_driver(
        "console.log(JSON.stringify(input.map((e) =>"
        " driver.eulerFromQuaternion(driver.fromEuler(e)))));",
        angles,
    )
    assert np.allclose(out, angles, atol=1e-9)


@pytest.mark.node
def test_at_the_pole_the_rotation_and_its_y_angle_survive():
    out = run_driver(
        "console.log(JSON.stringify(input.map((e) => {"
        " const back = driver.eulerFromQuaternion(driver.fromEuler(e));"
        " return [back, driver.fromEuler(back)]; })));",
        [[0.0, 90.0, 0.0], [20.0, -90.0, 0.0]],
    )
    assert out[0][0][1] == pytest.approx(90.0)
    assert out[1][0][1] == pytest.approx(-90.0)
    assert np.allclose(rotation_of(out[0][1]), euler((0, 90, 0))[:3, :3], atol=1e-6)
    assert np.allclose(rotation_of(out[1][1]), euler((20, -90, 0))[:3, :3], atol=1e-6)


def test_export_is_deterministic():
    assert glb_of(NAME) == glb_of(NAME)


def test_export_changes_nothing_built_and_is_no_input(keeps_the_export):
    rigged(NAME)
    before = pipeline.fingerprints()
    folder = BUILD / "bodies" / NAME
    files = ("wrapped.npy", "skeleton.json", "weights.npz", "fingerprint.json")
    hashes = {f: hashlib.sha256((folder / f).read_bytes()).hexdigest() for f in files}
    path = export_body(NAME)
    assert path == folder / "body.glb"
    document, _ = read_glb(path.read_bytes())
    info = document["extras"]["sotaigen"]
    assert (info["format"], info["version"]) == ("sotaigen/glb", 1)
    assert info["name"] == NAME
    assert info["generator"] == {"name": "sotaigen", "version": package_version(), "fingerprint": pipeline.read_fingerprint(folder)["rig"]}
    assert document["asset"]["generator"] == f"sotaigen {package_version()}"
    # The whole parameter file, to make the body again from the file alone.
    assert read_parameter_file(json.dumps(info["parameters"])) == {
        "sex": rigged(NAME).spec.sex, "body": rigged(NAME).spec.body, "base": None, "unknown": {}
    }
    assert hashes == {f: hashlib.sha256((folder / f).read_bytes()).hexdigest() for f in files}
    assert pipeline.fingerprints() == before
    for stage in pipeline.STAGES:
        reached = {key.split(":")[0] for key in sources.reached(pipeline.STAGE_CODE[stage])}
        assert not {"sotaigen/output/glb.py", "sotaigen/output/face.py"} & reached, stage


def test_check_poses_travel_with_the_file():
    document, _ = read_glb(glb_of(NAME))
    stored = document["extras"]["sotaigen"]["checkPoses"]
    assert stored.keys() == POSES.keys()
    crouch = stored["crouch"]
    rig = rigged(NAME)
    posed = pose_verts(rig.bones, rig.weights, rig.verts, crouch)
    feet = np.asarray(rig.base["regions"]) == "foot"
    assert abs(posed[feet, 2].min()) < 1e-6
    assert abs(np.mean(posed[feet, 1] - rig.verts[feet, 1])) < 1e-6
    assert crouch.pop("rootTranslation")[2] < 0
    for pose, rotations in POSES.items():
        assert stored[pose] == {name: list(value) for name, value in rotations.items()}


def test_a_uv_seam_adds_copies_after_the_original_vertices():
    base, verts, bones, weights = small_rig()
    base["uvs"] = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.5],
        [1.0, 0.5],
        [0.0, 1.0],
        [1.0, 1.0],
        [9.0, 9.0],
    ]
    # The second face's first corner has another UV, so vertex 2 is copied.
    base["faceUvs"] = [[0, 1, 3, 2], [6, 3, 5, 4]]
    document, data = read_glb(glb_bytes(base, verts, bones, weights))
    attributes = primitive(document)["attributes"]
    points = accessor(document, data, attributes["POSITION"])
    uvs = accessor(document, data, attributes["TEXCOORD_0"])
    joints = accessor(document, data, attributes["JOINTS_0"])
    assert len(points) == len(verts) + 1
    assert np.array_equal(points[:6], to_gltf(verts).astype(np.float32))
    assert np.array_equal(points[6], points[2])
    assert uvs[2].tolist() == [0.0, 0.5]
    assert uvs[6].tolist() == [9.0, 9.0]
    assert np.array_equal(joints[6], joints[2])
    assert document["extras"]["sotaigen"]["splitFrom"] == [2]
    triangles = accessor(document, data, primitive(document)["indices"]).reshape(-1, 3)
    assert 6 in triangles[2:]
    assert 2 not in triangles[2:]


def test_morph_targets_are_named_and_in_gltf_units():
    base, verts, bones, weights = small_rig()
    offsets = np.zeros_like(verts)
    offsets[3] = (0.0, -5.0, 0.0)
    document, data = read_glb(
        glb_bytes(base, verts, bones, weights, morphs={"blink": offsets, "aa": offsets * 2})
    )
    assert document["meshes"][0]["extras"]["targetNames"] == ["blink", "aa"]
    assert document["meshes"][0]["weights"] == [0.0, 0.0]
    targets = primitive(document)["targets"]
    assert len(targets) == 2
    moved = accessor(document, data, targets[0]["POSITION"])
    assert moved[3] == pytest.approx([0.0, 0.0, 0.005])
    row = document["accessors"][targets[0]["POSITION"]]
    assert row["max"] == moved.max(axis=0).tolist()


def test_a_rig_without_the_later_fields_gets_neither():
    base, verts, bones, weights = small_rig()
    document, _ = read_glb(glb_bytes(base, verts, bones, weights))
    assert "TEXCOORD_0" not in primitive(document)["attributes"]
    assert "targets" not in primitive(document)
    assert "splitFrom" not in document["extras"]["sotaigen"]


def test_an_exported_preset_carries_the_atlas(keeps_the_export):
    name = NAME
    path = export_body(name)
    document, data = read_glb(path.read_bytes())
    attributes = document["meshes"][0]["primitives"][0]["attributes"]
    uvs = accessor(document, data, attributes["TEXCOORD_0"])
    assert len(uvs) > len(rigged(name).verts)
    assert uvs.min() >= 0.0 and uvs.max() <= 1.0


def stand_in_part(**changes):
    verts = [[0.0, -10.0, 100.0], [5.0, -10.0, 100.0], [5.0, -10.0, 105.0], [0.0, -10.0, 105.0]]
    fields = {
        "name": "patch",
        "verts": np.array(verts),
        "faces": [[0, 1, 2, 3]],
        "material": Material((0.5, 0.5, 0.5, 1.0)),
        **changes,
    }
    return Part(**fields)


@pytest.mark.parametrize(
    ("changes", "complaint"),
    [
        ({"name": "two words"}, "name"),
        ({"name": ""}, "name"),
        ({"verts": np.zeros((4, 2))}, "verts"),
        ({"verts": np.full((4, 3), np.nan)}, "verts"),
        ({"faces": []}, "faces"),
        ({"faces": [[0, 1, 4]]}, "bad face"),
        ({"faces": [[0, 1]]}, "bad face"),
        ({"weights": np.ones((3, 2))}, "weights"),
        ({"weights": np.full((4, 2), 0.2)}, "weights"),
        ({"weights": np.full((4, 5), 0.2)}, "four"),
        ({"uvs": np.zeros((3, 2))}, "uvs"),
        ({"normals": np.zeros((4, 3))}, "length"),
        ({"morphs": {"blink": np.zeros((3, 3))}}, "morph blink"),
        ({"morphs": {"": np.zeros((4, 3))}}, "name"),
    ],
)
def test_a_part_refuses_what_is_not_the_contract(changes, complaint):
    with pytest.raises(ValueError, match=complaint):
        stand_in_part(**changes)


@pytest.mark.parametrize(
    "fields",
    [
        {"base_color": (1.0, 1.0, 1.0)},
        {"base_color": (2.0, 0.0, 0.0, 1.0)},
        {"base_color": (1.0, 1.0, 1.0, 1.0), "alpha": "opaque"},
        {"base_color": (1.0, 1.0, 1.0, 1.0), "alpha_cutoff": 2.0},
        {"base_color": (1.0, 1.0, 1.0, 1.0), "render_order": 0.5},
    ],
)
def test_a_material_refuses_what_gltf_cannot_say(fields):
    with pytest.raises(ValueError):
        Material(**fields)


def test_a_part_takes_plain_lists():
    part = stand_in_part(verts=stand_in_part().verts.tolist(), morphs={"aa": [[0.0, 0.0, 1.0]] * 4})
    assert part.verts.shape == (4, 3)
    assert part.morphs["aa"].shape == (4, 3)


def rig_with_head():
    """small_rig with its last bone called head, which is where a part goes by default."""
    base, verts, bones, weights = small_rig()
    bones[-1]["name"] = "head"
    return base, verts, bones, weights


def glb_with(parts, **changes):
    base, verts, bones, weights = rig_with_head()
    return read_glb(glb_bytes(base, verts, bones, weights, parts=parts, **changes))


def grid_part(name="patch", cells=10, **changes):
    """A flat grid of quads in front of the small rig's head, with (cells + 1)^2 vertices."""
    side = cells + 1
    xs, zs = np.meshgrid(np.linspace(-5.0, 5.0, side), np.linspace(250.0, 260.0, side))
    verts = np.column_stack((xs.ravel(), np.full(side * side, -10.0), zs.ravel()))
    faces = [
        [row * side + col, row * side + col + 1, (row + 1) * side + col + 1, (row + 1) * side + col]
        for row in range(cells)
        for col in range(cells)
    ]
    return Part(name, verts, faces, Material((0.2, 0.3, 0.4, 1.0)), **changes)


def test_each_part_is_a_node_with_its_own_mesh_and_material_on_the_one_skin():
    document, _ = glb_with([grid_part("eye"), grid_part("brow")])
    assert len(document["skins"]) == 1
    meshes = [row for row in document["nodes"] if "mesh" in row]
    assert [row["name"] for row in meshes] == ["body", "eye", "brow"]
    assert [row["skin"] for row in meshes] == [0, 0, 0]
    assert [row["mesh"] for row in meshes] == [0, 1, 2]
    assert [mesh["name"] for mesh in document["meshes"]] == ["body", "eye", "brow"]
    assert [row["name"] for row in document["materials"]] == ["skin", "eye", "brow"]
    for index, mesh in enumerate(document["meshes"]):
        assert mesh["primitives"][0]["material"] == index
    for index, node in enumerate(document["nodes"]):
        if "mesh" in node:
            assert not {"translation", "rotation", "scale", "matrix"} & node.keys()
            assert index in document["scenes"][0]["nodes"]


def test_adding_parts_leaves_the_body_where_it_was():
    alone_document, alone = glb_with([])
    document, data = glb_with([grid_part(morphs={"aa": np.ones((121, 3))})])
    assert data[: len(alone)] == alone
    for key in ("accessors", "bufferViews"):
        assert document[key][: len(alone_document[key])] == alone_document[key]
    assert document["meshes"][0] == alone_document["meshes"][0]
    assert document["materials"][0] == alone_document["materials"][0]
    assert document["skins"] == alone_document["skins"]


def test_no_parts_and_no_normals_are_what_leaving_them_out_gives():
    base, verts, bones, weights = rig_with_head()
    plain = glb_bytes(base, verts, bones, weights)
    assert plain == glb_bytes(base, verts, bones, weights, parts=(), normals=None)
    document, _ = read_glb(plain)
    assert len(document["meshes"]) == 1
    assert len(document["materials"]) == 1
    assert document["scenes"][0]["nodes"][-1] == len(document["nodes"]) - 1


def test_a_part_is_on_the_head_bone_unless_it_says_otherwise():
    document, data = glb_with([grid_part()])
    attributes = primitive(document, 1)["attributes"]
    joints = accessor(document, data, attributes["JOINTS_0"])
    steps = accessor(document, data, attributes["WEIGHTS_0"])
    assert (joints[:, 0] == 2).all()
    assert (steps[:, 0] == WEIGHT_STEPS).all()
    assert (steps[:, 1:] == 0).all()


def test_a_part_can_copy_its_weights_from_the_skin():
    weights = np.zeros((121, 3))
    weights[:, 1:] = (0.25, 0.75)
    document, data = glb_with([grid_part(weights=weights)])
    attributes = primitive(document, 1)["attributes"]
    joints = accessor(document, data, attributes["JOINTS_0"])
    steps = accessor(document, data, attributes["WEIGHTS_0"]) / WEIGHT_STEPS
    assert (joints[:, :2] == (2, 1)).all()
    assert np.allclose(steps[:, :2], (0.75, 0.25), atol=1 / WEIGHT_STEPS)


def test_a_part_refuses_a_rig_it_cannot_bind_to():
    base, verts, bones, weights = small_rig()
    with pytest.raises(ValueError, match="head"):
        glb_bytes(base, verts, bones, weights, parts=[grid_part()])
    with pytest.raises(ValueError, match="columns"):
        glb_bytes(base, verts, bones, weights, parts=[grid_part(weights=np.ones((121, 1)))])


@pytest.mark.parametrize("taken", ["body", "head", "eye"])
def test_a_part_cannot_take_a_name_already_in_the_file(taken):
    with pytest.raises(ValueError, match="taken"):
        glb_with([grid_part("eye"), grid_part(taken)])


def test_a_parts_vertices_are_stored_in_order_in_gltf_units():
    part = grid_part()
    document, data = glb_with([part])
    points = accessor(document, data, primitive(document, 1)["attributes"]["POSITION"])
    assert np.array_equal(points, to_gltf(part.verts).astype(np.float32))


def test_a_parts_triangles_do_not_depend_on_its_shape():
    part = grid_part()
    moved = Part(part.name, part.verts * (3.0, 0.1, 1.0), part.faces, part.material)
    lists = []
    for one in (part, moved):
        document, data = glb_with([one])
        lists.append(accessor(document, data, primitive(document, 1)["indices"]).ravel().tolist())
    assert lists[0] == lists[1]
    assert lists[0][:6] == [0, 1, 12, 0, 12, 11]


def test_a_parts_normals_are_computed_or_taken():
    document, data = glb_with([grid_part()])
    computed = accessor(document, data, primitive(document, 1)["attributes"]["NORMAL"])
    # Counter-clockwise from the front, which is -Y here and +Z in glTF.
    assert np.allclose(computed, (0.0, 0.0, 1.0), atol=1e-6)
    given = np.tile((0.0, -3.0, 4.0), (121, 1))
    document, data = glb_with([grid_part(normals=given)])
    stored = accessor(document, data, primitive(document, 1)["attributes"]["NORMAL"])
    assert np.allclose(stored, (0.0, 0.8, 0.6), atol=1e-6)


def test_a_parts_uvs_are_stored_only_if_it_has_them():
    document, _ = glb_with([grid_part()])
    assert "TEXCOORD_0" not in primitive(document, 1)["attributes"]
    uvs = np.tile((0.25, 0.75), (121, 1))
    document, data = glb_with([grid_part(uvs=uvs)])
    stored = accessor(document, data, primitive(document, 1)["attributes"]["TEXCOORD_0"])
    assert np.array_equal(stored, uvs.astype(np.float32))


def test_material_fields_become_gltf_and_defaults_stay_out():
    grid = grid_part()
    cut = Material(
        (0.1, 0.2, 0.3, 0.5), "MASK", 0.25, render_order=3, double_sided=True, outline=True
    )
    parts = [
        grid_part("plain"),
        Part("hair", grid.verts, grid.faces, cut),
        Part("veil", grid.verts, grid.faces, Material((1.0, 1.0, 1.0, 0.5), "BLEND")),
    ]
    document, _ = glb_with(parts)
    plain, hair, veil = document["materials"][1:]
    assert plain["pbrMetallicRoughness"]["baseColorFactor"] == [0.2, 0.3, 0.4, 1.0]
    assert not {"alphaMode", "alphaCutoff", "doubleSided", "extras"} & plain.keys()
    assert hair["alphaMode"] == "MASK"
    assert hair["alphaCutoff"] == 0.25
    assert hair["doubleSided"] is True
    assert hair["extras"]["sotaigen"] == {"renderOrder": 3, "outline": True}
    assert veil["alphaMode"] == "BLEND"
    assert "alphaCutoff" not in veil


def moved_offsets(count, moved):
    """V x 3 offsets with the first `moved` vertices each going 2 mm to the wearer's front."""
    offsets = np.zeros((count, 3))
    offsets[:moved, 1] = -2.0
    return offsets


def test_a_target_stores_only_the_vertices_it_moves():
    document, data = glb_with([grid_part(morphs={"blink": moved_offsets(121, 5)})])
    index = primitive(document, 1)["targets"][0]["POSITION"]
    row = document["accessors"][index]
    assert "bufferView" not in row
    assert row["count"] == 121
    assert row["sparse"]["count"] == 5
    assert row["sparse"]["indices"]["componentType"] == COMPONENT["uint16"]
    for view in ("indices", "values"):
        assert "target" not in document["bufferViews"][row["sparse"][view]["bufferView"]]
    offsets = accessor(document, data, index)
    assert (offsets[:5, 2] == np.float32(0.002)).all()
    assert not offsets[5:].any()
    assert row["min"] == offsets.min(axis=0).tolist()
    assert row["max"] == offsets.max(axis=0).tolist()
    assert document["meshes"][1]["extras"]["targetNames"] == ["blink"]
    assert document["meshes"][1]["weights"] == [0.0]


def test_a_target_that_moves_nearly_everything_stays_dense():
    document, data = glb_with([grid_part(morphs={"aa": moved_offsets(121, 120)})])
    index = primitive(document, 1)["targets"][0]["POSITION"]
    row = document["accessors"][index]
    assert "sparse" not in row
    assert document["bufferViews"][row["bufferView"]]["target"] == ARRAY_BUFFER
    assert accessor(document, data, index)[:120, 2].min() == np.float32(0.002)


def test_a_target_that_moves_nothing_stores_nothing():
    _, without = glb_with([grid_part()])
    document, data = glb_with([grid_part(morphs={"still": np.zeros((121, 3))})])
    index = primitive(document, 1)["targets"][0]["POSITION"]
    row = document["accessors"][index]
    assert "bufferView" not in row and "sparse" not in row
    assert row["min"] == row["max"] == [0.0, 0.0, 0.0]
    assert len(data) == len(without)
    assert not accessor(document, data, index).any()


def test_noise_is_not_a_move():
    offsets = moved_offsets(121, 5)
    offsets[5:] = MORPH_EPSILON_MM / 2
    document, _ = glb_with([grid_part(morphs={"blink": offsets})])
    row = document["accessors"][primitive(document, 1)["targets"][0]["POSITION"]]
    assert row["sparse"]["count"] == 5


def test_sparse_targets_take_fewer_bytes_until_they_would_not():
    def size(moved):
        base, verts, bones, weights = rig_with_head()
        offsets = moved_offsets(121, moved)
        return len(glb_bytes(base, verts, bones, weights, parts=[grid_part(morphs={"a": offsets})]))

    plain = size(0)
    assert size(10) - plain < 121 * 12 / 2
    # Whatever is chosen, it is never much worse than dense's 12 bytes a vertex and its JSON.
    assert size(121) - plain < 121 * 12 + 200


def test_every_part_keeps_its_targets_on_its_own_mesh():
    parts = [
        grid_part("eye", morphs={"blink": moved_offsets(121, 3), "aa": moved_offsets(121, 4)}),
        grid_part("mouth", morphs={"aa": moved_offsets(121, 6)}),
        grid_part("ear"),
    ]
    document, _ = glb_with(parts)
    assert "targets" not in primitive(document, 0)
    assert document["meshes"][1]["extras"]["targetNames"] == ["blink", "aa"]
    assert document["meshes"][2]["extras"]["targetNames"] == ["aa"]
    assert "targets" not in primitive(document, 3)
    assert "extras" not in document["meshes"][3]


def seamed_rig():
    """rig_with_head with vertex 2 copied at a UV seam."""
    base, verts, bones, weights = rig_with_head()
    base["uvs"] = [[0, 0], [1, 0], [0, 0.5], [1, 0.5], [0, 1], [1, 1], [0.9, 0.9]]
    base["faceUvs"] = [[0, 1, 3, 2], [6, 3, 5, 4]]
    return base, verts, bones, weights


def test_a_skin_morph_target_follows_a_seam_copy_when_sparse():
    base, verts, bones, weights = seamed_rig()
    offsets = np.zeros_like(verts)
    offsets[2] = (0.0, -5.0, 0.0)
    document, data = read_glb(glb_bytes(base, verts, bones, weights, morphs={"cheek": offsets}))
    index = primitive(document)["targets"][0]["POSITION"]
    stored = accessor(document, data, index)
    assert len(stored) == 7
    assert np.array_equal(np.flatnonzero(stored.any(axis=1)), [2, 6])
    assert np.array_equal(stored[6], stored[2])


def test_given_skin_normals_replace_the_computed_ones_and_seam_copies_follow():
    base, verts, bones, weights = seamed_rig()
    given = np.tile((0.0, -1.0, 0.0), (6, 1))
    given[2] = (6.0, 0.0, 8.0)
    document, data = read_glb(glb_bytes(base, verts, bones, weights, normals=given))
    stored = accessor(document, data, primitive(document)["attributes"]["NORMAL"])
    assert len(stored) == 7
    assert np.allclose(np.linalg.norm(stored, axis=1), 1.0, atol=1e-6)
    # The rig's (x, y, z) is glTF's (x, z, -y), and a normal is not scaled to metres.
    assert np.allclose(stored[0], (0.0, 0.0, 1.0), atol=1e-6)
    assert np.allclose(stored[2], (0.6, 0.8, 0.0), atol=1e-6)
    assert np.array_equal(stored[6], stored[2])


@pytest.mark.parametrize(
    "bad", [np.zeros((5, 3)), np.zeros((6, 2)), np.zeros((6, 3)), np.full((6, 3), np.nan)]
)
def test_given_skin_normals_must_fit_and_have_a_length(bad):
    base, verts, bones, weights = rig_with_head()
    with pytest.raises(ValueError, match="normals"):
        glb_bytes(base, verts, bones, weights, normals=bad)


def test_the_default_skin_normals_ignore_uv_seams():
    base, verts, bones, weights = seamed_rig()
    # A fold along the row the seam runs through, so each side alone would give another normal.
    verts = verts.copy()
    verts[4:, 1] = 100.0
    document, data = read_glb(glb_bytes(base, verts, bones, weights))
    stored = accessor(document, data, primitive(document)["attributes"]["NORMAL"])
    assert np.allclose(to_gltf(skin_normals(base, verts)) * 1000.0, stored[:6], atol=1e-6)
    assert np.array_equal(stored[6], stored[2])
    assert not np.allclose(stored[0], stored[2], atol=0.01)
    assert not np.allclose(stored[4], stored[2], atol=0.01)
    assert glb_bytes(base, verts, bones, weights) == glb_bytes(
        base, verts, bones, weights, normals=skin_normals(base, verts)
    )


def test_an_exported_preset_has_no_crease_at_its_uv_seams():
    rig = rigged(NAME)
    base = {**rig.base, **export_fields(rig.base["faces"])}
    document, data = read_glb(glb_bytes(base, rig.verts, rig.bones, rig.weights))
    stored = accessor(document, data, primitive(document)["attributes"]["NORMAL"])
    source = np.array(document["extras"]["sotaigen"]["splitFrom"])
    assert len(source) > 0
    assert len(stored) == len(rig.verts) + len(source)
    assert np.array_equal(stored[len(rig.verts) :], stored[source])
    # A seam's originals too: the surface decides their normals, not the atlas.
    plain = {key: value for key, value in base.items() if key not in ("uvs", "faceUvs")}
    without_seams, read = read_glb(glb_bytes(plain, rig.verts, rig.bones, rig.weights))
    unsplit = accessor(without_seams, read, primitive(without_seams)["attributes"]["NORMAL"])
    assert np.array_equal(stored[: len(rig.verts)], unsplit)


def face_part(rig):
    """A patch before the real head, with a target that moves part of it."""
    top = rig.verts[:, 2].max()
    xs, zs = np.meshgrid(np.linspace(-20.0, 20.0, 9), np.linspace(top - 80.0, top - 50.0, 7))
    verts = np.column_stack((xs.ravel(), np.full(63, -90.0), zs.ravel()))
    faces = [
        [r * 9 + c, r * 9 + c + 1, (r + 1) * 9 + c + 1, (r + 1) * 9 + c]
        for r in range(6)
        for c in range(8)
    ]
    offsets = np.zeros((63, 3))
    offsets[:20, 2] = -4.0
    return Part("face", verts, faces, Material((0.1, 0.1, 0.1, 1.0)), morphs={"blink": offsets})


@pytest.mark.parametrize("name", NAMES)
def test_a_stored_part_replays_numpy_in_every_check_pose_with_its_target(name):
    rig = rigged(name)
    part = face_part(rig)
    blob = glb_bytes(rig.base, rig.verts, rig.bones, rig.weights, parts=[part])
    document, data = read_glb(blob)
    head = np.zeros((len(part.verts), len(rig.bones)))
    head[:, [b["name"] for b in rig.bones].index("head")] = 1.0
    for weight in (0.0, 0.6):
        at = part.verts + weight * part.morphs["blink"]
        for pose, rotations in POSES.items():
            wanted = to_gltf(pose_verts(rig.bones, head, at, rotations)) * 1000.0
            moved = skin_stored(document, data, rotations, mesh=1, morph_weights={"blink": weight})
            worst = np.linalg.norm(moved - wanted, axis=1).max()
            assert worst < STORED_LIMIT_MM, f"{pose} at {weight}: {worst} mm"


def test_a_part_turns_with_the_head():
    rig = rigged(NAME)
    part = face_part(rig)
    document, data = read_glb(glb_bytes(rig.base, rig.verts, rig.bones, rig.weights, parts=[part]))
    rest = skin_stored(document, data, {}, mesh=1)
    turned = skin_stored(document, data, POSES["headTurn"], mesh=1)
    assert np.linalg.norm(turned - rest, axis=1).max() > 10.0
    legs_only = skin_stored(document, data, {"upperLeg": (30.0, 0.0, 0.0)}, mesh=1)
    assert np.allclose(legs_only, rest, atol=1e-3)


def stand_in_module():
    spec = importlib.util.spec_from_file_location("standin_parts", STAND_IN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_stand_in_takes_every_path_the_contract_has():
    module = stand_in_module()
    rig = rigged(NAME)
    data = module.stand_in(NAME)
    document, read = read_glb(module.standin_bytes(NAME, data))
    assert [node["name"] for node in document["nodes"] if "mesh" in node][1:] == [
        part.name for part in data.parts
    ]
    modes = {row.get("alphaMode", "OPAQUE") for row in document["materials"]}
    assert modes == {"OPAQUE", "MASK", "BLEND"}
    # Copied weights reach past the head bone, and the skin's target moves some seam copies.
    assert any(
        part.weights is not None and part.weights.max(axis=1).min() < 1 for part in data.parts
    )
    target = accessor(document, read, primitive(document)["targets"][0]["POSITION"])
    assert target[len(rig.verts) :].any()
    assert "sparse" in document["accessors"][primitive(document)["targets"][0]["POSITION"]]
    # Every name in the settings is a target somewhere.
    names = {
        name
        for mesh in document["meshes"]
        for name in mesh.get("extras", {}).get("targetNames", [])
    }
    for weights in module.SETTINGS.values():
        assert weights.keys() <= names


def test_the_stand_in_replays_numpy_in_every_check_pose_with_its_targets():
    # What the head's real parts don't take through the browser check: copied weights across the
    # neck's blend, and a morph target on the skin with seam copies in it.
    module = stand_in_module()
    rig = rigged(NAME)
    data = module.stand_in(NAME)
    document, read = read_glb(module.standin_bytes(NAME, data))
    source = np.array(document["extras"]["sotaigen"]["splitFrom"])
    order = np.concatenate((np.arange(len(rig.verts)), source))
    names = [bone["name"] for bone in rig.bones]
    meshes = [(0, rig.verts, np.asarray(rig.weights), data.morphs, order)]
    for index, part in enumerate(data.parts, start=1):
        weights = part.weights
        if weights is None:
            weights = np.zeros((len(part.verts), len(names)))
            weights[:, names.index("head")] = 1.0
        meshes.append((index, part.verts, weights, part.morphs, np.arange(len(part.verts))))
    setting = module.SETTINGS["mix"]
    for pose in POSES.values():
        for index, verts, weights, morphs, kept in meshes:
            given = {name: weight for name, weight in setting.items() if name in morphs}
            moved = verts + sum(
                (weight * morphs[name] for name, weight in given.items()),
                start=np.zeros_like(verts),
            )
            wanted = to_gltf(pose_verts(rig.bones, weights, moved, pose)[kept]) * 1000.0
            stored = skin_stored(document, read, pose, mesh=index, morph_weights=given)
            assert np.linalg.norm(stored - wanted, axis=1).max() < STORED_LIMIT_MM


def test_a_broken_preset_is_refused_before_any_bake_and_nothing_is_written(keeps_the_export, monkeypatch, capsys):
    reasons = ["flipped faces after the wrap: 3", "the skin is not closed: 2 edges without exactly two faces"]
    monkeypatch.setattr(glb_module, "broken", lambda _wrap, _faces: reasons)

    def no_bake(*_args, **_kwargs):
        raise AssertionError("a broken mesh got as far as making correctives")

    monkeypatch.setattr(Correctives, "production", no_bake)
    plain = BUILD / "bodies" / NAME / "body.glb"
    plain.unlink(missing_ok=True)
    with pytest.raises(quality.BrokenBody) as refused:
        export_body(NAME)
    assert refused.value.reasons == reasons
    assert not plain.exists()

    assert main([NAME]) == 1
    assert main([NAME, "--no-parts"]) == 1
    assert not plain.exists()
    said = capsys.readouterr().err
    assert said.count("not exported, the mesh is broken") == 2 and reasons[1] in said

    baked = BUILD / "bodies" / NAME / "body-ipc.glb"
    before = baked.read_bytes() if baked.exists() else None
    assert main([NAME, "--contact"]) == 1
    assert (baked.read_bytes() if baked.exists() else None) == before


def test_the_exports_quality_problems_travel_in_the_file(keeps_the_export):
    """Both levels are .blend's: the wrap's, the atlas's and the check poses' as the build measures them."""
    pipeline.ensure_built(NAME, ("metrics",))
    info = read_glb(export_body(NAME).read_bytes())[0]["extras"]["sotaigen"]
    wrap = json.loads((BUILD / "bodies" / NAME / "wrap.json").read_text())
    poses = json.loads((BUILD / "bodies" / NAME / "poses.json").read_text())
    rig = rigged(NAME)
    wanted = as_json(quality.quality(wrap, rig.verts, rig.base["faces"], poses))
    assert info["quality"]["found"][: len(wanted)] == wanted
    assert info["quality"]["checked"] == ["wrap", "uv", "parameters", "poses"]
    assert "broken" not in info


def test_the_quality_list_adds_what_was_found_and_the_failed_bakes(monkeypatch):
    rig = rigged(NAME)
    wrap = json.loads((BUILD / "bodies" / NAME / "wrap.json").read_text())
    plain = quality.quality(wrap, rig.verts, rig.base["faces"])
    args = (wrap, rig.base, rig.verts, rig.bones, rig.weights)
    sculpted = SimpleNamespace(contact="sculpted", bind=lambda verts, _pose: verts)
    seam = rule("wrist_thick_hand", 0.01, "warning")
    found = quality.parameter_notes([seam], {"headHeight": (1.0, 2.0)}, ["legsPart"])
    assert found == [
        "warning: the wrist is too thick for the hand, short by 0.010",
        "target: headHeight asked 1.0, reached 2.0",
        "surface: legsPart",
    ]
    assert [d.code for d in found] == ["rule.wrist_thick_hand", "target.missed", "surface.legsPart"]
    assert quality.glb_quality(*args, sculpted, found=found) == {
        "checked": ["wrap", "uv", "parameters"], "found": as_json([*plain, *found])
    }
    # The default correctives never baked anything, so what a cache says of this rig isn't asked.
    failed = Diagnostic("contact.failed", "warning", {"group": "axilla", "detail": "x"})
    monkeypatch.setattr(quality, "contact_failures", lambda *_args: [failed])
    assert quality.glb_quality(*args, sculpted)["found"] == as_json(plain)
    # With the IPC tables the check poses are measured too, so their lines come before the bakes'. This
    # stand-in's correctives do nothing, so its elbow and crouch collapse however the real ones would.
    baked = quality.glb_quality(*args, SimpleNamespace(contact="ipc", bind=sculpted.bind))
    assert baked["checked"] == ["wrap", "uv", "parameters", "poses", "contact"]
    assert baked["found"][-1] == failed.json()
    assert [d for d in baked["found"][:-1] if d["code"] != "pose.collapsed"] == as_json(plain)
    assert len(baked["found"]) > len(plain) + 1


def test_export_body_passes_parts_and_normals_along(keeps_the_export):
    rig = rigged(NAME)
    normals = skin_normals(rig.base, rig.verts)
    path = export_body(NAME, parts=[face_part(rig)], normals=normals)
    document, data = read_glb(path.read_bytes())
    assert document["meshes"][1]["name"] == "face"
    stored = accessor(document, data, primitive(document)["attributes"]["NORMAL"])
    wanted = (normals @ TO_GLTF.T).astype(np.float32)
    assert np.allclose(stored[: len(rig.verts)], wanted, atol=1e-5)


def groups_of(document, part):
    """The file's correctives groups of one part (axilla, groin, hang), without their sources."""
    return unsourced(
        group for group in document["extras"]["sotaigen"]["correctives"]["groups"] if group["name"].split(".")[0] == part
    )


def unsourced(groups):
    return [{key: value for key, value in group.items() if key != "source"} for group in groups]


def expected_hang_names(document):
    """The hanging arm's default (implicit-field) targets: the IPC table's names and driver."""
    table = hang_driver.driver_metadata(hang_driver.adduction_keys(rigged(NAME).bones))
    assert groups_of(document, "hang") == unsourced(hang_driver.groups(table))
    return [name for side in hang_driver.SIDES for name in table["sides"][side]["targets"]]


def expected_axilla_names(document=None):
    """The sculpted shoulder correctives' names, which an export has unless asked for contact tables."""
    if document is not None:
        groups = document["extras"]["sotaigen"]["correctives"]["groups"]
        assert {group["name"]: group["source"] for group in groups} == {
            "axilla.left": "sculpted", "axilla.right": "sculpted", "groin": "sculpted", "hang.left": "field", "hang.right": "field"
        }
        assert groups_of(document, "axilla") == unsourced(axilla.groups())
    return [
        f"axilla{side}{forward}" + (f"Spread{spread}" if spread else "")
        for side in ("Left", "Right")
        for forward in (0, 20, 35, 60, 75, 95)
        for spread in (-24, 0, 35, 72, 110)
        if (forward, spread) != (0, 0)
    ]


def test_the_command_writes_the_head_by_default(keeps_the_export, capsys):
    assert main([NAME]) == 0
    path = BUILD / "bodies" / NAME / "body.glb"
    assert str(path) in capsys.readouterr().out
    written = path.read_bytes()
    document, data = read_glb(written)
    parts = face_parts(NAME)
    assert [mesh["name"] for mesh in document["meshes"]] == ["body", *(part.name for part in parts)]
    stored = accessor(document, data, primitive(document)["attributes"]["NORMAL"])
    wanted = stored_normals(head_normals(NAME))
    assert np.array_equal(stored[: len(wanted)], wanted)
    assert document["extras"]["sotaigen"]["name"] == NAME
    # Every expression reaches the file, under its own name, and the file lists them.
    assert document["extras"]["sotaigen"]["expressions"] == list(EXPRESSIONS)
    names = [mesh.get("extras", {}).get("targetNames", []) for mesh in document["meshes"]]
    assert names[0] == [*expected_axilla_names(document), *groin.NAMES, *expected_hang_names(document)]
    assert names[1:] == [list(part.morphs) for part in parts]
    assert written == export_with_face(NAME).read_bytes()


def test_the_command_writes_the_skin_alone_when_asked(keeps_the_export, capsys):
    assert main([NAME, "--no-parts"]) == 0
    capsys.readouterr()
    written = (BUILD / "bodies" / NAME / "body.glb").read_bytes()
    document, data = read_glb(written)
    assert [mesh["name"] for mesh in document["meshes"]] == ["body"]
    assert document["meshes"][0]["extras"]["targetNames"] == [
        *expected_axilla_names(document),
        *groin.NAMES,
        *expected_hang_names(document),
    ]
    rig = rigged(NAME)
    stored = accessor(document, data, primitive(document)["attributes"]["NORMAL"])
    wanted = stored_normals(skin_normals(rig.base, rig.verts))
    assert np.array_equal(stored[: len(wanted)], wanted)
    assert written == export_body(NAME).read_bytes()


@pytest.mark.parametrize("first", ["sotaigen.output.glb", "sotaigen.output.face"])
def test_either_module_can_be_imported_first(first):
    done = subprocess.run(
        [sys.executable, "-c", f"import {first}, sotaigen.output.glb, sotaigen.output.face"],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr


def test_the_real_command_line_writes_the_head(keeps_the_export):
    # As __main__, this module is loaded a second time when sotaigen.output.face imports it.
    done = subprocess.run(
        [sys.executable, "-m", "sotaigen.output.glb", NAME],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""
    document, _ = read_glb((BUILD / "bodies" / NAME / "body.glb").read_bytes())
    assert len(document["meshes"]) == 1 + len(face_parts(NAME))


def test_the_skin_coloured_parts_have_the_skins_colour_in_the_file(keeps_the_export):
    document, _ = read_glb(export_with_face(NAME).read_bytes())
    colours = {
        row["name"]: row["pbrMetallicRoughness"]["baseColorFactor"] for row in document["materials"]
    }
    assert colours["nose"] == colours["skin"]
    assert colours["ears"] == colours["skin"]


def test_what_a_head_swap_relies_on_is_in_the_file_and_the_same_for_every_body():
    rows = []
    for name in NAMES[:3]:
        rig = rigged(name)
        rows.append(head_attachments(rig.base, rig.verts, rig.bones))
        segments = np.asarray(rig.base["segments"])
        start, count = rows[-1]["region"]
        assert (segments[start : start + count] == "head").all()
        assert (segments == "head").sum() == count
        assert sorted(rows[-1]["neckRing"]) == sorted(rig.base["openings"]["neck"])
        # The eyes as the head swap places them: by the eye bone, from the crown and the ring.
        eye = next(bone for bone in rig.bones if bone["name"] == "leftEye")["head"]
        crown, ring = rig.verts[rows[-1]["crownVertex"]], rig.verts[rows[-1]["neckRing"]]
        assert (rows[-1]["eyeLine"], rows[-1]["eyeX"]) == eye_place(crown, ring, eye)[:2]
        assert 0.3 < rows[-1]["eyeLine"] < 0.6 and 0.1 < rows[-1]["eyeX"] < 0.25
        assert "eyeU" not in rows[-1] and "referenceWidth" not in rows[-1]
    for key in ("bones", "region", "neckRing", "crownVertex"):
        assert all(row[key] == rows[0][key] for row in rows), key
    assert rows[0]["bones"] == ["head", "jaw", "leftEye", "rightEye"]


def test_a_baked_motion_carries_the_jaws_slide():
    rig = rigged(NAME)
    motions = [("open", 10.0, [{}, {"jaw": (20.0, 0.0, 0.0)}])]
    document, data = read_glb(glb_bytes(rig.base, rig.verts, rig.bones, rig.weights, motions=motions))
    node = [bone["name"] for bone in rig.bones].index("jaw")
    animation = document["animations"][0]
    channel = next(
        row for row in animation["channels"] if row["target"] == {"node": node, "path": "translation"}
    )
    places = accessor(document, data, animation["samplers"][channel["sampler"]]["output"])
    assert np.allclose(places[0], document["nodes"][node]["translation"], atol=1e-6)
    assert np.linalg.norm(places[1] - places[0]) == pytest.approx(
        20.0 * np.linalg.norm(rig.bones.glides["jaw"]) / 1000, abs=1e-6
    )


def test_the_exported_jaw_and_eyes_move_what_they_carry(keeps_the_export):
    document, data = read_glb(export_with_face(NAME).read_bytes())
    rig = rigged(NAME)
    meshes = [mesh["name"] for mesh in document["meshes"]]
    h = rig.L["head"]

    def moved(mesh, pose):
        index = meshes.index(mesh)
        return np.linalg.norm(skin_stored(document, data, pose, mesh=index) - skin_stored(document, data, {}, mesh=index), axis=1)

    opened = {"jaw": (20.0, 0.0, 0.0)}
    assert moved("body", opened).max() > 0.1 * h
    assert moved("mouth", opened).max() > 0.05 * h
    for still in ("iris", "nose", "ears", "hair", "lash"):
        assert moved(still, opened).max() == 0, still
    looked = {"leftEye": (0.0, 0.0, 12.0), "rightEye": (0.0, 0.0, 12.0)}
    for carried in ("iris", "pupil", "eyeLight"):
        assert moved(carried, looked).min() > 0.01 * h, carried
    for still in ("body", "eyeWhite", "mouth", "lash"):
        assert moved(still, looked).max() == 0, still


def test_reach_motion_replays_baked_weights_with_bones_and_keeps_extra_body_targets():
    from sotaigen.corrective import axilla

    rig = rigged("female-long")
    base = {**rig.base, **export_fields(rig.base["faces"])}
    poses = [
        {},
        {"leftUpperArm": (35, 0, 0), "rootTranslation": (0, -20, 10)},
        {"upperArm": (68, 12, 10)},
        {"rightUpperArm": (75, 0, 0)},
    ]
    fps = 30
    blob = glb_bytes(
        base,
        rig.verts,
        rig.bones,
        rig.weights,
        morphs={"expression": np.zeros_like(rig.verts)},
        motions=[("reach", fps, poses)],
        correctives=Correctives(axilla.targets(rig.base, rig.verts, rig.bones, rig.weights)),
    )
    document, data = read_glb(blob)
    names = document["meshes"][0]["extras"]["targetNames"]
    assert names == ["expression", *expected_axilla_names()]
    assert groups_of(document, "axilla") == unsourced(axilla.groups())
    (animation,) = document["animations"]
    (weights,) = [c for c in animation["channels"] if c["target"]["path"] == "weights"]
    assert document["nodes"][weights["target"]["node"]]["name"] == "body"
    sampler = animation["samplers"][weights["sampler"]]
    assert sampler["interpolation"] == "LINEAR"
    assert {s["input"] for s in animation["samplers"]} == {sampler["input"]}
    assert accessor(document, data, sampler["input"]).ravel() == pytest.approx(
        np.arange(len(poses)) / fps
    )
    assert document["accessors"][sampler["output"]]["type"] == "SCALAR"
    rows = accessor(document, data, sampler["output"]).reshape(len(poses), len(names))
    assert not rows[:, 0].any()
    assert rows[1, names.index("axillaLeft35")] == 1
    assert rows[-1, names.index("axillaRight75")] == 1
    deltas = axilla.targets(base, rig.verts, rig.bones, rig.weights)
    copies = document["extras"]["sotaigen"]["splitFrom"]
    source = np.r_[np.arange(len(rig.verts)), copies]
    primitive_row = primitive(document)
    # Additional expressions with no normal change get a zero accessor, and all
    # UV duplicates carry the identical direction change on each corrective.
    normal_rows = [accessor(document, data, t["NORMAL"]) for t in primitive_row["targets"]]
    assert not normal_rows[0].any()
    for delta in normal_rows[1:]:
        assert delta.any()
        np.testing.assert_array_equal(delta, delta[source])
    for i, pose in enumerate(poses):
        # Replay plain glTF TRS and weight channels with no semantic driver.
        for ch in animation["channels"]:
            node = ch["target"]
            if node["path"] != "weights":
                track = animation["samplers"][ch["sampler"]]["output"]
                document["nodes"][node["node"]][node["path"]] = accessor(document, data, track)[i]
        expected = rig.verts.copy()
        for key, weight in axilla.pose_weights(pose).items():
            expected += weight * deltas[key]
        expected = pose_verts(rig.bones, rig.weights, expected, pose)
        weights_at_frame = dict(zip(names, rows[i], strict=True))
        rendered = skin_stored(document, data, {}, morph_weights=weights_at_frame)
        assert np.max(abs(rendered - 1000 * to_gltf(expected[source]))) < STORED_LIMIT_MM


def test_corrected_shading_follows_the_bent_axillary_surface_in_the_glb():
    from sotaigen.rig.skinning import pose_skins

    rig = rigged("female-long")
    base = {**rig.base, **export_fields(rig.base["faces"])}
    from sotaigen.corrective import axilla

    shoulder = Correctives(axilla.targets(rig.base, rig.verts, rig.bones, rig.weights))
    doc, data = read_glb(glb_bytes(base, rig.verts, rig.bones, rig.weights, correctives=shoulder))
    mesh = primitive(doc)
    names = doc["meshes"][0]["extras"]["targetNames"]
    count = len(rig.verts)
    # Read glTF unit directions and metre positions back in the generator's frame.
    normal = accessor(doc, data, mesh["attributes"]["NORMAL"])[:count] @ TO_GLTF
    for angle in (75, 95):
        key = mesh["targets"][names.index(f"axillaLeft{angle}")]
        delta = (accessor(doc, data, key["POSITION"])[:count] @ TO_GLTF) * 1000
        normal_delta = accessor(doc, data, key["NORMAL"])[:count] @ TO_GLTF
        pose = {"leftUpperArm": (angle, 0, 0)}
        mesh_points = pose_verts(rig.bones, rig.weights, rig.verts + delta, pose)
        geometric = skin_normals(base, mesh_points)
        rotations = pose_skins(rig.bones, pose)[:, :3, :3].reshape(-1, 9)
        blend = (rig.weights @ rotations).reshape(-1, 3, 3)
        shaded = (blend @ (normal + normal_delta)[:, :, None])[:, :, 0]
        shaded /= np.linalg.norm(shaded, axis=1, keepdims=True)
        # A key's deltas are scaled by its strength: the same skin as at 5 mm in full.
        active = np.linalg.norm(delta, axis=1) > 5 * axilla.key_strength(angle, 0)
        cosines = np.sum(geometric[active] * shaded[active], axis=1)
        degrees = np.rad2deg(np.arccos(np.clip(cosines, -1, 1)))
        assert active.sum() > 100
        assert np.percentile(degrees, 90) < 3
        # Custom head normals and unaffected limbs are excluded from the correction.
        assert not normal_delta[np.asarray(base["segments"]) == "head"].any()


@pytest.mark.parametrize("planar", [False, True])
def test_body_specific_shoulder_table_survives_export_and_baked_animation(planar):
    from sotaigen.corrective import axilla
    from sotaigen.corrective.pose_grid import PoseGrid, from_samples

    rig = rigged("female-chibi")
    base = {**rig.base, **export_fields(rig.base["faces"])}
    grid = PoseGrid.rectilinear([[0, 35], [-24, 0]] if planar else [[0, 35], [-90, 0], [-24, 0]])
    # A small synthetic displacement tests the table/animation contract independently
    # of contact fitting. Both sides use distinct vertices and opposite directions.
    local = np.zeros((len(grid.points), 1, 3))
    local[:, 0, 0] = grid.points[:, 0] / 35
    samples = {side: {"points": grid.points, "cells": grid.cells,
                     "ids": [vertex], "deltas": local * sign}
               for side, vertex, sign in [("left", 1437, 1), ("right", 1438, -1)]}
    targets = from_samples(samples, len(rig.verts), axilla.TWIST_LIMIT)
    poses = [{}, {"leftUpperArm": (35, -90, -24)},
             {"leftUpperArm": (17, -32, -12), "rightUpperArm": (26, 14, 5)}]
    document, data = read_glb(glb_bytes(
        base, rig.verts, rig.bones, rig.weights, correctives=Correctives(targets),
        morphs={"expression": np.zeros_like(rig.verts)}, motions=[("contact", 30, poses)]))
    assert groups_of(document, "axilla") == unsourced(axilla.groups(targets.driver))
    names = document["meshes"][0]["extras"]["targetNames"]
    assert names == ["expression", *targets]
    animation, = document["animations"]
    channel, = [c for c in animation["channels"] if c["target"]["path"] == "weights"]
    track = animation["samplers"][channel["sampler"]]
    rows = accessor(document, data, track["output"]).reshape(len(poses), len(names))
    source = np.r_[np.arange(len(rig.verts)), document["extras"]["sotaigen"]["splitFrom"]]
    for i, pose in enumerate(poses):
        expected_weights = {"expression": 0., **axilla.pose_weights(pose, targets.driver)}
        assert rows[i] == pytest.approx([expected_weights[name] for name in names], abs=1e-7)
        for ch in animation["channels"]:
            node = ch["target"]
            if node["path"] != "weights":
                values = animation["samplers"][ch["sampler"]]["output"]
                document["nodes"][node["node"]][node["path"]] = accessor(document, data, values)[i]
        expected = pose_verts(rig.bones, rig.weights, axilla.bind_verts(rig.verts, targets, pose), pose)
        rendered = skin_stored(document, data, {}, morph_weights=dict(zip(names, rows[i], strict=True)))
        assert np.max(abs(rendered - 1000 * to_gltf(expected[source]))) < STORED_LIMIT_MM


@pytest.mark.bake
def test_pelvis_targets_replay_with_pose_and_are_baked_into_animation():
    rig = rigged("female-long")
    base = {**rig.base, **export_fields(rig.base["faces"])}
    poses = [{}, groin.sample_pose(40, 0), groin.sample_pose(20, 80)]
    document, data = read_glb(
        glb_bytes(
            base,
            rig.verts,
            rig.bones,
            rig.weights,
            correctives=Correctives(
                groin=groin_contact.targets(rig.base, rig.verts, rig.bones, rig.weights)
            ),
            motions=[("hip", 30, poses)],
        )
    )
    # The contact table keeps 90 and 110 on top of the sculpture's keys; the file's group says so.
    assert groups_of(document, "groin") == unsourced(groin.groups(keys=groin_contact.KEYS))
    names = document["meshes"][0]["extras"]["targetNames"]
    assert names == list(groin_contact.NAMES)
    animation = document["animations"][0]
    channel = next(c for c in animation["channels"] if c["target"]["path"] == "weights")
    sampler = animation["samplers"][channel["sampler"]]
    stored_weights = accessor(document, data, sampler["output"]).reshape(len(poses), len(names))
    targets = groin_contact.targets(rig.base, rig.verts, rig.bones, rig.weights)
    for i, pose in enumerate(poses):
        driven = groin.pose_weights(pose, groin_contact.KEYS)
        np.testing.assert_allclose(stored_weights[i], [driven[n] for n in names], atol=1e-7)
        actual = skin_stored(document, data, pose, morph_weights=driven)[: len(rig.verts)]
        expected = (
            to_gltf(
                pose_verts(
                    rig.bones, rig.weights, groin.bind_verts(rig.verts, targets, pose), pose
                )
            )
            * 1000
        )
        np.testing.assert_allclose(actual, expected, atol=0.03, rtol=0)
    primitive = document["meshes"][0]["primitives"][0]
    assert all("NORMAL" in target for target in primitive["targets"])


@pytest.mark.bake
def test_hanging_arm_targets_replay_on_top_of_the_shoulder_and_are_baked_into_animation():
    from sotaigen.contact.axilla_cache import targets as shoulder_targets
    from sotaigen.corrective import hang as hang_driver

    rig = rigged("female-long")
    base = {**rig.base, **export_fields(rig.base["faces"])}
    hang, table = hang_contact.targets(rig.base, rig.verts, rig.bones, rig.weights)
    shoulder = shoulder_targets(rig.base, rig.verts, rig.bones, rig.weights)
    # The IPC shoulder is a simplex table with its own targets and driver, read through the correctives'
    # own table as the file is; axilla.pose_weights alone names the sculpted shoulder's targets.
    correctives = Correctives(shoulder, hang=hang, hang_table=table)
    bottom = table["adductionKeys"][-1]
    poses = [{}, {"leftUpperArm": (0.0, 0.0, -bottom)}, {"upperArm": (12.0, 0.0, -27.0)},
             {"leftUpperArm": (7.0, 0.0, -bottom), "rightUpperArm": (0.0, 0.0, 26.0)}]
    document, data = read_glb(glb_bytes(
        base, rig.verts, rig.bones, rig.weights, correctives=correctives, motions=[("hang", 30, poses)]))
    assert groups_of(document, "hang") == unsourced(hang_driver.groups(table))
    names = document["meshes"][0]["extras"]["targetNames"]
    assert set(hang) <= set(names)
    animation = document["animations"][0]
    channel = next(c for c in animation["channels"] if c["target"]["path"] == "weights")
    sampler = animation["samplers"][channel["sampler"]]
    stored_weights = accessor(document, data, sampler["output"]).reshape(len(poses), len(names))
    for i, pose in enumerate(poses):
        driven = correctives.pose_weights(pose)
        np.testing.assert_allclose(stored_weights[i], [driven.get(n, 0.0) for n in names], atol=1e-7)
        actual = skin_stored(document, data, pose, morph_weights=driven)[: len(rig.verts)]
        expected = to_gltf(pose_verts(rig.bones, rig.weights, correctives.bind(rig.verts, pose), pose)) * 1000
        np.testing.assert_allclose(actual, expected, atol=0.03, rtol=0)


@pytest.mark.slow
def test_the_same_parameters_give_the_same_file_whichever_way_it_is_made(tmp_path, keeps_the_export):
    """The library's file of a typical body's parameters is the export's, metadata and all."""
    import sotaigen

    made = sotaigen.write_glb(sotaigen.generate(sotaigen.typical(NAME)), tmp_path / "made.glb")
    assert made.read_bytes() == export_with_face(NAME).read_bytes()
