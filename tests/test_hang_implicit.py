"""The default correctives keep a relaxed hanging arm's forearm and hand out of the hip and thigh.

The implicit-field table (sotaigen/corrective/hang_field.py) needs no bake, so these run on every body. The
upper arm against the chest side is the shoulder's table's, and is not counted here.
"""

from functools import cache

import numpy as np
import pytest

from sotaigen.corrective import hang as hang_driver
from sotaigen.corrective import hang_field
from sotaigen.corrective.crossings import SkinParts, edges_through_triangles
from sotaigen.corrective.recipe import Correctives
from sotaigen.mesh import opposite_vertices, triangulate, vertex_normals
from sotaigen.pipeline import bodies, rigged, surface
from sotaigen.rig.skinning import bone_local, pose_skins, pose_verts, rest_matrix, sided

LOWER = {"lowerArm", "hand"}
SIDE = {"trunk", "upperLeg"}


@cache
def shipped(name):
    rig = rigged(name)
    return rig, Correctives.production(
        rig.base, rig.verts, rig.bones, rig.weights, field=surface(name)[0]
    )


def lower_arm_crossings(rig, correctives, pose):
    parts = SkinParts(rig.base)
    posed = pose_verts(rig.bones, rig.weights, correctives.bind(rig.verts, pose), pose)
    count = 0
    for edge, triangle in edges_through_triangles(posed, parts.edges, parts.triangles).tolist():
        pair = {str(parts.groups[parts.edges[edge, 0]]), str(parts.groups[parts.triangles[triangle, 0]])}
        if pair & LOWER and pair & SIDE:
            count += 1
    return count


def both_arms(forward, adduction):
    return hang_driver.arm_pose("left", forward, adduction) | hang_driver.arm_pose("right", forward, adduction)


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(bodies()))
def test_both_forearms_and_hands_hang_to_the_target_without_crossing(name):
    rig, correctives = shipped(name)
    table = correctives.hang_table
    assert table is not None and correctives.contact == "sculpted"
    bottom = table["adductionKeys"][-1]
    assert hang_driver.rest_tilt(rig.bones) - bottom == pytest.approx(hang_driver.FROM_VERTICAL, abs=0.01)
    # At keys and between them, forward and down. Between the forward keys 0 and 5, and 5 and 10,
    # is where a narrow gap at the keys let the eased morphs cross again.
    middle = (bottom + table["adductionKeys"][-2]) / 2
    for forward, adduction in ((0, bottom), (2.5, bottom), (3.75, middle), (7.5, bottom),
                               (8.75, bottom), (12.5, (bottom + 24) / 2), (45, bottom),
                               (0, bottom - 1.0)):
        pose = both_arms(forward, adduction)
        assert lower_arm_crossings(rig, correctives, pose) == 0, (forward, adduction)


def test_without_the_table_the_forearm_crosses_there():
    """The check above can fail: the same poses cross without the table.

    On a body whose hips stand wider than its shoulders; on the others the arm hangs clear anyway.
    """
    rig, correctives = shipped("female-widehip")
    bare = Correctives(correctives.shoulder, correctives.groin)
    bottom = correctives.hang_table["adductionKeys"][-1]
    assert lower_arm_crossings(rig, bare, both_arms(0, bottom)) > 0


def test_the_table_is_driven_as_the_ipc_one_and_leaves_the_standing_body_alone():
    rig, correctives = shipped("female-long")
    table = correctives.hang_table
    assert table == hang_driver.driver_metadata(hang_driver.adduction_keys(rig.bones))
    names = [n for side in hang_driver.SIDES for n in table["sides"][side]["targets"]]
    assert list(correctives.hang) == names
    assert correctives.bind(rig.verts, {}) is rig.verts
    for pose in ({"upperArm": (0.0, 0.0, -24.0)}, {"upperArm": (90.0, 0.0, 0.0)},
                 both_arms(0, table["adductionKeys"][-1]) | {"lowerArm": (30.0, 0, 0)}):
        assert not any(hang_driver.pose_weights(pose, table).values()), pose


def test_the_right_side_mirrors_the_left_on_a_symmetric_body():
    rig, correctives = shipped("female-long")
    table, hang = correctives.hang_table, correctives.hang
    opposite = opposite_vertices(rig.verts)
    assert opposite is not None
    for a, b in zip(table["sides"]["left"]["targets"], table["sides"]["right"]["targets"], strict=True):
        np.testing.assert_array_equal(hang[b][opposite], hang[a] * hang_driver.FLIP)


def test_the_table_solved_in_parts_is_the_whole_to_the_bit():
    """What the editor relies on to solve it in two processes."""
    rig, correctives = shipped("female-long")
    table = hang_field.Table(rig.base, rig.verts, rig.bones, rig.weights, surface("female-long")[0])
    first = table.solve(correctives.shoulder, hang_driver.FORWARD[::2])
    second = table.solve(correctives.shoulder, hang_driver.FORWARD[1::2])
    assert not first.keys() & second.keys()
    assert first.keys() | second.keys() == correctives.hang.keys()
    for name, delta in (first | second).items():
        assert delta.tobytes() == correctives.hang[name].tobytes(), name
    # And their normal deltas, made for some of the targets at a time.
    triangles = triangulate(rig.base["faces"], np.asarray(rig.base["verts"]))
    normals = vertex_normals(rig.verts, triangles)
    args = (rig.base, rig.verts, rig.bones, rig.weights, correctives.shoulder, correctives.hang,
            correctives.hang_table, normals)
    some = set(list(second)[:6])
    part = hang_driver.normal_targets(*args, triangles=triangles, names=some)
    whole = hang_driver.normal_targets(*args, triangles=triangles)
    assert part.keys() == some
    for name in some:
        assert part[name].tobytes() == whole[name].tobytes(), name


def test_the_default_needs_the_field():
    rig = rigged("female-long")
    with pytest.raises(TypeError, match="field"):
        Correctives.production(rig.base, rig.verts, rig.bones, rig.weights)


def fresh_skins(bones, pose):
    """pose_skins as it reads, with every rest frame and inverse worked out again."""
    rotations = sided(pose)
    rest = {bone["name"]: rest_matrix(bone) for bone in bones}
    world = {}
    for bone in bones:
        name = bone["name"]
        local = bone_local(bone, rotations.get(name, (0.0, 0.0, 0.0)), bones.glides.get(name))
        if bone["parent"]:
            parent = bone["parent"]
            world[name] = world[parent] @ np.linalg.inv(rest[parent]) @ rest[name] @ local
        else:
            world[name] = rest[name] @ local
            world[name][:3, 3] += pose.get("rootTranslation", (0, 0, 0))
    return np.array([world[b["name"]] @ np.linalg.inv(rest[b["name"]]) for b in bones])


def test_kept_rest_frames_give_the_skins_to_the_bit():
    rig = rigged("female-long")
    rng = np.random.default_rng(0)
    poses = [{}, both_arms(10, 30), {"rootTranslation": (1.0, 2.0, 3.0), "spine": (0.0, 0.0, 0.0)}]
    poses += [{"jaw": (20.0, 0.0, 3.0)}]
    poses += [
        {"leftUpperArm": tuple(rng.uniform(-90, 90, 3)), "rightLowerArm": tuple(rng.uniform(-90, 90, 3)),
         "upperLeg": tuple(rng.uniform(-60, 60, 3)), "hand": tuple(rng.uniform(-40, 40, 3))}
        for _ in range(40)
    ]
    for pose in poses:
        expected = fresh_skins(rig.bones, pose).tobytes()
        assert pose_skins(rig.bones, pose).tobytes() == expected, pose
        assert pose_skins(rig.bones, pose).tobytes() == expected, pose


@pytest.mark.parametrize("name", ["female-balanced", "male-balanced", "female-long"])
def test_the_body_side_is_the_field_away_from_the_arm(name):
    """B drops the arm groups and nothing else: the breast, the pectoral plate and the profiles stay."""
    from scipy.spatial import cKDTree

    from sotaigen.form.anatomy import calibrated
    from sotaigen.form.bodies import load_bodies
    from sotaigen.params.rules import RANGES, sex_of
    from sotaigen.params.spec import GLUTE_CONTROLS

    rig = rigged(name)
    if name == "female-long":
        # The glute contours at the top of their ranges, a layer no typical body has.
        p = load_bodies()[name]
        field, _ = calibrated(p | {key: RANGES[sex_of(name)][key][1] for key in GLUTE_CONTROLS})
    else:
        field = surface(name)[0]
    assert "reference" in vars(field), "the body should have a layer over its groups"
    fields = hang_field.PosedFields(field, rig.bones, "left", None)
    arm = np.isin(SkinParts(rig.base).groups, ["upperArm", "lowerArm", "hand"])
    far = cKDTree(rig.verts[arm]).query(rig.verts[~arm])[0] > 0.15 * float(rig.L["T"])
    # Around the shoulder the deltoid, an arm mass, fills the body's side: B leaves it out on purpose.
    low = rig.verts[~arm][:, 2] < rig.L["pit"] - 0.2 * float(rig.L["T"])
    skin = rig.verts[~arm][far & low]
    # On the skin and a little inside and outside it, where the layers act.
    points = np.concatenate([skin + (0.0, dy, 0.0) for dy in (-4.0, 0.0, 4.0)])
    np.testing.assert_allclose(fields.B(points), field(points), atol=1e-9, rtol=0)
