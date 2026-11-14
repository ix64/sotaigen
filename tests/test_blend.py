"""The rigged .blend poses as the runtime does, opened with Auto Run Python Scripts off.

Each typical body's file is written by the command's own entry point (sotaigen.output.blend.blend_body), then
opened in a fresh Blender with factory settings and its controls turned to every check pose, with the
joint limits off and on. The skin and the head's parts must match sotaigen.rig.skinning.pose_verts with
Correctives.production's targets bound, the shape keys their weights, and the limits must clip a pose
as their table says. Tests marked `blender` are left out when there is no Blender (tests/conftest.py).
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from gltf_support import LIMIT_POSES

from sotaigen.corrective.recipe import Correctives
from sotaigen.diagnostics import as_json
from sotaigen.form.bodies import load_specs
from sotaigen.output.blend import blend_body, make_blend
from sotaigen.output.blender import run_blender, script
from sotaigen.output.face import face_parts
from sotaigen.output.glb import driver_table
from sotaigen.output.quality import BrokenBody, broken, quality
from sotaigen.pipeline import BUILD, rigged, surface, template
from sotaigen.rig.limits import clipped, joint_limits, rest_angles
from sotaigen.rig.poses import check_poses
from sotaigen.rig.skinning import bone_local, pose_verts, rest_matrix
from sotaigen.rig.touch import blend_poses, solve_all
from sotaigen.uv import export_fields

NAMES = list(load_specs("typical"))
# Blender skins in 32-bit floats, in metres: one step at 1 m is 6e-5 mm, and a vertex goes through
# some ten chained products, so 1e-6 of a 2 m body is 0.002 mm. The probe measured 0.0009 mm on one
# body (docs/systems/blend.md); the twelve typical bodies measure 0.0005-0.0021 mm, limits on or off.
# Half as much again is the ceiling; a missing mechanism moves the skin by millimetres.
SKIN_MM = 0.003
# The probe's largest corrective weight error was 2e-6.
KEY_WEIGHT = 1e-5
# A corrective missing from the file would move the skin at least this much in these poses.
CORRECTIVE_POSES = ("armsHigh", "reachHigh", "crouch")
CORRECTIVE_MM = 1.0
# Each turn as the joint-limit table reads it: (motion, base pose, turned pose, landmark on the
# wearer's left, which point, world axis, the sign it moves on the left). The right mirrors X.
AXES = [
    ("shoulder flexion is upperArm X+: the elbow goes forward", {}, {"upperArm": (30, 0, 0)}, "LowerArm", "head", 1, -1),
    ("shoulder abduction is upperArm Z+: the elbow goes up", {}, {"upperArm": (0, 0, 30)}, "LowerArm", "head", 2, 1),
    (
        "internal rotation is upperArm Y+: the bent forearm turns to the midline",
        {"lowerArm": (90, 0, 0)},
        {"lowerArm": (90, 0, 0), "upperArm": (0, 30, 0)},
        "Hand",
        "head",
        0,
        -1,
    ),
    ("elbow flexion is lowerArm X+: the wrist goes forward", {}, {"lowerArm": (30, 0, 0)}, "Hand", "head", 1, -1),
    ("wrist flexion is hand X+: the fingers go to the palm", {}, {"hand": (30, 0, 0)}, "MiddleDistal", "tail", 0, -1),
    ("radial deviation is hand Z+: the fingers go to the thumb", {}, {"hand": (0, 0, 30)}, "MiddleDistal", "tail", 1, -1),
    ("pronation is hand Y+: the thumb turns to the midline", {}, {"hand": (0, 30, 0)}, "ThumbDistal", "tail", 0, -1),
    ("finger flexion is X+: the tip goes to the palm", {}, {"indexProximal": (30, 0, 0)}, "IndexDistal", "tail", 0, -1),
    ("hip flexion is upperLeg X+: the knee goes forward", {}, {"upperLeg": (30, 0, 0)}, "LowerLeg", "head", 1, -1),
    ("hip abduction is upperLeg Z+: the knee goes out", {}, {"upperLeg": (0, 0, 30)}, "LowerLeg", "head", 0, 1),
    ("hip internal rotation is upperLeg Y+: the toes turn in", {}, {"upperLeg": (0, 30, 0)}, "Toes", "tail", 0, -1),
    ("knee flexion is lowerLeg X-: the ankle goes back", {}, {"lowerLeg": (-30, 0, 0)}, "Foot", "head", 1, 1),
    ("dorsiflexion is foot X+: the toes go up", {}, {"foot": (20, 0, 0)}, "Toes", "head", 2, 1),
    ("eversion is foot Y+: the sole turns out", {}, {"foot": (0, 20, 0)}, "Foot", "sole", 0, 1),
    ("lumbar flexion is spine X+: the neck goes forward", {}, {"spine": (20, 0, 0)}, "neck", "head", 1, -1),
    ("neck flexion is neck X+: the head goes forward", {}, {"neck": (20, 0, 0)}, "head", "tail", 1, -1),
    ("mouth opening is jaw X+: the chin goes down", {}, {"jaw": (20, 0, 0)}, "jaw", "tail", 2, -1),
    ("elevation is eye X+: the gaze goes up", {}, {"eye": (20, 0, 0)}, "Eye", "tail", 2, 1),
    ("abduction is eye Z+: the gaze goes out", {}, {"eye": (0, 0, 20)}, "Eye", "tail", 0, 1),
]
MIDLINE = ("neck", "head", "jaw")
# A turn that moves its landmark less than this checks nothing.
AXIS_MOVES_MM = 5.0
# Rx(90) Rz(-60) of the left upper arm, in the control's own Euler angles, the right mirrored.
ACROSS_PAST = (90.0, 60.0, 0.0)
# Beyond the check poses, which turn the arms and legs about one axis but for two: turns about all
# three at once, where reading the correctives by swing and twist differs most from reading Euler
# angles, a side's twist past the shoulder table's and keys between the tables'.
COMPOUND_POSES = {
    "armCompound": {"upperArm": (40, 30, 50), "lowerArm": (60, 0, 0), "hand": (20, 45, 10)},
    "armBehind": {"upperArm": (-40, -30, 20)},
    "armForwardUp": {"upperArm": (50, -15, 60), "shoulder": (0, 0, 8)},
    "armTwisted": {"upperArm": (60, 70, 72)},
    "hangBetweenKeys": {"upperArm": (12.5, 5, -27), "lowerArm": (8, 0, 0)},
    "asymmetric": {
        "leftUpperArm": (95, 0, 40),
        "rightUpperArm": (10, -20, -24),
        "leftUpperLeg": (95, 10, 5),
        "rightUpperLeg": (20, -15, -10),
    },
    "sit": {"upperLeg": (85, 15, 12), "lowerLeg": (-90, 0, 0), "spine": (5, 7, 0)},
    # Past the upper arm's limit, which splits the rotation as YZX, so that the limits-on read holds
    # them there as sotaigen.rig.limits.held says: a side raise, and a level reach 60 past the midline.
    "armPastSide": {"upperArm": (0, 0, 140)},
    "armPastAcross": {"upperArm": ACROSS_PAST},
}


class Posed(NamedTuple):
    rig: object
    correctives: Correctives
    poses: dict
    limits: dict
    data: dict
    info: dict
    blend: Path


def axis_poses():
    out = {}
    for i, (_, base, turned, *_) in enumerate(AXES):
        out[f"axis{i}:base"], out[f"axis{i}:turned"] = base, turned
    return out


def limit_poses():
    return {f"limit{i}": pose for i, pose in enumerate(LIMIT_POSES)}


def pose_file(blend, poses, central, folder):
    request = folder / f"{blend.stem}-poses.json"
    request.write_text(json.dumps({"poses": poses, "central": central, "limits": [False, True], "actions": True}))
    out = folder / f"{blend.stem}-posed"
    run_blender(script("pose"), request, out, blend=blend)
    with np.load(out.with_suffix(".npz")) as data:
        return dict(data), json.loads(out.with_suffix(".json").read_text())


@pytest.fixture(scope="module")
def posed(tmp_path_factory):
    """Every typical body's file, written and posed; two at a time, as builds run on a shared machine."""
    folder = tmp_path_factory.mktemp("blend")

    def one(name):
        path = blend_body(name, folder / f"{name}.blend")
        rig = rigged(name)
        base = {**rig.base, **export_fields(rig.base["faces"])}
        correctives = Correctives.production(base, rig.verts, rig.bones, rig.weights, field=surface(name)[0])
        poses = check_poses(base, rig.verts, rig.bones, rig.weights)
        request = {**poses, **COMPOUND_POSES, **axis_poses(), **limit_poses()}
        data, info = pose_file(path, request, driver_table(rig.bones)["central"], folder)
        return name, Posed(rig, correctives, poses, joint_limits(rig.bones), data, info, path)

    with ThreadPoolExecutor(2) as pool:
        return dict(pool.map(one, NAMES))


def limits_of(rig):
    return joint_limits(rig.bones)


def reference(rig, correctives, pose):
    return pose_verts(rig.bones, rig.weights, correctives.bind(rig.verts, pose), pose)


def part_reference(rig, part, pose):
    names = [bone["name"] for bone in rig.bones]
    weights = part.weights
    if weights is None:
        weights = np.zeros((len(part.verts), len(names)))
        weights[:, names.index("head")] = 1.0
    return pose_verts(rig.bones, weights, np.asarray(part.verts, float), pose)


def worst(a, b):
    d = np.linalg.norm(np.asarray(a, float) - b, axis=1)
    return float(d.max()), float(np.percentile(d, 99))


@pytest.mark.blender
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("limits", ["off", "on"])
def test_the_file_poses_as_the_runtime_does(posed, name, limits):
    """With the limits off every check pose and compound turn as it is; on, as the limits clip it."""
    body = posed[name]
    names = list(body.correctives.targets)
    assert body.info["keyNames"] == names
    parts = face_parts(name)
    rows, key_error, part_error = {}, 0.0, 0.0
    for pose_name, pose in {**body.poses, **COMPOUND_POSES}.items():
        given = clipped(pose, body.limits) if limits == "on" else pose
        prefix = f"{limits}:{pose_name}:"
        rows[pose_name] = worst(body.data[prefix + "body"], reference(body.rig, body.correctives, given))
        weights = body.correctives.pose_weights(given)
        keys = np.abs(body.data[prefix + "keys"] - np.array([weights[n] for n in names])).max()
        assert keys <= KEY_WEIGHT, (pose_name, keys)
        key_error = max(key_error, float(keys))
        for part in parts:
            error = worst(body.data[prefix + part.name], part_reference(body.rig, part, given))[0]
            assert error <= SKIN_MM, (pose_name, part.name, error)
            part_error = max(part_error, error)
    largest = max(rows.items(), key=lambda row: row[1][0])
    print(
        f"{name} limits {limits}: skin max {largest[1][0]:.5f} mm ({largest[0]}), "
        f"p99 {max(r[1] for r in rows.values()):.5f} mm; parts {part_error:.5f} mm; keys {key_error:.1e}"
    )
    assert largest[1][0] <= SKIN_MM, largest


@pytest.mark.blender
@pytest.mark.parametrize("name", NAMES)
def test_a_fade_turned_to_its_limit_drives_no_key_in_the_file(posed, name):
    """The same weights as Python and the runtime (tests/test_export.py), down to the zeros at the limit."""
    from sotaigen.corrective import drive

    body = posed[name]
    names = list(body.correctives.targets)
    for pose_name, pose in limit_poses().items():
        turns = {"intents": drive.intents(pose), "read": {}}
        # The groups a fade at its limit shuts: their keys are exactly 0, not float32's 1e-32. The tents
        # of the other groups may still leave 1e-14 where Python has 0, inside KEY_WEIGHT.
        shut = {
            target
            for group in body.correctives.groups()
            if any(
                drive.fade_share(drive.reading(turns, fade["bone"], fade["read"]), fade["limit"]) == 0
                for fade in group.get("fades", [])
            )
            for target in group["targets"]
        }
        assert shut, pose
        weights = body.correctives.pose_weights(pose)
        keys = dict(zip(names, body.data[f"off:{pose_name}:keys"], strict=True))
        assert all(keys[target] == 0 and weights[target] == 0 for target in shut), pose
        assert max(abs(keys[n] - weights[n]) for n in names) <= KEY_WEIGHT, pose


@pytest.mark.blender
@pytest.mark.parametrize("name", NAMES)
def test_the_contact_poses_are_in_the_file_and_pose_as_the_runtime_does(posed, name):
    """Each hand-to-head contact is a pose asset; played on the rest pose, with the limits off or on,
    it is what pose_verts makes of the solved angles: the limits clip none of them."""
    body = posed[name]
    rig = body.rig
    assets = blend_poses(solve_all({**rig.base}, rig.verts, rig.bones, rig.weights))
    assert sorted(assets) == body.info["poseAssets"]
    assert assets
    for asset, pose in assets.items():
        for limits in ("off", "on"):
            error = worst(body.data[f"{limits}:{asset}:body"], reference(rig, body.correctives, pose))[0]
            assert error <= SKIN_MM, (asset, limits, error)


@pytest.mark.blender
@pytest.mark.parametrize("name", NAMES)
def test_the_correctives_are_in_the_file(posed, name):
    """The reverse check: without the bound targets the reference is far from what the file shows."""
    body = posed[name]
    for pose_name in CORRECTIVE_POSES:
        pose = body.poses[pose_name]
        bare = pose_verts(body.rig.bones, body.rig.weights, body.rig.verts, pose)
        assert worst(body.data[f"off:{pose_name}:body"], bare)[0] > CORRECTIVE_MM, pose_name


@pytest.mark.blender
@pytest.mark.parametrize("name", NAMES)
def test_the_file_carries_no_python(posed, name):
    """Opened with Auto Run Python Scripts off, nothing was refused, no script is in the file, and
    every driver is a simple expression or a plain average, which Blender evaluates without Python."""
    info = posed[name].info
    assert info["autoexec"] is False
    assert info["autoexecFail"] is False
    assert info["texts"] == 0
    assert info["notSimple"] == []
    # One driver per hub property and shape key, and one switch per joint limit.
    limits = joint_limits(posed[name].rig.bones)
    assert info["constraints"]["LIMIT_ROTATION"] == len(limits)
    assert {"jaw", "leftEye", "rightEye"} <= set(info["deform"])
    assert info["info"]["head"]["bones"] == ["head", "jaw", "leftEye", "rightEye"]
    assert info["drivers"] > len(posed[name].correctives.targets) + len(limits)


@pytest.mark.blender
@pytest.mark.parametrize("name", NAMES)
def test_quality_problems_travel_in_the_file(posed, name):
    body = posed[name]
    out = BUILD / "bodies" / name
    wrap = json.loads((out / "wrap.json").read_text())
    poses = json.loads((out / "poses.json").read_text())
    stored = body.info["info"]
    assert (stored["format"], stored["version"], stored["name"]) == ("sotaigen/blend", 1, name)
    found = quality(wrap, body.rig.verts, body.rig.base["faces"], poses)
    assert stored["quality"]["found"][: len(found)] == as_json(found)
    assert stored["quality"]["checked"] == ["wrap", "uv", "parameters", "poses", "touch"]
    assert stored["restFrameError"] < 1e-5


def landmark(body, pose_name, bone, kind):
    index = body.info["deform"].index(bone)
    matrix = body.data[f"off:{pose_name}:bones"][index]
    spec = next(b for b in body.rig.bones if b["name"] == bone)
    if kind == "head":
        return matrix[:3, 3] * 1000
    if kind == "tail":
        length = np.linalg.norm(np.subtract(spec["tail"], spec["head"])) / 1000
        return (matrix @ np.array([0.0, length, 0.0, 1.0]))[:3] * 1000
    # The sole's normal, turned as the bone turns, scaled to a 1 m lever.
    return matrix[:3, :3] @ rest_matrix(spec)[:3, :3].T @ np.array([0.0, 0.0, -1.0]) * 1000


@pytest.mark.blender
@pytest.mark.parametrize("name", NAMES)
def test_the_limits_axes_turn_the_way_their_motions_do(posed, name):
    """range-of-motion.md inferred each control axis's motion from the check poses; posed in Blender,
    each turns its landmark the way the joint-limit table assumes, on both sides."""
    body = posed[name]
    for i, (motion, _, _, bone, kind, axis, sign) in enumerate(AXES):
        for side, mirror in (("left", 1), ("right", -1)):
            if bone in MIDLINE and side == "right":
                continue
            full = bone if bone in MIDLINE else f"{side}{bone}"
            moved = landmark(body, f"axis{i}:turned", full, kind) - landmark(body, f"axis{i}:base", full, kind)
            expected = sign * (mirror if axis == 0 else 1)
            assert moved[axis] * expected > AXIS_MOVES_MM, (motion, side, moved)


# Without Blender -----------------------------------------------------------------------------------


def test_the_jaw_slides_as_it_opens_only_past_rest():
    rig = rigged(NAMES[0])
    jaw = next(bone for bone in rig.bones if bone["name"] == "jaw")
    assert limits_of(rig)["jaw"][0] == [0.0, 27.0]
    glide = rig.bones.glides["jaw"]
    assert np.allclose(bone_local(jaw, (10.0, 0.0, 0.0), glide)[:3, 3], np.asarray(glide) * 10.0)
    assert (bone_local(jaw, (-3.0, 0.0, 0.0), glide)[:3, 3] == 0).all()


def test_a_broken_body_is_refused_before_blender_runs(tmp_path):
    rig = rigged(NAMES[0])
    wrap = {"flippedFaces": 3, "crossedMidline": 0}
    found = broken(wrap, template()["faces"])
    assert found == ["flipped faces after the wrap: 3"]
    with pytest.raises(BrokenBody, match="flipped faces"):
        make_blend(tmp_path / "x.blend", "x", rig.base, rig.verts, rig.bones, rig.weights, None, broken_by=found)
    assert not (tmp_path / "x.blend").exists()


def test_the_bundle_says_what_it_is_and_the_scripts_name_things_alike(tmp_path):
    """sotaigen-blender is installed apart: the bundle carries its format, version and names, and the
    GPL scripts' constants are those names, which convert.py checks before it builds anything."""
    import ast

    from sotaigen.output import blender
    from sotaigen.output.blend import write_bundle

    rig = rigged(NAMES[0])
    base = {**rig.base, **export_fields(rig.base["faces"])}
    write_bundle(tmp_path, "x", base, rig.verts, rig.bones, rig.weights, Correctives())
    spec = json.loads((tmp_path / "input.json").read_text())
    assert (spec["format"], spec["version"], spec["names"]) == (blender.BUNDLE_FORMAT, blender.BUNDLE_VERSION, blender.NAMES)
    scripts = Path(__file__).resolve().parents[1] / "blender" / "src" / "sotaigen_blender"

    def constants(path):
        tree = ast.parse(path.read_text())
        return {
            target.id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
            for target in node.targets
            if isinstance(target, ast.Name)
        }

    rig_names = constants(scripts / "rig.py")
    by_key = {"rig": "RIG", "body": "BODY", "headRegion": "HEAD_REGION", "neckRing": "NECK_RING", "info": "INFO"}
    assert {key: rig_names[constant] for key, constant in by_key.items()} == blender.NAMES
    convert = ast.parse((scripts / "convert.py").read_text())
    pair = next(
        node.value for node in convert.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple)
        and [t.id for t in node.targets[0].elts] == ["BUNDLE_FORMAT", "BUNDLE_VERSION"]
    )
    assert (pair.elts[0].value, pair.elts[1].value) == (blender.BUNDLE_FORMAT, blender.BUNDLE_VERSION)


def test_an_open_skin_is_broken():
    faces = template()["faces"]
    assert broken({"flippedFaces": 0, "crossedMidline": 0}, faces) == []
    found = broken({"flippedFaces": 0, "crossedMidline": 0}, faces[1:])
    assert len(found) == 1 and found[0].startswith("the skin is not closed")


@pytest.mark.parametrize("name", NAMES)
def test_the_right_limits_mirror_the_left(name):
    limits = joint_limits(rigged(name).bones)
    for bone, rows in limits.items():
        if not bone.startswith("left"):
            continue
        right = limits["right" + bone.removeprefix("left")]
        assert right[0] == rows[0]
        for mine, theirs in zip(rows[1:], right[1:], strict=True):
            assert theirs == (None if mine is None else [-mine[1], -mine[0]])


@pytest.mark.parametrize("name", NAMES)
def test_the_upper_arm_rests_raised_35_degrees(name):
    """range-of-motion.md: the side raise's limit counts from hanging straight down."""
    rest = rest_angles(rigged(name).bones)
    assert rest["leftUpperArm"][1] == pytest.approx(35, abs=1)
    limits = joint_limits(rigged(name).bones)
    assert limits["leftUpperArm"][2] == [pytest.approx(-rest["leftUpperArm"][1], abs=1e-3), pytest.approx(150 - rest["leftUpperArm"][1], abs=1e-3)]


@pytest.mark.parametrize("name", NAMES)
def test_no_check_pose_is_past_the_range_of_motion(name):
    """The check poses are things people do, so the joint limits let every one through."""
    rig = rigged(name)
    limits = joint_limits(rig.bones)
    base = {**rig.base, **export_fields(rig.base["faces"])}
    for pose_name, pose in check_poses(base, rig.verts, rig.bones, rig.weights).items():
        assert clipped(pose, limits) == clipped(pose, {}), pose_name


def test_clipping_splits_a_limb_only_by_its_limits():
    limits = {"leftHand": [None, [-88.0, 80.0], None], "rightHand": [None, [-80.0, 88.0], None]}
    assert clipped({"hand": (0, 90, 0)}, limits) == {"leftHand": (0, 80.0, 0), "rightHand": (0, -80.0, 0)}
    assert clipped({"rootTranslation": (1, 2, 3)}, limits) == {"rootTranslation": (1, 2, 3)}


def fake_blender(path, says):
    path.write_text(f"#!/bin/sh\necho '{says}'\n")
    path.chmod(0o755)
    return path


@pytest.mark.parametrize(("says", "refused"), [("Blender 4.5.3 LTS", True), ("Blender 5.2.0", False),
                                               ("Blender 6.0.1", False), ("something else", False)])
def test_a_blender_older_than_the_floor_is_refused_before_anything_runs(tmp_path, monkeypatch, says, refused):
    """One that doesn't say its version gets through: the scripts check again inside Blender."""
    from sotaigen.errors import BlenderMissing, BlenderTooOld
    from sotaigen.output import blender

    monkeypatch.setenv("BLENDER", str(fake_blender(tmp_path / "blender", says)))
    why = blender.can_write_blend()
    assert (why is not None and "older than 5.2" in why) == refused
    if refused:
        with pytest.raises(BlenderTooOld) as raised:
            blender.run_blender("unused.py")
        assert isinstance(raised.value, BlenderMissing)


def test_the_command_line_exits_3_for_a_blender_too_old(tmp_path, monkeypatch, capsys):
    from sotaigen import cli

    monkeypatch.setenv("BLENDER", str(fake_blender(tmp_path / "blender", "Blender 4.2.0")))
    assert cli.main(["blend", str(tmp_path / "body.json"), str(tmp_path / "body.blend")]) == 3
    assert "older than 5.2" in capsys.readouterr().err


def test_both_packages_hold_the_same_blender_floor():
    import ast

    from sotaigen.output.blender import MINIMUM_BLENDER

    init = Path(__file__).resolve().parents[1] / "blender" / "src" / "sotaigen_blender" / "__init__.py"
    found = next(
        ast.literal_eval(node.value)
        for node in ast.parse(init.read_text()).body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "MINIMUM_BLENDER"
    )
    assert found == MINIMUM_BLENDER
    scripts = init.parent
    for name in ("convert", "pose", "head_swap"):
        assert "    require_blender()\n" in (scripts / f"{name}.py").read_text(), name
