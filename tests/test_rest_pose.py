"""Another rest pose at export: the angles that reach it, and the file that takes it as its rest.

The bones reach what they reached from the A pose for any pose, so the skin is what's tested: at the new
rest and back at the A pose it must match sotaigen.rig.skinning.pose_verts with the correctives bound to float
precision, opened with Auto Run Python Scripts off. Away from both rests linear blend skinning from the
new rest differs from skinning from the A pose (docs/systems/blend.md); that is held under a ceiling.
Tests marked `blender` are left out when there is no Blender (tests/conftest.py).
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from sotaigen import cli
from sotaigen.corrective.recipe import Correctives
from sotaigen.form.bodies import body_specs, load_specs
from sotaigen.output import blend as blend_module
from sotaigen.output.blend import blend_body, write_bundle
from sotaigen.output.blender import run_blender, script
from sotaigen.output.face import face_parts
from sotaigen.output.glb import driver_table
from sotaigen.params import file as params_file
from sotaigen.pipeline import rigged, surface
from sotaigen.rig.limits import clipped, joint_limits
from sotaigen.rig.poses import check_poses
from sotaigen.rig.rest_pose import (
    CLEAN,
    T_POSE,
    PastLimits,
    add_arguments,
    asset_reference,
    euler_of,
    in_new_rest,
    request_of,
    solve,
    world_frames,
)
from sotaigen.rig.skinning import euler, pose_verts, rest_matrix
from sotaigen.rig.touch import blend_poses, solve_all
from sotaigen.uv import export_fields

NAMES = list(load_specs("typical"))
T = {"arm_abduction": T_POSE["armAbduction"], "palms_down": T_POSE["palmsDown"], "straight_arms": T_POSE["straightArms"]}
# tests/test_blend.py's ceiling: Blender's 32-bit floats in metres.
SKIN_MM = 0.003
KEY_WEIGHT = 1e-5
# The mesh's own coordinates aren't SKIN_MM's to hold. What must land within it is the skin under zero on
# the controls, which is a coordinate times its bones' matrices over their rest frames; Blender rounds a new
# rest frame by up to 5e-6 rad (0.016 mm at a fingertip), so the coordinate takes the opposite of that for the
# skin to land right. 0.0039 mm measured over the twelve bodies; half as much again, as SKIN_MM has over its
# own measurement, is this ceiling. No existing ceiling moved: tests/test_blend.py has no such check.
# The skin in the new rest, with the shape keys off or on, takes it too: it goes through the rounded rest
# frames as well as the coordinates, and the opposite rounding of the coordinate only nearly cancels it.
# On the twelve bodies it measured 0.0008-0.0036 mm (male-balanced's right little fingertip, 2026-11-12),
# moved by up to 0.0014 mm when the bones moved by 0.001 mm: past SKIN_MM by where the rounding falls.
BASIS_MM = 0.006
# Away from both rests, the largest difference from the A pose's file, in the check and compound poses:
# 11.9 mm over the twelve bodies (docs/systems/blend.md). Past this something other than the order of the
# skinning changed.
FAR_MM = 15.0
# A pose asset's deform bones against the runtime's: the conversion is exact, this is the float chain's
# drift, 0.0038 mm measured with the limits off.
ASSET_BONE_MM = 0.01
# A pose asset's skin differs as FAR_MM's poses do, by the blending the user accepted; a hand on the crown is
# farther from the new rest than any of them, 19.9 mm on male-balanced, the largest over the twelve bodies.
ASSET_FAR_MM = 25.0
# Beyond the check poses, what most reads the correctives and coupling, as tests/test_blend.py has them.
FAR_POSES = {
    "armCompound": {"upperArm": (40, 30, 50), "lowerArm": (60, 0, 0), "hand": (20, 45, 10)},
    "armTwisted": {"upperArm": (60, 70, 72)},
    "asymmetric": {
        "leftUpperArm": (95, 0, 40),
        "rightUpperArm": (10, -20, -24),
        "leftUpperLeg": (95, 10, 5),
        "rightUpperLeg": (20, -15, -10),
    },
}


def bare_request(argv):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return request_of(parser.parse_args(argv))


# Without Blender -----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_the_t_pose_is_what_its_values_say(name):
    """Arms 90 degrees off vertical on both sides, the hand along them with its palm down, within the
    joint limits and the accepted clean range: the side raise's 55."""
    bones = rigged(name).bones
    rest = solve(bones, joint_limits(bones), **T)
    assert rest["pastLimits"] == [] and rest["unclean"] == []
    measured = rest["measured"]
    assert measured["armAbductionLeft"] == pytest.approx(90, abs=1e-4)
    assert measured["armAbductionRight"] == pytest.approx(90, abs=1e-4)
    assert rest["pose"]["upperArm"][2] == pytest.approx(55, abs=1.0)
    assert measured["handOffUpperArm"] < 1e-3 and measured["palmOffDown"] < 1e-3
    # A hinge can't take out the forearm's sideways lean (blend.md).
    assert measured["forearmOffUpperArm"] < 5.0
    frames = world_frames(bones, {k: tuple(v) for k, v in rest["pose"].items()})
    assert frames["leftUpperArm"][:3, 1] == pytest.approx((1, 0, 0), abs=1e-5)
    assert frames["leftHand"][:3, 1] == pytest.approx((1, 0, 0), abs=1e-5)
    assert frames["rightHand"][:3, 1] == pytest.approx((-1, 0, 0), abs=1e-5)


@pytest.mark.parametrize("name", NAMES)
def test_the_ankle_spacing_is_between_the_ankle_centres_and_the_soles_stay(name):
    bones = rigged(name).bones
    still = world_frames(bones, {})
    wide = abs(still["leftFoot"][0, 3]) * 2 + 80
    rest = solve(bones, joint_limits(bones), ankle_spacing=wide)
    pose = {k: tuple(v) for k, v in rest["pose"].items()}
    frames = world_frames(bones, pose)
    assert np.linalg.norm(frames["leftFoot"][:3, 3] - frames["rightFoot"][:3, 3]) == pytest.approx(wide, abs=1e-3)
    assert pose["upperLeg"][2] > 0
    for bone in ("leftFoot", "rightFoot", "leftToes", "rightToes"):
        # The foot keeps how it stood and where it stood in height: the hips go down by what the shin lifts.
        assert frames[bone][:3, :3] == pytest.approx(still[bone][:3, :3], abs=1e-9)
        assert frames[bone][2, 3] == pytest.approx(still[bone][2, 3], abs=1e-6)
    assert pose["rootTranslation"][2] < 0
    # Arms left alone.
    assert "upperArm" not in pose and "hand" not in pose


@pytest.mark.parametrize("abduction", [10.0, 20.0, 35.4, 45.0, 70.0, 110.0])
def test_any_arm_abduction_is_reached_on_both_sides(abduction):
    """Across the shoulder's kink at the rest pose as well; the arm stays in the frontal plane."""
    bones = rigged(NAMES[0]).bones
    rest = solve(bones, joint_limits(bones), arm_abduction=abduction, past_limits=True)
    assert rest["measured"]["armAbductionLeft"] == pytest.approx(abduction, abs=1e-4)
    assert rest["measured"]["armAbductionRight"] == pytest.approx(abduction, abs=1e-4)
    frames = world_frames(bones, {k: tuple(v) for k, v in rest["pose"].items()})
    assert frames["leftUpperArm"][1, 1] == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("name", NAMES)
def test_palms_down_only_rolls_the_hand(name):
    """At the A pose's arm: the hand keeps its axis. It hangs with its palm to the body, turned a little
    down, so that is nearly as far to the floor as a roll about that axis can face it."""
    bones = rigged(name).bones
    limits = joint_limits(bones)
    rest = solve(bones, limits, palms_down=True)
    assert set(rest["pose"]) == {"hand"} and rest["pastLimits"] == []
    assert abs(rest["pose"]["hand"][1]) < 5.0
    still = world_frames(bones, {})
    frames = world_frames(bones, {"hand": tuple(rest["pose"]["hand"])})
    assert frames["leftHand"][:3, 1] == pytest.approx(still["leftHand"][:3, 1], abs=1e-9)
    axis = frames["leftHand"][:3, 1]
    assert rest["measured"]["palmOffDown"] == pytest.approx(np.degrees(np.arcsin(abs(axis[2]))), abs=1e-6)


def test_past_the_joint_limits_is_refused_unless_allowed():
    bones = rigged(NAMES[0]).bones
    limits = joint_limits(bones)
    with pytest.raises(PastLimits, match="upperArm Z"):
        solve(bones, limits, arm_abduction=175.0)
    rest = solve(bones, limits, arm_abduction=175.0, past_limits=True)
    assert any((d["values"]["limb"], d["values"]["axis"]) == ("upperArm", "Z") for d in rest["pastLimits"])
    assert rest["measured"]["armAbductionLeft"] == pytest.approx(175, abs=1e-3)


def test_past_the_clean_range_only_warns():
    bones = rigged(NAMES[0]).bones
    limits = joint_limits(bones)
    high = solve(bones, limits, arm_abduction=120.0)
    assert high["pastLimits"] == [] and any((d["values"]["limb"], d["values"]["axis"]) == ("upperArm", "Z") for d in high["unclean"])
    # Past the thighs' touch at rest: narrower than the A pose's stance.
    frames = world_frames(bones, {})
    narrow = solve(bones, limits, ankle_spacing=float(frames["leftFoot"][0, 3]))
    assert narrow["pastLimits"] == [] and any((d["values"]["limb"], d["values"]["axis"]) == ("upperLeg", "Z") for d in narrow["unclean"])
    assert CLEAN["upperArm"]["z"][1] == 55.0


def test_a_rest_pose_needs_something_to_do():
    assert bare_request([]) is None
    with pytest.raises(ValueError, match="at least one"):
        solve(rigged(NAMES[0]).bones, {})


def test_the_t_pose_is_a_set_of_values_the_arguments_can_change():
    assert bare_request(["--t-pose"]) == {
        "arm_abduction": 90.0,
        "ankle_spacing": None,
        "palms_down": True,
        "straight_arms": True,
        "past_limits": False,
    }
    asked = bare_request(["--t-pose", "--arm-abduction", "80", "--ankle-spacing", "200", "--no-palms-down"])
    assert asked["arm_abduction"] == 80.0 and asked["ankle_spacing"] == 200.0 and asked["palms_down"] is False
    assert asked["straight_arms"] is True
    assert bare_request(["--ankle-spacing", "200"])["arm_abduction"] is None


def test_a_pose_in_the_new_rest_composes_back():
    held = {"upperArm": (-0.2, 0.0, 54.6), "hand": (-16.0, 0.5, 5.0), "rootTranslation": (0.0, 0.0, -3.0)}
    pose = {"upperArm": (40, 30, 50), "leftHand": (20, 45, 10), "neck": (10, 0, 0), "rootTranslation": (0, 5, 0)}
    turned = in_new_rest(held, pose)
    assert np.allclose(euler((-0.2, 0, 54.6)) @ euler(turned["leftUpperArm"]), euler((40, 30, 50)))
    assert np.allclose(euler((-0.2, 0, -54.6)) @ euler(turned["rightUpperArm"]), euler((40, -30, -50)))
    assert np.allclose(euler((-16.0, 0.5, 5.0)) @ euler(turned["leftHand"]), euler((20, 45, 10)))
    assert np.allclose(euler((-16.0, -0.5, -5.0)) @ euler(turned["rightHand"]), np.eye(4))
    assert np.allclose(turned["neck"], (10, 0, 0))
    assert turned["rootTranslation"] == pytest.approx((0, 5, 3))
    # The A pose itself: each held turn undone.
    back = in_new_rest(held, {})
    assert np.allclose(euler(held["upperArm"]) @ euler(back["leftUpperArm"]), np.eye(4))


def test_euler_of_inverts_the_skinning_euler():
    for angles in ((10, 20, 30), (-40, 80, 5), (0, -30, 120)):
        assert euler_of(euler(angles)) == pytest.approx(angles)


def test_without_a_rest_pose_the_bundle_is_what_it_was(tmp_path):
    name = NAMES[0]
    rig = rigged(name)
    base = {**rig.base, **export_fields(rig.base["faces"])}
    correctives = Correctives.production(base, rig.verts, rig.bones, rig.weights, field=surface(name)[0])
    spec = write_bundle(tmp_path / "a", name, base, rig.verts, rig.bones, rig.weights, correctives)
    assert "restPose" not in spec
    assert "restPose" not in json.loads((tmp_path / "a" / "input.json").read_text())
    rest = solve(rig.bones, joint_limits(rig.bones), **T)
    other = write_bundle(tmp_path / "b", name, base, rig.verts, rig.bones, rig.weights, correctives, rest=rest)
    assert other["restPose"]["pose"] == rest["pose"]
    assert {k: v for k, v in other.items() if k != "restPose"} == spec


@pytest.mark.slow
def test_the_parameter_files_command_takes_the_same_arguments(tmp_path, monkeypatch):
    """`sotaigen blend` hands Blender the rest pose, and with no arguments none."""
    kept = []

    def fake_blender(_script, scratch, out):
        kept.append(json.loads((Path(scratch) / "input.json").read_text()))
        Path(out).write_bytes(b"blend")

    monkeypatch.setattr(blend_module, "run_blender", fake_blender)
    file = tmp_path / "body.json"
    file.write_text(params_file.write_parameter_file(body_specs()["female-slim"], "female"))
    assert cli.main(["blend", str(file), str(tmp_path / "a.blend")]) == 0
    assert cli.main(["blend", str(file), str(tmp_path / "t.blend"), "--t-pose"]) == 0
    assert "restPose" not in kept[0]
    assert kept[1]["restPose"]["request"]["armAbduction"] == 90.0
    assert kept[1]["restPose"]["pose"]["upperArm"][2] == pytest.approx(55, abs=1.0)
    with pytest.raises(SystemExit):
        cli.main(["glb", str(file), str(tmp_path / "t.glb"), "--t-pose"])


def test_a_rest_pose_past_the_limits_stops_before_blender_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(blend_module, "run_blender", lambda *a, **k: pytest.fail("Blender ran"))
    with pytest.raises(PastLimits):
        blend_body(NAMES[0], tmp_path / "x.blend", {"arm_abduction": 175.0})
    assert not (tmp_path / "x.blend").exists()


# In the file ---------------------------------------------------------------------------------------


class Posed(NamedTuple):
    rig: object
    correctives: Correctives
    limits: dict
    rest: dict
    poses: dict
    data: dict
    info: dict


def reference(body, pose):
    return pose_verts(body.rig.bones, body.rig.weights, body.correctives.bind(body.rig.verts, pose), pose)


def part_reference(rig, part, pose):
    weights = part.weights
    if weights is None:
        weights = np.zeros((len(part.verts), len(rig.bones)))
        weights[:, [b["name"] for b in rig.bones].index("head")] = 1.0
    return pose_verts(rig.bones, weights, np.asarray(part.verts, float), pose)


def farthest(a, b):
    return float(np.linalg.norm(np.asarray(a, float) - b, axis=1).max())


def write_and_pose(name, folder, request):
    rig = rigged(name)
    base = {**rig.base, **export_fields(rig.base["faces"])}
    correctives = Correctives.production(base, rig.verts, rig.bones, rig.weights, field=surface(name)[0])
    limits = joint_limits(rig.bones)
    rest = solve(rig.bones, limits, **request)
    path = blend_body(name, folder / f"{name}.blend", request)
    poses = {**check_poses(base, rig.verts, rig.bones, rig.weights), **FAR_POSES}
    asked = {pose: in_new_rest(rest["pose"], value) for pose, value in poses.items()}
    asked["newRest"] = {}
    (folder / f"{name}.json").write_text(
        json.dumps({"poses": asked, "central": driver_table(rig.bones)["central"], "limits": [False, True], "actions": True})
    )
    out = folder / f"{name}-posed"
    run_blender(script("pose"), folder / f"{name}.json", out, blend=path)
    with np.load(out.with_suffix(".npz")) as data:
        return name, Posed(rig, correctives, limits, rest, poses, dict(data), json.loads(out.with_suffix(".json").read_text()))


@pytest.fixture(scope="module")
def posed(tmp_path_factory):
    """Every typical body's T pose with the ankles apart, the whole request at once; two at a time."""
    folder = tmp_path_factory.mktemp("rest")

    def one(name):
        wide = float(np.linalg.norm(np.subtract(*(rest_matrix(b)[:3, 3] for b in rigged(name).bones if b["name"] in ("leftFoot", "rightFoot"))))) + 70
        return write_and_pose(name, folder, {**T, "ankle_spacing": wide})

    with ThreadPoolExecutor(2) as pool:
        return dict(pool.map(one, NAMES))


@pytest.mark.blender
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("limits", ["off", "on"])
def test_turned_back_to_the_a_pose_the_file_is_the_a_pose(posed, name, limits):
    """Every control turned back: the skin and the parts are the A pose's to float precision."""
    body = posed[name]
    prefix = f"{limits}:rest:"
    assert farthest(body.data[prefix + "body"], reference(body, {})) <= SKIN_MM
    for part in face_parts(name):
        assert farthest(body.data[prefix + part.name], part_reference(body.rig, part, {})) <= SKIN_MM
    # The correctives keep their names and order, and the rest keys after them are full on.
    keys = body.data[prefix + "keys"]
    names = body.info["keyNames"]
    carried = len(body.correctives.targets)
    assert names[:carried] == list(body.correctives.targets)
    assert names[carried:] and all(name.startswith("rest.") for name in names[carried:])
    assert np.abs(keys[carried:] - 1).max() <= KEY_WEIGHT


@pytest.mark.blender
@pytest.mark.parametrize("name", NAMES)
def test_zero_on_every_control_is_the_new_rest(posed, name):
    """The file's own mesh and rest bones are the pose, and zero on the controls stays there."""
    body = posed[name]
    held = {k: tuple(v) for k, v in body.rest["pose"].items()}
    want = reference(body, held)
    assert farthest(body.data["rest:body"], want) <= BASIS_MM
    assert farthest(body.data["off:newRest:body"], want) <= BASIS_MM
    assert farthest(body.data["on:newRest:body"], want) <= BASIS_MM
    assert np.abs(body.data["off:newRest:keys"]).max() <= KEY_WEIGHT
    for part in face_parts(name):
        assert farthest(body.data["rest:" + part.name], part_reference(body.rig, part, held)) <= BASIS_MM
        assert farthest(body.data["off:newRest:" + part.name], part_reference(body.rig, part, held)) <= BASIS_MM
    frames = world_frames(body.rig.bones, held)
    for i, bone in enumerate(body.info["deform"]):
        moved = frames[bone].copy()
        # The file keeps the root on the ground and moves what hangs from it; the root has no weights.
        if bone == "root":
            moved[:3, 3] -= held["rootTranslation"]
        moved[:3, 3] /= 1000
        assert np.abs(body.data["rest:bones"][i] - moved).max() < 2e-5, bone
        assert np.abs(body.data["rest:controls"][i] - body.data["rest:bones"][i]).max() < 2e-5, bone
        # The deform bone's frame under zero controls is its rest.
        assert np.abs(body.data["off:newRest:bones"][i] - body.data["rest:bones"][i]).max() < 2e-5, bone
    # The file says how to get back, and it agrees with the formula.
    stored = body.info["info"]["restPose"]["aPose"]
    back = in_new_rest(held, {})
    for bone, row in stored.items():
        assert np.allclose(euler(row["rotation"])[:3, :3], euler(back.get(bone, (0, 0, 0)))[:3, :3], atol=1e-5), bone


@pytest.mark.blender
@pytest.mark.parametrize("name", NAMES)
def test_the_file_with_a_new_rest_carries_no_python(posed, name):
    info = posed[name].info
    assert info["autoexec"] is False and info["autoexecFail"] is False
    assert info["texts"] == 0
    assert info["notSimple"] == []
    assert info["constraints"]["LIMIT_ROTATION"] == len(posed[name].limits)


@pytest.mark.blender
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("limits", ["off", "on"])
def test_away_from_both_rests_the_skin_stays_near_the_a_poses(posed, name, limits):
    """The bones reach the same place for any pose, the joint limits clipping as they do in the A pose's
    file; the skin differs by the order of the blending, and not at all where only one bone moves it."""
    body = posed[name]
    for pose_name, pose in body.poses.items():
        given = clipped(pose, body.limits) if limits == "on" else pose
        assert farthest(body.data[f"{limits}:{pose_name}:body"], reference(body, given)) <= FAR_MM, pose_name
        # The head's parts hang from one bone.
        for part in face_parts(name):
            assert farthest(body.data[f"{limits}:{pose_name}:{part.name}"], part_reference(body.rig, part, given)) <= SKIN_MM


@pytest.mark.blender
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("limits", ["off", "on"])
def test_the_pose_assets_are_carried_into_the_new_rest(posed, name, limits):
    """Each contact's asset stores Q = R^-1 P, so played on the new rest its bones reach what they do in the A
    pose's file, every control it doesn't key still holding the rest. The skin differs as it does away from
    both rests."""
    body = posed[name]
    base = {**body.rig.base, **export_fields(body.rig.base["faces"])}
    assets = blend_poses(solve_all(base, body.rig.verts, body.rig.bones, body.rig.weights))
    assert assets
    for asset, pose in assets.items():
        given = asset_reference(body.rest["pose"], pose)
        if limits == "on":
            given = clipped(given, body.limits)
        assert farthest(body.data[f"{limits}:{asset}:body"], reference(body, given)) <= ASSET_FAR_MM, asset
        frames = world_frames(body.rig.bones, given)
        for i, bone in enumerate(body.info["deform"]):
            if bone != "root":
                at = body.data[f"{limits}:{asset}:bones"][i][:3, 3].astype(float) * 1000
                assert np.linalg.norm(at - frames[bone][:3, 3]) <= ASSET_BONE_MM, (asset, bone)


@pytest.mark.blender
def test_a_rest_pose_past_its_limit_is_still_the_rest(tmp_path):
    """Let past a joint limit, the limit stretches to the rest pose, so zero on the controls stays there
    with the limits on; the file also says so."""
    _, body = write_and_pose(NAMES[0], tmp_path, {"arm_abduction": 175.0, "past_limits": True})
    held = {k: tuple(v) for k, v in body.rest["pose"].items()}
    want = reference(body, held)
    assert farthest(body.data["on:newRest:body"], want) <= BASIS_MM
    assert farthest(body.data["off:newRest:body"], want) <= BASIS_MM
    assert body.info["info"]["restPose"]["pastLimits"]
    assert any(row["code"] == "rest.past_limit" for row in body.info["info"]["quality"]["found"])
    assert "rest" in body.info["info"]["quality"]["checked"]


def test_the_bone_names_are_read_from_the_file():
    """The Blender side names no bone of the rig: it goes by the armature and the rest pose it's given."""
    source = (Path(__file__).resolve().parents[1] / "blender" / "src" / "sotaigen_blender" / "rebase.py").read_text()
    for word in ("UpperArm", "UpperLeg", "Hand", "Foot", "hips", "Shoulder"):
        assert word not in source, word
