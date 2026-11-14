"""A captured clip on a body's .blend: fitted to the controls, keyed on them, read back from the file.

sotaigen_dev/motion/clips.py's retarget is free to write any local rotation, which the runtime takes. The
delivered controls have hinges at the elbow and knee and carry the forearm's turn on the hand, so the
clip is fitted to them first (sotaigen_dev/motion/blend.py), and a keyed file leaves its limits off. Tests
marked `blender` are left out when there is no Blender (tests/conftest.py).
"""

from functools import cache

import numpy as np
import pytest

from sotaigen.corrective.recipe import Correctives
from sotaigen.form.bodies import load_specs
from sotaigen.output.blend import blend_body
from sotaigen.pipeline import rigged, surface
from sotaigen.rig.limits import clipped, held, joint_limits, limit_order, split_yzx
from sotaigen.rig.skinning import euler, pose_verts, rest_matrix
from sotaigen_dev.motion.blend import (
    CHAINS,
    SAMPLE_EVERY,
    controls_ready,
    key_and_read,
    world_deltas,
)
from sotaigen_dev.motion.clips import clip_names, load_clip, playback_samples
from sotaigen_dev.motion.foot_contact import contact_errors

NAMES = list(load_specs("typical"))
# Both ends of the typical bodies' proportions.
ENDS = [NAMES[0], NAMES[-1]]
# What fitting may cost the planted feet, in mm: measured at most 1.6 and 3.5 on these clips and bodies.
# The retarget's own contact solve holds them to 0.03; a fit that moved the ankles would cost centimetres.
FEET_FLOOR_MM = 2.5
FEET_STEP_MM = 5.0
# How far a limb's lower part may leave its captured direction, in degrees: the twist is smoothed
# over a few frames, because it's ill-defined while the limb is nearly straight. Measured at most 2.8.
LOWER_LIMB_DEGREES = 4.0
# Blender skins in 32-bit floats, in metres; a clip adds up to some metres of travel to what
# tests/test_blend.py's SKIN_MM allows at rest. Measured 0.0042 over 3 s of walking.
KEYED_SKIN_MM = 0.006


@cache
def fitted(name, clip_name):
    rig = rigged(name)
    clip = load_clip(clip_name)
    capture = clip.get("capture", clip)
    fps, poses = playback_samples(clip, rig)
    limits = joint_limits(rig.bones)
    return rig, capture, fps, poses, controls_ready(poses, rig.bones, limits, capture["rest"])


def angle_between(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1, 1)))


@pytest.mark.slow
@pytest.mark.parametrize("name", ENDS)
@pytest.mark.parametrize("clip_name", clip_names())
def test_a_fitted_clip_has_hinges_where_the_controls_do(name, clip_name):
    rig, _, _, _, poses = fitted(name, clip_name)
    limits = joint_limits(rig.bones)
    for _, lower, _ in CHAINS:
        angles = np.array([pose[lower] for pose in poses])
        assert np.all(angles[:, 1:] == 0), lower
        assert limits[lower][1] == [0.0, 0.0] and limits[lower][2] == [0.0, 0.0], lower
        low, high = limits[lower][0]
        assert angles[:, 0].min() >= low - 1e-3 and angles[:, 0].max() <= high + 1e-3, lower


@pytest.mark.parametrize("name", ENDS)
@pytest.mark.parametrize("clip_name", clip_names())
def test_fitting_keeps_the_limbs_directions(name, clip_name):
    rig, _, _, before, after = fitted(name, clip_name)
    rests = {bone["name"]: rest_matrix(bone)[:3, :3] for bone in rig.bones}
    old = [world_deltas(pose, rig.bones, rests) for pose in before]
    new = [world_deltas(pose, rig.bones, rests) for pose in after]
    for upper, lower, _ in CHAINS:
        for bone, ceiling in ((upper, 1e-2), (lower, LOWER_LIMB_DEGREES)):
            axis = rests[bone][:, 1]
            was = np.array([delta[bone] @ axis for delta in old])
            now = np.array([delta[bone] @ axis for delta in new])
            assert angle_between(was, now).max() <= ceiling, bone


@pytest.mark.parametrize("name", ENDS)
@pytest.mark.parametrize("clip_name", clip_names())
def test_fitting_keeps_the_planted_feet(name, clip_name):
    rig, capture, _, _, poses = fitted(name, clip_name)
    errors = contact_errors(rig, poses, capture["footContacts"])
    assert errors["worstFloorErrorMM"] <= FEET_FLOOR_MM
    assert errors["worstMarkerStepMM"] <= FEET_STEP_MM


@pytest.mark.parametrize("clip_name", clip_names())
def test_a_wrists_median_is_its_neutral(clip_name):
    *_, poses = fitted(ENDS[0], clip_name)
    for hand in ("leftHand", "rightHand"):
        angles = np.array([pose[hand] for pose in poses])
        assert np.abs(np.median(angles[:, [0, 2]], axis=0)).max() < 0.01, hand


def test_only_the_upper_arm_splits_its_limit_differently():
    orders = {name: limit_order(name) for name in ("leftUpperArm", "rightUpperArm", "leftHand", "leftUpperLeg", "neck")}
    assert orders == {
        "leftUpperArm": "YZX",
        "rightUpperArm": "YZX",
        "leftHand": "XYZ",
        "leftUpperLeg": "XYZ",
        "neck": "XYZ",
    }


@pytest.mark.parametrize("angles", [(33, -20, 47), (95, 40, -60), (-40, 30, 20), (0, 0, 110), (120, -70, 15)])
def test_the_yzx_split_gives_the_rotation_back(angles):
    matrix = euler(angles)[:3, :3]
    x, y, z = np.degrees(split_yzx(matrix))
    again = euler((x, 0, 0))[:3, :3] @ euler((0, 0, z))[:3, :3] @ euler((0, y, 0))[:3, :3]
    assert np.abs(again - matrix).max() < 1e-12


@pytest.mark.parametrize("single", [(0, 0, 110), (95, 0, 0), (-30, 0, 0), (0, 50, 0), (0, -70, 0)])
def test_a_turn_about_one_axis_reads_the_same_in_the_upper_arms_order(single):
    assert np.degrees(split_yzx(euler(single)[:3, :3])) == pytest.approx(single, abs=1e-9)


@pytest.mark.parametrize("name", ENDS)
def test_the_upper_arms_limit_stops_a_hanging_arm_and_not_a_reach_across(name):
    bones = rigged(name).bones
    limits = joint_limits(bones)
    bone = "leftUpperArm"
    frame = rest_matrix(next(b for b in bones if b["name"] == bone))[:3, :3]
    inward = limits[bone][2][0]

    # Hanging, past the body: held at the body.
    assert held(bone, (0.0, 0.0, inward - 15), limits[bone]) == pytest.approx((0.0, 0.0, inward), abs=1e-6)

    def reach(degrees):
        """The arm level, pointing this far across the midline from straight ahead."""
        angle = np.radians(degrees)
        return frame.T @ np.array([-np.sin(angle), -np.cos(angle), 0.0])

    def angles_for(direction):
        # Rx Rz takes the bone's Y onto (−sin z, cos z cos x, cos z sin x).
        z = np.arcsin(-direction[0])
        x = np.arctan2(direction[2], direction[1])
        matrix = euler((np.degrees(x), 0, 0))[:3, :3] @ euler((0, 0, np.degrees(z)))[:3, :3]
        return tuple(
            np.degrees(
                [np.arctan2(matrix[2, 1], matrix[2, 2]), np.arcsin(-matrix[2, 0]), np.arctan2(matrix[1, 0], matrix[0, 0])]
            )
        )

    # A hand to the far shoulder is a level reach some 40 past the midline, and passes.
    across = angles_for(reach(40))
    assert held(bone, across, limits[bone]) == across
    # Well past what the geometry allows, 45 at level, it does not.
    further = angles_for(reach(70))
    assert held(bone, further, limits[bone]) != further


def test_clipping_a_mirrored_arm_mirrors_the_left():
    limits = joint_limits(rigged(ENDS[0]).bones)
    pose = {"upperArm": (90.0, 60.0, 0.0)}
    both = clipped(pose, limits)
    left, right = np.array(both["leftUpperArm"]), np.array(both["rightUpperArm"])
    assert right == pytest.approx(left * [1, -1, -1], abs=1e-6)


@pytest.mark.blender
@pytest.mark.parametrize("name", ENDS)
def test_a_keyed_clip_plays_in_the_file_as_it_does_at_runtime(name, tmp_path):
    """Reopened with Auto Run off, limits off as saved, and on as sotaigen.rig.limits.clipped says."""
    rig, _, fps, _, poses = fitted(name, "walk")
    poses = poses[:60]
    blend = blend_body(name, tmp_path / "body.blend")
    sample = list(range(0, len(poses), SAMPLE_EVERY))
    info, data = key_and_read(blend, rig, fps, poses, sample, tmp_path)
    assert info["limitsAsSaved"] is False
    assert info["autoexecFail"] is False
    assert info["texts"] == 0
    assert info["frames"] == [1, len(poses)]
    limits = joint_limits(rig.bones)
    correctives = Correctives.production(rig.base, rig.verts, rig.bones, rig.weights, field=surface(name)[0])
    for frame in sample:
        for state, pose in (("off", poses[frame]), ("on", clipped(poses[frame], limits))):
            wanted = pose_verts(rig.bones, rig.weights, correctives.bind(rig.verts, pose), pose)
            error = np.linalg.norm(data[f"{state}:{frame}:skin"] - wanted, axis=1).max()
            assert error <= KEYED_SKIN_MM, (state, frame, error)
