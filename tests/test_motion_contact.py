"""Travelling motion, skin contact and a standard glTF player's take on every baked transform."""

import numpy as np
import pytest
from gltf_support import accessor, run_driver, skin_stored

from sotaigen.output.glb import glb_bytes, read_glb, to_gltf
from sotaigen.pipeline import rigged
from sotaigen.rig.skinning import euler, pose_skins, pose_verts
from sotaigen_dev.motion import clips as motions
from sotaigen_dev.motion.foot_contact import (
    SIDES,
    contact_tracks,
    filter_foot_samples,
    sole_samples,
)


def test_only_a_low_stationary_marker_is_a_plant():
    # The ankle stays put at either end, swinging midway. The ball travels throughout.
    xyz = np.zeros((7, 2, 3))
    xyz[:, 1, 1] = np.arange(7)
    xyz[2:5, 0, 2] = 0.2
    mask = np.array(contact_tracks(xyz, 10), bool)
    assert mask[:, 0].tolist() == [True, True, False, False, False, True, True]
    assert not mask[:, 1].any()


def test_walking_feet_rest_on_the_full_sole_and_still_roll_off():
    rig = rigged("female-long")
    clip = motions.load_clip("walk")["capture"]
    _, poses = motions.playback_samples(clip, rig)
    skins = np.array([pose_skins(rig.bones, pose) for pose in poses])
    names = [b["name"] for b in rig.bones]
    for side in SIDES:
        heel, ball = np.asarray(clip["footContacts"][side], bool).T
        markers, _ = sole_samples(rig, skins, side)
        flat = heel & ball
        assert flat.any()
        # A level foot can still point in the wrong direction. Compare horizontal
        # heel-to-ball travel axes against the original capture through each plant.
        source = side[0] + "foot"
        heading = motions.heading_frame(clip, 0)
        source_axes = np.array([
            heading @ motions.matrix_from(q) @ clip["rest"][source]
            for q in clip["rotations"][source]
        ])
        edges = np.diff(np.r_[False, flat, False].astype(int))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
            wanted = source_axes[start:end, :2].mean(axis=0)
            actual = (markers[start:end, 1, :2] - markers[start:end, 0, :2]).mean(axis=0)
            cosine = (wanted @ actual) / (np.linalg.norm(wanted) * np.linalg.norm(actual))
            assert cosine > np.cos(np.radians(3))
        # Both plantar markers, not just the lowest vertex. Each rests a share of the foot above the
        # floor (the ball 0.021 of it), so a plant is held to where the marker rests: the old flat 4 mm
        # was about that plus 0.9 on the old 148 mm foot.
        resting, _ = sole_samples(rig, pose_skins(rig.bones, {})[None], side)
        assert (markers[flat, :, 2] - resting[0, :, 2]).max() < 1
        for part in ("Foot", "Toes"):
            rotation = skins[:, names.index(side + part), :3, :3]
            assert rotation[flat, 2, 2].min() > np.cos(np.radians(3))
        assert np.max((markers[:, 1, 2] - markers[:, 0, 2])[heel & ~ball]) > 5
        assert np.max((markers[:, 0, 2] - markers[:, 1, 2])[~heel & ball]) > 5


def test_foot_capture_filter_rejects_a_short_reversal_without_delaying_a_swing():
    clean = np.array([euler((angle, 0, 0))[:3, :3] for angle in range(0, 42, 2)])
    noisy = clean.copy()
    noisy[9:11] = clean[1]
    filtered = filter_foot_samples(noisy, 60)
    error = np.linalg.norm(filtered - clean, axis=(1, 2))
    assert error.max() < 0.18  # Less than an eight-degree phase error at the dropout.
    np.testing.assert_allclose(filtered[3:6], clean[3:6], atol=1e-8)
    np.testing.assert_allclose(filtered[-6:-3], clean[-6:-3], atol=1e-8)


@pytest.mark.node
def test_root_translation_moves_all_vertices_in_project_coordinates():
    rig = rigged("female-long")
    offset = [43, -118, 26]
    pose = {"lowerLeg": (-30, 0, 0)}

    def skin(p):
        return pose_verts(rig.bones, rig.weights, rig.verts, p)

    assert skin({**pose, "rootTranslation": offset}) == pytest.approx(skin(pose) + offset)
    assert run_driver(
        "console.log(JSON.stringify(driver.rootTranslation(input)));", {"rootTranslation": offset}
    ) == pytest.approx(to_gltf(offset))
    for value in ([0, 1], [0, 0, np.inf]):
        with pytest.raises(ValueError, match="rootTranslation"):
            skin({"rootTranslation": value})


@pytest.mark.parametrize("preset", ["female-chibi", "female-balanced"])
@pytest.mark.parametrize("name", ["walk", "run", "sitLow"])
def test_contacts_follow_the_floor_at_both_leg_lengths(preset, name):
    rig = rigged(preset)
    clip = motions.load_clip(name)["capture"]
    _, poses = motions.playback_samples(clip, rig)
    skins = np.array([pose_skins(rig.bones, pose) for pose in poses])
    down, clearance = [], []
    for side in SIDES:
        active = np.asarray(clip["footContacts"][side], bool)
        markers, floor = sole_samples(rig, skins, side)
        planted = active.any(axis=1)
        assert np.max(abs(floor[planted])) < 0.1
        assert floor.min() > -0.1
        step = np.linalg.norm(np.diff(markers[:, :, :2], axis=0), axis=2)
        assert step[active[:-1] & active[1:]].mean() < 0.5
        down.append(planted)
        clearance.append(floor)
    root = np.array([p["rootTranslation"] for p in poses])
    if name == "sitLow":
        assert root[-1, 2] < root[0, 2] - 120
    else:
        assert np.linalg.norm(root[-1, :2] - root[0, :2]) > 800
    if name == "run":
        flying = ~np.logical_or(*down)
        assert np.min(clearance, axis=0)[flying].max() > 15


def test_animation_keys_drive_the_stored_skin_including_root_and_helpers():
    rig = rigged("female-long")
    clip = motions.load_clip("run")
    fps, poses = motions.playback_samples(clip, rig)
    document, data = read_glb(
        glb_bytes(rig.base, rig.verts, rig.bones, rig.weights, motions=[("run", fps, poses)])
    )
    (animation,) = document["animations"]
    for frame in (0, len(poses) // 2, len(poses) - 1):
        for channel in animation["channels"]:
            sampler = animation["samplers"][channel["sampler"]]
            assert accessor(document, data, sampler["input"])[frame, 0] == pytest.approx(
                frame / fps
            )
            value = accessor(document, data, sampler["output"])[frame].astype(float)
            target = channel["target"]
            document["nodes"][target["node"]][target["path"]] = value
        skin = skin_stored(document, data, {})
        expected = pose_verts(rig.bones, rig.weights, rig.verts, poses[frame])
        assert np.max(abs(skin - to_gltf(expected) * 1000)) < 0.1


def test_rounded_capture_quaternion_preserves_a_rigid_frame():
    from sotaigen_dev.motion.clips import matrix_from

    q = np.round([0.5**0.5, 0, 0.5**0.5, 0], 4)
    frame = matrix_from(q)
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
