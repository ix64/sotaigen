"""Contact and skin deformation on both hands of the wrapped bodies."""

import numpy as np
import pytest

from sotaigen.corrective.crossings import delivered_skin, edges_through_triangles
from sotaigen.form.bodies import load_bodies
from sotaigen.pipeline import rigged
from sotaigen.rig.poses import POSES
from sotaigen.rig.skinning import pose_skins, pose_verts, rest_matrix
from sotaigen_dev.motion.hand_contacts import (
    FINGERS,
    HANDLE_RADIUS,
    THIN_HANDLE_RADIUS,
    handle_contact,
    pad_faces,
    pinch_contact,
    sphere_clearances,
)
from sotaigen_dev.motion.hand_motion import cylinder_pickup, greeting, handle_pose


@pytest.fixture(params=load_bodies())
def hands(request):
    rig = rigged(request.param)
    base, verts, bones, weights = rig.base, rig.verts, rig.bones, rig.weights
    segments = np.array(base["segments"])
    rows = []
    for side, sign in (("left", 1), ("right", -1)):
        ids = np.flatnonzero(
            (np.array(base["regions"]) == "hand") & (np.array(base["verts"])[:, 0] * sign > 0)
        )
        selected = np.zeros(len(verts), bool)
        selected[ids] = True
        faces = [np.searchsorted(ids, f).tolist() for f in base["faces"] if selected[f].all()]
        i = next(i for i, b in enumerate(bones) if b["name"] == side + "Hand")
        skin = delivered_skin(faces, np.array(base["verts"])[ids])
        rows.append((side, sign, i, faces, segments[ids], weights[ids], verts[ids], skin))
    return bones, rows


def test_articulated_palm_and_contact_shapes(hands):
    bones, rows = hands
    by_name = {b["name"]: b for b in bones}
    for side, sign, i, faces, segments, weights, verts, (triangles, edges) in rows:
        wrist = rest_matrix(bones[i])
        length = np.linalg.norm(bones[i]["tail"] - bones[i]["head"])
        for name in (
            "handRelaxed",
            "handPoint",
            "handLightTouch",
            "handWave",
            "sphereGrip",
            "thinCylinderGrip",
            "littleOpposition",
        ):
            points = pose_verts(bones, weights, verts, POSES[name])
            assert not len(edges_through_triangles(points, edges, triangles)), (name, side)
            local = (points - wrist[:3, 3]) @ wrist[:3, :3] * [sign, 1, 1] / length
            if name in ("sphereGrip", "thinCylinderGrip"):
                gap = (
                    sphere_clearances(local, triangles)
                    if name == "sphereGrip"
                    else (
                        np.linalg.norm(handle_contact(local, triangles), axis=1)
                        - THIN_HANDLE_RADIUS
                    )
                )
                assert gap.min() >= 0, (name, side)
                for finger in FINGERS:
                    assert gap[(segments[triangles] == finger + "Distal").all(1)].min() < 0.02, (
                        name,
                        side,
                        finger,
                    )
            elif name == "littleOpposition":
                patches = [
                    pad_faces(f, sign, faces, segments, verts, by_name[side + f.title() + "Distal"])
                    for f in ("thumb", "little")
                ]
                gap, alignment, offset = pinch_contact(points, patches, length)
                assert gap < 0.025 and alignment < -0.94 and offset > 0, (
                    side,
                    gap,
                    alignment,
                    offset,
                )


@pytest.mark.slow
def test_pickup_path_and_wave_on_skinned_hands(hands):
    bones, rows = hands
    # The return half retraces the pickup. Check each 30 Hz key and the midpoint between keys.
    for side, sign, i, faces, _, weights, verts, (triangles, edges) in rows:
        wrist = rest_matrix(bones[i])
        length = np.linalg.norm(bones[i]["tail"] - bones[i]["head"])
        held = None
        for t in np.linspace(0, 0.5, 91):
            prop = pose_skins(bones, handle_pose(t))[i] @ wrist
            points = pose_verts(bones, weights, verts, cylinder_pickup(t))
            local = (points - prop[:3, 3]) @ prop[:3, :3] * [sign, 1, 1] / length
            gap = np.linalg.norm(handle_contact(local, triangles), axis=1) - HANDLE_RADIUS
            assert gap.min() >= 0, (side, t, gap.min())
            assert not len(edges_through_triangles(points, edges, triangles)), (side, t)
            if t >= 0.38:
                if held is None:
                    held = local
                np.testing.assert_allclose(local, held, atol=1e-4)
        for t in np.linspace(0, 1, 31):
            points = pose_verts(bones, weights, verts, greeting(t))
            assert not len(edges_through_triangles(points, edges, triangles)), (side, t)


def test_pickup_glb_moves_the_visible_handle_only_while_held(tmp_path):
    from gltf_support import accessor, trs

    from sotaigen.output.glb import TO_GLTF, read_glb, to_gltf
    from sotaigen_dev.motion.clips import cmd_glb

    target = tmp_path / "pickup.glb"
    cmd_glb(["female-long", "--clip", "cylinderPickup", "-o", str(target)])
    doc, data = read_glb(target.read_bytes())
    node = next(i for i, n in enumerate(doc["nodes"]) if n["name"] == "referenceHandle")
    assert node in doc["scenes"][doc["scene"]]["nodes"]
    assert "mesh" in doc["nodes"][node] and "skin" not in doc["nodes"][node]
    (animation,) = doc["animations"]
    tracks = {}
    for channel in animation["channels"]:
        if channel["target"]["node"] == node:
            sampler = animation["samplers"][channel["sampler"]]
            tracks[channel["target"]["path"]] = accessor(doc, data, sampler["output"])
            times = accessor(doc, data, sampler["input"]).ravel()
            assert times[0] == 0 and times[-1] == 3
    bones = rigged("female-long").bones
    i = next(i for i, b in enumerate(bones) if b["name"] == "leftHand")
    wrist = rest_matrix(bones[i])
    for frame in (0, 23, 35, 45, 67, 90):
        expected = pose_skins(bones, handle_pose(frame / 90))[i] @ wrist
        actual = trs({k: v[frame] for k, v in tracks.items()})
        np.testing.assert_allclose(actual[:3, 3], to_gltf(expected[:3, 3]), atol=1e-6)
        np.testing.assert_allclose(actual[:3, :3], TO_GLTF @ expected[:3, :3], atol=1e-6)
    assert tracks["translation"][45, 1] - tracks["translation"][0, 1] > 0.04


def test_thumb_pad_faces_the_grasped_surface(hands):
    from sotaigen.mesh import unit_face_normals
    from sotaigen_dev.motion.hand_contacts import (
        BALL_CENTER,
        BALL_RADIUS,
        HANDLE_AXIS,
        HANDLE_CENTER,
    )

    bones, rows = hands
    for side, sign, i, faces, segments, weights, verts, _ in rows:
        wrist = rest_matrix(bones[i])
        length = np.linalg.norm(bones[i]["tail"] - bones[i]["head"])
        bone = next(b for b in bones if b["name"] == side + "ThumbDistal")
        patches = pad_faces("thumb", sign, faces, segments, verts, bone)
        for name, radius in (
            ("cylinderGrip", HANDLE_RADIUS),
            ("thinCylinderGrip", THIN_HANDLE_RADIUS),
            ("sphereGrip", BALL_RADIUS),
        ):
            points = pose_verts(bones, weights, verts, POSES[name])
            centers = np.array([points[f].mean(0) for f in patches])
            centers = (centers - wrist[:3, 3]) @ wrist[:3, :3] * [sign, 1, 1] / length
            radial = centers - (BALL_CENTER if name == "sphereGrip" else HANDLE_CENTER)
            if name != "sphereGrip":
                radial -= np.outer(radial @ HANDLE_AXIS, HANDLE_AXIS)
            distances = np.linalg.norm(radial, axis=1)
            near = distances.argmin()
            normals = unit_face_normals(points, patches) @ wrist[:3, :3] * [sign, 1, 1]
            # The closest sample on the soft pulp must face into the object, not graze it sideways.
            assert 0 <= distances[near] - radius < 0.025, (name, side)
            assert normals[near] @ radial[near] / distances[near] < -0.94, (name, side)


def test_thumb_opens_the_web_before_curling_around_the_handle():
    ready = cylinder_pickup(0.16)
    closed = cylinder_pickup(0.38)
    # Leave most of the IP joint's bend for acquisition and unwind it before retreating.
    assert np.linalg.norm(ready["thumbDistal"]) < 0.4 * np.linalg.norm(closed["thumbDistal"])
    assert np.linalg.norm(ready["thumbMetacarpal"]) > np.linalg.norm(ready["thumbDistal"])
    np.testing.assert_allclose(cylinder_pickup(0.84)["thumbDistal"], ready["thumbDistal"])
