"""Shoulder surface, pose-grid interpolation and exported geometric normals."""

import json
import subprocess
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from sotaigen.corrective import axilla
from sotaigen.form.bodies import load_bodies
from sotaigen.metrics import RestMesh
from sotaigen.pipeline import rigged
from sotaigen.rig.skinning import MIDLINE_BONES, pose_skins, pose_verts
from sotaigen.template import before_loops


def test_pose_grid_keys_mirroring_and_overrides():
    # The table's points are twist-free swings (axilla.sample_pose), read back by swing and twist.
    for index, (forward, lateral) in enumerate(axilla.SAMPLES):
        key = axilla.sample_pose("left", forward, 0, lateral)["leftUpperArm"]
        both = axilla.pose_weights({"upperArm": key})
        assert {k for k, v in both.items() if v == pytest.approx(1, abs=1e-12)} == {
            axilla.NAMES[side][index] for side in ("left", "right")
        }
        assert sum(both.values()) == pytest.approx(2, abs=1e-12)
        explicit = axilla.pose_weights({
            "leftUpperArm": key,
            "rightUpperArm": axilla.sample_pose("right", forward, 0, lateral)["rightUpperArm"],
        })
        assert explicit == pytest.approx(both, abs=1e-12)
    overridden = {"rightUpperArm": (95, 0, -72), "upperArm": (20, 0, 0)}
    assert axilla.pose_weights(overridden) == axilla.pose_weights({"upperArm": (20, 0, 0)})
    weights = axilla.pose_weights(axilla.sample_pose("left", 67.5, 0, 53.5))
    assert sorted(v for v in weights.values() if v > 1e-12) == pytest.approx([0.25] * 4)
    # Twisted to the fade's limit, the same swing drives nothing.
    twisted = axilla.sample_pose("left", 95, axilla.TWIST_LIMIT, 72)["leftUpperArm"]
    assert max(axilla.pose_weights({"upperArm": twisted}).values()) < 1e-12
    for pose in ({}, {"upperArm": (-15, 0, 0)}, {"head": (20, 0, 0)}):
        assert not any(axilla.pose_weights(pose).values())
        verts = np.zeros((1, 3))
        assert axilla.bind_verts(verts, {}, pose) is verts


@pytest.mark.node
def test_browser_and_python_drive_the_same_grid():
    rng = np.random.default_rng(935)
    poses = [
        {"upperArm": (x, 0, z)} for x in axilla.KEYS for z in axilla.SPREAD_KEYS
    ]
    poses += [{"rightUpperArm": (67.5, -22.5, -53.5)},
              {"upperArm": (35, 10, 72), "rightUpperArm": (60, 0, 24)},
              {"rightUpperArm": (95, 0, -110), "upperArm": (0, 0, 0)}]
    poses += [{"upperArm": rng.uniform((-20, -100, -35), (120, 100, 130)).tolist()} for _ in range(80)]
    path = Path(__file__).resolve().parents[1] / "web/runtime/driver.mjs"
    code = f'''import {{ correctiveWeights }} from {json.dumps(path.as_uri())};
const payload = JSON.parse(await new Promise(resolve => {{
let s = ''; process.stdin.on('data', x => s += x); process.stdin.on('end', () => resolve(s));
}}));
console.log(JSON.stringify(payload.poses.map(p => correctiveWeights(p, payload.table, payload.central))));'''
    done = subprocess.run(["node", "--input-type=module", "-e", code], input=json.dumps({
        "poses": poses, "table": axilla.groups(), "central": list(MIDLINE_BONES),
    }), capture_output=True, text=True, check=True)
    for pose, actual in zip(poses, json.loads(done.stdout), strict=True):
        assert actual == pytest.approx(axilla.pose_weights(pose), abs=1e-12)


@cache
def correction(name):
    rig = rigged(name)
    return axilla.targets(rig.base, rig.verts, rig.bones, rig.weights)


@pytest.mark.slow
@pytest.mark.parametrize("name", load_bodies())
def test_axillary_floor_does_not_unfold_into_a_long_web(name):
    rig = rigged(name)
    raw = np.asarray(rig.base["verts"])
    floor = rig.base["landmarks"]["vertices"]["axillaFloor"]
    anchor = raw[floor]
    q = (raw - anchor) / anchor[0]
    patch = (raw[:, 0] > 0) & (((q / [1.5, 2.15, 2.55]) ** 2).sum(1) < 1)
    # Measured on the quads as extract.py cut them. The axilla loops quarter the quads in the
    # front crease, and one quarter of a pinched quad drops below a limit that the whole quad
    # clears, so a per-face limit on the cut faces would tighten each time a loop is added.
    faces = [f for f in before_loops(rig.base)["faces"] if patch[f].all()]
    rest = RestMesh(rig.verts, faces)
    arm = next(i for i, b in enumerate(rig.bones) if b["name"] == "leftUpperArm")
    deltas = correction(name)
    protected = np.isin(rig.base["segments"], ("head", "neck"))
    assert all(not delta[protected].any() for delta in deltas.values())
    for angles in ((0, 0, 0), (0, 0, 72), (0, 0, 110), (75, 0, 0), (95, 0, 0), (75, 20, 35)):
        pose = {"upperArm": angles}
        posed = pose_verts(rig.bones, rig.weights, axilla.bind_verts(rig.verts, deltas, pose), pose)
        area = rest.area_ratio(posed)
        joint = (pose_skins(rig.bones, pose)[arm] @ np.r_[rig.bones[arm]["head"], 1])[:3]
        # Engineering guards against the former 100x area expansion and low hanging floor.
        # These are not physiological tissue-strain limits or a substitute for visual review.
        assert np.linalg.norm(posed[floor] - joint) < 0.20 * rig.L["T"], (name, angles)
        assert area.max() < 8, (name, angles, area.max())
        assert area.min() >= 0.15, (name, angles, area.min())


def test_corrections_are_local_symmetric_and_continuous():
    rig = rigged("female-long")
    raw = np.asarray(rig.base["verts"])
    targets = correction("female-long")
    head = np.asarray(rig.base["segments"]) == "head"
    for side, sign in (("left", 1), ("right", -1)):
        for name in axilla.NAMES[side]:
            assert np.isfinite(targets[name]).all()
            assert not targets[name][raw[:, 0] * sign <= 0].any()
            assert not targets[name][head].any()
    for angle in axilla.KEYS:
        for spread in axilla.SPREAD_KEYS:
            for axis in (0, 2):
                nearby = []
                for offset in (-0.01, 0.01):
                    angles = np.array([angle, 0, spread], float)
                    angles[axis] += offset
                    pose = {"upperArm": angles}
                    nearby.append(pose_verts(rig.bones, rig.weights,
                        axilla.bind_verts(rig.verts, targets, pose), pose))
                assert np.linalg.norm(nearby[0] - nearby[1], axis=1).max() < 0.5


def test_reach_shading_matches_export_triangles_and_keeps_authored_head_normals():
    from sotaigen.mesh import triangulate, vertex_normals

    rig = rigged("female-long")
    faces = triangulate(rig.base["faces"], np.asarray(rig.base["verts"]))
    positions = correction("female-long")
    normals = vertex_normals(rig.verts, faces).astype(float)
    head = np.asarray(rig.base["segments"]) == "head"
    normals[head] = (0, -1, 0)
    deltas = axilla.normal_targets(
        rig.base, rig.verts, rig.bones, rig.weights, positions, normals, triangles=faces
    )
    assert list(deltas) == list(positions)
    for delta in deltas.values():
        assert np.isfinite(delta).all()
        assert not delta[head].any()
    # At the keys: the last one is a swing about both axes, as the table samples it.
    for pose in ({"upperArm": (75, 0, 0)}, {"rightUpperArm": (95, 0, 0)},
                 {"leftUpperArm": (0, 0, 72)}, axilla.sample_pose("right", 60, 0, 35)):
        bound = axilla.bind_verts(rig.verts, positions, pose)
        reference = vertex_normals(pose_verts(rig.bones, rig.weights, bound, pose), faces)
        blend = np.einsum("vb,bij->vij", rig.weights, pose_skins(rig.bones, pose)[:, :3, :3])
        shading = axilla.bind_verts(normals, deltas, pose)
        posed = (blend @ shading[:, :, None])[:, :, 0]
        posed /= np.linalg.norm(posed, axis=1, keepdims=True)
        moved = np.linalg.norm(bound - rig.verts, axis=1) > 5.0
        assert moved.any()
        np.testing.assert_allclose(posed[moved], reference[moved], atol=1e-6)
