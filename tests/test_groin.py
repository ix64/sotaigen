"""Production pelvic shapes preserve the approved sculpture and its fixed rig."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sotaigen.corrective import groin
from sotaigen.form import groin as groin_rest
from sotaigen.params.rules import RANGES
from sotaigen.pipeline import BUILD, rigged
from sotaigen.rig.skinning import MIDLINE_BONES, pose_verts

# The approved sculpture on the front of two bodies' pelvises: what the accepted review prototype gave,
# on inputs kept with it, so neither a new body nor a new template changes what this holds. A deliberate,
# approved change to the sculpture rewrites the file from groin.sculpture on those same inputs.
FROZEN = Path(__file__).with_name("data") / "pelvis-sculpture.npz"
ANGLES = (0, 40, 100)


def frozen_body(data, key):
    faces = np.split(data[f"{key}.faces"], np.cumsum(data[f"{key}.faceSizes"])[:-1])
    base = {"faces": [face.tolist() for face in faces], "segments": data[f"{key}.segments"]}
    bones = [
        {k: np.asarray(v) if k in ("head", "tail", "align") else v for k, v in bone.items()}
        for bone in json.loads(data[f"{key}.bones"].item())
    ]
    return base, bones, json.loads(data[f"{key}.L"].item())


@pytest.mark.parametrize("key", ("short", "balanced"))
@pytest.mark.parametrize("mode", ("single", "double"))
def test_production_reproduces_the_approved_sculpture(key, mode):
    with np.load(FROZEN) as data:
        base, bones, L = frozen_body(data, key)
        for angle in ANGLES:
            actual = groin.sculpture(
                base,
                data[f"{key}.reference"],
                data[f"{key}.rest"],
                bones,
                data[f"{key}.weights"],
                L,
                angle,
                angle if mode == "double" else 0,
            )
            np.testing.assert_allclose(actual, data[f"{key}.{mode}{angle}"], atol=1e-8, rtol=0)


@pytest.mark.parametrize("name", ("female-chibi", "female-slim", "female-balanced"))
def test_the_rest_shape_inverts_to_the_wrapped_skin(name):
    rig = rigged(name)
    reference = np.load(BUILD / "bodies" / name / "wrapped.npy")
    np.testing.assert_allclose(
        groin_rest.reference_verts(rig.base, rig.verts, rig.L), reference, atol=1e-8, rtol=0
    )


def test_the_sculpture_leaves_the_back_and_other_segments_on_the_plain_pose():
    rig = rigged("female-long")
    reference = np.load(BUILD / "bodies" / "female-long" / "wrapped.npy")
    segments = np.asarray(rig.base["segments"])
    fixed = ~np.isin(segments, ("trunk", "upperLeg")) | (reference[:, 1] >= 0)
    actual = groin.sculpture(rig.base, reference, rig.verts, rig.bones, rig.weights, rig.L, 100, 100)
    plain = pose_verts(rig.bones, rig.weights, rig.verts, groin.sample_pose(100, 100))
    np.testing.assert_array_equal(actual[fixed], plain[fixed])


def test_the_groin_amounts_scale_the_static_shape_and_stop_at_the_accepted_one():
    rig = rigged("female-balanced")
    reference = np.load(BUILD / "bodies" / "female-balanced" / "wrapped.npy")
    L = rig.L
    assert RANGES["female"]["groinDepth"][1] == groin_rest.DEPTH
    assert RANGES["female"]["groinRelief"][1] == groin_rest.RELIEF
    accepted = L | {"groinDepth": groin_rest.DEPTH, "groinRelief": groin_rest.RELIEF}
    np.testing.assert_array_equal(groin_rest.rest_verts(rig.base, reference, accepted), rig.verts)
    none = L | {"groinDepth": 0.0, "groinRelief": 0.0}
    np.testing.assert_array_equal(groin_rest.rest_verts(rig.base, reference, none), reference)
    half = L | {"groinDepth": groin_rest.DEPTH / 2, "groinRelief": groin_rest.RELIEF / 2}
    np.testing.assert_allclose(
        groin_rest.rest_verts(rig.base, reference, half) - reference,
        (rig.verts - reference) / 2,
        atol=1e-9,
        rtol=0,
    )
    # The millimetres say what the skin does: the valley alone, its peak on this body.
    valley_only = L | {"groinRelief": 0.0}
    moved = groin_rest.rest_verts(rig.base, reference, valley_only) - reference
    assert moved[:, 1].max() == pytest.approx(groin_rest.DEPTH * L["T"], rel=0.01)


def test_a_preset_asking_for_a_lighter_groin_gets_it_through_the_build():
    # The wide-hip test body asks for a lighter groin, groinRelief 2.0 mm, as no typical body does.
    rig = rigged("female-widehip")
    reference = np.load(BUILD / "bodies" / "female-widehip" / "wrapped.npy")
    kept = groin.landmarks(rig.bones)
    assert kept["groinRelief"] * kept["T"] == pytest.approx(2.0)
    assert "groinDepth" not in kept
    # The torso read back from the bones differs from the landmark's by rounding only.
    np.testing.assert_allclose(
        groin_rest.rest_verts(rig.base, reference, kept), rig.verts, atol=1e-9, rtol=0
    )
    accepted = groin_rest.rest_verts(rig.base, reference, kept | {"groinRelief": groin_rest.RELIEF})
    assert (accepted - rig.verts)[:, 1].min() < -3.0


def test_pose_targets_follow_the_amounts_kept_with_the_rig():
    rig = rigged("female-long")
    reference = np.load(BUILD / "bodies" / "female-long" / "wrapped.npy")
    L = rig.L | {"groinDepth": 0.008, "groinRelief": 0.003}
    bones = rig.bones.copy()
    groin.record_amounts(bones, L)
    assert {k: groin.landmarks(bones)[k] for k in groin.AMOUNTS} == {
        "groinDepth": 0.008,
        "groinRelief": 0.003,
    }
    rest = groin_rest.rest_verts(rig.base, reference, L)
    np.testing.assert_allclose(
        groin_rest.reference_verts(rig.base, rest, groin.landmarks(bones)), reference, atol=1e-8, rtol=0
    )
    deltas = groin.sculpted_targets(rig.base, rest, bones, rig.weights)
    pose = groin.sample_pose(40, 0)
    actual = pose_verts(bones, rig.weights, groin.bind_verts(rest, deltas, pose), pose)
    expected = groin.sculpture(rig.base, reference, rest, bones, rig.weights, L, 40, 0)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)


@pytest.fixture(scope="module")
def body_targets():
    rig = rigged("female-long")
    return rig, groin.sculpted_targets(rig.base, rig.verts, rig.bones, rig.weights)


def test_inverse_skinned_targets_hit_the_sculpture_at_asymmetric_keys(body_targets):
    rig, targets = body_targets
    reference = np.load(BUILD / "bodies" / "female-long/wrapped.npy")
    for a, b in ((0, 0), (40, 0), (0, 100), (20, 80), (100, 100)):
        pose = groin.sample_pose(a, b)
        bound = groin.bind_verts(rig.verts, targets, pose)
        actual = pose_verts(rig.bones, rig.weights, bound, pose)
        expected = groin.sculpture(
            rig.base, reference, rig.verts, rig.bones, rig.weights, rig.L, a, b
        )
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
    assert groin.bind_verts(rig.verts, targets, {}) is rig.verts
    assert not rig.verts.flags.writeable


@pytest.mark.node
def test_javascript_and_python_pelvis_drivers_agree():
    poses = [
        {},
        {"upperLeg": [70, 0, 0]},
        {"upperLeg": [85, 15, 10], "leftUpperLeg": [31, -20, -15]},
        {"leftUpperLeg": [100, 0, 0], "upperLeg": [40, 0, 0]},
        {"rightUpperLeg": [110, -5, 15], "leftUpperLeg": [-10, 0, 0]},
        {"upperLeg": [50, 60, 0]},
    ]
    code = "import {correctiveWeights} from './web/runtime/driver.mjs'; const input=JSON.parse(process.argv[1]); console.log(JSON.stringify(input.poses.map(p=>correctiveWeights(p,input.table,input.central))));"
    result = subprocess.run(
        [
            "node",
            "--input-type=module",
            "-e",
            code,
            json.dumps({"poses": poses, "table": groin.groups(), "central": list(MIDLINE_BONES)}),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for pose, actual in zip(poses, json.loads(result.stdout), strict=True):
        expected = groin.pose_weights(pose)
        assert actual.keys() == expected.keys()
        np.testing.assert_allclose(
            list(actual.values()), list(expected.values()), atol=1e-12, rtol=0
        )
    assert all(v == 0 for v in groin.pose_weights({}).values())
    assert groin.pose_weights({"upperLeg": [40, 0, 0]})["pelvisL40R40"] == 1
