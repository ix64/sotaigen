"""Observation math must not confuse rigid movement or ownership with tissue motion."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from sotaigen_dev.motion.groin_sheet import (
    check_comparison,
    frame_of,
    reference_columns,
    relative_displacement,
)


def test_reference_ownership_is_segment_and_side_not_height():
    bones = [{"name": n} for n in ("hips", "leftUpperLeg", "rightUpperLeg")]
    base = {
        "segments": ["trunk", "upperLeg", "upperLeg", "lowerLeg"],
        "verts": [[1, 0, -10], [1, 0, 10], [-1, 0, 10], [1, 0, -20]],
    }
    # Trunk below the joint remains pelvic; a high thigh point remains femoral.
    assert reference_columns(base, bones).tolist() == [0, 1, 2, -1]


def rigid():
    angle = np.radians(70)
    c, s = np.cos(angle), np.sin(angle)
    m = np.eye(4)
    m[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    m[:3, 3] = [20, -7, 30]
    return m


def apply(points, matrix):
    return (np.column_stack((points, np.ones(len(points)))) @ matrix.T)[:, :3]


def test_rigid_reference_motion_is_zero_even_after_rotation_and_translation():
    rest = np.array([[1.0, 2.0, 3.0], [3.0, 5.0, 7.0], [-1.0, 1.0, 9.0]])
    skins = np.array([np.eye(4), rigid()])
    columns = np.array([0, 1, -1])
    posed = rest.copy()
    posed[1] = apply(rest[1:2], skins[1])[0]
    lift, tangent = relative_displacement(
        rest, posed, skins, columns, np.tile([0.0, 0.0, 1.0], (3, 1))
    )
    np.testing.assert_allclose(lift[:2], 0, atol=1e-12)
    np.testing.assert_allclose(tangent[:2], 0, atol=1e-12)
    assert np.isnan(lift[2]) and np.isnan(tangent[2])


def test_normal_and_tangent_are_measured_in_the_rest_reference():
    rest = np.array([[1.0, 2.0, 3.0]])
    skins = np.array([rigid()])
    posed = apply(rest + [4.0, 0.0, 3.0], skins[0])
    lift, tangent = relative_displacement(
        rest, posed, skins, np.array([0]), np.array([[0.0, 0.0, 1.0]])
    )
    np.testing.assert_allclose(lift, 3.0, atol=1e-12)
    np.testing.assert_allclose(tangent, 4.0, atol=1e-12)


def test_hip_frame_uses_pelvic_up_instead_of_lateral_parent_connection():
    rig = SimpleNamespace(
        bones=[
            {"name": "hips", "head": np.array([0.0, 0.0, 1.0]), "tail": np.array([0.0, 0.0, 2.0])},
            {
                "name": "leftUpperLeg",
                "head": np.array([1.0, 0.0, 1.0]),
                "tail": np.array([1.0, 0.0, -1.0]),
                "align": np.array([0.0, -1.0, 0.0]),
            },
        ]
    )
    frame = frame_of(rig)
    np.testing.assert_allclose(frame.rotation @ frame.rotation.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(frame.upper, [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(frame.canonical(frame.centre), [0, 0, 1000], atol=1e-12)
    assert np.linalg.det(frame.rotation) > 0.999999

    # Canonical front remains anterior, and left stays on the image right.
    assert frame.axis[0] > 0.99
    anterior = frame.centre + np.array([0.0, -1.0, 0.0])
    assert frame.canonical(anterior)[1] < 0


def test_comparison_allows_new_files_but_rejects_a_changed_camera_scale():
    before = {
        "meshes": [
            {
                "source": "/before/single040.obj",
                "shots": [{"out": "/before/shot.png", "mm_per_px": 1.0, "look": "clay"}],
            }
        ]
    }
    after = copy.deepcopy(before)
    after["meshes"][0]["source"] = "/after/single040.obj"
    after["meshes"][0]["shots"][0]["out"] = "/after/shot.png"
    check_comparison(after, before)
    after["meshes"][0]["shots"][0]["mm_per_px"] = 2.0
    with pytest.raises(ValueError, match="scale"):
        check_comparison(after, before)
