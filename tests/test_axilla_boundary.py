"""Area retention must not turn a fading corrective boundary into an unbounded spike."""

import numpy as np

from sotaigen.corrective.axilla import _patch_topology, _retain_surface


def triangle_area(v):
    return np.linalg.norm(np.cross(v[1] - v[0], v[2] - v[0])) * 0.5


def test_area_protection_recovers_a_compressed_mobile_face():
    rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    goal = rest.copy()
    goal[2, 1] = 0.01
    result = _retain_surface(
        rest,
        goal,
        np.array([[0, 1, 2, 2]]),
        np.tile(np.eye(3), (3, 1, 1)),
        np.array([0.0, 0.0, 1.0]),
    )
    np.testing.assert_array_equal(result[:2], goal[:2])
    assert triangle_area(result) >= 0.27 * triangle_area(rest)
    assert np.linalg.norm(result - goal, axis=1).max() < 1


def test_nearly_fixed_boundary_converges_to_fixed_boundary():
    rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    goal = rest.copy()
    goal[2, 1] = 0.01
    faces = np.array([[0, 1, 2, 2]])
    rotations = np.tile(np.eye(3), (3, 1, 1))
    distances = []
    for epsilon in [0.0, 1e-6, 1e-4, 1e-2]:
        result = _retain_surface(rest, goal, faces, rotations, np.array([0.0, 0.0, epsilon]))
        np.testing.assert_array_equal(result[:2], goal[:2])
        distances.append(np.linalg.norm(result[2] - goal[2]))
    assert distances[0] == 0
    assert distances[1] < 1e-5
    assert distances[2] < 1e-3
    assert distances[3] < 0.02
    assert np.all(np.diff(distances) > 0)


def test_boundary_solve_is_covariant_under_scale_and_rotation():
    rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    goal = rest.copy()
    goal[2, 1] = 0.01
    faces = np.array([[0, 1, 2, 2]])
    free = np.array([0.0, 0.0, 0.05])
    rotations = np.tile(np.eye(3), (3, 1, 1))
    expected = _retain_surface(rest, goal, faces, rotations, free)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    shift = np.array([3.0, 5.0, 7.0])
    result = _retain_surface(
        rest * 100 @ rotation.T + shift, goal * 100 @ rotation.T + shift, faces, rotations, free
    )
    np.testing.assert_allclose(result, expected * 100 @ rotation.T + shift, atol=1e-10)


def test_patch_includes_the_other_half_of_a_boundary_quad():
    v = np.array([[0., 0, 0], [1., 0, 0], [1., 1, 0], [0., 1, 0]])
    ids, faces, polygons = _patch_topology([[0, 1, 2, 3]], v, np.array([0, 1., 0, 0]))
    np.testing.assert_array_equal(ids, np.arange(4))
    assert len(faces) == 2
    np.testing.assert_array_equal(ids[polygons[-1]], [0, 1, 2, 3])


def test_triangle_padding_does_not_change_material_correction():
    rest = np.array([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.]])
    goal = np.array([[0., 0., 0.], [1.4, .2, 0.], [.3, .04, 0.]])
    from scipy.spatial.transform import Rotation

    rotations = Rotation.from_euler("xyz", [[0., 0, 0], [.4, .1, 0], [-.3, .5, 0]]).as_matrix()
    free = np.ones(3)
    expected = _retain_surface(rest, goal, np.array([[0, 1, 2]]), rotations, free)
    # The old padded solve differed by 0.163 for cyclic orders of this one face.
    for face in [[0, 1, 2, 2], [1, 2, 0, 0], [2, 0, 1, 1], [0, 1, 1, 2]]:
        result = _retain_surface(rest, goal, np.array([face]), rotations, free)
        np.testing.assert_allclose(result, expected, atol=1e-12)


def test_fading_edge_budget_is_independent_of_face_winding():
    rest = np.array([[0., 0., 0.], [3., 0., 0.], [0., 1., 0.]])
    goal = rest.copy()
    goal[2, 1] = .01
    rotations = np.tile(np.eye(3), (3, 1, 1))
    free = np.array([0., .001, .02])
    forward = _retain_surface(rest, goal, np.array([[0, 1, 2]]), rotations, free)
    backward = _retain_surface(rest, goal, np.array([[2, 1, 0]]), rotations, free)
    np.testing.assert_allclose(backward, forward, atol=1e-12)
