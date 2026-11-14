"""Internal support must retain attachments without making the skin stick to it."""

import numpy as np
from scipy.spatial.transform import Rotation

from sotaigen.contact.axilla_ribbons import Ribbon, constrain, swing, transport


def test_chord_transport_rest_rigid_motion_and_endpoint_attachments():
    a, b = np.array([0., 0, 0]), np.array([2., 0, 0])
    points = np.array([a, b, [1., 1, 0]])
    identity = np.eye(4)
    np.testing.assert_allclose(transport(points, a, b, identity, identity), points)
    rigid = np.eye(4)
    rigid[:3, :3] = Rotation.from_euler('xyz', [.5, -.3, .7]).as_matrix()
    rigid[:3, 3] = [3., -2., 1.]
    np.testing.assert_allclose(transport(points, a, b, rigid, rigid),
                               points @ rigid[:3, :3].T + rigid[:3, 3], atol=1e-12)
    stretched = identity.copy()
    stretched[0, 3] = 2
    result = transport(points, a, b, identity, stretched)
    np.testing.assert_allclose(result, [[0, 0, 0], [4, 0, 0], [2, 1, 0]])


def test_half_turn_is_finite_and_stays_a_rotation():
    matrix = swing([1., 0, 0], [-1., 0, 0])
    np.testing.assert_allclose(matrix @ [1., 0, 0], [-1., 0, 0], atol=1e-12)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(matrix), 1.)


def fixture():
    points = np.array([[0., 1, 0], [1., 1, 0], [2., 1, 0], [1., 1, 2]])
    curve = np.array([[0., 0, 0], [2., 0, 0]])
    ribbon = Ribbon('front', 'left', np.array([0, 1, 2]), np.array([0, 1, 2]),
                    np.array([1., 1., 1., 0.]), 0, 1, 2., curve, 1.)
    normals = np.tile([0., 1., 0.], (4, 1))
    skins = np.tile(np.eye(4), (2, 1, 1))
    return points, ribbon, normals, skins


def test_support_pushes_locally_but_skin_can_separate_and_return():
    points, ribbon, normals, skins = fixture()
    np.testing.assert_array_equal(constrain(points, points, skins, [ribbon], normals), points)
    skins[:, 1, 3] = .5
    result = constrain(points, points, skins, [ribbon], normals)
    np.testing.assert_allclose(result[:3], points[:3] + [0., .5, 0.])
    np.testing.assert_array_equal(result[3], points[3])
    np.testing.assert_array_equal(constrain(points, points, skins, [ribbon], normals, ('back',)), points)
    skins[:, 1, 3] = -.5
    np.testing.assert_array_equal(constrain(points, points, skins, [ribbon], normals), points)
    skins[:, 1, 3] = 0
    np.testing.assert_array_equal(constrain(points, points, skins, [ribbon], normals), points)


def test_preexisting_clearance_and_inward_skin_are_not_pushed():
    points, ribbon, normals, skins = fixture()
    points[:, 1] *= .6
    np.testing.assert_allclose(constrain(points, points, skins, [ribbon], normals), points)
    skins[:, 1, 3] = .3
    np.testing.assert_array_equal(constrain(points, points, skins, [ribbon], -normals), points)


def test_support_respects_uniform_scale():
    points, ribbon, normals, skins = fixture()
    skins[:, 1, 3] = .5
    result = constrain(points, points, skins, [ribbon], normals)
    factor = 123.
    ribbon.centreline *= factor
    ribbon.radius *= factor
    skins[:, :3, 3] *= factor
    scaled = constrain(points * factor, points * factor, skins, [ribbon], normals)
    np.testing.assert_allclose(scaled, result * factor, atol=1e-10)


def test_support_is_covariant_under_reflection():
    points, ribbon, normals, skins = fixture()
    skins[:, 1, 3] = .5
    expected = constrain(points, points, skins, [ribbon], normals)
    reflection = np.diag([-1., 1., 1., 1.])
    ribbon.centreline = ribbon.centreline @ reflection[:3, :3]
    mirrored = points @ reflection[:3, :3]
    result = constrain(mirrored, mirrored, reflection @ skins @ reflection,
                       [ribbon], normals @ reflection[:3, :3])
    np.testing.assert_allclose(result, expected @ reflection[:3, :3], atol=1e-12)
