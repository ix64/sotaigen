"""The optional offline contact prototype must preserve sheet ordering and its collar."""

import ipctk
import numpy as np
import pytest

from sotaigen.contact.solver import ContactPatch, vertex_filter


def sheets():
    lower = np.array([[-.1, -.1, 0], [.1, -.1, 0], [0, .1, 0]])
    upper = lower * .7 + [0, 0, .03]
    v = np.vstack((lower, upper))
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    polygons = np.column_stack((faces, faces[:, -1]))
    return v, faces, polygons


def wavy_grid(size=7):
    """A bumpy sheet of quads, with a band of them split into triangles padded to four corners."""
    u, w = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size), indexing="ij")
    v = np.column_stack((u.ravel(), w.ravel(), .05 * np.sin(5 * u.ravel()) * np.cos(4 * w.ravel())))
    at = np.arange(size * size).reshape(size, size)
    quads = np.column_stack((at[:-1, :-1].ravel(), at[1:, :-1].ravel(), at[1:, 1:].ravel(), at[:-1, 1:].ravel()))
    split = np.arange(len(quads)) % 3 == 0
    halves = np.vstack((quads[split][:, [0, 1, 2]], quads[split][:, [0, 2, 3]]))
    faces = np.vstack((quads[~split][:, [0, 1, 2]], quads[~split][:, [0, 2, 3]], halves))
    return v, faces, np.vstack((quads[~split], np.column_stack((halves, halves[:, -1]))))


def test_area_hessian_layout_adds_blocks_in_scipys_order():
    # Overlapping blocks of random values: any other order of adding the duplicates rounds differently.
    from scipy import sparse

    v, faces, polygons = wavy_grid()
    patch = ContactPatch(v, faces, polygons, np.ones(len(v)))
    rng = np.random.default_rng(0)
    blocks = rng.normal(size=(len(polygons), 12, 12))
    blocks[rng.random(len(polygons)) < .4] = 0
    active = np.flatnonzero(blocks.any(axis=(1, 2)))
    rows = np.repeat(patch.corner_dofs, 12, axis=1).ravel()
    cols = np.tile(patch.corner_dofs, (1, 12)).ravel()
    expected = sparse.coo_matrix((blocks.ravel(), (rows, cols)), shape=(v.size, v.size)).tocsc()
    summed = patch._assemble_area(active, blocks[active])
    result = sparse.csc_matrix((summed, patch.area_indices, patch.area_indptr), shape=(v.size, v.size))
    for matrix in (expected, result):
        matrix.eliminate_zeros()
    np.testing.assert_array_equal(result.indptr, expected.indptr)
    np.testing.assert_array_equal(result.indices, expected.indices)
    np.testing.assert_array_equal(result.data, expected.data)


def test_free_block_is_the_sliced_hessian():
    from scipy import sparse

    v, faces, polygons = wavy_grid()
    mobility = np.ones(len(v))
    mobility[::5] = 0
    patch = ContactPatch(v, faces, polygons, mobility)
    matrix = sparse.random(v.size, v.size, density=.05, format="csc", random_state=1)
    expected = matrix[patch.dofs][:, patch.dofs]
    expected.sum_duplicates()
    result = patch._free_block(matrix)
    np.testing.assert_array_equal(result.indptr, expected.indptr)
    np.testing.assert_array_equal(result.indices, expected.indices)
    np.testing.assert_array_equal(result.data, expected.data)


def test_vertex_filter_drops_excluded_vertices_and_paired_pairs():
    rng = np.random.default_rng(2)
    excluded, paired = rng.random(40) < .2, rng.random(40) < .4
    allowed = vertex_filter(excluded, paired)
    for i in range(40):
        for j in range(40):
            if i != j:  # ipctk never asks about a vertex and itself
                assert allowed(i, j) == (not (excluded[i] or excluded[j] or (paired[i] and paired[j])))


def test_contact_follows_without_tunnelling_or_moving_fixed_sheet():
    ipctk.set_num_threads(1)
    v, f, p = sheets()
    patch = ContactPatch(v, f, p, np.array([0, 0, 0, 1., 1., 1.]))
    rotations = np.tile(np.eye(3), (len(v), 1, 1))
    for height in [.02, .01, 0, -.01, -.02]:
        goal = v.copy()
        goal[3:, 2] = height
        result, _ = patch.advance(goal, rotations)
        np.testing.assert_array_equal(result[:3], v[:3])
        assert np.all(result[3:, 2] > 0)
        assert not ipctk.has_intersections(patch.mesh, result)
    # A mere endpoint intersection check would accept a sheet that tunnelled through.
    assert not ipctk.has_intersections(patch.mesh, goal)
    assert np.all(goal[3:, 2] < 0)


def test_noncontact_translation_is_exact():
    v, f, p = sheets()
    patch = ContactPatch(v, f, p, np.ones(len(v)))
    goal = v + [.02, .01, 0]
    result, _ = patch.advance(goal, np.tile(np.eye(3), (len(v), 1, 1)))
    np.testing.assert_allclose(result, goal, atol=1e-9)


def test_intersected_start_is_rejected():
    v, f, p = sheets()
    v[3:, 2] = [-.01, .01, .01]
    with pytest.raises(ValueError, match="intersection-free"):
        ContactPatch(v, f, p, np.ones(len(v)))


def test_rest_is_unchanged_and_contact_clearance_is_not_assumed():
    v, f, p = sheets()
    patch = ContactPatch(v, f, p, np.ones(len(v)))
    result, _ = patch.advance(v, np.tile(np.eye(3), (len(v), 1, 1)))
    np.testing.assert_array_equal(result, v)
    v[3:, 2] = 1e-5
    with pytest.raises(ValueError, match="insufficient"):
        ContactPatch(v, f, p, np.ones(len(v)))


def test_release_after_contact_recovers_without_sticking_or_tunnelling():
    ipctk.set_num_threads(1)
    v, f, p = sheets()
    patch = ContactPatch(v, f, p, np.array([0, 0, 0, 1., 1., 1.]))
    rotations = np.tile(np.eye(3), (len(v), 1, 1))
    previous = v.copy()
    for height in [.01, -.02, -.01, 0., .01, .02, .03]:
        goal = v.copy()
        goal[3:, 2] = height
        result, _ = patch.advance(goal, rotations)
        assert ipctk.compute_collision_free_stepsize(patch.mesh, previous, result) == 1
        np.testing.assert_array_equal(result[:3], v[:3])
        assert np.all(result[3:, 2] > 0)
        previous = result.copy()
    np.testing.assert_allclose(result, v, atol=1e-8)


@pytest.mark.parametrize("intrinsic", [False, True])
def test_contact_area_normals_ignore_repeated_triangle_corners(intrinsic):
    from scipy.spatial.transform import Rotation

    ipctk.set_num_threads(1)
    v, faces, padded = sheets()
    goal = v * [.4, .5, 1.]
    rotations = Rotation.from_euler('xyz', [[0., 0, 0], [.4, .1, 0], [-.3, .5, 0]] * 2).as_matrix()
    expected, _ = ContactPatch(v, faces, faces, np.ones(len(v)),
                               intrinsic_area=intrinsic).advance(goal, rotations)
    for polygons in [padded, np.array([[1, 2, 0, 0], [4, 5, 3, 3]])]:
        result, _ = ContactPatch(v, faces, polygons, np.ones(len(v)),
                                 intrinsic_area=intrinsic).advance(goal, rotations)
        np.testing.assert_allclose(result, expected, atol=1e-10)


@pytest.mark.parametrize("intrinsic", [False, True])
def test_area_gradient_and_exact_hessian_match_finite_differences(intrinsic):
    v, f, p = sheets()
    patch = ContactPatch(v, f, p, np.ones(len(v)), intrinsic_area=intrinsic)
    points = v * [.4, .4, 1.]
    normals = patch.normal
    gradient, hessian = patch._area_terms(points, normals, True, project=False)
    epsilon = 1e-6
    numeric_gradient = np.zeros(points.size)
    numeric_hessian = np.zeros((points.size, points.size))
    for i in range(points.size):
        step = np.zeros(points.size)
        step[i] = epsilon
        plus, minus = points + step.reshape(points.shape), points - step.reshape(points.shape)
        numeric_gradient[i] = (patch._area_terms(plus, normals)
                               - patch._area_terms(minus, normals)) / (2 * epsilon)
        g_plus, _ = patch._area_terms(plus, normals, True, project=False)
        g_minus, _ = patch._area_terms(minus, normals, True, project=False)
        numeric_hessian[:, i] = (g_plus - g_minus) / (2 * epsilon)
    np.testing.assert_allclose(gradient, numeric_gradient, atol=1e-8)
    np.testing.assert_allclose(hessian.toarray(), numeric_hessian, atol=1e-7)
    _, projected = patch._area_terms(points, normals, True)
    assert np.linalg.eigvalsh(projected.toarray()).min() > -1e-8


def test_rigid_predictor_moves_nearby_skin_with_its_fixed_boundary():
    ipctk.set_num_threads(1)
    v, faces, polygons = sheets()
    mobility = np.array([0., 0, 0, 1, 1, 1])
    rotation = np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])
    goal = v @ rotation.T
    rotations = np.tile(rotation, (len(v), 1, 1))
    # Moving only the fixed sheet first creates an artificial collision with the
    # old position of the other sheet, even though their common motion is safe.
    patch = ContactPatch(v, faces, polygons, mobility)
    with pytest.raises(RuntimeError, match="Fixed collar"):
        patch.advance(goal, rotations)
    np.testing.assert_array_equal(patch.x, v)
    result, _ = patch.advance(goal, rotations, initial=goal)
    np.testing.assert_allclose(result, goal, atol=1e-10)
    assert ipctk.compute_collision_free_stepsize(patch.mesh, v, result) == 1


def test_intrinsic_area_allows_skin_normal_to_turn_without_compression():
    from scipy.spatial.transform import Rotation

    v, f, p = sheets()
    patch = ContactPatch(v, f, p, np.ones(len(v)), intrinsic_area=True)
    moved = v @ Rotation.from_euler('x', 130, degrees=True).as_matrix().T
    assert patch._area_terms(moved, patch.normal) == pytest.approx(0., abs=1e-25)
    compressed = moved * .4
    assert patch._area_terms(compressed, patch.normal) > 0


def test_material_step_cannot_jump_through_a_flattened_face():
    v, f, p = sheets()
    patch = ContactPatch(v, f, p, np.ones(len(v)), intrinsic_area=True)
    goal = v * [-1, 1, 1]
    # Both endpoints have exactly the original area, but the linear path flattens
    # every triangle at its midpoint. Endpoint-only area tests cannot catch this.
    assert patch._area_terms(goal, patch.normal) == 0
    alpha = patch._area_stepsize(v, goal - v)
    assert .44 < alpha < .5
    assert patch._area_stepsize(v, np.tile([.01, .02, .03], (len(v), 1))) == 1


def test_moving_boundary_solve_reaches_exact_targets_without_teleporting_a_sheet():
    ipctk.set_num_threads(1)
    v, f, p = sheets()
    patch = ContactPatch(v, f, p, np.array([0., 0, 0, 1, 1, 1]),
                         intrinsic_area=True, moving_boundary=True)
    rotation = np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])
    goal = v @ rotation.T
    result, stats = patch.advance(goal, np.tile(rotation, (len(v), 1, 1)))
    assert stats["boundaryIterations"] > 0
    np.testing.assert_array_equal(result[:3], goal[:3])
    np.testing.assert_allclose(result, goal, atol=2e-6)
    assert not ipctk.has_intersections(patch.mesh, result)
