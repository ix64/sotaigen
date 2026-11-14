"""The weight transfer on made-up meshes: the closest points are exact, what's trusted is copied, what
isn't is solved, and pieces with nothing near them, flipped faces and held vertices come out as promised."""

import numpy as np
import pytest

from sotaigen.output.skin_transfer import (
    closest_points,
    inpaint,
    laplacian,
    limited,
    transfer_weights,
)


def uv_sphere(radius=1.0, centre=(0, 0, 0), rows=12, sides=16):
    """A closed sphere as triangles, poles included, faces outward."""
    z = np.cos(np.linspace(0, np.pi, rows + 1))[1:-1]
    around = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = np.sqrt(1 - z * z)
    points = [(0, 0, 1)]
    for r, h in zip(ring, z, strict=True):
        points += [(r * np.cos(a), r * np.sin(a), h) for a in around]
    points.append((0, 0, -1))
    faces = [(0, 1 + k, 1 + (k + 1) % sides) for k in range(sides)]
    for i in range(rows - 2):
        for k in range(sides):
            a, b = 1 + i * sides + k, 1 + i * sides + (k + 1) % sides
            faces += [(a, a + sides, b + sides), (a, b + sides, b)]
    last = len(points) - 1
    faces += [(last, last - sides + (k + 1) % sides, last - sides + k) for k in range(sides)]
    return np.array(points) * radius + np.asarray(centre, float), np.array(faces)


def height_weights(points):
    """Two bones that share a body by height, as a neck and a head do."""
    share = np.clip((points[:, 2] + 1) / 2, 0, 1)
    return np.column_stack((1 - share, share))


def test_closest_points_match_brute_force():
    verts, tris = uv_sphere()
    rng = np.random.default_rng(1)
    points = rng.normal(size=(200, 3)) * 1.2
    squared, faces, bary = closest_points(points, verts, tris)
    # Far more samples than triangles can differ by: the exact answer is no further than any of them.
    samples = rng.dirichlet((1, 1, 1), 400)
    on = np.einsum("si,tij->tsj", samples, verts[tris]).reshape(-1, 3)
    brute = np.min(np.linalg.norm(points[:, None] - on[None], axis=2), axis=1) ** 2
    assert (squared <= brute + 1e-12).all()
    found = np.einsum("ni,nij->nj", bary, verts[tris[faces]])
    assert np.allclose(np.sum((found - points) ** 2, 1), squared)
    assert (bary >= -1e-12).all() and np.allclose(bary.sum(1), 1)


def test_the_laplacian_is_zero_on_a_constant_and_keeps_its_edges():
    verts, tris = uv_sphere()
    L, mass = laplacian(verts, tris)
    assert np.allclose(L @ np.ones(len(verts)), 0)
    assert (L.tocoo().data[L.tocoo().row != L.tocoo().col] > 0).all()
    assert mass.sum() == pytest.approx(4 * np.pi, rel=0.03)


def test_inpainting_gives_the_same_weights_in_any_unit():
    verts, tris = uv_sphere()
    weights = height_weights(verts)
    known = verts[:, 2] > 0.2
    known |= verts[:, 2] < -0.2
    in_one = inpaint(verts, tris, known, weights)
    in_other = inpaint(verts * 1000, tris, known, weights)
    assert np.allclose(in_one, in_other, atol=1e-9)
    # A smooth band between two fixed caps stays within what surrounds it.
    assert in_one[~known, 1].min() >= weights[known, 1].min() - 1e-9


def test_a_mesh_on_its_source_copies_it():
    verts, tris = uv_sphere()
    result = transfer_weights(verts, tris, height_weights(verts), verts * 1.001, tris, smooth=False)
    assert result.matched.all()
    assert np.allclose(result.weights, height_weights(verts), atol=2e-3)


def test_vertices_too_far_are_solved_and_stay_a_partition():
    source, tris = uv_sphere()
    target = source.copy()
    # A horn out of the side: past the distance limit, and its normals turned away from the source's.
    horn = (target[:, 0] > 0.7) & (np.abs(target[:, 2]) < 0.4)
    target[horn] *= 2.2
    result = transfer_weights(source, tris, height_weights(source), target, tris)
    assert not result.matched[horn].any()
    assert result.matched[~horn].mean() > 0.8
    assert np.allclose(result.weights.sum(1), 1)
    # The horn is at mid-height, so it takes a mix of the two bones from round its root.
    assert (0.15 < result.weights[horn, 1]).all() and (result.weights[horn, 1] < 0.85).all()


def test_flipped_faces_still_count_as_close():
    source, tris = uv_sphere()
    weights = height_weights(source)
    flipped = tris[:, ::-1]
    assert transfer_weights(source, tris, weights, source * 1.01, flipped).matched.all()
    assert not transfer_weights(source, tris, weights, source * 1.01, flipped, either_way=False).matched.any()


def test_a_piece_with_nothing_near_it_moves_as_one():
    source, tris = uv_sphere()
    weights = height_weights(source)
    near, near_tris = uv_sphere(1.01)
    far, far_tris = uv_sphere(0.1, centre=(3, 0, 0.5))
    target = np.vstack((near, far))
    faces = np.vstack((near_tris, far_tris + len(near)))
    result = transfer_weights(source, tris, weights, target, faces, distance=0.1)
    piece = result.weights[len(near) :]
    assert not result.matched[len(near) :].any()
    assert np.ptp(piece, axis=0).max() < 1e-6
    # With the weights of the source's point nearest to it: the equator's side, a little above.
    closest = height_weights(np.array([[1.0, 0.0, 0.16]]))[0]
    assert piece[0] == pytest.approx(closest, abs=0.1)


def test_a_vertex_no_face_uses_takes_its_closest_point():
    source, tris = uv_sphere()
    weights = height_weights(source)
    target = np.vstack((source * 1.01, [[0.0, 0.0, 1.05]]))
    result = transfer_weights(source, tris, weights, target, tris)
    assert result.weights[-1] == pytest.approx(height_weights(np.array([[0.0, 0.0, 1.0]]))[0], abs=1e-6)


def test_held_vertices_keep_exactly_their_rows():
    source, tris = uv_sphere()
    held = np.arange(5, 25)
    rows = np.tile([0.25, 0.75], (len(held), 1))
    result = transfer_weights(source, tris, height_weights(source), source, tris, fixed=(held, rows))
    assert np.array_equal(result.weights[held], rows)


def test_at_most_the_limit_of_bones_on_a_vertex():
    rows = limited(np.array([[0.4, 0.3, 0.2, 0.05, 0.05, -0.1]]), np.eye(6)[:1], 4)
    assert np.count_nonzero(rows) == 4
    assert rows.sum() == pytest.approx(1)
    assert rows[0, 5] == 0
    # A row with nothing left takes its fallback.
    assert limited(np.zeros((1, 6)), np.eye(6)[:1], 4)[0, 0] == 1
