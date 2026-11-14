"""Skin weights from one mesh onto another that has a different shape and different vertices.

The method of Abdrashitov et al., "Robust Skin Weights Transfer via Weight Inpainting" (SIGGRAPH Asia
2023 Technical Communications, doi:10.1145/3610543.3626180), rewritten on numpy and scipy alone. A
target vertex close to the source, whose normal agrees, takes the weights at its closest point; the
others are solved for, minimising a smoothness energy with the trusted ones held fixed. Its authors'
example (MIT) needs libigl for the closest points, the Laplacian and the solve; those three are here.

What differs from their example, all for meshes a person made in Blender rather than a scan:

- A normal may point either way: a shell with its faces flipped still counts as agreeing.
- A piece with no vertex close enough (a strand, an eye, a tooth) would leave the solve singular, so
  its vertex nearest the source is trusted anyway. With one fixed vertex the solve gives the whole
  piece that vertex's weights, so such a piece moves rigidly with the closest thing to it.
- The Laplacian's edge weights are floored above zero. That keeps the energy positive on bad triangles
  without an intrinsic triangulation, at the cost of some accuracy on them.

Nothing here knows about heads: docs/parts/head.md#权重转移 says how the head swap uses it, and clothes
are meant to use it unchanged.
"""

from typing import NamedTuple

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from sotaigen.mesh import triangulate, unit_normals

# The paper's two trust limits: a share of the target's bounding box diagonal, and degrees.
DISTANCE_SHARE = 0.05
ANGLE_DEG = 30.0
# Its optional smoothing of the solved vertices and what is near them.
SMOOTH_STEPS = 10
SMOOTH_ALPHA = 0.2
# glTF's and VRM's limit; the skin's own weights keep to it too.
MAX_INFLUENCES = 4
# Below this a weight is dropped before normalising.
SMALLEST_WEIGHT = 1e-4
# How many query points go to the KD-tree at once: candidates are held in memory per batch.
BATCH = 4096


class Transfer(NamedTuple):
    weights: np.ndarray  # V x B, each row summing to 1
    matched: np.ndarray  # which vertices took their weights from the closest point
    closest: np.ndarray  # V x B, the weights at the closest point, before anything else


def triangles_of(faces, verts):
    """A T x 3 array from polygons, or the array itself when it is one already."""
    if isinstance(faces, np.ndarray) and faces.ndim == 2 and faces.shape[1] == 3:
        return faces.astype(int)
    faces = [list(map(int, face)) for face in faces]
    return triangulate(faces, np.asarray(verts, float)).astype(int) if faces else np.empty((0, 3), int)


def closest_barycentric(points, a, b, c):
    """The closest point on each triangle (a, b, c) to each point, as N x 3 barycentric coordinates."""
    ab, ac = b - a, c - a
    ap, bp, cp = points - a, points - b, points - c
    d1, d2 = np.sum(ab * ap, 1), np.sum(ac * ap, 1)
    d3, d4 = np.sum(ab * bp, 1), np.sum(ac * bp, 1)
    d5, d6 = np.sum(ab * cp, 1), np.sum(ac * cp, 1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2

    def share(top, bottom):
        return top / np.where(np.abs(bottom) > 1e-30, bottom, 1e-30)

    whole = va + vb + vc
    v, w = share(vb, whole), share(vc, whole)
    # Each region replaces what came before, in the reverse of the order they are tested in.
    on_bc = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
    along = share(d4 - d3, (d4 - d3) + (d5 - d6))
    v, w = np.where(on_bc, 1 - along, v), np.where(on_bc, along, w)
    on_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    v, w = np.where(on_ac, 0.0, v), np.where(on_ac, share(d2, d2 - d6), w)
    on_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    v, w = np.where(on_ab, share(d1, d1 - d3), v), np.where(on_ab, 0.0, w)
    at_c = (d6 >= 0) & (d5 <= d6)
    v, w = np.where(at_c, 0.0, v), np.where(at_c, 1.0, w)
    at_b = (d3 >= 0) & (d4 <= d3)
    v, w = np.where(at_b, 1.0, v), np.where(at_b, 0.0, w)
    at_a = (d1 <= 0) & (d2 <= 0)
    v, w = np.where(at_a, 0.0, v), np.where(at_a, 0.0, w)
    return np.column_stack((1 - v - w, v, w))


def closest_points(points, verts, triangles):
    """The exact closest point on a triangle mesh to each point: squared distances, triangles and
    barycentric coordinates.

    A triangle can only hold the closest point if its centre is within the distance to the nearest
    corner plus the largest triangle's radius, so only those are tried.
    """
    points, verts, triangles = (np.asarray(points, float), np.asarray(verts, float), np.asarray(triangles, int))
    corners = verts[triangles]
    centres = corners.mean(1)
    reach = np.linalg.norm(corners - centres[:, None], axis=2).max()
    centre_tree = cKDTree(centres)
    corner_tree = cKDTree(verts[np.unique(triangles)])
    squared = np.empty(len(points))
    faces = np.empty(len(points), int)
    bary = np.empty((len(points), 3))
    for start in range(0, len(points), BATCH):
        batch = points[start : start + BATCH]
        upper = corner_tree.query(batch)[0]
        found = centre_tree.query_ball_point(batch, upper + reach * (1 + 1e-9) + 1e-9)
        rows = np.repeat(np.arange(len(batch)), [len(f) for f in found])
        candidates = np.concatenate([np.asarray(f, int) for f in found])
        tri = corners[candidates]
        b = closest_barycentric(batch[rows], tri[:, 0], tri[:, 1], tri[:, 2])
        on = np.einsum("ni,nij->nj", b, tri)
        d = np.sum((on - batch[rows]) ** 2, 1)
        # The smallest distance of each row comes first once sorted by row, then by distance.
        order = np.lexsort((d, rows))
        first = order[np.unique(rows[order], return_index=True)[1]]
        squared[start : start + len(batch)] = d[first]
        faces[start : start + len(batch)] = candidates[first]
        bary[start : start + len(batch)] = b[first]
    return squared, faces, bary


def laplacian(verts, triangles):
    """The cotangent Laplacian (positive off the diagonal) and the barycentric vertex areas.

    Works on any triangle soup: non-manifold edges add up, an unused vertex has an empty row.
    """
    verts = np.asarray(verts, float)
    n = len(verts)
    corners = verts[triangles]
    area2 = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    usable = np.maximum(area2, 1e-12 * max(float(np.mean(area2)) if len(area2) else 1.0, 1e-30))
    rows, cols, values = [], [], []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        u, v = corners[:, j] - corners[:, i], corners[:, k] - corners[:, i]
        rows.append(triangles[:, j])
        cols.append(triangles[:, k])
        values.append(0.5 * np.sum(u * v, 1) / usable)
    a, b = np.concatenate(rows), np.concatenate(cols)
    v = np.concatenate(values)
    summed = sparse.coo_matrix((np.r_[v, v], (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    summed.sum_duplicates()
    # Floored above zero, so that every edge still ties its ends together.
    summed.data = np.maximum(summed.data, 1e-6)
    summed.setdiag(0.0)
    summed.eliminate_zeros()
    L = summed - sparse.diags(np.asarray(summed.sum(1)).ravel())
    mass = np.zeros(n)
    for i in range(3):
        np.add.at(mass, triangles[:, i], area2 / 6)
    used = mass[mass > 0]
    floor = 1e-6 * (float(used.mean()) if len(used) else 1.0)
    return L.tocsr(), np.maximum(mass, floor)


def inpaint(verts, triangles, known, weights):
    """The paper's solve: Q = -L + L M^-1 L with the `known` rows of `weights` held fixed.

    The mesh is measured in units of its bounding box diagonal, so that the same mesh gives the same
    weights in millimetres and in metres.
    """
    verts = np.asarray(verts, float)
    free = np.flatnonzero(~known)
    if not len(free):
        return weights.copy()
    size = max(float(np.linalg.norm(verts.max(0) - verts.min(0))), 1e-12)
    L, mass = laplacian(verts / size, triangles)
    Q = (-L + L @ sparse.diags(1 / mass) @ L).tocsr()
    fixed = np.flatnonzero(known)
    inside = Q[free][:, free].tocsc()
    right = -(Q[free][:, fixed] @ weights[fixed])
    out = weights.copy()
    out[free] = splu(inside).solve(np.asarray(right))
    return out


def components_of(count, triangles):
    graph = sparse.coo_matrix(
        (np.ones(2 * len(triangles)), (np.r_[triangles[:, 0], triangles[:, 1]], np.r_[triangles[:, 1], triangles[:, 2]])),
        shape=(count, count),
    )
    return connected_components(graph, directed=False)[1]


def edge_lengths(verts, triangles):
    """A symmetric sparse graph of the mesh's edges by length, for scipy.sparse.csgraph."""
    a = np.r_[triangles[:, 0], triangles[:, 1], triangles[:, 2]]
    b = np.r_[triangles[:, 1], triangles[:, 2], triangles[:, 0]]
    lengths = np.linalg.norm(verts[a] - verts[b], axis=1)
    n = len(verts)
    graph = sparse.coo_matrix((np.r_[lengths, lengths], (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    # The same edge met twice adds up in a sparse sum; it should stay one length.
    graph.data = graph.data / np.maximum(
        sparse.coo_matrix((np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr().data, 1
    )
    return graph


def smoothed(verts, triangles, weights, matched, reach, held):
    """The paper's optional smoothing, of the solved vertices and those within `reach` along the mesh."""
    if matched.all():
        return weights
    graph = edge_lengths(verts, triangles)
    distance = dijkstra(graph, directed=False, indices=np.flatnonzero(~matched), min_only=True, limit=reach)
    chosen = np.isfinite(distance) & ~held
    neighbours = (graph > 0).astype(float)
    degree = np.asarray(neighbours.sum(1)).ravel()
    step = sparse.diags(1 / np.maximum(degree, 1)) @ neighbours
    out = weights.copy()
    moving = chosen & (degree > 0)
    for _ in range(SMOOTH_STEPS):
        mixed = (1 - SMOOTH_ALPHA) * out + SMOOTH_ALPHA * (step @ out)
        out[moving] = mixed[moving]
    return out


def limited(weights, fallback, max_influences):
    """Rows that are non-negative, keep the largest few and sum to 1. A row left with nothing takes
    its row in `fallback`."""
    out = np.clip(weights, 0.0, None)
    out[out < SMALLEST_WEIGHT] = 0.0
    if max_influences and out.shape[1] > max_influences:
        drop = np.argpartition(out, -max_influences, axis=1)[:, :-max_influences]
        np.put_along_axis(out, drop, 0.0, axis=1)
    totals = out.sum(1)
    empty = totals <= 0
    if empty.any():
        out[empty] = fallback[empty]
        totals = out.sum(1)
    return out / np.maximum(totals[:, None], 1e-30)


def transfer_weights(
    source_verts,
    source_faces,
    source_weights,
    target_verts,
    target_faces,
    *,
    fixed=None,
    distance=None,
    angle=ANGLE_DEG,
    either_way=True,
    smooth=True,
    max_influences=MAX_INFLUENCES,
):
    """The source's V x B weights on the target's vertices.

    Faces are polygons or a T x 3 array. `fixed` is {vertex index array: rows}, kept exactly: as a tuple
    (indices, B-wide rows). `distance` is the closest-point limit in the meshes' own units, by default
    DISTANCE_SHARE of the target's diagonal. `either_way` trusts a normal pointing the opposite way.
    A target of several pieces is fine, and so is a target with vertices no face uses.
    """
    source_verts = np.asarray(source_verts, float)
    target_verts = np.asarray(target_verts, float)
    source_weights = np.asarray(source_weights, float)
    source_tris = triangles_of(source_faces, source_verts)
    target_tris = triangles_of(target_faces, target_verts)
    n = len(target_verts)
    squared, hit, bary = closest_points(target_verts, source_verts, source_tris)
    corners = source_tris[hit]
    closest = np.einsum("ni,nib->nb", bary, source_weights[corners])
    normals = np.einsum("ni,nij->nj", bary, unit_normals(source_verts, source_tris)[corners])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    agree = np.sum(normals * unit_normals(target_verts, target_tris), 1)
    size = float(np.linalg.norm(target_verts.max(0) - target_verts.min(0)))
    limit = DISTANCE_SHARE * size if distance is None else distance
    cosine = np.cos(np.radians(angle))
    matched = (squared <= limit * limit) & ((np.abs(agree) if either_way else agree) >= cosine)
    weights = closest.copy()
    held = np.zeros(n, bool)
    if fixed is not None:
        indices, rows = fixed
        indices = np.asarray(indices, int)
        weights[indices] = rows
        held[indices] = True
    known = matched | held
    # Every piece needs something to hold it, or the solve has no answer.
    labels = components_of(n, target_tris)
    for label in np.flatnonzero(~np.bincount(labels, known.astype(int), minlength=labels.max() + 1).astype(bool)):
        members = np.flatnonzero(labels == label)
        known[members[np.argmin(squared[members])]] = True
    solved = inpaint(target_verts, target_tris, known, weights)
    if smooth:
        # The smoothing reaches as far as the trust limit, as in the paper's example.
        solved = smoothed(target_verts, target_tris, solved, known, limit, held)
    out = limited(solved, closest, max_influences)
    if fixed is not None:
        # Limited rows of the fixed vertices would move them; they are kept as given.
        out[indices] = rows
    return Transfer(out, matched, closest)
