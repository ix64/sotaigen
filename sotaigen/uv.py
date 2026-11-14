"""The skin's UV atlas and how to judge one: overlaps, flips and stretch, on plain arrays.

UVs belong to a face's corners, not its vertices, because an island's edge splits a vertex in two.
sotaigen/data/template/uv.npz is written by template/extract_uv.py and shared by every body.
"""


import numpy as np

from sotaigen.paths import DATA

UV_FILE = DATA / "template" / "uv.npz"
# Interiors must overlap by more than this, in UV units, to count. Neighbours share an edge, and
# stored UVs are rounded to 1/65535.
OVERLAP_TOLERANCE = 1e-5


def load_uv(faces, path=UV_FILE):
    """Each corner's (u, v), face after face, for this face list; refuses any other topology."""
    with np.load(path) as data:
        corners, sizes = data["corners"], data["sizes"]
    wanted = [len(face) for face in faces]
    if sizes.tolist() != wanted:
        raise ValueError(
            f"{path.name} has {len(sizes)} faces for a mesh of {len(wanted)}, or other face sizes: "
            "rerun template/extract_uv.py"
        )
    return corners / 65535.0


def export_fields(faces, path=UV_FILE):
    """The `uvs` and `faceUvs` that sotaigen.output.glb.glb_bytes takes: shared UVs, with V down as in glTF."""
    uv = load_uv(faces, path)
    uvs, index = np.unique(uv * (1.0, -1.0) + (0.0, 1.0), axis=0, return_inverse=True)
    index = index.ravel().tolist()
    corners, start = [], 0
    for face in faces:
        corners.append(index[start : start + len(face)])
        start += len(face)
    return {"uvs": uvs, "faceUvs": corners}


def corner_triangles(faces):
    """A fan over each face, as T x 3 corner indices, with each triangle's face beside it."""
    triangles, owners, start = [], [], 0
    for index, face in enumerate(faces):
        for k in range(1, len(face) - 1):
            triangles.append((start, start + k, start + k + 1))
            owners.append(index)
        start += len(face)
    return np.array(triangles), np.array(owners)


def signed_areas(uv, triangles):
    a, b, c = uv[triangles[:, 0]], uv[triangles[:, 1]], uv[triangles[:, 2]]
    u, v = b - a, c - a
    return (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) / 2


def _separated(first, second, tolerance):
    """For each pair of triangles (P x 3 x 2), whether an edge of `first` separates them."""
    edge = np.roll(first, -1, axis=1) - first
    normal = np.stack((edge[..., 1], -edge[..., 0]), axis=-1)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
    own = np.einsum("pei,pvi->pev", normal, first)
    other = np.einsum("pei,pvi->pev", normal, second)
    # Either winding: the triangles sit on opposite sides of this edge's line.
    return (
        (other.min(2) >= own.max(2) - tolerance) | (other.max(2) <= own.min(2) + tolerance)
    ).any(1)


def overlapping_triangles(uv, triangles, tolerance=OVERLAP_TOLERANCE, cells=128):
    """Pairs of triangles whose interiors overlap, as P x 2, from a grid and the separating axis."""
    points = uv[triangles]
    low, high = points.min(1), points.max(1)
    origin = uv.min(0)
    size = np.maximum((uv.max(0) - origin).max(), 1e-9) / cells
    first_cell = np.floor((low - origin) / size).astype(int)
    last_cell = np.floor((high - origin) / size).astype(int)
    buckets = {}
    for t, (a, b) in enumerate(zip(first_cell, last_cell, strict=True)):
        for i in range(a[0], b[0] + 1):
            for j in range(a[1], b[1] + 1):
                buckets.setdefault((i, j), []).append(t)
    candidates = set()
    for members in buckets.values():
        for x, first in enumerate(members):
            for second in members[x + 1 :]:
                candidates.add((first, second))
    if not candidates:
        return np.empty((0, 2), int)
    pairs = np.array(sorted(candidates))
    a, b = points[pairs[:, 0]], points[pairs[:, 1]]
    apart = _separated(a, b, tolerance) | _separated(b, a, tolerance)
    return pairs[~apart]


def stretch(points, uv, corner_verts, triangles):
    """Each triangle's singular values from surface millimetres to UV, normalised over the mesh.

    With the whole mesh's area ratio divided out, a triangle with (1, 1) has the atlas's average
    texel density and no distortion. Returns T x 2 with the larger value first, and each area in mm2.
    """
    p = points[corner_verts[triangles]]
    q = uv[triangles]
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    area = np.linalg.norm(np.cross(e1, e2), axis=1) / 2
    x = e1 / np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-9)
    normal = np.cross(e1, e2)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    y = np.cross(normal, x)
    # The triangle in its own plane, in mm, and the map from there to UV.
    source = np.empty((len(p), 2, 2))
    source[:, 0, 0] = np.sum(e1 * x, axis=1)
    source[:, 1, 0] = 0.0
    source[:, 0, 1] = np.sum(e2 * x, axis=1)
    source[:, 1, 1] = np.sum(e2 * y, axis=1)
    target = np.stack((q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]), axis=2)
    degenerate = area < 1e-9
    source[degenerate] = np.eye(2)
    jacobian = target @ np.linalg.inv(source)
    values = np.linalg.svd(jacobian, compute_uv=False)
    values /= np.sqrt(np.abs(signed_areas(uv, triangles)).sum() / max(area.sum(), 1e-9))
    return values, area


def weighted_percentile(values, weights, share):
    order = np.argsort(values)
    running = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(running, share * running[-1])])


def stretch_report(values, area):
    """Area-weighted summary: `anisotropy` is how far a circle goes oval, `density` is its texel size."""
    anisotropy = values[:, 0] / np.maximum(values[:, 1], 1e-9)
    density = np.sqrt(values[:, 0] * values[:, 1])
    # Sander's L2 stretch of the normalised map: 1 means no stretch at all.
    l2 = np.sqrt(np.sum((values**2).sum(1) / 2 * area) / area.sum())
    return {
        "l2": round(float(l2), 3),
        "anisotropyP50": round(weighted_percentile(anisotropy, area, 0.5), 2),
        "anisotropyP95": round(weighted_percentile(anisotropy, area, 0.95), 2),
        "anisotropyP999": round(weighted_percentile(anisotropy, area, 0.999), 2),
        "anisotropyMax": round(float(anisotropy.max()), 2),
        "densityP5": round(weighted_percentile(density, area, 0.05), 2),
        "densityP95": round(weighted_percentile(density, area, 0.95), 2),
    }
