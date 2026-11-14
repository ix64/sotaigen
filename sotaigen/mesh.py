"""Plain polygon-mesh helpers: points are an N x 3 array, faces a list of vertex-index lists."""

import numpy as np
from scipy.spatial import cKDTree

# A face below this share of its rest area counts as collapsed.
COLLAPSED = 0.15
# How near a vertex's mirror image must lie for a shape to count as symmetric, in millimetres. Far
# above the solver's own noise between the sides of a symmetric body (up to 3e-7), far below anything
# that could be seen or change a contact. Too strict only costs a bake.
MIRROR_TOLERANCE = 1e-5
MIRRORED = np.array([-1.0, 1.0, 1.0])


def write_obj(path, points, faces):
    with open(path, "w") as f:
        f.writelines(f"v {a:.3f} {b:.3f} {c:.3f}\n" for a, b, c in points)
        f.writelines("f " + " ".join(str(i + 1) for i in face) + "\n" for face in faces)


def triangulate(faces, template_verts):
    """Faces as a T x 3 array: a quad across its shorter diagonal on the template, others as fans.

    The template is the same for every body, which keeps one triangle list across bodies. A face
    with no template position (the head, whose vertices are made per body) ties, and takes the
    diagonal from its first corner.
    """
    out = []
    for face in faces:
        if len(face) == 4:
            a, b, c, d = face
            first = np.linalg.norm(template_verts[a] - template_verts[c])
            second = np.linalg.norm(template_verts[b] - template_verts[d])
            if second < first:
                out += [(b, c, d), (b, d, a)]
            else:
                out += [(a, b, c), (a, c, d)]
        else:
            out += [(face[0], face[i], face[i + 1]) for i in range(1, len(face) - 1)]
    return np.array(out)


def unit_normals(points, triangles):
    """Area-weighted unit normals, at the points' precision. Moving points along them stays exact."""
    corners = points[triangles]
    crosses = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals = np.zeros_like(points)
    for i in range(3):
        np.add.at(normals, triangles[:, i], crosses)
    sizes = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.where(sizes > 0, normals / np.maximum(sizes, 1e-30), (0.0, 1.0, 0.0))


def vertex_normals(points, triangles):
    """unit_normals as float32, so that they stay unit once stored."""
    normals = unit_normals(points, triangles).astype(np.float32)
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def face_normals(points, faces):
    """Each face's fan-summed cross products: outward, and as long as twice the face's area.

    They are not unit vectors; see unit_face_normals.
    """
    out = np.zeros((len(faces), 3))
    sizes = np.fromiter(map(len, faces), int, len(faces))
    for size in np.unique(sizes):
        rows = np.flatnonzero(sizes == size)
        p = points[np.array([faces[i] for i in rows])]
        for j in range(1, size - 1):
            out[rows] += np.cross(p[:, j] - p[:, 0], p[:, j + 1] - p[:, 0])
    return out


def unit_face_normals(points, faces):
    """face_normals scaled to unit length; a face with no area keeps a near-zero normal."""
    normals = face_normals(points, faces)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)


def face_areas(points, faces):
    return np.linalg.norm(face_normals(points, faces), axis=1) / 2


def face_edges(face):
    """A face's edges as vertex pairs, with each pair's lower index first."""
    return [(min(a, b), max(a, b)) for a, b in zip(face, face[1:] + face[:1], strict=True)]


def edges_of(faces):
    """Every edge once, sorted, as an E x 2 array."""
    return np.array(sorted({edge for face in faces for edge in face_edges(face)}))


def edge_owners(faces):
    """Each edge's faces, as {edge: [face index, ...]}."""
    owners = {}
    for i, face in enumerate(faces):
        for edge in face_edges(face):
            owners.setdefault(edge, []).append(i)
    return owners


def face_pairs(faces):
    """Pairs of faces that share an edge, as a P x 2 array; boundary edges give none."""
    return np.array([owners for owners in edge_owners(faces).values() if len(owners) == 2])


def adjacency(n, faces):
    """Vertex adjacency as index arrays for a uniform Laplacian: source, sink, count per vertex."""
    edges = edges_of(faces)
    pairs = np.vstack((edges, edges[:, ::-1]))
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    return pairs[:, 0], pairs[:, 1], np.maximum(np.bincount(pairs[:, 0], minlength=n), 1)


def vertex_neighbours(n, faces):
    """Each vertex's neighbours as a list per vertex, in no particular order."""
    out = [[] for _ in range(n)]
    for a, b in edges_of(faces).tolist():
        out[a].append(b)
        out[b].append(a)
    return out


def skin_normals(base, verts):
    """The skin's area-weighted normals in the rig's frame, before any UV seam splits a vertex.

    What glb_bytes stores unless told otherwise, and what to start from when replacing only some of
    them. A UV seam is no edge of the surface, so its copies share one normal and shading can't crease.
    """
    return vertex_normals(
        np.asarray(verts, float), triangulate(base["faces"], np.asarray(base["verts"], float))
    )


def opposite_vertices(verts):
    """Each vertex's mirror image across x = 0, within MIRROR_TOLERANCE, or None if the shape has none.

    From the rest shape and not the template, whose head vertices are all still at the origin.
    """
    verts = np.asarray(verts, float)
    distance, opposite = cKDTree(verts).query(verts * MIRRORED)
    unique = len(np.unique(opposite)) == len(verts)
    return opposite if unique and distance.max() <= MIRROR_TOLERANCE else None
