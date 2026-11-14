"""Where skin passes through skin: edge-triangle crossings, and which parts each is between.

The correctives' own fits and audits count crossings, and the pose reports (sotaigen.metrics) count
them on the correctives' results, so this sits below both.
"""

import numpy as np

from sotaigen.mesh import edges_of, triangulate
from sotaigen.rig.skinning import FINE

LEG_PARTS = ("upperLeg", "lowerLeg", "foot", "toes")
# Parts on the midline, which have no side of their own.
MIDDLE_PARTS = ("trunk", "neck", "head")


def grid_cells(low, high, size):
    """Boxes given by their corners as the grid cells they touch: (box index, cell key) per cell."""
    first = np.floor(low / size).astype(np.int64)
    span = np.floor(high / size).astype(np.int64) - first + 1
    count = span.prod(axis=1)
    box = np.repeat(np.arange(len(low)), count)
    # Each box's cells are numbered from 0; that number unravels into offsets along the axes.
    index = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    width, depth = span[box, 0], span[box, 1]
    offset = np.column_stack((index % width, (index // width) % depth, index // (width * depth)))
    cell = first[box] + offset
    # A million cells each way is far past any body, and the key still fits in 64 bits.
    size_in_cells = 1 << 20
    cell = cell + size_in_cells // 2
    return box, (cell[:, 0] * size_in_cells + cell[:, 1]) * size_in_cells + cell[:, 2]


def delivered_skin(faces, template_verts):
    """The triangles the exporter writes, and all their edges, diagonals included."""
    triangles = triangulate(faces, np.asarray(template_verts, float))
    return triangles, edges_of(triangles.tolist())


def edge_triangle_candidates(points, edges, triangles, size):
    """Pairs of an edge and a triangle that share a grid cell, without duplicates."""
    ends, corners = points[edges], points[triangles]
    edge_box, edge_key = grid_cells(ends.min(1), ends.max(1), size)
    tri_box, tri_key = grid_cells(corners.min(1), corners.max(1), size)
    order = np.argsort(tri_key)
    tri_box, tri_key = tri_box[order], tri_key[order]
    start = np.searchsorted(tri_key, edge_key, "left")
    count = np.searchsorted(tri_key, edge_key, "right") - start
    # The triangles of each edge's cell, laid end to end, as positions in the sorted list.
    position = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    triangle = tri_box[np.repeat(start, count) + position]
    pair = np.unique(np.repeat(edge_box, count) * len(triangles) + triangle)
    return np.column_stack((pair // len(triangles), pair % len(triangles)))


def edges_through_triangles(points, edges, triangles):
    """Pairs (edge, triangle) where an edge passes through a triangle it shares no vertex with.

    A crossing means two sheets of skin pass through each other there. Touching doesn't count,
    however close, and neither does a fold that stays on its own side. Returns an N x 2 array.
    """
    lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    # About one edge to a cell. Long edges just touch more cells.
    size = max(float(np.median(lengths)), 1.0)
    pairs = edge_triangle_candidates(points, edges, triangles, size)
    a, b = edges[pairs[:, 0], 0], edges[pairs[:, 0], 1]
    corners = triangles[pairs[:, 1]]
    apart = ~((corners == a[:, None]) | (corners == b[:, None])).any(axis=1)
    pairs, a, b, corners = pairs[apart], a[apart], b[apart], corners[apart]

    # Moller-Trumbore, with the line's parameter limited to the edge's two ends.
    direction = points[b] - points[a]
    side1 = points[corners[:, 1]] - points[corners[:, 0]]
    side2 = points[corners[:, 2]] - points[corners[:, 0]]
    s = points[a] - points[corners[:, 0]]
    h = np.cross(direction, side2)
    q = np.cross(s, side1)
    det = np.sum(side1 * h, axis=1)
    usable = np.abs(det) > 1e-12
    det = np.where(usable, det, 1.0)
    u = np.sum(s * h, axis=1) / det
    v = np.sum(direction * q, axis=1) / det
    t = np.sum(side2 * q, axis=1) / det
    return pairs[usable & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)]


class SkinParts:
    """Which part of the body owns each edge and triangle, to say what passes through what."""

    def __init__(self, base):
        segments = np.asarray(base["segments"])
        self.triangles, self.edges = delivered_skin(base["faces"], base["verts"])
        self.legs = np.isin(segments, LEG_PARTS)
        fingers = np.setdiff1d(FINE, ["toes"])
        self.groups = np.where(
            np.isin(segments, fingers), "hand", np.where(segments == "toes", "foot", segments)
        )
        # A side belongs to the template's topology, not to where a pose has put the vertex.
        self.side = np.where(np.asarray(base["verts"], float)[:, 0] >= 0, 1, -1)
        self.side[np.isin(segments, MIDDLE_PARTS)] = 0

    def label(self, vertex, other):
        """A pair of parts in a stable order; `<>` marks limbs of opposite sides."""
        first, second = sorted((str(self.groups[vertex]), str(self.groups[other])))
        across = self.side[vertex] * self.side[other] < 0
        return f"{first}{'<>' if across else '|'}{second}"

    def intersections(self, points):
        """How many edges pass through the skin elsewhere, in all and by pair of parts."""
        hits = edges_through_triangles(points, self.edges, self.triangles)
        pairs = {}
        for edge, triangle in hits.tolist():
            label = self.label(self.edges[edge, 0], self.triangles[triangle, 0])
            pairs[label] = pairs.get(label, 0) + 1
        return {"selfIntersections": len(hits), "intersectionsByPair": dict(sorted(pairs.items()))}

    def legs_past_midline(self, points):
        """Leg vertices on the far side of the midline. Legs crossed on purpose count too."""
        return int((points[self.legs, 0] * self.side[self.legs] < 0).sum())
