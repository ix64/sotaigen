"""Experimental internal supports for separate anterior/posterior axillary folds.

Rest surface paths locate two round support envelopes below the skin. The chest
end follows the thorax and the arm end the proximal humerus; a chord frame retains
the rest bow. Skin can slide/separate and is only pushed from the outward side.
These are geometric skin/support proxies, not tendon footprints, tissue mechanics,
or measured physiological thicknesses. Generation bakes these supports into pose morphs.
"""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.transform import Rotation

from sotaigen.mesh import edges_of, triangulate, vertex_normals


def fold_handles(base):
    """Each fold's template handles, from its landmarks: chest wall, fold crest, proximal arm surface.

    Posterior skin handles are NOT the actual anterior humeral tendon footprints.
    """
    marks = base["landmarks"]["vertices"]
    return {
        kind: tuple(marks[f"axilla{kind.title()}{end}"] for end in ("Chest", "Fold", "Arm"))
        for kind in ("front", "back")
    }


def moved(matrix, points):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def swing(source, target):
    """Minimal rotation between nonzero chords, including a deterministic half turn."""
    a, b = np.asarray(source, float), np.asarray(target, float)
    if min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-10:
        raise ValueError("Collapsed attachment chord")
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    cross, cosine = np.cross(a, b), np.clip(a @ b, -1., 1.)
    sine = np.linalg.norm(cross)
    if sine < 1e-10:
        if cosine > 0:
            return np.eye(3)
        basis = np.eye(3)[np.argmin(np.abs(a))]
        axis = np.cross(a, basis)
        return Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.pi).as_matrix()
    return Rotation.from_rotvec(cross / sine * np.arctan2(sine, cosine)).as_matrix()


def transport(verts, origin, insertion, origin_skin, insertion_skin):
    """Stretch along the endpoint chord; retain the rest bow and transverse offsets."""
    chord = insertion - origin
    t = np.clip((verts - origin) @ chord / (chord @ chord), 0., 1.)
    a, b = moved(origin_skin, origin), moved(insertion_skin, insertion)
    chest_rotation = origin_skin[:3, :3]
    rotation = swing(chest_rotation @ chord, b - a) @ chest_rotation
    offset = verts - origin - t[:, None] * chord
    return a + t[:, None] * (b - a) + offset @ rotation.T


def shortest_path(predecessors, source, target):
    path = [int(target)]
    while path[-1] != source:
        parent = int(predecessors[path[-1]])
        if parent < 0:
            raise ValueError("Disconnected material fold path")
        path.append(parent)
    return list(reversed(path))


@dataclass
class Ribbon:
    kind: str
    side: str
    handles: np.ndarray
    path: np.ndarray
    support: np.ndarray
    origin_bone: int
    insertion_bone: int
    width: float
    centreline: np.ndarray
    radius: float


def ribbons(base, verts, bones):
    """Bind compact material paths; widths come from fold-to-floor rest separation."""
    raw = np.asarray(base["verts"])
    edges = np.asarray(edges_of(base["faces"]))
    lengths = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
    graph = coo_matrix((np.tile(lengths, 2), (edges.T.ravel(), edges[:, ::-1].T.ravel())),
                       shape=(len(verts), len(verts))).tocsr()
    raw_lengths = np.linalg.norm(raw[edges[:, 0]] - raw[edges[:, 1]], axis=1)
    material_graph = coo_matrix((np.tile(raw_lengths, 2),
                                 (edges.T.ravel(), edges[:, ::-1].T.ravel())),
                                shape=(len(verts), len(verts))).tocsr()
    paths = {}
    held = fold_handles(base)
    for kind, ends in held.items():
        path = []
        for start, end in pairwise(ends):
            _, predecessor = dijkstra(material_graph, indices=start, return_predecessors=True)
            section = shortest_path(predecessor, start, end)
            path.extend(section if not path else section[1:])
        paths[kind] = np.asarray(path)
    names = [b["name"] for b in bones]
    normals = vertex_normals(verts, triangulate(base["faces"], raw))
    result = []
    for side, sign in [("left", 1), ("right", -1)]:
        floor_index = base["landmarks"]["vertices"]["axillaFloor"]
        floor = np.linalg.norm(raw - raw[floor_index] * [sign, 1, 1], axis=1).argmin()
        for kind, original in held.items():
            handles = np.array([np.linalg.norm(raw - raw[i] * [sign, 1, 1], axis=1).argmin()
                                for i in original])
            path = paths[kind]
            if sign < 0:
                path = np.array([np.linalg.norm(raw - raw[i] * [-1, 1, 1], axis=1).argmin()
                                 for i in path])
            distance = dijkstra(graph, indices=path, min_only=True)
            # A fraction of the fold-to-floor span makes separate rims, without treating the
            # two folds as symmetric thicknesses or carrying the whole breast.
            width = np.linalg.norm(verts[handles[1]] - verts[floor]) * .8
            t = np.clip(1 - distance / width, 0, 1)
            support = t * t * (3 - 2 * t)
            support *= (raw[:, 0] * sign > 0)
            radius = width * .45
            stations = np.r_[0., np.cumsum(np.linalg.norm(np.diff(verts[path], axis=0), axis=1))]
            samples = np.linspace(0., stations[-1], 48)
            surface = CubicSpline(stations, verts[path], bc_type="natural")(samples)
            normal = CubicSpline(stations, normals[path], bc_type="natural")(samples)
            normal /= np.linalg.norm(normal, axis=1, keepdims=True)
            centreline = surface - radius * normal
            result.append(Ribbon(kind, side, handles, path, support, names.index("upperChest"),
                                 names.index(side + "UpperArm"), float(width), centreline, radius))
    return result


def closest_curve(points, curve):
    a, span = curve[:-1], np.diff(curve, axis=0)
    t = np.clip(np.einsum("vsi,si->vs", points[:, None, :] - a, span)
                / np.sum(span * span, axis=1), 0, 1)
    closest = a + t[:, :, None] * span
    offset = points[:, None, :] - closest
    index = np.sum(offset * offset, axis=2).argmin(1)
    return offset[np.arange(len(points)), index]


def constrain(verts, posed, skins, bound, normals, kinds=("front", "back")):
    """One-sided internal support; skin may slide and separate from either ribbon.

    Preserve pre-existing clearance at rest. Only the outward side of the support
    can push skin; this is not a replacement for continuous skin/skin collision.
    """
    offset = np.zeros_like(verts)
    for ribbon in bound:
        if ribbon.kind not in kinds:
            continue
        ids = np.flatnonzero(ribbon.support > 0)
        rest_curve = ribbon.centreline
        curve = transport(rest_curve, rest_curve[0], rest_curve[-1],
                          skins[ribbon.origin_bone], skins[ribbon.insertion_bone])
        initial = np.linalg.norm(closest_curve(verts[ids], rest_curve), axis=1)
        radial = closest_curve(posed[ids], curve)
        distance = np.linalg.norm(radial, axis=1)
        radial /= np.maximum(distance[:, None], 1e-12)
        clearance = np.minimum(initial, ribbon.radius)
        deficit = np.maximum(clearance - distance, 0.)
        outward = np.clip(np.sum(radial * normals[ids], axis=1) / .5, 0, 1)
        gain = ribbon.support[ids] * outward * outward * (3 - 2 * outward)
        offset[ids] += radial * (deficit * gain)[:, None]
    return posed + offset
