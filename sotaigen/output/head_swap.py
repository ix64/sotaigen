"""Swap a body's head in its .blend for one made elsewhere, by reading both files and writing a third.

sotaigen swap-head BODY.blend HEAD.blend OUT.blend [--scale S] [--parts-only] [--keep-parts a,b]
    [--report FILE]

What a head file holds, how the join is made and what the checks mean are in docs/parts/head.md#换头工具.
Blender has no scipy, so this runs outside it: sotaigen_blender/head_swap.py reads the two files into arrays, this
module works out where everything goes and which bones move what, and sotaigen_blender/head_swap.py applies that
to the body's file. The body's mesh is changed in place, so its corrective shape keys keep their drivers
and a file that needs no Python to pose stays so. The output is written once it has been reopened with
factory settings and posed; a join that doesn't close or that flips faces writes nothing.
"""

import argparse
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from sotaigen.diagnostics import Diagnostic, as_json
from sotaigen.errors import SotaigenError
from sotaigen.mesh import (
    edge_owners,
    edges_of,
    face_normals,
    triangulate,
    unit_face_normals,
    unit_normals,
)
from sotaigen.metrics import RestMesh, pose_row
from sotaigen.output.blender import NAMES, can_write_blend, run_blender, script
from sotaigen.output.head_interface import eye_place
from sotaigen.output.skin_transfer import edge_lengths, transfer_weights
from sotaigen.rig.poses import POSES

BODY, HEAD_REGION = NAMES["body"], NAMES["headRegion"]
# The bones a head file may place: the head's own pivot stays where the neck needs it.
ATTACHED = ("jaw", "leftEye", "rightEye")
# The turn, the nod, the open mouth and the gaze: the check poses that move what a swap changes.
CHECK_POSES = ("headTurn", "headNod", "jawOpen", "jawTurn", "jawNod", "gaze")
CENTRAL = ("neck", "head", "jaw")
# How far up the new neck the join is faired, as shares of the head's height: at least the first, at
# most the second, and twice the gap between the two necks in between. Our own head has 0.07 of its
# height between the ring and the chin, so the first keeps below the chin.
SEAM_MIN, SEAM_MAX, SEAM_PER_GAP = 0.05, 0.10, 2.0
# The closest-point limit of the weight transfer, as a share of the new head's diagonal (the paper's).
MATCH_SHARE = 0.05
# Rows of the neck below the ring that the weights and the fairing read.
NECK_ROWS = 3
# The fairing's edge lengths settle in a few rounds.
FAIR_ROUNDS = 3
# In the fairing no edge counts as shorter than this share of the ring's mean edge. An opening that lies on
# the ring makes a strip with no height, whose edges would otherwise outweigh all the others and keep it so.
FAIR_SHORTEST = 0.1
# An opening whose vertices are nearer the ring's than this share of its mean edge lies on the ring: however
# a loop's vertices fall on it, they are at most half an edge from one of the ring's.
ON_RING = 0.5
# Two faces this far from coplanar on a join are a fold.
FOLDED_DEG = 90.0


class BrokenSwap(SotaigenError, ValueError):
    """A swap whose result would be broken, so nothing was written; the message says why."""


# Reading ------------------------------------------------------------------------------------------


class FileMesh(NamedTuple):
    name: str
    verts: np.ndarray
    faces: list
    uv: np.ndarray | None
    material: np.ndarray
    normals: np.ndarray | None
    corner_normals: np.ndarray
    groups: list
    group_rows: np.ndarray
    group_columns: np.ndarray
    group_values: np.ndarray
    keys: dict
    matrix: list

    def group(self, name):
        """A group's vertices and values, empty when there is no such group."""
        if name not in self.groups:
            return np.empty(0, int), np.empty(0)
        chosen = self.group_columns == self.groups.index(name)
        return self.group_rows[chosen], self.group_values[chosen]

    def weights(self, names):
        out = np.zeros((len(self.verts), len(names)))
        for j, name in enumerate(names):
            rows, values = self.group(name)
            out[rows, j] = values
        return out


class Dump(NamedTuple):
    meshes: list
    bones: dict
    empties: dict
    info: dict | None
    ignored: list
    texts: int

    def mesh(self, name):
        return next(m for m in self.meshes if m.name == name)


def read_dump(blend, out, keys=False):
    """What sotaigen_blender/head_swap.py reads out of a file."""
    run_blender(script("head_swap"), "read", out, *(["--keys"] if keys else []), blend=blend)
    summary = json.loads(Path(out).with_suffix(".json").read_text())
    data = np.load(Path(out).with_suffix(".npz"))
    meshes = []
    for i, row in enumerate(summary["meshes"]):
        prefix = f"obj{i}_"
        counts, corners = data[prefix + "counts"], data[prefix + "corners"]
        starts = np.concatenate(([0], np.cumsum(counts)))
        faces = [corners[a:b].tolist() for a, b in itertools.pairwise(starts)]
        meshes.append(
            FileMesh(
                row["name"],
                data[prefix + "verts"],
                faces,
                data.get(prefix + "uv", None),
                data[prefix + "material"],
                data.get(prefix + "normals", None),
                data[prefix + "cornerNormals"],
                row["groups"],
                data[prefix + "groupRows"],
                data[prefix + "groupColumns"],
                data[prefix + "groupValues"],
                {name: data[f"{prefix}key{j}"] for j, name in enumerate(row["keys"]) if f"{prefix}key{j}" in data},
                row["matrixWorld"],
            )
        )
    return Dump(meshes, summary["bones"], summary["empties"], summary["info"], summary["ignored"], summary["texts"])


# The head file ------------------------------------------------------------------------------------


def islands_of(mesh):
    """Which connected piece each vertex of a mesh is in."""
    n = len(mesh.verts)
    rows, columns = [], []
    for face in mesh.faces:
        rows += face
        columns += face[1:] + face[:1]
    graph = sparse.coo_matrix((np.ones(len(rows)), (rows, columns)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def boundary_loops(faces):
    """Each open boundary as an ordered list of vertices; a boundary that touches itself is refused. It
    says nothing of whose faces they are: find_skin does."""
    half = {(a, b) for face in faces for a, b in zip(face, face[1:] + face[:1], strict=True)}
    following = {}
    for a, b in half:
        if (b, a) in half:
            continue
        if a in following:
            raise BrokenSwap(f"the boundary at vertex {a} passes through it twice")
        following[a] = b
    loops = []
    while following:
        start, row = next(iter(following)), []
        vertex = start
        while vertex in following:
            row.append(vertex)
            vertex = following.pop(vertex)
        if vertex != start:
            raise BrokenSwap(f"the boundary at vertex {start} does not close")
        loops.append(row)
    return loops


class Skin(NamedTuple):
    mesh: FileMesh
    vertices: np.ndarray  # the piece's vertices, in the mesh's numbering
    faces: list  # its faces, by index into `vertices`
    corners: list  # the corner of the mesh each of those faces begins at
    face_rows: np.ndarray  # which faces of the mesh they are
    loop: np.ndarray  # the neck's opening, by index into `vertices`
    other_loops: int
    skipped: list  # the objects with a piece passed over for its boundary, in the file's order


def mesh_pieces(mesh):
    """Each connected piece of a mesh with faces: whether it's marked with headRegion, its size, the
    mesh, which vertices are in which piece, its own label and its faces' rows."""
    labels = islands_of(mesh)
    marked = set(labels[mesh.group(HEAD_REGION)[0]].tolist())
    first = np.array([face[0] for face in mesh.faces], int)
    for label in np.unique(labels[first]) if len(first) else ():
        yield int(label) in marked, int((labels == label).sum()), mesh, labels, int(label), np.flatnonzero(labels[first] == label)


def find_skin(head):
    """The head's skin: the piece marked with the group headRegion, or else the largest piece with an
    opening. Its neck is its lowest opening; any other one (a mouth, the eyes) stays open.

    Only the skin's boundary has to be clean, since only it is joined. A marked piece is the only one
    read, and refused if its boundary touches itself. With no mark that says nothing, as it may be a part
    (generated hair often has one), so such a piece is passed over and stays a part."""
    pieces = [piece for mesh in head.meshes for piece in mesh_pieces(mesh)]
    marked = [piece for piece in pieces if piece[0]]
    found, skipped = [], []
    if marked:
        _, _, mesh, labels, label, face_rows = max(marked, key=lambda piece: piece[1])
        try:
            loops = boundary_loops([mesh.faces[i] for i in face_rows])
        except BrokenSwap as error:
            raise BrokenSwap(f"the piece marked {HEAD_REGION} in {mesh.name}: {error}") from error
        if not loops:
            raise BrokenSwap(f"the piece marked {HEAD_REGION} in {mesh.name} has no opening to join to the neck ring")
    else:
        for _, size, mesh, labels, label, face_rows in pieces:
            try:
                loops = boundary_loops([mesh.faces[i] for i in face_rows])
            except BrokenSwap:
                skipped += [] if mesh.name in skipped else [mesh.name]
                continue
            if loops:
                found.append((size, mesh, labels, label, face_rows, loops))
        if not found:
            passed = f", and passed over pieces of {', '.join(skipped)}, whose boundaries touch themselves" if skipped else ""
            raise BrokenSwap(f"the head has no piece with an opening to join to the neck ring{passed}")
        _, mesh, labels, label, face_rows, loops = max(found, key=lambda row: row[0])
    vertices = np.flatnonzero(labels == label)
    index = np.full(len(mesh.verts), -1)
    index[vertices] = np.arange(len(vertices))
    faces = [index[np.asarray(mesh.faces[i])].tolist() for i in face_rows]
    lowest = min(loops, key=lambda row: mesh.verts[row, 2].mean())
    starts = np.concatenate(([0], np.cumsum([len(face) for face in mesh.faces])))
    return Skin(mesh, vertices, faces, starts[face_rows].tolist(), face_rows, index[np.asarray(lowest)], len(loops) - 1, skipped)


def markers(head):
    """Where the head file puts each attached bone: a bone of that name, or an empty, by place and length."""
    out = {}
    for name in ATTACHED:
        if name in head.bones:
            bone = head.bones[name]
            at, tail = np.array(bone["head"]) * 1000, np.array(bone["tail"]) * 1000
            out[name] = (at, float(np.linalg.norm(tail - at)))
        elif name in head.empties:
            out[name] = (np.array(head.empties[name]) * 1000, None)
    return out


# The join -----------------------------------------------------------------------------------------


def counter_clockwise(points):
    """Whether a loop runs counter-clockwise seen from above."""
    x, y = points[:, 0], points[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) > 0


def weld_offset(ring_points, loop_points):
    """How far to turn the loop's numbering so that it lies closest to the ring's, vertex by vertex."""
    costs = [np.sum((np.roll(loop_points, -k, axis=0) - ring_points) ** 2) for k in range(len(loop_points))]
    return int(np.argmin(costs))


def arc_shares(points):
    """Each vertex's place round a closed loop as a share of its length."""
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    return np.concatenate(([0.0], np.cumsum(lengths)[:-1])) / lengths.sum()


def resampled(points, count):
    """A closed loop at `count` even steps of its length, from its first vertex."""
    shares = np.r_[arc_shares(points), 1.0]
    closed = np.vstack((points, points[:1]))
    wanted = np.arange(count) / count
    return np.column_stack([np.interp(wanted, shares, closed[:, i]) for i in range(3)])


def zip_strip(ring_points, loop_points):
    """Triangles between two closed loops of different counts, both counter-clockwise from above, with
    the ring below: the same pattern as our own head's first row on the ring (sotaigen.form.head.topology).

    A corner is i for the ring's vertex i and len(ring) + j for the loop's vertex j. Returns the
    triangles and, for each vertex of the loop, the ring vertex it was zipped to first.
    """
    count, other = len(ring_points), len(loop_points)
    ring_at, loop_at = np.r_[arc_shares(ring_points), 1.0], np.r_[arc_shares(loop_points), 1.0]
    faces, nearest = [], np.zeros(other, int)
    i = j = 0
    while i < count or j < other:
        if j < other:
            nearest[j] = i % count
        if j == other or (i < count and ring_at[i + 1] <= loop_at[j + 1]):
            faces.append([i % count, (i + 1) % count, count + j % other])
            i += 1
        else:
            faces.append([i % count, count + (j + 1) % other, count + j % other])
            j += 1
    return faces, nearest


def outward(points, faces, centre):
    """Whether polygons face away from a point: the sign of the volume they enclose round it."""
    middles = np.array([points[f].mean(0) for f in faces]) - centre
    return float(np.sum(face_normals(points, faces) * middles)) > 0


def faired(points, faces, free, shortest):
    """The free vertices moved so that the surface bends as evenly as it can, the others held: the
    squared Laplacian, which reads the two rows on each side and so matches their slope.

    Each neighbour counts by the inverse of its edge's length, so that a straight line stays straight
    when its rows are unevenly spaced, as the head's fine rows are next to the neck's coarse ones. The
    lengths are those of the last result, since the first are of a neck that hasn't reached the ring.
    No edge counts as shorter than `shortest`.
    """
    edges = edges_of(faces)
    n = len(points)
    is_free = np.zeros(n, bool)
    is_free[free] = True
    a, b = np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]
    out = points.copy()
    for _ in range(FAIR_ROUNDS):
        closeness = 1 / np.maximum(np.linalg.norm(out[a] - out[b], axis=1), shortest)
        adjacency = sparse.coo_matrix((closeness, (a, b)), shape=(n, n)).tocsr()
        total = np.asarray(adjacency.sum(1)).ravel()
        L = sparse.eye(n, format="csr") - sparse.diags(1 / np.maximum(total, 1e-30)) @ adjacency
        # Only where the whole neighbourhood is in `faces`: the free vertices and those next to them.
        rows = np.flatnonzero(is_free | (np.asarray((adjacency[free] > 0).sum(0)).ravel() > 0))
        block = L[rows]
        energy = (block.T @ block).tocsr()
        held = np.flatnonzero(~is_free)
        out[free] = splu(energy[free][:, free].tocsc()).solve(np.asarray(-(energy[free][:, held] @ out[held])))
    return out


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def largest_crease(points, faces, among):
    """The largest angle in degrees between the two faces of any edge with a vertex in `among`."""
    among = set(np.asarray(among).tolist())
    pairs = [o for e, o in edge_owners(faces).items() if len(o) == 2 and (e[0] in among or e[1] in among)]
    if not pairs:
        return 0.0
    normals = unit_face_normals(points, faces)
    pairs = np.array(pairs)
    dots = np.clip(np.sum(normals[pairs[:, 0]] * normals[pairs[:, 1]], 1), -1, 1)
    return float(np.degrees(np.arccos(dots)).max())


def near_ring(faces, ring, avoid, rows=NECK_ROWS):
    """The vertices within a few edges of the ring, the ring among them, without crossing `avoid`."""
    neighbours = {}
    for face in faces:
        for a, b in zip(face, face[1:] + face[:1], strict=True):
            neighbours.setdefault(a, set()).add(b)
    reached = set(np.asarray(ring).tolist())
    front = set(reached)
    avoid = set(np.asarray(avoid).tolist())
    for _ in range(rows):
        front = {b for a in front for b in neighbours.get(a, ()) if b not in reached and b not in avoid}
        reached |= front
    return np.array(sorted(reached))


def ring_uvs(body, ring, region):
    """Each ring vertex's UV on the body's side of it."""
    out = np.zeros((len(ring), 2), np.float32)
    if body.uv is None:
        return out
    index = {int(v): i for i, v in enumerate(ring)}
    inside = set(np.asarray(region).tolist())
    corner = 0
    for face in body.faces:
        if not inside.intersection(face):
            for k, vertex in enumerate(face):
                if vertex in index:
                    out[index[vertex]] = body.uv[corner + k]
        corner += len(face)
    return out


class Join(NamedTuple):
    positions: np.ndarray  # the final mesh's vertices, the survivors first
    base: int  # how many survived
    survivor: np.ndarray  # each old vertex's place among the survivors, where `keep`
    keep: np.ndarray
    final_ring: np.ndarray
    face_counts: np.ndarray
    face_corners: np.ndarray  # the new faces, in the final mesh's numbering
    face_uv: np.ndarray
    face_material: np.ndarray  # 0 the body's skin, i + 1 the head's slot i
    new_index: np.ndarray  # each of the skin's vertices in the final mesh
    distance: np.ndarray  # how far along the new surface each final vertex is from the join
    seam_mm: float
    gap_mm: float
    welded: bool
    turned_out: bool  # the head's faces pointed in, and were turned
    new_faces: list
    near_faces: list  # the body's faces just below the ring


def mean_edge(points):
    """The mean length of a closed loop's edges."""
    return float(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).mean())


def turned_over_because(ring_points, loop_points, welded):
    """What the distance between the two necks says about a join that came out turned over."""
    if cKDTree(ring_points).query(loop_points)[0].mean() >= ON_RING * mean_edge(ring_points):
        return "the two necks are too far apart"
    reason = "the head's opening lies about on the ring, so distance isn't the cause: check that the head's own neck isn't folded there"
    return reason if welded else f"{reason}, or give the opening {len(ring_points)} vertices, which weld to the ring vertex by vertex"


def join_skin(body, ring, region, head_h, skin, aligned):
    """The new skin in the final numbering, joined to the ring and faired, refused if that flips it."""
    keep = np.ones(len(body.verts), bool)
    keep[region] = False
    if not keep[ring].all():
        raise BrokenSwap("the neck ring is inside the head's region: the file's groups are wrong")
    survivor = np.cumsum(keep) - 1
    base = int(keep.sum())
    final_ring = survivor[ring]
    ring_points = body.verts[ring]
    loop = skin.loop if counter_clockwise(aligned[skin.loop]) else skin.loop[::-1]
    welded = len(loop) == len(ring)
    turn = (
        weld_offset(ring_points, aligned[loop])
        if welded
        else int(np.argmin(np.linalg.norm(aligned[loop] - ring_points[0], axis=1)))
    )
    loop = np.roll(loop, -turn)
    loop_points = aligned[loop]
    gap = float(np.mean(np.linalg.norm(resampled(loop_points, len(ring)) - ring_points, axis=1)))
    seam = float(np.clip(SEAM_PER_GAP * gap, SEAM_MIN * head_h, SEAM_MAX * head_h))

    faces = [list(face) for face in skin.faces]
    corner_rows = [list(range(c, c + len(face))) for c, face in zip(skin.corners, faces, strict=True)]
    turned_out = not outward(aligned, faces, loop_points.mean(0))
    if turned_out:
        faces = [face[::-1] for face in faces]
        corner_rows = [row[::-1] for row in corner_rows]
    reference = unit_face_normals(aligned, faces)

    new_index = np.full(len(aligned), -1)
    if welded:
        new_index[loop] = final_ring
    rest = np.flatnonzero(new_index < 0)
    new_index[rest] = base + np.arange(len(rest))
    own_faces = [new_index[np.asarray(face)].tolist() for face in faces]
    strip_faces, strip_uv = [], []
    if not welded:
        strip, nearest = zip_strip(ring_points, loop_points)
        lookup = np.r_[final_ring, new_index[loop]]
        uvs = ring_uvs(body, ring, region)
        uv_lookup = np.vstack((uvs, uvs[nearest]))
        strip_faces = [lookup[np.asarray(face)].tolist() for face in strip]
        strip_uv = [uv_lookup[np.asarray(face)] for face in strip]
    new_faces = own_faces + strip_faces

    positions = np.zeros((base + len(rest), 3))
    positions[:base] = body.verts[keep]
    positions[new_index[rest]] = aligned[rest]
    # How far along the head's own surface from the join.
    graph = edge_lengths(positions, triangulate(own_faces, positions))
    seeds = final_ring if welded else new_index[loop]
    distance = dijkstra(graph, directed=False, indices=seeds, min_only=True)
    hops = dijkstra(graph, directed=False, indices=seeds, min_only=True, unweighted=True)
    free = new_index[rest]
    free = free[(distance[free] <= seam) | (hops[free] <= 2)]
    surviving = [face for face in body.faces if keep[face].all()]
    near = set(near_ring(surviving, ring, region).tolist())
    near_faces = [survivor[np.asarray(face)].tolist() for face in surviving if near.issuperset(face)]
    positions = faired(positions, near_faces + new_faces, free, FAIR_SHORTEST * mean_edge(ring_points))

    flips = int((np.sum(unit_face_normals(positions, own_faces) * reference, 1) < 0).sum())
    if flips:
        raise BrokenSwap(
            f"joining the head's neck to the ring flips {flips} of its faces: {turned_over_because(ring_points, loop_points, welded)}"
        )
    if strip_faces:
        outwards = np.array([positions[face].mean(0) for face in strip_faces]) - positions[final_ring].mean(0)
        outwards[:, 2] = 0.0
        if (np.sum(unit_face_normals(positions, strip_faces) * outwards, 1) < 0).any():
            raise BrokenSwap(
                f"the strip between the head's neck and the ring is turned inside out: {turned_over_because(ring_points, loop_points, welded)}"
            )

    face_uv = [skin.mesh.uv[row] if skin.mesh.uv is not None else np.zeros((len(row), 2)) for row in corner_rows]
    return Join(
        positions,
        base,
        survivor,
        keep,
        final_ring,
        np.array([len(face) for face in new_faces], np.int32),
        np.concatenate([np.asarray(face) for face in new_faces]).astype(np.int32),
        np.vstack([*face_uv, *strip_uv]).astype(np.float32),
        np.concatenate((skin.mesh.material[skin.face_rows] + 1, np.zeros(len(strip_faces), np.int32))).astype(np.int32),
        new_index,
        distance,
        seam,
        gap,
        welded,
        turned_out,
        new_faces,
        near_faces,
    )


def normal_rows(join, skin):
    """Normals for the final mesh, zero where Blender's own stand. A ring vertex has one smooth normal
    across both surfaces, and the new skin turns from that to its own (or its author's) up the seam's
    height, so that a toon step doesn't land on the join however sharply the two surfaces meet there.
    The neck below keeps Blender's, as our own head's seam does (sotaigen.output.face.blended_normals)."""
    rows = np.zeros_like(join.positions)
    new_triangles = triangulate(join.new_faces, join.positions)
    both = np.vstack((triangulate(join.near_faces, join.positions), new_triangles))
    across = unit_normals(join.positions, both)[join.final_ring]
    new = np.flatnonzero(join.new_index >= join.base)
    finals = join.new_index[new]
    own = unit_normals(join.positions, new_triangles)[finals]
    authored = skin.mesh.normals
    wanted = own if authored is None else authored[skin.vertices[new]]
    share = smoothstep(join.distance[finals] / join.seam_mm)[:, None]
    nearest = cKDTree(join.positions[join.final_ring]).query(join.positions[finals])[1]
    mixed = across[nearest] * (1 - share) + wanted * share
    mixed /= np.maximum(np.linalg.norm(mixed, axis=1, keepdims=True), 1e-9)
    # Past the seam a head without normals of its own keeps Blender's.
    rows[finals] = np.where(share >= 1, wanted if authored is not None else 0.0, mixed)
    return rows


# The plan -----------------------------------------------------------------------------------------


def align(skin, body, ring, crown, scale):
    """The head's uniform scale and shift: its neck's opening on the ring, its top at the body's crown."""
    loop = skin.mesh.verts[skin.vertices[skin.loop]]
    top = float(skin.mesh.verts[skin.vertices, 2].max())
    centre = body.verts[ring].mean(0)
    wanted = float(body.verts[crown, 2] - centre[2])
    if scale is None:
        high = top - float(loop[:, 2].mean())
        if high <= 0:
            raise BrokenSwap("the head's lowest opening is at or above its top: it can't be the neck")
        scale = wanted / high
    return float(scale), centre - scale * loop.mean(0), wanted


def bone_names(body_dump, body, region, ring, head):
    """The deform bones a swap writes: those on the old head and its neck, and those the new head names."""
    deform = [name for name, bone in body_dump.bones.items() if bone["deform"]]
    used = np.zeros(len(body.verts), bool)
    used[near_ring(body.faces, ring, region)] = True
    used[region] = True
    named = {group for mesh in head.meshes for group in mesh.groups}
    return [name for name in deform if name in named or used[body.group(name)[0]].any()]


def fit_matrix(mesh, s, t):
    """A mesh's world matrix, in metres, followed by the head's scale and shift."""
    fit = np.eye(4)
    fit[:3, :3] *= s
    fit[:3, 3] = t / 1000
    return (fit @ np.array(mesh.matrix)).tolist()


def part_rows(head, skin, names, s, t, source, reach):
    """Every piece that isn't the skin, by object: a piece with a bone's vertex group keeps its weights,
    normalised; the rest take theirs from `source` (verts, faces, weights). Returns the plan's rows, its
    arrays, what the report says about them and how much of the transfer found a close match."""
    rows, arrays, reports, pending = [], {}, [], []
    targets, target_faces = [], []
    offset = 0
    for mesh in head.meshes:
        labels = islands_of(mesh)
        keep = np.ones(len(mesh.verts), bool)
        if skin is not None and mesh is skin.mesh:
            keep[skin.vertices] = False
        if not keep.any():
            continue
        members = np.flatnonzero(keep)
        out = mesh.weights(names)
        bound = set(labels[out.sum(1) > 0].tolist())
        for label in bound:
            inside = labels == label
            empty = inside & (out.sum(1) <= 0)
            out[empty] = out[inside].mean(0)
        out = out[members] / np.maximum(out[members].sum(1, keepdims=True), 1e-30)
        unbound = np.array([labels[v] not in bound for v in members])
        if unbound.any():
            local = np.full(len(mesh.verts), -1)
            local[members[unbound]] = offset + np.arange(unbound.sum())
            targets.append(mesh.verts[members[unbound]] * s + t)
            target_faces += [local[np.asarray(f)].tolist() for f in mesh.faces if local[f[0]] >= 0]
            pending.append((len(rows), np.flatnonzero(unbound), offset))
            offset += int(unbound.sum())
        prefix = f"part{len(rows)}_"
        rows.append({"source": mesh.name, "prefix": prefix, "matrix": fit_matrix(mesh, s, t)})
        arrays[prefix + "keep"] = keep
        arrays[prefix + "weights"] = out
        reports.append(
            {
                "object": mesh.name,
                "vertices": len(members),
                "pieces": len(set(labels[members].tolist())),
                "boundByGroups": int((~unbound).sum()),
                "transferred": int(unbound.sum()),
            }
        )
    matched = -1.0
    if targets:
        result = transfer_weights(*source[:2], source[2], np.vstack(targets), target_faces, distance=reach)
        matched = float(result.matched.mean())
        for row, chosen, start in pending:
            arrays[rows[row]["prefix"] + "weights"][chosen] = result.weights[start : start + len(chosen)]
    return rows, arrays, reports, matched


def moved_bones(body_dump, marks, s, t):
    """The attached bones the head file places, in the body's frame, and the jaw's change of length."""
    moves, glide = {}, 1.0
    for name, (at, length) in marks.items():
        old = np.array(body_dump.bones[name]["head"]) * 1000
        old_length = float(np.linalg.norm(np.array(body_dump.bones[name]["tail"]) * 1000 - old))
        new_length = old_length * s if length is None else length * s
        moves[name] = {"head": ((at * s + t) / 1000).tolist(), "length": new_length / 1000}
        if name == "jaw":
            glide = new_length / old_length
    return moves, glide


def source_mesh(body, region, ring):
    """The old head and the neck under it, as rows of the body's vertices and faces among them."""
    rows = np.unique(np.r_[region, near_ring(body.faces, ring, region)])
    index = np.full(len(body.verts), -1)
    index[rows] = np.arange(len(rows))
    return rows, [index[np.asarray(f)].tolist() for f in body.faces if (index[np.asarray(f)] >= 0).all()]


def new_head_info(attachments, join, top, moves, body_dump):
    """rig["sotaigen"]["head"] after the swap: the same fields, the eyes placed by the same rule as before
    it (sotaigen.output.head_interface), on the new head."""
    eye = moves.get("leftEye")
    at = np.array(eye["head"] if eye else body_dump.bones["leftEye"]["head"]) * 1000
    line, across, height = eye_place(join.positions[top], join.positions[join.final_ring], at)
    out = dict(attachments)
    out.update(
        {
            "region": [join.base, len(join.positions) - join.base],
            "neckRing": [int(i) for i in join.final_ring],
            "crownVertex": int(top),
            "eyeLine": line,
            "eyeX": across,
        }
    )
    return out, height


def plan_swap(body_dump, head, *, scale=None, parts_only=False, keep_parts=()):
    """What sotaigen_blender/head_swap.py's apply does, as a JSON-able plan, its arrays and what the report needs."""
    if BODY not in [m.name for m in body_dump.meshes] or not body_dump.info or "head" not in body_dump.info:
        raise BrokenSwap("the body file has no skin called body, or no sotaigen head on its rig")
    if not head.meshes:
        raise BrokenSwap("the head file has no mesh in its view layer")
    body = body_dump.mesh(BODY)
    attachments = body_dump.info["head"]
    region = body.group(HEAD_REGION)[0]
    ring = np.array(attachments["neckRing"])
    if not len(region):
        raise BrokenSwap(f"the body file's skin has no vertex group {HEAD_REGION}: it was written before the head swap")
    missing = [name for name in keep_parts if name not in [m.name for m in body_dump.meshes]]
    if missing:
        raise BrokenSwap(f"--keep-parts names what the body file doesn't have: {', '.join(missing)}")
    remove = [m.name for m in body_dump.meshes if m.name != BODY and m.name not in keep_parts]
    names = bone_names(body_dump, body, region, ring, head)
    marks = markers(head)
    quality = [Diagnostic("head_swap.touch_stale", "warning")]
    unplaced = [name for name in ATTACHED if name not in marks]
    if unplaced:
        quality.append(Diagnostic("head_swap.unplaced", "warning", {"bones": unplaced}))
    source_rows, source_faces = source_mesh(body, region, ring)
    body_weights = body.weights(names)
    context = {"body": body, "ring": ring, "region": region}

    if parts_only:
        s, t = 1.0 if scale is None else float(scale), np.zeros(3)
        points = body.verts[region]
        reach = MATCH_SHARE * float(np.linalg.norm(points.max(0) - points.min(0)))
        source = (body.verts[source_rows], source_faces, body_weights[source_rows])
        rows, arrays, reports, part_matched = part_rows(head, None, names, s, t, source, reach)
        moves, glide = moved_bones(body_dump, marks, s, t)
        swap = {"mode": "parts", "scale": s, "partsMatched": part_matched, "ignored": ", ".join(head.ignored)}
        plan = {
            "mode": "parts",
            "removeParts": remove,
            "skinObject": None,
            "skinKeys": [],
            "appendOrder": sorted({row["source"] for row in rows}),
            "parts": rows,
            "boneNames": names,
            "bones": moves,
            "glideScale": glide,
            "head": attachments,
            "swap": swap,
            "quality": as_json(quality),
        }
        return plan, arrays, {**context, "reports": reports}

    skin = find_skin(head)
    s, t, wanted = align(skin, body, ring, int(attachments["crownVertex"]), scale)
    aligned = skin.mesh.verts[skin.vertices] * s + t
    join = join_skin(body, ring, region, wanted, skin, aligned)

    # The weights: the old head and the neck below it onto the new skin, the ring held as it was. What
    # the head's skin already has from its author is held too, away from the join.
    new_rows = np.arange(join.base, len(join.positions))
    target_rows = np.r_[join.final_ring, new_rows]
    local = np.full(len(join.positions), -1)
    local[target_rows] = np.arange(len(target_rows))
    target_faces = [local[np.asarray(face)].tolist() for face in join.new_faces]
    reach = MATCH_SHARE * float(np.linalg.norm(aligned.max(0) - aligned.min(0)))
    fixed_rows, fixed_weights = [np.arange(len(ring))], [body_weights[ring]]
    authored = skin.mesh.weights(names)[skin.vertices]
    own = np.flatnonzero((join.new_index >= join.base) & (authored.sum(1) > 0))
    own = own[join.distance[join.new_index[own]] > 2 * join.seam_mm]
    if len(own):
        fixed_rows.append(local[join.new_index[own]])
        fixed_weights.append(authored[own] / authored[own].sum(1, keepdims=True))
    result = transfer_weights(
        body.verts[source_rows],
        source_faces,
        body_weights[source_rows],
        join.positions[target_rows],
        target_faces,
        fixed=(np.concatenate(fixed_rows), np.vstack(fixed_weights)),
        distance=reach,
    )
    skin_weights = result.weights[len(ring) :]

    # The other pieces read the new skin and the neck.
    all_weights = np.zeros((len(join.positions), len(names)))
    all_weights[: join.base] = body_weights[join.keep]
    all_weights[new_rows] = skin_weights
    neck = {int(v) for face in join.near_faces for v in face}
    part_source = np.array(sorted(neck | set(new_rows.tolist())))
    part_index = np.full(len(join.positions), -1)
    part_index[part_source] = np.arange(len(part_source))
    faces = [part_index[np.asarray(face)].tolist() for face in [*join.new_faces, *join.near_faces]]
    source = (join.positions[part_source], faces, all_weights[part_source])
    rows, arrays, reports, part_matched = part_rows(head, skin, names, s, t, source, reach)

    moves, glide = moved_bones(body_dump, marks, s, t)
    if skin.skipped:
        quality.append(Diagnostic("head_swap.kept_parts", "warning", {"objects": list(skin.skipped), "group": HEAD_REGION}))
    if skin.other_loops:
        quality.append(Diagnostic("head_swap.openings", "warning", {"count": int(skin.other_loops)}))
    top = int(new_rows[np.argmax(join.positions[new_rows, 2])])
    head_info, height = new_head_info(attachments, join, top, moves, body_dump)
    swap = {
        "mode": "whole",
        "source": skin.mesh.name,
        "scale": s,
        "shiftMm": [float(v) for v in t],
        "neckLoop": len(skin.loop),
        "welded": join.welded,
        "turnedOut": join.turned_out,
        "gapMm": join.gap_mm,
        "seamMm": join.seam_mm,
        "headHeightMm": height,
        "skinMatched": float(result.matched[len(ring) :].mean()),
        "partsMatched": part_matched,
        "ignored": ", ".join(head.ignored),
    }
    arrays.update(
        {
            "removed": region.astype(np.int64),
            "skin_verts": join.positions[new_rows],
            "skin_weights": skin_weights,
            "face_counts": join.face_counts,
            "face_corners": join.face_corners,
            "face_uv": join.face_uv,
            "face_material": join.face_material,
            "normals": normal_rows(join, skin),
        }
    )
    kept = np.flatnonzero(join.new_index >= join.base)
    skin_keys = list(skin.mesh.keys)
    for j, name in enumerate(skin_keys):
        arrays[f"skin_key{j}"] = skin.mesh.keys[name][skin.vertices[kept]] * s
    plan = {
        "mode": "whole",
        "removeParts": remove,
        "skinObject": skin.mesh.name,
        "skinKeys": skin_keys,
        "appendOrder": sorted({skin.mesh.name, *(row["source"] for row in rows)}),
        "parts": rows,
        "boneNames": names,
        "bones": moves,
        "glideScale": glide,
        "head": head_info,
        "swap": swap,
        "quality": as_json(quality),
    }
    return plan, arrays, {**context, "reports": reports}


# The checks ---------------------------------------------------------------------------------------


def seam_measures(mesh, ring):
    """The join on a file's skin, for the old head and the new one alike: the largest crease at the ring,
    the largest angle between a ring vertex's corner normals (zero when it has one smooth normal), and
    the largest between the normals of a ring vertex and of a neighbour off the ring."""
    corners = np.concatenate([np.asarray(f) for f in mesh.faces])
    normals = mesh.corner_normals / np.maximum(np.linalg.norm(mesh.corner_normals, axis=1, keepdims=True), 1e-9)
    summed = np.zeros_like(mesh.verts)
    np.add.at(summed, corners, normals)
    summed /= np.maximum(np.linalg.norm(summed, axis=1, keepdims=True), 1e-9)
    on_ring = np.isin(corners, ring)
    dots = np.sum(normals[on_ring] * summed[corners[on_ring]], 1)
    members = set(np.asarray(ring).tolist())
    pairs = np.array(
        [
            (a, b)
            for face in mesh.faces
            if members.intersection(face)
            for a, b in zip(face, face[1:] + face[:1], strict=True)
            if (a in members) != (b in members)
        ]
    )
    steps = np.sum(summed[pairs[:, 0]] * summed[pairs[:, 1]], 1)
    return {
        "creaseDeg": largest_crease(mesh.verts, mesh.faces, ring),
        "cornerSpreadDeg": float(np.degrees(np.arccos(np.clip(dots, -1, 1))).max()),
        "normalStepDeg": float(np.degrees(np.arccos(np.clip(steps, -1, 1))).max()),
    }


def check_output(path, scratch):
    """Open the written file as a downstream user would and hold it to the swap's promises."""
    written = read_dump(path, scratch / "written")
    mesh = written.mesh(BODY)
    ring = np.array(written.info["head"]["neckRing"])
    owners = edge_owners(mesh.faces)
    members = set(ring.tolist())
    open_edges = [edge for edge, row in owners.items() if len(row) != 2]
    on_ring = [edge for edge in open_edges if edge[0] in members or edge[1] in members]
    if on_ring:
        raise BrokenSwap(f"the join is open: {len(on_ring)} edges on the neck ring don't have two faces")
    out = {"texts": written.texts, "openEdges": len(open_edges), **seam_measures(mesh, ring)}
    if out["creaseDeg"] >= FOLDED_DEG:
        raise BrokenSwap(f"the join folds: {out['creaseDeg']:.0f} degrees between two faces at the neck ring")
    request = scratch / "poses.json"
    request.write_text(
        json.dumps({"poses": {n: POSES[n] for n in CHECK_POSES}, "central": CENTRAL, "limits": [True], "actions": False})
    )
    run_blender(script("pose"), request, scratch / "posed", blend=path)
    posed = np.load(scratch / "posed.npz")
    report = json.loads((scratch / "posed.json").read_text())
    out["autoexecFail"] = report["autoexecFail"]
    out["notSimpleDrivers"] = len(report["notSimple"])
    rest = RestMesh(np.asarray(posed[f"rest:{BODY}"], float), mesh.faces)
    out["poses"] = {name: pose_row(rest, np.asarray(posed[f"on:{name}:{BODY}"], float)) for name in CHECK_POSES}
    return out


def swap_head(body_path, head_path, out_path, *, scale=None, parts_only=False, keep_parts=(), report=None):
    """Write OUT from BODY and HEAD, returning the report; BrokenSwap writes nothing."""
    body_path, head_path, out_path = Path(body_path), Path(head_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="head-swap-", dir=out_path.parent) as folder:
        scratch = Path(folder)
        body = read_dump(body_path, scratch / "body")
        head = read_dump(head_path, scratch / "head", keys=True)
        plan, arrays, context = plan_swap(body, head, scale=scale, parts_only=parts_only, keep_parts=keep_parts)
        (scratch / "plan.json").write_text(json.dumps(plan))
        np.savez(scratch / "plan.npz", **arrays)
        written = scratch / "written.blend"
        run_blender(script("head_swap"), "apply", scratch / "plan", head_path, written, blend=body_path)
        checks = check_output(written, scratch)
        checks["bodyBefore"] = seam_measures(context["body"], context["ring"])
        quality = [
            Diagnostic("head_swap.collapsed", "warning", {"pose": name, "count": row["collapsedFaces"]}).json()
            for name, row in checks["poses"].items()
            if row["collapsedFaces"]
        ]
        measured = {k: checks[k] for k in ("creaseDeg", "cornerSpreadDeg", "normalStepDeg", "openEdges")}
        (scratch / "found.json").write_text(json.dumps({"swap": measured, "quality": quality}))
        run_blender(script("head_swap"), "note", scratch / "found.json", written, blend=written)
        os.replace(written, out_path)
    out = {**plan["swap"], "checks": checks, "parts": context["reports"], "quality": [*plan["quality"], *quality]}
    report = Path(report) if report else out_path.with_suffix(".json")
    report.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def summary_lines(report):
    checks = report["checks"]
    lines = [f"{report['mode']} head, scale {report['scale']:.3f}"]
    before = checks["bodyBefore"]
    if report["mode"] == "whole":
        lines.append(
            f"neck: {report['neckLoop']} vertices on the ring's {'36, welded' if report['welded'] else '36, zipped'}, "
            f"{report['gapMm']:.1f} mm apart, faired over {report['seamMm']:.1f} mm"
        )
        lines.append(
            f"join: crease {checks['creaseDeg']:.1f} deg (old head {before['creaseDeg']:.1f}), "
            f"normal step {checks['normalStepDeg']:.1f} deg (old head {before['normalStepDeg']:.1f}), "
            f"open edges in the skin {checks['openEdges']}"
        )
        lines.append(f"weights: {report['skinMatched']:.0%} of the skin from its closest point")
    for row in report["parts"]:
        lines.append(
            f"{row['object']}: {row['pieces']} pieces, {row['boundByGroups']} vertices by their groups, "
            f"{row['transferred']} transferred"
        )
    collapsed = {name: row["collapsedFaces"] for name, row in checks["poses"].items()}
    lines.append(f"collapsed faces: {collapsed}")
    lines += [f"note: {Diagnostic.from_json(row)}" for row in report["quality"]]
    return lines


def add_arguments(parser):
    """The command's arguments, on `sotaigen swap-head`."""
    parser.add_argument("body", type=Path, help="a body's .blend, as `sotaigen blend` writes it")
    parser.add_argument("head", type=Path, help="the head's .blend: docs/parts/head.md#换头工具")
    parser.add_argument("out", type=Path, help="the file to write")
    parser.add_argument("--scale", type=float, help="the head's scale, rather than fitting its height to the body's")
    parser.add_argument("--parts-only", action="store_true", help="keep this head's skin: only change the face and the hair")
    parser.add_argument("--keep-parts", default="", help="placeholder parts to leave in, as ears,nose")
    parser.add_argument("--report", type=Path, help="where the report goes: by default OUT with .json")


def run(args):
    """The command on parsed arguments: 0 written, 1 refused, 2 a file that isn't there, 3 no Blender."""
    for given in (args.body, args.head):
        if not given.is_file():
            print(f"{given}: no such file", file=sys.stderr)
            return 2
    why = can_write_blend()
    if why:
        print(why, file=sys.stderr)
        return 3
    try:
        report = swap_head(
            args.body,
            args.head,
            args.out,
            scale=args.scale,
            parts_only=args.parts_only,
            keep_parts=tuple(name for name in args.keep_parts.split(",") if name),
            report=args.report,
        )
    except BrokenSwap as error:
        print(f"{args.out}: not written: {error}", file=sys.stderr)
        return 1
    except RuntimeError as error:
        # Blender's own output, which is long: its first line is what went wrong.
        print(f"{args.out}: not written: {str(error).splitlines()[0]}", file=sys.stderr)
        return 1
    print("\n".join(summary_lines(report)))
    print(args.out, args.out.stat().st_size, "bytes")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sotaigen swap-head", description=__doc__.splitlines()[0])
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
