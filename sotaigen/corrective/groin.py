"""The groin's pose sculptures on a fixed rig: the lower-abdomen / thigh transition as the hips bend.

They follow docs/motion/pelvic-flexion.toml on top of the rest shape (sotaigen/form/groin.py). The
pose grid stores inverse-skinned deltas, so Python, the editor and exported animation share one
result. This is stylised shape control, not a tissue or contact simulation.
"""


import re

import numpy as np

from sotaigen.corrective.drive import bound, weights
from sotaigen.form.field import ramp
from sotaigen.form.groin import DEPTH, coordinates, reference_verts
from sotaigen.mesh import adjacency, triangulate, vertex_normals
from sotaigen.params.spec import GROIN_CONTROLS
from sotaigen.rig.skinning import (
    HIPS_LEVEL,
    pose_skins,
    pose_verts,
    skin_verts,
    torso_length,
)

# 20° apart: the sculpture between them interpolates to within 0.9 mm of a dense grid on every typical
# body, as close as the 10° keys at 90° and 110° that the IPC contact table once needed (docs/parts/groin.md).
# That experimental table, baked on these same keys, lets 85-90° pass through again between them.
KEYS = (0, 20, 40, 60, 80, 100, 120)


def grid(keys):
    """(samples, target names) of a table keyed at `keys` on both legs, every pair but rest."""
    samples = tuple((a, b) for a in keys for b in keys if (a, b) != (0, 0))
    return samples, tuple(f"pelvisL{a}R{b}" for a, b in samples)


SAMPLES, NAMES = grid(KEYS)


def keys_of(targets):
    """The keys a table was made on, read from its targets' names: KEYS for the sculpture's, and the
    IPC contact table's own (sotaigen.contact.groin.KEYS), which also keeps 90 and 110."""
    if not targets:
        return KEYS
    found = {int(n) for name in targets for n in re.fullmatch(r"pelvisL(\d+)R(\d+)", name).groups()}
    return tuple(sorted(found))

# Where a leg's twist and side swing (skinning.swing_twist) have faded the sculpture out, in degrees.
# The sculpture is of a straight hip bend. A twist about the thigh hardly moves the crease, a side swing
# moves where it runs, so the twist is let much further. Set on the twelve typical bodies against the
# 60/60 that Euler angles had: the sitting poses keep their collapsed faces, wider spread brings more to
# a crossed leg (docs/systems/blend.md#腹股沟的淡出).
TWIST_LIMIT = 120
SPREAD_LIMIT = 80
AMOUNTS = GROIN_CONTROLS


def record_amounts(bones, L):
    """Keep a body's own amounts with its rig (Skeleton.groin): its pose targets have to undo the same
    rest shape."""
    bones.groin = {key: L[key] for key in AMOUNTS if key in L}


def landmarks(bones):
    hips = next(b for b in bones if b["name"] == "hips")
    torso = torso_length(bones)
    return {"T": torso, "crotch": hips["head"][2] - HIPS_LEVEL * torso, **bones.groin}


def sample_pose(a, b):
    return {
        "leftUpperLeg": (a, 0, 0),
        "leftLowerLeg": (-a, 0, 0),
        "rightUpperLeg": (b, 0, 0),
        "rightLowerLeg": (-b, 0, 0),
    }


class _Sculptor:
    """What the pose sculpture needs that no pose changes, kept to the skin it can move.

    Built once, it serves every sample. Elsewhere the sculpture is the plain skinned pose, exactly:
    a whole-body pass only adds zeros there. Each sum keeps that pass's order, so the moved skin
    gets the same bits too, and the Laplacian visits each vertex's neighbours in the same order.
    """

    def __init__(self, base, reference, rest, bones, weights, L):
        T, u, level, across, front, ends = coordinates(base, reference, L)
        n = len(rest)
        left = ramp(0.5 + reference[:, 0] / (0.12 * T))
        # Evaluate flow on the static surface with the outer fade removed.
        clear = rest.copy()
        old = np.exp(-0.5 * (across / 0.039) ** 2)
        outer = ramp((u - 0.52) / 0.42)
        new = np.exp(-0.5 * (across / (0.039 * (1 + 0.85 * outer))) ** 2) * (1 - 0.65 * outer)
        clear[:, 1] -= T * front * ends * L.get("groinDepth", DEPTH) * (new - old)
        triangles = triangulate(base["faces"], clear)
        corners = clear[triangles]
        crossed = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        normals = np.zeros_like(clear)
        for corner in range(3):
            np.add.at(normals, triangles[:, corner], crossed)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
        direction = np.column_stack(
            (0.22 * np.tanh(reference[:, 0] / (0.06 * T)), np.zeros(n), np.ones(n))
        )
        tangent = direction - np.sum(direction * normals, axis=1)[:, None] * normals
        belt = np.exp(-0.5 * (across / 0.13) ** 2) * front
        upper = np.exp(-0.5 * (u / 0.74) ** 2 - 0.5 * ((level - 0.28) / 0.14) ** 2)
        upper_gate = ramp((across + 0.025) / 0.12) * front
        lower = np.exp(-0.5 * ((across + 0.10) / 0.105) ** 2) * ends * front
        valley = np.exp(-0.5 * (across / 0.090) ** 2) * ends * front
        brush = ramp((u - 0.48) / 0.28) * ramp((1.18 - u) / 0.23)
        brush_gate = np.exp(-0.5 * ((across - 0.015) / 0.18) ** 2) * front
        source, sink, count = adjacency(n, base["faces"])

        # Each term is one of these fields times the activation, so off their supports it adds
        # exact zeros. The smoothing also reads the neighbours of what it moves, which only skin.
        smoothed = (brush != 0) & (brush_gate != 0)
        moved = (belt != 0) | ((upper != 0) & (upper_gate != 0)) | (lower != 0) | (valley != 0)
        moved |= smoothed
        edges = smoothed[source]
        read = moved.copy()
        read[sink[edges]] = True
        self.ids = np.flatnonzero(read)
        self.moved_ids = m = np.flatnonzero(moved)
        smoothed_ids = np.flatnonzero(smoothed)
        # Positions: of the moved and the smoothed among ids, and of the smoothed among the moved.
        self.moved = np.searchsorted(self.ids, m)
        self.smoothed = np.searchsorted(self.ids, smoothed_ids)
        self.brushed = np.searchsorted(m, smoothed_ids)
        # The Laplacian's edges in their order: from a position among the smoothed to one among ids.
        self.source = np.searchsorted(smoothed_ids, source[edges])
        self.sink = np.searchsorted(self.ids, sink[edges])
        self.count = count[smoothed]
        self.weights, self.T = weights[m], T
        self.names = [bone["name"] for bone in bones]
        self.left, self.right = left[m], (1 - left)[m]
        self.tangent, self.belt = tangent[m], belt[m]
        self.upper, self.upper_gate = upper[m], upper_gate[m]
        self.lower, self.valley = lower[m], valley[m]
        self.brush, self.brush_gate = brush[smoothed], brush_gate[smoothed]

    def blends(self, skins):
        """Each moved vertex's skinning rotation under a pose's bone skins."""
        return np.einsum("vb,bij->vij", self.weights, skins[:, :3, :3])

    def apply(self, result, a, b, skins, rotations):
        """The sculpted pose on self.ids, from the plain skinned pose there, which it changes.

        `rotations` are blends(skins).
        """
        T = self.T
        act = ramp(a / 100.0) * self.left + ramp(b / 100.0) * self.right
        moved = result[self.moved]
        moved += np.einsum(
            "vij,vj->vi", rotations, self.tangent * (0.025 * T * act * self.belt)[:, None]
        )
        upper = self.upper * (self.upper_gate * act)
        lower = self.lower * act
        valley = self.valley * act
        moved += (upper[:, None] * np.array([0.0, -0.060, 0.018]) * T) @ skins[
            self.names.index("hips"), :3, :3
        ].T
        for side, share in (("left", self.left), ("right", self.right)):
            delta = lower[:, None] * share[:, None] * np.array([0.0, -0.030, 0.0]) * T
            moved += delta @ skins[self.names.index(f"{side}UpperLeg"), :3, :3].T
        moved += np.einsum(
            "vij,vj->vi", rotations, valley[:, None] * np.array([0.0, 0.006, 0.0]) * T
        )
        result[self.moved] = moved
        rate = 0.48 * (self.brush * (self.brush_gate * act[self.brushed]))
        size = len(self.smoothed)
        for _ in range(18):
            average = np.column_stack(
                [
                    np.bincount(self.source, weights=result[self.sink, axis], minlength=size)
                    / self.count
                    for axis in range(3)
                ]
            )
            here = result[self.smoothed]
            here += (average - here) * rate[:, None]
            result[self.smoothed] = here
        return result


def sculpture(base, reference, rest, bones, weights, L, a, b):
    pose = sample_pose(a, b)
    result = pose_verts(bones, weights, rest, pose)
    if a == b == 0:
        return result
    sculptor = _Sculptor(base, reference, rest, bones, weights, L)
    skins = pose_skins(bones, pose)
    ids = sculptor.ids
    result[ids] = sculptor.apply(result[ids], a, b, skins, sculptor.blends(skins))
    return result


def sculpted_targets(base, verts, bones, weights, keys=KEYS):
    """The accepted sculpture on its own, before any contact: sotaigen.contact.groin.targets adds that.

    The deltas are zero off the skin the sculpture moves, so only that skin is posed and solved. `keys`
    is the grid to sample it on, KEYS but for the contact table's.
    """
    L = landmarks(bones)
    reference = reference_verts(base, verts, L)
    sculptor = _Sculptor(base, reference, verts, bones, weights, L)
    ids, moved = sculptor.ids, sculptor.moved
    near = weights[ids]
    result = {}
    samples, names = grid(keys)
    for name, (a, b) in zip(names, samples, strict=True):
        pose = sample_pose(a, b)
        skins = pose_skins(bones, pose)
        plain = pose_verts(bones, near, verts[ids], pose)
        blend = sculptor.blends(skins)
        goal = sculptor.apply(plain.copy(), a, b, skins, blend)
        delta = np.zeros_like(verts, dtype=float)
        moves = (goal - plain)[moved]
        delta[sculptor.moved_ids] = np.linalg.solve(blend, moves[:, :, None])[:, :, 0]
        result[name] = delta
    return result


def driver_metadata(keys=KEYS):
    samples, names = grid(keys)
    return {
        "keys": list(keys),
        "samples": [list(p) for p in samples],
        "targets": list(names),
        "twistLimit": TWIST_LIMIT,
        "spreadLimit": SPREAD_LIMIT,
    }


def groups(source="sculpted", keys=KEYS):
    """The table as one correctives group (sotaigen.corrective.drive): both legs' bend, faded by either's
    twist and side swing, whose sizes alone count. The group says its keys, so a file keyed otherwise,
    as the IPC contact table is, needs nothing else to be read."""
    samples, names = grid(keys)
    return [
        {
            "name": "groin",
            "source": source,
            "axes": [{"bone": f"{side}UpperLeg", "read": "swingX", "scale": 1, "keys": list(keys)} for side in ("left", "right")],
            "fades": [
                {"bone": f"{side}UpperLeg", "read": read, "limit": limit}
                for side in ("left", "right")
                for read, limit in (("twist", TWIST_LIMIT), ("swingZ", SPREAD_LIMIT))
            ],
            "samples": [list(p) for p in samples],
            "targets": list(names),
        }
    ]


def pose_weights(pose, keys=KEYS):
    """The legs are read by swing and twist (skinning.swing_twist). The samples bend the hip about X
    alone, a twist-free swing, so they read as their own angles. Only the sizes of the twist and the
    side swing count, so the right leg needs no mirroring."""
    return weights(pose, groups(keys=keys))


def bind_verts(verts, targets, pose):
    return bound(verts, targets, pose_weights(pose, keys_of(targets)))


def normal_targets(base, verts, bones, weights, deltas, normals, *, triangles=None):
    if triangles is None:
        triangles = triangulate(base["faces"], np.asarray(base["verts"]))
    triangles = np.asarray(triangles)
    result = {}
    samples, names = grid(keys_of(deltas))
    for name, (a, b) in zip(names, samples, strict=True):
        pose = sample_pose(a, b)
        moved = np.linalg.norm(deltas[name], axis=1) > 1e-8
        ids = np.unique(triangles[moved[triangles].any(1)])
        # A normal at `ids` reads only the triangles around them, so only their corners are skinned.
        # Kept in their order, the triangles add up at each of `ids` as they do over the whole skin.
        ring = triangles[np.isin(triangles, ids).any(1)]
        need, local = np.unique(ring, return_inverse=True)
        skins = pose_skins(bones, pose)
        posed = skin_verts(skins, weights[need], (verts + deltas[name])[need])
        geometric = vertex_normals(posed, local.reshape(ring.shape))[np.searchsorted(need, ids)]
        blend = np.einsum("vb,bij->vij", weights[ids], skins[:, :3, :3])
        bind = np.linalg.solve(blend, geometric[:, :, None])[:, :, 0]
        bind /= np.maximum(np.linalg.norm(bind, axis=1, keepdims=True), 1e-12)
        delta = np.zeros_like(verts)
        delta[ids] = bind - normals[ids]
        result[name] = delta
    return result
