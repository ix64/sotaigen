"""Default contact for the relaxed hanging arm, from the body's own implicit volumes, in about a second.

docs/DECISIONS.md ("肩腋与胸侧") chose this over a learned correction: the generated body's
implicit volumes move with the bones, and the skinned skin is pushed back out of the other part's
volume where they meet, in the spirit of implicit skinning (Vaillant et al., SIGGRAPH 2013).

* Fields. The calibrated field (sotaigen/form/field.py, built in sotaigen/form/anatomy.py) is folded at the midline, so a
  one-sided arm can't move inside it. It is taken apart instead. The body side B is every group but
  the two arm groups, left in place: the trunk and the thighs don't move in a relaxed hang, and the
  table fades out with hip flexion anyway. B keeps every layer the field has over its groups (the
  abdomen, thigh and glute profiles, the breast and the pectoral plate), in the same order, so it is
  the field itself away from the arm. The arm side A is rebuilt per segment from the same
  primitives and blends: the upper-arm cone and the deltoid read points through the upper arm's
  inverse skinning matrix, the forearm cone and its two masses (and the wrist cut) through the
  forearm's, so each moves rigidly with its bone. The field has no hand; the hand's volume is an
  implicit moving-least-squares surface (Kolluri 2008) of its own rest skin, moved by the hand bone.
* Projection. The skin is posed as the runtime delivers it without this table: with the shoulder's
  correctives (the pelvis's are driven by the hip, which fades this table out). Each vertex keeps the value its own part's field has there (Vaillant's
  per-vertex iso value), so a field that misses the finished skin by a millimetre moves nothing.
  Where the two parts overlap, the contact surface is where (1 - s) * A = s * B, s the arm's share
  of the overlap: arm vertices are projected along their own part's gradient onto the arm side, body
  vertices onto the body side, each a gap away. The hand is the exception: a finger is thinner than
  the thigh can overlap it, so it leaves along the body's gradient instead of giving way into itself
  to avoid crushing the fingertips in a near-vertical hang. No solver, no collision
  detection, no time steps.
  The moves are smoothed over each part's skin while every contact vertex is held at least as deep
  as its projection, so the side flattens and the area around gives way gently without a kink where
  the contact ends; everything fades with the same supports as the IPC table, and the body side
  fades out towards the armpit, which is the shoulder table's.
* Targets. The posed move is taken back through each vertex's blended skinning matrix to a bind-space
  delta at the hanging-arm table's own keys (sotaigen/corrective/hang.py), named and driven exactly as that
  table is, so the runtime and the GLB read it the same way. On a mirror-symmetric body the right
  side is the left's mirror image; otherwise it is solved on its own.

Only what a pose can touch is skinned and solved: the arm below the shoulder's top and the trunk and
thigh on that side, about a fifth of the skin.
"""

import copy

import numpy as np
from scipy.spatial import cKDTree

from sotaigen.corrective import axilla, hang
from sotaigen.corrective.crossings import SkinParts
from sotaigen.form.field import CLIP_BLEND, Body, ramp, smax, smin
from sotaigen.mesh import adjacency, opposite_vertices, triangulate, vertex_normals
from sotaigen.rig.skinning import pose_skins, torso_length

# The arm's share of an overlap: 0.5 meets halfway, 1 leaves the body untouched.
SHARE = 0.5
# Each side stops this far short of the contact surface, in mm: the two skins end 2 * GAP apart at
# the keys. The clearance must also cover the eased interpolation between keys: a gap that
# separates sampled poses can still let the skins cross between them. Check both with test_hang_implicit.
GAP = 2.0
# The line search along each vertex's normal: 1 mm marching steps up to MARCH mm, then bisection.
MARCH, BISECT = 30, 8
# The body answers the upper arm only from this share of its length down: see Side.solve.
UPPER_FROM = 0.35
# The body answers from this far below the shoulder joint, fading in over HEIGHT_FADE, in torso shares.
BELOW_SHOULDER, HEIGHT_FADE = 0.12, 0.06
# Central differences for the gradient, in mm.
STEP = 0.25
# Smoothing passes over the moves, each holding the contact at its projected depth.
RELAX = 30
# The hand's moving-least-squares surface: neighbours and kernel width, in mm.
HAND_K, HAND_H = 10, 4.0
ARM_FIELD_GROUPS = ("armBones", "armMasses")
# Arm parts that leave the body along its gradient instead of giving way into themselves.
OUTWARD = ("hand",)


def without_arms(field):
    """(the field without its two arm groups, under the same layers; the calibrated Body inside them).

    A layer (an abdomen, thigh or glute profile, a breast, a pectoral plate) is a Body that reads its
    `reference`; it is copied onto the layer below it with the arms taken out, so what it measured
    on the whole body stays as it was.
    """
    if "reference" not in vars(field):
        kept = [g for g in field.groups if g.get("name") not in ARM_FIELD_GROUPS]
        return Body(kept, field.clips, field.carve), field
    inner, core = without_arms(field.reference)
    layer = copy.copy(field)
    layer.reference = inner
    return layer, core


def arm_parts(body):
    """The arm's primitives and blends, by segment, from the calibrated field's two arm groups."""
    bones = next(g for g in body.groups if g.get("name") == "armBones")
    masses = next(g for g in body.groups if g.get("name") == "armMasses")
    upper_cone, fore_cone = bones["prims"]
    deltoid, *forearm = masses["prims"]
    return {
        "upper": (upper_cone, [deltoid]),
        "fore": (fore_cone, forearm),
        "k_bones": bones["k"],
        "k_masses": masses["k"],
        "join": masses["join"],
        "clips": bones.get("clips", ()),
    }


class HandSurface:
    """Kolluri's IMLS surface of the hand's rest skin: signed, smooth, negative inside."""

    def __init__(self, points, normals):
        self.points, self.normals = points, normals
        self.tree = cKDTree(points)

    def __call__(self, q):
        dist, idx = self.tree.query(q, HAND_K)
        w = np.exp(-((dist / HAND_H) ** 2))
        offsets = q[:, None, :] - self.points[idx]
        along = np.einsum("nkj,nkj->nk", offsets, self.normals[idx])
        total = w.sum(1)
        near = total > 1e-6
        out = dist[:, 0].copy()
        out[near] = (w * along).sum(1)[near] / total[near]
        return out


class PosedFields:
    """B, the body without its arms, left in place; A, one arm whose segments follow its bones."""

    def __init__(self, field, bones, side, hand):
        self.body, core = without_arms(field)
        self.parts = arm_parts(core)
        names = [b["name"] for b in bones]
        self.index = {
            seg: names.index(side + bone)
            for seg, bone in (("upper", "UpperArm"), ("fore", "LowerArm"), ("hand", "Hand"))
        }
        # The arm's primitives are the left arm's; the right arm reads them in its mirror image.
        self.flip = hang.FLIP if side == "right" else np.ones(3)
        self.hand = hand
        self.inverse = {seg: np.eye(4) for seg in self.index}

    def pose(self, skins):
        self.inverse = {seg: np.linalg.inv(skins[i]) for seg, i in self.index.items()}

    def local(self, seg, x):
        m = self.inverse[seg]
        return x @ m[:3, :3].T + m[:3, 3]

    def B(self, x):
        return self.body(x)

    def A(self, x):
        p = self.parts
        qu = self.local("upper", x) * self.flip
        qf = self.local("fore", x) * self.flip
        bones = smin(p["upper"][0](qu), p["fore"][0](qf), p["k_bones"])
        for point, normal in p["clips"]:
            bones = smax(bones, (qf - point) @ normal, CLIP_BLEND)
        masses = p["upper"][1][0](qu)
        for prim in p["fore"][1]:
            masses = smin(masses, prim(qf), p["k_masses"])
        arm = smin(bones, masses, p["join"])
        return np.minimum(arm, self.hand(self.local("hand", x)))


def gradient(f, x, step=STEP):
    """Central differences, all six shifts in one call of the field."""
    shifts = np.concatenate([np.eye(3) * step, -np.eye(3) * step])
    values = f((x[None] + shifts[:, None]).reshape(-1, 3)).reshape(6, len(x))
    return ((values[:3] - values[3:]) / (2 * step)).T


class Side:
    """One side's arm against its trunk and thigh, solved one pose at a time."""

    def __init__(self, base, verts, bones, weights, field, side, *, skin=None, share=SHARE):
        """Everything that doesn't need the shoulder's correctives: see use()."""
        self.bones, self.side, self.share = bones, side, share
        self.driver, self.targets = None, {}
        self.T = torso_length(bones)
        self.names = [b["name"] for b in bones]
        skin = skin or SkinParts(base)
        raw = np.asarray(base["verts"])
        here = raw[:, 0] > 0 if side == "left" else raw[:, 0] < 0
        # Who answers, as the IPC table has it: the arm from just below the shoulder's top, and the
        # trunk and thigh near where the arm hangs.
        arm = bones[self.names.index(side + "UpperArm")]
        head, tail = np.asarray(arm["head"], float), np.asarray(arm["tail"], float)
        along = (verts - head) @ (tail - head) / np.sum((tail - head) ** 2)
        limb = np.isin(skin.groups, hang.ARM_GROUPS) & here
        arm_support = limb * np.where(skin.groups == "upperArm", ramp((along - 0.15) / 0.2), 1.0)
        arm_ids = np.flatnonzero(arm_support > 0)
        body_ids = np.flatnonzero(np.isin(skin.groups, ("trunk", "upperLeg")) & here)
        # Everything below works on these vertices only, arm first: `work` indexes them.
        self.work = np.concatenate([arm_ids, body_ids])
        self.n_arm = len(arm_ids)
        self.arm_support = arm_support[arm_ids]
        # Fingers are thinner than the overlap can be: see solve().
        self.hand_rows = np.isin(skin.groups[arm_ids], OUTWARD)
        self.verts = np.asarray(verts, float)[self.work]
        self.weights = np.asarray(weights)[self.work]
        # Bones that move none of these vertices are left out of the blend.
        self.used = np.flatnonzero(np.abs(self.weights).sum(0) > 0)
        self.weights = self.weights[:, self.used]
        # Above the armpit the trunk meets the arm where the two blend; that is the shoulder's table.
        self.cut = head[2] - BELOW_SHOULDER * self.T
        self.below = ramp((self.cut - self.verts[self.n_arm :, 2]) / (HEIGHT_FADE * self.T))
        # The skin's edges between work vertices of the same part, for the smoothing.
        rows, cols, _ = adjacency(len(verts), base["faces"])
        position = np.full(len(verts), -1)
        position[self.work] = np.arange(len(self.work))
        rows, cols = position[rows], position[cols]
        keep = (rows >= 0) & (cols >= 0)
        rows, cols = rows[keep], cols[keep]
        same = (rows < self.n_arm) == (cols < self.n_arm)
        self.rows, self.cols = rows[same], cols[same]
        hand = here & (skin.groups == "hand")
        triangles = triangulate(base["faces"], raw)
        normals = vertex_normals(verts, triangles)
        self.fields = PosedFields(field, bones, side, HandSurface(verts[hand], normals[hand]))
        # Each vertex's own iso value at rest, and how deep it already sits in the other part there:
        # the armpit, where the two parts blend, overlaps at rest and is not a contact. The shoulder's
        # correctives are zero at rest.
        rest, _ = self.skinned({})
        self.rest_B = self.fields.B(rest[: self.n_arm])
        self.rest_A = self.fields.A(rest[self.n_arm :])

    def use(self, shoulder):
        """Sit on the shoulder's correctives, as the runtime delivers the arm without this table."""
        self.driver = axilla.driver_metadata(shoulder)
        self.targets = {
            name: delta[self.work] for name, delta in shoulder.items() if np.any(delta[self.work])
        }

    def skinned(self, pose):
        """(the work vertices as delivered in this pose, their blended skinning matrices)."""
        bound = self.verts.copy()
        if self.targets:
            for name, weight in axilla.pose_weights(pose, self.driver).items():
                if weight and name in self.targets:
                    bound += weight * self.targets[name]
        skins = pose_skins(self.bones, pose)
        self.fields.pose(skins)
        used = skins[self.used, :3].reshape(len(self.used), 12)
        blend = (self.weights @ used).reshape(-1, 3, 4)
        return (blend[:, :, :3] @ bound[:, :, None])[:, :, 0] + blend[:, :, 3], (skins, blend)

    def solve(self, forward, adduction):
        """(skin vertex ids, bind-space deltas there) for one pose of this side's arm."""
        pose = hang.arm_pose(self.side, forward, adduction)
        X, (skins, blend) = self.skinned(pose)
        n = self.n_arm
        # The trunk and thigh near the hanging arm, faded like the IPC table's support.
        reach = np.full(len(X) - n, np.inf)
        for bone in ("UpperArm", "LowerArm", "Hand"):
            i = self.names.index(self.side + bone)
            ends = [
                skins[i][:3, :3] @ np.asarray(self.bones[i][end], float) + skins[i][:3, 3]
                for end in ("head", "tail")
            ]
            if bone == "UpperArm":
                # The shoulder's top and the armpit's apex are where the two parts blend at rest:
                # the shoulder table's business, and no contact here.
                ends[0] = ends[0] + UPPER_FROM * (ends[1] - ends[0])
            reach = np.minimum(reach, hang.segment_distance(X[n:], *ends))
        body_support = ramp((hang.REACH - reach / self.T) / hang.FADE) * self.below
        near = np.flatnonzero(body_support > 0)
        s = self.share
        A, B = self.fields.A, self.fields.B
        # The arm answers only where it has come down to the trunk's lower half.
        low = np.flatnonzero(X[:n, 2] < self.cut)
        # With the arm taking all of the overlap the body answers nothing, but still gives way.
        pushed = near if s < 1 else near[:0]
        # Both parts in one pass, arm first: the fields cost per call more than per point.
        rows = np.concatenate([low, n + pushed])
        arm = np.arange(len(rows)) < len(low)
        x = X[rows]

        def level(a, b, arm, own, other):
            """Positive on the wrong side of the contact; each part ends GAP short of it."""
            arm_side = (1 - s) * (a - own) - s * (b - other)
            body_side = s * (b - own) - (1 - s) * (a - other)
            return np.where(arm, arm_side, body_side) + GAP

        def level_at(y, arm, own, other):
            return level(A(y), B(y), arm, own, other)

        a, b = A(x), B(x)
        own = np.where(arm, a, b)
        other = np.minimum(np.concatenate([self.rest_B[low], self.rest_A[pushed]]), 0.0)
        hit = level(a, b, arm, own, other) > 0
        rows, arm, y, own, other = rows[hit], arm[hit], x[hit], own[hit], other[hit]
        if not len(y):
            return np.zeros(0, int), np.zeros((0, 3))
        # Along the vertex's own inward normal, never the other part's gradient: a thin forearm
        # deep in the waist has its axis inside, and past the axis its gradient points the wrong way.
        # A finger is thinner than the thigh can overlap it: giving way into itself along its own
        # normals, which turn round its tip, crushes the tip. The hand leaves the body instead,
        # along the body's outward gradient, the same way for every finger.
        out = arm & np.concatenate([self.hand_rows[low], np.zeros(len(pushed), bool)])[hit]
        g = np.empty_like(y)
        for mask, field, sign in ((arm & ~out, A, 1.0), (~arm, B, 1.0), (out, B, -1.0)):
            if mask.any():
                g[mask] = sign * gradient(field, y[mask])
        g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
        # March to the first root, then bisect it.
        lo, hi = np.zeros(len(y)), np.full(len(y), np.nan)
        for step in np.arange(1.0, MARCH + 1.0):
            open_ = np.flatnonzero(np.isnan(hi))
            if not len(open_):
                break
            done = level_at(y[open_] - step * g[open_], arm[open_], own[open_], other[open_]) <= 0
            hi[open_[done]] = step
            lo[open_[~done]] = step
        # No way out within reach: a blend between the parts, not a contact. Leave it.
        lost = np.isnan(hi)
        hi = np.where(lost, lo, hi)
        for _ in range(BISECT):
            middle = (lo + hi) / 2
            over = level_at(y - middle[:, None] * g, arm, own, other) > 0
            lo = np.where(over, middle, lo)
            hi = np.where(over, hi, middle)
        pinned = rows[~lost]
        if not len(pinned):
            return np.zeros(0, int), np.zeros((0, 3))
        D = np.zeros_like(X)
        D[pinned] = -hi[~lost, None] * g[~lost]
        # Pin what was projected; relax the move into the rings around it, within each part.
        free = np.zeros(len(X), bool)
        free[:n] = True
        free[n + near] = True
        keep = free[self.cols]
        rows, cols = self.rows[keep], self.cols[keep]
        count = np.bincount(rows, minlength=len(X)).astype(float)
        active = free & (count > 0)
        # Smooth every move, the contact's too, then hold each contact vertex at least as deep as
        # its projection asked. Holding the contact fixed instead (a harmonic fill) leaves a kink
        # where the contact ends, which shows as a notch in the forearm's outline.
        depth = np.linalg.norm(D[pinned], axis=1)
        direction = D[pinned] / np.maximum(depth, 1e-9)[:, None]
        for _ in range(RELAX):
            total = np.column_stack([np.bincount(rows, D[cols, k], len(X)) for k in range(3)])
            mean = total / np.maximum(count, 1)[:, None]
            D[active] = 0.5 * D[active] + 0.5 * mean[active]
            short = depth - np.einsum("ij,ij->i", D[pinned], direction)
            D[pinned] += np.maximum(short, 0.0)[:, None] * direction
        # Everything fades with the support, the pinned contact too: towards the armpit the
        # correction gives way to the shoulder's table instead of crushing the fold there.
        support = np.zeros(len(X))
        support[:n] = self.arm_support
        support[n + near] = body_support[near]
        D *= support[:, None]
        moved = np.flatnonzero(np.abs(D).sum(1) > 0)
        delta = np.linalg.solve(blend[moved, :, :3], D[moved][:, :, None])[:, :, 0]
        return self.work[moved], delta


class Table:
    """The hanging arm's default table for one body: set up first, then solved on the shoulder's."""

    def __init__(self, base, verts, bones, weights, field, *, share=SHARE):
        self.keys = hang.adduction_keys(bones)
        self.verts = np.asarray(verts, float)
        self.opposite = opposite_vertices(self.verts) if self.keys else None
        self.sides = {}
        if self.keys:
            skin = SkinParts(base)
            # On a mirror-symmetric body the right side is the left's mirror image.
            for side in ("left",) if self.opposite is not None else hang.SIDES:
                self.sides[side] = Side(
                    base, self.verts, bones, weights, field, side, skin=skin, share=share
                )

    def metadata(self):
        return hang.driver_metadata(self.keys) if self.keys else None

    def solve(self, shoulder, forwards=None):
        """{name: whole-mesh bind deltas}, both sides, at every forward key or only at `forwards`.

        The poses don't depend on each other, so the table can be solved in parts, in any order.
        """
        if not self.keys:
            return {}
        rows = hang.sample_names(self.keys)
        out = {}
        for side, solver in self.sides.items():
            solver.use(shoulder)
            for forward, adduction, name in rows[side]:
                if forwards is not None and forward not in forwards:
                    continue
                ids, delta = solver.solve(forward, adduction)
                whole = np.zeros_like(self.verts)
                whole[ids] = delta
                out[name] = whole
        if self.opposite is not None:
            for (_, _, name), (_, _, mirrored) in zip(rows["left"], rows["right"], strict=True):
                if name in out:
                    right = np.zeros_like(self.verts)
                    right[self.opposite] = out[name] * hang.FLIP
                    out[mirrored] = right
        # In the table's own order, whichever part was asked for.
        order = [name for side in hang.SIDES for _, _, name in rows[side]]
        return {name: out[name] for name in order if name in out}


def targets(base, verts, bones, weights, field, shoulder, *, share=SHARE):
    """({name: whole-mesh bind deltas}, driver metadata), or ({}, None) when the arm hangs within reach.

    `shoulder` is the shoulder's correctives, which the table sits on. The pelvis's are driven by the
    hip, and this table fades out as the hip flexes.
    """
    table = Table(base, verts, bones, weights, field, share=share)
    return table.solve(shoulder), table.metadata()
