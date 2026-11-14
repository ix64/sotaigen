"""Generation-time contact between the lower abdomen and the thigh root.

The accepted sculpture in sotaigen/corrective/groin.py stays the target. IPC moves the skin only where that target
would pass through itself, and each sample stores just this difference. It is a geometric fit, not
tissue mechanics. A body whose bake fails is still generated on the sculpture alone, and failure()
says why, as for the shoulder.

The audit reads the table the way the browser will: the same eased weights, then skinning. It covers
hip flexion on a fixed pelvis, each leg at any angle up to the last key. Twist, spread and a leaning
trunk are outside it, and the driver only fades the correction there.
"""

import itertools
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from sotaigen.contact import cache
from sotaigen.contact.solver import ContactPatch, vertex_filter
from sotaigen.corrective import axilla, groin
from sotaigen.corrective.crossings import SkinParts, edges_through_triangles
from sotaigen.form import groin as groin_rest
from sotaigen.form.field import ramp
from sotaigen.mesh import face_areas
from sotaigen.paths import cache_dir
from sotaigen.rig.skinning import pose_skins, pose_verts
from sotaigen.sources import bake_key

# Shared by every worktree, like the shoulder's: a bake takes minutes, and its key covers all that decides it.
CACHE = cache_dir() / "groin-contact"
# What decides an entry or a failure record: bake() solves, audits and writes it.
BAKED_BY = ("sotaigen/contact/groin.py:bake",)
# The table's own keys: the sculpture's every 20 degrees, and 90 and 110 too. With 20 degrees between 100
# and 120 the belly passed through the thigh root at 105 on 13 of the 28 bodies, and between 80 and 100
# the patch folded flat; with these, female-rounded and male-heavyarm passed. The sculpture alone stays
# on its own keys. A file reads the keys from the group, so it needs nothing else.
KEYS = (0, 20, 40, 60, 80, 90, 100, 110, 120)
SAMPLES, NAMES = groin.grid(KEYS)
# How far around the crease the skin may answer, in shares of the torso, and how it fades to fixed.
REACH, FADE = 0.45, 0.20
# How high above the crotch, in shares of the torso, the patch may reach, and how it fades out there. The
# belly meets the thigh no higher than about 0.56 at 120 degrees on every typical body; the loops round the
# armpit start at 0.77, and inside the barrier at rest they would be pushed from the first degree.
TOP, TOP_FADE = 0.70, 0.10
# Below this no skin has come near yet, so the first step of a path can be long.
FIRST_STEP, STEP = 30.0, 5.0
AUDIT_STEP = 5
WORKERS = 4
# Where the barrier starts, in shares of the torso. Wider than the shoulder's: a key sits close to the
# skin it touches, and between keys the eased morphs stray from the arc the skin really follows.
GAP = 0.008
PATCH_COLLAPSED = 0.15
MIRROR_MM = 0.01
WHOLE_EVERY = 2
WHOLE_PAIRS = ("trunk|upperLeg", "trunk|trunk")
# When --jobs already runs bodies side by side, each takes fewer workers.
WORKERS_UNDER_JOBS = 2
# The midline pocket between the legs, in torso lengths: half width, front and back, above the crotch.
# The table's keys are 20 degrees apart up to 80, and with uneven legs the skin here still crosses itself
# between them. A few edges may, as the audit reports; any more fail the table.
POCKET = (0.045, -0.17, 0.0, 0.06)
POCKET_EDGES = 15


class GroinSampler:
    """Walk a patch around the crease from rest to any left and right hip flexion."""

    def __init__(self, base, verts, bones, weights):
        self.base, self.verts, self.bones, self.weights = base, verts, bones, weights
        self.L = groin.landmarks(bones)
        self.T = self.L["T"]
        self.reference = groin_rest.reference_verts(base, verts, self.L)
        raw = np.asarray(base["verts"])
        _, u, level, across, _, _ = groin_rest.coordinates(base, self.reference, self.L)
        forward = ramp((-self.reference[:, 1] / self.T + 0.12) / 0.12)
        band = ramp((REACH - np.abs(across)) / FADE) * ramp((1.35 - u) / 0.2) * ramp((TOP - level) / TOP_FADE)
        local = np.isin(base["segments"], ("trunk", "upperLeg"))
        knee, floor = self.other_folds()
        # The knee's crease follows the sculpture. The crotch floor moves with the skin around it. Neither
        # takes part in contact: held still inside the barrier, they would push everything near them away.
        support = band * forward * local * (1 - knee)
        self.ids, self.triangles, polygons = axilla._patch_topology(base["faces"], raw, support)
        self.solver = ContactPatch(
            verts[self.ids] / self.T,
            self.triangles,
            polygons,
            support[self.ids],
            intrinsic_area=True,
            moving_boundary=True,
            gap=GAP,
        )
        self.behind_knee, self.in_slot = knee > 0.5, floor > 0.5
        # One thread keeps the bake reproducible: the order ipctk finds contacts in sets the order the
        # barrier sums them. The bake already runs samples in parallel processes.
        self.solver.ipc.set_num_threads(1)
        # The floor only ignores itself: the belly can still come down onto it.
        self.solver.mesh.can_collide = vertex_filter(self.behind_knee[self.ids], self.in_slot[self.ids])
        self.solver.collisions.build(self.solver.mesh, self.solver.x, GAP)

    def other_folds(self):
        """The places that fold on their own: (behind the knee, inside the crotch), 0 to 1.

        A short thigh brings the knee's crease into the patch, though the front of that thigh has to
        stay in, since that is what meets the belly. The floor of the crotch lies inside the barrier
        when standing, so contact there would answer from the first degree, with nothing pressing.
        """
        knee = next(bone for bone in self.bones if bone["name"] == "leftLowerLeg")["head"][2]
        points = self.reference / self.T
        near_knee = ramp(1 - (points[:, 2] - knee / self.T) / 0.20)
        behind = ramp((points[:, 1] + 0.02) / 0.04)
        level = points[:, 2] - self.L["crotch"] / self.T
        floor = (
            ramp((0.11 - np.abs(points[:, 0])) / 0.04)
            * ramp((0.035 - level) / 0.025)
            * ramp((points[:, 1] + 0.13) / 0.03)
        )
        # The two sides stay closest along the midline, a little further forward and up than the floor.
        midline = (
            ramp((0.04 - np.abs(points[:, 0])) / 0.015)
            * ramp((0.07 - level) / 0.02)
            * ramp((points[:, 1] + 0.195) / 0.03)
        )
        return near_knee * behind, np.maximum(floor, midline)

    def pocket(self):
        half_width, front, back, top = POCKET
        points = self.reference / self.T
        level = points[:, 2] - self.L["crotch"] / self.T
        return (
            (np.abs(points[:, 0]) < half_width)
            & (points[:, 1] > front)
            & (points[:, 1] < back)
            & (level < top)
            & (level > -top)
        )

    def blends(self, a, b):
        skins = pose_skins(self.bones, groin.sample_pose(a, b))
        return np.einsum("vb,bij->vij", self.weights[self.ids], skins[:, :3])

    def goal(self, a, b):
        whole = groin.sculpture(
            self.base, self.reference, self.verts, self.bones, self.weights, self.L, a, b
        )
        blend = self.blends(a, b)
        u, _, vt = np.linalg.svd(blend[:, :, :3])
        u[:, :, -1] *= np.linalg.det(u @ vt)[:, None]
        return whole, whole[self.ids] / self.T, u @ vt, blend

    def advance(self, previous, target, depth=0):
        """One step of a path, halved until it converges. Returns the sculpture and the solved patch."""
        start = self.solver.x.copy()
        whole, goal, rotation, blend = self.goal(*target)
        old = self.blends(*previous)
        material = np.linalg.solve(old[:, :, :3], (start * self.T - old[:, :, 3])[:, :, None])[:, :, 0]
        predictor = (np.einsum("vij,vj->vi", blend[:, :, :3], material) + blend[:, :, 3]) / self.T
        try:
            result, stat = self.solver.advance(goal, rotation, initial=predictor)
        except RuntimeError:
            self.solver.x = start
            if depth == 5:
                raise
            middle = ((previous[0] + target[0]) / 2, (previous[1] + target[1]) / 2)
            self.advance(previous, middle, depth + 1)
            return self.advance(middle, target, depth + 1)
        return whole, result, stat

    def sample(self, a, b):
        """The bind-space difference from the sculpture at (a, b), always along the path from rest.

        Contact depends on the path, so every sample takes the same kind of path whatever was solved
        before it, and the table doesn't depend on the order it was filled in.
        """
        self.solver.x = self.verts[self.ids] / self.T
        reach = max(a, b)
        stops = np.arange(min(FIRST_STEP, reach), reach, STEP)
        previous = (0.0, 0.0)
        for stop in [*stops, reach]:
            fraction = stop / reach
            target = (a * fraction, b * fraction)
            whole, solved, _ = self.advance(previous, target)
            previous = target
        blend = self.blends(a, b)
        difference = solved * self.T - whole[self.ids]
        return np.linalg.solve(blend[:, :, :3], difference[:, :, None])[:, :, 0]


_worker = None


def _start_worker(base, verts, bones, weights):
    global _worker
    _worker = GroinSampler(base, verts, bones, weights)


def _solve_sample(sample):
    return _worker.sample(*sample)


def mirror_of(base, verts, ids):
    """Each patch vertex's twin across the midline, or None if this body's patch isn't symmetric."""
    from scipy.spatial import cKDTree

    raw = np.asarray(base["verts"], float)[ids]
    distance, twin = cKDTree(raw).query(raw * [-1, 1, 1])
    symmetric = (
        distance.max() < 1e-6
        and np.array_equal(twin[twin], np.arange(len(twin)))
        and np.abs(verts[ids][twin] * [-1, 1, 1] - verts[ids]).max() < MIRROR_MM
    )
    return twin if symmetric else None


def solve_table(base, verts, bones, weights, ids):
    """Every sample's difference, as {name: deltas on the patch}.

    A symmetric body solves each left and right pair once and mirrors the other, which the audit then
    reads like any other sample.
    """
    twin = mirror_of(base, verts, ids)
    wanted = [s for s in SAMPLES if twin is None or s[0] >= s[1]]
    context = multiprocessing.get_context("spawn")
    initargs = (base, np.array(verts), bones, np.array(weights))
    workers = WORKERS if multiprocessing.parent_process() is None else WORKERS_UNDER_JOBS
    with ProcessPoolExecutor(workers, context, _start_worker, initargs) as pool:
        solved = dict(zip(wanted, pool.map(_solve_sample, wanted), strict=True))
    return {
        name: solved[(a, b)] if (a, b) in solved else solved[(b, a)][twin] * [-1, 1, 1]
        for name, (a, b) in zip(NAMES, SAMPLES, strict=True)
    }


def audit(base, verts, bones, weights, ids, table):
    """Read the table as delivered, across all left and right angles and between neighbouring poses."""
    import ipctk

    sampler = GroinSampler(base, verts, bones, weights)
    if not np.array_equal(sampler.ids, ids):
        raise ValueError("Pelvic contact table was solved on a different patch")
    sculpted = {name: value[ids] for name, value in groin.sculpted_targets(base, verts, bones, weights, KEYS).items()}
    # The knee's own crease follows the sculpture, so its folded faces aren't this table's to answer for.
    local = [f for f in base["faces"] if np.isin(f, ids).all() and not sampler.behind_knee[f].any()]
    faces = [np.searchsorted(ids, f).tolist() for f in local]
    original = np.maximum(face_areas(verts[ids], faces), 1e-6)

    def posed(a, b):
        delta = np.zeros_like(verts[ids])
        for name, weight in groin.pose_weights(groin.sample_pose(a, b), KEYS).items():
            if weight:
                delta += weight * (sculpted[name] + table[name])
        blend = sampler.blends(a, b)
        bound = verts[ids] + delta
        return (np.einsum("vij,vj->vi", blend[:, :, :3], bound) + blend[:, :, 3]) / sampler.T

    pocket = sampler.pocket()
    mesh = sampler.solver.mesh
    mesh.can_collide = vertex_filter(sampler.behind_knee[ids], pocket[ids])
    angles = list(range(0, KEYS[-1] + 1, AUDIT_STEP))
    frames, crossed, collapsed, unsafe = {}, [], [], []

    def check_edge(first, second, depth=0):
        if ipctk.compute_collision_free_stepsize(mesh, frames[first], frames[second]) == 1:
            return
        middle = ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)
        frames[middle] = posed(*middle)
        if ipctk.has_intersections(mesh, frames[middle]) or depth == 6:
            unsafe.append([first, second])
            return
        check_edge(first, middle, depth + 1)
        check_edge(middle, second, depth + 1)

    for point in itertools.product(angles, angles):
        frames[point] = posed(*point)
        if ipctk.has_intersections(mesh, frames[point]):
            crossed.append(point)
        small = int((face_areas(frames[point] * sampler.T, faces) / original < PATCH_COLLAPSED).sum())
        if small:
            collapsed.append([*point, small])
    bad = set(crossed)
    # The patch can't see what it left out, so read the whole skin too, more coarsely.
    parts = SkinParts(base)
    whole = {name: value.copy() for name, value in groin.sculpted_targets(base, verts, bones, weights, KEYS).items()}
    for name, delta in table.items():
        whole[name][ids] += delta
    outside, pocketed = [], []
    for point in itertools.product(angles[:: WHOLE_EVERY], angles[:: WHOLE_EVERY]):
        pose = groin.sample_pose(*point)
        posed_body = pose_verts(bones, weights, groin.bind_verts(verts, whole, pose), pose)
        hits = edges_through_triangles(posed_body, parts.edges, parts.triangles)
        hits = [
            (edge, triangle)
            for edge, triangle in hits.tolist()
            if parts.label(parts.edges[edge, 0], parts.triangles[triangle, 0]) in WHOLE_PAIRS
        ]
        own = sum(bool(pocket[parts.edges[edge]].all() and pocket[parts.triangles[triangle]].all()) for edge, triangle in hits)
        if len(hits) > own:
            outside.append([*point, len(hits) - own])
        if own:
            pocketed.append([*point, own])
    for a, b in itertools.product(angles, angles):
        for next_point in ((a + AUDIT_STEP, b), (a, b + AUDIT_STEP)):
            if next_point in frames and (a, b) not in bad and next_point not in bad:
                check_edge((a, b), next_point)
    return {
        "scanStepDegrees": AUDIT_STEP,
        "poses": len(angles) ** 2,
        "crossedPoses": crossed,
        "unsafeIntervals": unsafe,
        "collapsedPoses": collapsed,
        "wholeBodyPoses": len(angles[:: WHOLE_EVERY]) ** 2,
        "wholeBodyCrossed": outside,
        "pocketCrossed": pocketed,
        "passed": not crossed
        and not unsafe
        and not collapsed
        and not outside
        and all(row[2] <= POCKET_EDGES for row in pocketed),
    }


def bake(base, verts, bones, weights, output):
    started = time.monotonic()
    ids = GroinSampler(base, verts, bones, weights).ids
    table = solve_table(base, verts, bones, weights, ids)
    solved = time.monotonic()
    report = audit(base, verts, bones, weights, ids, table)
    report |= {"solveSeconds": solved - started, "auditSeconds": time.monotonic() - solved}
    (output / "report.json").write_text(json.dumps(report, indent=1))
    np.savez_compressed(output / "table.npz", ids=ids, names=list(table), deltas=np.stack(list(table.values())))
    if not report["passed"]:
        # The pocket fails the table too when one pose has more than POCKET_EDGES crossed edges there.
        crowded = [row for row in report["pocketCrossed"] if row[2] > POCKET_EDGES]
        first = (
            report["crossedPoses"]
            or report["unsafeIntervals"]
            or report["collapsedPoses"]
            or report["wholeBodyCrossed"]
            or crowded
        )[0]
        raise RuntimeError(f"pelvic contact table fails its audit at {first}")


def fingerprint(base, verts, bones, weights):
    return bake_key(BAKED_BY, base, verts, bones, weights)


def failure(base, verts, bones, weights):
    """Why this input has no contact table, or None. Delete the record to bake it again."""
    return cache.failure(CACHE, fingerprint(base, verts, bones, weights))


def load(folder):
    with np.load(folder / "table.npz", allow_pickle=False) as saved:
        if list(saved["names"]) != list(NAMES):
            raise ValueError("Pelvic contact table has other samples")
        return saved["ids"].copy(), dict(zip(NAMES, saved["deltas"].copy(), strict=True))


def corrections(base, verts, bones, weights):
    """(patch ids, {name: deltas on the patch}), baked once per input, or None if the bake failed."""
    key = fingerprint(base, verts, bones, weights)
    table, why = cache.baked(CACHE, key, lambda folder: bake(base, verts, bones, weights, folder), load)
    if table is None:
        print(f"pelvic contact bake failed, using the sculpture alone: {why}", file=sys.stderr, flush=True)
    return table


def targets(base, verts, bones, weights):
    """The sculpture on this table's KEYS, with the skin kept out of itself where the table has an answer;
    without a table, the sculpture as the default correctives have it, on its own keys."""
    table = corrections(base, verts, bones, weights)
    if table is None:
        return groin.sculpted_targets(base, verts, bones, weights)
    result = groin.sculpted_targets(base, verts, bones, weights, KEYS)
    ids, deltas = table
    for name, delta in deltas.items():
        result[name][ids] += delta
    return result
