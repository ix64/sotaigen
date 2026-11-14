"""The hanging arm's IPC contact table: baked on top of the shoulder's, audited and cached.

What it is for and how its targets are driven are in sotaigen/corrective/hang.py; this module only
makes the IPC table, which the default correctives don't use (sotaigen.corrective.hang_field makes
the default one). A body whose bake fails is generated without it, and failure() says why.
"""

import itertools
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from sotaigen.contact import cache
from sotaigen.contact.axilla_cache import targets as shoulder_targets
from sotaigen.contact.solver import ContactPatch
from sotaigen.corrective import axilla
from sotaigen.corrective.crossings import SkinParts, edges_through_triangles
from sotaigen.corrective.hang import (
    ARM_GROUPS,
    EDGE,
    FADE,
    FLIP,
    FORWARD,
    REACH,
    SIDES,
    adduction_keys,
    arm_pose,
    bind_verts,
    driver_metadata,
    sample_names,
    segment_distance,
)
from sotaigen.form.field import ramp
from sotaigen.mesh import COLLAPSED, face_areas, opposite_vertices
from sotaigen.paths import cache_dir
from sotaigen.rig.skinning import pose_skins, pose_verts, torso_length
from sotaigen.sources import bake_key

CACHE = cache_dir() / "hang-contact"
# What decides an entry or a failure record: bake() solves on top of the shoulder table, audits and
# writes it, and load() reads it back.
BAKED_BY = ("sotaigen/contact/hang.py:bake", "sotaigen/contact/hang.py:load")


STEP = 1.0


# Barrier width, in shares of the torso. As wide as the groin's: at 0.004 the skin sat so close at the
# keys that the eased morphs between them crossed again.
GAP = 0.008


AUDIT_FORWARD = 5


AUDIT_ADDUCTION = 0.5


WHOLE_FORWARD = 10


WHOLE_ADDUCTION = 1.0


WORKERS = 4


WORKERS_UNDER_JOBS = 2


WHOLE_PARTS = ("upperArm", "lowerArm", "hand")


class HangSampler:
    """Walk one side's arm down from the shoulder table's edge, at one forward angle at a time."""

    def __init__(self, base, verts, bones, weights, side, shoulder, keys):
        self.base, self.verts, self.bones, self.weights = base, verts, bones, weights
        self.side, self.shoulder, self.keys = side, shoulder, keys
        self.T = torso_length(bones)
        names = [b["name"] for b in bones]
        groups = SkinParts(base).groups
        raw = np.asarray(base["verts"])
        here = raw[:, 0] > 0 if side == "left" else raw[:, 0] < 0
        # Where the arm will be at the deepest adduction, at any forward key.
        reach = np.full(len(verts), np.inf)
        for forward in FORWARD:
            skins = pose_skins(bones, arm_pose(side, forward, keys[-1]))
            for bone_name in (side + "UpperArm", side + "LowerArm", side + "Hand"):
                i = names.index(bone_name)
                ends = [skins[i][:3, :3] @ bones[i][end] + skins[i][:3, 3] for end in ("head", "tail")]
                reach = np.minimum(reach, segment_distance(verts, *ends))
        reach /= self.T
        body = np.isin(groups, ("trunk", "upperLeg")) & here
        body_support = body * ramp((REACH - reach) / FADE)
        arm = bones[names.index(side + "UpperArm")]
        head, tail = np.asarray(arm["head"], float), np.asarray(arm["tail"], float)
        along = (verts - head) @ (tail - head) / np.sum((tail - head) ** 2)
        limb = np.isin(groups, ARM_GROUPS) & here
        # The shoulder's top stays with the shoulder table; the arm answers from below it.
        arm_support = limb * np.where(groups == "upperArm", ramp((along - 0.15) / 0.2), 1.0)
        support = np.maximum(body_support, arm_support)
        self.ids, triangles, polygons = axilla._patch_topology(base["faces"], raw, support)
        self.hand = (groups[self.ids] == "hand").tolist()
        # Here so that reading the driver doesn't load the contact solver.

        start = self.delivered(0, EDGE)
        self.solver = ContactPatch(start[self.ids] / self.T, triangles, polygons, support[self.ids],
                                   intrinsic_area=True, moving_boundary=True, gap=GAP)
        # A Python filter is called from ipctk's threads while the caller holds the GIL: one thread only.
        self.solver.ipc.set_num_threads(1)
        hand = self.hand
        # Fingers lie inside each other's barrier at rest, and the hand needs no contact with itself.
        self.solver.mesh.can_collide = lambda i, j: not (hand[i] and hand[j])
        self.solver.collisions.build(self.solver.mesh, self.solver.x, GAP)

    def delivered(self, forward, adduction):
        """The skin as generation gives it without this table: the shoulder's correction held at its edge."""
        held = arm_pose(self.side, forward, min(adduction, EDGE))
        bound = axilla.bind_verts(self.verts, self.shoulder, held)
        return pose_verts(self.bones, self.weights, bound, arm_pose(self.side, forward, adduction))

    def blends(self, forward, adduction):
        skins = pose_skins(self.bones, arm_pose(self.side, forward, adduction))
        return np.einsum("vb,bij->vij", self.weights[self.ids], skins[:, :3])

    def advance(self, previous, target, forward, depth=0):
        start = self.solver.x.copy()
        whole = self.delivered(forward, target)
        blend = self.blends(forward, target)
        u, _, vt = np.linalg.svd(blend[:, :, :3])
        u[:, :, -1] *= np.linalg.det(u @ vt)[:, None]
        old = self.blends(forward, previous)
        material = np.linalg.solve(old[:, :, :3], (start * self.T - old[:, :, 3])[:, :, None])[:, :, 0]
        predictor = (np.einsum("vij,vj->vi", blend[:, :, :3], material) + blend[:, :, 3]) / self.T
        try:
            result, _ = self.solver.advance(whole[self.ids] / self.T, u @ vt, initial=predictor)
        except RuntimeError:
            self.solver.x = start
            if depth == 5:
                raise
            middle = (previous + target) / 2
            self.advance(previous, middle, forward, depth + 1)
            return self.advance(middle, target, forward, depth + 1)
        return whole, result

    def column(self, forward):
        """Bind-space differences at every key below the edge, for one forward angle, walked from the edge."""
        self.solver.x = self.delivered(forward, EDGE)[self.ids] / self.T
        out, previous = [], EDGE
        for key in self.keys[1:]:
            stops = [*np.arange(previous + STEP, key, STEP), key]
            for stop in stops:
                whole, solved = self.advance(previous, float(stop), forward)
                previous = float(stop)
            blend = self.blends(forward, key)
            difference = solved * self.T - whole[self.ids]
            out.append(np.linalg.solve(blend[:, :, :3], difference[:, :, None])[:, :, 0])
        return out


_worker = None


def _start_worker(base, verts, bones, weights, side, shoulder, keys):
    global _worker
    _worker = HangSampler(base, verts, bones, weights, side, shoulder, keys)


def _solve_column(forward):
    return _worker.column(forward)


def solve_side(base, verts, bones, weights, side, shoulder, keys):
    """(patch ids, deltas shaped [forward key, adduction key, vertex, 3])."""
    ids = HangSampler(base, verts, bones, weights, side, shoulder, keys).ids
    context = multiprocessing.get_context("spawn")
    initargs = (base, np.array(verts), bones, np.array(weights), side, shoulder, keys)
    workers = WORKERS if multiprocessing.parent_process() is None else WORKERS_UNDER_JOBS
    with ProcessPoolExecutor(workers, context, _start_worker, initargs) as pool:
        columns = list(pool.map(_solve_column, FORWARD))
    return ids, np.array(columns)


def solve(base, verts, bones, weights, shoulder, keys):
    """{side: (ids, deltas)}, and whether the right is the left's mirror image."""
    left = solve_side(base, verts, bones, weights, "left", shoulder, keys)
    opposite = opposite_vertices(verts)
    if opposite is None:
        right = solve_side(base, verts, bones, weights, "right", shoulder, keys)
    else:
        right = (opposite[left[0]], left[1] * FLIP)
    return {"left": left, "right": right}, opposite is not None


def full_targets(verts, keys, tables):
    """{name: deltas over the whole mesh}."""
    names = sample_names(keys)
    out = {}
    for side, (ids, deltas) in tables.items():
        rows = iter(names[side])
        for i, _ in enumerate(FORWARD):
            for j, _ in enumerate(keys[1:]):
                _, _, name = next(rows)
                delta = np.zeros_like(verts, dtype=float)
                delta[ids] = deltas[i, j]
                out[name] = delta
    return out


def audit(base, verts, bones, weights, shoulder, keys, tables, sides=SIDES):
    """Read the table as delivered, each side over its forward and adduction range, and between poses.

    On a mirror-symmetric body the right side's skin, table and poses are the left's mirror image, and so
    is any crossing or collapsed face, so the left's audit holds for both.
    """
    import ipctk

    table = driver_metadata(keys)
    targets = full_targets(verts, keys, tables)
    parts = SkinParts(base)
    groups = parts.groups
    forwards = list(range(0, FORWARD[-1] + 1, AUDIT_FORWARD))
    steps = round((keys[-1] - EDGE) / AUDIT_ADDUCTION)
    adductions = [EDGE + (keys[-1] - EDGE) * i / steps for i in range(steps + 1)]
    report = {"forwardStepDegrees": AUDIT_FORWARD, "adductionStepDegrees": AUDIT_ADDUCTION, "sides": {}}
    passed = True
    for side in sides:
        sampler = HangSampler(base, verts, bones, weights, side, shoulder, keys)
        ids = sampler.ids
        if not np.array_equal(np.sort(ids), np.sort(tables[side][0])):
            raise ValueError(f"Hanging-arm table for the {side} side was solved on a different patch")
        local = [f for f in base["faces"] if np.isin(f, ids).all()]
        order = np.argsort(ids)
        faces = [order[np.searchsorted(ids[order], f)].tolist() for f in local]
        original = np.maximum(face_areas(verts[ids], faces), 1e-6)
        mesh = sampler.solver.mesh

        def posed(forward, adduction, sampler=sampler, side=side, ids=ids):
            pose = arm_pose(side, forward, adduction)
            bound = axilla.bind_verts(verts, shoulder, arm_pose(side, forward, min(adduction, EDGE)))
            bound = bind_verts(bound, targets, pose, table)
            blend = sampler.blends(forward, adduction)
            return (np.einsum("vij,vj->vi", blend[:, :, :3], bound[ids]) + blend[:, :, 3]) / sampler.T

        frames, crossed, collapsed, unsafe = {}, [], [], []

        def check_edge(first, second, depth=0, frames=frames, unsafe=unsafe, mesh=mesh, posed=posed):
            if ipctk.compute_collision_free_stepsize(mesh, frames[first], frames[second]) == 1:
                return
            middle = ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)
            frames[middle] = posed(*middle)
            if ipctk.has_intersections(mesh, frames[middle]) or depth == 6:
                unsafe.append([first, second])
                return
            check_edge(first, middle, depth + 1)
            check_edge(middle, second, depth + 1)

        for point in itertools.product(forwards, adductions):
            frames[point] = posed(*point)
            if ipctk.has_intersections(mesh, frames[point]):
                crossed.append([point[0], round(point[1], 2)])
            small = int((face_areas(frames[point] * sampler.T, faces) / original < COLLAPSED).sum())
            if small:
                collapsed.append([point[0], round(point[1], 2), small])
        bad = {(f, a) for f, a in crossed}
        for i, forward in enumerate(forwards):
            for j, adduction in enumerate(adductions):
                for other in ((forwards[i + 1], adduction) if i + 1 < len(forwards) else None,
                              (forward, adductions[j + 1]) if j + 1 < len(adductions) else None):
                    if other and (forward, round(adduction, 2)) not in bad and (other[0], round(other[1], 2)) not in bad:
                        check_edge((forward, adduction), other)
        # The patch can't see what it left out, so read the whole skin too, more coarsely.
        whole = []
        coarse = round((keys[-1] - EDGE) / WHOLE_ADDUCTION)
        for forward in range(0, FORWARD[-1] + 1, WHOLE_FORWARD):
            for k in range(coarse + 1):
                adduction = EDGE + (keys[-1] - EDGE) * k / coarse
                pose = arm_pose(side, forward, adduction)
                bound = axilla.bind_verts(verts, shoulder, arm_pose(side, forward, min(adduction, EDGE)))
                bound = bind_verts(bound, targets, pose, table)
                body = pose_verts(bones, weights, bound, pose)
                hits = 0
                for edge, triangle in edges_through_triangles(body, parts.edges, parts.triangles).tolist():
                    pair = {str(groups[parts.edges[edge, 0]]), str(groups[parts.triangles[triangle, 0]])}
                    if pair & set(WHOLE_PARTS) and pair - {"hand"}:
                        hits += 1
                if hits:
                    whole.append([forward, round(adduction, 2), hits])
        ok = not crossed and not unsafe and not collapsed and not whole
        passed &= ok
        report["sides"][side] = {
            "poses": len(frames), "crossedPoses": crossed, "unsafeIntervals": unsafe,
            "collapsedPoses": collapsed, "wholeBodyCrossed": whole, "passed": ok,
        }
    report["passed"] = passed
    return report


def bake(base, verts, bones, weights, output):
    started = time.monotonic()
    keys = adduction_keys(bones)
    shoulder = shoulder_targets(base, verts, bones, weights)
    tables, mirrored = solve(base, verts, bones, weights, shoulder, keys)
    solved = time.monotonic()
    report = audit(base, verts, bones, weights, shoulder, keys, tables, ("left",) if mirrored else SIDES)
    report |= {"adductionKeys": list(keys), "mirrored": mirrored, "solveSeconds": solved - started,
               "auditSeconds": time.monotonic() - solved}
    (output / "report.json").write_text(json.dumps(report, indent=1))
    np.savez_compressed(
        output / "table.npz", keys=np.array(keys),
        **{f"{side}_ids": ids for side, (ids, _) in tables.items()},
        **{f"{side}_deltas": deltas for side, (_, deltas) in tables.items()},
    )
    if not report["passed"]:
        for side, row in report["sides"].items():
            for field in ("crossedPoses", "unsafeIntervals", "collapsedPoses", "wholeBodyCrossed"):
                if row[field]:
                    raise RuntimeError(f"hanging-arm table fails its audit on the {side} side at {row[field][0]}")


def fingerprint(base, verts, bones, weights):
    return bake_key(BAKED_BY, base, verts, bones, weights)


def failure(base, verts, bones, weights):
    """Why this input has no hanging-arm table, or None. Delete the record to bake it again."""
    return cache.failure(CACHE, fingerprint(base, verts, bones, weights))


def load(folder):
    with np.load(folder / "table.npz", allow_pickle=False) as saved:
        keys = tuple(float(k) for k in saved["keys"])
        tables = {side: (saved[f"{side}_ids"].copy(), saved[f"{side}_deltas"].copy()) for side in SIDES}
    return keys, tables


def corrections(base, verts, bones, weights):
    """(adduction keys, {side: (ids, deltas)}), baked once per input; None without keys or if it failed."""
    if not adduction_keys(bones):
        return None
    key = fingerprint(base, verts, bones, weights)
    table, why = cache.baked(CACHE, key, lambda folder: bake(base, verts, bones, weights, folder), load)
    if table is None:
        print(f"hanging-arm contact bake failed, the arm hangs without it: {why}", file=sys.stderr)
    return table


def targets(base, verts, bones, weights):
    """({name: whole-mesh deltas}, driver metadata), or ({}, None) when this body has no table."""
    table = corrections(base, verts, bones, weights)
    if table is None:
        return {}, None
    keys, tables = table
    return full_targets(verts, keys, tables), driver_metadata(keys)
