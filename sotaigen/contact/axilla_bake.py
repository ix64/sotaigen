"""Generate and audit local shoulder contact samples before pose-space export.

The audit covers sampled poses and adjacent intervals in the recorded domain and step size.
The default two-axis bake keeps the runtime driver's twist fade.
"""

import hashlib
import itertools
import json
import time
from pathlib import Path

import ipctk
import numpy as np

from sotaigen.contact.axilla import ShoulderSampler
from sotaigen.corrective import axilla
from sotaigen.corrective.pose_grid import PoseGrid
from sotaigen.mesh import COLLAPSED, MIRROR_TOLERANCE, face_areas, opposite_vertices

ROOT = Path(__file__).resolve().parents[2]
FLIP = np.array([-1., 1., 1.])
# The audit's scans, in degrees, each repeated until nothing fails or MAX_ROUNDS is spent.
AUDIT_STEPS = (5., 2.5, 1.25)
MAX_ROUNDS = 16


def bake_side(base, verts, bones, weights, output, *, side="left", input_hashes=None):
    ipctk.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sampler = ShoulderSampler(base, verts, bones, weights, side)
    axes = [axilla.KEYS, axilla.SPREAD_KEYS]
    grid = PoseGrid.rectilinear(axes)
    points = grid.points
    deltas = np.zeros((len(points), len(sampler.ids), 3))
    complete = np.zeros(len(points), bool)
    complete[np.flatnonzero((points == 0).all(axis=1))] = True
    report = {"side": side, "rounds": [], "samples": [],
              "scope": "twist-zero",
              "samplingPath": "straight-from-rest",
              **(input_hashes or {}),
              "codeSHA256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                             ["sotaigen/corrective/axilla.py", "sotaigen/contact/axilla_ribbons.py", "sotaigen/contact/solver.py",
                              "sotaigen/corrective/pose_grid.py", "sotaigen/contact/axilla.py", "sotaigen/contact/axilla_bake.py"]}}

    def angles(point):
        return np.array([point[0], 0., point[1]])

    def save():
        np.savez_compressed(output / "table.npz", points=grid.points, cells=grid.cells,
                            deltas=deltas, ids=sampler.ids, initialComplete=complete)
        report["seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, indent=2))

    while not complete.all():
        index = np.flatnonzero(~complete)[0]
        sample_start = time.monotonic()
        # Contact is path dependent. A nearest-key warm start can carry a different
        # fold into this pose, making both convergence and interpolation depend on
        # sample insertion order. Use the same canonical rest-to-pose path for all keys.
        deltas[index], stats = sampler.sample(angles(points[index]))
        report["samples"].append({"point": points[index].tolist(), "steps": len(stats),
                                  "seconds": time.monotonic() - sample_start,
                                  "maxIterations": max(row["iterations"] for row in stats),
                                  "unsafeConstructionChords": sum(
                                      row["straightChordSafeFraction"] < 1 for row in stats)})
        complete[index] = True
        save()
    report["initialBakeSeconds"] = time.monotonic() - started
    save()
    # Test delivered polygons as well as collision triangles, preserving the existing collapse bar.
    selected = [f for f in base["faces"] if np.isin(f, sampler.ids).all()]
    faces = [np.searchsorted(sampler.ids, f).tolist() for f in selected]
    original_area = np.maximum(face_areas(sampler.verts[sampler.ids], faces), 1e-6)

    for step in AUDIT_STEPS:
        scan_axes = [np.unique(np.r_[np.arange(axis[0], axis[-1], step), axis]) for axis in axes]
        scan = np.asarray(list(itertools.product(*scan_axes)))
        for iteration in range(MAX_ROUNDS):
            failed = []
            frames = []
            for point in scan:
                corners, weights = grid.locate(point)
                delta = np.einsum("k,kvi->vi", weights, deltas[corners])
                posed = sampler.posed(angles(point), delta)
                frames.append(posed / sampler.torso)
                crossing = ipctk.has_intersections(sampler.solver.mesh, posed / sampler.torso)
                collapsed = int((face_areas(posed, faces) / original_area < COLLAPSED).sum())
                if crossing or collapsed:
                    failed.append({"point": point.tolist(), "crossing": bool(crossing),
                                   "collapsedFaces": collapsed})
            # Check chords between neighbouring actual skinned poses, not bind deltas.
            # A chord can cut across a safe curved trajectory; subdivide such a chord
            # before deciding that the pose table itself needs another sample.
            frames = np.asarray(frames)
            failures = {tuple(row["point"]) for row in failed}
            edge_failures = []

            def check_edge(a, b, va, vb, depth=0, *, grid=grid, deltas=deltas,
                           edge_failures=edge_failures):
                if ipctk.compute_collision_free_stepsize(sampler.solver.mesh, va, vb) == 1:
                    return
                middle = (a + b) * .5
                corners, blend = grid.locate(middle)
                delta = np.einsum("k,kvi->vi", blend, deltas[corners])
                vm = sampler.posed(angles(middle), delta) / sampler.torso
                collapsed = (face_areas(vm * sampler.torso, faces) / original_area < COLLAPSED).any()
                if ipctk.has_intersections(sampler.solver.mesh, vm) or collapsed or depth == 8:
                    edge_failures.append({"point": middle.tolist(), "unsafeEdge": True,
                                          "interval": [a.tolist(), b.tolist()]})
                    return
                check_edge(a, middle, va, vm, depth + 1)
                check_edge(middle, b, vm, vb, depth + 1)

            shape = tuple(len(axis) for axis in scan_axes)
            for indices in itertools.product(*(range(n) for n in shape)):
                i = np.ravel_multi_index(indices, shape)
                if tuple(scan[i]) in failures:
                    continue
                for axis, length in enumerate(shape):
                    if indices[axis] + 1 == length:
                        continue
                    next_indices = list(indices)
                    next_indices[axis] += 1
                    j = np.ravel_multi_index(tuple(next_indices), shape)
                    if tuple(scan[j]) not in failures:
                        check_edge(scan[i], scan[j], frames[i], frames[j])
            failed.extend(edge_failures)
            # One pose may be reached from several failed intervals.
            failed = list({tuple(row["point"]): row for row in failed}.values())
            row = {"stepDegrees": step,
                   "iteration": iteration, "samples": len(grid.points), "tested": len(scan),
                   "unsafeIntervals": len(edge_failures),
                   "failed": failed, "seconds": time.monotonic() - started}
            report["rounds"].append(row)
            save()
            if not failed:
                break
            remaining = np.array([r["point"] for r in failed])
            for _ in range(min(20, len(remaining))):
                distance = np.linalg.norm(remaining[:, None] - grid.points, axis=2)
                index = distance.min(axis=1).argmax()
                if distance[index].min() < 1e-8:
                    raise RuntimeError(f"Exact shoulder sample fails: {remaining[index]}")
                point = remaining[index]
                sample_start = time.monotonic()
                delta, stats = sampler.sample(angles(point))
                report["samples"].append({"point": point.tolist(), "steps": len(stats),
                                          "seconds": time.monotonic() - sample_start,
                                          "maxIterations": max(row["iterations"] for row in stats),
                                          "unsafeConstructionChords": sum(
                                              row["straightChordSafeFraction"] < 1 for row in stats)})
                grid = grid.insert(point)
                deltas = np.concatenate((deltas, delta[None]))
                remaining = np.delete(remaining, index, axis=0)
                save()
            save()
        else:
            raise RuntimeError("Shoulder table still fails the stated audit; not ready for export")
    report["auditPassed"] = True
    save()


def mirror_side(base, verts, bones, weights, left, right):
    """Write the right side's table as the mirror image of the left's. False if it would not be one.

    On a symmetric body an independent bake of the right side only finds the left's result again,
    to within 5e-7 mm on all twelve bodies, and an intersection or a collapsed face has a mirror
    image of itself, so the left's audit holds. Nothing is written unless the rest shape mirrors,
    the solver's goals mirror at every sample, and the mirrored table, skinned on the right arm,
    mirrors the left's across the pose domain. Weights on bones an arm pose leaves still may differ:
    the hip and leg weights do, by up to 1.4%, without reaching either goal or skin.
    """
    opposite = opposite_vertices(verts)
    if opposite is None:
        return False
    with np.load(left / "table.npz", allow_pickle=False) as data:
        table = {name: data[name] for name in data.files}
    sampler = {side: ShoulderSampler(base, verts, bones, weights, side) for side in ("left", "right")}
    ids = sampler["right"].ids
    if not np.array_equal(np.sort(opposite[ids]), table["ids"]):
        return False
    source = np.searchsorted(table["ids"], opposite[ids])
    deltas = table["deltas"][:, source] * FLIP

    def angles(point):
        return np.array([point[0], 0., point[1]])

    largest = 0.
    for point in table["points"]:
        goals = {side: s.goal(angles(point))[0] * s.torso for side, s in sampler.items()}
        largest = max(largest, np.abs(goals["left"][source] * FLIP - goals["right"]).max())
    grid = PoseGrid(table["points"], table["cells"])
    scan = [np.unique(np.r_[np.arange(axis[0], axis[-1], 5.), axis])
            for axis in (axilla.KEYS, axilla.SPREAD_KEYS)]
    for point in np.vstack((table["points"], list(itertools.product(*scan)))):
        corners, blend = grid.locate(point)
        skinned = {side: sampler[side].posed(angles(point), np.einsum("k,kvi->vi", blend, values[corners]))
                   for side, values in (("left", table["deltas"]), ("right", deltas))}
        largest = max(largest, np.abs(skinned["left"][source] * FLIP - skinned["right"]).max())
    if largest > MIRROR_TOLERANCE:
        return False

    right.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(right / "table.npz", **{**table, "ids": ids, "deltas": deltas})
    report = json.loads((left / "report.json").read_text())
    (right / "report.json").write_text(json.dumps(
        {**report, "side": "right", "mirroredFromLeft": True, "mirrorErrorMillimetres": largest},
        indent=2))
    return True
