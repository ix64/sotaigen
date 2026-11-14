"""Stress the parameter space: wrap and rig bodies nobody designed, and see what breaks first.

uv run sotaigen-dev stress oat [--anchors female-long] [--rounds 3]
uv run sotaigen-dev stress random --sex female --count 100 --seed 3
uv run sotaigen-dev stress spec SPEC.json
uv run sotaigen-dev stress report build/reports/stress/<file>.jsonl

oat     one parameter at a time: try both ends of WIDE, then bisect towards the anchor where one
        breaks, so the table says where each parameter first breaks and how
random  random combinations from one sex's range table in sotaigen/params/spec.py, held to its rules
spec    named bodies from a file as scripts/sample_sheet.py reads it
report  what failed in a run

Nothing here touches data/ or build/bodies/<body>. Rows land in build/reports/stress/ as JSON lines, one body per
row, each carrying the full parameters that made it, so any row can be built again by hand.
Every result is deterministic: the pipeline uses no randomness and the sampling takes a seed.
"""

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from pathlib import Path

# One thread per worker: the pool is the parallelism, and BLAS would otherwise oversubscribe.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sotaigen.corrective.crossings import SkinParts
from sotaigen.form.anatomy import calibrated, landmarks
from sotaigen.form.bodies import load_bodies
from sotaigen.form.checks import field_failures, field_measures
from sotaigen.form.field import Body
from sotaigen.form.head import extended, neck_ring_order
from sotaigen.form.solve import solved_controls
from sotaigen.form.wrap import (
    FOOT_TO_GROUND_MM,
    SEAM_CREASE_DEG,
    SEAM_MISS_SHARE,
    seam_radius,
    wrap_body,
)
from sotaigen.mesh import face_areas
from sotaigen.metrics import RestMesh, joint_report, pose_row
from sotaigen.rig.poses import POSES
from sotaigen.rig.skeleton import skeleton, skin_weights
from sotaigen.rig.skinning import pose_verts
from sotaigen.template import load_template

OUT = ROOT / "build" / "reports" / "stress"
# Poses whose collapse the invariants of ARCHITECTURE 3.5 forbid.
GUARDED_POSES = ("armsUp", "headTurn", "headNod")
FOREARM_AND_HAND = ("lowerArm", "hand", "index", "middle", "ring", "little", "thumb")
# How far the coarse pass reaches for each parameter: well past any body, on purpose.
WIDE = {
    "stature": (800, 2000),
    "heads": (3.5, 9.0),
    "neck": (0.0, 0.6),
    "leg": (0.33, 0.60),
    "waist": (0.25, 0.60),
    "shoulderWidth": (0.40, 0.90),
    "shoulderRise": (-0.12, 0.10),
    "chestWidth": (0.30, 0.70),
    "chestDepth": (0.22, 0.52),
    "waistBreadth": (0.22, 0.70),
    "waistDepth": (0.16, 0.46),
    "hipBreadth": (0.30, 0.80),
    "hipDepth": (0.24, 0.60),
    "ribTilt": (-5, 25),
    "pelvisTilt": (-5, 30),
    "armRadius": (0.03, 0.12),
    "neckRadius": (0.03, 0.12),
    "thighRadius": (0.07, 0.20),
    "kneeRadius": (0.04, 0.14),
    "upperArm": (0.30, 0.75),
    "lowerArmLength": (0.25, 0.65),
    "hand": (0.14, 0.40),
    "foot": (0.20, 0.55),
    "calfRadius": (0.04, 0.14),
    "ankleRadius": (0.025, 0.09),
    "wristRadius": (0.02, 0.07),
    "stance": (0.0, 0.08),
    "headWidthRatio": (0.60, 1.00),
    "headDepth": (0.65, 1.05),
    "jaw": (0.30, 1.00),
    "definition": (0.0, 1.0),
    "softness": (0.5, 2.0),
}


def exit_with_parent(parent):
    """A worker's initializer: end when the process that started it is gone, however it ended."""

    def watch():
        while os.getppid() == parent:
            time.sleep(2.0)
        os._exit(1)

    threading.Thread(target=watch, daemon=True).start()


def worker_pool(workers):
    """A process pool that leaves nothing running if this process is killed."""
    return ProcessPoolExecutor(workers, initializer=exit_with_parent, initargs=(os.getpid(),))


@cache
def template():
    return extended(load_template())


@cache
def source_data():
    return load_bodies()


def field_row(p):
    """What the implicit surface measures, without wrapping anything."""
    body, L = calibrated(p)
    points = np.random.default_rng(1).uniform(-300, 300, (500, 3)) + [0, 0, L["H"] / 2]
    measures = field_measures(p)
    return {
        "T": round(float(L["T"]), 1),
        **{k: v if isinstance(v, bool) else round(v, 4) for k, v in measures.items()},
        "symmetric": bool(np.allclose(body(points), body(points * [-1, 1, 1]))),
    }


def arm_in_body(p, base, posed):
    """How many hand and forearm vertices of a posed mesh sit inside the trunk and the legs, and how deep.

    The rest field stands in for a pose that moves only the arms. The arm's own group is left out of
    it, or the rest arm would count. No invariant covers this; it is what shows in armsDown first.
    """
    body, _ = calibrated(p)
    trunk = Body([g for g in body.groups if "clips" not in g], [], body.carve)
    members = [str(name).startswith(FOREARM_AND_HAND) for name in base["segments"]]
    values = trunk(posed[np.array(members)])
    return int((values < 0).sum()), round(float(max(-values.min(), 0.0)), 1)


def chin_height(base, verts):
    """The head's lowest point well ahead of the neck, as tests/test_head.py measures it."""
    head = verts[np.asarray(base["segments"]) == "head"]
    ring = verts[neck_ring_order(load_template())]
    ahead = ring[:, 1].min() - 0.08 * (head[:, 2].max() - head[:, 2].min())
    return float(head[head[:, 1] < ahead, 2].min())


def forearm_girth(base, verts, bones, weights):
    """The forearm's girth after a rolled wrist as a share of its rest girth, as tests/test_rig.py."""
    segments = np.asarray(base["segments"])
    template_verts = np.asarray(base["verts"])
    ring = lambda points, key: points[base["rings"][key]].mean(0)
    e, w = ring(verts, "elbow"), ring(verts, "wrist")
    t = (verts - e) @ (w - e) / np.sum((w - e) ** 2)
    members = (segments == "lowerArm") & (template_verts[:, 0] > 0) & (t > 0.4)

    def girth(points):
        start, end = ring(points, "elbow"), ring(points, "wrist")
        axis = (end - start) / np.linalg.norm(end - start)
        offset = points[members] - start
        return np.linalg.norm(offset - np.outer(offset @ axis, axis), axis=1).mean()

    posed = pose_verts(bones, weights, verts, {"hand": (0, 90, 0)})
    return girth(posed) / girth(verts)


def wrap_row(p, out):
    """Wrap and rig one set of parameters; the mesh's and the poses' measures as a flat dict."""
    wrap = wrap_body(load_template(), p, out)
    base = template()
    verts = np.load(out / "wrapped.npy")
    L = landmarks(p)  # as sotaigen.pipeline.rig_landmarks
    bones = skeleton(base, verts, L)
    weights = skin_weights(base, bones, verts, L)
    rest = RestMesh(verts, base["faces"])
    area = np.maximum(face_areas(verts, base["faces"]), 1e-6)
    table = {
        pose: pose_row(rest, pose_verts(bones, weights, verts, rotations))
        for pose, rotations in POSES.items()
    }
    segments = np.asarray(base["segments"])
    head = verts[segments == "head"]
    rolls = {pose: POSES[pose] for pose in ("wristRoll", "armRoll")}
    joints = joint_report(base, verts, bones, weights, L["shoulderJoint"], rolls, calibrated(p)[0])
    inside, depth = arm_in_body(p, base, pose_verts(bones, weights, verts, POSES["armsDown"]))
    parts = SkinParts(base)
    row = {
        "legsPast": {
            pose: parts.legs_past_midline(pose_verts(bones, weights, verts, POSES[pose]))
            for pose in ("crouch", "stride")
        },
        "wristGirth": round(forearm_girth(base, verts, bones, weights), 3),
        "armInsideVerts": inside,
        "armInsideMm": depth,
        "flipped": wrap["flippedFaces"],
        "crossedMidline": wrap["crossedMidline"],
        "footLowest": wrap["footLowest"],
        "lowest": round(float(verts[:, 2].min()), 2),
        "surfaceMax": wrap["surfaceErrorMm"]["max"],
        "handScale": wrap["handScale"],
        "footScale": wrap["footScale"],
        "handSeam": wrap["handSeamMissMm"],
        "footSeam": wrap["footSeamMissMm"],
        "handShare": round(wrap["handSeamMissMm"] / seam_radius(template(), verts, "hand"), 3),
        "footShare": round(wrap["footSeamMissMm"] / seam_radius(template(), verts, "foot"), 3),
        "handCrease": wrap["handSeamCreaseDeg"],
        "footCrease": wrap["footSeamCreaseDeg"],
        "shortestEdge": wrap["shortestEdgeMm"],
        "cornerP1": wrap["cornerAngleDeg"]["p1"],
        "cornerP99": wrap["cornerAngleDeg"]["p99"],
        "edgeP1": wrap["edgeChangeFromWarp"]["p1"],
        "headTop": round((float(head[:, 2].max()) - L["H"]) / L["head"], 4),
        "headBottom": round((chin_height(base, verts) - L["chin"]) / L["head"], 4),
        "restMoves": bool(not np.allclose(pose_verts(bones, weights, verts, {}), verts, atol=1e-6)),
        "weightSumOff": bool(not np.allclose(weights.sum(1), 1.0)),
        "influencesMax": int((weights > 0).sum(1).max()),
        "smallestRestFace": round(float(area.min()), 3),
        "restCreases": joints["restCreases"],
        "collapsed": {pose: table[pose]["collapsedFaces"] for pose in POSES},
        "girth": {
            pose: {k: round(v, 3) for k, v in joints["poses"][pose]["girth"].items()}
            for pose in rolls
        },
    }
    return row


def failures(row):
    """Which invariants of ARCHITECTURE 3.5 a row breaks, by name, cheapest to read first."""
    if "error" in row:
        return ["crash"]
    bad = field_failures(row)
    if not row["symmetric"]:
        bad.append("symmetry")
    if "flipped" not in row:
        return bad
    if row["flipped"]:
        bad.append("flipped")
    if row["crossedMidline"]:
        bad.append("crossedMidline")
    if abs(row["footLowest"]) > FOOT_TO_GROUND_MM or abs(row["lowest"]) > FOOT_TO_GROUND_MM:
        bad.append("ground")
    for part, share, crease in (
        ("hand", row["handShare"], row["handCrease"]),
        ("foot", row["footShare"], row["footCrease"]),
    ):
        if share > SEAM_MISS_SHARE[part]:
            bad.append(f"{part}SeamMiss")
        if crease > SEAM_CREASE_DEG[part]:
            bad.append(f"{part}SeamCrease")
    if abs(row["headTop"]) > 0.03:
        bad.append("headTop")
    if abs(row["headBottom"]) > 0.05:
        bad.append("headBottom")
    if row["restMoves"]:
        bad.append("restMoves")
    if row["weightSumOff"] or row["influencesMax"] > 4:
        bad.append("weights")
    for pose in GUARDED_POSES:
        if row["collapsed"][pose]:
            bad.append(f"collapsed:{pose}")
    if row["wristGirth"] < 0.95:
        bad.append("wristGirth")
    if row["legsPast"]["crouch"] or row["legsPast"]["stride"]:
        bad.append("legsPast")
    return bad


def evaluate(p, keep=None, thorough=False):
    """All the measures of one body: the field first, then wrap and rig unless the field has failed.

    The field takes a fraction of a second and the rest some seconds, so a body already broken isn't
    worth wrapping. An exception is a result, because finding out what crashes is what this is for.
    `keep` names a folder to leave the mesh in.
    """
    start = time.perf_counter()
    row = {}
    try:
        row.update(field_row(p))
        if not failures(row) or thorough:
            with tempfile.TemporaryDirectory() as scratch:
                out = Path(keep) if keep else Path(scratch)
                row.update(wrap_row(p, out))
    except Exception as error:  # noqa: BLE001 - see the docstring
        row["error"] = f"{type(error).__name__}: {error}"
    row["fail"] = failures(row)
    row["seconds"] = round(time.perf_counter() - start, 1)
    return row


def job(spec):
    """Run one spec in a worker: the controls `p` or the measurements `body`, and any labels it came with."""
    return {**spec, **evaluate(spec.get("p") or solved_controls(spec["body"]))}


def run(specs, tag, workers):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{tag}.jsonl"
    start = time.perf_counter()
    rows = []
    with path.open("w") as sink, worker_pool(workers) as pool:
        for row in pool.map(job, specs):
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            rows.append(row)
    print(f"{len(rows)} bodies in {time.perf_counter() - start:.0f}s -> {path}")
    return rows


def spec_for(anchor_name, param, value):
    p = {**source_data()[anchor_name], param: value}
    return {"anchor": anchor_name, "param": param, "value": value, "p": p}


def cmd_oat(args):
    """Both ends of WIDE, then bisect towards the anchor from each end that breaks."""
    params = args.params.split(",") if args.params else list(WIDE)
    edges = []
    for name in args.anchors.split(","):
        for param in params:
            home = source_data()[name][param]
            for side, end in zip(("low", "high"), WIDE[param], strict=True):
                edges.append(
                    {"anchor": name, "param": param, "side": side, "good": home, "try": end}
                )
    tag = args.tag or "oat"
    rows = []
    for round_number in range(args.rounds + 1):
        live = [edge for edge in edges if edge.get("try") is not None]
        if not live:
            break
        specs = [spec_for(e["anchor"], e["param"], round(e["try"], 5)) for e in live]
        batch = run(specs, f"{tag}-{round_number}", args.workers)
        rows += batch
        for edge, row in zip(live, batch, strict=True):
            if row["fail"]:
                edge["bad"], edge["reason"] = row["value"], row["fail"]
            else:
                edge["good"] = row["value"]
            edge["try"] = (edge["good"] + edge["bad"]) / 2 if "bad" in edge else None
    (OUT / f"{tag}-edges.json").write_text(json.dumps(edges, indent=1))
    print_summary(rows)
    for edge in edges:
        bad = f"bad {edge['bad']:<9.5g} {' '.join(edge['reason'])}" if "bad" in edge else "no break"
        print(
            f"{edge['anchor']:<16}{edge['param']:<22}{edge['side']:<5}"
            f" good {edge['good']:<9.5g} {bad}"
        )


def cmd_random(args):
    from sotaigen.form.checks import sample_spec

    rng = np.random.default_rng(args.seed)
    specs = []
    for index in range(args.count):
        specs.append({"index": index, "seed": args.seed, "body": sample_spec(rng, args.sex)})
    print_summary(run(specs, args.tag or f"random-{args.sex}-{args.seed}", args.workers))


def cmd_spec(args):
    specs = json.loads(Path(args.file).read_text())
    named = [{"index": name, "body": spec["body"]} for name, spec in specs.items()]
    print_summary(run(named, args.tag or Path(args.file).stem, args.workers))


def print_summary(rows):
    broken = [row for row in rows if row["fail"]]
    print(f"{len(broken)} of {len(rows)} break an invariant")
    counts = {}
    for row in broken:
        for reason in row["fail"]:
            counts[reason] = counts.get(reason, 0) + 1
    for reason, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"  {reason:<24}{count}")


def cmd_report(args):
    rows = [json.loads(line) for line in Path(args.file).read_text().splitlines()]
    print_summary(rows)
    for row in rows:
        if row["fail"]:
            where = f"#{row['index']}" if "index" in row else f"{row['param']}={row['value']:g}"
            print(f"{where}: {' '.join(row['fail'])} {row.get('error', '')}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sotaigen-dev stress", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    oat = commands.add_parser("oat")
    oat.add_argument("--anchors", default="female-long")
    oat.add_argument("--params", help="comma list, default all")
    oat.add_argument("--rounds", type=int, default=3, help="bisections after the two ends")
    oat.add_argument("--tag")
    rand = commands.add_parser("random")
    rand.add_argument("--sex", required=True, help="female or male: whose ranges and rules")
    rand.add_argument("--count", type=int, default=100)
    rand.add_argument("--seed", type=int, default=1)
    rand.add_argument("--tag")
    named = commands.add_parser("spec")
    named.add_argument("file")
    named.add_argument("--tag")
    report = commands.add_parser("report")
    report.add_argument("file")
    for sub in (oat, rand, named):
        sub.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    commands = {"oat": cmd_oat, "random": cmd_random, "spec": cmd_spec, "report": cmd_report}
    commands[args.command](args)


if __name__ == "__main__":
    main()
