"""The clean range of each joint on the typical bodies: sweep, then sheets to judge by eye.

uv run scripts/clean_range.py measure [body ...] [--joints a,b]
    [--write-poses]
uv run scripts/clean_range.py summary
uv run scripts/clean_range.py sheets [--only arm-up,elbow]
uv run scripts/clean_range.py sheet <joint> [--angles 30,55]
    [--names pose,pose] [--bodies ...] [--view front] [--look clay,toon] [--focus shoulder]
    [--context 2.6] [--rows body|pose|view]

`measure` turns each joint of docs/systems/rig.md's table on its own from rest, both sides
together, skinned the way the build skins its check poses (the production correctives bound into
the rest skin, then pose_verts). Per joint, angle and body it records collapsed faces by segment,
edges newly through skin by pair of parts (beyond rest), newly folded edges, the elbow card's
expected-fold split, and for the arm the upper arm's angle from straight down. A few one-sided
probes (the left hand to the mouth, near the face, on top of the head) add how far the hand is
from the head. It writes build/reports/clean-range/bodies/<body>.json and, with --write-poses, the posed
meshes the sheets render, as a build tree under build/reports/clean-range/poses that sotaigen_dev/review.py
reads.

`summary` merges the bodies into build/reports/clean-range/sweep.json, prints the worst body per joint and
angle, and writes first-worse.txt: per joint and body, the first angle at which each measure gets
worse. `sheets` renders SHEETS into build/reports/clean-range/sheets/, one view and one template a sheet.
None of this passes or fails anything: the clean range is judged from the sheets.
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sotaigen.corrective.crossings import SkinParts
from sotaigen.corrective.recipe import Correctives
from sotaigen.mesh import COLLAPSED, face_pairs, write_obj
from sotaigen.metrics import RestMesh, count_by, folded_edges
from sotaigen.pipeline import BUILD, fingerprints, rigged, surface
from sotaigen.rig.skinning import pose_skins, pose_verts

OUT = BUILD / "reports" / "clean-range"
TYPICAL = [
    f"{sex}-{style}"
    for sex in ("female", "male")
    for style in ("chibi", "compact", "slim", "long", "balanced", "rounded")
]


def finger_curl(amount):
    return {
        f"{finger}{joint}": (amount * share, 0, 0)
        for finger in ("index", "middle", "ring", "little")
        for joint, share in (("Proximal", 0.8), ("Intermediate", 1.0), ("Distal", 0.65))
    }


# A joint: its pose for an angle, the angles, and the pairs of parts that matter there. The poses
# are scripts/limits.py's, with more angles; the table in docs/systems/rig.md has the same rows.
JOINTS = {
    "arm up": (lambda a: {"upperArm": (0, 0, a)}, (15, 30, 45, 55, 65, 72, 80, 90, 110, 130)),
    "arm forward": (
        lambda a: {"upperArm": (a, 0, 0)},
        (15, 30, 45, 60, 75, 90, 95, 110, 130, 150),
    ),
    "arm back": (lambda a: {"upperArm": (-a, 0, 0)}, (10, 20, 30, 45, 60)),
    "arm down": (lambda a: {"upperArm": (0, 0, -a)}, (6, 12, 18, 24, 30, 36)),
    "arm roll": (lambda a: {"upperArm": (0, a, 0)}, (30, 60, 90)),
    "arm roll back": (lambda a: {"upperArm": (0, -a, 0)}, (30, 60, 90)),
    "elbow": (lambda a: {"lowerArm": (a, 0, 0)}, (30, 45, 60, 90, 120, 140, 150)),
    "wrist bend": (lambda a: {"hand": (a, 0, 0)}, (15, 30, 45, 60, 80)),
    "wrist back": (lambda a: {"hand": (-a, 0, 0)}, (15, 30, 45, 60, 80)),
    "wrist side": (lambda a: {"hand": (0, 0, a)}, (15, 30, 45)),
    "wrist side back": (lambda a: {"hand": (0, 0, -a)}, (15, 30, 45)),
    "wrist roll": (lambda a: {"hand": (0, a, 0)}, (45, 90, 120)),
    "fingers": (finger_curl, (30, 60, 90, 110)),
    "neck nod": (lambda a: {"neck": (a, 0, 0), "head": (a, 0, 0)}, (5, 10, 15, 20, 30, 45)),
    "neck back": (lambda a: {"neck": (-a, 0, 0), "head": (-a, 0, 0)}, (5, 10, 15, 20, 30)),
    "neck turn": (lambda a: {"neck": (0, a, 0), "head": (0, a, 0)}, (10, 15, 20, 30, 45)),
    "neck side": (lambda a: {"neck": (0, 0, a), "head": (0, 0, a)}, (5, 10, 15, 20, 30)),
    "spine bend": (lambda a: {"spine": (a, 0, 0), "chest": (a, 0, 0)}, (10, 20, 30, 45)),
    "spine back": (lambda a: {"spine": (-a, 0, 0), "chest": (-a, 0, 0)}, (10, 20, 30)),
    "spine side": (lambda a: {"spine": (0, 0, a), "chest": (0, 0, a)}, (10, 20, 30)),
    "spine twist": (lambda a: {"spine": (0, a, 0), "chest": (0, a, 0)}, (15, 30, 45)),
    "hip flex": (lambda a: {"upperLeg": (a, 0, 0)}, (15, 30, 45, 60, 75, 85, 100, 120)),
    "hip extend": (lambda a: {"upperLeg": (-a, 0, 0)}, (10, 20, 30, 45)),
    "hip out": (lambda a: {"upperLeg": (0, 0, a)}, (10, 20, 30, 45, 60)),
    "hip in": (lambda a: {"upperLeg": (0, 0, -a)}, (5, 10, 20, 30)),
    "hip roll": (lambda a: {"upperLeg": (0, a, 0)}, (15, 30, 45)),
    "hip roll back": (lambda a: {"upperLeg": (0, -a, 0)}, (15, 30, 45)),
    "knee": (lambda a: {"lowerLeg": (-a, 0, 0)}, (30, 45, 60, 90, 115, 130, 150)),
    "ankle up": (lambda a: {"foot": (-a, 0, 0)}, (15, 30, 45)),
    "ankle down": (lambda a: {"foot": (a, 0, 0)}, (15, 30, 45, 60)),
    "toes": (lambda a: {"toes": (-a, 0, 0)}, (15, 30, 45, 60)),
}

# One-sided probes for the downstream targets: the left hand to the head and to the face. Each is
# named by its upper arm (forward, spread) and elbow angles.
PROBE_ARMS = {
    "hand to head": [
        (forward, spread, elbow)
        for forward in (75, 90, 110, 130)
        for spread in (0, 30)
        for elbow in (110, 130, 145)
    ],
    # The hand on top of the head: the arm raised to the side and rolled out (local Y -90), the
    # elbow bent. Without the roll a bent elbow takes the hand forward, away from the head.
    "hand over head": [
        (0, spread, elbow, -90) for spread in (90, 110, 130) for elbow in (100, 110, 130)
    ],
    "hand near face": [
        (forward, spread, elbow)
        for forward in (30, 45, 60)
        for spread in (-20, 0)
        for elbow in (120, 135, 145)
    ],
}


def probe_pose(forward, spread, elbow, roll=0):
    return {"leftUpperArm": (forward, roll, spread), "leftLowerArm": (elbow, 0, 0)}


def probe_name(forward, spread, elbow, roll=0):
    return f"f{forward}s{spread}e{elbow}" + (f"r{roll}" if roll else "")


class Body:
    """One typical body, its correctives and what every posed copy is measured against."""

    def __init__(self, preset):
        self.preset = preset
        rig = rigged(preset)
        self.base, self.verts, self.bones, self.weights = rig.base, rig.verts, rig.bones, rig.weights
        self.correctives = Correctives.production(
            self.base, self.verts, self.bones, self.weights, field=surface(preset)[0]
        )
        self.faces = self.base["faces"]
        self.rest = RestMesh(self.verts, self.faces)
        self.parts = SkinParts(self.base)
        self.pairs = face_pairs(self.faces)
        self.rest_folded = folded_edges(self.verts, self.faces, self.pairs)
        self.rest_pairs = self.parts.intersections(self.verts)["intersectionsByPair"]
        segments = np.asarray(self.base["segments"])
        self.face_segment = np.array([segments[f[0]] for f in self.faces])
        regions = np.asarray(self.base["regions"])
        left = np.asarray(self.base["verts"], float)[:, 0] >= 0
        self.left_hand = np.nonzero((regions == "hand") & left)[0]
        self.head = np.nonzero(segments == "head")[0]
        self.names = [bone["name"] for bone in self.bones]
        self.elbow = fold_card()

    def posed(self, pose):
        return pose_verts(self.bones, self.weights, self.correctives.bind(self.verts, pose), pose)

    def measure(self, pose, posed):
        area = self.rest.area_ratio(posed)
        bad = area < COLLAPSED
        crossing = self.parts.intersections(posed)["intersectionsByPair"]
        new = {k: v - self.rest_pairs.get(k, 0) for k, v in crossing.items()}
        new = {k: v for k, v in new.items() if v > 0}
        folded = folded_edges(posed, self.faces, self.pairs) & ~self.rest_folded
        row = {
            "collapsed": int(bad.sum()),
            "collapsedBySegment": count_by(self.face_segment[bad]),
            "areaP1": round(float(np.percentile(area, 1)), 3),
            "newCrossings": sum(new.values()),
            "newByPair": dict(sorted(new.items())),
            "newFolds": int(folded.sum()),
            "newFoldsBySegment": count_by(self.face_segment[self.pairs[folded, 0]]),
        }
        if any(name in pose for name in ("upperArm", "leftUpperArm")):
            row["armFromDownDeg"] = round(self.arm_from_down(pose), 1)
        return row

    def arm_from_down(self, pose):
        """The left upper arm's angle from straight down, shoulder blade's share included."""
        skins = pose_skins(self.bones, pose)
        bone = self.bones[self.names.index("leftUpperArm")]
        head = skins[self.names.index("leftUpperArm")] @ np.append(bone["head"], 1)
        tail = skins[self.names.index("leftUpperArm")] @ np.append(bone["tail"], 1)
        along = (tail - head)[:3]
        along /= np.linalg.norm(along)
        return float(np.degrees(np.arccos(np.clip(-along[2], -1, 1))))

    def hand_to_head(self, posed):
        """The nearest left-hand vertex to the head's skin, in mm."""
        tree = cKDTree(posed[self.head])
        return float(tree.query(posed[self.left_hand])[0].min())


def fold_card():
    from sotaigen_dev.motion.cards import load_card

    return load_card("elbow")


def elbow_folds(body, posed):
    from sotaigen_dev.motion.sheet import fold_counts

    card = body.elbow
    guard = card["guards"][0]
    return fold_counts(card, guard, body.base, body.verts, posed, body.bones, card["sweeps"][0])


def pose_tree(preset, name, posed, faces):
    """A posed mesh where sotaigen_dev/review.py finds a pose named `name` of `body`."""
    folder = OUT / "poses" / preset / "poses"
    folder.mkdir(parents=True, exist_ok=True)
    write_obj(folder / f"{name}.obj", posed, faces)
    (OUT / "poses" / preset / "fingerprint.json").write_text(
        json.dumps({"pose": fingerprints()["pose"]})
    )


def slug(joint, angle):
    return f"{joint.replace(' ', '-')}-{angle}"


def cmd_measure(args):
    presets = args.presets or TYPICAL
    joints = args.joints.split(",") if args.joints else [*JOINTS, *PROBE_ARMS]
    for preset in presets:
        path = OUT / "bodies" / f"{preset}.json"
        table = json.loads(path.read_text()) if path.exists() else {}
        table.setdefault("joints", {})
        table.setdefault("probes", {})
        body = Body(preset)
        for joint in joints:
            if joint in PROBE_ARMS:
                rows = table["probes"][joint] = {}
                for arm in PROBE_ARMS[joint]:
                    pose = probe_pose(*arm)
                    posed = body.posed(pose)
                    row = body.measure(pose, posed)
                    row["handToHeadMm"] = round(body.hand_to_head(posed), 1)
                    rows.setdefault(probe_name(*arm), {})[preset] = row
                    if args.write_poses:
                        pose_tree(preset, f"{joint.replace(' ', '-')}-{probe_name(*arm)}",
                                  posed, body.faces)
                continue
            make, angles = JOINTS[joint]
            rows = table["joints"][joint] = {}
            for angle in (0, *angles):
                pose = make(angle)
                posed = body.posed(pose)
                row = body.measure(pose, posed)
                if joint == "elbow":
                    row["elbowFolds"] = elbow_folds(body, posed)
                rows.setdefault(str(angle), {})[preset] = row
                if args.write_poses:
                    pose_tree(preset, slug(joint, angle), posed, body.faces)
        print(preset, flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(table, indent=1))
    return 0


def merged():
    """Every body's sweep in one table, by joint, angle, then body; written as sweep.json."""
    table = {"joints": {}, "probes": {}}
    for path in sorted((OUT / "bodies").glob("*.json")):
        body = json.loads(path.read_text())
        for kind, joints in table.items():
            for joint, rows in body[kind].items():
                for angle, cells in rows.items():
                    joints.setdefault(joint, {}).setdefault(angle, {}).update(cells)
    (OUT / "sweep.json").write_text(json.dumps(table, indent=1))
    return table


def first_worse_table(table):
    """Per joint and body, the first angle with collapsed faces, new folds, new crossings."""
    short = [f"{p[0]}-{p.split('-')[1][:4]}" for p in TYPICAL]
    lines = ["joint            measure    " + " ".join(f"{s:>7}" for s in short)]
    measures = (
        ("collapsed", lambda c: c["collapsed"] > 0),
        ("folds", lambda c: c["newFolds"] > 0),
        ("crossings", lambda c: c["newCrossings"] > 0),
    )
    for joint, rows in table["joints"].items():
        angles = sorted(rows, key=float)[1:]
        for name, worse in measures:
            cells = [
                next((a for a in angles if preset in rows[a] and worse(rows[a][preset])), "-")
                for preset in TYPICAL
            ]
            lines.append(f"{joint:16} {name:10} " + " ".join(f"{c:>7}" for c in cells))
    return "\n".join(lines) + "\n"


def cmd_summary(args):
    table = merged()
    text = first_worse_table(table)
    (OUT / "first-worse.txt").write_text(text)
    print(text)
    for joint, rows in table["joints"].items():
        print(joint)
        for angle in sorted(rows, key=float):
            cells = rows[angle]
            worst = max(cells, key=lambda p: cells[p]["collapsed"])
            cross = max(cells, key=lambda p: cells[p]["newCrossings"])
            fold = max(cells, key=lambda p: cells[p]["newFolds"])
            pairs = {}
            for cell in cells.values():
                for k, v in cell["newByPair"].items():
                    pairs[k] = max(pairs.get(k, 0), v)
            top = sorted(pairs.items(), key=lambda kv: -kv[1])[:3]
            arm = [c.get("armFromDownDeg") for c in cells.values() if "armFromDownDeg" in c]
            arm_text = f" arm {min(arm):.0f}-{max(arm):.0f}" if arm else ""
            print(
                f"  {angle:>4}{arm_text}  collapsed {cells[worst]['collapsed']:3} ({worst})  "
                f"folds {cells[fold]['newFolds']:3} ({fold})  "
                f"through {cells[cross]['newCrossings']:4} ({cross})  {top}"
            )
    return 0


def cmd_sheet(args):
    from sotaigen_dev import review

    review.BUILD = OUT / "poses"
    if args.focus and args.context:
        # Wider than review's own crop, so a whole axilla or hip fold stays in the frame.
        review.FOCUS[args.focus] = replace(review.FOCUS[args.focus], context=args.context)
    presets = args.presets.split(",") if args.presets else TYPICAL
    if args.names:
        names = args.names.split(",")
    elif args.joint in PROBE_ARMS:
        names = [f"{args.joint.replace(' ', '-')}-{a}" for a in args.angles.split(",")]
    else:
        angles = args.angles.split(",") if args.angles else ["0", *map(str, JOINTS[args.joint][1])]
        names = [slug(args.joint, a) for a in angles]
    review.POSES = {name: {} for name in names}
    axes = review.select_axes(
        presets, pose=",".join(names), view=args.view, focus=args.focus, look=args.look
    )
    rows, cols = {
        "preset": (("look", "preset"), ("pose", "view")),
        "pose": (("look", "pose"), ("preset", "view")),
        "view": (("look", "preset", "view"), ("pose",)),
    }[args.rows]
    out = Path(args.out) if args.out else OUT / "sheets" / f"{args.joint.replace(' ', '-')}.png"
    review.make_sheet(
        axes, rows=rows, cols=cols, cell_px=args.cell, mm_per_px=args.mm_per_px, out=out
    )
    return 0


def names(joint, *angles):
    return [slug(joint, a) for a in angles]


# The sheets to judge the clean range by: name, pose names, views, focus and its context.
SHEETS = [
    ("arm-up", names("arm up", 0, 30, 55, 72, 90, 110, 130),
     ("front", "threequarter", "backquarter"), "shoulder", 2.6),
    ("arm-forward", names("arm forward", 0, 30, 45, 60, 75, 95, 130),
     ("side", "threequarter", "front"), "shoulder", 2.6),
    ("arm-back", names("arm back", 0, 20, 30, 45, 60), ("side", "backquarter"), "shoulder", 2.6),
    ("arm-down", names("arm down", 0, 12, 24, 30, 36), ("front",), "shoulder", 2.6),
    ("arm-roll", [*names("arm roll back", 90, 60), *names("arm roll", 0, 60, 90)],
     ("front", "backquarter"), "shoulder", 2.6),
    ("elbow", names("elbow", 0, 45, 90, 120, 140, 150), ("side", "back"), "elbow", None),
    ("wrist-flex", [*names("wrist back", 80, 60, 45), *names("wrist bend", 0, 45, 60, 80)],
     ("front", "back"), "wrist", None),
    ("wrist-side", [*names("wrist side back", 45, 30), *names("wrist side", 0, 30, 45),
                    *names("wrist roll", 90, 120)], ("side",), "wrist", None),
    ("fingers", names("fingers", 0, 30, 60, 90, 110), ("threequarter",), "hand", None),
    ("neck-turn", names("neck turn", 0, 15, 20, 30, 45), ("front", "threequarter"), "neck", 2.0),
    ("neck-nod", [*names("neck back", 30, 20, 10), *names("neck nod", 0, 10, 20, 30, 45)],
     ("side", "threequarter"), "neck", 2.0),
    ("neck-side", names("neck side", 0, 10, 20, 30), ("front",), "neck", 2.0),
    ("hand-to-head", [f"hand-to-head-{n}" for n in
                      ("f90s0e130", "f110s0e130", "f110s30e130", "f130s30e130", "f130s0e145")],
     ("front", "threequarter"), "shoulder", 3.4),
    ("hand-over-head", [f"hand-over-head-{n}" for n in
                        ("f0s90e110r-90", "f0s110e110r-90", "f0s130e110r-90", "f0s110e130r-90")],
     ("front", "backquarter"), "shoulder", 3.4),
    ("hand-near-face", [f"hand-near-face-{n}" for n in
                        ("f45s-20e145", "f60s-20e135", "f60s0e135", "f60s0e145")],
     ("front", "threequarter"), "shoulder", 3.4),
    ("hip-flex", names("hip flex", 0, 30, 45, 60, 85, 100, 120), ("side", "threequarter"), "hip",
     None),
    ("hip-extend", names("hip extend", 0, 10, 20, 30, 45), ("side", "backquarter"), "hip", None),
    ("hip-spread", [*names("hip in", 10, 5), *names("hip out", 0, 10, 20, 30, 45, 60)],
     ("front",), "hip", None),
    ("hip-roll", [*names("hip roll back", 45, 30), *names("hip roll", 0, 15, 30, 45)],
     ("front",), "hip", None),
    ("knee", names("knee", 0, 45, 90, 115, 130, 150), ("side", "threequarter"), "knee", None),
    ("ankle", [*names("ankle up", 45, 30), *names("ankle down", 0, 30, 45, 60),
               *names("toes", 30, 60)], ("side",), "ankle", None),
    ("spine-flex", [*names("spine back", 30, 20), *names("spine bend", 0, 20, 30, 45)],
     ("side",), None, None),
    ("spine-side", [*names("spine side", 0, 10, 20, 30), *names("spine twist", 15, 30, 45)],
     ("front",), None, None),
]


def cmd_sheets(args):
    """Every sheet in SHEETS, one per view and template: looks as rows over the six bodies."""
    wanted = args.only.split(",") if args.only else [row[0] for row in SHEETS]
    for name, poses, views, focus, context in SHEETS:
        if name not in wanted:
            continue
        for sex in args.sexes.split(","):
            presets = ",".join(p for p in TYPICAL if p.startswith(sex))
            for view in views:
                out = OUT / "sheets" / f"{name}-{view}-{sex}.png"
                cmd_sheet(argparse.Namespace(
                    joint=name, names=",".join(poses), angles=None, presets=presets, view=view,
                    look=args.look, focus=focus, context=context, rows="preset", cell=args.cell,
                    mm_per_px=None, out=str(out),
                ))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    measure = commands.add_parser("measure")
    measure.add_argument("presets", nargs="*")
    measure.add_argument("--joints", help="comma list of joints or probes; default all")
    measure.add_argument("--write-poses", action="store_true", help="also write OBJs for sheets")
    commands.add_parser("summary")
    sheet = commands.add_parser("sheet")
    sheet.add_argument("joint")
    sheet.add_argument("--angles")
    sheet.add_argument("--names", help="pose names as written, across joints: arm-up-55,elbow-90")
    sheet.add_argument("--presets")
    sheet.add_argument("--view", default="front")
    sheet.add_argument("--look", default="clay,toon")
    sheet.add_argument("--focus")
    sheet.add_argument("--rows", choices=("preset", "pose", "view"), default="preset")
    sheet.add_argument("--context", type=float, help="focus context, wider than review's")
    sheet.add_argument("--cell", type=int)
    sheet.add_argument("--mm-per-px", type=float)
    sheet.add_argument("--out")
    sheets = commands.add_parser("sheets")
    sheets.add_argument("--only", help="comma list of sheet names")
    sheets.add_argument("--sexes", default="female,male")
    sheets.add_argument("--look", default="clay,toon")
    sheets.add_argument("--cell", type=int, default=240)
    args = parser.parse_args(argv)
    run = {"measure": cmd_measure, "summary": cmd_summary, "sheet": cmd_sheet, "sheets": cmd_sheets}
    return run[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
