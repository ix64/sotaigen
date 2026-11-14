"""Walk the head swap through with the made-up test head, on a few bodies, and look at it.

uv run scripts/head_swap_study.py [body ...] [--coarse body,body] [--no-sheet]

For each body: writes its file if it has none, makes the test head from that file (blender/fixtures/made_up_head.py),
runs `sotaigen swap-head`, and compares the swapped file's poses with the original's. Writes
build/reports/head-swap/<body>.blend, <body>.json (the command's report) and report.json, and renders
build/reports/head-swap/sheet.png. A body in --coarse gets the head with as many neck vertices as the ring.

Nothing here bakes a contact table: `blend_body` uses the sculpted correctives.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sotaigen.output.blend import blend_body, blend_path
from sotaigen.output.blender import run_blender, script
from sotaigen.output.head_swap import (
    BODY,
    CENTRAL,
    CHECK_POSES,
    read_dump,
    summary_lines,
    swap_head,
)
from sotaigen.pipeline import BUILD
from sotaigen.rig.poses import POSES

OUT = BUILD / "reports" / "head-swap"
TEST_HEAD = ROOT / "blender" / "fixtures" / "made_up_head.py"
RENDER = ROOT / "blender" / "review" / "head_swap_render.py"
DEFAULT = ("female-compact", "male-balanced", "female-chibi")
CELL = 420
MM = 0.001
# (title, file, pose, view, how wide, hidden objects)
COLUMNS = (
    ("old head", "old", "rest", "threeQuarter", 1.5, ()),
    ("new, front", "new", "rest", "front", 1.5, ()),
    ("three quarter", "new", "rest", "threeQuarter", 1.5, ()),
    ("side", "new", "rest", "side", 1.5, ()),
    ("back", "new", "rest", "back", 1.5, ()),
    ("headTurn", "new", "headTurn", "threeQuarter", 1.7, ()),
    ("headNod", "new", "headNod", "side", 1.7, ()),
    ("jawOpen", "new", "jawOpen", "threeQuarter", 1.5, ()),
    ("jawOpen, no skin", "new", "jawOpen", "side", 1.5, (BODY, "hair")),
    ("seam, under", "new", "rest", "under", 0.8, ()),
)


def test_head(blend, out, coarse):
    run_blender(TEST_HEAD, out, *(["--coarse"] if coarse else []), blend=blend)
    return out


def posed(blend, folder, tag):
    request = folder / f"{tag}-poses.json"
    request.write_text(
        json.dumps({"poses": {n: POSES[n] for n in CHECK_POSES}, "central": CENTRAL, "limits": [True], "actions": False})
    )
    run_blender(script("pose"), request, folder / f"{tag}-posed", blend=blend)
    return np.load(folder / f"{tag}-posed.npz")


def compare_body(old, new, keep):
    """How far the skin below the head moved from the original file in each check pose (mm)."""
    return {
        name: float(np.linalg.norm(old[f"on:{name}:{BODY}"][keep] - new[f"on:{name}:{BODY}"][: len(keep)], axis=1).max())
        for name in CHECK_POSES
    }


def run_body(name, coarse):
    blend = blend_path(name)
    if not blend.exists():
        blend_body(name)
    folder = OUT / "work" / name
    folder.mkdir(parents=True, exist_ok=True)
    head = test_head(blend, folder / "test-head.blend", coarse)
    out = OUT / f"{name}.blend"
    report = swap_head(blend, head, out)
    print(name)
    print("\n".join(f"  {line}" for line in summary_lines(report)))
    old, new = posed(blend, folder, "old"), posed(out, folder, "new")
    region = read_dump(blend, folder / "old-dump").mesh(BODY).group("headRegion")[0]
    keep = np.delete(np.arange(len(old[f"rest:{BODY}"])), region)
    report["bodyBelowHeadMovedMm"] = compare_body(old, new, keep)
    return report


def sheet(names):
    shots = {"old": {}, "new": {}}
    frames = {}
    for name in names:
        bones = json.loads((BUILD / "bodies" / name / "skeleton.json").read_text())["bones"]
        head = next(b for b in bones if b["name"] == "head")
        report = json.loads((OUT / f"{name}.json").read_text())
        height = report["headHeightMm"]
        centre = (np.array(head["head"]) + np.array([0.0, 0.0, 0.42 * height])) * MM
        frames[name] = (centre, height * MM)
        for row, (_, which, pose, view, width, hide) in enumerate(COLUMNS):
            low = centre - np.array([0.0, 0.0, 0.45 * height * MM]) if view == "under" else centre
            shots[which].setdefault(name, []).append(
                {
                    "pose": POSES[pose],
                    "view": view,
                    "centre": low.tolist(),
                    "width": width * height * MM,
                    "hide": list(hide),
                    "out": str(OUT / "shots" / f"{name}-{row}.png"),
                }
            )
    (OUT / "shots").mkdir(parents=True, exist_ok=True)
    for name in names:
        for which, blend in (("old", blend_path(name)), ("new", OUT / f"{name}.blend")):
            request = OUT / "shots" / f"{name}-{which}.json"
            request.write_text(json.dumps({"central": CENTRAL, "size": CELL, "shots": shots[which][name]}))
            run_blender(RENDER, request, blend=blend)
    margin, title = 8, 26
    width = len(COLUMNS) * (CELL + margin) + 150
    height = len(names) * (CELL + margin) + title
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for col, column in enumerate(COLUMNS):
        draw.text((150 + col * (CELL + margin) + 6, 6), column[0], fill="black")
    for row, name in enumerate(names):
        y = title + row * (CELL + margin)
        draw.text((6, y + CELL // 2), name, fill="black")
        for col in range(len(COLUMNS)):
            shot = Image.open(OUT / "shots" / f"{name}-{col}.png").convert("RGB")
            image.paste(shot, (150 + col * (CELL + margin), y))
    path = OUT / "sheet.png"
    image.save(path)
    print(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", default=list(DEFAULT))
    parser.add_argument("--coarse", default="male-balanced", help="bodies that get the head whose neck welds to the ring")
    parser.add_argument("--no-sheet", action="store_true")
    parser.add_argument("--only-sheet", action="store_true")
    args = parser.parse_args(argv)
    names = args.names or list(DEFAULT)
    if not args.only_sheet:
        coarse = set(args.coarse.split(","))
        rows = {name: run_body(name, name in coarse) for name in names}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "report.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    if not args.no_sheet:
        sheet(names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
