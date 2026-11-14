"""The hand-to-head contacts on the typical bodies: the solve's numbers, and the sheets to look at.

    uv run scripts/touch_study.py [body ...] [--no-sheets]

Writes build/reports/touch/report.json: per body and contact, how the solve came out (sotaigen/rig/touch.py), the seconds
it took and the crossings that are new against the rest pose, for one hand and for both. Poses and sheets
go through scripts/upper_body_study.py, under build/reports/upper-body/. Nothing here bakes a contact table.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import upper_body_study as study

from sotaigen.pipeline import BUILD
from sotaigen.rig.touch import CONTACTS, Arm, solve_contact, touch_name

OUT = BUILD / "reports" / "touch"
LOOKS = "clay,toon"
VIEWS = "front,threequarter,side"
# What a contact's hand may not pass through, as the test counts it (tests/test_touch.py).
INTO_HEAD = ("hand|head", "hand|neck", "head|lowerArm", "lowerArm|neck", "head|upperArm", "neck|upperArm", "hand|hand")


def measure(preset):
    body = study.Body(preset)
    arm = Arm(body.base, body.verts, body.bones, body.weights)
    rows = {}
    for contact in CONTACTS:
        started = time.perf_counter()
        solved = solve_contact(arm, contact)
        seconds = time.perf_counter() - started
        row = {"seconds": round(seconds, 2), **solved.report(), "angles": solved.angles}
        for label, side in (("left", "left"), ("both", "both")) if contact.both else (("left", "left"),):
            posed = body.posed(solved.pose(side))
            found = body.parts.intersections(posed)["intersectionsByPair"]
            row[f"new-{label}"] = {
                pair: count - body.rest_pairs.get(pair, 0)
                for pair, count in sorted(found.items())
                if count > body.rest_pairs.get(pair, 0)
            }
        rows[contact.name] = row
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("presets", nargs="*")
    parser.add_argument("--no-sheets", action="store_true")
    args = parser.parse_args(argv)
    presets = args.presets or study.TYPICAL
    report = {preset: measure(preset) for preset in presets}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print(OUT / "report.json")
    if args.no_sheets:
        return 0
    names = [touch_name(contact.name, "left") for contact in CONTACTS]
    both = [touch_name(contact.name, "both") for contact in CONTACTS if contact.both]
    study.main(["poses", *presets, "--only", ",".join([*names, *both])])
    for name in names:
        study.main(
            ["sheet", f"touch-{name}", "--poses", name, "--presets", ",".join(presets), "--view", VIEWS,
             "--look", LOOKS, "--focus", "head", "--context", "3.0", "--rows", "preset", "--cell", "300"]
        )
    study.main(
        ["sheet", "touch-both", "--poses", ",".join(both), "--presets", ",".join(presets), "--view", "front,threequarter",
         "--look", "clay", "--focus", "head", "--context", "3.0", "--rows", "preset", "--cell", "300"]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
