"""Look at IPC contact on the groin patch, along one path or from the table generation uses.

The sculpture stays the target; contact only moves the skin where the target would pass through
itself. It walks one path at a time and draws the sculpture beside the solved skin; the bake that
generation uses is sotaigen/contact/groin.py.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sotaigen.contact import groin as groin_contact
from sotaigen.contact.groin import GroinSampler
from sotaigen.corrective import groin
from sotaigen.corrective.crossings import SkinParts
from sotaigen.mesh import write_obj
from sotaigen.pipeline import BUILD, rigged
from sotaigen.rig.skinning import pose_verts
from sotaigen_dev.motion.groin_sheet import grid, make_shot
from sotaigen_dev.review import find_blender

PAIR = "trunk|upperLeg"


def pose_along(k, path):
    return (float(k), float(k)) if path == "double" else (float(k), 0.0)


def run(name, path, angle, step, out):
    rig = rigged(name)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sampler = GroinSampler(rig.base, rig.verts, rig.bones, rig.weights)
    parts = SkinParts(rig.base)
    print(f"patch {len(sampler.ids)} vertices, {time.monotonic() - started:.1f}s", flush=True)
    faces = rig.base["faces"]
    report = []
    previous = (0.0, 0.0)
    for k in np.arange(step, angle + 1e-6, step):
        target = pose_along(k, path)
        tick = time.monotonic()
        whole, solved, stat = sampler.advance(previous, target)
        previous = target
        full = whole.copy()
        full[sampler.ids] = solved * sampler.T
        before = parts.intersections(whole)["intersectionsByPair"].get(PAIR, 0)
        after = parts.intersections(full)["intersectionsByPair"].get(PAIR, 0)
        moved = np.linalg.norm(full - whole, axis=1)
        row = {
            "angle": float(k),
            "goalPair": before,
            "solvedPair": after,
            "maxMoveMm": round(float(moved.max()), 2),
            "seconds": round(time.monotonic() - tick, 1),
            "iterations": stat["iterations"],
        }
        report.append(row)
        print(row, flush=True)
        (out / "report.json").write_text(json.dumps(report, indent=1))
        if k % 20 == 0 or k in (70, 90, 110):
            write_obj(out / f"goal-{int(k):03d}.obj", whole, faces)
            write_obj(out / f"solved-{int(k):03d}.obj", full, faces)


def from_table(name, out, angles):
    """Write the delivered body beside the sculpture alone, through the driver, and count what's left."""
    rig = rigged(name)
    out.mkdir(parents=True, exist_ok=True)
    parts = SkinParts(rig.base)
    sculpted = groin.sculpted_targets(rig.base, rig.verts, rig.bones, rig.weights)
    delivered = groin_contact.targets(rig.base, rig.verts, rig.bones, rig.weights)
    for mode in ("double", "single"):
        for angle in angles:
            pose = groin.sample_pose(angle, angle if mode == "double" else 0)
            row = {}
            for variant, table in (("goal", sculpted), ("solved", delivered)):
                bound = groin.bind_verts(rig.verts, table, pose)
                posed = pose_verts(rig.bones, rig.weights, bound, pose)
                write_obj(out / f"{mode}-{variant}-{angle:03d}.obj", posed, rig.base["faces"])
                found = parts.intersections(posed)
                row[variant] = [found["selfIntersections"], found["intersectionsByPair"].get(PAIR, 0)]
            print(mode, angle, row, flush=True)


VIEWS = [("high", "clay"), ("threequarter", "clay"), ("front", "clay"), ("threequarter", "toon")]


def render(name, out, angles, prefix=""):
    rig = rigged(name)
    names = (f"{prefix}goal", f"{prefix}solved")
    meshes = [
        {
            "source": str(out / f"{variant}-{angle:03d}.obj"),
            "shots": [
                make_shot(out / f"{variant}-{angle:03d}-{view}-{look}.png", rig, "region", view, look)
                for view, look in VIEWS
            ],
        }
        for variant in names
        for angle in angles
    ]
    manifest = out / "render.json"
    manifest.write_text(json.dumps({"meshes": meshes}, indent=2))
    with (out / "render.log").open("w") as log:
        subprocess.run(
            [
                str(find_blender()),
                "-b",
                "--factory-startup",
                "-t",
                "2",
                "--python-exit-code",
                "1",
                "--python",
                str(ROOT / "blender/review/render.py"),
                "--",
                str(manifest),
            ],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    rows = [
        [
            (f"{label} {angle}° · {view} {look}", out / f"{variant}-{angle:03d}-{view}-{look}.png")
            for angle in angles
            for variant, label in zip(names, ("求解前", "接触后"), strict=True)
        ]
        for view, look in VIEWS
    ]
    grid(
        out / f"{prefix}overview.png",
        f"{name} · 腹部与大腿根接触",
        rows,
        "已选雕塑是目标，接触只在会穿过时移动皮肤。",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", metavar="name")
    parser.add_argument("--path", choices=("double", "single"), default="double")
    parser.add_argument("--angle", type=float, default=100.0)
    parser.add_argument("--step", type=float, default=5.0)
    parser.add_argument("--render", help="comma list of saved angles to draw, instead of solving")
    parser.add_argument("--table", help="comma list of angles to draw from the delivered table")
    args = parser.parse_args()
    if args.table:
        angles = [int(a) for a in args.table.split(",")]
        out = BUILD / "reports" / "groin-contact-study" / args.preset / "table"
        from_table(args.preset, out, angles)
        for mode in ("double", "single"):
            render(args.preset, out, angles, f"{mode}-")
        return
    out = BUILD / "reports" / "groin-contact-study" / args.preset / args.path
    if args.render:
        render(args.preset, out, [int(a) for a in args.render.split(",")])
    else:
        run(args.preset, args.path, args.angle, args.step, out)


if __name__ == "__main__":
    main()
