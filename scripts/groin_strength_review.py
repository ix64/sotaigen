"""Lighter static groins beside the accepted one, standing and in hip flexion, on one fixed rig."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sotaigen.contact import groin as groin_contact
from sotaigen.corrective import groin
from sotaigen.form import groin as groin_rest
from sotaigen.mesh import write_obj
from sotaigen.pipeline import BUILD, rigged
from sotaigen.rig.skinning import pose_verts
from sotaigen_dev.motion.groin_sheet import grid, make_shot
from sotaigen_dev.review import find_blender

VARIANTS = {
    "accepted": ("默认", 1.0, 1.0),
    "half": ("两项减半", 0.5, 0.5),
    "no-valley": ("沟 0，起伏保留", 0.0, 1.0),
    "no-relief": ("沟保留，起伏 0", 1.0, 0.0),
    "none": ("两项都 0", 0.0, 0.0),
}
SHOTS = {
    0: [("front", "clay"), ("threequarter", "clay"), ("front", "toon"), ("threequarter", "toon")],
    70: [("high", "clay")],
    100: [("high", "clay"), ("threequarter", "toon")],
}


def run(name):
    rig = rigged(name)
    reference = np.load(BUILD / "bodies" / name / "wrapped.npy")
    out = BUILD / "reports" / "groin-strength-review" / name
    out.mkdir(parents=True, exist_ok=True)
    meshes = []
    for variant, (_, depth, relief) in VARIANTS.items():
        L = rig.L | {"groinDepth": groin_rest.DEPTH * depth, "groinRelief": groin_rest.RELIEF * relief}
        bones = [dict(bone) for bone in rig.bones]
        groin.record_amounts(bones, L)
        rest = groin_rest.rest_verts(rig.base, reference, L)
        targets = groin_contact.targets(rig.base, rest, bones, rig.weights)
        for angle, shots in SHOTS.items():
            pose = groin.sample_pose(angle, angle)
            posed = pose_verts(bones, rig.weights, groin.bind_verts(rest, targets, pose), pose)
            source = out / f"{variant}-{angle:03d}.obj"
            write_obj(source, posed, rig.base["faces"])
            meshes.append(
                {
                    "source": str(source),
                    "shots": [
                        make_shot(out / f"{variant}-{angle:03d}-{view}-{look}.png", rig, "region", view, look)
                        for view, look in shots
                    ],
                }
            )
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
    compose(name)


def compose(name):
    out = BUILD / "reports" / "groin-strength-review" / name
    rows = []
    for angle, shots in SHOTS.items():
        for view, look in shots:
            rows.append(
                [
                    (f"{label} · 屈髋 {angle}° · {view} {look}", out / f"{variant}-{angle:03d}-{view}-{look}.png")
                    for variant, (label, _, _) in VARIANTS.items()
                ]
            )
    grid(
        out / "overview.png",
        f"{name} · 静态腹股沟强度",
        rows,
        "同一骨架和权重；只改静态的沟深与邻近起伏，屈髋动作造型不变。屈髋为双腿。",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("presets", nargs="+")
    parser.add_argument("--compose-only", action="store_true")
    args = parser.parse_args()
    for name in args.presets:
        compose(name) if args.compose_only else run(name)


if __name__ == "__main__":
    main()
