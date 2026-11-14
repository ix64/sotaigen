"""A body's file with another rest pose against the A pose's: how far apart they pose, and what it looks like.

uv run scripts/rest_pose_study.py measure [preset ...] [--jobs 2]
uv run scripts/rest_pose_study.py sheet <name> [--presets a,b] [--look clay,toon]

`measure` writes each body's file with the T pose, and with the T pose and the ankles 0.75 of the hip
width apart (sotaigen.rig.rest_pose), opens each in a fresh Blender with factory settings, turns its new controls
to the A pose and to every check and compound pose, limits off and on, and compares the skin with
sotaigen.rig.skinning.pose_verts and the correctives as tests/test_blend.py does for the A pose's file. Writes
build/reports/rest-pose/measure.json and summary.txt. `sheet` renders the new rest's skin as the file holds it,
and its poses next to the A pose's (sheet_mesh), with sotaigen_dev/review.py's looks.

Nothing here bakes a contact table.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sotaigen.corrective.recipe import Correctives
from sotaigen.form.anatomy import landmarks
from sotaigen.form.bodies import load_bodies, load_specs
from sotaigen.mesh import write_obj
from sotaigen.output.blend import blend_body
from sotaigen.output.blender import run_blender, script
from sotaigen.output.face import face_parts
from sotaigen.output.glb import driver_table
from sotaigen.pipeline import BUILD, rigged, surface
from sotaigen.rig.limits import clipped, joint_limits
from sotaigen.rig.poses import check_poses
from sotaigen.rig.rest_pose import T_POSE, asset_reference, in_new_rest, solve, world_frames
from sotaigen.rig.skinning import pose_verts
from sotaigen.rig.touch import blend_poses, solve_all
from sotaigen.uv import export_fields

OUT = BUILD / "reports" / "rest-pose"
TYPICAL = list(load_specs("typical"))
T = {"arm_abduction": T_POSE["armAbduction"], "palms_down": True, "straight_arms": True}
# The compound poses of tests/test_blend.py, repeated rather than imported from the tests.
COMPOUND_POSES = {
    "armCompound": {"upperArm": (40, 30, 50), "lowerArm": (60, 0, 0), "hand": (20, 45, 10)},
    "armBehind": {"upperArm": (-40, -30, 20)},
    "armForwardUp": {"upperArm": (50, -15, 60), "shoulder": (0, 0, 8)},
    "armTwisted": {"upperArm": (60, 70, 72)},
    "hangBetweenKeys": {"upperArm": (12.5, 5, -27), "lowerArm": (8, 0, 0)},
    "asymmetric": {
        "leftUpperArm": (95, 0, 40),
        "rightUpperArm": (10, -20, -24),
        "leftUpperLeg": (95, 10, 5),
        "rightUpperLeg": (20, -15, -10),
    },
    "sit": {"upperLeg": (85, 15, 12), "lowerLeg": (-90, 0, 0), "spine": (5, 7, 0)},
}


def hip_width(name):
    p = load_bodies()[name]
    return float(p["hipBreadth"] * landmarks(p)["T"])


def requests(name):
    return {"t": T, "tWide": {**T, "ankle_spacing": 0.75 * hip_width(name)}}


def hip_band_width(name, verts, regions):
    """The widest the skin is across the hips, as the shelved stance probe measured the hip width: from
    0.05 T under the crotch to 0.25 T over it, here off the mesh, without the arms."""
    L = landmarks(load_bodies()[name])
    z = verts[:, 2]
    keep = (z > L["crotch"] - 0.05 * L["T"]) & (z < L["crotch"] + 0.25 * L["T"]) & ~np.isin(regions, ("arm", "hand"))
    return float(2 * np.abs(verts[keep, 0]).max())


def reference(rig, correctives, pose):
    return pose_verts(rig.bones, rig.weights, correctives.bind(rig.verts, pose), pose)


def distances(a, b):
    return np.linalg.norm(np.asarray(a, float) - b, axis=1)


def measure_one(name):
    rig = rigged(name)
    base = {**rig.base, **export_fields(rig.base["faces"])}
    correctives = Correctives.production(base, rig.verts, rig.bones, rig.weights, field=surface(name)[0])
    limits = joint_limits(rig.bones)
    central = driver_table(rig.bones)["central"]
    poses = {**check_poses(base, rig.verts, rig.bones, rig.weights), **COMPOUND_POSES}
    regions = np.asarray(base["regions"])
    parts = face_parts(name)
    folder = OUT / "files" / name
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for label, request in requests(name).items():
        path = folder / f"{label}.blend"
        blend_body(name, path, request)
        rest = solve(rig.bones, limits, **request)
        turned = {**{k: in_new_rest(rest["pose"], v) for k, v in poses.items()}, "newRest": {}}
        ask = folder / f"{label}-poses.json"
        ask.write_text(json.dumps({"poses": turned, "central": central, "limits": [False, True], "actions": True}))
        run_blender(script("pose"), ask, folder / f"{label}-posed", blend=path)
        with np.load(folder / f"{label}-posed.npz") as data, open(folder / f"{label}-posed.json") as f:
            info = json.load(f)
            held = {k: tuple(v) for k, v in rest["pose"].items()}
            row = {
                "file": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "pose": rest["pose"],
                "measured": rest["measured"],
                "unclean": rest["unclean"],
                "pastLimits": rest["pastLimits"],
                "drivers": info["drivers"],
                "notSimple": info["notSimple"],
                "autoexec": info["autoexec"],
                "autoexecFail": info["autoexecFail"],
                "texts": info["texts"],
                "constraints": info["constraints"],
                "errorInFile": info["info"]["restPose"]["errorMm"],
            }
            want_rest = reference(rig, correctives, held)
            row["restMesh"] = float(distances(data["rest:body"], want_rest).max())
            row["hipBandMm"] = [hip_band_width(name, rig.verts, regions), hip_band_width(name, data["rest:body"].astype(float), regions)]
            row["newRest"] = float(distances(data["off:newRest:body"], want_rest).max())
            row["restKeyValues"] = float(np.abs(data["off:newRest:keys"]).max())
            row["assets"] = {}
            for asset, pose in blend_poses(solve_all(base, rig.verts, rig.bones, rig.weights)).items():
                given = asset_reference(rest["pose"], pose)
                d = distances(data[f"off:{asset}:body"], reference(rig, correctives, given))
                frames = world_frames(rig.bones, given)
                bones = [
                    np.linalg.norm(data[f"off:{asset}:bones"][i][:3, 3] * 1000 - frames[bone][:3, 3])
                    for i, bone in enumerate(info["deform"])
                    if bone != "root"
                ]
                row["assets"][asset] = {"max": float(d.max()), "p99": float(np.percentile(d, 99)), "bones": float(max(bones))}
            row["poses"] = {}
            for pose_name, pose in poses.items():
                for limit in ("off", "on"):
                    given = clipped(pose, limits) if limit == "on" else pose
                    d = distances(data[f"{limit}:{pose_name}:body"], reference(rig, correctives, given))
                    part_error = 0.0
                    for part in parts:
                        weights = part.weights
                        if weights is None:
                            weights = np.zeros((len(part.verts), len(rig.bones)))
                            weights[:, [b["name"] for b in rig.bones].index("head")] = 1.0
                        part_error = max(
                            part_error,
                            float(distances(data[f"{limit}:{pose_name}:{part.name}"], pose_verts(rig.bones, weights, np.asarray(part.verts, float), given)).max()),
                        )
                    row["poses"][f"{limit}:{pose_name}"] = {
                        "max": float(d.max()),
                        "p99": float(np.percentile(d, 99)),
                        "where": str(regions[int(d.argmax())]),
                        "parts": part_error,
                    }
        out[label] = row
    return name, out


def summary(rows):
    lines = []
    for name, row in rows.items():
        for label, one in row.items():
            a = one["poses"]["off:rest"]
            al = one["poses"]["on:rest"]
            others = {k: v for k, v in one["poses"].items() if k.startswith("off:") and k != "off:rest"}
            big = max(others.items(), key=lambda kv: kv[1]["max"])
            parts = max(v["parts"] for v in one["poses"].values())
            lines.append(
                f"{name:16s} {label:6s} newRest {one['newRest']:.4f} basis {one['restMesh']:.4f} "
                f"A {a['max']:.4f}/{al['max']:.4f} parts {parts:.4f} "
                f"far {big[1]['max']:.2f} ({big[0][4:]}, {big[1]['where']}) "
                f"arm {one['pose'].get('upperArm', [0, 0, 0])[2]:.2f} "
                f"forearm {one['measured']['forearmOffUpperArm']:.2f} palm {one['measured']['palmOffDown']:.2f} "
                f"ankle {one['measured']['ankleSpacing']:.1f} "
                f"keys0 {one['restKeyValues']:.1e} notSimple {len(one['notSimple'])} unclean {len(one['unclean'])}"
            )
    table = {}
    for row in rows.values():
        for label, one in row.items():
            for pose_name, values in one["poses"].items():
                if pose_name.startswith("off:"):
                    entry = table.setdefault((label, pose_name[4:]), [0.0, 0.0, 0.0, ""])
                    if values["max"] > entry[0]:
                        entry[0], entry[3] = values["max"], values["where"]
                    entry[1] = max(entry[1], values["p99"])
                    entry[2] = max(entry[2], values["parts"])
    lines.append("")
    for (label, pose_name), (largest, p99, parts, where) in table.items():
        lines.append(f"{label:6s} {pose_name:18s} max {largest:7.3f} p99 {p99:7.3f} parts {parts:.4f} {where}")
    return "\n".join(lines) + "\n"


def cmd_measure(args):
    names = args.names or TYPICAL
    with ThreadPoolExecutor(args.jobs) as pool:
        rows = dict(pool.map(measure_one, names))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "measure.json").write_text(json.dumps(rows, indent=1))
    text = summary(rows)
    (OUT / "summary.txt").write_text(text)
    print(text)
    return 0


def sheet_mesh(name, label):
    """The skin a sheet's label names: `t` the file's own new rest as pose.py read it, `t~reach` that file
    posed to a check pose, `a~reach` the A pose's skin in that pose, `a~side55` raised 55 to the side."""
    file, _, pose = label.partition("~")
    if file == "a":
        rig = rigged(name)
        base = {**rig.base, **export_fields(rig.base["faces"])}
        correctives = Correctives.production(base, rig.verts, rig.bones, rig.weights, field=surface(name)[0])
        poses = {**check_poses(base, rig.verts, rig.bones, rig.weights), **COMPOUND_POSES, "side55": {"upperArm": (0, 0, 55)}}
        return reference(rig, correctives, poses[pose])
    with np.load(OUT / "files" / name / f"{file}-posed.npz") as data:
        return data[f"off:{pose}:body" if pose else "rest:body"].astype(float)


def cmd_sheet(args):
    from sotaigen_dev import review

    presets = args.presets.split(",") if args.presets else TYPICAL
    labels = args.labels.split(",")
    root = OUT / "sheets-input"
    for name in presets:
        rig = rigged(name)
        folder = root / name / "poses"
        folder.mkdir(parents=True, exist_ok=True)
        for label in labels:
            write_obj(folder / f"{label}.obj", sheet_mesh(name, label), rig.base["faces"])
        # review checks the build's fingerprint and frames a focus by the skeleton.
        for keep in ("fingerprint.json", "skeleton.json", "weights.npz", "wrap.json"):
            source = BUILD / "bodies" / name / keep
            if source.exists():
                (root / name / keep).write_bytes(source.read_bytes())
    review.BUILD = root
    review.POSES = {label: {} for label in labels}
    axes = review.select_axes(presets, pose=",".join(labels), view=args.view, focus=args.focus, look=args.look)
    rows, cols = (("look", "preset"), ("pose", "view"))
    out = OUT / "sheets" / f"{args.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    review.make_sheet(axes, rows=rows, cols=cols, cell_px=args.cell, out=out)
    print(out)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    measure = commands.add_parser("measure")
    measure.add_argument("names", nargs="*")
    measure.add_argument("--jobs", type=int, default=2)
    measure.set_defaults(run=cmd_measure)
    sheet = commands.add_parser("sheet")
    sheet.add_argument("name")
    sheet.add_argument("--presets")
    sheet.add_argument("--labels", default="t")
    sheet.add_argument("--look", default="clay,toon")
    sheet.add_argument("--view", default="front,threequarter,back")
    sheet.add_argument("--focus")
    sheet.add_argument("--cell", type=int, default=360)
    sheet.set_defaults(run=cmd_sheet)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
