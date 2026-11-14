"""Pelvic observation backend for motion_sheet; never changes the rig or skin."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sotaigen.contact import groin as groin_contact
from sotaigen.corrective import groin
from sotaigen.mesh import triangulate, unit_normals, write_obj
from sotaigen.pipeline import BUILD, ROOT, fingerprints, rigged
from sotaigen.rig.skinning import pose_skins, pose_verts, rest_matrix
from sotaigen_dev.motion.cards import ANCHOR, Hinge, limb_coordinates, plane_section, sweep_poses
from sotaigen_dev.pictures import font

BACKGROUND = (240, 241, 242)
VIEWS = {"front": (0, -1, 0), "threequarter": (0.64, -0.77, 0.08), "high": (0.64, -0.77, 0.8)}


def grid(path, title, rows, note):
    """Rows of labelled images on one sheet, under a title and above a note; rows must be as wide."""
    widths, heights = [], []
    for row in rows:
        dimensions = []
        for _, p in row:
            with Image.open(p) as im:
                dimensions.append(im.size)
        widths.append(sum(w for w, _ in dimensions))
        heights.append(max(h for _, h in dimensions) + 46)
    if len(set(widths)) != 1:
        raise ValueError("all rows must have equal widths")
    canvas = Image.new("RGB", (widths[0], 115 + sum(heights) + 66), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), title, font=font(29), fill=(30, 38, 48))
    draw.text(
        (24, 65),
        "同骨架、同权重、同机位；髋局部输入，膝输入取反；骨盆与躯干固定。",
        font=font(20),
        fill=(70, 78, 88),
    )
    y = 115
    for row, height in zip(rows, heights):
        x = 0
        for label, p in row:
            draw.text((x + 16, y + 10), label, font=font(22), fill=(30, 38, 48))
            with Image.open(p) as im:
                rgba = im.convert("RGBA")
                canvas.paste(rgba, (x, y + 46), rgba)
                x += im.width
        y += height
    draw.text((24, y + 18), note, font=font(19), fill=(70, 78, 88))
    canvas.save(path)


def frame_of(rig):
    """The hip's flexion axis, with pelvic up; no femur-based recentering between poses."""
    names = [b["name"] for b in rig.bones]
    p, t = (names.index(name) for name in ("hips", "leftUpperLeg"))
    groin, thigh = rig.bones[p], rig.bones[t]
    centre = np.array(thigh["head"])
    upper = np.array(groin["tail"]) - groin["head"]
    lower = np.array(thigh["tail"]) - centre
    length = np.linalg.norm(lower)
    axis = rest_matrix(thigh)[:3, 0]
    if axis[0] < 0:
        axis = -axis
    return Hinge(
        centre,
        axis,
        upper / np.linalg.norm(upper),
        lower / length,
        float(length),
        p,
        t,
    )


def reference_columns(base, bones):
    """Use explicit rest template segments, never a point's height relative to the hip."""
    names = [b["name"] for b in bones]
    segments = np.asarray(base["segments"])
    columns = np.full(len(segments), -1, int)
    columns[segments == "trunk"] = names.index("hips")
    left = np.asarray(base["verts"])[:, 0] >= 0
    for side, mask in (("left", left), ("right", ~left)):
        columns[(segments == "upperLeg") & mask] = names.index(f"{side}UpperLeg")
    return columns


def relative_displacement(rest, posed, skins, columns, normals):
    """Normal/tangential displacement in each selected rigid reference; not tissue slip."""
    lift = np.full(len(rest), np.nan)
    tangent = np.full(len(rest), np.nan)
    for column in np.unique(columns[columns >= 0]):
        chosen = columns == column
        back = (
            np.column_stack((posed[chosen], np.ones(chosen.sum()))) @ np.linalg.inv(skins[column]).T
        )
        delta = back[:, :3] - rest[chosen]
        normal = np.sum(delta * normals[chosen], axis=1)
        lift[chosen] = normal
        tangent[chosen] = np.linalg.norm(delta - normal[:, None] * normals[chosen], axis=1)
    return lift, tangent


def behaviour(path, rig, hinge, frames, columns, triangles):
    # Reuse the elbow tool's plotting primitives, not its arm-only framing or ownership rule.
    from sotaigen_dev.motion.sheet import (
        draw_heat,
        draw_section,
        lift_colour,
        moved_by,
        slide_colour,
    )

    cell, gutter, top, bottom = 320, 170, 52, 110
    labels = ["左髋轴截面", "中央截面", "法向位移 · 正面", "切向位移 · 正面"]
    im = Image.new(
        "RGB", (gutter + cell * len(frames), top + cell * len(labels) + bottom), BACKGROUND
    )
    draw = ImageDraw.Draw(im)
    keep = (columns[triangles] >= 0).all(axis=1)
    whole = triangles[keep]
    half_mm = hinge.length * 0.70
    centre_x = float(hinge.canonical(np.array([[0.0, *hinge.centre[1:]]]))[0, 0])
    for col, (name, data) in enumerate(frames.items()):
        x = gutter + col * cell
        draw.text((x + 14, 15), name, font=font(20), fill=(30, 35, 45))
        posed, skins, lift, tangent = data
        canonical = hinge.canonical(posed)
        for row, cut in enumerate((0.0, centre_x)):
            rigid = []
            for bone in np.unique(columns[columns >= 0]):
                tris = triangles[(columns[triangles] == bone).all(axis=1)]
                rigid.append(
                    plane_section(hinge.canonical(moved_by(skins[bone], rig.verts)), tris, cut)
                    - ANCHOR[1:]
                )
            actual = plane_section(canonical, whole, cut) - ANCHOR[1:]
            draw_section(im, (x, top + row * cell, cell), half_mm, actual, rigid)
        for row, values, colour in ((2, lift, lift_colour), (3, tangent, slide_colour)):
            draw_heat(
                im,
                (x, top + row * cell, cell),
                half_mm,
                canonical,
                whole,
                np.nan_to_num(values),
                colour,
                "front",
            )
    for i, label in enumerate(labels):
        draw.text((10, top + i * cell + 14), label, font=font(18), fill=(30, 35, 45))
    y = top + cell * len(labels) + 12
    notes = [
        "黑：当前皮肤；蓝：躯干随骨盆、大腿随各自股骨刚性移动。蓝线不是生理目标，分段接缝处可中断。",
        "红 / 蓝：相对各自参照的向外 / 向内位移（满色 ±12 mm）；紫：切向分量（满色 25 mm）。颜色饱和不是阈值。",
        "热图隐去小腿，开口来自分段截断；整体灰模保留全身。模板分段处参照改变会产生颜色分界；这些量不是筋膜滑移。",
    ]
    for i, note in enumerate(notes):
        draw.text((12, y + i * 29), note, font=font(18), fill=(50, 60, 70))
    im.save(path)


def make_shot(out, rig, focus, view, look):
    scale = rig.L["H"] / 1400.0
    if focus == "full":
        width, height, mm = 480, 720, 2.15 * scale
        centre = [0.0, -65.0 * scale, rig.L["H"] * 0.507]
    else:
        width, height, mm = 600, 600, 1.0 * scale
        centre = [0.0, -60.0 * scale, rig.L["crotch"] + 0.105 * rig.L["T"]]
    toward = np.array(VIEWS[view], float)
    toward /= np.linalg.norm(toward)
    up = np.array([0.0, 0.0, 1.0]) - toward[2] * toward
    up /= np.linalg.norm(up)
    return {
        "width": width,
        "height": height,
        "mm_per_px": mm,
        "center": centre,
        "toward": toward.tolist(),
        "up": up.tolist(),
        "right": np.cross(up, toward).tolist(),
        "camera": "ortho",
        "look": look,
        "wire": False,
        "studio_light": "studio.sl",
        "out": str(out),
    }


def board(folder, sweep, preset, name, title, cells, prior=None):
    """Comparable panels use the same framing; whole and regional scales are labelled."""
    len(cells)
    width = sum(Image.open(p).width for _, p in cells[0])
    heights = [max(Image.open(p).height for _, p in row) + 42 for row in cells]
    im = Image.new("RGB", (width, 96 + sum(heights) + 52), BACKGROUND)
    draw = ImageDraw.Draw(im)
    draw.text((20, 14), f"{preset} · {sweep} · {title}", font=font(27), fill=(30, 35, 45))
    draw.text(
        (20, 54),
        "局部髋输入；膝为负髋角。骨盆、躯干固定。姿势间同机位；不按褶皱数量判断好坏。",
        font=font(19),
        fill=(65, 75, 85),
    )
    y = 96
    for row, h in zip(cells, heights):
        x = 0
        for label, p in row:
            draw.text((x + 15, y + 9), label, font=font(21), fill=(30, 35, 45))
            with Image.open(p) as src:
                rgba = src.convert("RGBA")
                im.paste(rgba, (x, y + 42), rgba)
                x += src.width
        y += h
    draw.text(
        (20, y + 12),
        "全身、整体灰模保留完整模型；角度范围是观察采样，不是生理阈值。",
        font=font(19),
        fill=(65, 75, 85),
    )
    path = folder / name
    im.save(path)
    if prior is not None:
        old = prior / preset / name
        if not old.exists():
            raise ValueError(f"missing comparison image: {old}")
        with Image.open(old) as before:
            if before.size != im.size:
                raise ValueError("comparison framing differs; regenerate with the same card")
            pair = Image.new("RGB", (im.width * 2, im.height + 40), BACKGROUND)
            pair.paste(before, (0, 40))
            pair.paste(im, (im.width, 40))
            d = ImageDraw.Draw(pair)
            d.text((20, 8), "BEFORE", font=font(22), fill=(30, 35, 45))
            d.text((im.width + 20, 8), "AFTER", font=font(22), fill=(30, 35, 45))
            pair.save(folder / f"compare-{name}")


def check_comparison(current, previous):
    """Do not label images comparable when cameras, scales, lighting, or poses differ."""

    def layout(manifest):
        return [
            (
                Path(mesh["source"]).stem,
                [
                    {key: value for key, value in shot.items() if key != "out"}
                    for shot in mesh["shots"]
                ],
            )
            for mesh in manifest["meshes"]
        ]

    if layout(current) != layout(previous):
        raise ValueError("comparison cameras, scale, lighting, or sampled poses differ")


def run(card, args):
    from sotaigen_dev.review import find_blender

    looks = args.look.split(",")
    if any(look not in {"clay", "toon"} for look in looks):
        raise ValueError("pelvic cards support clay,toon")
    blender = find_blender() if not args.no_render else None
    if not args.no_render and not blender:
        raise RuntimeError("Blender is required for pelvic sweep renders")
    for preset in args.presets:
        rig = rigged(preset)
        pelvic = groin_contact.targets(rig.base, rig.verts, rig.bones, rig.weights)
        folder = BUILD / "reports" / "motion" / card["region"] / preset
        folder.mkdir(parents=True, exist_ok=True)
        hinge = frame_of(rig)
        columns = reference_columns(rig.base, rig.bones)
        rest = np.asarray(rig.verts)
        triangles = triangulate(rig.base["faces"], rest)
        normals = unit_normals(rest, triangles)
        # Sampling coordinates only; reference ownership always follows explicit segments.
        axial, azimuth = limb_coordinates(hinge, card, rig.base, rig.verts)
        meshes, all_frames = [], {}
        for sweep in card["sweeps"]:
            frames = {}
            for name, pose in sweep_poses(sweep).items():
                skins = pose_skins(rig.bones, pose)
                posed = pose_verts(
                    rig.bones, rig.weights, groin.bind_verts(rig.verts, pelvic, pose), pose
                )
                if not np.isfinite(posed).all():
                    raise ValueError(f"nonfinite pose: {preset}/{name}")
                lift, tangent = relative_displacement(rig.verts, posed, skins, columns, normals)
                frames[name] = (posed, skins, lift, tangent)
                source = folder / f"{name}.obj"
                write_obj(source, posed, rig.base["faces"])
                shots = []
                for look in looks:
                    for view in VIEWS:
                        output = folder / f"{name}-region-{view}-{look}.png"
                        shots.append(make_shot(output, rig, "region", view, look))
                if int(name[-3:]) in (0, 40, 100):
                    output = folder / f"{name}-full.png"
                    shots.append(make_shot(output, rig, "full", "threequarter", "clay"))
                meshes.append({"source": str(source), "shots": shots})
            behaviour(
                folder / f"behaviour-{sweep['name']}.png", rig, hinge, frames, columns, triangles
            )
            all_frames.update(frames)
        report = {
            "source_main": subprocess.check_output(
                ["git", "merge-base", "HEAD", "main"], cwd=ROOT, text=True
            ).strip(),
            "rig_fingerprint": fingerprints()["rig"],
            "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "card_sha256": hashlib.sha256(
                (ROOT / "docs/motion" / f"{card['region']}.toml").read_bytes()
            ).hexdigest(),
            "skeleton_sha256": hashlib.sha256(
                (BUILD / "bodies" / preset / "skeleton.json").read_bytes()
            ).hexdigest(),
            "weights_sha256": hashlib.sha256(rig.weights.tobytes()).hexdigest(),
            "poses": {name: pose for s in card["sweeps"] for name, pose in sweep_poses(s).items()},
            "reference": "Rest template trunk -> hips; upperLeg -> same-side UpperLeg. "
            "Normal and tangential rigid-reference displacements, NOT tissue slip.",
        }
        (folder / "card-snapshot.json").write_text(json.dumps(card, ensure_ascii=False, indent=2))
        (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(
            folder / "observations.npz",
            rest=rig.verts,
            reference=columns,
            axial=axial,
            azimuth=azimuth,
            **{f"{k}_posed": v[0] for k, v in all_frames.items()},
            **{f"{k}_normal": v[2] for k, v in all_frames.items()},
            **{f"{k}_tangent": v[3] for k, v in all_frames.items()},
        )
        manifest = folder / "render.json"
        if args.against:
            prior = Path(args.against) / preset
            if prior.resolve() == folder.resolve():
                raise ValueError("comparison must use a separate saved run")
            check_comparison({"meshes": meshes}, json.loads((prior / "render.json").read_text()))
        manifest.write_text(json.dumps({"meshes": meshes}, indent=2))
        if blender:
            with (folder / "render.log").open("w") as log:
                subprocess.run(
                    [
                        str(blender),
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
            for sweep in card["sweeps"]:
                key = sweep["name"]
                prior = Path(args.against) if args.against else None
                board(
                    folder,
                    key,
                    preset,
                    f"whole-{key}.png",
                    "全身关系",
                    [[(f"{a}°", folder / f"{key}{a:03d}-full.png") for a in (0, 40, 100)]],
                    prior,
                )
                for view in VIEWS:
                    board(
                        folder,
                        key,
                        preset,
                        f"sweep-{key}-{view}.png",
                        f"整体观察 / {view}",
                        [
                            [
                                (
                                    f"{a}° / {look}",
                                    folder / f"{key}{a:03d}-region-{view}-{look}.png",
                                )
                                for a in sweep["angles"]
                            ]
                            for look in looks
                        ],
                        prior,
                    )
        print(folder, flush=True)
    return 0
