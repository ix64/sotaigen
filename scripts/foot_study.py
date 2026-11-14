"""Look at one foot from the sides a body sheet can't: the inner side and the sole.

uv run scripts/foot_study.py [body ...] [--against DIR] [--views inner,under] [--cell-px 460]
    [-o out.png] [--no-render]

Prints the foot's profile as shares of its length: the sole's height along the inner edge, the
dorsum's height, and the footprint's arch index. Renders the left leg alone below the mid shin, so
the other foot can't hide the arch. Reads wrapped.npy, so build first.
"""

import argparse
import itertools
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sotaigen.form.head import extended
from sotaigen.mesh import triangulate, write_obj
from sotaigen.output.blender import find_blender
from sotaigen.pipeline import BUILD, bodies_of
from sotaigen.template import load_template

ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "blender" / "review" / "render.py"

# Directions to the camera for the left foot, whose inner side faces -X.
VIEWS = {
    "inner": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "outer": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "front": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "back": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "under": ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
    "inner3q": ((-0.72, -0.62, 0.32), (0.0, 0.0, 1.0)),
}
CONTACT_MM_PER_LENGTH = 0.012


def profile(points, is_foot):
    """The foot's outline by station from the toe tip (0) to the heel (1), as shares of its length."""
    foot = points[is_foot]
    length = np.ptp(foot[:, 1])
    y_toe = foot[:, 1].min()
    floor = foot[:, 2].min()
    rows = []
    for start in np.arange(0.0, 1.0, 0.1):
        band = foot[(foot[:, 1] >= y_toe + start * length) & (foot[:, 1] < y_toe + (start + 0.1) * length)]
        if not len(band):
            continue
        third = band[:, 0].min() + np.ptp(band[:, 0]) / 3
        inner = band[band[:, 0] <= third]
        outer = band[band[:, 0] >= band[:, 0].max() - np.ptp(band[:, 0]) / 3]
        rows.append(
            {
                "station": round(float(start + 0.05), 2),
                "innerSole": round(float((inner[:, 2].min() - floor) / length * 100), 1),
                "outerSole": round(float((outer[:, 2].min() - floor) / length * 100), 1),
                "dorsum": round(float((band[:, 2].max() - floor) / length * 100), 1),
                "width": round(float(np.ptp(band[:, 0]) / length * 100), 1),
            }
        )
    return rows


def arch_index(points, triangles, is_foot):
    """Cavanagh and Rodgers' ratio: contact area of the middle third over the toeless footprint."""
    foot = points[is_foot]
    length = np.ptp(foot[:, 1])
    floor = foot[:, 2].min()
    keep = is_foot[triangles].all(1)
    a, b, c = (points[triangles[keep, i]] for i in range(3))
    normal = np.cross(b - a, c - a)
    down = normal[:, 2] < 0
    low = np.maximum.reduce((a[:, 2], b[:, 2], c[:, 2])) - floor < CONTACT_MM_PER_LENGTH * length
    area = np.abs(normal[:, 2]) / 2
    station = ((a + b + c)[:, 1] / 3 - foot[:, 1].min()) / length
    contact = down & low
    # The toes are taken as the front quarter of the length.
    span = np.linspace(0.25, 1.0, 4)
    thirds = [area[contact & (station >= s) & (station < e)].sum() for s, e in itertools.pairwise(span)]
    total = sum(thirds)
    return round(float(thirds[1] / total), 3) if total else None


def left_lower_leg(points, faces, is_foot):
    """The left leg below the mid shin, as an OBJ's points and faces."""
    ankle_z = points[is_foot, 2].max()
    keep = (points[:, 0] > 0) & (points[:, 2] < ankle_z * 2.0)
    remap = -np.ones(len(points), int)
    remap[keep] = np.arange(keep.sum())
    kept = [[int(remap[i]) for i in face] for face in faces if keep[list(face)].all()]
    return points[keep], kept


def require_blender():
    """Blender as body.blend finds it: BLENDER, then PATH, then the macOS app."""
    blender = find_blender()
    if blender is None:
        sys.exit("Blender was not found: put blender on PATH or set BLENDER=/path/to/blender")
    return blender


def render(meshes, out, tmp, views, cell_px):
    """Render each (label, points, faces) from the views, at one scale."""
    span = max(np.ptp(m[1], axis=0).max() for m in meshes) * 1.18
    mm_per_px = span / cell_px
    manifest = []
    paths = {}
    for row, (_label, points, faces) in enumerate(meshes):
        source = Path(tmp) / f"mesh{row}.obj"
        write_obj(source, points, faces)
        center = (points.min(0) + points.max(0)) / 2
        shots = []
        for view in views:
            toward, up = (np.array(v) for v in VIEWS[view])
            toward = toward / np.linalg.norm(toward)
            up = up - (up @ toward) * toward
            up = up / np.linalg.norm(up)
            target = Path(tmp) / f"{row}-{view}.png"
            paths[row, view] = target
            shots.append(
                {
                    "out": str(target),
                    "look": "clay",
                    "camera": "ortho",
                    "right": np.cross(up, toward).tolist(),
                    "up": up.tolist(),
                    "toward": toward.tolist(),
                    "center": center.tolist(),
                    "width": cell_px,
                    "height": cell_px,
                    "mm_per_px": mm_per_px,
                    "wire": False,
                }
            )
        manifest.append({"source": str(source), "shots": shots})
    listing = Path(tmp) / "manifest.json"
    listing.write_text(json.dumps({"meshes": manifest}))
    subprocess.run(
        [str(require_blender()), "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(RENDERER), "--", str(listing)],
        check=True,
        capture_output=True,
    )
    sheet = Image.new("RGB", (cell_px * len(views), (cell_px + 22) * len(meshes)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (label, _points, _faces) in enumerate(meshes):
        for column, view in enumerate(views):
            shot = Image.open(paths[row, view]).convert("RGBA")
            x, y = column * cell_px, row * (cell_px + 22) + 22
            sheet.paste(shot, (x, y), shot)
            draw.text((x + 6, y - 18), f"{label} / {view}", fill="black")
    sheet.save(out)
    print(out, f"{mm_per_px:.2f} mm/px")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("presets", nargs="*", default=["female-long"])
    parser.add_argument("--against", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=ROOT / "build" / "reports" / "foot-study.png")
    parser.add_argument("--views", default=",".join(VIEWS))
    parser.add_argument("--cell-px", type=int, default=460)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    full = extended(load_template())
    faces = full["faces"]
    triangles = np.asarray(triangulate(faces, np.asarray(full["verts"])))
    regions = np.asarray(full["regions"])
    meshes = []
    for name in args.presets:
        sources = [("now", BUILD / "bodies")]
        if args.against:
            sources.insert(0, ("before", bodies_of(args.against)))
        for label, folder in sources:
            points = np.load(folder / name / "wrapped.npy")
            is_foot = (regions == "foot") & (points[:, 0] > 0)
            print(f"{name} {label}: archIndex {arch_index(points, triangles, is_foot)}")
            for row in profile(points, is_foot):
                print("  ", row)
            meshes.append((f"{name} {label}", *left_lower_leg(points, faces, is_foot)))
    if not args.no_render:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            render(meshes, args.out, tmp, args.views.split(","), args.cell_px)


if __name__ == "__main__":
    main()
