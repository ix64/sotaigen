"""Judge the UV atlas: overlaps and flips once, stretch and texel density per body and region.

uv run scripts/uv_check.py [body ...] [--atlas <png>] [--texture-px 2048]
"""

import argparse

import numpy as np
from PIL import Image, ImageDraw

from sotaigen.form.bodies import load_bodies
from sotaigen.output.glb import split_uv_seams
from sotaigen.pipeline import BUILD, ensure_built, template
from sotaigen.uv import (
    corner_triangles,
    export_fields,
    load_uv,
    overlapping_triangles,
    signed_areas,
    stretch,
    stretch_report,
)

COLOURS = {
    "trunk": (118, 124, 196),
    "neck": (196, 124, 196),
    "head": (226, 170, 120),
    "leg": (120, 188, 128),
    "arm": (196, 170, 96),
    "hand": (216, 96, 96),
    "foot": (92, 188, 196),
}


def draw_atlas(path, faces, uv, regions, size=1400):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    start = 0
    for face in faces:
        corners = uv[start : start + len(face)]
        start += len(face)
        outline = [(u * (size - 1), (1 - v) * (size - 1)) for u, v in corners]
        draw.polygon(outline, fill=COLOURS[regions[face[0]]], outline=(40, 40, 40))
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("presets", nargs="*")
    parser.add_argument("--atlas", help="also draw the layout to this PNG")
    parser.add_argument("--texture-px", type=int, default=2048)
    args = parser.parse_args()

    base = template()
    faces = base["faces"]
    regions = np.array(base["regions"])
    uv = load_uv(faces)
    triangles, owners = corner_triangles(faces)
    corner_verts = np.concatenate([np.asarray(face) for face in faces])
    areas = signed_areas(uv, triangles)
    overlaps = overlapping_triangles(uv, triangles)
    fields = export_fields(faces)
    split, _, _ = split_uv_seams(faces, len(base["verts"]), fields["uvs"], fields["faceUvs"])
    print(
        f"atlas: {len(faces)} faces, range {uv.min(0).round(4).tolist()} to "
        f"{uv.max(0).round(4).tolist()}, coverage {np.abs(areas).sum():.1%}"
    )
    print(
        f"  overlapping triangle pairs {len(overlaps)}, flipped {(areas < 0).sum()}, "
        f"without area {(np.abs(areas) < 1e-10).sum()}"
    )
    print(f"  vertices {len(base["verts"])} -> {len(split)} once split along seams")
    if args.atlas:
        draw_atlas(args.atlas, faces, uv, regions, 1400)
        print(f"  layout drawn to {args.atlas}")

    face_region = regions[np.array([face[0] for face in faces])][owners]
    uv_area = np.abs(areas).sum()
    names = args.presets or list(load_bodies())
    for name in names:
        ensure_built(name, ("wrap",))
        points = np.load(BUILD / "bodies" / name / "wrapped.npy")
        values, area = stretch(points, uv, corner_verts, triangles)
        print(name, stretch_report(values, area))
        px_per_mm = np.sqrt(uv_area / area.sum()) * args.texture_px
        for region in COLOURS:
            mine = face_region == region
            density = np.sqrt(values[mine, 0] * values[mine, 1])
            report = stretch_report(values[mine], area[mine])
            print(
                f"  {region:6s} {px_per_mm * np.average(density, weights=area[mine]):5.1f} px/mm "
                f"at {args.texture_px}, anisotropy p50 {report['anisotropyP50']} "
                f"p95 {report['anisotropyP95']} max {report['anisotropyMax']}"
            )


if __name__ == "__main__":
    main()
