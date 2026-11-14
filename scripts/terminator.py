"""Where a toon terminator comes from: the blended surface, the wrapped mesh, or the pose.

uv run scripts/terminator.py <body> --view backquarter [--pose armsDown]
    [--build DIR] [--old-fold] [--mm-per-px 1.5] [-o out.png]

Panels use the camera, key light and threshold of the toon look in blender/review/render.py:
field   the implicit surface's own gradient per pixel; what the masses alone would shade
mesh    the wrapped mesh at rest with smooth vertex normals; differs from field through the wrap
pose    the same mesh in the pose; differs from mesh through the skinning
angle   degrees between the mesh normal and the field's gradient, white 0 to black 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sotaigen.form import field as field_module
from sotaigen.form.anatomy import calibrated
from sotaigen.form.bodies import load_bodies
from sotaigen.form.head import extended
from sotaigen.mesh import unit_normals
from sotaigen.pipeline import BUILD, bodies_of
from sotaigen.template import load_template

VIEWS = {
    "front": (0, -1, 0),
    "threequarter": (0.64, -0.77, 0.08),
    "side": (1, 0, 0),
    "backquarter": (0.64, 0.77, 0.08),
    "back": (0, 1, 0),
}
KEY = np.array((-0.45, 0.55, 0.70))
TERMINATOR = 0.35
LIT = np.array((235, 232, 228), np.uint8)
SHADE = np.array((165, 165, 180), np.uint8)
FIELD_SEGMENTS = {"trunk", "upperArm", "lowerArm", "upperLeg", "lowerLeg", "neck"}


def read_obj(path):
    verts = []
    with open(path) as file:
        for line in file:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
    return np.array(verts)


def triangles(faces):
    out = []
    for face in faces:
        out += [(face[0], face[i], face[i + 1]) for i in range(1, len(face) - 1)]
    return np.array(out)


def camera(view):
    toward = np.array(VIEWS[view], float)
    toward /= np.linalg.norm(toward)
    right = np.cross((0.0, 0.0, 1.0), toward)
    right /= np.linalg.norm(right)
    up = np.cross(toward, right)
    light = np.column_stack((right, up, toward)) @ (KEY / np.linalg.norm(KEY))
    return right, up, toward, light


def rasterise(verts, tris, normals, view, mm_per_px, frame=None):
    """Depth-buffered points and normals per pixel, returned with the frame so panels line up."""
    right, up, toward, _ = camera(view)
    screen = np.column_stack((verts @ right, -(verts @ up))) / mm_per_px
    if frame is None:
        frame = screen.min(0) - 4, screen.max(0) + 4
    low, high = frame
    width, height = (high - low).astype(int) + 1
    screen = screen - low
    depth = np.full((height, width), -np.inf)
    point = np.zeros((height, width, 3))
    normal = np.zeros((height, width, 3))
    near = verts @ toward
    for tri in tris:
        a, b, c = screen[tri]
        x0, y0 = np.floor(np.minimum(np.minimum(a, b), c)).astype(int)
        x1, y1 = np.ceil(np.maximum(np.maximum(a, b), c)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width - 1), min(y1, height - 1)
        if x1 < x0 or y1 < y0:
            continue
        area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(area) < 1e-9:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        w1 = ((xs - a[0]) * (c[1] - a[1]) - (ys - a[1]) * (c[0] - a[0])) / area
        w2 = ((b[0] - a[0]) * (ys - a[1]) - (b[1] - a[1]) * (xs - a[0])) / area
        w0 = 1 - w1 - w2
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z = w0 * near[tri[0]] + w1 * near[tri[1]] + w2 * near[tri[2]]
        row, col = ys[inside], xs[inside]
        better = z[inside] > depth[row, col]
        row, col = row[better], col[better]
        weights = np.column_stack((w0[inside][better], w1[inside][better], w2[inside][better]))
        depth[row, col] = z[inside][better]
        point[row, col] = weights @ verts[tri]
        normal[row, col] = weights @ normals[tri]
    size = np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-9)
    return np.isfinite(depth), point, normal / size, frame


def shade(mask, normal, light):
    image = np.full((*mask.shape, 3), 255, np.uint8)
    lit = (normal @ light) > TERMINATOR
    image[mask & lit] = LIT
    image[mask & ~lit] = SHADE
    return image


def outward(body, points, steps=4):
    """The field's own unit normal at the surface point nearest to each given point."""
    for _ in range(steps):
        grad = body.gradient(points, step=1.5)
        squared = np.maximum(np.sum(grad * grad, axis=1, keepdims=True), 1e-9)
        points = points - 0.8 * grad * body(points)[:, None] / squared
    grad = body.gradient(points, step=1.5)
    return grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", metavar="name")
    parser.add_argument("--view", default="backquarter", choices=VIEWS)
    parser.add_argument("--pose", default="armsDown")
    parser.add_argument("--build", type=Path, default=BUILD)
    parser.add_argument("--mm-per-px", type=float, default=1.5)
    parser.add_argument(
        "--old-fold", action="store_true", help="the fold of builds made before tanh rounding"
    )
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()

    if args.old_fold:
        band = field_module.SOFT_MIDLINE
        field_module.soft_fold = lambda x: np.sqrt(x * x + band * band) - band
    body, _ = calibrated(load_bodies()[args.preset])
    body.clips = []
    for group in body.groups:
        group.pop("clips", None)

    base = extended(load_template())
    tris = triangles(base["faces"])
    segments = np.asarray(base["segments"])
    rest = read_obj(bodies_of(args.build) / args.preset / "poses" / "rest.obj")
    posed = read_obj(bodies_of(args.build) / args.preset / "poses" / f"{args.pose}.obj")
    _, _, _, light = camera(args.view)

    rest_normals = unit_normals(rest, tris)
    keep = np.isin(segments[tris[:, 0]], tuple(FIELD_SEGMENTS))
    mask, _point, normal, frame = rasterise(rest, tris, rest_normals, args.view, args.mm_per_px)
    mask_field, point_field, normal_field, _ = rasterise(
        rest, tris[keep], rest_normals, args.view, args.mm_per_px, frame
    )
    field_normal = np.zeros_like(normal)
    field_normal[mask_field] = outward(body, point_field[mask_field])
    angle = np.zeros(mask.shape)
    cosine = np.sum(field_normal[mask_field] * normal_field[mask_field], axis=1)
    angle[mask_field] = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    deviation = np.full((*mask.shape, 3), 255, np.uint8)
    deviation[mask] = (225, 225, 225)
    value = 255 - np.clip(angle[mask_field] / 20.0, 0, 1) * 255
    deviation[mask_field] = np.column_stack((value, value, value)).astype(np.uint8)

    posed_normals = unit_normals(posed, tris)
    mask_pose, _, normal_pose, _ = rasterise(posed, tris, posed_normals, args.view, args.mm_per_px, frame)

    panels = (
        shade(mask_field, field_normal, light),
        shade(mask, normal, light),
        shade(mask_pose, normal_pose, light),
        deviation,
    )
    gap = np.full((panels[0].shape[0], 8, 3), 255, np.uint8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    strip = [p for panel in panels for p in (panel, gap)][:-1]
    Image.fromarray(np.hstack(strip)).save(args.out)
    print(args.out, f"mesh-field angle p50 {np.percentile(angle[mask_field], 50):.1f}"
          f" p95 {np.percentile(angle[mask_field], 95):.1f} max {angle[mask_field].max():.1f}")


if __name__ == "__main__":
    main()
