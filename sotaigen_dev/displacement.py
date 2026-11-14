"""Where two builds' meshes differ, vertex by vertex: a summary for agents and heat maps for both.

uv run sotaigen-dev displacement --against DIR [body ...] [--build DIR] [--pose a,b]
    [--images rest|all|none|a,b] [--views front,side,back] [--scale MM] [--mm-per-px 2]
    [--allow-stale] [-o OUT]

Every body has the template's vertex order, so the same index is the same place on both sides and
their difference is exact: no matching, no sampling. For each body and pose both sides have, it
reads poses/<pose>.obj from --against (before) and from --build (after, default this checkout's
build) and splits each vertex's move into the part along the before mesh's normal (positive is
outward, the body got fuller there) and the part along the surface (a slide, which a shaded picture
hides). Nothing is built, and contact tables are never touched.

Writes OUT (default build/reports/displacement):

    displacement.json     every body and pose: the whole body, each template region and segment
                          split by side, and the largest moves far enough apart to be separate places
    <body>-<pose>.png   for the poses --images names: the after mesh, coloured on two rows, the
                          normal part (blue in, red out) and the size of the move (light to dark), on
                          a shared scale, the largest moves numbered as in the JSON

Images are orthographic at --mm-per-px, as sotaigen_dev/review.py's, so lengths can be read off them.
Without --scale, a sheet's scale is its own largest move rounded up to 1, 2 or 5 times a power of
ten, which the sheet prints; pass one to compare sheets with each other.

Exit codes: 0 compared, 2 something could not be compared (a missing file, another vertex count).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sotaigen.mesh import triangulate, unit_normals
from sotaigen.pipeline import BUILD, bodies_of, fingerprints, read_fingerprint, template

EXIT_INCOMPLETE = 2
# Below this a vertex has not moved: what writing an OBJ at three decimals leaves is 0.0005 mm.
STILL_MM = 0.01
# The regions and segments of a limb are told apart by side; the trunk, neck and head are not.
SIDED_REGIONS = {"arm", "hand", "leg", "foot"}
# A vertex this close to the midline, at rest on the before side, is on neither side.
MIDLINE_MM = 1.0
# The largest moves listed, each at least this far from the ones before it.
TOP = 8
TOP_APART_MM = 40.0
# A numbered vertex this far behind the nearest surface at its pixel is hidden, and not numbered.
HIDDEN_MM = 3.0
# The close-up around the largest move: see close_up.
CLOSE_MM = 150.0
CLOSE_SHARE = 0.2
CLOSE_PAD_MM = 25.0
CLOSE_LOCAL = 0.5
CLOSE_PX = 420
CLOSE_MIN_MM_PER_PX = 0.2
MARGIN_PX = 24
CAPTION_PX, LABEL_PX, BAR_PX = 28, 22, 46
# glTF's +x is the body's left: the build has the body looking down -y with z up.
VIEWS = {
    "front": (0, -1, 0),
    "threequarter": (0.64, -0.77, 0.08),
    "side": (1, 0, 0),
    "backquarter": (0.64, 0.77, 0.08),
    "back": (0, 1, 0),
}
DEFAULT_VIEWS = "front,side,back"
# The same key as scripts/terminator.py and the toon look: left, above and towards the camera.
KEY = np.array((-0.45, 0.55, 0.70)) / np.linalg.norm((-0.45, 0.55, 0.70))
BACKGROUND = (255, 255, 255)
INK = (30, 30, 34)
# Blue to grey to red, and light grey to dark red: both readable by size alone, and in grey.
DIVERGING = np.array(((0.16, 0.32, 0.75), (0.86, 0.86, 0.86), (0.75, 0.10, 0.12)))
SEQUENTIAL = np.array(
    ((0.88, 0.88, 0.88), (0.98, 0.76, 0.36), (0.84, 0.26, 0.18), (0.33, 0.05, 0.28))
)


# --- reading ---------------------------------------------------------------------------------


def read_points(path):
    """The vertices of an OBJ, as N x 3 float64."""
    rows = [line.split()[1:4] for line in path.read_text().splitlines() if line.startswith("v ")]
    return np.array(rows, float).reshape(-1, 3)


def pose_names(folder):
    return (
        {path.stem for path in (folder / "poses").glob("*.obj")}
        if (folder / "poses").is_dir()
        else set()
    )


def preset_folders(folder):
    return {path.name for path in folder.iterdir() if path.is_dir() and pose_names(path)}


def stale_presets(names, build):
    """Bodies of this checkout whose posed meshes no finished build vouches for."""
    current = fingerprints()["pose"]
    return [
        name
        for name in names
        if (build / name).is_dir() and read_fingerprint(build / name).get("pose") != current
    ]


# --- measuring -------------------------------------------------------------------------------


def sides(rest_x):
    """'left', 'right' or '' for each vertex, by where it is at rest."""
    return np.where(rest_x > MIDLINE_MM, "left", np.where(rest_x < -MIDLINE_MM, "right", ""))


def group_keys(labels, side, sided):
    """`labels` as group names, with the side after those in `sided`: 'arm.left', 'trunk'."""
    out = labels.astype(object)
    split = np.isin(labels, list(sided)) & (side != "")
    out[split] = out[split] + "." + side[split]
    return out


def split_move(before, after, triangles):
    """Each vertex's move, its part along the before normal, and the size of the rest."""
    move = after - before
    normal = unit_normals(before, triangles)
    along = np.einsum("ij,ij->i", move, normal)
    slide = np.linalg.norm(move - along[:, None] * normal, axis=1)
    return move, along, slide


def stats(size, along, share_of):
    """A group's numbers, in mm and rounded to what an OBJ holds."""
    if not len(size):
        return {"vertices": 0}
    return {
        "vertices": len(size),
        "moved": int((size > STILL_MM).sum()),
        "max": round(float(size.max()), 3),
        "p95": round(float(np.percentile(size, 95)), 3),
        "mean": round(float(size.mean()), 3),
        "outwardMean": round(float(along.mean()), 3),
        "outwardMin": round(float(along.min()), 3),
        "outwardMax": round(float(along.max()), 3),
        "share": round(float(size.sum() / share_of), 3) if share_of > 0 else 0.0,
    }


def grouped(keys, size, along):
    total = float(size.sum())
    groups = {
        key: stats(size[keys == key], along[keys == key], total) for key in dict.fromkeys(keys)
    }
    return dict(sorted(groups.items(), key=lambda item: -item[1].get("share", 0)))


def largest(points, size, along, slide, segment_keys, count=TOP, apart=TOP_APART_MM):
    """The largest moves, each at least `apart` mm from the ones already taken."""
    taken = []
    for index in np.argsort(-size):
        if size[index] <= STILL_MM or len(taken) == count:
            break
        if all(np.linalg.norm(points[index] - points[other]) >= apart for other in taken):
            taken.append(int(index))
    return [
        {
            "vertex": index,
            "segment": str(segment_keys[index]),
            "mm": round(float(size[index]), 3),
            "outward": round(float(along[index]), 3),
            "slide": round(float(slide[index]), 3),
            "at": [round(float(x), 1) for x in points[index]],
        }
        for index in taken
    ]


def summarise(before, after, triangles, regions, segments, rest_x):
    """One body and pose: the numbers, and the per-vertex arrays the pictures are made from."""
    move, along, slide = split_move(before, after, triangles)
    size = np.linalg.norm(move, axis=1)
    side = sides(rest_x)
    sided_segments = set(segments[np.isin(regions, list(SIDED_REGIONS))])
    segment_keys = group_keys(segments, side, sided_segments)
    whole = stats(size, along, float(size.sum()))
    if size.max() <= STILL_MM:
        return {"identical": True, "whole": whole}, size, along
    summary = {
        "identical": False,
        "whole": whole,
        "regions": grouped(group_keys(regions, side, SIDED_REGIONS), size, along),
        "segments": grouped(segment_keys, size, along),
        "largest": largest(after, size, along, slide, segment_keys),
    }
    return summary, size, along


# --- drawing ---------------------------------------------------------------------------------


def camera(view):
    toward = np.array(VIEWS[view], float)
    toward /= np.linalg.norm(toward)
    right = np.cross((0.0, 0.0, 1.0), toward)
    right /= np.linalg.norm(right)
    up = np.cross(toward, right)
    return right, up, toward


def rasterise(points, triangles, view, mm_per_px):
    """For each pixel of an orthographic view: the nearest triangle and its barycentric weights.

    Framed on the points the triangles use. Returns (triangle index, -1 where empty; weights
    H x W x 3; each point's screen position in pixels; each point's depth towards the camera, in mm;
    the depth buffer).
    """
    right, up, toward = camera(view)
    screen = np.column_stack((points @ right, -(points @ up))) / mm_per_px
    used = screen[np.unique(triangles)]
    screen = screen - (used.min(0) - MARGIN_PX)
    width, height = (used.max(0) - used.min(0) + 2 * MARGIN_PX).astype(int) + 1
    near = points @ toward
    depth = np.full((height, width), -np.inf)
    owner = np.full((height, width), -1)
    weights = np.zeros((height, width, 3))
    for number, triangle in enumerate(triangles):
        a, b, c = screen[triangle]
        x0, y0 = np.maximum(np.floor(np.minimum(np.minimum(a, b), c)).astype(int), 0)
        x1, y1 = np.minimum(
            np.ceil(np.maximum(np.maximum(a, b), c)).astype(int), (width - 1, height - 1)
        )
        area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if x1 < x0 or y1 < y0 or abs(area) < 1e-12:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        w1 = ((xs - a[0]) * (c[1] - a[1]) - (ys - a[1]) * (c[0] - a[0])) / area
        w2 = ((b[0] - a[0]) * (ys - a[1]) - (b[1] - a[1]) * (xs - a[0])) / area
        w0 = 1 - w1 - w2
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z = w0 * near[triangle[0]] + w1 * near[triangle[1]] + w2 * near[triangle[2]]
        row = (ys[inside] - 0.5).astype(int)
        col = (xs[inside] - 0.5).astype(int)
        better = z[inside] > depth[row, col]
        row, col = row[better], col[better]
        depth[row, col] = z[inside][better]
        owner[row, col] = number
        weights[row, col] = np.column_stack(
            (w0[inside][better], w1[inside][better], w2[inside][better])
        )
    return owner, weights, screen, near, depth


def colour_map(stops, t):
    """Colours along `stops` for t in [0, 1]."""
    t = np.clip(t, 0, 1) * (len(stops) - 1)
    low = np.minimum(t.astype(int), len(stops) - 2)
    part = (t - low)[..., None]
    return stops[low] * (1 - part) + stops[low + 1] * part


def nice_scale(value):
    """`value` rounded up to 1, 2 or 5 times a power of ten, and never below 0.1 mm."""
    value = max(value, 0.1)
    power = 10 ** math.floor(math.log10(value))
    return next(step * power for step in (1, 2, 5, 10) if step * power >= value - 1e-12)


def draw_view(raster, triangles, normals, values, stops, low, high, view, marks):
    """One rasterised view coloured by a per-vertex value, shaded by the key so the form reads."""
    owner, weights, screen, near, depth = raster
    image = np.full((*owner.shape, 3), BACKGROUND, float) / 255
    hit = owner >= 0
    corners = triangles[owner[hit]]
    value = np.einsum("ij,ij->i", weights[hit], values[corners])
    normal = np.einsum("ij,ijk->ik", weights[hit], normals[corners])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    right, up, toward = camera(view)
    light = np.column_stack((right, up, toward)) @ KEY
    lit = 0.62 + 0.38 * np.clip(normal @ light, 0, 1)
    image[hit] = colour_map(stops, (value - low) / (high - low)) * lit[:, None]
    picture = Image.fromarray((image * 255).round().astype(np.uint8))
    canvas = ImageDraw.Draw(picture)
    font = ImageFont.load_default(size=13)
    shown = set(np.unique(triangles))
    for number, vertex in marks:
        x, y = screen[vertex]
        row, col = int(y), int(x)
        if vertex not in shown or not (0 <= row < depth.shape[0] and 0 <= col < depth.shape[1]):
            continue
        # Only where the vertex is in front, so that a number never points through the body.
        if near[vertex] < depth[row, col] - HIDDEN_MM:
            continue
        canvas.ellipse((x - 7, y - 7, x + 7, y + 7), outline=INK, width=2)
        canvas.text((x + 9, y - 9), str(number), fill=INK, font=font)
    return picture


def colour_bar(stops, low, high, caption, width=260):
    bar = np.linspace(0, 1, width)
    strip = (colour_map(stops, bar)[None].repeat(14, 0) * 255).round().astype(np.uint8)
    picture = Image.new("RGB", (width + 300, BAR_PX), BACKGROUND)
    picture.paste(Image.fromarray(strip), (0, 4))
    canvas = ImageDraw.Draw(picture)
    font = ImageFont.load_default(size=12)
    for at, value in ((0, low), (width // 2, (low + high) / 2), (width - 1, high)):
        canvas.line((at, 18, at, 22), fill=INK)
        text = f"{value:+g}" if low < 0 else f"{value:g}"
        canvas.text((max(0, at - 14), 24), text, fill=INK, font=font)
    canvas.text((width + 10, 4), caption, fill=INK, font=font)
    return picture


def close_up(points, size, vertex, height_mm, mm_per_px):
    """The box around one vertex's move and a finer scale for it, or None if the move isn't local.

    The box holds the vertices within CLOSE_MM of it that moved at least CLOSE_SHARE as much,
    CLOSE_PAD_MM larger all round. The vertex is the first of `largest`, not argmax: a body's two
    sides tie, and the close-up has to show the place numbered 1.
    """
    centre = points[vertex]
    moved = size >= CLOSE_SHARE * size[vertex]
    near = (np.linalg.norm(points - centre, axis=1) < CLOSE_MM) & moved
    low, high = points[near].min(0) - CLOSE_PAD_MM, points[near].max(0) + CLOSE_PAD_MM
    if np.linalg.norm(high - low) > CLOSE_LOCAL * height_mm:
        return None
    finer = max(float((high - low).max()) / CLOSE_PX, CLOSE_MIN_MM_PER_PX)
    if finer > 0.75 * mm_per_px:
        return None
    return (low, high), finer


def block(points, triangles, normals, rows, views, mm_per_px, marks, caption):
    """Rows of views at one scale, each row over its colour bar, under a caption."""
    rasters = [rasterise(points, triangles, view, mm_per_px) for view in views]
    drawn = [
        [
            draw_view(raster, triangles, normals, values, stops, low, high, view, marks)
            for view, raster in zip(views, rasters, strict=True)
        ]
        for _, values, stops, low, high in rows
    ]
    widths = [max(row[i].width for row in drawn) for i in range(len(views))]
    heights = [max(view.height for view in row) for row in drawn]
    total = CAPTION_PX + sum(heights) + len(rows) * (LABEL_PX + BAR_PX)
    picture = Image.new("RGB", (max(sum(widths), 580), total), BACKGROUND)
    canvas = ImageDraw.Draw(picture)
    label = ImageFont.load_default(size=13)
    canvas.text((8, 6), caption, fill=INK, font=ImageFont.load_default(size=15))
    y = CAPTION_PX
    for (bar_caption, _, stops, low, high), row, height in zip(rows, drawn, heights, strict=True):
        x = 0
        for view, view_picture, width in zip(views, row, widths, strict=True):
            canvas.text((x + 8, y + 3), view, fill=INK, font=label)
            picture.paste(view_picture, (x, y + LABEL_PX))
            x += width
        y += LABEL_PX + height
        picture.paste(colour_bar(stops, low, high, bar_caption), (8, y + 2))
        y += BAR_PX
    return picture


def sheet(name, pose, after, triangles, size, along, largest_moves, views, mm_per_px, scale):
    """The after mesh on two rows, the normal part and the size of the move, a column per view;
    then the same around the largest move, closer and cut out of the body, when it is local."""
    normals = unit_normals(after, triangles)
    marks = [(number, move["vertex"]) for number, move in enumerate(largest_moves, 1)]
    rows = (
        ("outward (mm): red is out, blue is in", along, DIVERGING, -scale, scale),
        ("size of move (mm), sliding included", size, SEQUENTIAL, 0.0, scale),
    )
    head = f"{name} / {pose}   max {size.max():.2f} mm   colours to {scale:g} mm"
    whole = block(
        after, triangles, normals, rows, views, mm_per_px, marks, f"{head}   {mm_per_px:g} mm/px"
    )
    blocks = [whole]
    first = largest_moves[0]["vertex"]
    found = close_up(after, size, first, float(np.ptp(after[:, 2])), mm_per_px)
    if found:
        (low, high), finer = found
        inside = np.all((after >= low) & (after <= high), axis=1)
        cut = triangles[inside[triangles].all(axis=1)]
        caption = f"around 1, cut out of the body   {finer:.2g} mm/px"
        blocks.append(block(after, cut, normals, rows, views, finer, marks, caption))
    width = max(part.width for part in blocks)
    picture = Image.new("RGB", (width, sum(part.height for part in blocks)), BACKGROUND)
    y = 0
    for part in blocks:
        picture.paste(part, (0, y))
        y += part.height
    return picture


# --- the command -----------------------------------------------------------------------------


def comma_list(text):
    return [item for item in text.split(",") if item]


def brief(name, pose, summary):
    """One line for a pose that changed: how much, where most, and which regions hold most of it."""
    whole, top = summary["whole"], summary["largest"][0]
    regions = ", ".join(
        f"{key} {group['share']:.0%}"
        for key, group in list(summary["regions"].items())[:3]
        if group["share"] >= 0.01
    )
    return (
        f"{name} / {pose}: max {whole['max']:.2f} mm at {top['segment']} (vertex {top['vertex']}), "
        f"p95 {whole['p95']:.2f}, outward mean {whole['outwardMean']:+.2f}; most of it: {regions}"
    )


def compare_pose(folder_before, folder_after, pose, labels, triangles, rest_x):
    before = read_points(folder_before / "poses" / f"{pose}.obj")
    after = read_points(folder_after / "poses" / f"{pose}.obj")
    if len(before) != len(after) or len(after) != len(labels[0]):
        return (
            None,
            f"{len(before)} vertices before, {len(after)} after, {len(labels[0])} in the template",
        )
    summary, size, along = summarise(before, after, triangles, *labels, rest_x)
    return (summary, after, size, along), None


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sotaigen-dev displacement", description=__doc__.splitlines()[0])
    parser.add_argument("presets", nargs="*", metavar="name", help="default: all both sides have")
    parser.add_argument("--against", required=True, type=Path, help="the build directory before")
    parser.add_argument("--build", type=Path, default=BUILD, help="the build directory after")
    parser.add_argument(
        "--pose", type=comma_list, help="only these poses; default: all both sides have"
    )
    parser.add_argument("--images", default="rest", help="poses to draw: names, all or none")
    parser.add_argument("--views", type=comma_list, default=comma_list(DEFAULT_VIEWS))
    parser.add_argument(
        "--scale", type=float, help="mm at the ends of the colours; default: each sheet's own"
    )
    parser.add_argument("--mm-per-px", type=float, default=2.0)
    parser.add_argument(
        "--allow-stale", action="store_true", help="accept out-of-date posed meshes"
    )
    parser.add_argument("-o", "--out", type=Path, default=BUILD / "reports" / "displacement")
    args = parser.parse_args(argv)
    for flag, folder in (("--against", args.against), ("--build", args.build)):
        if not folder.is_dir():
            parser.error(f"{flag}: {folder} is not a directory")
    before_root, after_root = bodies_of(args.against), bodies_of(args.build)
    if before_root.resolve() == after_root.resolve():
        parser.error("--against and --build are the same directory")
    unknown = [view for view in args.views if view not in VIEWS]
    if unknown:
        parser.error(f"--views: unknown {', '.join(unknown)}; known: {', '.join(VIEWS)}")

    names = args.presets or sorted(preset_folders(before_root) & preset_folders(after_root))
    if not names:
        print(
            f"displacement: no body has posed meshes in both {before_root} and {after_root}",
            file=sys.stderr,
        )
        return EXIT_INCOMPLETE
    if after_root.resolve() == (BUILD / "bodies").resolve():
        stale = stale_presets(names, after_root)
        if stale and not args.allow_stale:
            listed = " ".join(stale)
            print(
                f"displacement: posed meshes out of date for {listed}\nrun `uv run sotaigen-dev build {listed} "
                "--stages pose` first, or pass --allow-stale to use them anyway",
                file=sys.stderr,
            )
            return EXIT_INCOMPLETE

    base = template()
    triangles = triangulate(base["faces"], np.array(base["verts"], float))
    labels = (np.array(base["regions"]), np.array(base["segments"]))
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "before": str(before_root.resolve()),
        "after": str(after_root.resolve()),
        "unit": "mm",
        "presets": {},
    }
    problems, lines = [], []
    for name in names:
        before_poses, after_poses = pose_names(before_root / name), pose_names(after_root / name)
        common = sorted(before_poses & after_poses)
        poses = [pose for pose in args.pose if pose in common] if args.pose else common
        missing = sorted(
            (set(args.pose) if args.pose else before_poses | after_poses) - set(common)
        )
        if missing:
            problems.append(f"{name}: not on both sides: {', '.join(missing)}")
        if "rest" not in common:
            problems.append(f"{name}: no rest pose on both sides, so no sides or images")
            continue
        rest_x = read_points(before_root / name / "poses" / "rest.obj")[:, 0]
        drawn = (
            set(common)
            if args.images == "all"
            else set()
            if args.images == "none"
            else set(comma_list(args.images))
        )
        entry = report["presets"][name] = {}
        identical = []
        for pose in poses:
            found, problem = compare_pose(
                before_root / name, after_root / name, pose, labels, triangles, rest_x
            )
            if problem:
                problems.append(f"{name} / {pose}: {problem}")
                continue
            summary, after, size, along = found
            entry[pose] = summary
            if summary["identical"]:
                identical.append(pose)
                continue
            lines.append(brief(name, pose, summary))
            print(lines[-1])
            if pose in drawn:
                scale = args.scale or nice_scale(float(size.max()))
                picture = sheet(
                    name,
                    pose,
                    after,
                    triangles,
                    size,
                    along,
                    summary["largest"],
                    args.views,
                    args.mm_per_px,
                    scale,
                )
                path = args.out / f"{name}-{pose}.png"
                picture.save(path)
                summary["image"] = {
                    "path": str(path),
                    "scale": scale,
                    "mmPerPx": args.mm_per_px,
                    "views": args.views,
                }
        if identical:
            lines.append(f"{name}: identical in {len(identical)} poses: {', '.join(identical)}")
            print(lines[-1])
    (args.out / "displacement.json").write_text(json.dumps(report, indent=1))
    (args.out / "summary.txt").write_text("".join(f"{line}\n" for line in lines))
    print(f"wrote {args.out / 'displacement.json'} and summary.txt")
    for problem in problems:
        print(f"displacement: {problem}", file=sys.stderr)
    return EXIT_INCOMPLETE if problems else 0


if __name__ == "__main__":
    sys.exit(main())
