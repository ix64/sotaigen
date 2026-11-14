"""See a body part move the way its card in docs/motion says to look at it.

uv run sotaigen-dev motion sweep <card> <body ...> [--look clay,toon] [--no-render]
    [--against DIR]
uv run sotaigen-dev motion folds <card> <body ...> [--against BUILD]

`sweep` turns the joint through the card's angles and writes, under build/reports/motion/<card>/<body>/:

    sweep.png      the review renders, with the joint stood upright so that `side` looks along the
                   bend axis, whatever the rest pose does; --against adds another run's poses
    behaviour.png  the section through the bend plane against the two limbs moved rigidly, and the
                   skin's lift and slide relative to its own limb, drawn without Blender
    report.json    the same lift, slide and radius by the card's zones, also printed

`folds` splits a pose's collapsed faces by whether the card expects a fold there. None of this
passes or fails anything: the regression gate stays sotaigen_dev/compare.py, and a good look stays the
user's call. It builds nothing but what sotaigen.pipeline.rigged needs.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sotaigen.mesh import triangulate, write_obj
from sotaigen.metrics import COLLAPSED, RestMesh
from sotaigen.pipeline import BUILD, bodies_of, fingerprints, rigged
from sotaigen.rig.poses import POSES
from sotaigen.rig.skeleton import load_rig
from sotaigen.rig.skinning import pose_skins, pose_verts
from sotaigen_dev.motion.cards import (
    ANCHOR,
    CardError,
    faces_in_zones,
    hinge_of,
    limb_coordinates,
    load_card,
    plane_section,
    skin_behaviour,
    sweep_poses,
    zone_mask,
    zone_summary,
)

CELL = 320
LABEL_WIDTH = 96
HEADER = 26
LEGEND = 44
# Half the window of a behaviour cell, in lengths of the lower limb.
WINDOW = 0.8
LIFT_RANGE_MM = 12.0
SLIDE_RANGE_MM = 25.0
# What of the body stays in a joint's review renders, and where the rest of it goes.
LIMB_REGIONS = ("arm", "hand")
FAR_UP = 6.0
BACKGROUND = (250, 250, 248)
RIGID_LINE = (120, 165, 215)
SKIN_LINE = (20, 20, 20)
# Each view: image right, image up, and towards the camera, in the canonical frame.
PROJECTIONS = {
    "side": ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    "back": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "front": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
}


def label_font(size=14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size)


def blend(low, high, share):
    return tuple(int(a + (b - a) * share) for a, b in zip(low, high))


def lift_colour(value):
    """Blue where skin sinks towards the limb, red where it rises off it, grey where neither."""
    share = float(np.clip(abs(value) / LIFT_RANGE_MM, 0, 1))
    return blend((225, 225, 225), (214, 39, 40) if value > 0 else (31, 90, 200), share)


def slide_colour(value):
    return blend((235, 235, 235), (120, 40, 160), float(np.clip(value / SLIDE_RANGE_MM, 0, 1)))


def to_pixels(uv, box, half_mm):
    """Canonical (right, up) around the joint to pixels inside a square cell."""
    x, y, size = box
    scale = size / (2 * half_mm)
    return np.column_stack((x + size / 2 + uv[:, 0] * scale, y + size / 2 - uv[:, 1] * scale))


def draw_segments(draw, segments, box, half_mm, colour, width):
    ends = to_pixels(segments.reshape(-1, 2), box, half_mm).reshape(-1, 2, 2)
    for start, end in ends:
        draw.line([tuple(start), tuple(end)], fill=colour, width=width)


def draw_section(canvas, box, half_mm, actual, rigid_sets):
    """The skin's section over the rigid limbs: the gap between them is what the skin did."""
    cell = Image.new("RGB", (box[2], box[2]), BACKGROUND)
    draw = ImageDraw.Draw(cell)
    local = (0, 0, box[2])
    for rigid in rigid_sets:
        draw_segments(draw, rigid, local, half_mm, RIGID_LINE, 3)
    draw_segments(draw, actual, local, half_mm, SKIN_LINE, 2)
    centre = local[2] / 2
    draw.ellipse((centre - 3, centre - 3, centre + 3, centre + 3), fill=(200, 60, 60))
    canvas.paste(cell, box[:2])


def draw_heat(canvas, box, half_mm, points, triangles, values, colour, view):
    """Faces painted far to near, shaded a little so the form stays readable."""
    right, up, towards = (np.array(axis, float) for axis in PROJECTIONS[view])
    corners = points[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    facing = normals @ towards
    order = np.argsort(corners.mean(axis=1) @ towards)
    cell = Image.new("RGB", (box[2], box[2]), BACKGROUND)
    draw = ImageDraw.Draw(cell)
    uv = np.column_stack(((points - ANCHOR) @ right, (points - ANCHOR) @ up))
    pixels = to_pixels(uv, (0, 0, box[2]), half_mm)
    face_values = values[triangles].mean(axis=1)
    for index in order:
        if facing[index] <= 0:
            continue
        shade = 0.55 + 0.45 * float(facing[index])
        fill = tuple(int(c * shade) for c in colour(face_values[index]))
        draw.polygon([tuple(pixels[i]) for i in triangles[index]], fill=fill)
    canvas.paste(cell, box[:2])


def limb_triangles(triangles, axial, left):
    """The exported triangles wholly on the left limb, then those on each limb alone."""
    on_limb = np.isfinite(axial) & left
    whole = on_limb[triangles].all(axis=1)
    side = axial[triangles]
    upper = whole & (side < 0).all(axis=1)
    lower = whole & (side >= 0).all(axis=1)
    return triangles[whole], triangles[upper], triangles[lower]


# Each row of behaviour.png: what it paints, and from where.
BEHAVIOUR_ROWS = {
    "section": "section",
    "lift side": ("lift", "side"),
    "lift back": ("lift", "back"),
    "lift front": ("lift", "front"),
    "slide side": ("slide", "side"),
}


def write_behaviour(path, hinge, rest, base, axial, frames, preset):
    """One column per angle: the section, then the skin's lift and slide from three sides."""
    triangles = triangulate(base["faces"], np.asarray(rest, float))
    left = np.asarray(base["verts"])[:, 0] >= 0
    limb, upper_tris, lower_tris = limb_triangles(triangles, axial, left)
    # The hand closes the forearm's end in the heat views, in the neutral colour.
    painted = left & np.isin(base["regions"], LIMB_REGIONS)
    painted_triangles = triangles[painted[triangles].all(axis=1)]
    rows = BEHAVIOUR_ROWS
    half_mm = WINDOW * hinge.length
    width = LABEL_WIDTH + CELL * len(frames)
    height = HEADER + CELL * len(rows) + LEGEND
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font = label_font()
    for column, (name, frame) in enumerate(frames.items()):
        x = LABEL_WIDTH + CELL * column
        draw.text((x + 8, 5), name, fill=(0, 0, 0), font=font)
        points = hinge.canonical(frame["posed"])
        for row, (title, what) in enumerate(rows.items()):
            box = (x, HEADER + CELL * row, CELL)
            if what == "section":
                rigid = [
                    plane_section(
                        hinge.canonical(moved_by(frame["skins"][bone], rest)), tris, ANCHOR[0]
                    )
                    - ANCHOR[1:]
                    for bone, tris in (
                        (hinge.upper_column, upper_tris),
                        (hinge.lower_column, lower_tris),
                    )
                ]
                actual = plane_section(points, limb, ANCHOR[0]) - ANCHOR[1:]
                draw_section(canvas, box, half_mm, actual, rigid)
            else:
                measure, view = what
                values = np.nan_to_num(getattr(frame["skin"], measure))
                paint = lift_colour if measure == "lift" else slide_colour
                draw_heat(canvas, box, half_mm, points, painted_triangles, values, paint, view)
    for row, title in enumerate(rows):
        draw.text((8, HEADER + CELL * row + 8), title, fill=(0, 0, 0), font=font)
    note = (
        f"{preset}  scale {2 * half_mm / CELL:.2f} mm/px  front of the arm is to the left in side "
        f"views.  section: black skin, blue limbs moved rigidly, red bend axis."
    )
    bottom = HEADER + CELL * len(rows)
    draw.text((8, bottom + 4), note, fill=(0, 0, 0), font=font)
    draw.text(
        (8, bottom + 22),
        f"lift: red rises off its own limb, blue sinks, full colour at {LIFT_RANGE_MM:g} mm.  "
        f"slide: purple, full at {SLIDE_RANGE_MM:g} mm.  Skin is judged from its own limb, so "
        "colour may step where the two limbs meet, and skin handed to the other limb reads as sinking.",
        fill=(0, 0, 0),
        font=font,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def moved_by(skin_matrix, rest):
    return (np.column_stack((rest, np.ones(len(rest)))) @ skin_matrix.T)[:, :3]


def sweep_frames(card, sweep, rig):
    """Each pose of a sweep, skinned the way the build does it. Elbows need no axilla targets."""
    # The joint's front comes from the card's first sweep: a later one may turn the other way.
    hinge = hinge_of(card, rig.bones, sample_pose(card["sweeps"][0]))
    axial, azimuth = limb_coordinates(hinge, card, rig.base, rig.verts)
    frames = {}
    for name, pose in sweep_poses(sweep).items():
        skins = pose_skins(rig.bones, pose)
        posed = pose_verts(rig.bones, rig.weights, rig.verts, pose)
        skin = skin_behaviour(hinge, card, rig.base, rig.verts, posed, skins, axial)
        canonical = hinge.canonical(posed)
        frames[name] = {"posed": posed, "skins": skins, "skin": skin, "canonical": canonical}
    return hinge, axial, azimuth, frames


def sample_pose(sweep):
    """A small turn the way the sweep goes, to tell the joint's front from its back."""
    return {bone: [axis * 5 for axis in axes] for bone, axes in sweep["pose"].items()}


def zone_report(card, axial, azimuth, frames, left):
    zones = {key: zone_mask(zone, axial, azimuth) & left for key, zone in card["zones"].items()}
    return {
        name: {
            key: {
                **zone_summary(frame["skin"], mask),
                "reachMm": round(axis_reach(frame["canonical"], mask), 1),
            }
            for key, mask in zones.items()
        }
        for name, frame in frames.items()
    }


def axis_reach(canonical, mask):
    """How far a zone's skin gets from the bend axis: this says if a corner stays out, in mm."""
    offset = canonical[mask][:, 1:] - ANCHOR[1:]
    return float(np.linalg.norm(offset, axis=1).max())


def print_report(report):
    columns = (
        "liftMeanMm",
        "liftMinMm",
        "liftMaxMm",
        "slideMeanMm",
        "slideMaxMm",
        "radiusShare",
        "reachMm",
    )
    print(f"  {'pose':10}{'zone':18}" + "".join(f"{column:>13}" for column in columns))
    for pose, zones in report.items():
        for key, row in zones.items():
            print(f"  {pose:10}{key:18}" + "".join(f"{row[column]:>13}" for column in columns))


def limb_only(hinge, base, points):
    """Canonical points with all but the left arm pulled to one spot far up the upper limb.

    Standing the joint upright tilts the whole body, and a tilted trunk behind the limb only
    distracts. The topology stays, because review's focus needs it, and the pulled faces vanish.
    """
    keep = np.isin(base["regions"], LIMB_REGIONS) & (np.asarray(base["verts"])[:, 0] >= 0)
    kept = np.array(points, float)
    kept[~keep] = ANCHOR + (0.0, 0.0, FAR_UP * hinge.length)
    return kept


def write_review_tree(root, preset, hinge, base, frames):
    """The poses as a build tree that sotaigen_dev/review.py reads, stood upright around the joint."""
    out = root / preset / "poses"
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        points = limb_only(hinge, base, hinge.canonical(frame["posed"]))
        write_obj(out / f"{name}.obj", points, base["faces"])
    (root / preset / "fingerprint.json").write_text(json.dumps({"pose": fingerprints()["pose"]}))


def review_sheet(card, sweep, root, preset, names, look, against, out):
    from sotaigen_dev import review

    review.BUILD = root
    review.POSES = {name: {} for name in names}
    views = list(dict.fromkeys(view for row in card["phenomena"] for view in row["views"]))
    axes = review.select_axes(
        [preset],
        pose=",".join(names),
        view=",".join(views),
        focus=card["joint"]["focus"],
        look=look,
    )
    review.make_sheet(axes, rows=("look", "view"), cols=("pose",), against=against, out=out)


def cmd_sweep(args):
    card = load_card(args.card, poses_known=POSES)
    if card["region"] == "shoulder-chest":
        # A shoulder needs its thorax and existing axilla correctives. The elbow's
        # single-hinge frame and arm-only crop would erase the phenomena on this card.
        from sotaigen_dev.motion.shoulder_sheet import run
        for preset in args.presets:
            run(
                preset,
                render=not args.no_render,
                look=args.look,
                against=args.against,
                contact=args.contact,
            )
        return 0


    if card.get("review", {}).get("renderer") == "pelvis":
        from sotaigen_dev.motion.groin_sheet import run

        return run(card, args)
    for preset in args.presets:
        rig = rigged(preset)
        folder = BUILD / "reports" / "motion" / card["region"] / preset
        report = {}
        for sweep in card["sweeps"]:
            hinge, axial, azimuth, frames = sweep_frames(card, sweep, rig)
            left = np.asarray(rig.base["verts"])[:, 0] >= 0
            report[sweep["name"]] = zone_report(card, axial, azimuth, frames, left)
            print(f"{preset} / {sweep['name']}")
            print_report(report[sweep["name"]])
            write_behaviour(
                folder / f"behaviour-{sweep['name']}.png",
                hinge,
                rig.verts,
                rig.base,
                axial,
                frames,
                preset,
            )
            if not args.no_render:
                root = BUILD / "reports" / "motion" / card["region"] / "poses"
                write_review_tree(root, preset, hinge, rig.base, frames)
                review_sheet(
                    card,
                    sweep,
                    root,
                    preset,
                    list(frames),
                    args.look,
                    args.against,
                    folder / f"sweep-{sweep['name']}.png",
                )
        (folder / "report.json").write_text(json.dumps(report, indent=1) + "\n")
        print(folder)
    return 0


def read_obj_verts(path):
    with open(path, encoding="ascii") as handle:
        rows = [line.split()[1:4] for line in handle if line.startswith("v ")]
    return np.array(rows, float)


def fold_counts(card, guard, base, rest, posed, bones, sweep):
    """Collapsed faces of a pose: all of them, and those the card expects in a fold."""
    hinge = hinge_of(card, bones, sample_pose(sweep))
    axial, azimuth = limb_coordinates(hinge, card, base, rest)
    masks = [zone_mask(card["zones"][key], axial, azimuth) for key in guard["expected_zones"]]
    collapsed = RestMesh(rest, base["faces"]).area_ratio(posed) < COLLAPSED
    expected = faces_in_zones(base["faces"], masks)
    return {
        "collapsed": int(collapsed.sum()),
        "inExpectedFold": int((collapsed & expected).sum()),
        "elsewhere": int((collapsed & ~expected).sum()),
    }


def cmd_folds(args):
    card = load_card(args.card, poses_known=POSES)
    if not card.get("guards"):
        print(f"card {card['region']} declares no expected folds", file=sys.stderr)
        return 2
    sweep = card["sweeps"][0]
    for preset in args.presets:
        rig = rigged(preset)
        for guard in card["guards"]:
            pose = guard["pose"]
            posed = read_obj_verts(BUILD / "bodies" / preset / "poses" / f"{pose}.obj")
            after = fold_counts(card, guard, rig.base, rig.verts, posed, rig.bones, sweep)
            line = f"{preset} / {pose}: {after}"
            if args.against:
                other = bodies_of(args.against) / preset
                old_rest = np.load(other / "wrapped.npy")
                old_posed = read_obj_verts(other / "poses" / f"{pose}.obj")
                old_bones, _ = load_rig(other)
                before = fold_counts(card, guard, rig.base, old_rest, old_posed, old_bones, sweep)
                line = f"{preset} / {pose}: {before} -> {after}"
            print(line)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sotaigen-dev motion", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    sweep = commands.add_parser("sweep")
    sweep.add_argument("card")
    sweep.add_argument("presets", nargs="+")
    sweep.add_argument("--look", default="clay,toon")
    sweep.add_argument("--no-render", action="store_true", help="skip Blender and sweep.png")
    sweep.add_argument("--against", help="saved run: limb cards use build/reports/motion/<card>/poses; pelvis uses build/reports/motion/<card>")
    sweep.add_argument(
        "--contact",
        action="store_true",
        help="shoulder card only: read the IPC contact table, baking it if missing",
    )
    folds = commands.add_parser("folds")
    folds.add_argument("card")
    folds.add_argument("presets", nargs="+")
    folds.add_argument("--against", help="another build directory")
    args = parser.parse_args(argv)
    try:
        return cmd_sweep(args) if args.command == "sweep" else cmd_folds(args)
    except CardError as error:
        print(f"motion_sheet: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
