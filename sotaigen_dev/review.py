"""Scale-true review renders and sheets of posed bodies.

uv run sotaigen-dev review render <body ...> [--pose a,b] [--view front,side]
    [--look clay,toon,silhouette] [--camera ortho,persp] [--allow-stale]
uv run sotaigen-dev review sheet [name ...] [--sex female,male] [--group long,balanced] [--pose rest]
    [--view front,side] [--look clay,toon,silhouette,toon+persp] [--camera ortho,persp]
    [--focus hip]
    [--rows look] [--cols view] [--against DIR] [--wire]
    [--mm-per-px 2] [--cell-px 900] [--allow-stale] [-o out.png]
uv run sotaigen-dev review verify [body ...] [--pose rest] [--view front,side]
    [--allow-stale]

Every orthographic cell of a sheet is rendered at one mm per pixel and pasted without resampling, so
sizes compare directly across a sheet. A perspective cell holds that scale only on the plane through
its camera's aim, and says so. Inputs are build/bodies/<body>/poses/<pose>.obj, refused when missing or
out of date; `render` writes build/bodies/<body>/renders/<pose>/<view>.png, with -<look>-<camera> before
the extension for anything but clay and ortho, and exits 3 when Blender cannot be found.

A silhouette sheet with --against has a third pane in each cell, CHANGE: both meshes rendered
into one frame and laid over each other, grey where both have body, red where only AFTER has it
(the outline grew) and blue where only BEFORE has it (it shrank). One frame means one camera,
so the two line up to the pixel without resampling either.
"""

import argparse
import functools
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sotaigen.form.bodies import load_bodies, load_specs
from sotaigen.form.head import extended
from sotaigen.output.blender import BLENDER_APP, find_blender
from sotaigen.pipeline import BUILD, bodies_of, fingerprints, read_fingerprint
from sotaigen.rig.poses import POSES
from sotaigen.template import load_template
from sotaigen_dev.pictures import font

ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "blender" / "review" / "render.py"

EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_NO_BLENDER = 3

# Directions from the body to the camera. Front looks along +Y, so the wearer's left (+X) is on the
# image's right; the side and quarter views all look at the wearer's left.
VIEWS = {
    "front": (0, -1, 0),
    "threequarter": (0.64, -0.77, 0.08),
    "side": (1, 0, 0),
    "backquarter": (0.64, 0.77, 0.08),
    "back": (0, 1, 0),
}
DEFAULT_VIEWS = ("front", "threequarter", "side", "back")

LOOKS = ("clay", "toon", "silhouette")
CAMERAS = ("ortho", "persp")
# --look may name its camera too, so that a sheet can have clay, toon and toon+persp as its rows.
LOOK_CHOICES = LOOKS + tuple(f"{look}+{camera}" for look in LOOKS for camera in CAMERAS)
# The perspective view: a 50 mm lens framing 1.08 statures of height. Only the camera's position
# decides the perspective, so every pose and part of a body is seen from this one distance.
LENS_MM = 50.0
SENSOR_MM = 36.0
FRAME_PER_STATURE = 1.08

# Scales a sheet may pick from. Round values read easily and let sheets share cached renders.
SCALES = (0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 8)
BODY_CELL_PX = 900
FOCUS_CELL_PX = 600

# `render` output: one canvas for every pose and view, so frames can be flipped through.
RENDER_MM_PER_PX = 1.5
RENDER_CANVAS = (1100, 1200)
RENDER_GROUND_PX = 40

# Sheet layout, in pixels.
SIDE_PAD = 8
TOP_PAD = 8
GROUND_PAD = 6
MARGIN = 16
GUTTER = 12
PANE_GAP = 8
MIN_PANE = 130
MIN_CELL = 170
PERSP_SLOT = 280
PERSP_CELL_NOTE = "perspective: scale only on centre plane"
PERSP_SHEET_NOTE = (
    "Perspective cells: mm per pixel holds only on the plane through the camera's aim; "
    "the ground line marks each body's lowest point, not a shared floor."
)
LINE = 17
HEADER = 26
FOOTER = 48
FONT_SIZE = 14
HEADER_FONT_SIZE = 16
CHAR_PX = 9
LABEL_CHAR_PX = 7
BAR_MIN_PX = 80
# A pixel counts as body when darker than halfway between white and the lit grey.
SILHOUETTE_LUMINANCE = 200
# A CHANGE pane's colours: body on both sides, only after, only before.
BOTH_RGB = (47, 49, 54)
GAINED_RGB = (214, 48, 39)
LOST_RGB = (38, 112, 212)
CHANGE_TAG = "CHANGE"
CHANGE_NOTE = "CHANGE: grey in both, red only AFTER (grew), blue only BEFORE (shrank)"
MAX_RATIO_ERROR = 0.01


class ReviewError(Exception):
    """A failure to report in one line, with the process exit code it maps to."""

    def __init__(self, message, code=EXIT_FAILED):
        super().__init__(message)
        self.code = code


# --- Selection ---


def split_names(text):
    return [name.strip() for name in text.split(",") if name.strip()]


def pick_names(text, known, what):
    """A comma list checked against the known names; `all` means every one of them."""
    names = split_names(text)
    if names == ["all"]:
        return list(known)
    unknown = [name for name in names if name not in known]
    if unknown or not names:
        raise ReviewError(
            f"unknown {what}: {', '.join(unknown) or text!r}; known: {', '.join(known)}",
            EXIT_USAGE,
        )
    return names


def catalog():
    """The body names, and the bodies and styles they are made of, in file order."""
    names = list(load_bodies())
    parts = [name.rsplit("-", 1) for name in names]
    return names, list(dict.fromkeys(p[0] for p in parts)), list(dict.fromkeys(p[1] for p in parts))


def load_typical():
    """The typical bodies' names: what a sheet shows when it is not told what to show."""
    return list(load_specs("typical"))


def typical_styles(names, styles):
    """The groups of the typical bodies among these names; every style when none is typical."""
    typical = [name for name in load_typical() if name in names]
    return list(dict.fromkeys(name.rsplit("-", 1)[1] for name in typical)) or styles


def split_look(value):
    """A --look value as (look, camera), where camera is None unless the value names one."""
    look, _, camera = value.partition("+")
    return look, camera or None


def pick_looks(look, camera):
    """The --look and --camera values, checked. A camera named twice would be ambiguous."""
    looks = list(LOOKS) if look == "all" else pick_names(look or "clay", LOOK_CHOICES, "look")
    cameras = pick_names(camera or "ortho", CAMERAS, "camera")
    if camera and any(split_look(value)[1] for value in looks):
        raise ReviewError("a --look that names its camera cannot go with --camera", EXIT_USAGE)
    return looks, cameras


def select_axes(
    presets=(), sex=None, group=None, pose=None, view=None, focus=None, look=None, camera=None
):
    """What a sheet shows, as axis name to values. Names replace the sex and group axes."""
    names, bodies, styles = catalog()
    if presets and (sex or group):
        raise ReviewError("give body names or --sex/--group, not both", EXIT_USAGE)
    if presets:
        axes = {"preset": pick_names(",".join(presets), names, "preset")}
    else:
        axes = {
            "sex": pick_names(sex, bodies, "sex") if sex else bodies,
            "group": pick_names(group, styles, "group") if group else typical_styles(names, styles),
        }
        # Each sex has its own groups, so not every sex and group make a body. Then the
        # sheet lists the bodies that do, in file order, instead of a grid with holes in it.
        wanted = {f"{b}-{s}" for b in axes["sex"] for s in axes["group"]}
        if not wanted <= set(names):
            axes = {"preset": [name for name in names if name in wanted]}
            if not axes["preset"]:
                raise ReviewError("no body has these --sex and --group", EXIT_USAGE)
    axes["pose"] = pick_names(pose, POSES, "pose") if pose else ["rest"]
    axes["view"] = pick_names(view, VIEWS, "view") if view else list(DEFAULT_VIEWS)
    if focus:
        axes["focus"] = pick_names(focus, FOCUS, "focus")
    looks, cameras = pick_looks(look, camera)
    if look:
        axes["look"] = looks
    if camera:
        axes["camera"] = cameras
    return axes


@dataclass(frozen=True)
class GridCell:
    row: int
    col: int
    coord: dict

    @property
    def preset(self):
        return self.coord.get("preset") or f"{self.coord['sex']}-{self.coord['group']}"

    @property
    def look(self):
        return split_look(self.coord.get("look", LOOKS[0]))[0]

    @property
    def camera(self):
        named = split_look(self.coord.get("look", LOOKS[0]))[1]
        return named or self.coord.get("camera", CAMERAS[0])


@dataclass(frozen=True)
class Grid:
    cells: list
    row_titles: list
    col_titles: list


ROW_FIRST = ("look", "camera", "group", "pose", "focus", "view")


def wrap_width(count):
    """Columns for one long list: short ones stay in a row, long ones go roughly square."""
    return count if count <= 5 else math.ceil(math.sqrt(count * 1.6))


def arrange(axes, rows=(), cols=()):
    """Lay the axes on a grid; multi-valued axes nobody placed go to rows by ROW_FIRST, then cols.

    Bodies stay in columns by default: a row shares one ground line, so heights read directly.
    """
    rows, cols = list(rows), list(cols)
    for name in rows + cols:
        if name not in axes:
            raise ReviewError(
                f"unknown axis: {name}; this sheet has: {', '.join(axes)}", EXIT_USAGE
            )
    if len(set(rows + cols)) != len(rows + cols):
        raise ReviewError(f"an axis is used twice: rows={rows} cols={cols}", EXIT_USAGE)

    free = [n for n in axes if len(axes[n]) > 1 and n not in rows + cols]
    if not rows and not cols and len(free) == 1:
        values = axes[free[0]]
        fixed = {name: values[0] for name, values in axes.items() if len(values) == 1}
        width = wrap_width(len(values))
        cells = [
            GridCell(i // width, i % width, {**fixed, free[0]: value})
            for i, value in enumerate(values)
        ]
        return Grid(cells, [""] * (cells[-1].row + 1), [""] * width)
    if not rows and free:
        rows.append(min(free, key=lambda name: ROW_FIRST.index(name) if name in ROW_FIRST else 99))
        free.remove(rows[0])
    cols += free

    fixed = {n: v[0] for n, v in axes.items() if n not in rows + cols}
    row_keys = list(itertools.product(*(axes[n] for n in rows)))
    col_keys = list(itertools.product(*(axes[n] for n in cols)))
    cells = [
        GridCell(r, c, {**fixed, **dict(zip(rows, rk)), **dict(zip(cols, ck))})
        for r, rk in enumerate(row_keys)
        for c, ck in enumerate(col_keys)
    ]
    return Grid(cells, [" / ".join(k) for k in row_keys], [" / ".join(k) for k in col_keys])


def sheet_name(axes, against=None, wire=False, scale=None):
    """A file name that says what the sheet shows; long lists shrink to a count and a hash."""
    parts = []
    for name, values in axes.items():
        if len(values) <= 4:
            parts.append("+".join(values))
        else:
            digest = hashlib.sha256(",".join(values).encode()).hexdigest()[:6]
            parts.append(f"{len(values)}{name}s-{digest}")
    if against:
        digest = hashlib.sha256(str(Path(against).resolve()).encode()).hexdigest()[:6]
        parts.append(f"vs-{digest}")
    if wire:
        parts.append("wire")
    if scale:
        parts.append(scale)
    return "_".join(parts) + ".png"


# --- Meshes and parts ---


@dataclass(frozen=True, eq=False)
class Mesh:
    path: Path
    digest: str
    verts: np.ndarray


def read_mesh(path):
    data = Path(path).read_bytes()
    rows = [line.split()[1:4] for line in data.splitlines() if line.startswith(b"v ")]
    return Mesh(Path(path), hashlib.sha256(data).hexdigest(), np.array(rows, dtype=float))


def pose_path(build, preset, pose):
    return Path(build) / preset / "poses" / f"{pose}.obj"


def stale_presets(build, presets):
    """Bodies whose poses no finished build vouches for as made from the current sources."""
    current = fingerprints()["pose"]
    return [
        preset
        for preset in dict.fromkeys(presets)
        if read_fingerprint(Path(build) / preset).get("pose") != current
    ]


def check_fresh(build, presets, allow_stale=False):
    """Refuse poses that may not show the current sources. Returns the bodies let through stale."""
    stale = stale_presets(build, presets)
    if stale and not allow_stale:
        names = " ".join(stale)
        raise ReviewError(
            f"poses out of date for {names}: sources changed since they were built, or the build "
            f"did not finish\nrun `uv run sotaigen-dev build {names} --stages pose` first, "
            "or pass --allow-stale to look at them anyway"
        )
    if stale:
        print(f"review: showing out-of-date poses for {' '.join(stale)}", file=sys.stderr)
    return stale


def check_inputs(paths):
    missing = [path for path in dict.fromkeys(paths) if not path.exists()]
    if missing:
        listed = "\n  ".join(str(path) for path in missing[:5])
        more = f"\n  and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise ReviewError(f"missing input:\n  {listed}{more}\nrun `uv run sotaigen-dev build` first")


FINGERS = tuple(
    f"{finger}{joint}"
    for finger, joints in (
        ("thumb", ("Metacarpal", "Proximal", "Distal")),
        ("index", ("Proximal", "Intermediate", "Distal")),
        ("middle", ("Proximal", "Intermediate", "Distal")),
        ("ring", ("Proximal", "Intermediate", "Distal")),
        ("little", ("Proximal", "Intermediate", "Distal")),
    )
    for joint in joints
)


@dataclass(frozen=True)
class Focus:
    """Where to look: whole segments, or the seam where two meet, with room around it."""

    parts: tuple = ()
    seam: tuple = ()
    context: float = 1.0
    left_only: bool = True


FOCUS = {
    "head": Focus(parts=("head",), context=1.2, left_only=False),
    "neck": Focus(parts=("neck",), context=1.8, left_only=False),
    "shoulder": Focus(seam=("trunk", "upperArm"), context=1.7),
    "elbow": Focus(seam=("upperArm", "lowerArm"), context=3.0),
    "wrist": Focus(seam=("lowerArm", "hand"), context=3.0),
    "hand": Focus(parts=("hand", *FINGERS), context=1.15),
    "hip": Focus(seam=("trunk", "upperLeg"), context=1.6),
    "knee": Focus(seam=("upperLeg", "lowerLeg"), context=2.6),
    "ankle": Focus(seam=("lowerLeg", "foot"), context=2.6),
    "foot": Focus(parts=("foot", "toes"), context=1.15),
}


def focus_region(name, base):
    """Vertex indices of a part. Segments and faces are fixed, so these hold in every pose."""
    focus = FOCUS[name]
    segments = np.asarray(base["segments"])
    if focus.parts:
        mask = np.isin(segments, focus.parts)
    else:
        first, second = focus.seam
        mask = np.zeros(len(segments), bool)
        for face in base["faces"]:
            found = {segments[i] for i in face}
            if first in found and second in found:
                mask[[i for i in face if segments[i] in focus.seam]] = True
    if focus.left_only:
        mask &= np.asarray(base["verts"])[:, 0] >= 0
    return np.nonzero(mask)[0]


@functools.cache
def template_base():
    return extended(load_template())


@functools.cache
def focus_regions():
    return {name: focus_region(name, template_base()) for name in FOCUS}


# --- Geometry and framing ---


def view_axes(view):
    """The camera's right, up and toward-camera unit vectors, keeping world Z upright."""
    toward = np.asarray(VIEWS[view], float)
    toward /= np.linalg.norm(toward)
    up = np.array([0.0, 0.0, 1.0]) - toward[2] * toward
    up /= np.linalg.norm(up)
    return np.cross(up, toward), up, toward


def project(points, view):
    """Points as (u, v) millimetres on the view plane: u to the image's right, v up."""
    right, up, _ = view_axes(view)
    return np.column_stack((points @ right, points @ up))


@functools.cache
def stature(preset):
    return load_bodies()[preset]["stature"]


def camera_distance(preset):
    """Millimetres from a perspective camera to its aim: the body's height, not the pose's."""
    return stature(preset) * FRAME_PER_STATURE * LENS_MM / SENSOR_MM


def camera_aim(verts, preset):
    """Where a perspective camera aims: the body's axis, half a stature above its lowest point."""
    return np.array([0.0, 0.0, verts[:, 2].min() + stature(preset) / 2])


def perspective_plane(points, view, distance, aim):
    """Points through the lens, as (u, v) on the plane through the aim, where project() agrees."""
    right, up, toward = view_axes(view)
    relative = points - aim
    enlarge = distance / (distance - relative @ toward)
    aim_u, aim_v = project(aim[None], view)[0]
    return np.column_stack((relative @ right * enlarge + aim_u, relative @ up * enlarge + aim_v))


@dataclass(frozen=True)
class Frame:
    """One render: a window on the view plane at a fixed scale, sized in whole pixels."""

    view: str
    u: float
    v: float
    width: int
    height: int
    mm_per_px: float
    ground: int | None = None
    look: str = "clay"
    camera: str = "ortho"
    distance: float | None = None
    aim: tuple | None = None

    def world_center(self):
        right, up, _ = view_axes(self.view)
        return right * self.u + up * self.v

    def lens(self):
        """Lens and shift that frame this window from the camera's fixed place, with no resample."""
        side = max(self.width, self.height) * self.mm_per_px
        aim_u, aim_v = project(np.asarray(self.aim)[None], self.view)[0]
        return self.distance * SENSOR_MM / side, (self.u - aim_u) / side, (self.v - aim_v) / side


def extent(verts, view, distance=None, aim=None):
    uv = project(verts, view) if distance is None else perspective_plane(verts, view, distance, aim)
    return uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max()


def body_frame(ext, view, mm_per_px, u_center=None, half_width=None):
    """Frame a whole body with its lowest vertex exactly on the frame's ground row."""
    u_min, u_max, v_min, v_max = ext
    if u_center is None:
        u_center = (u_min + u_max) / 2
    if half_width is None:
        half_width = (u_max - u_min) / 2
    width = math.ceil(2 * half_width / mm_per_px) + 2 * SIDE_PAD
    height = math.ceil((v_max - v_min) / mm_per_px) + TOP_PAD + GROUND_PAD
    v_center = v_min + (height / 2 - GROUND_PAD) * mm_per_px
    return Frame(
        view, float(u_center), float(v_center), width, height, mm_per_px, height - GROUND_PAD
    )


def focus_sphere(verts, region, context):
    """Centre of a part and the frame side that holds it from any direction."""
    points = verts[region]
    center = (points.min(0) + points.max(0)) / 2
    return center, 2 * float(np.linalg.norm(points - center, axis=1).max()) * context


def focus_frame(center, side_mm, view, mm_per_px):
    u, v = project(center[None], view)[0]
    side = math.ceil(side_mm / mm_per_px)
    return Frame(view, float(u), float(v), side, side, mm_per_px)


def as_tuple(aim):
    return None if aim is None else tuple(float(value) for value in aim)


def pick_scale(largest_mm, target_px):
    """The finest listed scale that fits the largest thing on the sheet in the target pixels."""
    for scale in SCALES:
        if largest_mm / scale <= target_px:
            return scale
    return SCALES[-1]


def render_canvas(verts, view, look="clay", camera="ortho", preset=None):
    """The shared `render` frame: body axis centred, feet on a fixed row, grown only if needed."""
    distance = aim = None
    if camera == "persp":
        distance, aim = camera_distance(preset), camera_aim(verts, preset)
    u_min, u_max, v_min, v_max = extent(verts, view, distance, aim)
    width, height = RENDER_CANVAS
    width = max(width, math.ceil(2 * max(-u_min, u_max) / RENDER_MM_PER_PX) + 2 * SIDE_PAD)
    needed = math.ceil((v_max - v_min) / RENDER_MM_PER_PX) + RENDER_GROUND_PX + TOP_PAD
    height = max(height, needed)
    v_center = v_min + (height / 2 - RENDER_GROUND_PX) * RENDER_MM_PER_PX
    frame = Frame(
        view, 0.0, float(v_center), width, height, RENDER_MM_PER_PX, height - RENDER_GROUND_PX
    )
    return replace(frame, look=look, camera=camera, distance=distance, aim=as_tuple(aim))


@dataclass(frozen=True)
class Pane:
    """One render in a cell. `under`, for a CHANGE pane, is the mesh laid beneath `mesh`."""

    tag: str
    mesh: Mesh
    under: Mesh | None = None

    def extent(self, view, distance=None, aim=None):
        """Both meshes' extent when there are two, so that neither is cropped."""
        own = extent(self.mesh.verts, view, distance, aim)
        if self.under is None:
            return own
        other = extent(self.under.verts, view, distance, aim)
        return (
            min(own[0], other[0]),
            max(own[1], other[1]),
            min(own[2], other[2]),
            max(own[3], other[3]),
        )


@dataclass(frozen=True)
class CellSpec:
    grid: GridCell
    panes: tuple

    @property
    def view(self):
        return self.grid.coord["view"]

    @property
    def focus(self):
        return self.grid.coord.get("focus")

    @property
    def look(self):
        return self.grid.look

    @property
    def camera(self):
        return self.grid.camera

    def lens(self, verts):
        """The distance and aim this cell's camera has for these vertices, or None for ortho."""
        if self.camera != "persp":
            return None, None
        return camera_distance(self.grid.preset), camera_aim(verts, self.grid.preset)


def focus_through_lens(verts, region, center, side, context, view, distance, aim):
    """A part's centre on the aim's plane, and the frame side that holds it as the lens shows it."""
    points = perspective_plane(verts[region], view, distance, aim)
    middle = (points.min(0) + points.max(0)) / 2
    radius = side / (2 * context)
    _, _, toward = view_axes(view)
    nearest = (center - aim) @ toward + radius
    enlarged = side * distance / (distance - nearest)
    return middle, max(enlarged, 2 * float(np.abs(points - middle).max()) * context)


def through_lens(frame, spec, verts):
    """A frame carrying its cell's look and, for perspective, the camera's distance and aim."""
    distance, aim = spec.lens(verts)
    return replace(frame, look=spec.look, camera=spec.camera, distance=distance, aim=as_tuple(aim))


def plan_frames(specs, regions=None, mm_per_px=None, cell_px=None):
    """The one scale of a sheet and each pane's frame. Sizes differ; the scale never does."""
    extents = {}
    spheres = {}
    largest = 0.0
    for i, spec in enumerate(specs):
        if spec.focus:
            focus = FOCUS[spec.focus]
            region = regions[spec.focus] if regions else None
            for j, pane in enumerate(spec.panes):
                verts = pane.mesh.verts
                center, side = focus_sphere(verts, region, focus.context)
                distance, aim = spec.lens(verts)
                if distance is not None:
                    center, side = focus_through_lens(
                        verts, region, center, side, focus.context, spec.view, distance, aim
                    )
                spheres[i, j] = (center, side)
                largest = max(largest, side)
        else:
            for j, pane in enumerate(spec.panes):
                distance, aim = spec.lens(pane.mesh.verts)
                extents[i, j] = pane.extent(spec.view, distance, aim)
                u_min, u_max, v_min, v_max = extents[i, j]
                largest = max(largest, u_max - u_min, v_max - v_min)

    if mm_per_px is None:
        target = cell_px or (FOCUS_CELL_PX if spheres else BODY_CELL_PX)
        mm_per_px = pick_scale(largest, target)

    frames = []
    for i, spec in enumerate(specs):
        row = []
        if spec.focus:
            for j in range(len(spec.panes)):
                # One frame size on the sheet, so no render edge shows inside a larger cell.
                if spec.camera == "persp":
                    u, v = spheres[i, j][0]
                    pixels = math.ceil(largest / mm_per_px)
                    frame = Frame(spec.view, float(u), float(v), pixels, pixels, mm_per_px)
                else:
                    frame = focus_frame(spheres[i, j][0], largest, spec.view, mm_per_px)
                row.append(through_lens(frame, spec, spec.panes[j].mesh.verts))
        else:
            # Paired panes share one horizontal centre, so a changed outline shows as a shift.
            own = [extents[i, j] for j in range(len(spec.panes))]
            low, high = min(e[0] for e in own), max(e[1] for e in own)
            for ext, pane in zip(own, spec.panes):
                frame = body_frame(ext, spec.view, mm_per_px, (low + high) / 2, (high - low) / 2)
                row.append(through_lens(frame, spec, pane.mesh.verts))
        frames.append(row)
    return mm_per_px, frames


# --- Layout ---


@dataclass(frozen=True)
class Placed:
    """A pane's render at its whole-pixel place on the sheet."""

    mesh: Mesh
    frame: Frame
    x: int
    y: int
    preset: str
    pose: str
    tag: str
    under: Mesh | None = None


@dataclass(frozen=True)
class Text:
    x: int
    y: int
    text: str
    header: bool = False
    warn: bool = False


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    mm_per_px: float
    placed: list
    texts: list
    ground_lines: list
    boxes: list
    bar: tuple


def scale_bar(mm_per_px, min_px=BAR_MIN_PX):
    """The shortest 1-2-5 length in mm at least min_px long, and that length in pixels."""
    exponent = math.floor(math.log10(min_px * mm_per_px))
    for power in (exponent, exponent + 1):
        for step in (1, 2, 5):
            mm = step * 10**power
            if mm / mm_per_px >= min_px:
                return mm, round(mm / mm_per_px)
    raise AssertionError("a length this long always exists")


def caption(pane, frame):
    if frame.ground is None or pane.under is not None:
        return pane.tag
    z = pane.mesh.verts[:, 2]
    height = f"H {z.max() - z.min():.0f} mm"
    return f"{pane.tag}  {height}" if pane.tag else height


def cell_subject(spec):
    """What a cell shows beyond its body, as its label's second line."""
    coord = spec.grid.coord
    parts = [coord["pose"], spec.focus, spec.view] if spec.focus else [coord["pose"], spec.view]
    if "look" in coord or "camera" in coord:
        parts.append(f"{spec.look} {spec.camera}")
    return " / ".join(parts)


def slot_width(frame, min_slot):
    """A pane's width on the sheet: the render, or more when its labels need the room."""
    return max(frame.width, min_slot, PERSP_SLOT if frame.camera == "persp" else 0)


def layout_sheet(grid, specs, frames, mm_per_px, notes=()):
    """Whole-pixel positions: one ground line per row, cells as wide and tall as they need."""
    rows, cols = len(grid.row_titles), len(grid.col_titles)
    min_slot = MIN_PANE if any(len(s.panes) > 1 for s in specs) else MIN_CELL
    any_persp = any(spec.camera == "persp" for spec in specs)
    all_persp = all(spec.camera == "persp" for spec in specs)
    above, below, width = [0] * rows, [0] * rows, [0] * cols
    for spec, row in zip(specs, frames):
        r, c = spec.grid.row, spec.grid.col
        above[r] = max(above[r], *(f.ground if f.ground is not None else f.height for f in row))
        below[r] = max(below[r], *(f.height - f.ground if f.ground is not None else 0 for f in row))
        slots = sum(slot_width(f, min_slot) for f in row) + PANE_GAP * (len(row) - 1)
        width[c] = max(width[c], slots, len(cell_subject(spec)) * LABEL_CHAR_PX + 8)

    gutter = 0
    if any(grid.row_titles):
        gutter = max(len(title) for title in grid.row_titles) * CHAR_PX + GUTTER
    top = MARGIN + (HEADER if any(grid.col_titles) else 0)
    label = (4 if any_persp else 3) * LINE + 4

    col_x = [MARGIN + gutter]
    for w in width:
        col_x.append(col_x[-1] + w + GUTTER)
    row_y = [top]
    for a, b in zip(above, below):
        row_y.append(row_y[-1] + label + a + b + GUTTER)

    placed, boxes, grounds = [], [], []
    texts = [
        Text(col_x[c], MARGIN, title, True) for c, title in enumerate(grid.col_titles) if title
    ]
    for r, title in enumerate(grid.row_titles):
        if title:
            middle = row_y[r] + label + above[r] // 2
            texts.append(Text(MARGIN, middle, title, True))

    for spec, row in zip(specs, frames):
        cell, r, c = spec.grid, spec.grid.row, spec.grid.col
        x0, y0 = col_x[c], row_y[r]
        pose = cell.coord["pose"]
        texts.append(Text(x0 + 4, y0 + 2, cell.preset))
        texts.append(Text(x0 + 4, y0 + 2 + LINE, cell_subject(spec)))
        boxes.append((x0, y0, x0 + width[c], y0 + label + above[r] + below[r]))
        ground_y = y0 + label + above[r]
        if row[0].ground is not None:
            grounds.append((x0, x0 + width[c], ground_y))

        if spec.camera == "persp":
            texts.append(Text(x0 + 4, y0 + 2 + 3 * LINE, PERSP_CELL_NOTE, warn=True))
        slots = [slot_width(f, min_slot) for f in row]
        used = sum(slots) + PANE_GAP * (len(row) - 1)
        x = x0 + (width[c] - used) // 2
        for i, (pane, frame, slot) in enumerate(zip(spec.panes, row, slots)):
            if i:
                divider = x - PANE_GAP // 2
                boxes.append((divider, y0, divider, y0 + label + above[r] + below[r]))
            text = caption(pane, frame)
            if text:
                texts.append(Text((x if len(row) > 1 else x0) + 4, y0 + 2 + 2 * LINE, text))
            y = ground_y - frame.ground if frame.ground is not None else y0 + label
            px = x + (slot - frame.width) // 2
            placed.append(Placed(pane.mesh, frame, px, y, cell.preset, pose, pane.tag, pane.under))
            x += slot + PANE_GAP

    bottom = row_y[-1]
    bar_mm, bar_px = scale_bar(mm_per_px)
    bar_x, bar_y = MARGIN + gutter, bottom + 16
    # Text at the foot must not be cut off on a sheet of one narrow cell.
    total_width = col_x[-1] - GUTTER + MARGIN
    total_height = bottom + FOOTER
    if all_persp:
        # No perspective cell can back a scale bar, so there is none to misread.
        bar = None
    else:
        bar = (bar_x, bar_y, bar_px)
        where = "orthographic cells only" if any_persp else "orthographic"
        legend = f"{bar_mm:g} mm   ({mm_per_px:g} mm per pixel, {where})"
        texts.append(Text(bar_x + bar_px + 10, bar_y - 8, legend))
        total_width = max(total_width, bar_x + bar_px + 10 + len(legend) * CHAR_PX + MARGIN)
    lines = [(note, False) for note in notes]
    if any_persp:
        lines.append((PERSP_SHEET_NOTE, True))
    for n, (note, warn) in enumerate(lines):
        texts.append(Text(bar_x, bottom + FOOTER - 8 + n * LINE, note, warn=warn))
        total_width = max(total_width, bar_x + len(note) * LABEL_CHAR_PX + MARGIN)
    total_height += LINE * len(lines)
    return Layout(
        total_width,
        total_height,
        mm_per_px,
        placed,
        texts,
        grounds,
        boxes,
        bar,
    )


# --- Rendering ---


def renderer_digest():
    return hashlib.sha256(RENDERER.read_bytes()).hexdigest()


def render_key(digest, frame, wire, renderer=""):
    """Names a render by what goes into it, never by a path or a modification time."""
    payload = {
        "mesh": digest,
        "view": list(VIEWS[frame.view]),
        "center": [round(frame.u, 3), round(frame.v, 3)],
        "size": [frame.width, frame.height],
        "mm_per_px": frame.mm_per_px,
        "wire": bool(wire),
        "look": frame.look,
        "camera": frame.camera,
        "distance": None if frame.distance is None else round(frame.distance, 3),
        "aim": None if frame.aim is None else [round(value, 3) for value in frame.aim],
        "renderer": renderer,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cache_dir():
    return BUILD / "cache" / "review"


def cache_path(mesh, frame, wire):
    return cache_dir() / f"{render_key(mesh.digest, frame, wire, renderer_digest())}.png"


def require_blender():
    blender = find_blender()
    if blender is None:
        named = os.environ.get("BLENDER")
        problem = f"BLENDER={named} does not exist" if named else "Blender not found"
        raise ReviewError(
            f"{problem}. Install Blender and put `blender` on PATH, or set BLENDER to its "
            f"executable, for example BLENDER={BLENDER_APP}",
            EXIT_NO_BLENDER,
        )
    return blender


def shot_entry(frame, wire, out):
    right, up, toward = view_axes(frame.view)
    lens = {}
    if frame.camera == "persp":
        lens_mm, shift_x, shift_y = frame.lens()
        lens = {
            "eye": (np.asarray(frame.aim) + toward * frame.distance).tolist(),
            "distance_mm": frame.distance,
            "lens_mm": lens_mm,
            "sensor_mm": SENSOR_MM,
            "shift": [shift_x, shift_y],
        }
    return {
        **lens,
        "out": str(out),
        "look": frame.look,
        "camera": frame.camera,
        "right": right.tolist(),
        "up": up.tolist(),
        "toward": toward.tolist(),
        "center": frame.world_center().tolist(),
        "width": frame.width,
        "height": frame.height,
        "mm_per_px": frame.mm_per_px,
        "wire": bool(wire),
    }


def run_blender(blender, meshes):
    with tempfile.TemporaryDirectory() as folder:
        manifest = Path(folder) / "manifest.json"
        manifest.write_text(json.dumps({"meshes": meshes}))
        command = [
            str(blender),
            "-b",
            "--factory-startup",
            "--python-exit-code",
            str(EXIT_FAILED),
            "--python",
            str(RENDERER),
            "--",
            str(manifest),
        ]
        done = subprocess.run(
            command, check=False, capture_output=True, text=True, errors="replace"
        )
    if done.returncode != 0:
        tail = "\n".join((done.stdout + done.stderr).splitlines()[-15:])
        raise ReviewError(f"Blender failed with exit code {done.returncode}:\n{tail}")


def ensure_rendered(jobs):
    """Render whatever the cache lacks, all in one Blender process. Returns (rendered, reused)."""
    cache_dir().mkdir(parents=True, exist_ok=True)
    wanted = {}
    for mesh, frame, wire in jobs:
        wanted[cache_path(mesh, frame, wire)] = (mesh, frame, wire)
    missing = {out: job for out, job in wanted.items() if not out.exists()}
    if missing:
        blender = require_blender()
        by_source = {}
        for out, (mesh, frame, wire) in missing.items():
            by_source.setdefault(str(mesh.path.resolve()), []).append(shot_entry(frame, wire, out))
        run_blender(blender, [{"source": s, "shots": shots} for s, shots in by_source.items()])
        absent = [str(out) for out in missing if not out.exists()]
        if absent:
            raise ReviewError(f"Blender finished without writing {absent[0]}")
    return len(missing), len(wanted) - len(missing)


# --- Sheets ---


def change_tile(before, after):
    """Two silhouettes of one frame as one: grey in both, red only after, blue only before.

    Coverage is each render's alpha, so an antialiased edge splits between the colours.
    """
    a = np.asarray(before.convert("RGBA"), float)[..., 3] / 255
    b = np.asarray(after.convert("RGBA"), float)[..., 3] / 255
    both, gained, lost = np.minimum(a, b), np.maximum(b - a, 0), np.maximum(a - b, 0)
    cover = np.maximum(a, b)
    rgb = sum(
        part[..., None] * np.array(colour, float)
        for part, colour in ((both, BOTH_RGB), (gained, GAINED_RGB), (lost, LOST_RGB))
    )
    rgb = rgb / np.maximum(cover, 1e-9)[..., None]
    pixels = np.dstack((rgb, cover * 255)).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


def compose(layout, wire):
    """Paste renders at whole pixels. Nothing here resamples an image."""
    sheet = Image.new("RGBA", (layout.width, layout.height), "white")
    draw = ImageDraw.Draw(sheet)
    for box in layout.boxes:
        draw.rectangle(box, outline=(232, 232, 232))
    for x0, x1, y in layout.ground_lines:
        draw.line((x0, y, x1, y), fill=(170, 170, 170))
    for item in layout.placed:
        tile = Image.open(cache_path(item.mesh, item.frame, wire)).convert("RGBA")
        if item.under is not None:
            tile = change_tile(Image.open(cache_path(item.under, item.frame, wire)), tile)
        if tile.size != (item.frame.width, item.frame.height):
            raise ReviewError(f"{item.mesh.path}: render has size {tile.size}, not the frame's")
        sheet.alpha_composite(tile, (item.x, item.y))
    draw = ImageDraw.Draw(sheet)
    small, large = font(FONT_SIZE), font(HEADER_FONT_SIZE)
    for text in layout.texts:
        color = (170, 70, 0) if text.warn else "black"
        draw.text((text.x, text.y), text.text, fill=color, font=large if text.header else small)
    if layout.bar:
        x, y, length = layout.bar
        draw.rectangle((x, y - 2, x + length - 1, y + 2), fill="black")
    return sheet.convert("RGB")


def cell_specs(grid, build, against=None):
    """Each cell's panes, reading every mesh once. Checks all inputs before any work."""
    paths = []
    for cell in grid.cells:
        paths.append(pose_path(build, cell.preset, cell.coord["pose"]))
        if against:
            paths.append(pose_path(against, cell.preset, cell.coord["pose"]))
    check_inputs(paths)
    meshes = {path: read_mesh(path) for path in dict.fromkeys(paths)}
    specs = []
    for cell in grid.cells:
        current = meshes[pose_path(build, cell.preset, cell.coord["pose"])]
        if against:
            before = meshes[pose_path(against, cell.preset, cell.coord["pose"])]
            panes = (Pane("BEFORE", before), Pane("AFTER", current))
            if cell.look == "silhouette":
                panes += (Pane(CHANGE_TAG, current, before),)
        else:
            panes = (Pane("", current),)
        specs.append(CellSpec(cell, panes))
    return specs


def check_topology(specs, count):
    for spec in specs:
        for pane in spec.panes:
            if len(pane.mesh.verts) != count:
                raise ReviewError(
                    f"{pane.mesh.path} has {len(pane.mesh.verts)} vertices, not the template's "
                    f"{count}: --focus needs the same topology"
                )


def make_sheet(
    axes,
    rows=(),
    cols=(),
    against=None,
    wire=False,
    mm_per_px=None,
    cell_px=None,
    out=None,
    allow_stale=False,
):
    """Render what is missing, compose, save. Returns the layout and the output path."""
    if against and not Path(against).is_dir():
        raise ReviewError(f"--against: {against} is not a directory")
    grid = arrange(axes, rows, cols)
    if wire and any(cell.look != "clay" for cell in grid.cells):
        raise ReviewError("--wire draws on clay only; toon and silhouette have none", EXIT_USAGE)
    specs = cell_specs(grid, bodies_of(BUILD), against and bodies_of(against))
    # Only the current build: --against shows an earlier state on purpose.
    stale = check_fresh(bodies_of(BUILD), [cell.preset for cell in grid.cells], allow_stale)
    regions = None
    if "focus" in axes:
        regions = focus_regions()
        check_topology(specs, len(template_base()["segments"]))
    scale, frames = plan_frames(specs, regions, mm_per_px, cell_px)
    notes = [f"BEFORE = {Path(against).resolve()}"] if against else []
    if any(pane.under is not None for spec in specs for pane in spec.panes):
        notes.append(CHANGE_NOTE)
    if stale:
        notes.append(f"OUT OF DATE, not the current sources: {', '.join(stale)}")
    layout = layout_sheet(grid, specs, frames, scale, notes)
    jobs = [(p.mesh, p.frame, wire) for p in layout.placed]
    jobs += [(p.under, p.frame, wire) for p in layout.placed if p.under is not None]
    rendered, reused = ensure_rendered(jobs)
    sheet = compose(layout, wire)
    if out is None:
        label = None
        if mm_per_px or cell_px:
            label = f"{scale:g}mm"
        out = BUILD / "reports" / "review" / sheet_name(axes, against, wire, label)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(
        f"{out}  {sheet.width}x{sheet.height}  {scale:g} mm/px  rendered {rendered}, cached {reused}"
    )
    return layout, out


def silhouette_rows(pixels, x, y, width, height):
    """First and last pixel rows of a window holding body, from a greyscale sheet array."""
    window = pixels[y : y + height, x : x + width]
    rows = np.nonzero((window < SILHOUETTE_LUMINANCE).any(axis=1))[0]
    if rows.size == 0:
        raise ReviewError(f"no silhouette in the window at ({x}, {y})")
    return y + int(rows[0]), y + int(rows[-1])


def measure_sheet(path, layout):
    """Each body's height as seen on the saved sheet, against its mesh's Z span."""
    pixels = np.asarray(Image.open(path).convert("L"))
    results = []
    for item in layout.placed:
        if item.frame.camera != "ortho":
            raise ReviewError("verify measures orthographic cells only: a lens has no one scale")
        ground = item.frame.ground if item.frame.ground is not None else item.frame.height
        top, last = silhouette_rows(pixels, item.x, item.y, item.frame.width, ground)
        z = item.mesh.verts[:, 2]
        results.append(
            {
                "preset": item.preset,
                "pose": item.pose,
                "view": item.frame.view,
                "z_span": float(z.max() - z.min()),
                "pixels": last - top + 1,
                "gap_to_ground": item.y + ground - 1 - last,
            }
        )
    return results


def ratio_errors(results):
    """Worst disagreement between pixel-height and Z-span ratios, per view and pose."""
    groups = {}
    for row in results:
        groups.setdefault((row["pose"], row["view"]), []).append(row)
    errors = {}
    for key, rows in groups.items():
        worst = 0.0
        for a, b in itertools.combinations(rows, 2):
            worst = max(worst, abs(a["pixels"] / b["pixels"] / (a["z_span"] / b["z_span"]) - 1))
        errors[key] = worst
    return errors


# --- Commands ---


def add_selection(parser):
    parser.add_argument("--pose", help="comma list of poses, or all (default rest)")
    parser.add_argument("--view", help=f"comma list from {', '.join(VIEWS)}, or all")


def render_path(build, preset, pose, view, look="clay", camera="ortho"):
    """The agreed path for clay through ortho; other looks sit beside it, never in its place."""
    suffix = "" if (look, camera) == (LOOKS[0], CAMERAS[0]) else f"-{look}-{camera}"
    return Path(build) / preset / "renders" / pose / f"{view}{suffix}.png"


def look_combinations(look, camera):
    """Every (look, camera) a --look and --camera pair asks for."""
    looks, cameras = pick_looks(look, camera)
    combos = []
    for value in looks:
        name, named = split_look(value)
        combos += [(name, cam) for cam in ([named] if named else cameras)]
    return list(dict.fromkeys(combos))


def cmd_render(argv):
    parser = argparse.ArgumentParser(prog="sotaigen-dev review render")
    parser.add_argument("presets", nargs="+")
    parser.add_argument("--pose", help="comma list of poses (default every pose)")
    parser.add_argument("--view", help=f"comma list from {', '.join(VIEWS)} (default every view)")
    parser.add_argument("--look", help="clay, toon, silhouette, or as toon+persp (default clay)")
    parser.add_argument("--camera", help="ortho or persp (default ortho)")
    parser.add_argument("--allow-stale", action="store_true", help="accept out-of-date poses")
    args = parser.parse_intermixed_args(argv)
    # First, so a machine without Blender always gets the same answer whatever is on disk.
    require_blender()
    names, _, _ = catalog()
    presets = pick_names(",".join(args.presets), names, "preset")
    poses = pick_names(args.pose, POSES, "pose") if args.pose else list(POSES)
    views = pick_names(args.view, VIEWS, "view") if args.view else list(VIEWS)
    combos = [(preset, pose) for preset in presets for pose in poses]
    looks = look_combinations(args.look, args.camera)
    bodies = bodies_of(BUILD)
    check_inputs([pose_path(bodies, preset, pose) for preset, pose in combos])
    check_fresh(bodies, presets, args.allow_stale)
    jobs, targets = [], []
    for preset, pose in combos:
        mesh = read_mesh(pose_path(bodies, preset, pose))
        for view, (look, camera) in itertools.product(views, looks):
            frame = render_canvas(mesh.verts, view, look, camera, preset)
            jobs.append((mesh, frame, False))
            target = render_path(bodies, preset, pose, view, look, camera)
            targets.append((mesh, frame, target))
    rendered, reused = ensure_rendered(jobs)
    for mesh, frame, target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache_path(mesh, frame, False), target)
    print(f"wrote {len(targets)} renders under {bodies}: rendered {rendered}, cached {reused}")
    return 0


def cmd_sheet(argv):
    parser = argparse.ArgumentParser(prog="sotaigen-dev review sheet")
    parser.add_argument("presets", nargs="*", metavar="name", help="body names, instead of --sex and --group")
    parser.add_argument("--sex", help="comma list of sexes, or all (default all)")
    parser.add_argument("--group", help="comma list of groups, or all (default the typical bodies' groups)")
    add_selection(parser)
    parser.add_argument("--look", help="comma list from clay, toon, silhouette, toon+persp, or all")
    parser.add_argument("--camera", help="comma list from ortho, persp")
    parser.add_argument("--focus", help=f"comma list from {', '.join(FOCUS)}, or all")
    parser.add_argument("--rows", default="", help="axes down the sheet, such as sex or pose")
    parser.add_argument("--cols", default="", help="axes across the sheet, such as view or group")
    parser.add_argument("--against", help="another build directory, shown as BEFORE in each cell")
    parser.add_argument("--wire", action="store_true", help="draw the mesh's edges")
    parser.add_argument("--mm-per-px", type=float, help="fix the scale (default fit --cell-px)")
    parser.add_argument("--cell-px", type=int, help="pixels for the largest thing on the sheet")
    parser.add_argument("--allow-stale", action="store_true", help="accept out-of-date poses")
    parser.add_argument("-o", "--out", help="output path (default build/reports/review/<selection>.png)")
    args = parser.parse_intermixed_args(argv)
    if args.mm_per_px is not None and args.mm_per_px <= 0:
        raise ReviewError("--mm-per-px must be positive", EXIT_USAGE)
    axes = select_axes(
        args.presets,
        args.sex,
        args.group,
        args.pose,
        args.view,
        args.focus,
        args.look,
        args.camera,
    )
    make_sheet(
        axes,
        split_names(args.rows),
        split_names(args.cols),
        args.against,
        args.wire,
        args.mm_per_px,
        args.cell_px,
        args.out,
        args.allow_stale,
    )
    return 0


def cmd_verify(argv):
    """Measure a real sheet: pixel heights must keep the meshes' Z-span ratios within 1%."""
    parser = argparse.ArgumentParser(prog="sotaigen-dev review verify")
    parser.add_argument("presets", nargs="*", help="default every typical body")
    add_selection(parser)
    parser.add_argument("--allow-stale", action="store_true", help="accept out-of-date poses")
    parser.add_argument("-o", "--out", help="output path of the measured sheet")
    args = parser.parse_intermixed_args(argv)
    names = catalog()[0]
    presets = args.presets or [name for name in load_typical() if name in names] or names
    axes = select_axes(presets, pose=args.pose, view=args.view)
    out = args.out or BUILD / "reports" / "review" / "verify-scale.png"
    layout, path = make_sheet(
        axes, rows=["pose", "view"], cols=["preset"], out=out, allow_stale=args.allow_stale
    )
    results = measure_sheet(path, layout)
    for row in results:
        measured = row["pixels"] * layout.mm_per_px
        print(
            f"{row['preset']:16} {row['pose']:10} {row['view']:13} Z span {row['z_span']:7.1f} mm"
            f"  {row['pixels']:4d} px = {measured:7.1f} mm  ground gap {row['gap_to_ground']} px"
        )
    errors = ratio_errors(results)
    for (pose, view), error in errors.items():
        print(f"{pose:10} {view:13} worst ratio error {error * 100:.3f}%")
    worst = max(errors.values(), default=0.0)
    if worst > MAX_RATIO_ERROR:
        raise ReviewError(f"ratio error {worst * 100:.3f}% is over {MAX_RATIO_ERROR * 100:g}%")
    print(f"ok: worst ratio error {worst * 100:.3f}%")
    return 0


COMMANDS = {"render": cmd_render, "sheet": cmd_sheet, "verify": cmd_verify}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv in (["--help"], ["-h"]):
        print(__doc__)
        return 0
    if not argv or argv[0] not in COMMANDS:
        print(__doc__, file=sys.stderr)
        return EXIT_USAGE
    try:
        return COMMANDS[argv[0]](argv[1:])
    except ReviewError as error:
        print(f"review: {error}", file=sys.stderr)
        return error.code


if __name__ == "__main__":
    sys.exit(main())
