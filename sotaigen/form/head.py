"""A neutral head from the same masses and blends as the body, joined to the template's neck ring.

It is a head with a skull, a jaw and a front plane, not a face: features are parts of their own.
The mesh has one fixed topology for every body: rings of 72 sides from the neck to a pole at the
crown, with the first band doubling the neck ring's 36 vertices.

The neck flows into the head through a blend, as every other junction of the body does. The ring
stops below that blend, so the seam between template and head falls on the plain neck.
"""

import numpy as np

from sotaigen.form.anatomy import calibrated
from sotaigen.form.field import Body, Ellipsoid, ramp, tilt_x
from sotaigen.params.spec import JAW_BASE, JAW_SPAN

SIDES = 72
ROWS = 46
# Polar samples per column, from which the rows are spaced evenly along the surface.
SAMPLES = 180
# How much closer together the rows are on the face than on the skull.
FACE_DENSITY = 2.2
# Column spacing at the front of the face against an even spread; the back takes up the rest.
FRONT_SPACING = 0.6
# How far the jaw's mass tips down at the front, in degrees.
JAW_TIP = 36
# The chin's centre stays at least this far ahead of the neck's front, as a share of the head: the
# neck's thickness doesn't follow the head's depth, and a chin over a thick neck would sit on it.
CHIN_CLEAR = 0.125
# How far down the head the chin's centre is before the fit, as a share of the head. Enough below the
# jaw that the chin stays the head's lowest point, whatever the jaw's width does to the blend.
CHIN_LOW = 0.955
# Blend widths of the neck into the head, as shares of the head: tight under the jaw, where the
# throat turns sharply, and wide at the nape, where the neck flows into the skull.
BLEND_THROAT = 0.08
BLEND_NAPE = 0.16
# Where the neck ring sits: how far below the chin at the head's axis, as a share of the head, and
# how much it rises per millimetre toward the back. Close to the slope the template's ring already
# has: a ring lifted unevenly folds the neck's top rows, on the shortest necks first.
SEAT_BELOW_CHIN = 0.07
SEAT_LEAN = 0.32
# A cubic blend reaches this many times its width along the surface it leaves.
BLEND_REACH = 1.5


def _head_solid(p, L):
    """The head, solved so its crown and chin land on their landmarks whatever the blends add."""
    fit = {"crown": 0.0, "chin": 0.0, "neck": _neck_front(p, L)}
    for _ in range(5):
        field = _head_masses(p, L, fit)
        centre = _centre(L)[None]
        top = _cast(field, centre, np.array([[0.0, 0.0, 1.0]]), 1.2 * L["head"])[0, 2]
        lowest = _lowest(field, p, L)
        fit["crown"] += L["H"] - top
        fit["chin"] += L["chin"] - lowest
    return _head_masses(p, L, fit)


class _SeatPlane:
    """Where wrap.seat_neck_ring stops the ring: a plane below the chin, rising toward the back.

    It leans the way the template's ring already does, so the neck is only lifted a little and
    evenly, and it clears the throat's and the nape's blends, so the seam between template and
    head falls where the neck is still plain.
    """

    def __init__(self, L):
        self.level = L["chin"] - SEAT_BELOW_CHIN * L["head"]

    def __call__(self, points):
        return self.level + SEAT_LEAN * points[:, 1] - points[:, 2]


def head_field(p, L):
    """What wrap.seat_neck_ring slides the ring up to; the head's own shape is _head_solid."""
    return _SeatPlane(L)


def _neck_front(p, L):
    """How far ahead of the head's axis the neck's surface is, at the height of the neck ring."""
    start = np.array([[0.0, 0.0, _SeatPlane(L).level]])
    return -_cast(_unclipped_body(p), start, np.array([[0.0, -1.0, 0.0]]), L["head"])[0, 1]


def _centre(L):
    """Inside the head and over the neck, so rays from here reach the whole skin once."""
    return np.array([0.0, 0.0, L["H"] - 0.52 * L["head"]])


def _lowest(field, p, L):
    """The lowest point of the head on the midline, searching the chin's reach front to back."""
    h = L["head"]
    d = p["headDepth"] * h / 2
    ys = np.linspace(-d, 0.1 * h, 23)
    starts = np.column_stack((np.zeros(23), ys, np.full(23, _centre(L)[2])))
    down = np.tile((0.0, 0.0, -1.0), (23, 1))
    return _cast(field, starts, down, 1.0 * h)[:, 2].min()


def _head_masses(p, L, fit):
    h, H = L["head"], L["H"]
    w, d = p["headWidthRatio"] * h / 2, p["headDepth"] * h / 2
    # The masses' own control, from the jaw's breadth over the head's.
    jaw = (p["jawBreadthRatio"] - JAW_BASE) / JAW_SPAN
    # Heights are measured down from the crown as shares of the head; the chin's follow its fit.
    at = lambda t: H - t * h
    low = lambda t: at(t) + fit["chin"]
    # The skull's base falls above the chin, so it overhangs the nape a little behind the neck.
    skull = Ellipsoid(
        (0.0, -0.03 * d, at(0.43)),
        (w, 0.90 * d, 0.43 * h + fit["crown"]),
        paired=False,
        low=0.40 * h,
    )
    # Face: hangs forward of the skull; its front falling back toward the jaw makes the side view
    # read as facing one way.
    face = Ellipsoid(
        (0.0, -0.36 * d, at(0.58)),
        (0.88 * w, 0.56 * d, 0.42 * h),
        paired=False,
        low=0.34 * h,
    )
    # Jaw: one mass tipped down at the front, so its lower edge climbs from the chin to a soft corner
    # under the ear. Its width is the jaw parameter.
    mandible = Ellipsoid(
        (0.0, -0.37 * d, low(0.79)),
        ((JAW_BASE + JAW_SPAN * jaw) * w, 0.46 * d, 0.18 * h),
        rotation=tilt_x(-JAW_TIP),
        paired=False,
    )
    # Chin: a small round point on the jaw; the jaw parameter sets how wide it stays.
    chin = Ellipsoid(
        (0.0, -max(0.56 * d, fit["neck"] + CHIN_CLEAR * h), low(CHIN_LOW)),
        ((0.05 + 0.07 * jaw) * h, 0.052 * h, 0.05 * h),
        paired=False,
    )
    # Cheek: fullness below the eye, shallow and inside the outline, turning front into side.
    cheek = Ellipsoid((0.40 * w, -0.42 * d, at(0.66)), (0.34 * w, 0.42 * d, 0.16 * h))
    groups = [
        {"prims": [skull, face], "k": 0.12 * h, "join": 0.0},
        {"prims": [mandible, chin], "k": 0.08 * h, "join": 0.06 * h},
        {"prims": [cheek], "k": 0.05 * h, "join": 0.10 * h},
    ]
    return Body(groups, [])


def _unclipped_body(p):
    """The body's own surface, whose neck carries on up inside the head."""
    body, _ = calibrated(p)
    body.clips = []
    for group in body.groups:
        group.pop("clips", None)
    return body


def _blend_width(points, L):
    """How wide the neck's blend into the head is at these points: tight in front, wide behind."""
    h = L["head"]
    behind = ramp((points[:, 1] + 0.10 * h) / (0.20 * h))
    return (BLEND_THROAT + (BLEND_NAPE - BLEND_THROAT) * behind) * h


class _Skin:
    """The surface the head's mesh lies on: the head with the neck blended into it."""

    def __init__(self, solid, neck, L):
        self.solid = solid
        self.neck = neck
        self.L = L

    def __call__(self, points):
        solid, neck = self.solid(points), self.neck(points)
        k = BLEND_REACH * _blend_width(points, self.L)
        h = np.maximum(k - np.abs(solid - neck), 0.0) / k
        return np.minimum(solid, neck) - h * h * h * k / 6.0

    def gradient(self, points, step=1.0):
        return Body.gradient(self, points, step)


def neck_ring_order(base):
    """The neck ring's vertices by azimuth, counter-clockwise seen from above."""
    ring = np.asarray(base["openings"]["neck"])
    raw = np.asarray(base["verts"], float)[ring]
    centre = raw.mean(0)
    return ring[np.argsort(np.arctan2(raw[:, 1] - centre[1], raw[:, 0] - centre[0]))]


def source_count(base):
    """How many of the template's vertices are the source mesh's, the ones before any added loop."""
    refined = base.get("refined")
    return refined["from"] if refined else len(base["verts"])


def topology(base):
    """Faces of the head, numbered after the source mesh's vertices and before any added loops'."""
    ring = neck_ring_order(base)
    assert len(ring) * 2 == SIDES, "the head expects a neck ring of half its sides"
    first = source_count(base)
    row = lambda i, k: first + (i - 1) * SIDES + k % SIDES
    pole = first + ROWS * SIDES
    faces = []
    for j in range(len(ring)):
        a, b = int(ring[j]), int(ring[(j + 1) % len(ring)])
        faces.append([a, row(1, 2 * j + 1), row(1, 2 * j)])
        faces.append([a, b, row(1, 2 * j + 1)])
        faces.append([b, row(1, 2 * j + 2), row(1, 2 * j + 1)])
    for i in range(1, ROWS):
        for k in range(SIDES):
            faces.append([row(i, k), row(i, k + 1), row(i + 1, k + 1), row(i + 1, k)])
    for k in range(SIDES):
        faces.append([row(ROWS, k), row(ROWS, k + 1), pole])
    return faces, ROWS * SIDES + 1


def extended(base):
    """The template with the head's vertices and faces in, ready for the rig.

    The head's block comes right after the source mesh, the added loops' after it, so that adding
    more loops only ever appends: nothing already numbered moves.
    """
    faces, count = topology(base)
    first = source_count(base)
    shifted = lambda i: i + count if i >= first else i
    out = dict(base)
    out["verts"] = insert_head(np.asarray(base["verts"], float), np.zeros((count, 3)), first)
    out["faces"] = [[shifted(i) for i in face] for face in base["faces"]] + faces
    out["regions"] = [*base["regions"][:first], *["head"] * count, *base["regions"][first:]]
    out["segments"] = [*base["segments"][:first], *["head"] * count, *base["segments"][first:]]
    if "landmarks" in base:
        marks = base["landmarks"]
        out["landmarks"] = marks | {"vertices": {k: shifted(v) for k, v in marks["vertices"].items()}}
    if base.get("refined"):
        refined = dict(base["refined"])
        refined["from"] = first + count
        refined["head"] = [first, count]
        refined["templateFaces"] = len(base["faces"])
        refined["parents"] = [[shifted(i) for i in pair] for pair in refined["parents"]]
        out["refined"] = refined
    return out


def insert_head(skin, head, first):
    """The skin's rows with the head's between the source mesh's and the added loops'."""
    return np.vstack((skin[:first], head, skin[first:]))


def template_rows(points, base):
    """The rows of a whole skin (wrapped.npy, a pose) that are the template's, in its order."""
    first = source_count(base)
    count = len(points) - len(base["verts"])
    return np.concatenate((points[:first], points[first + count :]))


def _cast(field, starts, directions, limit, step=4.0):
    """Where rays from inside first leave the field, to a fraction of a millimetre."""
    near = np.zeros(len(directions))
    far = np.full(len(directions), float(limit))
    pending = np.arange(len(directions))
    for t in np.arange(step, limit, step):
        if not len(pending):
            break
        left = field(starts[pending] + directions[pending] * t) > 0
        far[pending[left]] = t
        near[pending[~left]] = t
        pending = pending[~left]
    for _ in range(8):
        middle = (near + far) / 2
        inside = field(starts + directions * middle[:, None]) <= 0
        near = np.where(inside, middle, near)
        far = np.where(inside, far, middle)
    return starts + directions * ((near + far) / 2)[:, None]


def _project(field, points, members):
    value = field(points[members])
    grad = field.gradient(points[members], step=1.0)
    size = np.maximum(np.sum(grad * grad, axis=1, keepdims=True), 1e-6)
    points[members] -= 0.8 * grad * value[:, None] / size


def _turned_to_face(azimuth):
    """Azimuths with the columns drawn toward the front (-Y), keeping the midline where it is."""
    from_front = (azimuth + np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
    share = np.abs(from_front) / np.pi
    turned = from_front * (FRONT_SPACING + (1 - FRONT_SPACING) * share)
    return turned - np.pi / 2


def _face_weight(z, L):
    """How many rows a stretch of skin at this height earns, relative to the skull."""
    t = (L["H"] - z) / L["head"]
    rise = np.clip((t - 0.38) / 0.14, 0.0, 1.0)
    return 1.0 + (FACE_DENSITY - 1.0) * rise * rise * (3 - 2 * rise)


def build_head(base, body_points, p, L):
    """Head vertices for a wrapped body, in the order topology() numbers them."""
    h = L["head"]
    ring = neck_ring_order(base)
    ring_points = body_points[ring]
    skin = _Skin(_head_solid(p, L), _unclipped_body(p), L)
    centre = _centre(L)

    # Ring positions at the head's doubled resolution, with midpoints between the neck vertices.
    doubled = np.empty((SIDES, 3))
    doubled[0::2] = ring_points
    doubled[1::2] = (ring_points + np.roll(ring_points, -1, axis=0)) / 2
    toward = doubled - centre
    polar = np.arccos(np.clip(toward[:, 2] / np.linalg.norm(toward, axis=1), -1, 1))
    azimuth = np.arctan2(toward[:, 1], toward[:, 0])

    # Each column is a curve on the skin from its ring vertex to the crown. Columns start at the
    # ring's azimuths and gather toward the face as they climb.
    share = np.linspace(0.0, 1.0, SAMPLES)
    climb = ramp(share / 0.22)
    angles = polar[None, :] * (1 - share[:, None])
    turned = _turned_to_face(azimuth)
    around = azimuth[None, :] + climb[:, None] * ((turned - azimuth + np.pi) % (2 * np.pi) - np.pi)
    directions = np.stack(
        (np.sin(angles) * np.cos(around), np.sin(angles) * np.sin(around), np.cos(angles)), axis=2
    )
    rays = directions.reshape(-1, 3)
    curves = _cast(skin, np.tile(centre, (len(rays), 1)), rays, 1.2 * h).reshape(SAMPLES, SIDES, 3)

    # Rows are spaced along each curve by arc length, closer together over the face.
    points = np.empty((ROWS * SIDES + 1, 3))
    targets = np.arange(1, ROWS + 1) / (ROWS + 1)
    for k in range(SIDES):
        # A ray that grazes the flaring neck leaves it far below the ring; each curve starts at the
        # ring and takes only what lies above it.
        above = curves[:, k, 2] > doubled[k, 2] + 0.5
        curve = np.vstack((doubled[k], curves[np.argmax(above) :, k]))
        step = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        heights = (curve[1:, 2] + curve[:-1, 2]) / 2
        along = np.concatenate(([0.0], np.cumsum(step * _face_weight(heights, L))))
        place = targets * along[-1]
        for axis in range(3):
            points[k + SIDES * np.arange(ROWS), axis] = np.interp(place, along, curve[:, axis])
    points[-1] = _cast(skin, centre[None], np.array([[0.0, 0.0, 1.0]]), 1.2 * h)[0]

    # Even out the rows along the surface.
    free = np.arange(ROWS * SIDES)
    for _ in range(4):
        grid = points[:-1].reshape(ROWS, SIDES, 3)
        mean = (
            np.roll(grid, 1, axis=1)
            + np.roll(grid, -1, axis=1)
            + np.vstack((grid[:1], grid[:-1]))
            + np.vstack((grid[1:], np.broadcast_to(points[-1], (1, SIDES, 3))))
        ) / 4
        points[free] += 0.3 * (mean.reshape(-1, 3)[free] - points[free])
        _project(skin, points, free)
    _project(skin, points, free)
    return points
