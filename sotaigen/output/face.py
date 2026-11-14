"""Features, ears and placeholder hair as separate meshes, and the smooth normals of the head's skin.

    from sotaigen.output.face import face_parts, head_normals
    export_body(name, parts=face_parts(name), normals=head_normals(name))

Nothing here changes the skin. Every part has one material and the same vertices in the same order in
every body. The mouth rides the skin under it, so the jaw; the iris, pupil and highlight ride the
eyes; the rest are on the head bone (sotaigen.output.glb's default for a Part). The shapes are plain curves,
drawn for a neutral animation-style face: nobody's face and no hairstyle design.

Features live in face coordinates (u, t): u is x over the head's height, scaled with the head's width
so a feature keeps its share of the face; t is the share of the head down from the crown. A curve is
cast along the view onto the skin and lifted along the skin's normal by the feature's layer, so a
morph target is the same vertices at the same layer on another curve.

This module imports sotaigen.output.glb, which imports sotaigen.pipeline, which imports sotaigen.form.head: neither of those
may import it.
"""

from dataclasses import replace
from functools import cache

import numpy as np

from sotaigen.form.anatomy import calibrated
from sotaigen.form.field import Body, ramp
from sotaigen.form.head import ROWS, SIDES, _cast, _centre, _head_solid, _Skin, _unclipped_body
from sotaigen.mesh import skin_normals
from sotaigen.output.parts import Material, Part
from sotaigen.pipeline import bodies, rigged, surface
from sotaigen.rig.skeleton import EYE_AT, EYE_CLEAR_MM, REFERENCE_HALF_WIDTH

# The march along a ray before bisecting, in mm: a ray from inside the head crosses the skin once.
RAY_STEP = 8.0

# VRM's preset names: one target name means one expression across the parts it moves.
EXPRESSIONS = ("blink", "happy", "aa")

# How far each thin shell stands off the skin, in mm.
LAYERS = {
    "eyeWhite": 0.25,
    "mouth": 0.30,
    "iris": 0.40,
    "brow": 0.40,
    "pupil": 0.50,
    "eyeLight": 0.60,
    "lash": 0.70,
}

# The head width at which u is plain x over the head's height.
REFERENCE_WIDTH = 2 * REFERENCE_HALF_WIDTH

# How far up from the neck the head's skin takes to turn from its own normals to the smooth ones, as
# a share of the head.
NORMAL_BLEND = 0.14
# How much of the smooth normals' up and down turn the face loses, where that starts and how long it
# takes to reach full strength, in t.
LIFT = 0.90
LIFT_FROM, LIFT_OVER = 0.42, 0.22

# Samples along a feature's curve, inner corner to outer.
CURVE = np.linspace(-1.0, 1.0, 15)
EYE_ROWS = 5
DISC_RINGS = 3
DISC_SIDES = 24

EAR_RINGS = 4
EAR_SIDES = 16
EAR_TILT_DEG = 28

CAP_COLUMNS = 72
CAP_ROWS = 22
LOCK_STATIONS = 8
LOCK_SIDES = 6


def _linear(rgb):
    """An sRGB colour as linear, opaque RGBA."""
    c = np.asarray(rgb, float)
    return (*np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).tolist(), 1.0)


# The skin's material in the export, linear. Nose and ears have to read as the same skin.
SKIN_COLOR = (0.8, 0.72, 0.68, 1.0)
COLORS = {
    "eyeWhite": (0.98, 0.98, 0.97),
    "iris": (0.35, 0.42, 0.55),
    "pupil": (0.10, 0.11, 0.16),
    "eyeLight": (0.98, 0.98, 0.97),
    "lash": (0.13, 0.11, 0.12),
    "brow": (0.33, 0.27, 0.24),
    "mouth": (0.42, 0.20, 0.20),
    "hair": (0.30, 0.24, 0.23),
}


class HeadShape:
    """A body's head from its parameters: its skin as a field, and the ways parts reach it.

    `L` is the body's calibrated landmarks, when the caller has them already.
    """

    def __init__(self, p, L=None):
        self.L = calibrated(p)[1] if L is None else L
        self.h = self.L["head"]
        self.H = self.L["H"]
        self.spread = p["headWidthRatio"] / REFERENCE_WIDTH
        self.solid = _head_solid(p, self.L)
        self.skin = _Skin(self.solid, _unclipped_body(p), self.L)
        # The skull and the face alone: no jaw or cheek to break the shading.
        self.smooth = Body(self.solid.groups[:1], [])

    def skin_normal(self, points):
        grad = self.skin.gradient(points, step=1.0)
        return grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-9)

    def smooth_normal(self, points):
        grad = self.smooth.gradient(points, step=2.0)
        return grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-9)

    def on_front(self, u, t, offset=0.0):
        """Face coordinates cast along the view onto the skin, then `offset` mm along its normal."""
        u, t = np.asarray(u, float), np.asarray(t, float)
        x, z = u * self.spread * self.h, self.H - t * self.h
        starts = np.column_stack((x, np.zeros(len(x)), z))
        ahead = np.tile((0.0, -1.0, 0.0), (len(x), 1))
        points = _cast(self.skin, starts, ahead, 0.8 * self.h, step=RAY_STEP)
        return points + self.skin_normal(points) * np.reshape(offset, (-1, 1))

    def around(self, phi, t, offset=0.0):
        """From the head's axis at height t toward phi, onto the skin, then `offset` mm further.

        phi is in radians from the front, positive toward the wearer's left.
        """
        phi, t = np.asarray(phi, float), np.asarray(t, float)
        z = self.H - t * self.h
        starts = np.column_stack((np.zeros(len(z)), np.zeros(len(z)), z))
        out = np.column_stack((np.sin(phi), -np.cos(phi), np.zeros(len(z))))
        hits = _cast(self.skin, starts, out, 0.8 * self.h, step=RAY_STEP)
        return hits + out * np.reshape(offset, (-1, 1))


@cache
def head_shape(name):
    """A body's head, shared: don't change it."""
    return HeadShape(bodies()[name], surface(name)[1])


def unit_rows(values):
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-9)


def mesh_normals(verts, faces):
    """Area-weighted vertex normals of faces given as lists of corners."""
    normals = np.zeros_like(verts)
    for face in faces:
        corners = verts[face]
        total = np.zeros(3)
        for j in range(1, len(face) - 1):
            total += np.cross(corners[j] - corners[0], corners[j + 1] - corners[0])
        normals[face] += total
    return unit_rows(normals)


# The feature curves. All take s in -1..1, inner corner to outer, on the wearer's left; the right side
# is the mirror image. A smaller t is higher on the face.
EYE_CENTRE = (0.175, 0.605)
EYE_HALF_WIDTH = 0.072


def eye_curves(expression, s=CURVE):
    """The eye's (u, upper lid t, lower lid t)."""
    u = EYE_CENTRE[0] + EYE_HALF_WIDTH * s
    tilt = -0.010 * s
    ct = EYE_CENTRE[1]
    if expression == "blink":
        closed = ct + 0.018 * (1 - s * s) + tilt
        return u, closed, closed + 1e-3
    if expression == "happy":
        closed = ct - 0.022 * (1 - s * s) + tilt
        return u, closed, closed + 1e-3
    return u, ct - 0.060 * (1 - s * s) ** 0.6 + tilt, ct + 0.044 * (1 - s * s) ** 0.8 + tilt


def lash_curves(expression, s=CURVE):
    u, upper, _ = eye_curves(expression, s)
    return u, upper - (0.008 + 0.006 * np.clip(s, 0, 1) + 0.002 * (1 - s * s)), upper


def brow_curves(expression, s=CURVE):
    u = 0.178 + 0.082 * s
    middle = 0.490 - 0.014 * (1 - s * s) + 0.010 * s
    if expression == "happy":
        middle = middle - 0.010
    half = 0.0055 * (1 - 0.6 * s * s)
    return u, middle - half, middle + half


def mouth_curves(expression, s=CURVE):
    t0 = 0.828
    if expression == "happy":
        rise = 0.016 * s * s
        return 0.055 * s, t0 + 0.0035 - rise, t0 + 0.0095 - rise
    if expression == "aa":
        return 0.036 * s, t0 - 0.016 * (1 - s * s) ** 0.5, t0 + 0.040 * (1 - s * s) ** 0.5
    return 0.045 * s, t0 - 0.0025 * np.ones_like(s), t0 + 0.0025 * np.ones_like(s)


def eye_disc(expression, centre, radii):
    """A disc in rings of (u, t), squeezed between the lids so that it closes with them."""
    cu = centre[0]
    angle = np.linspace(0, 2 * np.pi, DISC_SIDES, endpoint=False)
    rings = np.arange(1, DISC_RINGS + 1)[:, None] / DISC_RINGS
    u = np.concatenate(([cu], (cu + radii[0] * rings * np.cos(angle)).ravel()))
    t = np.concatenate(([centre[1]], (centre[1] + radii[1] * rings * np.sin(angle)).ravel()))
    s = np.clip((u - EYE_CENTRE[0]) / EYE_HALF_WIDTH, -1, 1)
    _, upper, lower = eye_curves(expression, s)
    return u, np.clip(t, upper, lower)


def strip_points(u, upper, lower, rows):
    share = np.linspace(0, 1, rows)[:, None]
    return np.tile(u, (rows, 1)).ravel(), (upper + (lower - upper) * share).ravel()


def strip_faces(rows, cols):
    """Quads of a strip whose rows run down the face and columns toward the wearer's left."""
    return [
        [i * cols + j, (i + 1) * cols + j, (i + 1) * cols + j + 1, i * cols + j + 1]
        for i in range(rows - 1)
        for j in range(cols - 1)
    ]


def disc_faces():
    """A disc's faces, for angles that turn from +u toward +t, which is clockwise seen from outside."""
    faces = [[0, 1 + (k + 1) % DISC_SIDES, 1 + k] for k in range(DISC_SIDES)]
    for ring in range(DISC_RINGS - 1):
        for k in range(DISC_SIDES):
            a = 1 + ring * DISC_SIDES + k
            b = 1 + ring * DISC_SIDES + (k + 1) % DISC_SIDES
            faces.append([a, b, b + DISC_SIDES, a + DISC_SIDES])
    return faces


def flipped(faces):
    return [list(reversed(face)) for face in faces]


def paired(faces, count):
    """Faces of one side followed by the same faces on its mirror image, the next `count` vertices."""
    return faces + [[i + count for i in face] for face in flipped(faces)]


IRIS = (EYE_AT, (0.037, 0.054))
PUPIL = ((0.172, 0.610), (0.018, 0.028))
LIGHT = ((0.160, 0.588), (0.009, 0.012))


def shell_shapes(expression):
    """Each shell's (u, t) on both sides, the right side's mirrored in u."""
    shapes = {}

    def both(make):
        left, right = make(1), make(-1)
        return np.r_[left[0], -right[0]], np.r_[left[1], right[1]]

    u, up, low = eye_curves(expression)
    shapes["eyeWhite"] = both(lambda _: strip_points(u, up, low, EYE_ROWS))
    for name, (centre, radii) in (("iris", IRIS), ("pupil", PUPIL), ("eyeLight", LIGHT)):
        shapes[name] = both(lambda _, c=centre, r=radii: eye_disc(expression, c, r))
    u2, up2, low2 = lash_curves(expression)
    shapes["lash"] = both(lambda _: strip_points(u2, up2, low2, 2))
    u3, up3, low3 = brow_curves(expression)
    shapes["brow"] = both(lambda _: strip_points(u3, up3, low3, 2))
    shapes["mouth"] = strip_points(*mouth_curves(expression), 3)
    return shapes


def shell_faces():
    n = len(CURVE)
    eye = strip_faces(EYE_ROWS, n)
    disc = disc_faces()
    disc_count = 1 + DISC_RINGS * DISC_SIDES
    line = strip_faces(2, n)
    return {
        "eyeWhite": paired(eye, EYE_ROWS * n),
        "iris": paired(disc, disc_count),
        "pupil": paired(disc, disc_count),
        "eyeLight": paired(disc, disc_count),
        "lash": paired(line, 2 * n),
        "brow": paired(line, 2 * n),
        "mouth": strip_faces(3, n),
    }


def shell_parts(head):
    neutral = {
        name: head.on_front(u, t, LAYERS[name]) for name, (u, t) in shell_shapes(None).items()
    }
    moved = {name: {} for name in neutral}
    for expression in EXPRESSIONS:
        for name, (u, t) in shell_shapes(expression).items():
            offsets = head.on_front(u, t, LAYERS[name]) - neutral[name]
            if (np.abs(offsets) > 1e-4).any():
                moved[name][expression] = offsets
    faces = shell_faces()
    return [
        Part(
            name=name,
            verts=verts,
            faces=faces[name],
            material=Material(_linear(COLORS[name])),
            normals=_lifted(head.smooth_normal(verts), verts, head),
            morphs=moved[name],
        )
        for name, verts in neutral.items()
    ]


NOSE_ROWS = 10
NOSE_COLUMNS = 9
# Below the skin at its edge, so the nose has no visible seam on the skin.
NOSE_SINK_MM = 0.8


def nose_part(head):
    """A small wedge on the midline, highest just above its underside, like a soft fold of skin."""
    # Rows spread down the bridge and gather at the underside, which turns back to the skin quickly.
    share = np.concatenate((np.linspace(0, 0.78, NOSE_ROWS - 3, endpoint=False), [0.78, 0.86, 0.94]))
    share = np.append(share[:-1], 1.0)
    top, bottom = 0.650, 0.742
    t = top + (bottom - top) * share
    across = np.linspace(-1, 1, NOSE_COLUMNS)
    half = 0.028 * (0.30 + 0.70 * np.clip(share / 0.82, 0, 1) ** 0.8)
    ridge = np.where(
        share < 0.82,
        (share / 0.82) ** 1.6,
        np.clip((1 - share) / 0.18, 0, 1) ** 0.6,
    )
    high = 0.026 * head.h
    u_grid = np.outer(half, across)
    t_grid = np.tile(t[:, None], (1, NOSE_COLUMNS))
    lift = (1 - across**2) ** 1.2
    offset = -NOSE_SINK_MM + (high + NOSE_SINK_MM) * np.outer(ridge, lift)
    verts = head.on_front(u_grid.ravel(), t_grid.ravel(), offset.ravel())
    faces = strip_faces(NOSE_ROWS, NOSE_COLUMNS)
    # The edge shades like the skin around it, and only the wedge's own middle like itself.
    own = mesh_normals(verts, faces)
    edge = np.outer(ridge, lift).ravel()[:, None] ** 0.5
    normals = unit_rows(_lifted(head.smooth_normal(verts), verts, head) * (1 - edge) + own * edge)
    return Part(
        name="nose",
        verts=verts,
        faces=faces,
        material=Material(SKIN_COLOR, outline=True),
        normals=normals,
    )


# Where the ear's root meets the skin, in t, and how far behind the head's axis, as shares of the head.
EAR_TOP, EAR_BOTTOM = 0.52, 0.70
EAR_BACK = 0.03
EAR_REACH = 0.075
EAR_DEPTH = 0.016


def one_ear(head):
    """The wearer's left ear: a shallow bowl and a back, joined at the rim, its front in the skin."""
    h = head.h
    middle, half_height = (EAR_TOP + EAR_BOTTOM) / 2, (EAR_BOTTOM - EAR_TOP) / 2
    z = head.H - middle * h
    root = _cast(
        head.skin,
        np.array([[0.0, EAR_BACK * h, z]]),
        np.array([[1.0, 0.0, 0.0]]),
        1.2 * h,
    )[0]
    tilt = np.radians(EAR_TILT_DEG)
    away = np.array([np.sin(tilt), np.cos(tilt), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    outward = np.cross(down, away)
    if outward[0] < 0:
        outward = -outward
    centre = root + away * (0.55 * EAR_REACH * h)
    angle = np.linspace(0, 2 * np.pi, EAR_SIDES, endpoint=False)
    # Wider at the top than the lobe. sin(angle) > 0 is down the ear.
    across = EAR_REACH * h * np.cos(angle) * (1.0 - 0.35 * np.sin(angle))
    along = half_height * h * np.sin(angle)
    verts, rings = [], np.arange(1, EAR_RINGS + 1) / EAR_RINGS
    verts.append(centre - outward * (0.5 * EAR_DEPTH * h))
    for ring in rings:
        bowl = EAR_DEPTH * h * (ring**2 - 0.5)
        verts.extend(
            centre + away * (a * ring) + down * (b * ring) + outward * bowl
            for a, b in zip(across, along, strict=True)
        )
    for ring in rings[::-1][1:]:
        back = -EAR_DEPTH * h * (1.2 - ring**2)
        verts.extend(
            centre + away * (a * ring) + down * (b * ring) + outward * back
            for a, b in zip(across, along, strict=True)
        )
    verts.append(centre - outward * (1.2 * EAR_DEPTH * h))
    verts = np.array(verts)

    # Seen from outside, angle turns from `away` toward `down`.
    clockwise = np.dot(np.cross(away, down), outward) < 0
    faces = []
    ring_at = lambda i, k: 1 + i * EAR_SIDES + k % EAR_SIDES
    last_ring = 2 * EAR_RINGS - 2
    for k in range(EAR_SIDES):
        faces.append([0, ring_at(0, k), ring_at(0, k + 1)])
        faces.append([len(verts) - 1, ring_at(last_ring, k + 1), ring_at(last_ring, k)])
    for i in range(last_ring):
        for k in range(EAR_SIDES):
            faces.append([ring_at(i, k), ring_at(i + 1, k), ring_at(i + 1, k + 1), ring_at(i, k + 1)])
    return verts, flipped(faces) if clockwise else faces


def ears_part(head):
    left, faces = one_ear(head)
    right = left * np.array([-1.0, 1.0, 1.0])
    count = len(left)
    verts = np.vstack((left, right))
    return Part(
        name="ears",
        verts=verts,
        faces=paired(faces, count),
        material=Material(SKIN_COLOR, outline=True),
    )


def smooth_table(degrees, values, blur=25):
    """A curve over |phi| in degrees through the given keys, with its corners rounded."""
    table = np.interp(np.arange(0, 361), degrees, values)
    wrapped = np.r_[table[::-1][:-1], table, table[::-1][1:]]
    kernel = np.ones(blur) / blur
    return np.convolve(np.convolve(wrapped, kernel, "same"), kernel, "same")[360:721]


# Hairline: t where the cap ends, from the forehead round to the nape. Clear of the ear's root.
HAIRLINE = smooth_table(
    [0, 30, 58, 82, 105, 130, 180, 360],
    [0.30, 0.32, 0.45, 0.50, 0.51, 0.70, 0.84, 0.84],
)
# How much the cap stands off the skin toward its rim, as shares of the head, front to back.
CAP_LIFT = smooth_table([0, 60, 110, 180, 360], [0.016, 0.022, 0.030, 0.050, 0.050])
CAP_CROWN = 0.045
# The rim stands this far off the skin before folding under to below it, in mm.
RIM_MM = 1.5
RIM_SINK_MM = 1.0


def of_phi(table, phi):
    return np.interp(np.degrees(np.abs(phi)), np.arange(361), table)


def _polar_rays(phi, polar):
    """Directions from the head's centre: phi round from the front, polar down from the crown."""
    phi, polar = np.broadcast_arrays(phi, polar)
    sin = np.sin(polar)
    return np.stack((sin * np.sin(phi), -sin * np.cos(phi), np.cos(polar)), axis=-1)


def cap_mesh(head):
    """A shell over the skull from the crown to the hairline, its rim folded under to the skin.

    Returns the vertices, the faces and which vertices are the rim's.
    """
    h = head.h
    centre = _centre(head.L)
    phi = (np.arange(CAP_COLUMNS) * 2 * np.pi / CAP_COLUMNS + np.pi) % (2 * np.pi) - np.pi
    rows_out = CAP_ROWS - 1

    # Each column's polar angle at the hairline, read off a table of where its rays meet the skin.
    table = np.linspace(0, np.radians(135), 60)[1:]
    rays = _polar_rays(phi[:, None], table[None, :]).reshape(-1, 3)
    hits = _cast(head.skin, np.tile(centre, (len(rays), 1)), rays, 0.9 * h, step=RAY_STEP)
    hit_t = ((head.H - hits[:, 2]) / h).reshape(CAP_COLUMNS, -1)
    last = np.array(
        [np.interp(of_phi(HAIRLINE, phi[j]), hit_t[j], table) for j in range(CAP_COLUMNS)]
    )

    share = np.arange(1, rows_out + 1) / rows_out
    # The fold repeats the last row's direction, below the skin.
    polar = np.outer(last, np.append(share, 1.0))
    rays = _polar_rays(phi[:, None], polar)
    starts = np.tile(centre, (rays.size // 3, 1))
    skin = _cast(head.skin, starts, rays.reshape(-1, 3), 0.9 * h, step=RAY_STEP)
    skin = skin.reshape(CAP_COLUMNS, CAP_ROWS, 3)

    fade = ramp(share / 0.55)
    # The last stretch comes down to the rim as a quarter circle.
    edge = np.clip((1 - share) / 0.22, 0, 1)
    edge = np.sqrt(1 - (1 - edge) ** 2)
    side = of_phi(CAP_LIFT, phi)[:, None] * h
    lift = CAP_CROWN * h + (side - CAP_CROWN * h) * fade
    lift = RIM_MM + (lift - RIM_MM) * edge
    lift = np.column_stack((lift, np.full(CAP_COLUMNS, -RIM_SINK_MM)))
    ring = skin + rays * lift[..., None]

    top = _cast(head.skin, centre[None], np.array([[0.0, 0.0, 1.0]]), 1.2 * h)[0]
    verts = np.vstack((ring.reshape(-1, 3), top + np.array([0.0, 0.0, CAP_CROWN * h])))
    rim = np.zeros(len(verts), bool)
    rim[:-1] = np.tile(np.arange(CAP_ROWS) >= CAP_ROWS - 2, CAP_COLUMNS)
    pole = len(verts) - 1
    faces = []
    at = lambda i, j: (j % CAP_COLUMNS) * CAP_ROWS + i
    for j in range(CAP_COLUMNS):
        faces.append([pole, at(0, j), at(0, j + 1)])
        for i in range(CAP_ROWS - 1):
            faces.append([at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)])
    return verts, faces, rim


def lock_mesh(line, widths, thickness, outward):
    """A closed flat strand along `line`, S x 3, tapering to a point at its last station."""
    stations = len(line) - 1
    tangent = np.gradient(line, axis=0)
    tangent = unit_rows(tangent)
    side = unit_rows(np.cross(tangent, outward))
    up = unit_rows(np.cross(side, tangent))
    angle = np.linspace(0, 2 * np.pi, LOCK_SIDES, endpoint=False)
    verts = [
        line[i]
        + side[i] * (widths[i] * np.cos(a))
        + up[i] * (thickness[i] * np.sin(a))
        for i in range(stations)
        for a in angle
    ]
    verts.append(line[-1])
    faces = []
    at = lambda i, k: i * LOCK_SIDES + k % LOCK_SIDES
    # The root's end, sunk in the cap, closed so that every lock is a solid.
    faces.append([at(0, k) for k in range(LOCK_SIDES)])
    for i in range(stations - 1):
        for k in range(LOCK_SIDES):
            faces.append([at(i, k), at(i, k + 1), at(i + 1, k + 1), at(i + 1, k)])
    tip = stations * LOCK_SIDES
    for k in range(LOCK_SIDES):
        faces.append([at(stations - 1, k), at(stations - 1, k + 1), tip])
    return np.array(verts), faces


# A lock: root and tip direction in degrees from the front, root and tip t, half width as a share of
# the head. Bangs over the forehead, locks before each ear, and a row hanging down the back. They
# overlap, so they read as one mass with a broken edge.
BANGS = [
    (0, 0, 0.19, 0.455, 0.058),
    (15, 21, 0.18, 0.430, 0.060),
    (31, 38, 0.19, 0.465, 0.060),
    (47, 54, 0.23, 0.500, 0.055),
]
SIDE_LOCKS = [(62, 68, 0.28, 0.66, 0.045), (74, 80, 0.34, 0.80, 0.040)]
BACK_LOCKS = [
    (106, 110, 0.48, 0.78, 0.055),
    (122, 124, 0.52, 0.90, 0.070),
    (138, 140, 0.56, 0.95, 0.075),
    (155, 156, 0.60, 0.99, 0.080),
    (171, 172, 0.62, 1.01, 0.080),
]


def lock_width(half, s, root=0.6):
    """A lock's half width along it: fuller than the root soon after it, pointed only near the tip."""
    return half * np.minimum(root + 2.5 * s, 1.0) * np.sqrt(np.maximum(1.0 - s**2.2, 0.0))


def mirrored(locks):
    """Each lock and its mirror image, one lock on the midline staying single."""
    out = []
    for lock in locks:
        out.append(lock)
        if lock[0] != 0:
            out.append((-lock[0], -lock[1], *lock[2:]))
    return out


def front_lock(head, root_deg, tip_deg, t_root, t_tip, half):
    """A lock following the skin at a small height above it, gathered at its root under the cap."""
    h = head.h
    s = np.linspace(0, 1, LOCK_STATIONS + 1)
    phi = np.radians(root_deg + (tip_deg - root_deg) * s)
    t = t_root + (t_tip - t_root) * s
    lift = (0.016 + 0.008 * (1 - s)) * h
    line = head.around(phi, t, lift)
    outward = np.column_stack((np.sin(phi), -np.cos(phi), np.zeros(len(phi))))
    widths = lock_width(half * h, s)
    thickness = 0.008 * h * (1 - s) ** 0.5
    return lock_mesh(line, widths, thickness, outward)


def back_lock(head, root_deg, tip_deg, t_root, t_tip, half):
    """A lock hanging from under the cap: straight down off the skull, clear of the narrower neck."""
    h = head.h
    s = np.linspace(0, 1, LOCK_STATIONS + 1)
    phi = np.radians(root_deg + (tip_deg - root_deg) * s)
    t = t_root + (t_tip - t_root) * s
    skin = head.around(phi, t)
    radial = np.hypot(skin[:, 0], skin[:, 1])
    z = skin[:, 2]
    # The lock comes in as it falls, at least fast enough to reach the neck's clearance by its tip: a
    # big head on a slim neck (a toddler's) narrows far more over the same drop than an adult's.
    drop = max(z[0] - z[-1], 1e-9)
    slope = max(0.30, (radial[0] + 0.02 * h - (radial[-1] + 0.022 * h)) / drop)
    rest = radial[0] + 0.02 * h - slope * (z[0] - z)
    reach = np.maximum(radial + 0.022 * h, rest)
    out = np.column_stack((np.sin(phi), -np.cos(phi), np.zeros(len(phi))))
    line = np.column_stack((out[:, 0] * reach, out[:, 1] * reach, z))
    widths = lock_width(half * h, s)
    thickness = 0.010 * h * (1 - s) ** 0.5
    return lock_mesh(line, widths, thickness, out)


def hair_part(head):
    """The cap and the locks: enough to hide the crown's back and the nape, not a hairstyle."""
    cap, cap_faces, _ = cap_mesh(head)
    meshes = [(cap, cap_faces)]
    for lock in mirrored(BANGS) + mirrored(SIDE_LOCKS):
        meshes.append(front_lock(head, *lock))
    for lock in mirrored(BACK_LOCKS):
        meshes.append(back_lock(head, *lock))
    verts, faces, at = [], [], 0
    for mesh_verts, mesh_faces in meshes:
        verts.append(mesh_verts)
        faces += [[i + at for i in face] for face in mesh_faces]
        at += len(mesh_verts)
    verts = np.vstack(verts)
    return Part(
        name="hair",
        verts=verts,
        faces=faces,
        material=Material(_linear(COLORS["hair"]), outline=True),
    )


def head_parts(head):
    """Every part of a head. The same parts, vertices and order for every body."""
    return (*shell_parts(head), nose_part(head), ears_part(head), hair_part(head))


# The shells that turn with the eyes.
EYE_SHELLS = ("iris", "pupil", "eyeLight")
# Each row of the mouth's share of the jaw, top to bottom, at its middle. The top row keeps some: the
# skin under it comes forward as it stretches, and a row left on the head would go under it.
MOUTH_ROWS = (0.45, 0.7, 1.0)


def on_eye(head, eye, u, t, offset):
    """Face coordinates cast along the view onto an eye's sphere, `offset` mm off it. `eye` is its bone."""
    centre = np.asarray(eye["head"], float)
    radius = float(np.linalg.norm(np.subtract(eye["tail"], eye["head"]))) + offset
    x, z = np.asarray(u, float) * head.spread * head.h, head.H - np.asarray(t, float) * head.h
    across = (x - centre[0]) ** 2 + (z - centre[2]) ** 2
    y = centre[1] - np.sqrt(np.maximum(radius**2 - across, 0.0))
    return np.column_stack((x, y, z))


def eye_shell(head, part, bones):
    """A shell of the eye on its bone's sphere: open, it turns on that sphere's surface; shut, it lies
    on the skin under the lash as before, so a closed eye shows no shell above the lid."""
    by_name = {bone["name"]: bone for bone in bones}
    u, t = shell_shapes(None)[part.name]
    half = len(u) // 2
    offset = LAYERS[part.name] - EYE_CLEAR_MM

    def placed(u, t):
        return np.vstack(
            [on_eye(head, by_name[f"{side}Eye"], u[rows], t[rows], offset)
             for side, rows in (("left", slice(None, half)), ("right", slice(half, None)))]
        )

    verts = placed(u, t)
    morphs = {}
    for expression in part.morphs:
        moved_u, moved_t = shell_shapes(expression)[part.name]
        morphs[expression] = head.on_front(moved_u, moved_t, LAYERS[part.name]) - verts
    return replace(part, verts=verts, morphs=morphs, normals=_lifted(head.smooth_normal(verts), verts, head))


def bound_parts(head, parts, rig):
    """head_parts on a rig's bones: the eyes' shells on their spheres, and the weights of what moves
    off the head bone. `rig` has verts, bones and weights, as rigged() and the editor's exact body do."""
    names = [bone["name"] for bone in rig.bones]
    out = []
    for part in parts:
        if part.name in EYE_SHELLS:
            part = eye_shell(head, part, rig.bones)
            half = len(part.verts) // 2
            weights = np.zeros((len(part.verts), len(names)))
            weights[:half, names.index("leftEye")] = 1.0
            weights[half:, names.index("rightEye")] = 1.0
            part = replace(part, weights=weights)
        elif part.name == "mouth":
            part = replace(part, weights=mouth_weights(names))
        out.append(part)
    return tuple(out)


def mouth_weights(names):
    """The mouth opens on the jaw: its bottom row on the jaw and its top row mostly on the head, its
    corners halfway between them, as the skin's corners are (sotaigen.rig.skeleton.jaw_share)."""
    rows = np.array(MOUTH_ROWS)[:, None]
    share = 0.5 + (rows - 0.5) * np.sqrt(1 - CURVE**2)
    weights = np.zeros((share.size, len(names)))
    weights[:, names.index("jaw")] = share.ravel()
    weights[:, names.index("head")] = 1 - share.ravel()
    return weights


@cache
def face_parts(name):
    """Every part of a body's head, bound to its rig."""
    head = head_shape(name)
    return bound_parts(head, head_parts(head), rigged(name))


def head_normals(name):
    """A body's whole skin's normals for export, as blended_normals."""
    rig = rigged(name)
    return blended_normals(rig.base, rig.verts, head_shape(name))


def blended_normals(base, verts, head):
    """The whole skin's normals for export: smooth over the head, its own elsewhere.

    The head takes the normals of the skull and face alone, which takes the jaw and the nape's blend
    out of a toon shadow. The turn from its own to those is measured up each column from the row next
    to the template's neck, where they are still its own, so nothing jumps across the seam.
    """
    normals = skin_normals(base, verts)
    members = np.flatnonzero(np.asarray(base["segments"]) == "head")
    points = verts[members]
    # The rows' vertices are in order, a pole last; the first row lies next to the neck.
    heights = points[:, 2] - np.append(np.tile(points[:SIDES, 2], ROWS), points[0, 2])
    share = ramp(heights / (NORMAL_BLEND * head.h))[:, None]
    smooth = _lifted(head.smooth_normal(points), points, head)
    normals[members] = unit_rows(normals[members] * (1 - share) + smooth * share)
    return normals


def _lifted(normals, points, head):
    """Smooth normals with the face's up and down turn mostly taken out, front of the head only.

    The face mass curves under toward the chin, which would put everything below the nose past the
    toon step under any light from above; and under a nod, bands of light and shade would cross the
    face. Flattened, the face takes light together. Left to right it keeps its turn, as the jaw's
    side keeps its own, so the shadow still comes round the far side of the face.
    """
    t = (head.H - points[:, 2]) / head.h
    low = np.clip((t - LIFT_FROM) / LIFT_OVER, 0, 1)
    front = np.clip(-points[:, 1] / (0.20 * head.h), 0, 1)
    share = LIFT * low * low * (3 - 2 * low) * front * front * (3 - 2 * front)
    out = normals.copy()
    out[:, 2] = out[:, 2] * (1 - share)
    return unit_rows(out)
