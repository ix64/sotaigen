"""Give the template's foot the structures it lacks, before it is placed on the leg.

The template's sole is nearly flat and its dorsum a straight ramp. Everything here is a displacement
in the template's own frame, as a share of the foot's length, so each foot scales with its body:

    t  along the foot, 0 at the toe tip and 1 at the heel
    s  across it at that station, 0 on the inner edge and 1 on the outer
    h  above the floor

The length, the lowest point and the ankle ring don't move, which is what placing the foot relies on.
"""

import numpy as np

from sotaigen.form.field import ramp
from sotaigen.mesh import triangulate, vertex_normals

STATIONS = 21

# The medial longitudinal arch: lifted from behind the first metatarsal head to the front of the heel
# pad, highest under the navicular. The outer band stays on the floor as the footprint's isthmus.
ARCH_FROM, ARCH_PEAK, ARCH_TO = 0.30, 0.60, 0.86
ARCH_LIFT = 0.05
ARCH_OUTER_EDGE = 0.62
# The arch's inner wall leans in beneath it, which is what shows it from the side.
ARCH_UNDERCUT = 0.02
# How far up the foot's own height the sole's changes reach.
SOLE_REACH = 0.7

# The transverse arch behind the metatarsal heads: a shallow lift between the ball's two sides.
TRANSVERSE_FROM, TRANSVERSE_TO = 0.30, 0.52
TRANSVERSE_LIFT = 0.012

# The calcaneus rounds up towards the back of the heel.
HEEL_FROM = 0.87
HEEL_LIFT = 0.03

# The instep sits lower than the template's ramp where the arch height index measures it.
INSTEP_FROM, INSTEP_PEAK, INSTEP_TO = 0.36, 0.56, 0.72
INSTEP_DROP = 0.02

# Bumps pushed along X, as (t, h, amplitude, reach along t, reach along h, side): the inner malleolus
# stands higher and further forward than the outer; the hollows behind them frame the Achilles tendon.
INNER, OUTER = -1.0, 1.0
BUMPS = (
    (0.60, 0.205, 0.012, 0.055, 0.045, INNER),  # navicular tuberosity
    (0.76, 0.350, 0.024, 0.045, 0.045, INNER),  # medial malleolus
    (0.80, 0.295, 0.026, 0.045, 0.045, OUTER),  # lateral malleolus
    (0.91, 0.330, -0.012, 0.035, 0.060, INNER),  # behind the medial malleolus
    (0.91, 0.300, -0.012, 0.035, 0.060, OUTER),  # behind the lateral malleolus
)
# Which part of the width a bump works on: its side's share, fading out by the middle.
BUMP_SIDE = 0.45

# Nothing moves near the ankle ring, so the seam stays where the limb expects it.
RING_FADE_FROM, RING_FADE_TO = 0.39, 0.44
# The toes keep their shapes.
TOES_FROM, TOES_TO = 0.16, 0.26


def bump(t, start, peak, end):
    """Rises from start to peak and falls to end."""
    return np.where(t < peak, ramp((t - start) / (peak - start)), 1.0 - ramp((t - peak) / (end - peak)))


def station_line(t, values, pick):
    """`pick` of the values in each station along the foot, softened: the stations and the line."""
    centers = (np.arange(STATIONS) + 0.5) / STATIONS
    bins = np.minimum((t * STATIONS).astype(int), STATIONS - 1)
    picked = np.array([pick(values[bins == i]) if (bins == i).any() else np.nan for i in range(STATIONS)])
    known = ~np.isnan(picked)
    picked = np.interp(centers, centers[known], picked[known])
    padded = np.r_[picked[0], picked, picked[-1]]
    return centers, 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]


def stations(t, values, pick):
    """The same line, read back at each t."""
    return np.interp(t, *station_line(t, values, pick))


def displacement(points, soles):
    """The change to the left foot's points, in its own units. `soles` marks the downward-facing ones."""
    length = np.ptp(points[:, 1])
    floor = points[:, 2].min()
    t = (points[:, 1] - points[:, 1].min()) / length
    h = (points[:, 2] - floor) / length

    inner = stations(t, points[:, 0], np.min)
    outer = stations(t, points[:, 0], np.max)
    s = np.clip((points[:, 0] - inner) / np.maximum(outer - inner, 1e-6), 0.0, 1.0)
    lowest = np.interp(t, *station_line(t[soles], h[soles], np.min))
    top = stations(t, h, np.max)
    rise = np.clip((h - lowest) / np.maximum(top - lowest, 1e-6), 0.0, 1.0)

    behind_toes = ramp((t - TOES_FROM) / (TOES_TO - TOES_FROM))
    near_sole = 1.0 - ramp(rise / SOLE_REACH)
    away_from_ring = 1.0 - ramp((h - RING_FADE_FROM) / (RING_FADE_TO - RING_FADE_FROM))

    arch = bump(t, ARCH_FROM, ARCH_PEAK, ARCH_TO) * (1.0 - ramp(s / ARCH_OUTER_EDGE))
    across = bump(s, 0.15, 0.5, 0.85)
    transverse = bump(t, TRANSVERSE_FROM, 0.5 * (TRANSVERSE_FROM + TRANSVERSE_TO), TRANSVERSE_TO) * across
    heel = ramp((t - HEEL_FROM) / (1.0 - HEEL_FROM)) ** 2

    # The sole comes down to the floor first, keeping how its edges curve up.
    lift = -lowest + ARCH_LIFT * arch + TRANSVERSE_LIFT * transverse + HEEL_LIFT * heel
    dz = lift * near_sole * behind_toes

    instep = bump(t, INSTEP_FROM, INSTEP_PEAK, INSTEP_TO) * ramp((rise - 0.5) / 0.5)
    dz = dz - INSTEP_DROP * instep

    dx = ARCH_UNDERCUT * arch * near_sole
    for at_t, at_h, amount, reach_t, reach_h, side in BUMPS:
        spot = np.exp(-(((t - at_t) / reach_t) ** 2) - ((h - at_h) / reach_h) ** 2)
        on_side = 1.0 - ramp((s if side == INNER else 1.0 - s) / BUMP_SIDE)
        dx = dx + side * amount * spot * on_side

    change = np.zeros_like(points)
    change[:, 0] = dx * length * away_from_ring
    change[:, 2] = dz * length * away_from_ring
    # The softened sole line can run just above the real one: nothing goes through the floor.
    change[:, 2] = np.maximum(change[:, 2], floor - points[:, 2])
    return change


def sculpted(base):
    """The template's vertices with both feet given their structures."""
    raw = np.array(base["verts"], float)
    regions = np.array(base["regions"])
    left = np.nonzero((regions == "foot") & (raw[:, 0] > 0))[0]
    normals = vertex_normals(raw, triangulate(base["faces"], raw))
    change = displacement(raw[left], normals[left, 2] < -0.5)
    out = raw.copy()
    out[left] += change
    by_position = {tuple(v): i for i, v in enumerate(raw.tolist())}
    right = [by_position[(-x, y, z)] for x, y, z in raw[left].tolist()]
    out[right] = out[left] * (-1.0, 1.0, 1.0)
    return out
