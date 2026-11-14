"""Local coordinates for shaping the fused posterior pelvis.

Every control is a distance on the surface, as a share of the torso, and zero leaves the pelvis as it
is. They are measured against the same body without them, not against any survey.
"""

import numpy as np

from sotaigen.form.field import Layer, ramp
from sotaigen.params.spec import FOLD_SPAN

FOLD_HALF_HEIGHT = 0.054
# How far the two flanks' windows may carry the outline up or down. Past this the map folds over, so a
# fullness asked for beyond it stops there: see flank_strength.
UPPER_TRAVEL, LOWER_TRAVEL = 0.043, 0.039
# Where fullness is read: above and below the peak, where each window is flat, on the ridge of the buttock
# at this distance from the midline.
UPPER_STATION, LOWER_STATION = 0.12, -0.10
RIDGE = 0.11
# The crease across the hip, in half hip widths: its depth has halved here on the inner side and this far
# in from its end on the outer one.
FOLD_INNER, FOLD_END_IN = 0.1, 0.085


def contour_ramp(t):
    """C2 easing for a silhouette displacement, with no curvature jump at its ends."""
    u = np.clip(t, 0.0, 1.0)
    return u**3 * (10 - 15 * u + 6 * u * u)


def posterior_depths(body, L, x, z, halvings=14):
    """Find the rear surface at a mesh of lateral positions and torso-height fractions."""
    points = np.column_stack((x.ravel(), np.zeros(x.size), L["at"](z.ravel())))
    low = np.zeros(x.size)
    high = np.full(x.size, 0.60 * L["T"])
    for _ in range(halvings):
        points[:, 1] = (low + high) / 2
        inside = body(points) < 0
        low = np.where(inside, points[:, 1], low)
        high = np.where(inside, high, points[:, 1])
    return low.reshape(z.shape)


def peak_level(body, L, half_width):
    """Find the neutral rear prominence, including the adjoining pelvis and proximal thigh."""
    x, z = np.meshgrid(np.linspace(0, 0.85 * half_width, 16), np.linspace(0.015, 0.24, 46))
    depth = posterior_depths(body, L, x, z)
    return z.ravel()[np.argmax(depth)]


def fold_level(body, L, half_width, original, upper):
    """Keep a low fold inside the steep posterior transition from thigh to buttock."""
    if original >= upper:
        return original
    levels = np.linspace(-0.14, 0.05, 39)
    x, z = np.meshgrid(np.array([0.24, 0.34, 0.44]) * half_width, levels)
    depth = posterior_depths(body, L, x, z).mean(axis=1)
    slope = (depth[4:] - depth[:-4]) / (0.02 * L["T"])
    # Average the broad steep shoulder of the profile rather than selecting one sample.
    weights = np.maximum(slope - (slope.min() + 0.75 * np.ptp(slope)), 0)
    if weights.sum() < 1e-6:
        return original
    transition = np.average(levels[2:-2], weights=weights)
    # The measured strips are centred at x=0.34. Convert to the medial intercept
    # of the oblique crease.
    medial = transition + 0.036 * 0.34 - 0.012 * 0.34**2
    return np.clip(medial, original, upper)


def flank_strength(body, L, peak, station, travel, asked):
    """How much of a flank's travel moves the rear surface at `station` back by `asked`, and what it reaches.

    The window is flat at the station, so moving it reads the neutral outline a little nearer the peak.
    A flat outline can't give what it hasn't got, so the strength stops at one and says how far it got.
    """
    strengths = np.linspace(-1.0, 1.0, 41)
    levels = peak + station - np.sign(station) * travel * strengths
    rear = posterior_depths(body, L, np.full(len(levels), RIDGE * L["T"]), levels, halvings=20) / L["T"]
    gained = rear - rear[len(rear) // 2]
    strength = float(np.interp(asked, np.maximum.accumulate(gained), strengths))
    return strength, float(np.interp(strength, strengths, gained))


class GluteContours(Layer):
    """Read the neutral field through smooth, symmetric local coordinate changes."""

    def __init__(self, body, L, p):
        super().__init__(body)
        self.torso = L["T"]
        self.crotch = L["crotch"]
        self.half_width = self.torso * p["hipBreadth"] / 2
        self.projection = p.get("gluteProjection", 0)
        self.height = p.get("glutePeakShift", 0)
        self.outer = p.get("outerHipShift", 0)
        self.outer_upper = min(0.55, (L["waist"] - L["crotch"]) / L["T"])
        self.fold_depth = 0
        span = 2 * p.get("gluteFoldLength", FOLD_SPAN)
        self.fold_end = span + FOLD_INNER + FOLD_END_IN
        self.peak = peak_level(body, L, self.half_width)
        self.upper, self.upper_reached = 0.0, 0.0
        self.lower, self.lower_reached = 0.0, 0.0
        if p.get("gluteUpperFullness", 0):
            self.upper, self.upper_reached = flank_strength(
                body, L, self.peak, UPPER_STATION, UPPER_TRAVEL, p["gluteUpperFullness"]
            )
        if p.get("gluteLowerFullness", 0):
            self.lower, self.lower_reached = flank_strength(
                body, L, self.peak, LOWER_STATION, LOWER_TRAVEL, p["gluteLowerFullness"]
            )
        if p.get("gluteFoldDepth", 0):
            lower_bound = min(-0.029, self.peak + self.height - 0.17)
            # Up to the thigh transition, never above 0.029 T below the crotch. On the typical bodies
            # the transition sits high enough that the band reaches the lower-fullness section; the
            # user chose the fold at the transition over keeping that section clear (2026-11-06).
            self.fold_level = fold_level(self, L, self.half_width, lower_bound, -0.029)
            self.fold_depth = p["gluteFoldDepth"]

    def _projection(self, x, y, z):
        band = ramp((z + 0.20) / 0.17) * ramp((0.42 - z) / 0.17)
        # Full behind 0.12 T: the flattest buttock, male-compact's, stands 0.146 T behind the centre
        # plane when pulled in all the way, and a ramp to 0.15 T took 0.1% of the asked distance off.
        rear = ramp(y / 0.12) * ramp((1.20 - x) / 0.35)
        depth = self.projection * rear * band
        if self.fold_depth:
            # The inner attachment stays put; only the lateral termination advances.
            crease = self.fold_level - 0.036 * x + 0.012 * x * x
            across = ramp((x - 0.025) / 0.15) * ramp((self.fold_end - x) / 0.17)
            across_fold = np.maximum(1 - ((z - crease) / FOLD_HALF_HEIGHT) ** 2, 0) ** 3
            depth -= self.fold_depth * across * across_fold * ramp(y / 0.13)
        return depth

    def _vertical_shift(self, x, neutral_y, z):
        # Move the lateral prominence vertically. The two changes of height below are
        # monotone at every supported setting, so each neutral point has one image.
        band = ramp((z + 0.20) / 0.17) * ramp((0.42 - z) / 0.17)
        lateral = ramp((x - 0.60) / 0.33) * ramp((1.40 - x) / 0.25)
        rear = ramp(neutral_y / 0.15) * ramp((1.20 - x) / 0.35)
        # The lateral contour needs a longer upper transition than the posterior
        # peak. Sharing the short posterior band made a shoulder above a low hip.
        outer_band = ramp((z + 0.20) / 0.17) * contour_ramp(
            (self.outer_upper - z) / (self.outer_upper - 0.11)
        )
        shift = self.outer * lateral * outer_band + self.height * rear * band
        dz = z - shift - self.peak
        # Displacement and slope vanish at the peak. Stretch either flank of the
        # neutral posterior outline while preserving the peak and the opposite flank.
        upper = ramp(dz / 0.11) * ramp((0.28 - dz) / 0.13)
        lower = ramp(-dz / 0.095) * ramp((0.24 + dz) / 0.12)
        return shift + rear * (UPPER_TRAVEL * self.upper * upper - LOWER_TRAVEL * self.lower * lower)

    def to_reference(self, points):
        q = np.array(points, dtype=float, copy=True)
        x = np.abs(q[:, 0]) / self.half_width
        y = q[:, 1] / self.torso
        z = (q[:, 2] - self.crotch) / self.torso
        depth = self._projection(x, y, z)
        q[:, 1] -= self.torso * depth
        q[:, 2] -= self.torso * self._vertical_shift(x, y - depth, z)
        return q

    def from_reference(self, points):
        """Carry the fitted vertices onto the shaped field without resampling the peak."""
        placed = np.array(points, dtype=float, copy=True)
        active = np.any(self.to_reference(points) != points, axis=1)
        x = np.abs(points[active, 0]) / self.half_width
        y = points[active, 1] / self.torso
        z = (points[active, 2] - self.crotch) / self.torso

        def invert(target, max_shift, shift):
            # Outside the maximum displacement the two ends bracket the unique root.
            low, high = target - max_shift, target + max_shift
            for _ in range(34):
                mid = (low + high) / 2
                below = mid - shift(mid) < target
                low, high = np.where(below, mid, low), np.where(below, high, mid)
            return (low + high) / 2

        shaped_z = invert(z, 0.15, lambda level: self._vertical_shift(x, y, level))
        shaped_y = invert(y, abs(self.projection) + self.fold_depth + 0.001, lambda depth: self._projection(x, depth, shaped_z))
        placed[active, 2] += (shaped_z - z) * self.torso
        placed[active, 1] += (shaped_y - y) * self.torso
        return placed
