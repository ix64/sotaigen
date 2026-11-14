"""Measured anterior abdominal arc and lumbar hollow.

Distances are signed fractions of torso length, measured on the mid-sagittal surface. The three
height stations are model conventions, not detected ribs, iliac crests or vertebral landmarks.
"""

import numpy as np

from sotaigen.form.field import Layer, ramp

LOWER, STATION, UPPER = 0.25, 0.45, 0.70
# The least depth a target leaves either side of the central plane, in shares of the torso.
PLANE_CLEARANCE = 0.04


def sagittal_depths(body, L, levels=(LOWER, STATION, UPPER)):
    """Front/back distances from Y=0, in mm, at X=0 at the requested heights."""
    result = []
    for sign in (-1, 1):
        points = np.zeros((len(levels), 3))
        points[:, 2] = L["at"](np.array(levels))
        if np.any(body(points) >= 0):
            raise ValueError("A sagittal measurement station is outside the torso")
        low, high = np.zeros(len(levels)), np.full(len(levels), 0.6 * L["T"])
        for _ in range(26):
            middle = (low + high) / 2
            points[:, 1] = sign * middle
            inside = body(points) < 0
            low, high = np.where(inside, middle, low), np.where(inside, high, middle)
        result.append((low + high) / 2)
    return np.array(result)


def profile_measures(body, L):
    front, back = sagittal_depths(body, L) / L["T"]
    fraction = (STATION - LOWER) / (UPPER - LOWER)
    front_chord = front[0] * (1 - fraction) + front[2] * fraction
    back_chord = back[0] * (1 - fraction) + back[2] * fraction
    return {
        "frontArc": float(front[1] - front_chord),
        "backHollow": float(back_chord - back[1]),
        "depth": float(front[1] + back[1]),
    }


class AbdominalProfile(Layer):
    """Smoothly map a fused body onto measured anterior/posterior chord targets.

    front_arc > 0 means bulging in front of the front chord; back_hollow > 0 means recessed from
    the back chord. Zero means straight at the measurement station, not an unchanged body.
    Only Y moves; both anchor sections, Y=0 and the opposite half of the body stay put.
    """

    def __init__(self, reference, L, p, *, front_arc=None, back_hollow=None):
        super().__init__(reference)
        self.L = L
        self.half_width = p["waistBreadth"] * L["T"] / 2
        measured = profile_measures(reference, L)
        front, back = sagittal_depths(reference, L)[:, 1] / L["T"]
        target_front = front if front_arc is None else front + front_arc - measured["frontArc"]
        target_back = back if back_hollow is None else back - back_hollow + measured["backHollow"]
        # What the map can bend to: its numerical domain, not a human anatomical range, and clear of the
        # central plane. A target past it goes as far as it can, and `reached` says how far, in shares.
        reach_front, reach_back = self._within(target_front, front), self._within(target_back, back)
        self.front_scale, self.back_scale = reach_front / front, reach_back / back
        self.reached = {}
        if front_arc is not None and reach_front != target_front:
            self.reached["abdomenFrontArc"] = front_arc + reach_front - target_front
        if back_hollow is not None and reach_back != target_back:
            self.reached["lumbarHollow"] = back_hollow - (reach_back - target_back)

    @staticmethod
    def _within(target, depth):
        """The target, or the nearest depth the map reaches from this one."""
        return min(max(target, 0.4 * depth, PLANE_CLEARANCE), 1.8 * depth)

    def _scale(self, points):
        level = (points[:, 2] - self.L["crotch"]) / self.L["T"]
        band = ramp(
            np.minimum((level - LOWER) / (STATION - LOWER), (UPPER - level) / (UPPER - STATION))
        )
        lateral = ramp((1.3 - np.abs(points[:, 0]) / self.half_width) / 0.4)
        side = ramp(np.abs(points[:, 1]) / (0.03 * self.L["T"]))
        scale = np.where(points[:, 1] < 0, self.front_scale, self.back_scale)
        return 1 + (scale - 1) * band * lateral * side

    def to_reference(self, points):
        q = np.array(points, dtype=float, copy=True)
        q[:, 1] /= self._scale(q)
        return q

    def from_reference(self, points):
        q = np.array(points, dtype=float, copy=True)
        active = self._scale(q) != 1
        original = q[active].copy()
        trial = original.copy()
        low = np.minimum(original[:, 1] * 0.39, original[:, 1] * 1.81)
        high = np.maximum(original[:, 1] * 0.39, original[:, 1] * 1.81)
        for _ in range(35):
            trial[:, 1] = (low + high) / 2
            below = self.to_reference(trial)[:, 1] < original[:, 1]
            low, high = np.where(below, trial[:, 1], low), np.where(below, high, trial[:, 1])
        q[active, 1] = (low + high) / 2
        return q


def mesh_profile_measures(vertices, triangles, L):
    """The same sagittal distances on the piecewise-linear delivered triangle mesh."""
    surface = vertices[triangles]
    origin, a, b = surface[:, 0], surface[:, 1] - surface[:, 0], surface[:, 2] - surface[:, 0]
    determinant = a[:, 0] * b[:, 2] - a[:, 2] * b[:, 0]
    usable = np.abs(determinant) > 1e-10
    origin, a, b, determinant = origin[usable], a[usable], b[usable], determinant[usable]
    depths = []
    for level in (LOWER, STATION, UPPER):
        x, z = -origin[:, 0], L["at"](level) - origin[:, 2]
        u = (x * b[:, 2] - z * b[:, 0]) / determinant
        v = (a[:, 0] * z - a[:, 2] * x) / determinant
        inside = (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8)
        hits = (origin[:, 1] + u * a[:, 1] + v * b[:, 1])[inside]
        if len(hits) < 2:
            raise ValueError("No anterior/posterior mesh intersections at the measurement station")
        depths.append([-hits.min(), hits.max()])
    front, back = np.array(depths).T / L["T"]
    fraction = (STATION - LOWER) / (UPPER - LOWER)
    return {
        "frontArc": float(front[1] - (1 - fraction) * front[0] - fraction * front[2]),
        "backHollow": float((1 - fraction) * back[0] + fraction * back[2] - back[1]),
        "depth": float(front[1] + back[1]),
    }
