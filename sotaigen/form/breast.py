"""The breast as soft tissue laid on the chest wall: a thickness field over a footprint, read through the
body's field as a shift along the surface's normal, on top of the pectoral plate (sotaigen/form/pectoral.py).

A mass blended into the chest rounds its upper and lower edge alike, so the shape could only be steered
by unitless warps. A thickness field is what the measurements are: `breastProjection` is the most it
stands out of the surface under it, `breastBaseBreadth` the footprint's width (its height is the same, a
round footprint), `breastFoldDrop` how far below the suprasternal notch the inframammary fold lies,
`breastUpperPoleRatio` the share of the footprint's height above the apex (Mallucci's 45:55 reads 0.45), and
`breastGap` the distance between the two footprints' inner edges (below zero they lap over the midline,
where the two domes meet in a crease; nothing presses them flat), and `breastApexRatio` the share of the
footprint's width between its inner edge and the apex (a body without it has the apex in the middle,
0.5; women's nipples lie about 0.64 of the way out, docs/parts/chest.md). A study; the shape is
stylised, not tissue mechanics.

The tissue stands out along the normal of the surface under it. Pushed straight forward, the outer half of
a breast placed at a real spacing lay on the side of the chest, where forward is nearly along the surface,
and came out twisted.
"""

# Finite-difference step for the surface's normal, mm.
NORMAL_STEP = 0.25

import numpy as np

from sotaigen.form.field import Layer, ramp, soft_fold


class BreastEnvelope(Layer):
    """The body with its chest's front read `thickness` further forward."""

    def __init__(self, body, L, p):
        super().__init__(body)
        T = L["T"]
        self.torso = T
        self.size = p["breastProjection"] * T
        width = p["breastBaseBreadth"] * T
        self.half = width / 2
        self.middle = p.get("breastGap", 0.0) * T / 2 + self.half
        # The apex's distance from the midline, and the footprint's width inside and outside it.
        self.inner_pole = p.get("breastApexRatio")
        if self.inner_pole is not None:
            self.inside = self.inner_pole * width
            self.outside = width - self.inside
            self.apex_x = p.get("breastGap", 0.0) * T / 2 + self.inside
        self.fold = L["pit"] - p["breastFoldDrop"] * T
        self.top = self.fold + width
        self.apex = self.top - p["breastUpperPoleRatio"] * width

    def thickness(self, points):
        # One dome over the footprint, its radius measured from the apex: inwards in the inner pole's
        # width and outwards in the outer's, up in the upper pole's height and down in the lower pole's,
        # so the apex sits where the pole shares say. A product of an across and an up-down profile pinched the apex to a point.
        if self.inner_pole is None:
            u = (soft_fold(points[:, 0]) - self.middle) / self.half
        else:
            du = soft_fold(points[:, 0]) - self.apex_x
            u = np.where(du < 0, du / self.inside, du / self.outside)
        dz = points[:, 2] - self.apex
        v = np.where(dz < 0, dz / (self.apex - self.fold), dz / (self.top - self.apex))
        r2 = u * u + v * v
        # The edge runs out flat at the sternum, the side and the top, and steepens towards the bottom
        # into the fold, where the lower pole meets the chest.
        down = np.clip(-v / np.sqrt(np.maximum(r2, 1e-12)), 0.0, 1.0)
        power = 1.5 - 1.0 * down**2
        dome = np.maximum(1 - r2, 0.0) ** power
        front = ramp(-points[:, 1] / (0.06 * self.torso))
        return self.size * dome * front

    def normal(self, points):
        """The outward unit normal of the field under the tissue: its gradient, by central differences."""
        h = NORMAL_STEP
        grad = np.empty_like(points)
        for axis in range(3):
            step = np.zeros(3)
            step[axis] = h
            grad[:, axis] = self.reference(points + step) - self.reference(points - step)
        return grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-12)

    def to_reference(self, points):
        q = np.array(points, dtype=float, copy=True)
        depth = self.thickness(q)
        moved = depth > 0
        if moved.any():
            # A point stands on the shaped surface when the point `depth` inside it along the normal
            # stands on the surface under the tissue.
            q[moved] -= depth[moved, None] * self.normal(q[moved])
        return q
