"""Silhouette extrema of a body: where the side and front outlines turn, to find lumps."""

import sys

import numpy as np

from sotaigen.form.anatomy import calibrated
from sotaigen.form.bodies import load_bodies


def turns(values, heights, prominence=0.8):
    """Local extrema of an outline, ignoring ripples below `prominence` mm."""
    out, last, direction, index = [], values[0], 0, 0
    for i, v in enumerate(values):
        if direction >= 0 and v > last:
            last, index, direction = v, i, 1
        elif direction <= 0 and v < last:
            last, index, direction = v, i, -1
        elif abs(v - last) > prominence and direction != 0:
            out.append(("max" if direction > 0 else "min", round(heights[index]), round(last, 1)))
            direction, last, index = -direction, v, i
    return out


for name in sys.argv[1:] or load_bodies():
    body, L = calibrated(load_bodies()[name])
    T = L["T"]
    zs = np.arange(L["at"](1.0), L["knee"], -5.0)
    xs = np.arange(0.0, 0.34 * T, 4.0)
    ys = np.arange(-0.45 * T, 0.45 * T, 1.0)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    front, back, side = [], [], []
    for z in zs:
        pts = np.column_stack((grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, z)))
        inside = (body(pts) <= 0).reshape(grid_x.shape)
        at_y = np.nonzero(inside.any(0))[0]
        at_x = np.nonzero(inside.any(1))[0]
        front.append(-float(ys[at_y[0]]))
        back.append(float(ys[at_y[-1]]))
        side.append(float(xs[at_x[-1]]))
    print(
        name,
        f"pit {L['at'](1.0):.0f} waist {L['waist']:.0f} crotch {L['crotch']:.0f} knee {L['knee']:.0f}",
    )
    print("  front", turns(front, zs))
    print("  back ", turns(back, zs))
    print("  side ", turns(side, zs, prominence=2.0))
