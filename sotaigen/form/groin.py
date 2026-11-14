"""The groin's accepted rest shape: the valley between the lower abdomen and the thigh, at rest.

What the masses can't make, sculpted onto the finished skin by the rig stage (sotaigen.pipeline), and
read by the solve where it aims through it. Its pose sculptures are sotaigen/corrective/groin.py's.
Stylised shape control, not a tissue or contact simulation (docs/motion/pelvic-flexion.toml).
"""

import numpy as np

from sotaigen.form.field import ramp
from sotaigen.params.spec import GROIN_DEPTH as DEPTH
from sotaigen.params.spec import GROIN_RELIEF as RELIEF


def coordinates(base, verts, L):
    T = L["T"]
    local = np.isin(base["segments"], ("trunk", "upperLeg"))
    level = (verts[:, 2] - L["crotch"]) / T
    half_width = np.max(np.abs(verts[local & (level > 0) & (level < 0.3), 0]))
    u = np.abs(verts[:, 0]) / half_width
    line = 0.01 + 0.28 * u**1.3
    slope = 0.28 * 1.3 * T / half_width * np.maximum(u, 1e-8) ** 0.3
    across = (level - line) / np.sqrt(1 + slope * slope)
    front = ramp((-verts[:, 1] / T - 0.01) / 0.11) * local
    ends = ramp(u / 0.22) * ramp((1.07 - u) / 0.33)
    return T, u, level, across, front, ends


def rest_verts(base, reference, L):
    T, u, level, across, front, ends = coordinates(base, reference, L)
    outer = ramp((u - 0.52) / 0.42)
    valley = ends * np.exp(-0.5 * (across / (0.039 * (1 + 0.85 * outer))) ** 2)
    valley *= 1 - 0.65 * outer
    abdomen = np.exp(-0.5 * (u / 0.46) ** 2 - 0.5 * ((level - 0.15) / 0.13) ** 2)
    thigh = np.exp(-0.5 * ((u - 0.56) / 0.28) ** 2 - 0.5 * ((across + 0.14) / 0.12) ** 2)
    depth = L.get("groinDepth", DEPTH)
    relief = L.get("groinRelief", RELIEF) / RELIEF
    result = reference.copy()
    result[:, 1] += T * front * (depth * valley - 0.014 * relief * abdomen - 0.008 * relief * thigh)
    return result


def sculpted_depths(depths, heights, L):
    """Front depths, in mm, that rest_verts makes of these at X=0: what the solve aims through.

    Runs the sculpture itself, so that the two can't part. The width only reaches X=0 through a slope
    that vanishes there, and the extra point gives it a value.
    """
    points = np.zeros((len(depths) + 1, 3))
    points[:-1, 1], points[:-1, 2] = -np.asarray(depths), heights
    points[-1] = L["T"] / 2, 0, L["crotch"] + 0.1 * L["T"]
    return -rest_verts({"segments": np.full(len(points), "trunk")}, points, L)[:-1, 1]


def reference_verts(base, rest, L):
    """Invert only the Y sculpture; X/Z and the rig remain those of the reference surface."""
    reference = rest.copy()
    for _ in range(32):
        delta = rest_verts(base, reference, L) - reference
        reference = rest - delta
    if not np.allclose(rest_verts(base, reference, L), rest, atol=1e-8, rtol=0):
        raise ValueError("pelvic rest sculpture could not be inverted")
    return reference
