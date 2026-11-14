"""What the finished surface measures, so that a body can be asked for in those terms.

Every quantity is in millimetres, on the standing surface with the arms left off the trunk, before the
breast, the muscles and the local contour controls (glutes, abdomen, upper thigh), which move what they move on top of
this. The heights are the ones the survey in docs/PARAMETERS.md measures at, where it has one.
"""

import numpy as np

from sotaigen.form.abdomen import sagittal_depths
from sotaigen.form.anatomy import calibrated
from sotaigen.form.field import Body, frame, unit
from sotaigen.form.thigh import field_sections, section_measures
from sotaigen.params.spec import CONTOURS, layer

# The anthropometric chest point, in shares of the torso above the crotch.
CHEST_LEVEL = 0.71
HIP_LEVELS = (0.05, 0.11, 0.17)
# How far below the chin the neck is read, as a share of the head: under the jaw, clear of the trapezius.
NECK_BELOW_CHIN = 0.008
UPPER_ARM_AT = 0.55
WRIST_ABOVE = 0.04
CALF_AT = (0.3, 0.4, 0.5)
ANKLE_AT = (0.08, 0.16, 0.25)
# The posture measures read how far the trunk's centre line, halfway from front to back on the midline, has
# moved between two heights, in shares of the torso above the crotch. A chord's bulge barely sees a rotation,
# and one surface alone follows the depth of the masses as much as their lean: from the waist to the upper
# chest this leaves the chest's depth nearly out of it, and from the hip to the lower belly the ribs.
UPPER_TRUNK_LEVELS = (0.45, 0.85)
PELVIS_LEVELS = (0.05, 0.25)

# Measured before the local contours, the breast and the muscles, which move what they move on top of these.
LOCAL = (*CONTOURS, *layer("breast", "muscle"))


def neutral(p):
    """The controls these measurements read: no breast, no muscles and no local contours."""
    return {key: value for key, value in p.items() if key not in LOCAL}


def first_exit(field, origins, directions, reach, coarse=48, halvings=9):
    """How far each ray goes before it first leaves the surface.

    Nine halvings resolve reach / 24576: about 0.015 mm at an adult torso, finer than the solve's
    tolerance. Seven left 0.06 mm steps there, which the solve could not settle inside.
    """
    ts = np.linspace(0.0, reach, coarse + 1)
    points = origins[:, None] + ts[None, :, None] * directions[:, None]
    outside = field(points.reshape(-1, 3)).reshape(len(origins), -1) > 0
    # A ray that starts outside or never leaves has no answer here; the caller's station guarantees it does.
    first = np.maximum(outside.argmax(axis=1), 1)
    low, high = ts[first - 1], ts[first]
    for _ in range(halvings):
        middle = (low + high) / 2
        out = field(origins + middle[:, None] * directions) > 0
        low, high = np.where(out, low, middle), np.where(out, middle, high)
    return (low + high) / 2


def ring(angles):
    return np.cos(angles), np.sin(angles)


def section(field, centre, axis, reach, count=48):
    """Girth round an axis: the outline as a tape would follow a convex limb."""
    rotation = frame(unit(axis))
    cos, sin = ring(np.linspace(0, 2 * np.pi, count, endpoint=False))
    directions = np.outer(cos, rotation[:, 0]) + np.outer(sin, rotation[:, 1])
    r = first_exit(field, np.tile(centre, (count, 1)), directions, reach)
    flat = np.column_stack((cos * r, sin * r))
    return float(np.linalg.norm(np.roll(flat, -1, axis=0) - flat, axis=1).sum())


def convex_girth(points):
    """A tape round a horizontal outline: its convex hull's perimeter."""

    def half(sequence):
        hull = []
        for q in sequence:
            while len(hull) > 1 and (
                (hull[-1][0] - hull[-2][0]) * (q[1] - hull[-2][1])
                - (hull[-1][1] - hull[-2][1]) * (q[0] - hull[-2][0])
            ) <= 0:
                hull.pop()
            hull.append(q)
        return hull[:-1]

    order = sorted(map(tuple, points))
    hull = np.array(half(order) + half(order[::-1]))
    return float(np.linalg.norm(np.roll(hull, -1, axis=0) - hull, axis=1).sum())


def level(field, z, y, reach, count=72):
    """A horizontal section from a centre on the midline: its width, depth and tape girth."""
    cos, sin = ring(np.linspace(0, 2 * np.pi, count, endpoint=False))
    directions = np.column_stack((cos, sin, np.zeros(count)))
    r = first_exit(field, np.tile((0.0, y, z), (count, 1)), directions, reach)
    flat = np.column_stack((cos * r, sin * r))
    return {
        "width": float(np.ptp(flat[:, 0])),
        "depth": float(np.ptp(flat[:, 1])),
        "girth": convex_girth(flat),
    }


def measure(p):
    """The measured quantities of a body given by its internal controls."""
    body, L = calibrated(neutral(p))
    T, at = L["T"], L["at"]
    arms = ("armBones", "armMasses")
    # The rib cage, before the axillary hollow (the last carve) eats into its side. That pocket follows the
    # arm and not the ribs, so with it the width barely answers the chest at all (0.2 to 0.5 per mm).
    trunk = Body([g for g in body.groups if g.get("name") not in arms], [], body.carve[:-1])
    shoulder, elbow, wrist = L["shoulderJoint"], L["elbow"], L["wrist"]
    hip, knee, ankle = L["hipJoint"], L["kneeJoint"], L["ankleJoint"]
    shin = ankle - knee
    reach = 0.7 * T
    chest = level(trunk, at(CHEST_LEVEL), 0.0, reach)
    front, back = sagittal_depths(trunk, L, (*PELVIS_LEVELS, *UPPER_TRUNK_LEVELS, p["waist"]))
    hip_line, belly_line, waist_line, chest_line, _ = (back - front) / 2
    return {
        "chestBreadth": chest["width"],
        "chestDepth": chest["depth"],
        "hipDepth": max(level(trunk, at(f), 0.03 * T, reach)["depth"] for f in HIP_LEVELS),
        "neckGirth": level(trunk, L["chin"] - NECK_BELOW_CHIN * L["head"], 0.0, reach)["girth"],
        "upperArmGirth": section(body, shoulder + (elbow - shoulder) * UPPER_ARM_AT, elbow - shoulder, reach),
        "wristGirth": section(body, wrist - L["foreDir"] * WRIST_ABOVE * T, L["foreDir"], reach),
        "kneeGirth": section(body, knee, ankle - hip, reach),
        "calfGirth": max(section(body, knee + shin * f, shin, reach) for f in CALF_AT),
        "ankleGirth": min(section(body, ankle - shin * f, -shin, reach) for f in ANKLE_AT),
        "upperTrunkLean": float(chest_line - waist_line),
        "pelvisLean": float(hip_line - belly_line),
        # The upper thigh where its profile measures it (sotaigen.form.thigh.stations), and the waist's
        # depth at its landmark, front to back along the midline.
        "upperThighGirth": float(section_measures(field_sections(body, L), L)["thighGirth"] * T),
        "waistDepth": float(front[-1] + back[-1]),
    }
