"""Local shoulder surface constraints, independent of the skinning implementation.

The axillary fascia suspends a concave floor between the anterior pectoral and posterior
latissimus/teres folds. A rounded negative volume reserves that hollow in the blended body;
it is a surface proxy, not a model of individual muscles or measured tissue mechanics.
"""

import numpy as np

from sotaigen.form.field import RoundCone, ramp, unit

# How much of a shoulder's setback the axillary pocket follows.
POCKET_FOLLOWS = 0.85
# How wide the pocket blends in, in T: on its medial side, into the chest wall, and on its lateral
# side, into the upper arm. Narrower on the medial side leaves a small S where it meets the wall,
# which the axilla's loops pick up; wider on the lateral side thins the arm where its girth is
# measured, and the solve thickens the whole arm to make up (docs/parts/shoulder-axilla.md).
POCKET_BLEND = (0.06, 0.035)
# Across this much of T about the pocket's axis the blend goes from one width to the other.
POCKET_BLEND_SPAN = 0.05
# The wider medial blend fades out toward the front and back between these distances from the axis,
# in T: a thick arm leaves only a thin wall between the pocket and the front fold's crease, and a
# blend that changes width inside it folds the skin there.
POCKET_BLEND_DEPTH = (0.04, 0.07)


def axillary_vertices(base):
    """The template's anterior fold, suspended floor and posterior fold, from its landmarks."""
    marks = base["landmarks"]["vertices"]
    return marks["axillaFrontFold"], marks["axillaFloor"], marks["axillaBackFold"]


def gap_direction(L):
    """Bisector of the chest wall and the resting upper arm, pointing into the arm gap."""
    return unit(L["armDir"] + (0, 0, -1))


def hollow(L):
    """An open rounded pocket below the joint; its depth leaves the front/back folds intact."""
    T = L["T"]
    # The fossa is medial to the humerus. Centring its cap directly on the gap
    # bisector cuts into the upper arm instead of leaving its muscular wall intact.
    shoulder = L["shoulderJoint"]
    down = gap_direction(L)
    # A shoulder set back leaves the pocket between the two folds, behind only part of the way: at the
    # arm's own place it comes out through the back and takes the posterior fold with it.
    behind = np.array([0.0, POCKET_FOLLOWS * L["shoulderSetback"] * T, 0.0])
    start = shoulder - np.array([0.0, L["shoulderSetback"] * T, 0.0]) + behind
    top, bottom = start + down * 0.11 * T + (-0.035 * T, 0, 0), start + down * 0.48 * T
    pocket = RoundCone(
        top,
        bottom,
        0.025 * T,
        0.09 * T,
        flat=(1.0, 2.5),
        paired=False,
    )

    def local_pocket(points):
        # A raised waist approaches the lower opening. Bend that opening outward
        # smoothly instead of clipping it with a plane (which notches the upper arm).
        waist_gap = (shoulder[2] - L["waist"]) / T
        clearance = 0.06 * T * np.clip((0.4 - waist_gap) / 0.12, 0, 1)
        t = np.clip((shoulder[2] - 0.15 * T - points[:, 2]) / (0.12 * T), 0, 1)
        local = points.copy()
        local[:, 0] -= clearance * t * t * (3 - 2 * t)
        return pocket(local)

    axis = bottom - top
    medial, lateral = POCKET_BLEND

    def blend(points):
        # Medial is toward the midline: points come folded onto the left, so a smaller x.
        along = np.clip((points - top) @ axis / (axis @ axis), 0, 1)
        off = points[:, 0] - (top[0] + along * axis[0])
        share = ramp(0.5 - off / (POCKET_BLEND_SPAN * T))
        depth = np.abs(points[:, 1] - (top[1] + along * axis[1])) / T
        near, far = POCKET_BLEND_DEPTH
        share = share * ramp((far - depth) / (far - near))
        return (lateral + (medial - lateral) * share) * T

    return local_pocket, blend


def landmark_rays(L):
    """The floor and its front/back boundaries get distinct material correspondences."""
    down = unit(gap_direction(L) * 0.11 + (-0.035, 0, 0))
    return [unit(down + (0, y, 0)) for y in (-0.65, 0.0, 0.65)]


def patch_weights(verts, shoulder, torso):
    """Compact shoulder/axilla support, fading before the sternum, neck and elbow."""
    local = (verts - shoulder) / torso
    radius2 = ((local / (0.28, 0.24, 0.34)) ** 2).sum(1)
    return ramp((1 - radius2) / 0.5)
