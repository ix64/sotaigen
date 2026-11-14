"""Skeleton landmarks and masses of the body, derived from a small set of semantic parameters.

Layers: proportions -> landmarks (bones) -> masses attached to bones -> blended field.
Z is height above the floor, the front faces -Y and the left is +X. Lengths are mm.
"""

from collections.abc import Callable
from functools import lru_cache
from typing import NamedTuple

import numpy as np

from sotaigen.form.abdomen import AbdominalProfile
from sotaigen.form.axilla import hollow
from sotaigen.form.breast import BreastEnvelope
from sotaigen.form.field import Body, Ellipsoid, RoundCone, frame, peeled, ramp, tilt_x, unit
from sotaigen.form.glutes import GluteContours
from sotaigen.form.pectoral import PectoralPlate
from sotaigen.form.thigh import ThighProfile
from sotaigen.params.spec import (
    ABDOMEN_CONTROLS,
    ARM_REST_DEG,
    CONTOURS,
    GLUTE_CONTROLS,
    GROIN_CONTROLS,
    PROFILES,
    THIGH_CONTROLS,
)
from sotaigen.template import template_landmarks

# A small carrying angle: the forearm leans out from the upper arm, so a hanging hand clears the hip.
# People lean it 5 to 15 degrees; drawn larger, the whole turn at one point reads as a kink.
FOREARM_OUT_DEG = 4.0
FOREARM_FORWARD = 0.12
# The knee is narrower across and front to back than the radius it inherits from the soft limb.
KNEE_SECTION = (0.88, 0.92)


# How much wider the neck's top is than its radius parameter says. Only the end inside the head moves,
# so the base where it meets the trapezius, and everything below, keeps its shape.
NECK_TOP = 1.5
# The neck's and the trapezius's blends into the trunk, as shares of the torso, are at most this many neck
# radii wide: from a radius of 0.0667 T up they keep the full 0.10 T, and a thinner neck keeps a fillet in
# proportion to itself, where a fixed one widens a short thin neck from the shoulders up to the chin.
NECK_BLEND_RADII = 1.5
# The full fillet buries the clavicle's inner end. A narrower one leaves that end standing proud of the top of
# the chest, held only by its own narrow blend: a ledge with an undercut smaller than the template's rows,
# which the wrap folds over. So the end sinks back by this share of the width the fillet gives up
# (docs/parts/head.md).
CLAVICLE_SINK = 0.5
# The humerus's head sits under the acromion, covered by the deltoid. Started at the joint itself, the
# upper arm's round end stood above where the trapezius came down to, and the blends bridged the two into
# a flat shelf over the acromion with a tight bend off its outer end, which a raised arm turns up into a
# knob and then a step. So the bone starts this share of the torso down the arm.
HUMERUS_DROP = 0.03
# How far the deltoid reaches up past its centre, over the humerus's head, in upper arm radii beyond its
# centre's distance down the arm. Short of the head it left a dip between the head and its own belly, a
# bump on the outer arm that the raised arm brings to the top.
DELTOID_REACH = 0.6
# The setback, as a share of the torso, at which the masses from the trunk to the shoulder are all there.
SCAPULAR_FULL_SETBACK = 0.06


def landmarks(p):
    """Bones and heights. Torso fractions come from adult anthropometry, not from any one character."""
    H = p["stature"]
    head = H / p["heads"]
    chin = H - head
    pit = chin - p["neck"] * head
    crotch = p["leg"] * H
    T = pit - crotch
    at = lambda f: crotch + f * T
    knee = crotch * p["kneeHeightRatio"]
    foot_length = p["foot"] * T
    ankle = p["ankleHeight"] * T

    # The femur runs down the middle of the half hip: the thigh fills it from midline to outer edge.
    hip_x = p["hipBreadth"] * T / 4
    # A fuller thigh grows backwards from the femur, so its front stays behind the chest line.
    hip_joint = np.array([hip_x, max(p["thighRadius"] - 0.125, 0.0) * T, at(0.17)])
    # Legs hang so a small gap of `stance` remains between the knees.
    knee_x = p["kneeRadius"] * T * KNEE_SECTION[0] + p["stance"] * T / 2
    knee_joint = np.array([min(knee_x, hip_x), -0.012 * T, knee])
    # The feet keep the same gap: the ankle stands far enough out that the foot stays on its side.
    inner = foot_length * template_landmarks()["footInnerPerFoot"]
    ankle_x = max(knee_joint[0], inner + p["stance"] * T / 2)
    ankle_joint = np.array([ankle_x, 0.012 * T, ankle])

    shoulder_half = p["shoulderWidth"] * T / 2
    setback = 0.012 + p.get("shoulderSetback", 0.0)
    acromion = np.array([shoulder_half - 0.03 * T, setback * T, at(0.985 + p["shoulderRise"])])
    shoulder_joint = acromion + np.array([-0.025 * T, -0.004 * T, -0.05 * T])
    out = np.radians(ARM_REST_DEG)
    arm_dir = np.array([np.sin(out), 0.0, -np.cos(out)])
    elbow = shoulder_joint + arm_dir * p["upperArm"] * T
    fore_out = np.radians(ARM_REST_DEG + FOREARM_OUT_DEG)
    fore_dir = unit([np.sin(fore_out), -FOREARM_FORWARD * np.cos(fore_out), -np.cos(fore_out)])
    wrist = elbow + fore_dir * p["lowerArmLength"] * T
    return {
        **{key: p[key] for key in GROIN_CONTROLS if key in p},
        "H": H,
        "head": head,
        "chin": chin,
        "pit": pit,
        "c7": at(1.10),
        "crotch": crotch,
        "T": T,
        "at": at,
        "knee": knee,
        "ankle": ankle,
        "footLength": foot_length,
        "handLength": p["hand"] * T,
        "hipJoint": hip_joint,
        "kneeJoint": knee_joint,
        "ankleJoint": ankle_joint,
        "shoulderHalf": shoulder_half,
        "shoulderSetback": p.get("shoulderSetback", 0.0),
        "acromion": acromion,
        "shoulderJoint": shoulder_joint,
        "armDir": arm_dir,
        "foreDir": fore_dir,
        "elbow": elbow,
        "wrist": wrist,
        "waist": at(p["waist"]),
    }


def waist_band(level, center):
    """Weight between the fixed upper pelvis and upper chest, in torso-length units."""
    return np.clip(
        np.minimum((level - 0.25) / (center - 0.25), (0.85 - level) / (0.85 - center)), 0, 1
    )


class Trunk(NamedTuple):
    """Breadth and depth of the trunk's masses, in mm."""

    chest_w: float
    chest_d: float
    waist_w: float
    waist_d: float
    hip_w: float
    hip_d: float


class Thigh(NamedTuple):
    """The femur's line and what the thigh's masses are sized from, in mm."""

    point: Callable
    rotation: np.ndarray
    length: float
    radius: float
    side: float
    outer_side: float


class HipSeat(NamedTuple):
    """Where the masses on the pelvis stand, and how far calibration has moved them, in mm."""

    trochanter_x: float
    glute_x: float
    offset: float
    at_midline: bool


def limb_points(start, axis, rotation):
    def point(f, front=0.0, side=0.0):
        """A point on the limb; side is towards the wearer's outside, front towards the front."""
        return start + axis * f + rotation @ np.array([-side, -front, 0.0])

    return point


def thigh_of(p, L):
    axis = L["kneeJoint"] - L["hipJoint"]
    rotation = frame(axis)
    radius = p["thighRadius"] * L["T"]
    # The thigh's depth is its radius; the half hip limits its width.
    side = min(0.92 * p["hipBreadth"] * L["T"] / 4, radius)
    point = limb_points(L["hipJoint"], axis, rotation)
    return Thigh(point, rotation, np.linalg.norm(axis), radius, side, 0.30 * side)


def hip_seat(T, trunk, thigh, fit):
    trochanter_x, glute_x = trunk.hip_w / 2 - 0.145 * T, 0.115 * T
    # One offset moves the trochanter directly, the glute half as far, and the outer
    # thigh along its local lateral axis. Stop where the first paired centre reaches X=0.
    outer_thigh_room = thigh.point(0.24, side=thigh.outer_side)[0] / -thigh.rotation[0, 0]
    inward_room = min(trochanter_x, 2 * glute_x, outer_thigh_room)
    offset = max(fit.get("hip", 0.0), -inward_room)
    return HipSeat(trochanter_x, glute_x, offset, bool(offset <= -inward_room))


def core_masses(p, L, trunk, hip_offset):
    T, at = L["T"], L["at"]
    # The thorax leans back and the pelvis tips forward: the wedge between them is the lumbar curve.
    rib_lean = p["ribTilt"]
    pelvis_lean = -p["pelvisTilt"]
    return [
        # Thorax: fullest at the lower ribs, narrowing up to the first rib and closing above the waist.
        Ellipsoid(
            (0.0, 0.008 * T, at(0.69)),
            (trunk.chest_w / 2, trunk.chest_d / 2, 0.34 * T),
            tilt_x(rib_lean),
            taper=0.12,
            paired=False,
            low=0.25 * T,
        ),
        # Shoulder girdle on top of the ribs, so the chest reaches the shoulders without a gap.
        Ellipsoid(
            (0.0, 0.022 * T, at(0.90)),
            (L["shoulderHalf"] * 0.80, trunk.chest_d / 2 * 0.66, 0.15 * T),
            tilt_x(rib_lean),
            paired=False,
        ),
        # Soft belly between the rib cage and the pelvis: its width is the waist.
        Ellipsoid(
            (0.0, -0.006 * T, at(p["waist"] - 0.05)),
            (trunk.waist_w / 2, trunk.waist_d / 2 * 0.94, 0.21 * T),
            paired=False,
        ),
        # Pelvic bowl: its top is the iliac crest.
        Ellipsoid(
            (0.0, 0.028 * T, at(0.12)),
            (trunk.hip_w / 2 * 0.80 + hip_offset, trunk.hip_d / 2 * 0.78, 0.34 * T),
            tilt_x(pelvis_lean),
            taper=0.34,
            paired=False,
            low=0.15 * T,
        ),
    ]


def trunk_form_masses(p, L, trunk):
    T, at = L["T"], L["at"]
    tone = p["definition"]
    chest_w, chest_d, waist_d, hip_d = trunk.chest_w, trunk.chest_d, trunk.waist_d, trunk.hip_d
    lat = p.get("latissimusThickness", 0.0) * T
    return [
        # Latissimus: from below the armpit towards the waist, the V of the back. It is what keeps
        # the chest's side a single wall up to the armpit instead of closing over the ribs.
        # A male body's `latissimus` flares it out below the armpit and fills it towards the waist.
        RoundCone(
            (chest_w / 2 * 0.76 + 1.2 * lat, 0.03 * T + 0.5 * lat, at(0.88) - 1.5 * lat),
            (chest_w / 2 * 0.72 + 0.2 * lat, 0.03 * T, at(0.56)),
            0.05 * T * (0.9 + 0.6 * tone) + 1.6 * lat,
            0.024 * T * (0.9 + 0.6 * tone) + 0.5 * lat,
            flat=(0.8, 1.0),
        ),
        # Erector columns: a shallow furrow down the lower back.
        RoundCone(
            (0.042 * T, chest_d / 2 * (0.6 + 0.3 * tone), at(0.70)),
            (0.044 * T, waist_d / 2 * (0.62 + 0.3 * tone), at(0.40)),
            0.034 * T * (0.7 + 0.75 * tone),
            0.032 * T * (0.7 + 0.75 * tone),
        ),
        RoundCone(
            (0.044 * T, waist_d / 2 * (0.62 + 0.3 * tone), at(0.40)),
            (0.044 * T, hip_d / 2 * (0.5 + 0.25 * tone), at(0.26)),
            0.032 * T * (0.7 + 0.75 * tone),
            0.026 * T * (0.7 + 0.75 * tone),
        ),
    ]


def neck_blend(p):
    """How wide the neck and the trapezius blend into the trunk, as a share of the torso before softness."""
    return min(0.10, NECK_BLEND_RADII * p["neckRadius"])


def clavicle_masses(p, L, trunk):
    """Its strength is the style's definition, from buried to a visible ridge."""
    T, at = L["T"], L["at"]
    tone = p["definition"]
    sink = CLAVICLE_SINK * (0.10 - neck_blend(p)) * T * p["softness"]
    return [
        RoundCone(
            (0.05 * T, -trunk.chest_d / 2 * 0.50 + sink, at(0.99)),
            L["acromion"] + np.array([-0.03 * T, -0.016 * T, -0.006 * T]),
            (0.004 + 0.012 * tone) * T,
            (0.008 + 0.010 * tone) * T,
        ),
    ]


def trapezius_masses(p, L):
    T, at = L["T"], L["at"]
    tone = p["definition"]
    # A male body's `trapezius` thickens the sheet and lifts its neck end: a steeper neck-to-shoulder line.
    trap = p.get("trapeziusThickness", 0.0) * T
    return [
        # A sloping sheet from the neck to the acromion.
        RoundCone(
            (0.05 * T, 0.05 * T, at(1.02 + 0.11 * tone) + 1.2 * trap),
            L["acromion"] + np.array([-0.04 * T, 0.008 * T, -0.018 * T]),
            (0.035 + 0.035 * tone) * T + trap,
            0.03 * T + 0.5 * trap,
            flat=(1.0, 0.9),
        ),
    ]


def scapular_masses(p, L):
    """A shoulder set back stands off the ribs: these carry the upper back and the chest's side out to it.
    None without a setback, and they come in gradually, so a small setback stays near the body without
    one."""
    setback = p.get("shoulderSetback", 0.0)
    if not setback:
        return []
    T, joint = L["T"], L["shoulderJoint"]
    share = min(setback / SCAPULAR_FULL_SETBACK, 1.0)
    return [
        RoundCone(
            (0.17 * T, 0.085 * T, joint[2] + 0.02 * T),
            joint + np.array([-0.01 * T, -0.005 * T, 0.01 * T]),
            0.055 * T * share,
            0.042 * T * share,
        ),
        # The front of the armpit, which would otherwise open into a slot: the chest's side out to the arm.
        RoundCone(
            (0.17 * T, -0.02 * T, joint[2] - 0.01 * T),
            joint + np.array([-0.01 * T, -0.025 * T, -0.02 * T]),
            0.05 * T * share,
            0.036 * T * share,
        ),
    ]


def arm_masses(p, L):
    """The arm's bones and the masses on them."""
    T = L["T"]
    tone = p["definition"]
    delt = p.get("deltoidThickness", 0.0) * T
    arm_out = unit(np.cross(L["armDir"], (0.0, 1.0, 0.0)))
    arm_out = arm_out if arm_out[0] > 0 else -arm_out
    arm_r = p["armRadius"] * T
    wrist_r = p["wristRadius"] * T
    # The elbow is the arm's slimmest turn: mostly the upper arm's size, only part of the wrist's.
    elbow_r = arm_r * 0.60 + wrist_r * 0.50
    fore_len = np.linalg.norm(L["wrist"] - L["elbow"])
    fore_rot = frame(L["foreDir"])
    fore_out = unit(np.cross(L["foreDir"], (0.0, 1.0, 0.0)))
    fore_out = fore_out if fore_out[0] > 0 else -fore_out
    # Upper arm and forearm as one chain, like the leg's, so the elbow doesn't swell into a ring.
    bones = [
        RoundCone(L["shoulderJoint"] + L["armDir"] * HUMERUS_DROP * T, L["elbow"], arm_r * 1.08, elbow_r),
        # Forearm continues past the wrist so the template's ring finds a surface; cut when meshed.
        # Round at the elbow, flattening towards the wrist as the two bones lie side by side.
        RoundCone(
            L["elbow"],
            L["wrist"] + L["foreDir"] * 0.12 * T,
            elbow_r,
            wrist_r * 0.92,
            flat_end=(1.0, 0.80),
        ),
    ]
    masses = [
        # Deltoid wraps the head of the arm: a cap over the joint, drawn out and narrowing slowly down
        # the arm so that a raised arm reads as one form, without a ball on top or a step below it.
        # Its upper end reaches over the humerus's head (DELTOID_REACH), so from the trapezius down the
        # outer arm the front outline is one convex arc.
        # A male body's `deltoid` thickens the cap on its own, apart from the arm's girth.
        # It grows outwards from the arm, so none of it reaches into the armpit.
        Ellipsoid(
            L["shoulderJoint"] + L["armDir"] * 0.08 * T + arm_out * 0.8 * delt,
            (arm_r * (0.92 + 0.3 * tone) + delt, arm_r * (0.98 + 0.3 * tone) + 0.6 * delt, 0.32 * T),
            frame(L["armDir"]),
            taper=0.45,
            low=0.08 * T + DELTOID_REACH * arm_r,
        ),
        # Forearm: extensors make the outer peak high by the elbow, flexors the inner one lower.
        Ellipsoid(
            L["elbow"] + L["foreDir"] * fore_len * 0.20 + fore_out * 0.22 * elbow_r,
            (elbow_r * (0.72 + 0.2 * tone), elbow_r * (0.72 + 0.2 * tone), fore_len * 0.34),
            fore_rot,
        ),
        Ellipsoid(
            L["elbow"] + L["foreDir"] * fore_len * 0.36 - fore_out * 0.22 * elbow_r,
            (elbow_r * (0.70 + 0.2 * tone), elbow_r * (0.72 + 0.2 * tone), fore_len * 0.40),
            fore_rot,
        ),
    ]
    return bones, masses


def neck_masses(p, L):
    T, at = L["T"], L["at"]
    neck_r = p["neckRadius"] * T
    return [
        RoundCone(
            (0.0, 0.04 * T, at(0.98)),
            (0.0, 0.0, L["chin"] + 0.42 * L["head"]),
            neck_r * 1.4,
            neck_r * 0.92 * NECK_TOP,
            paired=False,
        ),
    ]


def leg_masses(p, L, thigh, hip_offset):
    """The leg's bones, the masses on the thigh and the masses on the shin."""
    T = L["T"]
    hip_joint, knee_joint, ankle_joint = L["hipJoint"], L["kneeJoint"], L["ankleJoint"]
    # Muscle offsets and detail radii grow with definition; 0.4 is their anatomical reading.
    form = 0.4 + 1.5 * p["definition"]
    along = thigh.point
    thigh_r, thigh_side = thigh.radius, thigh.side
    knee_r = p["kneeRadius"] * T
    calf_r = p["calfRadius"] * T
    ankle_r = p["ankleRadius"] * T
    shin_axis = ankle_joint - knee_joint
    shin_len = np.linalg.norm(shin_axis)
    shin_rot = frame(shin_axis)
    down = limb_points(knee_joint, shin_axis, shin_rot)

    # Femur and shin as one chain with a shared knee: blended widely they would swell into a ring
    # exactly where the limb should be slimmest.
    bones = [
        RoundCone(
            hip_joint,
            knee_joint,
            thigh_r,
            knee_r,
            flat=(thigh_side / thigh_r, 1.03),
            flat_end=KNEE_SECTION,
        ),
        RoundCone(
            knee_joint,
            ankle_joint + np.array([0.0, 0.0, -0.08 * T]),
            knee_r,
            ankle_r * 0.92,
            flat=KNEE_SECTION,
            flat_end=(1.0, 1.0),
        ),
    ]
    on_thigh = [
        # One anterior envelope represents the combined quadriceps contour.
        Ellipsoid(
            along(0.40, front=0.26 * thigh_r * form),
            (thigh_side * 0.80, thigh_r * 0.70, thigh.length * 0.38),
            thigh.rotation,
        ),
        # One posterior envelope represents the combined hamstring contour.
        Ellipsoid(
            along(0.42, front=-0.28 * thigh_r * form),
            (thigh_side * 0.76, thigh_r * 0.62, thigh.length * 0.42),
            thigh.rotation,
        ),
        # Adductors fill the inner thigh high up and taper fast.
        Ellipsoid(
            along(0.20, side=-0.34 * thigh_side),
            (thigh_side * 0.60, thigh_r * 0.68, thigh.length * 0.22),
            thigh.rotation,
        ),
        # Outer thigh: carries the width of the hip down the leg instead of stopping at the fold.
        Ellipsoid(
            along(0.24, side=thigh.outer_side + min(hip_offset, 0.0)),
            (thigh_side * 0.72, thigh_r * 0.82, thigh.length * 0.34),
            thigh.rotation,
        ),
    ]
    # The shin bone runs straight at the front; the calf hangs behind it: straight against curve.
    on_shin = [
        # Outer head higher, inner head lower: the two outlines peak apart. Offset only a little, as
        # a stronger or higher outer head reads as a shelf just below the knee.
        Ellipsoid(
            down(0.33, front=-0.33 * calf_r, side=0.12 * calf_r),
            (calf_r * 0.74, calf_r * 0.77, shin_len * 0.27),
            shin_rot,
        ),
        Ellipsoid(
            down(0.41, front=-0.34 * calf_r, side=-0.14 * calf_r),
            (calf_r * 0.74, calf_r * 0.77, shin_len * 0.30),
            shin_rot,
        ),
    ]
    return bones, on_thigh, on_shin


def hip_masses(L, trunk, seat):
    """What sits on the pelvis rather than on the femur."""
    T, at = L["T"], L["at"]
    return [
        # Greater trochanter and its padding: the true width of the hip, well below the crest.
        Ellipsoid(
            # The wide blend into the pelvis adds about 0.035 T, so the mass starts inside by that.
            (seat.trochanter_x + seat.offset, 0.012 * T, at(0.05)),
            # Its front stays behind the lower belly's, which goes with the hip's depth: a shallow
            # pelvis whose narrow hip draws it in would otherwise stand the thigh roots ahead of the
            # midline. From 0.3125 T of hip depth up it is the full 0.125 T.
            (0.105 * T, min(0.125 * T, 0.40 * trunk.hip_d), 0.17 * T),
            low=0.26 * T,
        ),
        # Gluteal mass: tilted, fullest below the hip joint, ending at the fold. Its long upper side
        # rises slowly out of the lumbar curve, so the back reads as one slope down to the fullness.
        Ellipsoid(
            (seat.glute_x + seat.offset * 0.5, trunk.hip_d / 2 * 0.46, at(0.08)),
            (0.125 * T, trunk.hip_d / 2 * 0.46, 0.20 * T),
            tilt_x(-16),
            low=0.12 * T,
        ),
    ]


def lower_belly_masses(p, L, trunk):
    T, at = L["T"], L["at"]
    return [
        # The lower abdominal wall narrows onto the pubis between the two thigh roots.
        Ellipsoid(
            (0.0, -0.215 * trunk.hip_d, at(0.115)),
            (0.25 * trunk.hip_w, 0.20 * trunk.hip_d, 0.18 * T),
            tilt_x(-p["pelvisTilt"]),
            low=0.155 * T,
            paired=False,
        ),
    ]


def thigh_offset(p, L, thigh):
    """thighFrontShift's shift of the thigh towards the back at each height, in mm, or None if not asked.

    Around the same femur, with no shear at either joint: it peaks halfway between them.
    """
    bias = p.get("thighFrontShift", 0.0)
    if not bias:
        return None
    knee_z, hip_z = L["kneeJoint"][2], L["hipJoint"][2]

    def offset(z):
        u = np.clip((z - knee_z) / (hip_z - knee_z), 0, 1)
        return bias * L["T"] * (4 * u * (1 - u)) ** 2

    return offset


def offset_coordinates(offset):
    """The group entry that reads paired masses through a thigh offset, if there is one."""
    if offset is None:
        return {}

    def coordinates(points):
        shifted = points.copy()
        shifted[:, 1] += offset(points[:, 2])
        return shifted

    return {"paired_coordinates": coordinates}


def thigh_coordinates(p, L, trunk, thigh):
    """The coordinates the paired thigh masses are read through."""
    T, at = L["T"], L["at"]
    hip_w, hip_d = trunk.hip_w, trunk.hip_d
    offset = thigh_offset(p, L, thigh)

    def coordinates(points):
        # The front of the thigh recedes under the abdominal wall along the
        # oblique course from the pubis toward the anterior iliac spine.
        x, y, z = points.T
        lateral = x / (0.5 * hip_w)
        level = at(0.01 + 0.22 * lateral)
        root = np.exp(-(((z - level) / (0.11 * T)) ** 2) - ((lateral - 0.48) / 0.48) ** 4)
        front = np.clip(-y / (0.28 * hip_d), 0, 1)
        shifted = points.copy()
        shifted[:, 1] -= 0.065 * T * root * front**2 * (3 - 2 * front)
        if offset is not None:
            shifted[:, 1] += offset(z)
        return shifted

    return coordinates


def waist_shape(p, L, fit):
    """The group entry that moves the waist band, if asked for.

    It relocates the surface landmark L["waist"]; the skeleton and lower-body fit use the neutral ones.
    """
    offset = p.get("waistContourShift", 0.0)
    if not offset:
        return {}
    T = L["T"]
    center = p["waist"] + offset
    L["waist"] = L["at"](center)

    def coordinates(points):
        level = (points[:, 2] - L["crotch"]) / T
        local = points.copy()
        influence = ramp(waist_band(level, center))
        local[:, 2] -= offset * T * influence
        local[:, 0] /= 1 + (fit.get("waistContour", 1.0) - 1) * influence
        return local

    return {"coordinates": coordinates}


def crotch_slot(p, L):
    """The legs part at the crotch whatever the thigh volume: a slot opening towards the knees."""
    T = L["T"]
    return RoundCone(
        (0.0, 0.0, L["crotch"] - 0.012 * T),
        (0.0, 0.0, -0.1 * T),
        0.010 * T,
        p["stance"] * T * 0.8,
        flat=(1.0, 30.0),
        paired=False,
    )


# The local contours over the masses, innermost first, each as (make it on the field so far, whether
# the parameters ask for it). These are part of the field the waist and hip are calibrated on.
LAYERS = (
    (PectoralPlate, lambda p: p.get("pectoralThickness", 0.0) > 0),
    # The breast lies on the pectoral: both are thicknesses, so they add.
    (BreastEnvelope, lambda p: p.get("breastProjection", 0.0) > 0),
    # Outermost, because the wrap fits to what the glutes wrap and carries the fit across.
    (GluteContours, lambda p: p.get("gluteFoldDepth", 0) or any(p.get(key, 0) for key in GLUTE_CONTROLS)),
)
# The profiles aim at what the calibrated field measures, so they go on after it, in this order.
MEASURED_LAYERS = (
    (
        lambda body, L, p: AbdominalProfile(
            body, L, p, front_arc=p.get("abdomenFrontArc"), back_hollow=p.get("lumbarHollow")
        ),
        lambda p: any(key in p for key in ABDOMEN_CONTROLS),
    ),
    (
        lambda body, L, p: ThighProfile(body, L, arc=p.get("outerThighArc")),
        lambda p: any(key in p for key in THIGH_CONTROLS),
    ),
)


def build(p, fit=None):
    """The blended body of these controls, and its landmarks."""
    fit = fit or {}
    L = landmarks(p)
    T = L["T"]
    tone = p["definition"]
    soft = p["softness"]
    neck_join = neck_blend(p)
    trunk = Trunk(
        p["chestWidth"] * T,
        p["chestDepth"] * T,
        p["waistBreadth"] * T * fit.get("waist", 1.0),
        p["waistDepth"] * T,
        p["hipBreadth"] * T,
        p["hipDepth"] * T,
    )
    thigh = thigh_of(p, L)
    seat = hip_seat(T, trunk, thigh, fit)
    L["hipTissueAtMidline"] = seat.at_midline

    arm_bones, arm = arm_masses(p, L)
    leg_bones, leg, shin = leg_masses(p, L, thigh, seat.offset)
    thigh_profile = {"paired_coordinates": thigh_coordinates(p, L, trunk, thigh)}
    waist = waist_shape(p, L, fit)

    groups = [
        {**waist, "prims": core_masses(p, L, trunk, seat.offset), "k": 0.13 * T * soft, "join": 0.0},
        {**waist, "prims": trunk_form_masses(p, L, trunk), "k": 0.05 * T * soft, "join": 0.12 * T * soft},
        # The glute's lower end blends into the back of the thigh, so it goes where thighFrontShift takes the
        # thigh: left behind, it would hold the thigh's inner back as the rest moves.
        {
            **waist,
            **offset_coordinates(thigh_offset(p, L, thigh)),
            "prims": hip_masses(L, trunk, seat),
            "k": 0.10 * T * soft,
            "join": 0.14 * T * soft,
        },
        {**waist, "prims": lower_belly_masses(p, L, trunk), "k": 0.0, "join": 0.035 * T * soft},
        {"prims": neck_masses(p, L), "k": 0.03 * T * soft, "join": neck_join * T * soft},
        {"prims": trapezius_masses(p, L) + scapular_masses(p, L), "k": 0.08 * T * soft, "join": neck_join * T * soft},
        # A narrow blend of its own, so a buried clavicle doesn't raise the chest around it.
        {"prims": clavicle_masses(p, L, trunk), "k": 0.0, "join": (0.012 + 0.08 * tone) * T},
        {
            "name": "armBones",
            "prims": arm_bones,
            "k": 0.012 * T,
            "join": 0.07 * T * soft,
            "clips": [(L["wrist"], L["foreDir"])],
        },
        {
            "name": "armMasses",
            "prims": arm,
            "k": 0.035 * T * soft,
            "join": 0.035 * T * soft,
        },
        {"prims": leg_bones, "k": 0.012 * T, "join": 0.08 * T * soft, **thigh_profile},
        {"prims": leg, "k": 0.05 * T * soft, "join": 0.05 * T * soft, **thigh_profile},
        {"prims": shin, "k": 0.04 * T * soft, "join": 0.04 * T * soft},
    ]
    clips = [
        # Flat cuts for the meshed preview: the head and the feet come from elsewhere.
        (np.array([0.0, 0.0, L["chin"] + 0.04 * L["head"]]), np.array([0.0, 0.0, 1.0])),
        (np.array([0.0, 0.0, L["ankle"]]), np.array([0.0, 0.0, -1.0])),
    ]
    body = Body(groups, clips, carve=[(crotch_slot(p, L), 0.03 * T), hollow(L)])
    for make, wanted in LAYERS:
        if wanted(p):
            body = make(body, L, p)
    return body, L


def width_at(body, z):
    """Width of the trunk at a height, measured from the midline out to the first surface."""
    ts = np.arange(0.0, 500.0, 0.5)
    values = body(np.column_stack((ts, np.zeros_like(ts), np.full_like(ts, z))))
    outside = np.nonzero(values > 0)[0]
    return 2 * float(ts[outside[0]]) if len(outside) else 0.0


@lru_cache(maxsize=32)
def solved_offsets(neutral_items, rounds):
    """Cache scalar waist, hip and local contour offsets, never mutable fields."""
    neutral = dict(neutral_items)
    fit = {"waist": 1.0, "hip": 0.0}
    reference = (
        neutral | {"waistContourShift": 0.0} if neutral.get("waistContourShift", 0) else neutral
    )
    for _ in range(rounds):
        body, L = build(reference, fit)
        T, at = L["T"], L["at"]
        waist = width_at(body, L["waist"])
        hip = max(width_at(body, at(f)) for f in np.linspace(-0.05, 0.25, 13))
        fit["waist"] *= neutral["waistBreadth"] * T / max(waist, 1.0)
        fit["hip"] += (neutral["hipBreadth"] * T - hip) / 2
    fit["waistContour"] = 1.0
    if neutral.get("waistContourShift", 0):
        # Correct only the moved waist band; its neighbours retain the neutral calibration.
        for _ in range(rounds):
            body, L = build(neutral, fit)
            fit["waistContour"] *= (
                neutral["waistBreadth"] * L["T"] / max(width_at(body, L["waist"]), 1.0)
            )
    return fit["waist"], fit["hip"], fit["waistContour"]


def calibrated(p, rounds=8):
    """Solve the internal offsets so the measured waist and hip equal what the parameters ask for.

    Only the offsets are remembered. Every call builds its own field, because wrap_body takes the
    clips off the one it gets.
    """
    # Set the width on the neutral pelvis before moving its posterior and outer contours.
    neutral = {k: v for k, v in p.items() if k not in CONTOURS}
    waist, hip, contour = solved_offsets(tuple(neutral.items()), rounds)
    body, L = build(p, {"waist": waist, "hip": hip, "waistContour": contour})
    for make, wanted in MEASURED_LAYERS:
        if wanted(p):
            body = make(body, L, p)
    return body, L


def skin_reference(p, vertices):
    """Bind local abdomen/thigh shape changes to their reference skin on the same skeleton.

    Diffusing and fitting weights again on the changed skin perturbs untouched groin faces.
    The inverse map recovers the exact skin used for the reference binding; the same reference defines the skeleton and weights.
    """
    if not any(key in p for key in PROFILES):
        return vertices
    body, _ = calibrated(p)
    reference = vertices
    for layer in peeled(body, (ThighProfile, AbdominalProfile))[1]:
        reference = layer.to_reference(reference)
    return reference
