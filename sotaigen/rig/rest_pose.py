"""Which pose a body's .blend takes as its new rest pose, when an export asks for one.

Modelling, the correctives, the contact tables, the clean ranges and the constraints all know only the
fixed A pose (docs/DECISIONS.md, 静止姿势). An export can still be asked for arms at another abduction,
for the ankles a given distance apart, for straight arms and for palms facing down: this finds the
control angles that put the body there, in the language of the check poses, measured on the deform
bones the runtime poses (sotaigen.rig.skinning.pose_skins). blender/rebase.py then makes that pose the file's
rest pose. Nothing here changes the rig; no build stage reaches it.

A control angle past the joint limits (sotaigen.rig.limits.joint_limits) is refused unless the caller allows it;
past the accepted clean range (docs/systems/rig.md, 关节的已知极限) it only warns.
"""

import argparse
import math

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from sotaigen.diagnostics import Diagnostic
from sotaigen.errors import SotaigenError
from sotaigen.rig.limits import limit_order, split_yzx
from sotaigen.rig.skinning import capital, euler, names_one_bone, pose_skins, rest_matrix

# A T pose: arms 90 degrees off vertical, 55 more than the rest pose's 35, straight, palms down. The
# ankles stay where they are. Only one set of values among those a caller can ask for.
T_POSE = {"armAbduction": 90.0, "palmsDown": True, "straightArms": True}
# Accepted clean ranges, in the controls' Euler degrees on the wearer's left (docs/systems/rig.md, 关节的
# 已知极限): {limb: {axis: (low, high)}}. A rest pose outside them warns; the correctives and the skin
# past there aren't what the user accepted. A degree of slack: the side raise's 55 is "about horizontal".
CLEAN = {
    "upperArm": {"x": (-60.0, 95.0), "y": (-90.0, 90.0), "z": (-30.0, 55.0)},
    "lowerArm": {"x": (-7.7, 120.0)},
    "hand": {"x": (-60.0, 45.0), "y": (-120.0, 120.0), "z": (-45.0, 45.0)},
    "upperLeg": {"z": (0.0, 30.0)},
    "foot": {"x": (-45.0, 45.0)},
}
SLACK = 1.0
# Newton's steps for the arm's two angles, and how close it must come, in degrees of the bone's direction.
STEPS, CLOSE = 20, 1e-6
LEFT = "left"


class PastLimits(SotaigenError, ValueError):
    """A rest pose needs a control past its joint limit; `reasons` holds each one's diagnostic."""

    def __init__(self, reasons):
        super().__init__(
            "the rest pose is past the joint limits: " + "; ".join(reasons) + " (allow it with --past-limits)"
        )
        self.reasons = list(reasons)


def world_frames(bones, pose):
    """{bone: its 4 x 4 frame under the pose}, as sotaigen.rig.skinning.pose_skins places it."""
    skins = pose_skins(bones, pose)
    return {bone["name"]: skins[i] @ rest_matrix(bone) for i, bone in enumerate(bones)}


def direction(frames, name):
    return frames[name][:3, 1]


def angle_between(a, b):
    return math.degrees(math.atan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))


def euler_of(matrix):
    """The XYZ Euler angles (degrees) of a rotation, as sotaigen.rig.skinning.euler composes them, |y| <= 90."""
    m = np.asarray(matrix)
    y = math.asin(max(-1.0, min(1.0, -m[2, 0])))
    x = math.atan2(m[2, 1], m[2, 2])
    z = math.atan2(m[1, 0], m[0, 0])
    return tuple(math.degrees(v) for v in (x, y, z))


def local_turning(frames, name, target):
    """The local rotation that takes bone `name`, as it stands in `frames` with no turn of its own, to
    the world rotation `target` (3 x 3)."""
    return frames[name][:3, :3].T @ target


def minimal_rotation(a, b):
    """The smallest rotation taking unit vector a to unit vector b."""
    axis = np.cross(a, b)
    s, c = np.linalg.norm(axis), float(np.dot(a, b))
    if s < 1e-12:
        return np.eye(3)
    k = axis / s
    skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * skew + (1 - c) * skew @ skew


def arm_angles(bones, abduction):
    """The upper arm control's (x, 0, z), so that the deform upper arm, the shoulder's share taken, points
    `abduction` degrees off straight down, in the body's frontal plane, to the wearer's left."""
    a = math.radians(abduction)
    # Two directions across the wanted one, (sin a, 0, -cos a): the frontal plane's, and to the front.
    across = (np.array([math.cos(a), 0.0, math.sin(a)]), np.array([0.0, 1.0, 0.0]))
    name = f"{LEFT}UpperArm"
    now = direction(world_frames(bones, {}), name)
    # The first guess: what the rest pose is short of the angle, in the plane.
    angles = np.array([0.0, abduction - math.degrees(math.atan2(now[0], -now[2]))])

    def residual(v):
        d = direction(world_frames(bones, {"upperArm": (v[0], 0.0, v[1])}), name)
        return np.array([d @ across[0], d @ across[1]])

    for _ in range(STEPS):
        r = residual(angles)
        if math.degrees(np.linalg.norm(r)) < CLOSE:
            break
        # One-sided steps away from zero, where the shoulder's share has a kink.
        steps = [(e * (1e-2 if angles @ e >= 0 else -1e-2)) for e in np.eye(2)]
        jacobian = np.column_stack([(residual(angles + step) - r) / step.sum() for step in steps])
        angles = angles - np.linalg.solve(jacobian, r)
    else:
        raise ValueError(f"no upper arm angle reaches {abduction} degrees: {math.degrees(np.linalg.norm(r)):.4f} off")
    return (float(angles[0]), 0.0, float(angles[1]))


def elbow_angle(bones, arm):
    """The elbow's X that lines the forearm up with the upper arm; a hinge can't take out what's left."""

    def apart(x):
        frames = world_frames(bones, {"upperArm": arm, "lowerArm": (x, 0.0, 0.0)})
        return angle_between(direction(frames, f"{LEFT}LowerArm"), direction(frames, f"{LEFT}UpperArm"))

    return float(minimize_scalar(apart, bounds=(-30.0, 30.0), method="bounded", options={"xatol": 1e-5}).x)


def palm_normal(frames):
    """The left palm's outward normal: across the knuckles, from the wrist. A positive turn of the fingers
    about the hand's X closes them on this side."""
    wrist = frames[f"{LEFT}Hand"][:3, 3]
    index, little = (frames[f"{LEFT}{f}Proximal"][:3, 3] for f in ("Index", "Little"))
    middle = frames[f"{LEFT}MiddleProximal"][:3, 3]
    n = np.cross(little - index, middle - wrist)
    return n / np.linalg.norm(n)


def hand_angles(bones, pose, straight, palm_down):
    """The hand control's Euler angles: its axis along the upper arm when `straight` (the elbow's hinge
    can leave the forearm a little off that line, the hand needn't be), the palm then rolled about that
    axis to face the floor as nearly as it can when `palm_down`."""
    frames = world_frames(bones, pose)
    hand = direction(frames, f"{LEFT}Hand")
    axis = direction(frames, f"{LEFT}UpperArm") if straight else hand
    turn = minimal_rotation(hand, axis)
    if palm_down:
        normal = turn @ palm_normal(frames)
        down = np.array([0.0, 0.0, -1.0])
        have, want = normal - axis * (normal @ axis), down - axis * (down @ axis)
        if np.linalg.norm(want) < 1e-6:
            raise ValueError("the arm points straight down or up, so no roll of the hand faces the palm down")
        have, want = have / np.linalg.norm(have), want / np.linalg.norm(want)
        turn = minimal_rotation(have, want) @ turn
    return euler_of(local_turning(frames, f"{LEFT}Hand", turn @ frames[f"{LEFT}Hand"][:3, :3]))


def ankle_distance(frames):
    return float(np.linalg.norm(frames[f"{LEFT}Foot"][:3, 3] - frames["rightFoot"][:3, 3]))


def thigh_angle(bones, spacing):
    """The thigh control's Z (out, on both sides) that sets the ankle centres `spacing` mm apart."""

    def miss(z):
        return ankle_distance(world_frames(bones, {"upperLeg": (0.0, 0.0, z)})) - spacing

    # The distance only grows outward from the angle where the two ankles meet at the midline.
    low = 0.0
    while world_frames(bones, {"upperLeg": (0.0, 0.0, low)})[f"{LEFT}Foot"][0, 3] > 0 and low > -60.0:
        low -= 1.0
    high = 89.0
    if miss(low) * miss(high) > 0:
        raise ValueError(f"no thigh angle sets the ankles {spacing} mm apart")
    return float(brentq(miss, low, high, xtol=1e-6))


def foot_flat(bones, pose):
    """The foot control's Euler angles that keep the sole as it stood, and how far that lifts the ankle:
    the thigh's turn takes the shin with it."""
    before, frames = world_frames(bones, {}), world_frames(bones, pose)
    name = f"{LEFT}Foot"
    angles = euler_of(local_turning(frames, name, before[name][:3, :3]))
    return angles, float(frames[name][2, 3] - before[name][2, 3])


def checks(pose, limits):
    """(what's past the joint limits, what's past the clean range), each a list of diagnostics. A limit reads
    its angles in the order it splits the rotation (sotaigen.rig.limits.limit_order); the clean ranges are the check
    poses' Euler angles."""
    past, unclean = [], []
    for limb, angles in pose.items():
        if limb == "rootTranslation":
            continue
        bone = LEFT + capital(limb)
        rows = limits.get(bone, [None, None, None])
        if limit_order(bone) == "XYZ":
            parts = angles
        else:
            parts = [math.degrees(v) for v in split_yzx(euler(angles)[:3, :3])]
        for axis, angle, part, row in zip("xyz", angles, parts, rows, strict=True):
            if row is not None and not row[0] <= part <= row[1] and abs(part) > 1e-9:
                past.append(
                    Diagnostic("rest.past_limit", "warning", {"limb": limb, "axis": axis.upper(), "degrees": round(part, 1), "low": row[0], "high": row[1]})
                )
            clean = CLEAN.get(limb, {}).get(axis)
            if clean is not None and not clean[0] - SLACK <= angle <= clean[1] + SLACK and abs(angle) > 1e-9:
                unclean.append(
                    Diagnostic("rest.unclean", "warning", {"limb": limb, "axis": axis.upper(), "degrees": round(angle, 1), "low": clean[0], "high": clean[1]})
                )
    return past, unclean


def solve(bones, limits, *, arm_abduction=None, ankle_spacing=None, palms_down=False, straight_arms=False, past_limits=False):
    """What body.blend hands to blender/rebase.py: {"request", "pose", "measured", "pastLimits", "unclean"}.

    `pose` names controls as the check poses do, a limb for both sides and `rootTranslation` to keep the
    soles where they stood, in the file's units of degrees and mm. A pose past `limits`
    (sotaigen.rig.limits.joint_limits) raises PastLimits unless `past_limits` allows it.
    """
    if arm_abduction is None and ankle_spacing is None and not palms_down and not straight_arms:
        raise ValueError("a rest pose needs at least one of the arm's abduction, the ankle spacing, straight arms or palms down")
    pose = {}
    if arm_abduction is not None:
        pose["upperArm"] = arm_angles(bones, arm_abduction)
    if straight_arms:
        pose["lowerArm"] = (elbow_angle(bones, pose.get("upperArm", (0.0, 0.0, 0.0))), 0.0, 0.0)
    if palms_down or straight_arms:
        pose["hand"] = hand_angles(bones, pose, straight_arms, palms_down)
    if ankle_spacing is not None:
        pose["upperLeg"] = (0.0, 0.0, thigh_angle(bones, ankle_spacing))
        pose["foot"], lift = foot_flat(bones, pose)
        pose["rootTranslation"] = (0.0, 0.0, -lift)
    past, unclean = checks(pose, limits)
    if past and not past_limits:
        raise PastLimits(past)
    frames = world_frames(bones, pose)
    measured = {
        "armAbductionLeft": math.degrees(math.atan2(direction(frames, f"{LEFT}UpperArm")[0], -direction(frames, f"{LEFT}UpperArm")[2])),
        "armAbductionRight": math.degrees(math.atan2(-direction(frames, "rightUpperArm")[0], -direction(frames, "rightUpperArm")[2])),
        "forearmOffUpperArm": angle_between(direction(frames, f"{LEFT}LowerArm"), direction(frames, f"{LEFT}UpperArm")),
        "handOffUpperArm": angle_between(direction(frames, f"{LEFT}Hand"), direction(frames, f"{LEFT}UpperArm")),
        "palmOffDown": angle_between(palm_normal(frames), (0.0, 0.0, -1.0)),
        "ankleSpacing": ankle_distance(frames),
        "ankleHeight": float(frames[f"{LEFT}Foot"][2, 3]),
    }
    request = {
        "armAbduction": arm_abduction,
        "ankleSpacing": ankle_spacing,
        "palmsDown": bool(palms_down),
        "straightArms": bool(straight_arms),
    }
    return {
        "request": request,
        "pose": {k: [float(v) for v in value] for k, value in pose.items()},
        "measured": measured,
        "pastLimits": [d.json() for d in past],
        "unclean": [d.json() for d in unclean],
    }


def by_control(pose):
    """A pose's angles per control, as sotaigen.rig.skinning.sided names them: a limb poses both sides, the
    right mirrored. Without the coupling that sided adds."""
    out = {}
    for name, (x, y, z) in pose.items():
        if name == "rootTranslation":
            continue
        if names_one_bone(name):
            out[name] = (x, y, z)
        else:
            out[f"left{capital(name)}"] = (x, y, z)
            out[f"right{capital(name)}"] = (x, -y, -z)
    return out


def in_new_rest(rest_pose, pose):
    """A pose named by the A pose's angles, the check poses' language, as the new rest's controls name it:
    each control's turn after the rest pose it holds. The rest pose's translation is on the root's
    children in the file, so moving the root by the difference comes to the same thing for the skin."""
    held = by_control(rest_pose)
    wanted = by_control(pose)
    out = {}
    for bone in {*held, *wanted}:
        turn = euler(held.get(bone, (0.0, 0.0, 0.0))).T @ euler(wanted.get(bone, (0.0, 0.0, 0.0)))
        out[bone] = euler_of(turn)
    shift = np.subtract(pose.get("rootTranslation", (0.0, 0.0, 0.0)), rest_pose.get("rootTranslation", (0.0, 0.0, 0.0)))
    if np.any(shift):
        out["rootTranslation"] = tuple(float(v) for v in shift)
    return out


def asset_reference(rest_pose, pose):
    """What a pose asset (A pose's angles) shows played on the new rest: the controls it keys at those angles,
    every other control and the translation still holding the rest pose. For sotaigen.rig.skinning.pose_verts."""
    out = {**by_control(rest_pose), **by_control(pose)}
    if "rootTranslation" in rest_pose:
        out["rootTranslation"] = tuple(rest_pose["rootTranslation"])
    return out


def add_arguments(parser):
    """The command line's way of asking for a rest pose, on `uv run sotaigen-dev blend` and `sotaigen blend`."""
    group = parser.add_argument_group("another rest pose, instead of the A pose (docs/systems/blend.md)")
    group.add_argument("--t-pose", action="store_true", help="arms 90 degrees off vertical and straight, palms down")
    group.add_argument("--arm-abduction", type=float, metavar="DEG", help="upper arms this far off straight down")
    group.add_argument("--ankle-spacing", type=float, metavar="MM", help="this far between the ankle centres")
    group.add_argument("--palms-down", action=argparse.BooleanOptionalAction, default=None, help="roll the hands to face the palms down")
    group.add_argument("--straight-arms", action=argparse.BooleanOptionalAction, default=None, help="elbows straight, hands in line with the arms")
    group.add_argument("--past-limits", action="store_true", help="go past the joint limits if the pose needs to; they stretch to it")


def request_of(args):
    """solve's arguments from add_arguments's, or None when no rest pose was asked for."""
    preset = T_POSE if args.t_pose else {}
    request = {
        "arm_abduction": preset.get("armAbduction") if args.arm_abduction is None else args.arm_abduction,
        "ankle_spacing": args.ankle_spacing,
        "palms_down": preset.get("palmsDown", False) if args.palms_down is None else args.palms_down,
        "straight_arms": preset.get("straightArms", False) if args.straight_arms is None else args.straight_arms,
    }
    if request["arm_abduction"] is None and request["ankle_spacing"] is None and not request["palms_down"] and not request["straight_arms"]:
        return None
    return {**request, "past_limits": args.past_limits}
