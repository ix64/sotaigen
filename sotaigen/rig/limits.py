"""Each control's range-of-motion limits, from the sourced joint ranges, in the control's own Euler angles.

The ranges are docs/motion/range-of-motion.md's; which motion an axis of which control stands for, and
what a joint's zero is measured from, are AXES and ZERO_DOWN below. The .blend writes these limits
onto its controls (sotaigen.output.blend), and the rest pose and the contact poses stay inside them.
"""

import json
import math
from pathlib import Path

import numpy as np

from sotaigen.paths import DATA
from sotaigen.rig.skinning import euler, names_one_bone

RANGE_OF_MOTION = DATA / "range-of-motion.json"

# mirrors Y and Z, as sotaigen.rig.skinning.sided does. A pair is (motion towards minus, motion towards plus),
# each (joint, motion[, share of it]), "lock" for an axis the joint has no motion about, or None where the data has no
# value. Missing rows leave the axis free. The signs are checked by posing the file (tests/test_blend.py).
AXES = {
    "UpperArm": {
        "x": (("shoulder", "extension"), ("shoulder", "flexion")),
        "y": (("shoulder", "externalRotation"), ("shoulder", "internalRotation")),
        # Down to the arm hanging straight: the data has no adduction past the body.
        "z": ("vertical", ("shoulder", "abduction")),
    },
    "LowerArm": {"x": (("elbow", "hyperextension"), ("elbow", "flexion")), "y": "lock", "z": "lock"},
    "Hand": {
        "x": (("wrist", "extension"), ("wrist", "flexion")),
        "y": (("forearm", "supination"), ("forearm", "pronation")),
        "z": (("wrist", "ulnarDeviation"), ("wrist", "radialDeviation")),
    },
    "UpperLeg": {
        "x": (("hip", "extension"), ("hip", "flexion")),
        "y": (("hip", "externalRotation"), ("hip", "internalRotation")),
        "z": (("hip", "adduction"), ("hip", "abduction")),
    },
    "LowerLeg": {"x": (("knee", "flexion"), ("knee", "hyperextension")), "y": "lock", "z": "lock"},
    "Foot": {
        "x": (("ankle", "plantarflexion"), ("ankle", "dorsiflexion")),
        # Posed in Blender: Y+ turns the sole outward.
        "y": (("ankle", "inversion"), ("ankle", "eversion")),
    },
    "spine": {
        "x": (("lumbar", "extension"), ("lumbar", "flexion")),
        "y": (("lumbar", "rotation"), ("lumbar", "rotation")),
        "z": (("lumbar", "lateralFlexion"), ("lumbar", "lateralFlexion")),
    },
}
for _finger in ("Index", "Middle", "Ring", "Little"):
    AXES[f"{_finger}Proximal"] = {
        "x": (("fingers", "mcpExtension"), ("fingers", "mcpFlexion")),
        "y": "lock",
        "z": (("fingers", "mcpAbduction"), ("fingers", "mcpAbduction")),
    }
    AXES[f"{_finger}Intermediate"] = {
        "x": (("fingers", "pipExtension"), ("fingers", "pipFlexion")),
        "y": "lock",
        "z": "lock",
    }
    AXES[f"{_finger}Distal"] = {
        "x": (("fingers", "dipExtension"), ("fingers", "dipFlexion")),
        "y": "lock",
        "z": "lock",
    }
# The neck and the head share the cervical range evenly.
for _bone in ("neck", "head"):
    AXES[_bone] = {
        "x": (("cervical", "extension"), ("cervical", "flexion")),
        "y": (("cervical", "rotation"), ("cervical", "rotation")),
        "z": (("cervical", "lateralFlexion"), ("cervical", "lateralFlexion")),
        "share": 0.5,
    }
# The jaw opens from rest and swings about its Z, the chin aside; it has no roll. The eye's rest is its
# zero, X+ looks up, Z+ out to the wearer's left on the left eye.
AXES["jaw"] = {
    "x": ("rest", ("jaw", "opening")),
    "y": "lock",
    "z": (("jaw", "lateralExcursion"), ("jaw", "lateralExcursion")),
}
AXES["Eye"] = {
    "x": (("eye", "depression"), ("eye", "elevation")),
    "y": "lock",
    "z": (("eye", "adduction"), ("eye", "abduction")),
}
# The trunk's three controls by where their heads sit, as shares of the torso from the crotch to the
# jugular notch (sotaigen.rig.skeleton.spine_table, sotaigen.form.anatomy.core_masses): `spine` at 0.36, inside the pelvic
# bowl whose top (the iliac crest) is at 0.46, turns everything above it, so it is the lumbar spine;
# `chest` at 0.56, above the rib cage's lower edge (0.44), stands for the lower thoracic segments up to
# `upperChest` at 0.76, about the mid-thorax (the thorax's centre is 0.69), which stands for the upper
# ones up to the neck at 1.03 (C7/T1). The thoracic totals are split between them by the segmental
# ranges measured each side: rotation T1-T7 9.5 against T7-L1 11.9 (Fujimori 2012), lateral bending
# 6.9 against 10.9 (Fujimori 2014). The flexion-extension arc has no neutral to split it by, so X is free.
THORACIC_LOWER = {"rotation": 11.9 / 21.4, "lateralFlexion": 10.9 / 17.8}
for _bone, _upper in (("chest", False), ("upperChest", True)):
    _shares = {m: 1 - v if _upper else v for m, v in THORACIC_LOWER.items()}
    AXES[_bone] = {
        "y": tuple(("thoracic", "rotation", _shares["rotation"]) for _ in range(2)),
        "z": tuple(("thoracic", "lateralFlexion", _shares["lateralFlexion"]) for _ in range(2)),
    }
# What a joint's zero is measured from, for the rest pose's own angle (range-of-motion.md 换到骨骼上):
# the upper arm and thigh from hanging straight down, the forearm and shin from their parent's line.
# The hand, the fingers and the foot keep the rest as their zero: their bones run between joint
# centres, not along the metacarpal, the phalanges or the sole, and measured that way a fingertip
# would rest 15-22 degrees past straight and the foot 27 degrees pointed.
ZERO_DOWN = ("UpperArm", "UpperLeg")
ZERO_PARENT = ("LowerArm", "LowerLeg")


def range_of_motion(path=RANGE_OF_MOTION):
    return json.loads(Path(path).read_text())["joints"]


def limb(name):
    """('left', 'UpperArm') for leftUpperArm, ('', 'spine') for the midline."""
    for side in ("left", "right"):
        if name.startswith(side):
            return side, name[len(side) :]
    return "", name


# The order a control's limit splits its rotation in, where it isn't the control's own XYZ. Split as
# XYZ (Rz Ry Rx), an upper arm raised to the front has Z along itself, so the limit that keeps a
# hanging arm out of the ribs stops its roll instead. Split as YZX (Rx Rz Ry), Y is the roll and Z how
# far the arm is out of the plane X swings it in, whatever X is: hanging it stops at the body, level
# in front it reaches 45 degrees past the midline. A turn about one axis reads the same in both.
LIMIT_ORDER = {"UpperArm": "YZX"}


def limit_order(name):
    return LIMIT_ORDER.get(limb(name)[1], "XYZ")


def wrapped(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def split_yzx(matrix):
    """Radians (x, y, z) with Rx Rz Ry = matrix: of the two solutions the one with the smaller sum of
    magnitudes, as Blender takes it."""
    m = matrix
    c = math.hypot(m[0, 0], m[0, 2])
    z = math.atan2(-m[0, 1], c)
    if c < 1e-9:
        return (math.atan2(-m[1, 2], m[2, 2]), 0.0, z)
    one = (math.atan2(m[2, 1], m[1, 1]), math.atan2(m[0, 2], m[0, 0]), z)
    two = (wrapped(one[0] + math.pi), wrapped(one[1] + math.pi), wrapped(math.pi - z))
    return min(one, two, key=lambda solution: sum(abs(v) for v in solution))


def nearest_xyz(matrix, near):
    """Degrees (x, y, z) with Rz Ry Rx = matrix: of the two solutions the one nearer to `near`, as
    Blender's channels read a control that has Euler angles of its own."""
    m = matrix
    y = math.asin(max(-1.0, min(1.0, -m[2, 0])))
    x, z = math.atan2(m[2, 1], m[2, 2]), math.atan2(m[1, 0], m[0, 0])
    solutions = (
        (x, y, z),
        (wrapped(x + math.pi), wrapped(math.pi - y), wrapped(z + math.pi)),
    )
    def distance(solution):
        return sum(abs(wrapped(v - math.radians(n))) for v, n in zip(solution, near, strict=True))
    return tuple(math.degrees(v) for v in min(solutions, key=distance))


def from_split(bone, parts, near):
    """A control's Euler angles for degrees (x, y, z) as its limit splits its rotation: of the two
    readings of that turn the one nearer to `near`."""
    if limit_order(bone) == "XYZ":
        return tuple(parts)
    x, y, z = parts
    turned = euler((x, 0, 0))[:3, :3] @ euler((0, 0, z))[:3, :3] @ euler((0, y, 0))[:3, :3]
    return nearest_xyz(turned, near)


def held(bone, angles, rows):
    """A control's Euler angles as its limit lets them through: unchanged when it holds nothing."""
    if limit_order(bone) == "XYZ":
        return tuple(
            a if row is None else min(max(a, row[0]), row[1]) for a, row in zip(angles, rows, strict=True)
        )
    parts = [math.degrees(v) for v in split_yzx(euler(angles)[:3, :3])]
    kept = [v if row is None else min(max(v, row[0]), row[1]) for v, row in zip(parts, rows, strict=True)]
    if kept == parts:
        return tuple(angles)
    return from_split(bone, kept, angles)


def rest_angles(bones):
    """Each left control's angle at rest from its joint's anatomical zero, about local X and Z, in
    degrees: the arms rest raised 35 degrees."""
    frames = {}
    for bone in bones:
        y = np.subtract(bone["tail"], bone["head"])
        y = y / np.linalg.norm(y)
        z = bone["align"] - y * (bone["align"] @ y)
        z = z / np.linalg.norm(z)
        frames[bone["name"]] = np.column_stack((np.cross(y, z), y, z))
    parents = {bone["name"]: bone["parent"] for bone in bones}
    out = {}
    for name, frame in frames.items():
        side, part = limb(name)
        if side != "left":
            continue
        if part in ZERO_DOWN:
            zero = np.array([0.0, 0.0, -1.0])
        elif part in ZERO_PARENT:
            zero = frames[parents[name]][:, 1]
        else:
            continue
        d = frame.T @ zero
        out[name] = (math.degrees(math.atan2(-d[2], d[1])), math.degrees(math.atan2(d[0], d[1])))
    return out


def joint_limits(bones, rom=None):
    """{control's bone: [x, y, z]}, each [min, max] in the control's local Euler degrees or None for
    a free axis; only bones with a limit on some axis are there."""
    rom = rom or range_of_motion()
    rest = rest_angles(bones)
    out = {}
    for bone in bones:
        name = bone["name"]
        side, part = limb(name)
        axes = AXES.get(part)
        if axes is None:
            continue
        offsets = rest.get(f"left{part}" if side else name, (0.0, 0.0))
        offset = {"x": offsets[0], "y": 0.0, "z": offsets[1]}
        rows = []
        for axis in "xyz":
            spec = axes.get(axis)
            if spec is None:
                rows.append(None)
                continue
            if spec == "lock":
                rows.append([0.0, 0.0])
                continue
            ends = []
            for i, end in enumerate(spec):
                if end in ("vertical", "rest"):
                    ends.append(0.0)
                    continue
                motion = rom[end[0]]["motions"].get(end[1])
                share = end[2] if len(end) > 2 else axes.get("share", 1.0)
                ends.append(None if motion is None else motion["value"] * share * (1 if i else -1))
            if ends[0] is None and ends[1] is None:
                rows.append(None)
                continue
            low = -180.0 if ends[0] is None else ends[0] - offset[axis]
            high = 180.0 if ends[1] is None else ends[1] - offset[axis]
            if side == "right" and axis in "yz":
                low, high = -high, -low
            rows.append([round(low, 3), round(high, 3)])
        if any(row is not None for row in rows):
            out[name] = rows
    return out


def clipped(pose, limits):
    """A pose as the controls' limits let it through, by the same names: what the file shows with
    the limits on. A limb without a side is split into its two sides when a limit clips it."""
    out = {}
    for name, value in pose.items():
        if name == "rootTranslation":
            out[name] = value
            continue
        sides = (
            [(name, tuple(value))]
            if names_one_bone(name)
            else [
                (f"left{name[0].upper()}{name[1:]}", tuple(value)),
                (f"right{name[0].upper()}{name[1:]}", (value[0], -value[1], -value[2])),
            ]
        )
        for bone, angles in sides:
            rows = limits.get(bone)
            if rows:
                angles = held(bone, angles, rows)
            out[bone] = angles
    return out
