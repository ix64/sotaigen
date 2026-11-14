"""The check poses: rotations in degrees by bone name, or by limb for both sides."""

import numpy as np

from sotaigen.rig.skinning import pose_verts


def finger_curls(rows):
    """Long finger flexion about X; spread at the knuckle about Z."""
    pose = {}
    for finger, (knuckle, middle, tip, spread) in rows.items():
        pose[f"{finger}Proximal"] = (knuckle, 0, spread)
        pose[f"{finger}Intermediate"] = (middle, 0, 0)
        pose[f"{finger}Distal"] = (tip, 0, 0)
    return pose


POSES = {
    "rest": {},
    "armsDown": {"upperArm": (0, 0, -24)},
    "armRoll": {"upperArm": (0, 60, 0)},
    # Forearm pronation reaches 80 degrees from thumb-up (sotaigen/data/range-of-motion.json).
    "wristRoll": {"hand": (0, 80, 0)},
    "armsUp": {"upperArm": (0, 0, 72)},
    "armsHigh": {"upperArm": (0, 0, 110)},
    "reachHigh": {"upperArm": (95, 0, 0)},
    "reachDiagonal": {"upperArm": (75, 20, 35)},
    "reach": {"upperArm": (75, 0, 0), "lowerArm": (5, 0, 0)},
    "elbowBend": {"upperArm": (20, 0, -18), "lowerArm": (120, 0, 0)},
    "crouch": {
        "spine": (12, 0, 0),
        "upperLeg": (85, 0, 0),
        "lowerLeg": (-115, 0, 0),
        "foot": (28, 0, 0),
    },
    "stride": {
        "leftUpperLeg": (26, 0, 0),
        "leftLowerLeg": (-20, 0, 0),
        "rightUpperLeg": (-16, 0, 0),
        "rightLowerLeg": (-12, 0, 0),
    },
    "headTurn": {"neck": (0, 15, 0), "head": (0, 30, 0)},
    "headNod": {"neck": (10, 0, 0), "head": (20, 0, 0)},
    "jawOpen": {"jaw": (20, 0, 0)},
    "jawTurn": {"neck": (0, 15, 0), "head": (0, 30, 0), "jaw": (20, 0, 0)},
    "jawNod": {"neck": (10, 0, 0), "head": (20, 0, 0), "jaw": (20, 0, 0)},
    # The eyes carry no skin, so the head turns too: a pose that moves nothing checks nothing.
    "gaze": {"head": (0, 10, 0), "leftEye": (8, 0, 12), "rightEye": (8, 0, 12)},
    # Within each part's share of the trunk's rotation: lumbar 7, lower and upper thorax 13.9 and 11.1.
    "twist": {"spine": (0, 7, 0), "chest": (0, 13, 0), "upperChest": (0, 10, 0)},
    "handRelaxed": {
        "palmCup": (0, -12, 0),
        "thumbMetacarpal": (8, -5, -8),
        **finger_curls(
            {
                "index": (8, 13, 6, 2),
                "middle": (14, 22, 10, 0),
                "ring": (19, 28, 15, -1),
                "little": (23, 33, 18, -2),
            }
        ),
    },
    "handPoint": {
        "palmCup": (0, -8, 0),
        "thumbMetacarpal": (8, 0, 8),
        "thumbProximal": (0, 0, -8),
        **finger_curls(
            {
                "index": (0, 4, 2, 4),
                "middle": (32, 52, 27, 3),
                "ring": (53, 65, 44, -7),
                "little": (55, 69, 46, -5),
            }
        ),
    },
    "handWave": {
        "palmCup": (0, -7, 0),
        "thumbMetacarpal": (3, 0, 14),
        **finger_curls(
            {
                "index": (2, 5, 2, 8),
                "middle": (0, 4, 2, 2),
                "ring": (5, 8, 4, -4),
                "little": (10, 12, 6, -12),
            }
        ),
    },
    "handLightTouch": {
        "palmCup": (0, -10, 0),
        "thumbMetacarpal": (10, -5, -4),
        **finger_curls(
            {
                "index": (14, 9, 2, 5),
                "middle": (14, 28, 13, 1),
                "ring": (18, 36, 19, -2),
                "little": (22, 40, 23, -5),
            }
        ),
    },
    "thinCylinderGrip": {
        "thumbMetacarpal": (29.8, -13.4, 45.6),
        "thumbProximal": (14.9, 0, -46.0),
        "thumbDistal": (5.4, 0, -11.1),
        **finger_curls(
            {
                "index": (38.6, 27.3, 51.3, 4),
                "middle": (51, 25.3, 53.8, -3),
                "ring": (49.1, 28.6, 50.3, -1),
                "little": (34.6, 27.9, 41.7, -2),
            }
        ),
    },
    "sphereGrip": {
        "palmCup": (0, -17, 0),
        "thumbMetacarpal": (44.6, 3.4, 44.6),
        "thumbProximal": (22.9, 0, -13.7),
        "thumbDistal": (9.5, 0, -10.3),
        **finger_curls(
            {
                "index": (22.9, 17, 48.6, 4),
                "middle": (22.4, 12.7, 50.4, 5),
                "ring": (15.2, 15.1, 46.5, -11),
                "little": (11.4, 16.7, 39.1, -15),
            }
        ),
    },
    "littleOpposition": {
        "palmCup": (0, -32, 0),
        "thumbMetacarpal": (75.3, 17.5, 67.2),
        "thumbProximal": (34.2, 0, -23.6),
        "thumbDistal": (14.1, 0, 16.4),
        **finger_curls(
            {
                "index": (3, 13, 6, 9),
                "middle": (4, 22, 10, 5),
                "ring": (4, 14, 15, -16),
                # The tip pressed back, as far as the joint goes (7 degrees past straight).
                "little": (52, 42.9, -7, 5.3),
            }
        ),
    },
    "fist": {
        f"{finger}{joint}": (angle, 0, 0)
        for finger in ("index", "middle", "ring", "little")
        for joint, angle in (("Proximal", 70), ("Intermediate", 90), ("Distal", 60))
    },
    # Fit against opposing patches of the two distal pads on the wrapped skin. The first
    # metacarpal rolls about local Y; the thumb hinges bend across the palm about Z.
    "padPinch": {
        "thumbMetacarpal": (44.6, 23.4, 35),
        "thumbProximal": (23.7, 0, -31.7),
        "thumbDistal": (12.3, 0, 15),
        **finger_curls(
            {
                "index": (32.5, 63.7, 7.8, 0),
                "middle": (18, 28, 16, 0),
                "ring": (18, 28, 16, 0),
                "little": (18, 28, 16, 0),
            }
        ),
    },
    # A transverse handle of radius 0.258 times wrist-to-middle-knuckle length. The
    # thumb approaches from the other side; the four fingers do not share a curl angle.
    "cylinderGrip": {
        "thumbMetacarpal": (21.3, -19.4, 19.4),
        "thumbProximal": (24.3, 11.3, -1.5),
        "thumbDistal": (-2.6, 5.9, -30.4),
        **finger_curls(
            {
                "index": (31.6, 21.9, 49.8, 4),
                "middle": (44.7, 20.4, 52.5, -3),
                "ring": (42.5, 23.8, 49.1, -1),
                "little": (26.7, 23.2, 40.8, -2),
            }
        ),
    },
}


def check_poses(base, verts, bones, weights):
    """Place the symmetric crouch over its original footprint on the actual ground plane."""
    # Generic meshes accepted by the GLB writer need not be humanoids with labelled feet.
    if "regions" not in base or "foot" not in base["regions"]:
        return POSES
    feet = np.asarray(base["regions"]) == "foot"
    standing = np.asarray(verts)[feet]
    bent = pose_verts(bones, np.asarray(weights)[feet], standing, POSES["crouch"])
    # Restore fore/aft position and land the lowest sole vertex. Symmetry preserves stance width.
    shift = [0.0, float(np.mean(standing[:, 1] - bent[:, 1])), -float(bent[:, 2].min())]
    return {**POSES, "crouch": {**POSES["crouch"], "rootTranslation": shift}}
