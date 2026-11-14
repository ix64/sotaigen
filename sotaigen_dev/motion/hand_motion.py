"""Authored hand performances, measured in the same wrist frame as hand_contacts."""

import math

from sotaigen.rig.poses import POSES
from sotaigen.rig.skinning import capital

# The handle remains at this arm's final reach until lifted, then returns here before release.
AT_HANDLE = {"upperArm": (16, 0, 10), "lowerArm": (68, 0, 0)}
READY = {"upperArm": (15, 0, 26), "lowerArm": (63, 0, 0)}
LIFTED = {"upperArm": (22, 0, 10), "lowerArm": (85, 0, 0)}


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def blend(a, b, t):
    return {
        key: tuple(v + (w - v) * t for v, w in zip(a.get(key, (0, 0, 0)), b.get(key, (0, 0, 0))))
        for key in a.keys() | b.keys()
    }


def cylinder_pickup(t):
    """Reach, close, lift, put down and release. t is the fraction of the whole performance."""
    # Reverse the approach after putting down. Open the thumb while the hand is still clear.
    u = min(t, 1 - t)
    grip = POSES["cylinderGrip"]
    relaxed = POSES["handRelaxed"]
    fingers = blend(relaxed, {}, ease(u / 0.10))
    # Open the first web space on approach; keep the IP joint nearly straight.
    # Once the thumb clears the handle, oppose at the base, then close MCP and IP.
    for key, opened, start, end in (
        ("thumbMetacarpal", (14, -8, 45), 0.16, 0.34),
        ("thumbProximal", (4, 0, -5), 0.17, 0.355),
        ("thumbDistal", (1, 0, -4), 0.20, 0.365),
    ):
        rest = relaxed.get(key, (0, 0, 0))
        opening = ease(u / 0.16)
        closing = ease((u - start) / (end - start))
        fingers[key] = tuple(
            a + (b - a) * opening + (c - b) * closing
            for a, b, c in zip(rest, opened, grip[key], strict=True)
        )
    arm = blend(READY, AT_HANDLE, ease((u - 0.10) / 0.14))
    # Keep the thumb on its own side of the handle while the long fingers wrap around it.
    for finger, delay in (
        ("middle", 0),
        ("ring", 0.01),
        ("index", 0.02),
        ("little", 0.03),
    ):
        finish = {k: v for k, v in grip.items() if k.startswith(finger)}
        start = {k: fingers.get(k, (0, 0, 0)) for k in finish}
        fingers.update(blend(start, finish, ease((u - 0.24 - delay) / 0.09)))
    if u >= 0.38:
        arm = handle_pose(t)
    return {**arm, **fingers}


def greeting(t):
    opening = ease(min(t, 1 - t) / 0.19)
    fingers = blend(POSES["handRelaxed"], POSES["handWave"], opening)
    # Wrist deviation carries the wave. Give the loose outer fingers a small delayed follow.
    cycle = 4 * math.pi * t
    for finger, amount, lag in (("ring", 3, 0.25), ("little", 6, 0.45)):
        key = f"{finger}Proximal"
        x, y, z = fingers[key]
        fingers[key] = (x + amount * opening * math.sin(cycle - lag), y, z)
    return {
        **fingers,
        "upperArm": (12, 0, 21),
        "lowerArm": (104, 0, 0),
        "hand": (3 * opening, 0, 18 * opening * math.sin(cycle)),
    }


PERFORMANCES = {"handGreeting": greeting, "cylinderPickup": cylinder_pickup}


def hand_sample(name, t, side="left"):
    """Explicit bone names let the other hand remain available for a different action."""
    sign = 1 if side == "left" else -1
    return {
        side + capital(key): (x, sign * y, sign * z)
        for key, (x, y, z) in PERFORMANCES[name](t).items()
    }


def handle_pose(t):
    """Pose of the arm carrying the reference prop, fixed in world space outside the lift."""
    u = min(t, 1 - t)
    return blend(AT_HANDLE, LIFTED, ease((u - 0.38) / 0.10))
