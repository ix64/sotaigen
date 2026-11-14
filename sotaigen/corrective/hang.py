"""Generation-time contact for a hanging arm: the arm against the chest side, the hip and the thigh.

docs/motion/arm-hang.toml asks for a relaxed hang down to 5 degrees from vertical, with a simple soft
contact. The shoulder table ends at its lowest spread input; below it, the arm's skin would pass into
the side. This table starts where the shoulder's ends and, on top of what the shoulder delivers there,
moves the skin only where it would pass through itself. It is a geometric fit, not tissue mechanics.

Only a relaxed hang is solved: a bent elbow, a twisted arm, a turned hand or a flexed hip on that side
moves the arm off the solved place, and the driver fades the correction out there, as the user chose.

This module names, drives and binds the table's targets. The table itself comes from the implicit
field by default (sotaigen.corrective.hang_field), or from an IPC bake (sotaigen.contact.hang).
"""

import math

import numpy as np

from sotaigen.corrective import axilla
from sotaigen.corrective.drive import bound, weights
from sotaigen.form.field import ramp
from sotaigen.mesh import triangulate, vertex_normals
from sotaigen.rig.skinning import pose_skins, swing_euler

SIDES = ("left", "right")
FLIP = np.array([-1.0, 1.0, 1.0])
# Forward keys. Up to 50 degrees the forearm sweeps across the hip, and keys 15 to 25 degrees apart, as
# the shoulder table has, cross again between them; above that only the upper arm meets the chest.
# Avoid a key at 45: the shoulder table can leave too little clearance there to start the contact solve.
FORWARD = (0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 75, 95)
# The hang table starts at the end of the shoulder table's spread range.
EDGE = float(-axilla.SPREAD_KEYS[0])
# The user's target, in degrees of the upper arm from vertical, in front view.
FROM_VERTICAL = 5.0
# Adduction keys are at most this far apart, in degrees; the solver walks a degree at a time.
KEY_STEP = 2.5
# Where the correction has faded out, in degrees. Implementation limits, not physiology.
TWIST_LIMIT = 90
ELBOW_LIMIT = 30
HAND_LIMIT = 30
HIP_LIMIT = 30
# How far from the hanging arm the side may answer, and over what it fades, in shares of the torso.
REACH, FADE = 0.16, 0.06
ARM_GROUPS = ("upperArm", "lowerArm", "hand")


def rest_tilt(bones, side="left"):
    """The upper arm's angle from vertical at rest, in degrees, seen from the front."""
    bone = next(b for b in bones if b["name"] == side + "UpperArm")
    d = np.asarray(bone["tail"], float) - np.asarray(bone["head"], float)
    return math.degrees(math.atan2(abs(d[0]), -d[2]))


def adduction_keys(bones):
    """Adduction inputs, positive degrees below rest, from the shoulder's edge to the target; or ()."""
    bottom = max(rest_tilt(bones, side) for side in SIDES) - FROM_VERTICAL
    if bottom <= EDGE + 0.5:
        return ()
    count = math.ceil((bottom - EDGE) / KEY_STEP)
    return tuple(round(EDGE + (bottom - EDGE) * i / count, 2) for i in range(count + 1))


def sample_names(keys):
    """{side: [(forward, adduction, name)]}, every key but the shoulder's edge, where nothing is stored."""
    return {
        side: [(f, a, f"hang{side[0].upper()}F{f}A{round(a * 100)}") for f in FORWARD for a in keys[1:]]
        for side in SIDES
    }


def arm_pose(side, forward, adduction):
    """A table point as a pose: the twist-free swing forward and below rest, as pose_weights reads it."""
    sign = 1 if side == "left" else -1
    return {side + "UpperArm": swing_euler(float(forward), 0.0, -sign * float(adduction))}


def ease(value):
    return float(ramp(value))


def groups(table, source="field"):
    """The table as correctives groups (sotaigen.corrective.drive), a side each: forward and below the
    shoulder table's edge, faded by the arm's twist, a bent elbow, a turned hand and a bent hip."""
    out = []
    for side in SIDES:
        rows = table["sides"][side]
        out.append({
            "name": f"hang.{side}",
            "source": source,
            "axes": [
                {"bone": f"{side}UpperArm", "read": "swingX", "scale": 1, "keys": list(table["forwardKeys"])},
                # Degrees below rest: the left's side raise turned over, which the right's already is.
                {"bone": f"{side}UpperArm", "read": "swingZ", "scale": -1 if side == "left" else 1,
                 "keys": list(table["adductionKeys"])},
            ],
            "fades": [
                {"bone": f"{side}UpperArm", "read": "twist", "limit": table["twistLimit"]},
                {"bone": f"{side}LowerArm", "read": "largest", "limit": table["elbowLimit"]},
                {"bone": f"{side}Hand", "read": "largest", "limit": table["handLimit"]},
                {"bone": f"{side}UpperLeg", "read": "largest", "limit": table["hipLimit"]},
            ],
            "samples": [list(p) for p in rows["samples"]],
            "targets": list(rows["targets"]),
        })
    return out


def pose_weights(pose, table):
    """Target weights for a pose, from the driver metadata. Zero at and above the shoulder's edge."""
    return weights(pose, groups(table))


def driver_metadata(keys):
    names = sample_names(keys)
    return {
        "forwardKeys": list(FORWARD),
        "adductionKeys": list(keys),
        "sides": {side: {"samples": [[f, a] for f, a, _ in rows], "targets": [n for _, _, n in rows]}
                  for side, rows in names.items()},
        "twistLimit": TWIST_LIMIT,
        "elbowLimit": ELBOW_LIMIT,
        "handLimit": HAND_LIMIT,
        "hipLimit": HIP_LIMIT,
    }


def bind_verts(verts, targets, pose, table):
    return bound(verts, targets, pose_weights(pose, table))


def segment_distance(points, a, b):
    ab = b - a
    t = np.clip((points - a) @ ab / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def normal_targets(
    base, verts, bones, weights, shoulder, deltas, table, normals, *, triangles=None, names=None
):
    """Bind-space normal deltas: the normals with this table, less those the shoulder already gives.

    Only for the targets in `names` when it is given; each target's are the same either way.
    """
    if triangles is None:
        triangles = triangulate(base["faces"], np.asarray(base["verts"]))
    triangles = np.asarray(triangles)
    result = {}
    held = {}
    for side in SIDES:
        rows = table["sides"][side]
        for (forward, adduction), name in zip(rows["samples"], rows["targets"], strict=True):
            if names is not None and name not in names:
                continue
            pose = arm_pose(side, forward, adduction)
            moved = np.linalg.norm(deltas[name], axis=1) > 1e-8
            ids = np.unique(triangles[moved[triangles].any(1)])
            # A normal at `ids` reads only the triangles around them, so only their corners are skinned.
            # Kept in their order, the triangles add up at each of `ids` as they do over the whole skin.
            ring = triangles[np.isin(triangles, ids).any(1)]
            need, local = np.unique(ring, return_inverse=True)
            local = local.reshape(ring.shape)
            at = np.searchsorted(need, ids)
            if (side, forward) not in held:
                held[side, forward] = axilla.bind_verts(verts, shoulder, arm_pose(side, forward, EDGE))
            start = held[side, forward][need]
            skins = pose_skins(bones, pose)
            # Only the bones that move these vertices, as matrix products: most of the cost otherwise.
            near = weights[need]
            used = np.flatnonzero(near.any(0))
            corners = (near[:, used] @ skins[used, :3].reshape(len(used), 12)).reshape(-1, 3, 4)
            blend = corners[at, :, :3]
            bind = []
            for skin in (start, start + deltas[name][need]):
                posed = (corners[:, :, :3] @ skin[:, :, None])[:, :, 0] + corners[:, :, 3]
                geometric = vertex_normals(posed, local)[at]
                local_normal = np.linalg.solve(blend, geometric[:, :, None])[:, :, 0]
                bind.append(
                    local_normal / np.maximum(np.linalg.norm(local_normal, axis=1, keepdims=True), 1e-12)
                )
            delta = np.zeros_like(normals)
            delta[ids] = bind[1] - bind[0]
            result[name] = delta
    return result
