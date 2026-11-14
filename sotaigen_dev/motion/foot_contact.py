"""Footplant detection in source leg lengths and contact correction on a target's skinned soles."""

import numpy as np

from sotaigen.rig.skinning import pose_skins, rest_matrix

SIDES = ("left", "right")


def contact_tracks(samples, fps):
    """Intervals at rest on a level floor, using ankle and ball as heel/forefoot proxies.

    Speed is measured across each pair of samples, so a stationary interval covers both endpoints.
    The low height of each landmark is estimated separately, relative to the capture floor.
    """
    points = np.asarray(samples)
    low = points[:, :, 2] <= np.quantile(points[:, :, 2], 0.1, axis=0) + 0.065
    slow = np.linalg.norm(np.diff(points, axis=0), axis=2) * fps <= 0.42
    locked = np.zeros_like(low)
    edges = slow & low[1:] & low[:-1]
    locked[:-1] |= edges
    locked[1:] |= edges
    return locked.astype(int).tolist()


def sole_samples(rig, skins, side):
    """Skinned plantar surface, with persistent mesh markers below the heel and ball."""
    sign = 1 if side == "left" else -1
    foot = (np.asarray(rig.base["regions"]) == "foot") & (rig.verts[:, 0] * sign > 0)
    rest = rig.verts[foot]
    weights = rig.weights[foot]
    bones = weights.any(axis=0)
    weighted = np.column_stack((rest, np.ones(len(rest))))[:, None, :] * weights[:, bones, None]
    moved = np.einsum("fbij,vbj->fvi", skins[:, bones, :3, :], weighted, optimize=True)
    index = {b["name"]: b for b in rig.bones}
    markers = []
    for name in ("Foot", "Toes"):
        xy = index[side + name]["head"][:2]
        # Closest sole point to the ankle / metatarsal joint projected to z=0.
        markers.append(np.argmin(np.linalg.norm(rest - np.r_[xy, 0], axis=1)))
    return moved[:, markers], moved[:, :, 2].min(axis=1)


def contact_errors(rig, poses, contacts):
    """Compare planted skin height and horizontal travel of continuing heel/ball markers."""
    skins = np.array([pose_skins(rig.bones, p) for p in poses])
    heights, steps = [], []
    for side in SIDES:
        markers, floor = sole_samples(rig, skins, side)
        active = np.asarray(contacts[side], bool)
        heights.extend(abs(floor[active.any(axis=1)]))
        distance = np.linalg.norm(np.diff(markers[:, :, :2], axis=0), axis=2)
        steps.extend(distance[active[:-1] & active[1:]])
    return {
        "plantedSideFrames": len(heights),
        "markerSteps": len(steps),
        "worstFloorErrorMM": float(np.max(heights)) if heights else None,
        "meanMarkerStepMM": float(np.mean(steps)) if steps else None,
        "worstMarkerStepMM": float(np.max(steps)) if steps else None,
    }


def anchor_offsets(markers, active):
    """Cancel the displacement of each continuing plant, including heel-to-ball transfers.

    When both are down, their mean displacement is the least-squares translation for a rigid foot.
    One offset per continuous support interval preserves its overall location on the captured path.
    """
    offsets = np.zeros((len(markers), 3))
    shared = active[1:] & active[:-1]
    steps = -np.diff(markers[:, :, :2], axis=0)
    steps = (steps * shared[:, :, None]).sum(axis=1) / np.maximum(shared.sum(axis=1)[:, None], 1)
    edges = shared.any(axis=1)
    changes = np.diff(np.r_[False, edges, False].astype(int))
    for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True):
        span = offsets[start : end + 1, :2]
        span[1:] = np.cumsum(steps[start:end], axis=0)
        span -= span.mean(axis=0)
    return offsets


def aligning(a, b):
    """The smallest rotation that takes direction a to direction b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = a @ b
    if c < -1 + 1e-10:
        axis = np.cross(a, np.eye(3)[np.argmin(abs(a))])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + skew + skew @ skew / (1 + c)


def euler_angles(matrix):
    """Degrees (x, y, z) such that sotaigen.rig.skeleton.euler gives this rotation back: Rz Ry Rx."""
    y = np.arcsin(np.clip(-matrix[2, 0], -1, 1))
    x = np.arctan2(matrix[2, 1], matrix[2, 2])
    z = np.arctan2(matrix[1, 0], matrix[0, 0])
    return tuple(round(float(v), 3) for v in np.degrees((x, y, z)))


def bend_leg(joints, target, forward):
    """Two-link IK preserving segment lengths and the captured knee's bend plane."""
    hip, knee, ankle = joints
    length1 = np.linalg.norm(knee - hip)
    length2 = np.linalg.norm(ankle - knee)
    axis = target - hip
    size = np.linalg.norm(axis)
    if size < 1e-8:
        axis, size = ankle - hip, np.linalg.norm(ankle - hip)
    axis /= max(size, 1e-8)
    distance = np.clip(size, abs(length1 - length2) + 1e-6, length1 + length2 - 1e-6)
    # Use the original bend plane throughout, including when the target crosses the old knee.
    original = ankle - hip
    original /= max(np.linalg.norm(original), 1e-8)
    bend = knee - hip - original * ((knee - hip) @ original)
    if np.linalg.norm(bend) < 1e-5:
        bend = forward
    pole = bend - axis * (bend @ axis)
    if np.linalg.norm(pole) < 1e-8:
        basis = np.eye(3)[np.argmin(abs(axis))]
        pole = basis - axis * (basis @ axis)
    along = (length1**2 - length2**2 + distance**2) / (2 * distance)
    height = np.sqrt(max(0, length1**2 - along**2))
    return hip + along * axis + height * pole / np.linalg.norm(pole), hip + distance * axis


def ease_outside(values, active, width):
    """Fade a correction over neighbouring swing samples, retaining every constrained sample."""
    result = values.copy()
    for i in np.flatnonzero(~active):
        near = np.flatnonzero(active[max(0, i - width) : i + width + 1]) + max(0, i - width)
        if len(near):
            distance = abs(near - i)
            nearest = near[distance == distance.min()]
            t = 1 - distance.min() / (width + 1)
            result[i] = t * t * (3 - 2 * t) * values[nearest].mean(axis=0)
    return result


def nearest_rotations(matrices):
    """Project averages of orientation matrices back onto proper rotations."""
    u, _, vh = np.linalg.svd(matrices)
    u[..., :, -1] *= np.linalg.det(u @ vh)[..., None]
    return u @ vh


def filter_foot_samples(rotations, fps):
    """Reject short reversals in the distal capture without delaying heel strike or toe-off."""
    radius = round(fps / 30)
    if radius < 1:
        return rotations
    n = len(rotations)
    window = np.clip(np.arange(n)[:, None] + np.arange(-radius, radius + 1), 0, n - 1)
    samples = rotations[window]
    # The rotation medoid is an observed orientation. A brief excursion cannot pull
    # it as far as an average; in a steady turn the middle sample is already the medoid.
    distances = np.linalg.norm(samples[:, :, None] - samples[:, None, :], axis=(-2, -1))
    best = distances.sum(axis=2).argmin(axis=1)
    clean = samples[np.arange(n), best]
    previous = np.maximum(np.arange(n) - 1, 0)
    following = np.minimum(np.arange(n) + 1, n - 1)
    return nearest_rotations((clean[previous] + 2 * clean + clean[following]) / 4)


def level_headings(foot_rotations, foot_bone):
    """Yaw of the captured ankle-to-ball direction relative to the model's footprint."""
    axis = foot_bone["tail"] - foot_bone["head"]
    direction = foot_rotations @ axis
    yaw = np.arctan2(direction[:, 1], direction[:, 0]) - np.arctan2(axis[1], axis[0])
    c, s = np.cos(yaw), np.sin(yaw)
    level = np.tile(np.eye(3), (len(yaw), 1, 1))
    level[:, 0, 0] = level[:, 1, 1] = c
    level[:, 0, 1], level[:, 1, 0] = -s, s
    return level


def sole_rotations(rotations, level, dual, width):
    """Transfer swing relative to a level sole pointing along the recorded footprint."""
    # Different ankle heights and oblique bone axes give the capture and mesh
    # different rest frames. Estimate one fixed alignment from observed flat soles.
    offsets = rotations[dual].transpose(0, 2, 1) @ level[dual]
    calibrated = rotations @ nearest_rotations(np.median(offsets, axis=0))
    correction = np.zeros_like(calibrated)
    changes = np.diff(np.r_[False, dual, False].astype(int))
    for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True):
        planted = nearest_rotations(level[start:end].mean(axis=0))
        correction[start:end] = planted @ calibrated[start:end].transpose(0, 2, 1) - np.eye(3)
    # Blend the correction, preserving the motion itself during heel strike and toe-off.
    blend = np.eye(3) + ease_outside(correction, dual, width)
    return nearest_rotations(blend) @ calibrated


def plant_rotation(rig, poses, skins, indexes, planted, width, fps):
    """Calibrate the hindfoot and toes separately; both soles are level at foot-flat."""
    for side in SIDES:
        dual = planted[side].all(axis=1)
        if not dual.any():
            continue
        parent = skins[:, indexes[side + "LowerLeg"], :3, :3]
        foot = indexes[side + "Foot"]
        level = level_headings(skins[:, foot, :3, :3], rig.bones[foot])
        for part in ("Foot", "Toes"):
            name = side + part
            idx = indexes[name]
            axes = rest_matrix(rig.bones[idx])[:3, :3]
            captured = filter_foot_samples(skins[:, idx, :3, :3], fps)
            world = sole_rotations(captured, level, dual, width)
            local = axes.T @ parent.transpose(0, 2, 1) @ world @ axes
            for pose, rotation in zip(poses, local, strict=True):
                pose[name] = euler_angles(rotation)
            parent = world


def ground_motion(rig, poses, contacts, fps):
    """Apply level-floor contacts to a whole clip. Swing and airborne frames keep their travel."""
    skins = np.array([pose_skins(rig.bones, pose) for pose in poses])
    indexes = {b["name"]: i for i, b in enumerate(rig.bones)}
    planted = {s: np.asarray(contacts[s], dtype=bool) for s in SIDES}
    width = max(1, round(0.10 * fps))
    plant_rotation(rig, poses, skins, indexes, planted, width, fps)
    skins = np.array([pose_skins(rig.bones, pose) for pose in poses])
    surface = {side: sole_samples(rig, skins, side) for side in SIDES}
    # One constant places the measured vertical trajectory on this target's floor.
    samples = np.concatenate([surface[s][1][planted[s].any(axis=1)] for s in SIDES])
    if not len(samples):
        return
    level = np.median(samples)
    downward = np.zeros(len(poses))
    goals = {}
    for side in SIDES:
        markers, lowest = surface[side]
        on = planted[side].any(axis=1)
        offsets = anchor_offsets(markers, planted[side])
        offsets[:, 2] = np.where(on, level - lowest, 0)
        offsets = ease_outside(offsets, on, width)
        offsets[:, 2] = np.maximum(offsets[:, 2], level - lowest)
        ids = [indexes[side + part] for part in ("UpperLeg", "LowerLeg", "Foot")]
        rests = [rest_matrix(rig.bones[j]) for j in ids]
        world = skins[:, ids] @ np.array(rests)
        joints = world[:, :, :3, 3]
        target = joints[:, 2] + offsets
        length = sum(np.linalg.norm(joints[0, j + 1] - joints[0, j]) for j in (0, 1))
        reach = target - joints[:, 0]
        # Lower the pelvis only where a support goal would stretch a leg.
        max_height = np.sqrt(np.maximum(0, length**2 - (reach[:, :2] ** 2).sum(axis=1)))
        downward = np.maximum(downward, np.where(on, -reach[:, 2] - max_height, 0))
        goals[side] = (
            ids,
            rests,
            joints,
            target,
            on,
            np.maximum(lowest + offsets[:, 2] - level, 0),
        )
    for frame, pose in enumerate(poses):
        root_drop = np.array([0.0, 0.0, downward[frame]])
        pose["rootTranslation"] = (pose["rootTranslation"] - root_drop - [0, 0, level]).tolist()
        for ids, rests, joints, targets, active, clearance in goals.values():
            original = joints[frame]
            target = targets[frame].copy()
            # With an airborne foot, only compensate the drop if it would cross the floor.
            target[2] += (
                downward[frame] if active[frame] else max(0, downward[frame] - clearance[frame])
            )
            if np.linalg.norm(target - original[2]) < 1e-7:
                continue
            parent = skins[frame, indexes["hips"], :3, :3]
            knee, ankle = bend_leg(original, target, parent @ [0, -1, 0])
            solved = (original[0], knee, ankle)
            for j, (index, rest) in enumerate(zip(ids, rests, strict=True)):
                rotation = skins[frame, index, :3, :3]
                if j < 2:
                    rotation = (
                        aligning(original[j + 1] - original[j], solved[j + 1] - solved[j])
                        @ rotation
                    )
                axes = rest[:3, :3]
                pose[rig.bones[index]["name"]] = euler_angles(axes.T @ parent.T @ rotation @ axes)
                parent = rotation
