"""What every pose corrective's driver does the same way: where a reading falls between two keys,
and the rest skin with the targets a pose drives added."""

from bisect import bisect_left

import numpy as np

from sotaigen.form.field import ramp


def interval(keys, value):
    """{key: weight} of the two keys either side of `value`, eased between them, clamped to the ends."""
    value = np.clip(value, keys[0], keys[-1])
    index = min(max(bisect_left(keys, value), 1), len(keys) - 1)
    a, b = keys[index - 1 : index + 1]
    t = float(ramp((value - a) / (b - a)))
    return {a: 1 - t, b: t}


def bound(verts, targets, weights):
    """The rest skin with each target added at its weight. Unchanged, and the same array, when a pose
    drives nothing; never changed in place."""
    out = verts
    for name, weight in weights.items():
        if weight:
            if out is verts:
                out = verts.copy()
            out += weight * targets[name]
    return out


# Every pose corrective's table has one shape (extras.sotaigen.correctives, docs/systems/export.md): groups of
# targets, each weighted by a few axes and fades that read one bone's swing and twist.
#
#   {"name", "source", "axes": [{"bone", "read", "scale", "keys"}], "fades": [{"bone", "read", "limit"}],
#    "samples": [[key of each axis], ...], "targets": [name, ...]}
#
# A target's weight is the product of each axis's interval weight at its sample's key, times the fades:
# smoothstep(1 - |reading| / limit), multiplied bone by bone in the order the bones first come, and 0 from
# LIMIT_SLACK below the limit on: a pose turned to the limit reads a hair short of it through the swing and
# twist split, which left a weight of 1e-32 there instead of none (a .blend reads float32 angles). A
# reading is the bone's rotation in the pose as swing and twist (sotaigen.rig.skinning.swing_twist),
# "swingX", "twist", "swingZ", or "largest" for the largest of the three by size, times `scale` (+1 or
# -1, which reads a right-side bone as the left). A pose names a bone, or a limb for both sides with the
# right mirrored, as sotaigen.rig.skinning.sided does; later entries win. A group with "kind" is the
# experimental IPC shoulder table (sotaigen.corrective.pose_grid), which isn't part of the contract.
READS = {"swingX": 0, "twist": 1, "swingZ": 2}
# Degrees: web/runtime/driver.mjs and blender/src/sotaigen_blender/rig.py hold the same.
LIMIT_SLACK = 1e-4


def intents(pose):
    """The rotation the pose asks of each bone, by bone, before any helper takes a share."""
    from sotaigen.rig.skinning import capital, names_one_bone

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


def reading(turns, bone, read, scale=1):
    """One reading of a bone's swing and twist, in degrees: `turns` is intents' and its swing_twist cache."""
    from sotaigen.rig.skinning import swing_twist

    if bone not in turns["read"]:
        turns["read"][bone] = swing_twist(turns["intents"].get(bone, (0.0, 0.0, 0.0)))
    values = turns["read"][bone]
    value = max(abs(v) for v in values) if read == "largest" else values[READS[read]]
    return value * scale if scale != 1 else value


def fade_share(value, limit):
    """A fade's share at a reading: smoothstep(1 - |value| / limit), and 0 within LIMIT_SLACK of the limit."""
    if abs(value) >= limit - LIMIT_SLACK:
        return 0.0
    return float(ramp(1 - abs(value) / limit))


def group_weights(pose, group, turns=None):
    """{target: weight} of one group for a pose."""
    turns = turns or {"intents": intents(pose), "read": {}}
    if group.get("kind") == "simplices":
        from sotaigen.corrective.pose_grid import side_weights

        return side_weights(turns, group)
    by_bone = {}
    for fade in group["fades"]:
        share = fade_share(reading(turns, fade["bone"], fade["read"]), fade["limit"])
        by_bone[fade["bone"]] = by_bone[fade["bone"]] * share if fade["bone"] in by_bone else share
    gain = 1.0
    for index, share in enumerate(by_bone.values()):
        gain = share if index == 0 else gain * share
    tents = [
        interval(axis["keys"], reading(turns, axis["bone"], axis["read"], axis.get("scale", 1))) for axis in group["axes"]
    ]
    out = {}
    for sample, name in zip(group["samples"], group["targets"], strict=True):
        weight = tents[0].get(sample[0], 0)
        for tent, key in zip(tents[1:], sample[1:], strict=True):
            weight = weight * tent.get(key, 0)
        out[name] = weight * gain
    return out


def weights(pose, groups):
    """{target: weight} of every group for a pose."""
    turns = {"intents": intents(pose), "read": {}}
    out = {}
    for group in groups:
        out |= group_weights(pose, group, turns)
    return out
