"""The top of a shoulder raised sideways: a wide shallow hollow at the acromion, the deltoid gathered.

With the arm up the deltoid shortens and its bulk draws in under the acromion (Leijnse et al., 2008,
https://pmc.ncbi.nlm.nih.gov/articles/PMC2666142/, for its origins and tendons; nothing measured
says how the surface reads, so this is a stylised shape chosen by eye). A shape in the rest skin:
the skin just outside the acromion sinks along its normal, and the skin on the back of the arm's top
slides towards the joint and swells. Added to the shoulder's targets after they are made, under the
same names and driver, as sotaigen/corrective/pectoral.py is, so the runtime, the GLB and the contact tables'
keys need nothing new.
"""

import numpy as np

from sotaigen.corrective.axilla import NAMES, SAMPLES, driver_metadata
from sotaigen.form.field import ramp
from sotaigen.mesh import triangulate, vertex_normals
from sotaigen.rig.skinning import torso_length

# The hollow: its depth and width in torso shares, its centre from the shoulder joint, out and over it:
# on the step between the trapezius and the arm. Further in it only deepens the valley by the neck.
HOLLOW_DEPTH = 0.014
HOLLOW_WIDTH = 0.09
HOLLOW_CENTRE = (0.07, 0.008, 0.035)
# The gather, in torso shares: how far the skin slides towards the joint and how much it swells, over
# how far down the arm from the joint; only on the skin facing back and a little out.
GATHER_SHIFT = 0.008
GATHER_SWELL = 0.004
GATHER_REACH = 0.30
# How much of each comes in at each side-raise key of the table, and where a forward raise takes it off.
HOLLOW_GAIN = {35: 0.0, 72: 0.5, 110: 1.0}
GATHER_GAIN = {35: 0.4, 72: 0.8, 110: 1.0}
FORWARD_OFF = 75.0


def side_raises(targets):
    """(side, target name, forward, spread) for every shoulder target, in the table's angles."""
    table = driver_metadata(targets)
    if table.get("kind") != "simplices":
        return [
            (side, name, x, z)
            for side, names in NAMES.items()
            for name, (x, z) in zip(names, SAMPLES, strict=True)
        ]
    out = []
    axes = table.get("axes", [0, 1, 2])
    for side, local in table["sides"].items():
        for name, point in zip(local["targets"], local["points"], strict=True):
            if name:
                angles = np.zeros(3)
                angles[axes] = point
                out.append((side, name, float(angles[0]), float(angles[2])))
    return out


def _gain(gains, spread):
    """The share at this side raise, linear between the keys, none below the first."""
    keys = sorted(gains)
    return float(np.interp(spread, keys, [gains[k] for k in keys], left=0.0))


def shapes(base, verts, bones):
    """{side: (hollow, gather)}: the full shapes in the rest skin, each V x 3."""
    verts = np.asarray(verts, float)
    raw = np.asarray(base["verts"])
    segments = np.asarray(base["segments"])
    T = torso_length(bones)
    lookup = {bone["name"]: bone for bone in bones}
    normals = vertex_normals(verts, triangulate(base["faces"], raw))
    out = {}
    for side, sign in (("left", 1.0), ("right", -1.0)):
        scale = np.array((sign, 1.0, 1.0))
        on_side = raw[:, 0] * sign > 0
        # The hollow, on what faces up at the acromion: trunk and the arm's top, not the neck or head.
        centre = np.asarray(lookup[f"{side}UpperArm"]["head"], float) + np.array(HOLLOW_CENTRE) * T * scale
        offset = verts - centre
        spread = np.hypot(offset[:, 0], offset[:, 1] / 1.2) / (HOLLOW_WIDTH * T)
        bell = np.exp(-(spread**2)) * ramp((normals[:, 2] - 0.1) / 0.4)
        bell = bell * ramp((0.14 * T - np.abs(offset[:, 2])) / (0.06 * T))
        bell = np.where(on_side & np.isin(segments, ("trunk", "upperArm")), bell, 0.0)
        hollow = -HOLLOW_DEPTH * T * bell[:, None] * normals
        # The gather, along the arm from the joint.
        arm = lookup[f"{side}UpperArm"]
        head = np.asarray(arm["head"], float)
        axis = np.asarray(arm["tail"], float) - head
        axis = axis / np.linalg.norm(axis)
        along = (verts - head) @ axis
        fade = ramp((GATHER_REACH * T - along) / (0.7 * GATHER_REACH * T))
        fade = fade * ramp((along + 0.08 * T) / (0.08 * T))
        # What faces back and a little out: the top of a raised arm stays smooth, a swell there is a step.
        out_of_trunk = np.array((0.5 * sign, 1.0, 0.0))
        out_of_trunk = out_of_trunk - axis * (out_of_trunk @ axis)
        out_of_trunk = out_of_trunk / np.linalg.norm(out_of_trunk)
        facing = ramp((normals @ out_of_trunk + 0.3) / 0.8)
        reach = np.where(on_side & np.isin(segments, ("trunk", "upperArm")), fade * facing, 0.0)
        swell = np.exp(-(((along - 0.08 * T) / (0.12 * T)) ** 2))
        gather = (
            -GATHER_SHIFT * T * reach[:, None] * axis
            + GATHER_SWELL * T * (reach * swell)[:, None] * normals
        )
        out[side] = hollow, gather
    return out


def gather(targets, base, verts, bones):
    """The shoulder's targets with the raised shoulder's shape added at the side-raise keys."""
    if not targets:
        return targets
    made = shapes(base, verts, bones)
    out = {name: delta.copy() for name, delta in targets.items()}
    for side, name, forward, spread in side_raises(targets):
        off = float(ramp(1 - forward / FORWARD_OFF))
        hollow, arm = made[side]
        out[name] += off * (_gain(HOLLOW_GAIN, spread) * hollow + _gain(GATHER_GAIN, spread) * arm)
    if hasattr(targets, "driver"):
        return type(targets)(out, targets.driver)
    return out
