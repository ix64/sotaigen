"""The pectoral plate's outer part carried towards a raised upper arm, added to the shoulder's targets.

The plate (sotaigen/form/pectoral.py) is a thickness laid on the chest wall at rest, and skinning keeps it on
the ribs, so a raised arm leaves the muscle where it was: its lower border stays level instead of
running up into the front fold of the armpit. The muscle's outer fibres converge on the humerus
(Haładaj et al. 2019, doi:10.1155/2019/6212039) and pivot about the sternum as their insertion rises,
so here the thickness is read back through a share of the move of a point on the humerus below the
joint: none by the sternum, `FOLLOW` at the plate's outer end. Turning the chest about the shoulder
joint instead pulls it down, since the chest lies inside the joint.
What the arm carries away is thinned, as a stretched muscle flattens. A stylised shape, not the
muscle's mechanics: the shares are engineering values chosen by eye.

The deltas go on the shoulder's own targets, under the same names and driver, so the runtime and the
GLB need nothing new, and they are added after the shoulder's targets are made, so the shoulder's
sculpting and its baked contact tables (and their cache keys) are untouched. A body without a plate
gets its targets back as they were.
"""

import numpy as np

from sotaigen.corrective.axilla import NAMES, SAMPLES, driver_metadata, sample_pose
from sotaigen.form.field import ramp
from sotaigen.form.pectoral import PectoralPlate
from sotaigen.mesh import triangulate, vertex_normals
from sotaigen.rig.skinning import pose_skins

# The share of the insertion's move the plate follows: none inside INNER, all of FOLLOW outside OUTER
# (torso shares from the midline); how much thinner it gets per quarter turn of the arm it is carried;
# the insertion, on the humerus this far (a torso share) below the joint; and how wide (a torso share,
# per quarter turn of the arm) the thickness is averaged. The plate's lower border drops to nothing within
# about two of the mesh's edges, so a border carried across the rows of edges still comes out as steps.
INNER, OUTER, FOLLOW, THIN, INSERTION, BLUR = 0.06, 0.22, 0.6, 0.35, 0.12, 0.06
# Unit offsets across the chest's front (x, z): the centre and a ring of six.
RING = np.array([(0.0, 0.0)] + [(np.cos(a), np.sin(a)) for a in np.arange(6) * np.pi / 3])


def plate_of(field):
    """The pectoral plate in a body's field, or None."""
    while field is not None and not isinstance(field, PectoralPlate):
        field = getattr(field, "__dict__", {}).get("reference")
    return field


def samples(targets):
    """(side, target name, local upper-arm Euler angles) for every shoulder target.

    The tables hold the left arm's swing and twist; the right's are mirrored, as axilla.targets and
    axilla.normal_targets pose them.
    """
    def posed(side, angle, twist, spread):
        return sample_pose(side, angle, twist, spread)[side + "UpperArm"]

    table = driver_metadata(targets)
    if table.get("kind") != "simplices":
        return [
            (side, name, posed(side, angle, 0, spread))
            for side, names in NAMES.items()
            for name, (angle, spread) in zip(names, SAMPLES, strict=True)
        ]
    out = []
    for side, local in table["sides"].items():
        for name, point in zip(local["targets"], local["points"], strict=True):
            if name:
                angles = np.zeros(3)
                angles[table.get("axes", [0, 1, 2])] = point
                out.append((side, name, posed(side, *angles)))
    return out


def _turn(rotation):
    """The angle of a rotation matrix, in radians."""
    return float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))


def follow(targets, base, verts, bones, field):
    """The shoulder's targets with the plate's change under each one's pose added."""
    plate = plate_of(field)
    if plate is None or not targets:
        return targets
    T = plate.torso
    verts = np.asarray(verts, dtype=float)
    lookup = {bone["name"]: i for i, bone in enumerate(bones)}
    normals = vertex_normals(verts, triangulate(base["faces"], np.asarray(base["verts"])))
    rest = plate.thickness(verts)
    # Where the plate is or can be carried to: the chest's front between the navel and the neck.
    height = (verts[:, 2] - plate.crotch) / T
    near = np.flatnonzero((verts[:, 1] < 0) & (height > 0.55) & (height < 1.1) & (np.abs(verts[:, 0]) < 0.4 * T))
    out = {name: delta.copy() for name, delta in targets.items()}
    for side, name, angles in samples(targets):
        sign = 1 if side == "left" else -1
        bone = bones[lookup[side + "UpperArm"]]
        skin = pose_skins(bones, {side + "UpperArm": angles})[lookup[side + "UpperArm"]]
        head, tail = (np.asarray(bone[end], dtype=float) for end in ("head", "tail"))
        insertion = head + INSERTION * T * (tail - head) / np.linalg.norm(tail - head)
        carry = skin[:3, :3] @ insertion + skin[:3, 3] - insertion
        rows = near[verts[near, 0] * sign > 0]
        points = verts[rows]
        share = FOLLOW * ramp((np.abs(points[:, 0]) / T - INNER) / (OUTER - INNER))
        # Where the thickness now at a point came from: its share of the insertion's move back.
        read = points - share[:, None] * carry
        carried = share * _turn(skin[:3, :3]) / (np.pi / 2)
        # The whole border turns, not only the part carried furthest, so it is all softened alike.
        radius = BLUR * T * min(_turn(skin[:3, :3]) / (np.pi / 2), 1)
        ring = np.zeros((len(RING), len(read), 3))
        ring[:, :, [0, 2]] = RING[:, None, :] * radius
        thickness = plate.thickness((read[None] + ring).reshape(-1, 3)).reshape(len(RING), -1).mean(0)
        moved = thickness * np.clip(1 - THIN * carried, 0.5, 1)
        out[name][rows] += (moved - rest[rows])[:, None] * normals[rows]
    if hasattr(targets, "driver"):
        return type(targets)(out, targets.driver)
    return out
