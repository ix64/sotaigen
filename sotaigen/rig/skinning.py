"""The skeleton's names and how a pose moves the skin: posing, blend skinning, delta mush.

Kept apart from rig.py, which places the bones and paints the weights, so that what only poses a
finished rig (the correctives, the contact bakes, the metrics) does not depend on how it was made. A limb's local Y runs along the bone, local Z leans to the
front and local X is the bend axis, so raising a limb forward is positive.
"""

import numpy as np

from sotaigen.mesh import adjacency

FINGERS = {
    "thumb": ("Metacarpal", "Proximal", "Distal"),
    "index": ("Proximal", "Intermediate", "Distal"),
    "middle": ("Proximal", "Intermediate", "Distal"),
    "ring": ("Proximal", "Intermediate", "Distal"),
    "little": ("Proximal", "Intermediate", "Distal"),
}
FINE = tuple(f"{finger}{joint}" for finger, joints in FINGERS.items() for joint in joints) + (
    "toes",
)
# A half bone sits at a joint, hangs from the parent and takes half of the limb's swing.
HALF_BONES = {
    "UpperArm": "Shoulder",
    "LowerArm": "UpperArm",
    "UpperLeg": "hips",
    "LowerLeg": "UpperLeg",
    "Hand": "LowerArm",
}
# A half bone hanging from the parent would keep none of the limb's roll, and where it holds the skin
# between two bones that do, rolling the limb cuts a ring around it. So the hand's half bone takes this
# share of the hand's true twist, midway between the forearm's last twist bone (0.75) and the hand.
HALF_ROLLS = {"Hand": 0.875}
# Twist bones spread a roll along a limb. Each is at a share of the limb's length and follows a
# share of the driver's roll: the upper arm cancels its own roll near the shoulder, the forearm
# takes on the hand's roll towards the wrist. The roll is the driver's true twist about its own Y
# (swing_twist), the upper arm's after the shoulder has taken its share, not the Euler Y angle.
TWISTS = {
    "UpperArm": {"driver": "UpperArm", "bones": ((0.2, -0.8), (0.55, -0.45)), "fullAt": 0.9},
    "LowerArm": {"driver": "Hand", "bones": ((0.4, 0.4), (0.75, 0.75)), "fullAt": None},
}
# The shoulder blade moves with the arm: a share of a raised or forward arm is done by the girdle.
SHOULDER_RHYTHM = 0.2
# A forward raise also elevates the girdle, by this share of the input, the arm turned back by as much so
# that it still points where it was asked to: the shoulder blade's upward rotation in flexion and abduction
# differs by 2-3 degrees (Ludewig et al., 2009). The input's forward degree raises the arm off vertical by
# about half of what a side degree does, so the share is about half of SHOULDER_RHYTHM.
FORWARD_LIFT = 0.1
MIDLINE_BONES = ("hips", "spine", "chest", "upperChest", "neck", "head", "jaw")
# Where the spine table puts the hips and the neck, as shares of the torso length.
HIPS_LEVEL, NECK_LEVEL = 0.17, 1.03


def capital(name):
    return name[0].upper() + name[1:]


class Skeleton(list):
    """A rig's bones, in the order of their weight columns, with what the rig records besides them.

    skeleton.json keeps the bones under "bones" and these at its top (sotaigen.rig.skeleton.save_rig):
    T       the torso length the bones were placed from, in mm
    glides  {bone: the mm it slides per degree of opening, a vector in its rest frame}; the jaw's
    groin   the body's own static groin amounts, which its pose targets undo (sotaigen.corrective.groin)

    copy() keeps the record; a list made from one some other way keeps only the bones.
    """

    def __init__(self, bones=(), *, T, glides=None, groin=None):
        super().__init__(bones)
        self.T = T
        self.glides = dict(glides or {})
        self.groin = dict(groin or {})

    def copy(self):
        return Skeleton(self, T=self.T, glides=self.glides, groin=self.groin)


def glides_of(bones):
    """Skeleton.glides, or none for a plain list of bones."""
    return getattr(bones, "glides", {})


def torso_length(bones):
    """The torso length T these bones were placed from (Skeleton.T)."""
    return bones.T


def rest_matrix(bone):
    """Bone-to-world at rest: local Y along the bone, local Z as near `align` as it can be."""
    y = bone["tail"] - bone["head"]
    y = y / np.linalg.norm(y)
    z = bone["align"] - y * (bone["align"] @ y)
    z = z / np.linalg.norm(z)
    x = np.cross(y, z)
    matrix = np.eye(4)
    matrix[:3, 0], matrix[:3, 1], matrix[:3, 2], matrix[:3, 3] = x, y, z, bone["head"]
    return matrix


def euler(degrees):
    x, y, z = np.radians(degrees)
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    rx = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]])
    ry = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]])
    rz = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return rz @ ry @ rx


def _quaternion(degrees):
    """euler(degrees) as (x, y, z, w), w not negative: rz * ry * rx, as driver.mjs fromEuler."""
    x, y, z = np.radians(degrees) / 2
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    q = (
        cz * cy * sx - sz * sy * cx,
        cz * sy * cx + sz * cy * sx,
        sz * cy * cx - cz * sy * sx,
        cz * cy * cx + sz * sy * sx,
    )
    return tuple(float(v) for v in q) if q[3] >= 0 else tuple(-float(v) for v in q)


def swing_twist(degrees):
    """How the pose correctives read a rotation: (swing x, twist, swing z) in degrees.

    The rotation splits into a twist about local Y followed by a swing that leaves Y's own component
    out, as Blender's Swing and Y Twist channels split it. The swing is given as its rotation vector
    (x, 0, z), so a turn about one axis reads as its own angle; Blender's channels give 2 asin of the
    swing quaternion's components instead, and blender/rig.py turns them into this. The right side's
    mirrored rotation (x, -y, -z) reads as (x, -twist, -z).
    """
    if sum(v != 0 for v in degrees) <= 1 and all(abs(v) <= 180 for v in degrees):
        # A turn about one axis, exactly, without the rounding of the general case.
        return tuple(float(v) for v in degrees)
    qx, qy, qz, qw = _quaternion(degrees)
    r = np.hypot(qw, qy)
    if r < 1e-12:
        # A half-turn swing: Blender's twist is then atan2(0, 0), zero.
        twist, sx, sz = 0.0, qx, qz
    else:
        twist = 2 * np.arctan2(qy, qw)
        sx, sz = (qx * qw + qz * qy) / r, (qz * qw - qx * qy) / r
    n = np.hypot(sx, sz)
    scale = 2 * np.arctan2(n, r) / n if n > 1e-12 else 2.0
    return float(np.degrees(scale * sx)), float(np.degrees(twist)), float(np.degrees(scale * sz))


def swing_euler(x, twist, z):
    """The Euler angles (degrees) that swing_twist reads back as (x, twist, z): the twist about local
    Y, then the swing whose rotation vector is (x, 0, z). The y angle comes out within +-90, but for
    a twist alone."""
    if sum(v != 0 for v in (x, twist, z)) <= 1 and all(abs(v) <= 180 for v in (x, twist, z)):
        # A turn about one axis, exactly, as swing_twist reads it back.
        return float(x), float(twist), float(z)
    angle = np.radians(np.hypot(x, z))
    s = np.sin(angle / 2) / np.radians(np.hypot(x, z)) if angle > 0 else 0.0
    sw = (np.radians(x) * s, 0.0, np.radians(z) * s, np.cos(angle / 2))
    t = np.radians(twist) / 2
    tw = (0.0, np.sin(t), 0.0, np.cos(t))
    # sw * tw, both as (x, y, z, w).
    qx = sw[3] * tw[0] + sw[0] * tw[3] + sw[1] * tw[2] - sw[2] * tw[1]
    qy = sw[3] * tw[1] - sw[0] * tw[2] + sw[1] * tw[3] + sw[2] * tw[0]
    qz = sw[3] * tw[2] + sw[0] * tw[1] - sw[1] * tw[0] + sw[2] * tw[3]
    qw = sw[3] * tw[3] - sw[0] * tw[0] - sw[1] * tw[1] - sw[2] * tw[2]
    sin_y = np.clip(2 * (qw * qy - qx * qz), -1, 1)
    return (
        float(np.degrees(np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))),
        float(np.degrees(np.arcsin(sin_y))),
        float(np.degrees(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))),
    )


def half_swing(degrees):
    """Half of a rotation's swing with its roll about local Y removed, as a matrix."""
    moved = euler(degrees)[:3, 1]
    axis = np.cross((0.0, 1.0, 0.0), moved)
    size = np.linalg.norm(axis)
    if size < 1e-9:
        return np.eye(4)
    axis = axis / size
    angle = np.arctan2(size, moved[1]) / 2
    skew = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
    )
    out = np.eye(4)
    out[:3, :3] = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew
    return out


def names_one_bone(name):
    """Whether a pose's name is a bone's own, rather than a limb on both sides."""
    return name.startswith(("left", "right")) or name in MIDLINE_BONES


def bones_named(name):
    """The bones a name in a pose stands for."""
    return (name,) if names_one_bone(name) else (f"left{capital(name)}", f"right{capital(name)}")


def check_pose(pose, bones):
    """Refuse a name no bone answers to: posing would ignore it and the pose would look fine."""
    known = {bone["name"] for bone in bones}
    unknown = [
        name
        for name in pose
        if name != "rootTranslation" and not known.issuperset(bones_named(name))
    ]
    if "rootTranslation" in pose:
        offset = np.asarray(pose["rootTranslation"], dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("rootTranslation must be three finite millimetre coordinates")
    if unknown:
        raise ValueError(f"unknown bone in pose: {', '.join(sorted(unknown))}")


def sided(pose):
    """A name without a side poses both; the right side mirrors, flipping roll and swing."""
    out = {}
    for name, (x, y, z) in pose.items():
        if name == "rootTranslation":
            continue
        if names_one_bone(name):
            out[name] = (x, y, z)
        else:
            out[f"left{capital(name)}"] = (x, y, z)
            out[f"right{capital(name)}"] = (x, -y, -z)
    for side, sign in (("left", 1.0), ("right", -1.0)):
        arm = out.get(f"{side}UpperArm")
        if arm is not None:
            forward = max(arm[0], 0.0) * SHOULDER_RHYTHM
            up = max(arm[2] * sign, 0.0) * SHOULDER_RHYTHM * sign
            up += max(arm[0], 0.0) * FORWARD_LIFT * sign
            x, y, z = out.get(f"{side}Shoulder", (0.0, 0.0, 0.0))
            out[f"{side}Shoulder"] = (x + forward, y, z + up)
            out[f"{side}UpperArm"] = (arm[0] - forward, arm[1], arm[2] - up)
        for name in HALF_BONES:
            if f"{side}{name}" in out:
                half = half_swing(out[f"{side}{name}"])
                if name in HALF_ROLLS:
                    twist = swing_twist(out[f"{side}{name}"])[1]
                    half = half @ euler((0.0, twist * HALF_ROLLS[name], 0.0))
                out[f"{side}{name}Half"] = half
        for name, spec in TWISTS.items():
            driver = out.get(f"{side}{spec['driver']}")
            if driver is None:
                continue
            twist = swing_twist(driver)[1]
            for i, (_at, share) in enumerate(spec["bones"], 1):
                out[f"{side}{name}Twist{i}"] = (0.0, twist * share, 0.0)
    return out


def bone_local(bone, rotation, glide=None):
    """A bone's local transform from sotaigen.rig.skeleton.sided's value for it, which is Euler degrees or a matrix.

    A bone with a `glide` (the jaw's, Skeleton.glides) also slides that many mm per degree of opening,
    its X past zero, along a vector in its rest frame: the turn and the slide come from the one angle.
    """
    if isinstance(rotation, np.ndarray):
        return rotation
    local = euler(rotation)
    if glide is not None:
        local[:3, 3] = np.asarray(glide, float) * max(rotation[0], 0.0)
    return local


# A skeleton's rest frames and their inverses, by the bones' names, parents and placement, for
# pose_skins. At most a few skeletons.
_REST_FRAMES = {}


def rest_frames(bones):
    """({name: rest_matrix}, {name: its inverse}) for these bones, computed once per skeleton."""
    key = tuple(
        (bone["name"], bone["parent"])
        + tuple(
            (np.asarray(bone[k]).dtype.str, np.asarray(bone[k]).tobytes())
            for k in ("head", "tail", "align")
        )
        for bone in bones
    )
    frames = _REST_FRAMES.get(key)
    if frames is None:
        rest = {bone["name"]: rest_matrix(bone) for bone in bones}
        frames = rest, {name: np.linalg.inv(matrix) for name, matrix in rest.items()}
        if len(_REST_FRAMES) >= 8:
            _REST_FRAMES.clear()
        _REST_FRAMES[key] = frames
    return frames


def pose_skins(bones, pose):
    """What each bone does to the skin under local rotations, as a bones x 4 x 4 array.

    The rest frames and their inverses are kept between calls (rest_frames), which makes it about five
    times faster than working them out each time, to the same bits.
    """
    check_pose(pose, bones)
    rotations = sided(pose)
    rest, inverse = rest_frames(bones)
    glides = glides_of(bones)
    world = {}
    for bone in bones:
        name = bone["name"]
        local = bone_local(bone, rotations.get(name, (0.0, 0.0, 0.0)), glides.get(name))
        if bone["parent"]:
            parent = bone["parent"]
            world[name] = world[parent] @ inverse[parent] @ rest[name] @ local
        else:
            world[name] = rest[name] @ local
            world[name][:3, 3] += pose.get("rootTranslation", (0, 0, 0))
    return np.array([world[b["name"]] @ inverse[b["name"]] for b in bones])


def skin_verts(skins, weights, verts):
    """Linear blend skinning with a pose's pose_skins already in hand."""
    homogeneous = np.column_stack((verts, np.ones(len(verts))))
    moved = np.einsum("bij,vj->bvi", skins, homogeneous)
    return np.einsum("vb,bvi->vi", weights, moved)[:, :3]


def pose_verts(bones, weights, verts, pose):
    """Linear blend skinning under local rotations. A name no bone answers to is a ValueError."""
    return skin_verts(pose_skins(bones, pose), weights, verts)


def triangle_fan(faces):
    """Every face split into a fan, as a T x 3 array."""
    return np.array(
        [(face[0], face[j], face[j + 1]) for face in faces for j in range(1, len(face) - 1)]
    )


def smoothed(points, source, sink, count, iterations, rate=0.5, free=None):
    """Uniform Laplacian smoothing, of `free` vertices only when given."""
    out = points.copy()
    for _ in range(iterations):
        total = np.zeros_like(out)
        np.add.at(total, source, out[sink])
        step = (total / count[:, None] - out) * rate
        out = out + (step if free is None else step * free[:, None])
    return out


def vertex_frames(points, source, sink, triangles):
    """A right-handed frame per vertex: the area-weighted normal, and one neighbour's direction."""
    corners = points[triangles]
    area = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normal = np.zeros_like(points)
    for i in range(3):
        np.add.at(normal, triangles[:, i], area)
    normal = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    first = np.zeros(len(points), int)
    first[source[::-1]] = sink[::-1]
    tangent = points[first] - points
    tangent = tangent - normal * np.sum(tangent * normal, axis=1, keepdims=True)
    tangent = tangent / np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    return np.stack((tangent, np.cross(normal, tangent), normal), axis=2)


class Mush:
    """Delta mush (Mancewicz et al., 2014) of a skinned mesh: smooth the pose, put the detail back.

    Fingers, toes and the head are left to LBS: their detail is the point of them, and smoothing
    them over gains nothing where rigid motion is already right.
    """

    def __init__(self, verts, faces, free, iterations=25):
        self.source, self.sink, self.count = adjacency(len(verts), faces)
        self.triangles = triangle_fan(faces)
        self.iterations = iterations
        self.free = free
        rest = smoothed(verts, self.source, self.sink, self.count, iterations, free=free)
        frames = vertex_frames(rest, self.source, self.sink, self.triangles)
        self.deltas = np.einsum("vji,vj->vi", frames, verts - rest)

    def __call__(self, posed):
        smooth = smoothed(posed, self.source, self.sink, self.count, self.iterations, free=self.free)
        frames = vertex_frames(smooth, self.source, self.sink, self.triangles)
        return smooth + np.einsum("vij,vj->vi", frames, self.deltas)


def mush_zone(base):
    """Smoothing weights: everything but fingers, toes, foot, hand and head."""
    regions = np.asarray(base["regions"])
    segments = np.asarray(base["segments"])
    return (np.isin(regions, ("trunk", "leg", "arm")) & (segments != "head")).astype(float)
