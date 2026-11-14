"""A hand placed on its own head: the arm's angles that put it there, solved for one finished body.

The same joint angles land elsewhere on a body with another head or other arms, so a pose that touches
the head has to be solved per body. What this gives is angles, which go through sotaigen.rig.skinning.pose_verts
and everything after it unchanged: nothing here reaches the runtime or the GLB. See docs/systems/touch.md.

A contact names a place on the head, read off the head's own skin, and the part of the hand that goes
there. The solve moves the arm's seven joints within the joint limits (sotaigen.rig.limits.joint_limits) so that
the hand arrives at that place, turned as the contact says, CLEARANCE_MM off the skin, and of the
solutions that leaves picks the one with the elbow low and out, the wrist near neutral and no arm in the
head, the neck or the trunk. It poses with sotaigen.rig.skinning, so a change to the shoulder's rhythm or to the
wrist's bones reaches it without a second copy of the skeleton's rules.
"""

from dataclasses import dataclass, field, replace
from itertools import pairwise

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from sotaigen.diagnostics import Diagnostic
from sotaigen.mesh import triangulate
from sotaigen.rig.limits import from_split, joint_limits, limit_order
from sotaigen.rig.skinning import capital, pose_skins

# What the hand keeps off the skin, in mm: touching, no pressing, no air anyone would see.
CLEARANCE_MM = 1.5
# The face's eye line, in head heights below the crown (sotaigen/output/face.py's eyes). A head swapped in says its own.
EYE_T = 0.605
# Below this the hand has arrived, whatever the reader is looking at: mm, and degrees of the palm's turn.
ARRIVED_MM = 2.0
ARRIVED_DEG = 25.0
# Fingertips arrive with the fingers curled, so the palm's own turn says less.
ARRIVED_TIPS_DEG = 35.0
# The bones on the left that neither carry the arm nor hang from it.
OFF_ARM_BONES = ("UpperLeg", "LowerLeg", "Foot", "Toes", "Eye")
LONG_FINGERS = ("index", "middle", "ring", "little")
JOINTS = ("Proximal", "Intermediate", "Distal")
# Upper arm x, y, z as its limit splits them; elbow; wrist x, y, z, each from the joint limits. Behind the
# back there's no limit in the data, but no contact asks for one.
FREE_BACK_DEG = -60.0
# A pose is rounded to 0.001 degrees in the control's own Euler angles, which a limit that splits the turn
# another way reads as a little more or less: the solve stops this short of such a limit.
SPLIT_LIMIT_MARGIN_DEG = 0.01
# A turn has two splits whose magnitudes add up to 540 degrees, and a limit reads the smaller one
# (sotaigen.rig.limits.split_yzx). The solve's upper arm stays clear under half of that, so that what it asks is
# what's read: at 270 and the lock the two are one, and 32-bit floats pick either.
SPLIT_READ_UP_TO_DEG = 260.0
# Within this of 90 degrees on the axis that locks, reading the upper arm's angles in a file's 32-bit floats
# moves the skin by 0.01 mm, more than anything else between the file and the runtime: the solve keeps off.
ANGLES_LOCK_DEG = 1.5
ANGLES_LOCK_PER_DEG = 10.0
# Where the wrist can get to, coarsely, for the solve's starts: the upper arm's x, z and y, the elbow.
GRID_FORWARD = (-20.0, 20.0, 60.0, 100.0, 140.0)
GRID_SPREAD = (-30.0, 10.0, 50.0, 90.0, 112.0)
GRID_ROLL = (-80.0, -40.0, 0.0, 40.0)
GRID_ELBOW = (70.0, 105.0, 140.0)
STARTS_KEPT = 6
STARTS_REFINED = 2
# The forearm's roll has three wells for a palm that has to face one way.
START_FOREARM = (-60.0, 0.0, 50.0)
# Starts this near each other in mm, at the elbow, are one start.
STARTS_APART_MM = 60.0
# How much a mm off the place weighs in the last stretch, and how many times the aim is moved to settle.
ARRIVING = 15.0
SETTLING = 4


@dataclass(frozen=True)
class Contact:
    """A place on the head and how the hand arrives there.

    `places` is where, in order, each as (across, below eye, fingers): `across` is the share of the face's
    half width out from the midline towards the hand's own side, `below_eye` is in head heights below the
    eye line (not used from the top), `fingers` is the way the fingers point in the head's frame for the
    left hand (x to the wearer's left, y back, z up), taken along the skin when `palm_on_skin`. The first
    place that an arm reaches is the place; short arms under a big head get a later one. `from_` says which
    way a place is cast onto the skin: "top" down from above, "front" back along the view, "back" forward
    from behind. `part` is the hand's "palm" or "fingertips". `both` says that both hands can do this at
    once, each staying on its own side.
    """

    name: str
    from_: str
    places: tuple
    part: str
    both: bool = False
    # Whether the palm's side lies against the skin. Pointing fingertips don't say which way it faces.
    palm_on_skin: bool = True
    # What an elbow forward of the shoulder pays per mm, for a hand that has to go behind the head.
    elbow_back: float = 0.0
    # How far off the midline the elbow stays, as a share of the shoulder joint's distance from it.
    elbow_off_midline: float = 0.1
    # Where only the fingertips arrive, for an arm that can't lay the palm on any of `places`.
    tips_only: tuple = ()


CONTACTS = (
    Contact(
        "crown",
        "top",
        (
            (0.3, 0.0, (-0.5, 0.85, 0.0)),
            (0.5, 0.0, (-0.6, 0.6, 0.5)),
            (0.7, 0.0, (-0.6, 0.2, 0.8)),
            (0.85, 0.0, (-0.5, 0.1, 0.9)),
        ),
        "palm",
        elbow_off_midline=0.8,
        tips_only=(
            (0.5, 0.0, (-0.8, 0.0, 0.6)),
            (0.7, 0.0, (-0.8, 0.0, 0.6)),
            (0.85, 0.0, (-0.8, 0.0, 0.6)),
        ),
    ),
    Contact("forehead", "front", ((0.0, -0.255, (-0.75, 0.0, 0.65)),), "palm", elbow_off_midline=0.4),
    Contact("cheek", "front", ((0.78, 0.115, (0.0, 0.25, 1.0)),), "palm", both=True, elbow_off_midline=0.3),
    Contact("mouth", "front", ((0.22, 0.223, (-0.5, 0.6, 0.5)),), "fingertips", palm_on_skin=False),
    Contact(
        "occiput",
        "back",
        (
            (0.85, -0.12, (-0.8, 0.5, 0.2)),
            (0.92, -0.12, (-0.8, 0.5, 0.2)),
            (0.98, -0.12, (-0.8, 0.5, 0.2)),
        ),
        "fingertips",
        both=True,
        elbow_back=0.15,
    ),
)
BY_NAME = {contact.name: contact for contact in CONTACTS}

# The fingers' curl at a palm's contact before they close on the skin, and the most they close to.
FLAT_CURL = {"index": (3, 5, 2), "middle": (4, 6, 3), "ring": (5, 7, 3), "little": (6, 8, 4)}
CLOSED_CURL = {"index": (26, 40, 25), "middle": (30, 44, 28), "ring": (32, 46, 30), "little": (34, 48, 30)}
# Light fingertips, at the mouth: POSES["handLightTouch"]'s curl, and the two that arrive a little straighter.
TIPS_CURL = {"index": (20, 22, 10), "middle": (18, 26, 13), "ring": (24, 40, 22), "little": (28, 44, 25)}
THUMB = {"thumbMetacarpal": (10, -5, -4)}
# The fingers fan out a little, as POSES["handLightTouch"]'s do: curled parallel, their tips meet.
SPREAD = {"index": 5.0, "middle": 1.0, "ring": -2.0, "little": -5.0}
PALM_CUP = -6.0
TIPS_CUP = -10.0
# How much further than its neighbour a finger closes, as a share of the way from flat to closed.
NEIGHBOUR_CURL = 0.25
# How the seven joints' discomfort and the elbow's place weigh against mm off the target.
WRIST_PER_DEG = 0.12
# Past these a wrist isn't just off neutral, it's strained: flexion, forearm roll, deviation.
WRIST_EASY = ((-35.0, 60.0), (-50.0, 45.0), (-18.0, 10.0))
WRIST_STRAIN_PER_DEG = 0.6
ROLL_PER_DEG = 0.04
ELBOW_LOW_PER_MM = 0.12
ELBOW_OUT_PER_MM = 0.03
# An elbow further in than the contact's `elbow_off_midline` has the upper arm across the chest or the
# face: it pays this.
ELBOW_INBOARD_PER_MM = 0.5
# The elbow's rule: under the shoulder is low, a third of the upper arm out from it is out.
ELBOW_OUT_SHARE = 0.33
LEVER_PALM_MM = 100.0
LEVER_FINGERS_MM = 35.0
# How near the arm comes to the head and to the trunk before it pays, mm.
ARM_MARGIN_MM = 5.0
TRUNK_MARGIN_MM = 3.0
# Within this of the skin, a point's distance is along the skin's normal.
NEAR_SKIN_MM = 12.0


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


class Solid:
    """Signed distance to a stretch of skin that a pose doesn't move, from samples over its triangles.

    Near the skin the distance is along the nearest sample's normal, so that it doesn't depend on how
    far apart the samples are.
    """

    def __init__(self, verts, triangles, steps=2):
        corners = verts[triangles]
        weights = np.array(
            [(a, b, steps - a - b) for a in range(steps + 1) for b in range(steps + 1 - a)],
            float,
        ) / steps
        self.points = np.einsum("wc,tci->twi", weights, corners).reshape(-1, 3)
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        self.normals = np.repeat(normals, len(weights), axis=0)
        self.tree = cKDTree(self.points)
        self.triangles = corners
        self.centre = self.points.mean(0)
        self.reach = float(np.linalg.norm(self.points - self.centre, axis=1).max())

    def distance(self, points, beyond=40.0):
        """Signed distance, negative inside; far points are `beyond` without asking the tree."""
        out = np.full(len(points), float(beyond))
        near = np.linalg.norm(points - self.centre, axis=1) < self.reach + beyond
        if near.any():
            gap, nearest = self.tree.query(points[near])
            side = np.sum((points[near] - self.points[nearest]) * self.normals[nearest], axis=1)
            # Past the stretch's open edge the nearest sample's normal says nothing: that isn't inside.
            inside = side < -0.7 * gap
            out[near] = np.where(gap < NEAR_SKIN_MM, side, np.where(inside, -gap, gap))
        return out

    def cast(self, start, direction):
        """Where a ray from `start` first meets the skin, and the outward normal there."""
        corners = self.triangles
        e1, e2 = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
        h = np.cross(direction, e2)
        det = np.sum(e1 * h, axis=1)
        usable = np.abs(det) > 1e-12
        det = np.where(usable, det, 1.0)
        s = start - corners[:, 0]
        u = np.sum(s * h, axis=1) / det
        q = np.cross(s, e1)
        v = (q @ direction) / det
        t = np.sum(e2 * q, axis=1) / det
        hit = usable & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        if not hit.any():
            raise ValueError("the ray misses the skin")
        best = np.flatnonzero(hit)[np.argmin(t[hit])]
        normal = unit(np.cross(e1[best], e2[best]))
        if normal @ direction > 0:
            normal = -normal
        return start + direction * t[best], normal


@dataclass(frozen=True)
class HeadAnchors:
    """Where a head's skin puts its crown, its eye line and its face's width.

    `crown_z` is the skin's top, `height` from there to the chin, `half_width` the face's half width at
    the eye line, `axis_y` the skull's front to back middle. All of it is measured on the skin, so a head
    swapped in gets its own; only `eye_t` is told.
    """

    crown_z: float
    height: float
    half_width: float
    axis_y: float
    eye_t: float = EYE_T

    @classmethod
    def from_skin(cls, base, verts, eye_t=EYE_T):
        segments = np.asarray(base["segments"])
        head = np.asarray(verts)[segments == "head"]
        crown = float(head[:, 2].max())
        middle = np.abs(head[:, 0]) < 1.0
        # The chin is the midline's lowest point on the face's front: forward of the skull's middle by
        # this share of the way to the nose's level, which leaves the throat out on every body.
        axis_y = float((head[:, 1].min() + head[:, 1].max()) / 2)
        front = middle & (head[:, 1] < axis_y - 0.45 * (axis_y - head[:, 1].min()))
        height = crown - float(head[front, 2].min())
        eye_z = crown - eye_t * height
        band = np.abs(head[:, 2] - eye_z) < 0.03 * height
        return cls(crown, height, float(np.abs(head[band, 0]).max()), axis_y, eye_t)


def place(contact, where, anchors, skin):
    """One of a contact's places on the skin and the skin's outward normal there, for the wearer's left."""
    across, below_eye, _ = where
    x = across * anchors.half_width
    z = anchors.crown_z - (anchors.eye_t + below_eye) * anchors.height
    far = 2.0 * anchors.height
    if contact.from_ == "top":
        return skin.cast(np.array((x, anchors.axis_y, anchors.crown_z + far)), np.array((0.0, 0.0, -1.0)))
    y, way = (-far, 1.0) if contact.from_ == "front" else (far, -1.0)
    return skin.cast(np.array((x, anchors.axis_y + y, z)), np.array((0.0, way, 0.0)))


def along_skin(direction, normal):
    d = np.asarray(direction, float)
    return unit(d - normal * (d @ normal))


@dataclass
class Solved:
    """A contact's angles for the left hand, and what the solve can say about them."""

    contact: Contact
    angles: dict
    reached: bool
    arrivalMm: float
    palmDeg: float
    fingersDeg: float
    gapMm: float
    armGapMm: float
    elbow: tuple
    target: tuple
    # Which of the contact's places: 0 is the one it's named for. With `part`, what arrived there.
    place: int = 0
    part: str = "palm"
    notes: list = field(default_factory=list)

    def pose(self, side="left"):
        """The pose for "left", "right" (mirrored as sotaigen.rig.skinning.sided mirrors) or "both"."""
        if side == "left":
            return dict(self.angles)
        if side == "both":
            return {name[4].lower() + name[5:]: value for name, value in self.angles.items()}
        return {f"right{name[4:]}": (v[0], -v[1], -v[2]) for name, v in self.angles.items()}

    def report(self):
        return {
            "reached": self.reached,
            "arrivalMm": round(self.arrivalMm, 2),
            "palmDeg": round(self.palmDeg, 1),
            "fingersDeg": round(self.fingersDeg, 1),
            "gapMm": round(self.gapMm, 2),
            "armGapMm": round(self.armGapMm, 2),
            "elbow": [round(v, 1) for v in self.elbow],
            "place": self.place,
            "part": self.part,
            "notes": list(self.notes),
        }


class Arm:
    """The left arm of one finished body, ready to be posed and measured many times over."""

    def __init__(self, base, verts, bones, weights, limits=None):
        self.from_split = from_split
        self.limit_order = limit_order
        self.bones = bones
        names = [bone["name"] for bone in bones]
        verts = np.asarray(verts, float)
        self.verts, self.weights = verts, np.asarray(weights, float)
        # Only the arm and what carries it move, so only they are posed.
        by = {bone["name"]: bone for bone in bones}
        carrying, name = set(), "leftShoulder"
        while name:
            carrying.add(name)
            name = by[name]["parent"]
        self.chain = [
            bone
            for bone in bones
            if bone["name"] in carrying
            or (bone["name"].startswith("left") and not any(part in bone["name"] for part in OFF_ARM_BONES))
        ]
        self.at = {bone["name"]: i for i, bone in enumerate(self.chain)}
        # A bone off the chain doesn't move: it is the identity at the end of the chain's skins.
        self.to_chain = np.array([self.at.get(n, len(self.chain)) for n in names])
        segments = np.asarray(base["segments"])
        regions = np.asarray(base["regions"])
        left = np.asarray(base["verts"], float)[:, 0] > 0
        triangles = np.asarray(triangulate(base["faces"], np.asarray(base["verts"], float)))
        over = lambda mask: triangles[mask[triangles].all(axis=1)]
        self.skin = Solid(verts, over(np.isin(segments, ("head", "neck"))))
        hand_verts = np.flatnonzero((regions == "hand") & left)
        self.frame(by, verts[hand_verts])
        chest = (segments == "trunk") & (verts[:, 2] > self.shoulder[2] - 1.3 * self.upper_length)
        self.trunk = Solid(verts, over(chest), steps=1)
        self.anchors = HeadAnchors.from_skin(base, verts)
        self.low, self.high = self.bounds(limits or joint_limits(bones))
        forearm = np.flatnonzero((segments == "lowerArm") & left)
        upper = np.flatnonzero((segments == "upperArm") & left)
        self.hand = self.subset(hand_verts)
        self.hand_few = self.subset(hand_verts[::8])
        self.forearm = self.subset(forearm[::4])
        self.upper = self.subset(upper[::4])
        # The upper arm away from the shoulder, where it doesn't belong against the trunk.
        along = (verts[upper] - self.shoulder) @ unit(self.elbow_rest - self.shoulder) / self.upper_length
        self.upper_far = self.subset(upper[along > 0.45][::3])
        self.fingers = {
            finger: self.subset(np.flatnonzero(np.isin(segments, [f"{finger}{j}" for j in JOINTS]) & left))
            for finger in LONG_FINGERS
        }
        self.reachable = None

    def bounds(self, limits):
        rows = (
            limits["leftUpperArm"][0],
            limits["leftUpperArm"][1],
            limits["leftUpperArm"][2],
            limits["leftLowerArm"][0],
            *limits["leftHand"],
        )
        low, high = np.array([row[0] for row in rows]), np.array([row[1] for row in rows])
        low[0] = max(low[0], FREE_BACK_DEG)
        if self.limit_order("leftUpperArm") != "XYZ":
            low[:3] += SPLIT_LIMIT_MARGIN_DEG
            high[:3] -= SPLIT_LIMIT_MARGIN_DEG
        return low, high

    def subset(self, indices):
        """Skinning restricted to some vertices and the bones they have weight on."""
        used = np.flatnonzero((self.weights[indices] > 0).any(axis=0))
        return (
            np.column_stack((self.verts[indices], np.ones(len(indices)))),
            self.weights[indices][:, used],
            self.to_chain[used],
        )

    def skins(self, pose):
        """The chain's skins, then the identity for a bone off the chain."""
        return np.concatenate((pose_skins(self.chain, pose), np.eye(4)[None]))

    def skinned(self, skins, subset):
        points, weights, used = subset
        moved = np.einsum("bij,vj->bvi", skins[used][:, :3], points)
        return np.einsum("vb,bvi->vi", weights, moved)

    def frame(self, by, hand):
        """The hand's rest frame and its reference points, from its bones and the palm's skin."""
        wrist = np.asarray(by["leftHand"]["head"], float)
        knuckles = {f: np.asarray(by[f"left{capital(f)}Proximal"]["head"], float) for f in LONG_FINGERS}
        across = knuckles["index"] - knuckles["little"]
        along = unit((knuckles["middle"] + knuckles["ring"]) / 2 - wrist)
        normal = unit(np.cross(across, along))
        # Flexing a finger takes it to the palm: the side the bones' local Z turns to at X+.
        flex = unit(np.subtract(by["leftHand"]["tail"], wrist))
        align = np.asarray(by["leftHand"]["align"], float)
        if normal @ (align - flex * (align @ flex)) < 0:
            normal = -normal
        self.rest_normal, self.rest_along = normal, along
        centre = (wrist + sum(knuckles.values()) / 4 * 2) / 3
        rel = hand - centre
        off_axis = rel - np.outer(rel @ normal, normal)
        near = np.linalg.norm(off_axis, axis=1) < 0.25 * np.linalg.norm(knuckles["middle"] - wrist)
        self.rest_palm = centre + normal * float((rel[near] @ normal).max())
        self.rest_tips = sum(np.asarray(by[f"left{f}Distal"]["tail"], float) for f in ("Index", "Middle")) / 2
        self.wrist_rest = wrist
        self.shoulder = np.asarray(by["leftUpperArm"]["head"], float)
        self.elbow_rest = np.asarray(by["leftLowerArm"]["head"], float)
        self.upper_length = float(np.linalg.norm(self.elbow_rest - self.shoulder))

    def grid(self):
        """Where the wrist and the elbow get to over the coarse grid of the upper arm and the elbow."""
        if self.reachable is None:
            rows = []
            for x in GRID_FORWARD:
                for z in GRID_SPREAD:
                    for y in GRID_ROLL:
                        for e in GRID_ELBOW:
                            q = np.clip((x, y, z, e, 0.0, 0.0, 0.0), self.low, self.high)
                            skins = self.skins(self.arm_pose(q))
                            rows.append(
                                (q, self.place_of(skins, self.wrist_rest, "leftHand"),
                                 self.place_of(skins, self.elbow_rest, "leftLowerArm"))
                            )
            self.reachable = tuple(np.array([row[i] for row in rows]) for i in range(3))
        return self.reachable

    def place_of(self, skins, point, bone):
        matrix = skins[self.at[bone]]
        return matrix[:3, :3] @ point + matrix[:3, 3]

    def read_split(self, split):
        """The solve's upper arm, which is its limit's split so that the solve's bounds are the limit, as
        that limit reads it. Scaling towards zero stays inside the bounds."""
        if self.limit_order("leftUpperArm") == "XYZ":
            return split
        return split * min(SPLIT_READ_UP_TO_DEG / max(np.abs(split).sum(), 1e-9), 1.0)

    def upper_arm(self, split):
        """The control's Euler angles for the solve's upper arm: the reading nearest zero, which is the one
        a file's shoulder rhythm reads off the turn, whatever the control's channels say."""
        return self.from_split("leftUpperArm", self.read_split(split), (0.0, 0.0, 0.0))

    def angles_locking(self, split):
        """How far into ANGLES_LOCK_DEG of the lock the upper arm's Euler angles and its limit's split are."""
        return sum(
            max(ANGLES_LOCK_DEG - abs(abs(v) - 90.0), 0.0) for v in (self.upper_arm(split)[1], self.read_split(split)[2])
        )

    def arm_pose(self, q):
        return {
            "leftUpperArm": self.upper_arm(q[:3]),
            "leftLowerArm": (q[3], 0.0, 0.0),
            "leftHand": (q[4], q[5], q[6]),
        }

    def pose_of(self, q, curls, cup=PALM_CUP):
        pose = {**self.arm_pose(q), "leftPalmCup": (0.0, cup, 0.0)}
        for finger, row in curls.items():
            for joint, angle in zip(JOINTS, row, strict=True):
                fan = SPREAD[finger] if joint == JOINTS[0] else 0.0
                pose[f"left{capital(finger)}{joint}"] = (float(angle), 0.0, fan)
        for name, row in THUMB.items():
            pose[f"left{capital(name)}"] = tuple(float(v) for v in row)
        return pose

    def reference(self, skins, part):
        """Where the hand's reference point is, which way the palm faces and the fingers point."""
        matrix = skins[self.at["leftHand"]]
        rotation = matrix[:3, :3]
        rest = self.rest_palm if part == "palm" else self.rest_tips
        return rotation @ rest + matrix[:3, 3], rotation @ self.rest_normal, rotation @ self.rest_along


def hinge(distance, margin):
    """One number for how far some points are inside a margin."""
    short = np.maximum(margin - distance, 0.0)
    return float(np.sqrt(np.sum(short * short)))


def residuals(q, arm, contact, target, normal, fingers, curls, cup, weight, bodies):
    """What's wrong with these angles. Without `bodies` the skin isn't asked: fast, for the starts."""
    skins = arm.skins(arm.pose_of(q, curls, cup))
    point, palm, along = arm.reference(skins, contact.part)
    elbow = arm.place_of(skins, arm.elbow_rest, "leftLowerArm")
    rows = [weight * (point - target)]
    if contact.palm_on_skin:
        rows.append(LEVER_PALM_MM * (palm + normal))
    rows.append(LEVER_FINGERS_MM * (along - fingers))
    rows.append(
        [
            ELBOW_LOW_PER_MM * (elbow[2] - (arm.shoulder[2] - arm.upper_length)),
            ELBOW_OUT_PER_MM * (elbow[0] - (arm.shoulder[0] + ELBOW_OUT_SHARE * arm.upper_length)),
            ELBOW_INBOARD_PER_MM * max(contact.elbow_off_midline * arm.shoulder[0] - elbow[0], 0.0),
            contact.elbow_back * max(arm.shoulder[1] - elbow[1], 0.0),
            WRIST_PER_DEG * q[4],
            WRIST_PER_DEG * q[5],
            WRIST_PER_DEG * q[6],
            ROLL_PER_DEG * q[1],
            ANGLES_LOCK_PER_DEG * arm.angles_locking(q[:3]),
            *(
                WRIST_STRAIN_PER_DEG * (max(low - angle, 0.0) + max(angle - high, 0.0))
                for angle, (low, high) in zip(q[4:], WRIST_EASY, strict=True)
            ),
        ]
    )
    if bodies:
        hand = arm.skinned(skins, arm.hand_few)
        forearm = arm.skinned(skins, arm.forearm)
        far = arm.skinned(skins, arm.upper_far)
        rows.append(
            [
                # What the arm can't go through. Hand against the head is the contact's own business.
                hinge(arm.skin.distance(forearm), ARM_MARGIN_MM),
                hinge(arm.skin.distance(arm.skinned(skins, arm.upper)), ARM_MARGIN_MM),
                hinge(arm.trunk.distance(hand), TRUNK_MARGIN_MM),
                hinge(arm.trunk.distance(forearm), TRUNK_MARGIN_MM),
                hinge(arm.trunk.distance(far), TRUNK_MARGIN_MM),
                # Both hands at once: the whole arm keeps to its own side.
                hinge(hand[:, 0], ARM_MARGIN_MM) if contact.both else 0.0,
                hinge(forearm[:, 0], ARM_MARGIN_MM) if contact.both else 0.0,
                hinge(far[:, 0], ARM_MARGIN_MM) if contact.both else 0.0,
            ]
        )
    return np.concatenate([np.ravel(row) for row in rows])


def angle_between(a, b):
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1.0, 1.0))))


def close_fingers(arm, q, cup):
    """Curl each long finger until it rests on the skin: the most it can without coming too near."""
    low = {f: 0.0 for f in LONG_FINGERS}
    high = {f: 1.0 for f in LONG_FINGERS}

    def curls(share):
        return {
            f: tuple(a + share[f] * (b - a) for a, b in zip(FLAT_CURL[f], CLOSED_CURL[f], strict=True))
            for f in LONG_FINGERS
        }

    for _ in range(6):
        mid = {f: (low[f] + high[f]) / 2 for f in LONG_FINGERS}
        skins = arm.skins(arm.pose_of(q, curls(mid), cup))
        for f in LONG_FINGERS:
            if arm.skin.distance(arm.skinned(skins, arm.fingers[f])).min() >= CLEARANCE_MM:
                low[f] = mid[f]
            else:
                high[f] = mid[f]
    # Neighbours curled far apart would pass through each other: the one further on waits for the other.
    for order in (LONG_FINGERS, LONG_FINGERS[::-1]):
        for before, after in pairwise(order):
            low[after] = min(low[after], low[before] + NEIGHBOUR_CURL)
    return curls(low)


def starts_for(arm, contact, aim, fingers):
    """Distinct arm positions that bring the wrist near where the hand needs it, lowest elbow first."""
    angles, wrists, elbows = arm.grid()
    reach = np.linalg.norm((arm.rest_palm if contact.part == "palm" else arm.rest_tips) - arm.wrist_rest)
    wanted = aim - fingers * reach
    cost = np.linalg.norm(wrists - wanted, axis=1) + 0.1 * (elbows[:, 2] - arm.shoulder[2])
    chosen = []
    for i in np.argsort(cost):
        if all(np.linalg.norm(elbows[i] - elbows[j]) > STARTS_APART_MM for j in chosen):
            chosen.append(i)
        if len(chosen) == STARTS_KEPT:
            break
    return angles[chosen]


def solve_place(arm, contact, where):
    """The left hand at one place: the best of a few starts, settled off the skin, the fingers closed."""
    target, normal = place(contact, where, arm.anchors, arm.skin)
    fingers = along_skin(where[2], normal) if contact.palm_on_skin else unit(where[2])
    curls = dict(FLAT_CURL) if contact.part == "palm" else dict(TIPS_CURL)
    cup = PALM_CUP if contact.part == "palm" else TIPS_CUP

    def run(q, aim, weight, evaluations, bodies):
        return least_squares(
            residuals,
            np.clip(q, arm.low + 1e-3, arm.high - 1e-3),
            args=(arm, contact, aim, normal, fingers, curls, cup, weight, bodies),
            bounds=(arm.low, arm.high),
            max_nfev=evaluations,
            x_scale=10.0,
        )

    aim = target + normal * CLEARANCE_MM

    def wrong(q, weight=1.0, bodies=True):
        row = residuals(q, arm, contact, aim, normal, fingers, curls, cup, weight, bodies)
        return float(row @ row)

    first = []
    for q in starts_for(arm, contact, aim, fingers):
        rolled = [np.array((*q[:5], roll, q[6])) for roll in START_FOREARM]
        first.append(run(min(rolled, key=lambda q: wrong(q, bodies=False)), aim, 1.0, 10, False))
    # Without the skin an arm through the head looks fine, so they're told apart with it.
    first.sort(key=lambda found: wrong(found.x))
    best = min((run(found.x, aim, 3.0, 8, True) for found in first[:STARTS_REFINED]), key=lambda found: found.cost)
    best = run(best.x, aim, ARRIVING, 8, True)
    # Settle: what touches first may not be the reference point, so move the aim along the normal.
    for _ in range(SETTLING):
        skins = arm.skins(arm.pose_of(best.x, curls, cup))
        gap = float(arm.skin.distance(arm.skinned(skins, arm.hand)).min())
        if abs(gap - CLEARANCE_MM) < 0.15:
            break
        aim = aim + normal * (CLEARANCE_MM - gap)
        best = run(best.x, aim, ARRIVING, 5, True)
    if contact.part == "palm":
        curls = close_fingers(arm, best.x, cup)
    pose = arm.pose_of(best.x, curls, cup)
    pose = {k: tuple(round(float(v), 3) for v in row) for k, row in pose.items()}
    return measured(arm, contact, pose, target, normal, fingers, aim)


def solve_contact(arm, contact):
    """The first of a contact's places this arm arrives at, or the nearest try at the last."""
    tips = replace(contact, part="fingertips", palm_on_skin=False)
    tries = [(contact, where) for where in contact.places] + [(tips, where) for where in contact.tips_only]
    for i, (asked, where) in enumerate(tries):
        solved = solve_place(arm, asked, where)
        solved.place, solved.part, solved.contact = i, asked.part, contact
        if solved.reached:
            break
    return solved


def measured(arm, contact, pose, target, normal, fingers, aim):
    skins = arm.skins(pose)
    point, palm, along = arm.reference(skins, contact.part)
    gap = float(arm.skin.distance(arm.skinned(skins, arm.hand)).min())
    forearm = float(arm.skin.distance(arm.skinned(skins, arm.forearm)).min())
    upper = float(arm.skin.distance(arm.skinned(skins, arm.upper)).min())
    elbow = arm.place_of(skins, arm.elbow_rest, "leftLowerArm")
    arrival = float(np.linalg.norm(point - aim))
    palm_deg = angle_between(palm, -normal) if contact.palm_on_skin else 0.0
    fingers_deg = angle_between(along, fingers)
    in_body = min(forearm, upper) < 0.0
    palm_limit = ARRIVED_DEG if contact.part == "palm" else ARRIVED_TIPS_DEG
    reached = (
        arrival <= ARRIVED_MM
        and palm_deg <= palm_limit
        and CLEARANCE_MM - 0.5 <= gap <= CLEARANCE_MM + 3.0
        and not in_body
    )
    notes = []
    if arrival > ARRIVED_MM:
        notes.append(f"short of the place by {arrival:.1f} mm")
    if palm_deg > palm_limit:
        notes.append(f"the palm turned off by {palm_deg:.0f} degrees")
    if gap < CLEARANCE_MM - 0.5:
        notes.append(f"the hand {gap:.1f} mm off the skin")
    if gap > CLEARANCE_MM + 3.0:
        notes.append(f"the hand stops {gap:.1f} mm short of the skin")
    if in_body:
        notes.append("the arm goes through the head")
    return Solved(
        contact,
        pose,
        reached,
        arrival,
        palm_deg,
        fingers_deg,
        gap,
        min(forearm, upper),
        tuple(float(v) for v in elbow),
        tuple(float(v) for v in target),
        notes,
    )


def touch_name(contact, side=None):
    """A contact's pose name, `contact` being its name: crown is touchCrown, and its pose for a side, the
    names the page and the .blend use, touchCrown.left, touchCrown.right or touchCrown.both."""
    return f"touch{capital(contact)}" + (f".{side}" if side else "")


def solve_all(base, verts, bones, weights, contacts=CONTACTS):
    """Every contact's solve for one finished body, by name."""
    arm = Arm(base, verts, bones, weights)
    return {contact.name: solve_contact(arm, contact) for contact in contacts}


def pose_table(solved):
    """What the page asks for: the poses by name, left hand or both, and how each came out, as JSON values."""
    poses, reports = {}, {}
    for name, row in solved.items():
        for both in (False, True) if row.contact.both else (False,):
            key = touch_name(name, "both" if both else "left")
            poses[key] = {bone: list(angles) for bone, angles in row.pose("both" if both else "left").items()}
            reports[key] = row.report()
    return {"poses": poses, "reports": reports}


def blend_poses(solved):
    """What a .blend stores: a pose asset per contact and side, for the contacts this body's arm reaches."""
    out = {}
    for name, row in solved.items():
        if not row.reached:
            continue
        for side in ("left", "right", "both") if row.contact.both else ("left", "right"):
            out[touch_name(name, side)] = row.pose(side)
    return out


def unreached(solved):
    """A diagnostic for each contact this body's arm doesn't reach, for the file's quality notes."""
    return [
        Diagnostic(
            "touch.unreached",
            "warning",
            {
                "contact": name,
                "arrivalMm": round(float(row.arrivalMm), 1),
                "palmDegrees": round(float(row.palmDeg)),
                "gapMm": round(float(row.gapMm), 1),
                "throughHead": "the arm goes through the head" in row.notes,
            },
        )
        for name, row in solved.items()
        if not row.reached
    ]
