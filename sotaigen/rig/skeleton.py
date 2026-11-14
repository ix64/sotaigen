"""Skeleton and skin weights for any wrapped body, placed from its landmarks.

Bones use VRM humanoid names; left is the wearer's left, +X. A limb's local Y runs along the bone,
local Z leans to the front and local X is the bend axis, so raising a limb forward is positive.
Every length here is a share of the torso or the shoulder, never a character's millimetres.
Posing a finished rig is in skinning.py.
"""

import json

import numpy as np

from sotaigen.form.field import ramp
from sotaigen.form.head import _SeatPlane
from sotaigen.mesh import adjacency, edges_of
from sotaigen.rig.skinning import (
    FINE,
    FINGERS,
    HALF_BONES,
    HIPS_LEVEL,
    NECK_LEVEL,
    TWISTS,
    Mush,
    Skeleton,
    capital,
    mush_zone,
    pose_skins,
    pose_verts,
    rest_matrix,
)
from sotaigen.template import coarse, prolonged

FRONT = (0.0, -1.0, 0.0)
UP = (0.0, 0.0, 1.0)
# The finger joints are the template's, carried with the hand to where the wrap placed it, not the
# centres of their skin rings. The wrist seam's correction reaches the thumb's base and moves those
# rings with the forearm's girth and direction, by up to 0.05 palm lengths, and the shared grip
# angles have no room for that: a thick wrist (the toddler's, 1.32 hand lengths) put the thumb into
# the props. The placement is read from the segments the correction never reaches.
PLACED_WHOLE = tuple(
    f"{finger}{joint}"
    for finger in ("index", "middle", "ring", "little")
    for joint in ("Intermediate", "Distal")
) + ("thumbDistal",)
# Where the correction seats the thumb's two inner joints on the median body, in template
# millimetres (the nine bodies of 2026-11-06 spread 3-4 mm around it), so the thumb's joints stay
# near their swollen skin and the grips fitted on those bodies still hold.
THUMB_SEAT = {
    "thumbMetacarpal": np.array([-0.2, -0.15, -2.75]),
    "thumbProximal": np.array([0.05, 0.05, -0.9]),
}
MAX_INFLUENCES = 4
# The neck's weights are smoothed over this many head heights of skin, after the diffusion that
# counts edges. The template's top rows of neck crowd together under the head on a short neck, and
# the head's weight rose across them in a few millimetres: turning the head slid one row past the
# next along their slant and flattened faces between them. The face, above NECK_FACE head heights
# down from the crown, stays out of that smoothing, so the features and ears do not slide on the skin.
NECK_SMOOTH = 0.10
NECK_FACE = 0.9
MIN_WEIGHT = 0.01
# The jaw turns at the temporomandibular joints, just ahead of the ear's root and level with its
# lower half: shares of the head, TMJ_T down from the crown and TMJ_Y behind the head's axis.
TMJ_T, TMJ_Y = 0.63, 0.0
# Opening also slides the condyle forward and down its articular eminence, 35 degrees below the
# horizontal on average (Krohn et al., 2020). The share is of the jaw's length per degree. Rotation
# makes 77% of the incisors' travel (Sforza et al., 2009), which would be 0.005 here; but this chin is
# further below the joint than ahead of it, unlike a person's, so at that the chin mostly goes back
# into the neck. About two and a half times as much takes it down instead (docs/motion/jaw.toml).
JAW_GLIDE = 0.0125
JAW_GLIDE_SLOPE = 35.0
# What rides the jaw, in shares of the head: skin below a line from the joint to the mouth. The line
# ends at the mouth, MOUTH_T down from the crown and MOUTH_Y ahead of the joint, and the blend across
# it stays as wide as JAW_BLEND_BACK along the cheek and narrows to JAW_BLEND_MOUTH only near the
# mouth, so that the mouth's corners take about half of the chin's travel and the cheek has no edge
# to fold at. It fades out over JAW_BEHIND up to JAW_EAR_Y behind the head's axis, where the ear's
# root is, so the skin doesn't slide out from under the ear. Under the chin it fades to nothing at the
# neck ring over JAW_SEAT, so the ring and the neck stay off it.
MOUTH_T = 0.828
MOUTH_Y = 0.30
JAW_BLEND_BACK = 0.14
JAW_BLEND_MOUTH = 0.036
JAW_BEHIND = 0.10
JAW_EAR_Y = 0.03
JAW_SEAT = 0.18
# The jaw's share is smoothed over JAW_SMOOTH head heights of skin below JAW_SMOOTH_BELOW, as the neck's
# weights are, which is under the nose. Without it opening flattens faces where the rows crowd along
# the jaw's lower edge, stretches the skin at the mouth line to six or seven times its area, and
# leaves streaks down the cheek where the columns cross the blend.
JAW_SMOOTH = 0.06
JAW_SMOOTH_BELOW = 0.75
# The eyes turn about the middle of the socket: inward along the skin's normal from the iris, whose
# place is in face coordinates (sotaigen/output/face.py). Their length, as a share of the head, is the eyeball's
# radius: the placeholder's iris lies on that sphere, so a turn keeps it on the same surface, and the
# skin takes the sphere's edge as a lid would.
EYE_AT = (0.172, 0.604)
EYE_RADIUS = 0.14
# The sphere stands EYE_CLEAR_MM off the skin over a patch around the rest iris: half its width in u
# with EYE_LOOK degrees of gaze added, half its height in shares of the head.
EYE_CLEAR_MM = 0.40
EYE_PATCH = (0.037, 0.054)
EYE_LOOK = 12.0
# Half the head's width, in shares of the head, at which u is plain x over the head's height
# (sotaigen.output.face.REFERENCE_WIDTH). The rig has no parameters, so it reads the width off the skin.
REFERENCE_HALF_WIDTH = 0.40
# How far the wrist and ankle blend reaches, as a share of the forearm or shin, and how long it runs.
# It fades out over END_EDGE more of that radius: a hard edge left the thumb's web crease just
# outside it on most bodies, and inside on some, so its weights jumped with the hand's size.
END_BLEND = 0.17
END_EDGE = 0.3
END_SPREAD = 14
# The back of the wrist bends over a long arc while the palm folds in one crease, so only the back's
# weights are smoothed further, over this many hand lengths of skin. Azimuth is measured from the
# palm: full effect from WRIST_BACK[1] degrees towards the back, none at WRIST_BACK[0].
WRIST_BACK_REACH = 0.15
WRIST_BACK = (70.0, 150.0)
# The elbow's tip rides the ulna. On the back of the elbow the skin leaves the half bone's arc: it
# follows the forearm up to TIP_RISE elbow radii above the joint, then the upper arm, over a band of
# TIP_BAND radii. Azimuth is measured from the flexing side: full effect from TIP_BACK[1] degrees
# towards the back, none at TIP_BACK[0]. Near the far edge of TIP_FADE radii the old weights return.
TIP_RISE = 0.8
TIP_BAND = 1.0
TIP_BACK = (70.0, 150.0)
TIP_FADE = (2.2, 3.2)
# Rounds of diffusion over the handed-over skin. The columns of skin around the limb are coarse, so
# without it the side of the tip folds into a thin line at large angles.
TIP_SMOOTH = 1
# The waist-down fit: where it applies, which bones it may use and what it learns from. The band
# from FIT_FULL_BELOW (a share of the torso above the crotch) up to the waist is where it fades out.
FIT_PARTS = ("trunk", "upperLeg", "lowerLeg")
FIT_BONES = ("hips", "spine", "chest", "UpperLeg", "UpperLegHalf", "LowerLeg", "LowerLegHalf")
FIT_FULL_BELOW = 0.22
# How far down the shin it has faded out, as a share of the way from knee to ankle.
FIT_SHIN_OUT = 0.5
# Written out, not taken from sotaigen/rig/poses.py: changing a check pose must not change the weights.
FIT_EXAMPLES = (
    {"spine": (12, 0, 0), "upperLeg": (85, 0, 0), "lowerLeg": (-115, 0, 0), "foot": (28, 0, 0)},
    {"upperLeg": (45, 0, 0), "lowerLeg": (-60, 0, 0)},
    {"upperLeg": (30, 0, 0)},
    {"upperLeg": (60, 0, 0)},
    {"upperLeg": (90, 0, 0)},
    {"upperLeg": (-30, 0, 0)},
    {"upperLeg": (0, 0, 45)},
    {
        "leftUpperLeg": (26, 0, 0),
        "leftLowerLeg": (-20, 0, 0),
        "rightUpperLeg": (-16, 0, 0),
        "rightLowerLeg": (-12, 0, 0),
    },
    {
        "rightUpperLeg": (26, 0, 0),
        "rightLowerLeg": (-20, 0, 0),
        "leftUpperLeg": (-16, 0, 0),
        "leftLowerLeg": (-12, 0, 0),
    },
    {"spine": (20, 0, 0), "chest": (10, 0, 0)},
)


# Where the shoulder cap's reach ahead of a set-back joint fades out, as shares of the torso, and the setback
# at which that is all in force.
SETBACK_CAP_AHEAD = 0.05
SETBACK_CAP_FADE = 0.05
SETBACK_CAP_FULL = 0.06


def spine_table(L):
    T, at = L["T"], L["at"]
    return [
        ("hips", at(HIPS_LEVEL), 0.004 * T),
        ("spine", at(0.36), 0.012 * T),
        ("chest", at(0.56), 0.0),
        ("upperChest", at(0.76), 0.0),
        ("neck", at(NECK_LEVEL), 0.03 * T),
        ("head", L["chin"] + 0.10 * L["head"], 0.0),
    ]


def template_hand(base, verts):
    """The similarity that carries the template's left hand to where the wrap placed it.

    The wrap scales and turns the template hand whole, then corrects only the part near the wrist,
    so the fingers' outer phalanges moved by exactly this transform.
    """
    segments = np.asarray(base["segments"])
    raw = np.asarray(base["verts"], float)
    outer = np.isin(segments, PLACED_WHOLE) & (raw[:, 0] > 0)
    source, target = raw[outer], verts[outer]
    source_mean, target_mean = source.mean(0), target.mean(0)
    u, s, vt = np.linalg.svd((target - target_mean).T @ (source - source_mean))
    flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    turn = u @ flip @ vt
    scale = np.trace(np.diag(s) @ flip) / np.sum((source - source_mean) ** 2)
    return lambda points: (points - source_mean) @ turn.T * scale + target_mean


def skeleton(base, verts, L):
    """A Skeleton of bones as dicts of name, head, tail, parent and the vector their local Z leans toward."""
    T = L["T"]
    regions = np.asarray(base["segments"])
    # A side belongs to the template's topology; a deformed vertex may sit close to the midline.
    left = np.asarray(base["verts"])[:, 0] > 0
    placed_hand = template_hand(base, verts)
    template = np.asarray(base["verts"], float)

    def ring(name):
        if name in FINE and name != "toes":
            seat = template[base["rings"][name]].mean(axis=0) + THUMB_SEAT.get(name, 0.0)
            return placed_hand(seat)
        return verts[base["rings"][name]].mean(axis=0)

    def tip(segment, start):
        points = verts[(regions == segment) & left]
        return points[np.argmax(np.linalg.norm(points - start, axis=1))]

    across = ring("littleProximal") - ring("indexProximal")
    along = ring("middleProximal") - ring("wrist")
    palm = np.cross(along, across)
    palm = -palm / np.linalg.norm(palm)

    bones = [
        {
            "name": "root",
            "head": np.zeros(3),
            "tail": np.array([0.0, 0.0, 100.0]),
            "parent": None,
            "align": np.array(FRONT),
        }
    ]
    previous = "root"
    table = spine_table(L)
    for i, (name, z, y) in enumerate(table):
        head = np.array([0.0, y, z])
        if i + 1 < len(table):
            tail = np.array([0.0, table[i + 1][2], table[i + 1][1]])
        else:
            tail = head + np.array([0.0, 0.0, 0.6 * L["head"]])
        bones.append(
            {"name": name, "head": head, "tail": tail, "parent": previous, "align": np.array(FRONT)}
        )
        previous = name

    limbs = []

    def limb(name, head, tail, parent, align=FRONT):
        limbs.append((name, np.asarray(head, float), np.asarray(tail, float), parent, align))

    shoulder = L["shoulderJoint"]
    limb("Shoulder", (0.06 * T, 0.0, L["at"](0.99)), shoulder, "upperChest")
    limb("UpperArm", shoulder, ring("elbow"), "Shoulder")
    limb("LowerArm", ring("elbow"), ring("wrist"), "UpperArm")
    limb("Hand", ring("wrist"), ring("middleProximal"), "LowerArm", tuple(palm))
    # Roll the ulnar palm around the longitudinal middle ray. The little knuckle
    # travels farther than the ring knuckle because it is farther from this axis.
    limb("PalmCup", ring("wrist") + 0.28 * along, ring("middleProximal"), "Hand", tuple(palm))
    for finger, joints in FINGERS.items():
        parent = "PalmCup" if finger in ("ring", "little") else "Hand"
        for i, joint in enumerate(joints):
            key = f"{finger}{joint}"
            head = ring(key)
            following = f"{finger}{joints[i + 1]}" if i + 1 < len(joints) else None
            tail = ring(following) if following else tip(key, head)
            bone = f"{capital(finger)}{joint}"
            limb(bone, head, tail, parent, tuple(palm))
            parent = bone
    # The hip's X comes from the skin, as the knee and ankle do: right above the knee ring the thigh
    # hangs plumb from the front, so a raised leg swings straight ahead and the shin follows it.
    # Slanted, by a hip further out or in, it carries the knee and ankle towards the midline.
    hip = np.array(L["hipJoint"], float)
    hip[0] = ring("knee")[0]
    limb("UpperLeg", hip, ring("knee"), "hips")
    limb("LowerLeg", ring("knee"), ring("ankle"), "UpperLeg")
    limb("Foot", ring("ankle"), ring("toes"), "LowerLeg", UP)
    limb("Toes", ring("toes"), tip("toes", ring("toes")), "Foot", UP)
    placed = {row[0]: row for row in limbs}
    for name, parent in HALF_BONES.items():
        _, start, end, _, align = placed[name]
        limb(f"{name}Half", start, start + (end - start) * 0.2, parent, align)
    for name, spec in TWISTS.items():
        _, start, end, _, align = placed[name]
        for i, (at, _share) in enumerate(spec["bones"], 1):
            head = start + (end - start) * at
            limb(f"{name}Twist{i}", head, head + (end - start) * 0.1, name, align)

    for side, flip in (("left", 1.0), ("right", -1.0)):
        scale = np.array((flip, 1.0, 1.0))
        for name, head, tail, parent, align in limbs:
            bones.append(
                {
                    "name": f"{side}{name}",
                    "head": head * scale,
                    "tail": tail * scale,
                    "parent": parent if parent[0].islower() else f"{side}{parent}",
                    "align": np.asarray(align, float) * scale,
                }
            )
    head, glide = head_bones(base, verts, L)
    bones += head
    heads = {bone["name"]: bone["head"] for bone in bones}
    # T as the spine table placed the bones, which is L["T"] to within a rounding step.
    placed = (heads["neck"][2] - heads["hips"][2]) / (NECK_LEVEL - HIPS_LEVEL)
    return Skeleton(bones, T=placed, glides={"jaw": glide})


def head_bones(base, verts, L):
    """The jaw and the eyes, last in the skeleton so that every other bone keeps its column, and how
    far the jaw slides as it opens (Skeleton.glides).

    The jaw's local X is the wearer's left, as the head's, so opening is positive as nodding is; it
    runs from the joints to the chin. An eye's local Y is its gaze, its Z up: X+ looks up and Z+ to
    the wearer's left. Its tail is straight ahead, as far as the eyeball's radius.
    """
    h, H = L["head"], L["H"]
    head = verts[np.asarray(base["segments"]) == "head"]
    ring = verts[base["openings"]["neck"]]
    # The head's mesh starts on the neck, so the chin is the lowest skin well ahead of the ring.
    ahead = head[head[:, 1] < ring[:, 1].min() - 0.08 * h]
    chin = ahead[np.argmin(ahead[:, 2])]
    joint = np.array([0.0, TMJ_Y * h, H - TMJ_T * h])
    tail = np.array([0.0, chin[1], chin[2]])
    length = float(np.linalg.norm(tail - joint))
    jaw = {"name": "jaw", "head": joint, "tail": tail, "parent": "head", "align": np.array((0.0, 0.0, -1.0))}
    slope = np.radians(JAW_GLIDE_SLOPE)
    along = np.array((0.0, -np.cos(slope), -np.sin(slope)))
    glide = (rest_matrix(jaw)[:3, :3].T @ along * JAW_GLIDE * length).tolist()
    bones = [jaw]

    level = H - EYE_AT[1] * h
    spread = np.abs(head[:, 0]).max() / (REFERENCE_HALF_WIDTH * h)
    x, radius = EYE_AT[0] * spread * h, EYE_RADIUS * h
    half_width = EYE_PATCH[0] * spread * h + radius * np.sin(np.radians(EYE_LOOK))
    half_height = EYE_PATCH[1] * h
    near = head[
        (head[:, 1] < 0) & (((head[:, 0] - x) / half_width) ** 2 + ((head[:, 2] - level) / half_height) ** 2 <= 1)
    ]
    # The skin's slant over the patch: the face turns away from the midline, so the socket's middle
    # is nearer the midline than the iris, as a person's is. y = slope_x (x - x0) + slope_z (z - z0) + c.
    plane = np.column_stack((near[:, 0] - x, near[:, 2] - level, np.ones(len(near))))
    slope_x, slope_z, _ = np.linalg.lstsq(plane, near[:, 1], rcond=None)[0]
    tilt = radius / np.sqrt(1 + slope_x**2 + slope_z**2)
    at = np.array((x - slope_x * tilt, level - slope_z * tilt))
    # The sphere's front at each skin point of the patch stays EYE_CLEAR_MM ahead of the skin.
    reach = np.sum((near[:, [0, 2]] - at) ** 2, axis=1)
    depth = float(np.min(near[:, 1] + np.sqrt(radius**2 - reach))) - EYE_CLEAR_MM
    for side, flip in (("left", 1.0), ("right", -1.0)):
        centre = np.array([flip * at[0], depth, at[1]])
        bones.append(
            {
                "name": f"{side}Eye",
                "head": centre,
                "tail": centre + np.array(FRONT) * radius,
                "parent": "head",
                "align": np.array(UP),
            }
        )
    return bones, glide


def diffuse(weights, edges, rate, iterations):
    """Spread weights along the mesh only, so nothing jumps between surfaces that merely touch."""
    a, b = edges[:, 0], edges[:, 1]
    ends = np.concatenate((a, b))
    neighbours = np.concatenate((b, a))
    size = len(weights)
    degree = np.bincount(ends, minlength=size)[:, None]
    rate = rate[:, None]
    for _ in range(iterations):
        # One index array for both directions: adding in this order keeps every sum's rounding.
        incoming = weights[neighbours].T
        total = np.column_stack([np.bincount(ends, column, size) for column in incoming])
        weights = weights + (total / degree - weights) * rate
    return weights


def smooth_over_length(weights, verts, edges, zone, reach):
    """Smooth the weights inside `zone` over `reach` millimetres of skin, whatever the rows' spacing.

    Solves (I + reach² Δ) w = w₀ there, Δ the mesh Laplacian with each edge weighted by the inverse
    square of its length, the skin around the zone held as it is. Counting edges, as `diffuse`
    does, blends over fewer millimetres where the rows crowd together.
    """
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu

    free = np.flatnonzero(zone)
    if not len(free):
        return weights
    index = np.full(len(verts), -1)
    index[free] = np.arange(len(free))
    length = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
    stiffness = (reach / np.maximum(length, 1e-3)) ** 2
    ends = np.concatenate((edges[:, 0], edges[:, 1]))
    others = np.concatenate((edges[:, 1], edges[:, 0]))
    stiffness = np.concatenate((stiffness, stiffness))
    inside = index[ends] >= 0
    ends, others, stiffness = ends[inside], others[inside], stiffness[inside]
    row = index[ends]
    diagonal = 1.0 + np.bincount(row, stiffness, len(free))
    coupled = index[others] >= 0
    matrix = csc_matrix(
        (
            np.concatenate((diagonal, -stiffness[coupled])),
            (
                np.concatenate((np.arange(len(free)), row[coupled])),
                np.concatenate((np.arange(len(free)), index[others[coupled]])),
            ),
        ),
        shape=(len(free), len(free)),
    )
    held = ~coupled
    target = weights[free].copy()
    np.add.at(target, row[held], stiffness[held, None] * weights[others[held]])
    used = np.flatnonzero(target.any(axis=0))
    out = weights.copy()
    out[np.ix_(free, used)] = splu(matrix).solve(target[:, used])
    out = np.maximum(out, 0.0)
    return out / out.sum(axis=1, keepdims=True)


def limit(weights):
    """Keep the strongest influences without a jump: an influence reaches zero as it leaves the top.

    Dropping all but the strongest outright makes neighbouring vertices keep different bones, which
    tears a joint under load. Subtracting the first dropped weight is continuous across that change.
    """
    ordered = np.sort(weights, axis=1)
    dropped = ordered[:, -MAX_INFLUENCES - 1][:, None]
    weights = np.maximum(weights - dropped, 0.0)
    weights = weights / weights.sum(axis=1, keepdims=True)
    weights[weights < MIN_WEIGHT] = 0.0
    return weights / weights.sum(axis=1, keepdims=True)


def setback_cap_share(verts, L):
    """How much of the shoulder's cap rides with the arm, once the shoulder stands back.

    The cap is read from side to side and by height, so a shoulder set back leaves the skin ahead of it
    on the arm, and a raised arm then folds the top of the chest. Skin further ahead of the joint than a
    body without a setback has any stays with the girdle; 1 everywhere without a setback.
    """
    setback = L["shoulderSetback"]
    if not setback:
        return 1.0
    T = L["T"]
    ahead = L["shoulderJoint"][1] - verts[:, 1]
    return 1 - min(setback / SETBACK_CAP_FULL, 1.0) * ramp((ahead - SETBACK_CAP_AHEAD * T) / (SETBACK_CAP_FADE * T))


def skin_weights(
    base,
    bones,
    verts,
    L,
    shoulder_spread=30,
    hip_spread=8,
    cap_z=(0.92, 1.04),
    neck_spread=20,
    fit_lower=True,
):
    """Per-vertex weights: template parts seed the limbs, height seeds the trunk, the mesh diffuses.

    Then the skin below the waist is refitted (see fit_lower_body). `fit_lower=False` gives the
    weights before that, only to compare the two: everything that plays the body uses the fit.

    Diffusion counts edges, so loops added at a joint would narrow its blend. The weights are found
    without them and carried to them (sotaigen.template.prolonged), which keeps every blend's length.
    """
    if base.get("refined"):
        small, keep = coarse(base)
        options = (shoulder_spread, hip_spread, cap_z, neck_spread, fit_lower)
        weights = prolonged(base, skin_weights(small, bones, verts[keep], L, *options), keep)
        added = np.setdiff1d(np.arange(len(verts)), keep)
        weights[added] = limit(weights[added])
        return weights
    T, at = L["T"], L["at"]
    half_width = L["shoulderHalf"]
    names = [bone["name"] for bone in bones]
    column = {name: i for i, name in enumerate(names)}
    segments = np.asarray(base["segments"])
    weights = np.zeros((len(verts), len(names)))
    x, z = verts[:, 0], verts[:, 2]

    trunk = np.isin(segments, ("trunk", "neck"))
    table = spine_table(L)
    remaining = np.ones(len(verts))
    for i, (name, _, _) in enumerate(table):
        if i + 1 < len(table):
            blend = 0.05 * T if table[i + 1][0] == "head" else 0.08 * T
            rising = ramp((z - table[i + 1][1] + blend) / (2 * blend))
            weights[trunk, column[name]] = (remaining * (1 - rising))[trunk]
            remaining = remaining * rising
        else:
            weights[trunk, column[name]] = remaining[trunk]

    # The girdle comes in at the trunk's upper corners; the very cap rides with the arm.
    girdle = ramp((np.abs(x) - 0.42 * half_width) / (0.38 * half_width))
    girdle = girdle * ramp((z - at(0.78)) / (0.135 * T))
    girdle = np.where(trunk, girdle, 0.0)
    cap = ramp((np.abs(x) - 0.80 * half_width) / (0.34 * half_width))
    cap = cap * ramp((z - at(cap_z[0])) / ((cap_z[1] - cap_z[0]) * T))
    cap = np.where(trunk, cap, 0.0)
    cap = cap * setback_cap_share(verts, L)
    weights *= ((1 - girdle) * (1 - cap))[:, None]
    side = np.where(np.asarray(base["verts"])[:, 0] >= 0, "left", "right")
    for i in np.flatnonzero((girdle > 0) | (cap > 0)):
        weights[i, column[f"{side[i]}Shoulder"]] += girdle[i] * (1 - cap[i])
        weights[i, column[f"{side[i]}UpperArm"]] += cap[i]
    head = segments == "head"
    weights[head, column["head"]] = 1.0
    for i in np.flatnonzero(~trunk & ~head):
        weights[i, column[f"{side[i]}{capital(segments[i])}"]] = 1.0

    for side_name in ("left", "right"):
        hand = next(b for b in bones if b["name"] == f"{side_name}Hand")
        little = next(b for b in bones if b["name"] == f"{side_name}LittleProximal")
        axes = rest_matrix(hand)[:3, :3]
        local = (verts - hand["head"]) @ axes
        span = (little["head"] - hand["head"]) @ axes[:, 0]
        length = np.linalg.norm(hand["tail"] - hand["head"])
        share = ramp(local[:, 0] / span) * ramp((local[:, 1] / length - 0.18) / 0.58)
        hand_col = column[hand["name"]]
        transfer = weights[:, hand_col] * share
        weights[:, hand_col] -= transfer
        weights[:, column[f"{side_name}PalmCup"]] = transfer

    edges = edges_of(base["faces"])
    rate = np.where(np.isin(segments, FINE), 0.25, 0.5)
    weights = diffuse(weights, edges, rate, 6)
    regions = np.asarray(base["regions"])
    free = ~np.isin(regions, ("hand", "foot"))
    shoulder_zone = ramp((z - at(0.745)) / (0.08 * T)) * ramp((at(1.01) - z) / (0.08 * T))
    hip_zone = ramp((z - at(-0.05)) / (0.06 * T)) * ramp((at(0.22) - z) / (0.06 * T))
    weights = diffuse(weights, edges, rate * shoulder_zone * free, shoulder_spread)
    weights = diffuse(weights, edges, rate * hip_zone * free, hip_spread)
    # The neck meets a head that moves as one piece, so the turn is spread over the neck's length.
    neck_zone = np.isin(segments, ("neck",)) | (head & (z < L["chin"] + 0.45 * L["head"]))
    weights = diffuse(weights, edges, rate * neck_zone, neck_spread)
    face = head & ((L["H"] - z) / L["head"] < NECK_FACE)
    weights = smooth_over_length(weights, verts, edges, neck_zone & ~face, NECK_SMOOTH * L["head"])
    share = jaw_share(verts, L) * head
    # Only how the head's weight is split: a turn or a nod of the head moves the skin nearly as without it.
    below = np.column_stack((share, 1 - share))
    lower = (L["H"] - z) / L["head"] > JAW_SMOOTH_BELOW
    share = smooth_over_length(below, verts, edges, head & lower, JAW_SMOOTH * L["head"])[:, 0]
    weights[:, column["jaw"]] = weights[:, column["head"]] * share
    weights[:, column["head"]] *= 1 - share
    # Rows of skin lie closer at the wrist and ankle than at the elbow and knee, so the same number
    # of edges would blend them over less length, and they'd bend at a hard line.
    ends = np.zeros(len(verts))
    for joint, start in (("ankle", "knee"), ("wrist", "elbow")):
        centre = verts[base["rings"][joint]].mean(axis=0)
        radius = END_BLEND * np.linalg.norm(centre - verts[base["rings"][start]].mean(axis=0))
        for flip in (1.0, -1.0):
            share = np.linalg.norm(verts - centre * (flip, 1.0, 1.0), axis=1) / radius
            ends = np.maximum(ends, ramp((1.0 + END_EDGE - share) / END_EDGE))
    weights = diffuse(weights, edges, 0.5 * ends, END_SPREAD)
    weights = round_wrist_back(base, bones, verts, weights, edges, ends > 0)
    weights = limit(weights)
    weights = limit(split_half(weights, column))
    weights = limit(split_twist(weights, column, bones, verts))
    weights, tip_zone = seat_elbow_tip(base, bones, verts, weights, column)
    weights = limit(diffuse(limit(weights), edges, 0.5 * (tip_zone > 0), TIP_SMOOTH))
    return fit_lower_body(base, bones, verts, L, weights) if fit_lower else weights


def jaw_share(verts, L):
    """How much of the head's weight each point gives to the jaw: see MOUTH_T and the rest."""
    h = L["head"]
    t = (L["H"] - verts[:, 2]) / h
    # How far from the joint to the mouth, in side view.
    along = np.clip((TMJ_Y - verts[:, 1] / h) / MOUTH_Y, 0.0, 1.0)
    line = TMJ_T + (MOUTH_T - TMJ_T) * along
    blend = JAW_BLEND_BACK + (JAW_BLEND_MOUTH - JAW_BLEND_BACK) * along**3
    below = ramp((t - line + blend / 2) / blend)
    ahead = ramp((JAW_EAR_Y - verts[:, 1] / h) / JAW_BEHIND)
    seat = ramp(-_SeatPlane(L)(verts) / (JAW_SEAT * h))
    return below * ahead * seat


def round_wrist_back(base, bones, verts, weights, edges, band):
    """Spread the back of the wrist's blend evenly over its length, so that flexing bends it in an arc.

    Counting edges leaves steps in the weights where the rows crowd together at the hand's seam, and
    the outline turns at those rows. The palm keeps them: its one crease is meant to stay there.
    `band` is the skin the wrist and ankle blends reach.
    """
    segments = np.asarray(base["segments"])
    by_name = {bone["name"]: bone for bone in bones}
    template_x = np.asarray(base["verts"])[:, 0]
    effect = np.zeros(len(verts))
    length = 0.0
    for side, on_side in (("left", template_x >= 0), ("right", template_x < 0)):
        hand, lower = by_name[f"{side}Hand"], by_name[f"{side}LowerArm"]
        length = float(np.linalg.norm(hand["tail"] - hand["head"]))
        palm = rest_matrix(hand)[:3, 2]
        for segment, bone in (("hand", hand), ("lowerArm", lower)):
            members = (segments == segment) & on_side
            along = (bone["tail"] - bone["head"]) / np.linalg.norm(bone["tail"] - bone["head"])
            offset = verts[members] - hand["head"]
            around = offset - np.outer(offset @ along, along)
            around /= np.maximum(np.linalg.norm(around, axis=1, keepdims=True), 1e-9)
            front = palm - along * (palm @ along)
            front = front / np.linalg.norm(front)
            azimuth = np.degrees(np.arccos(np.clip(around @ front, -1, 1)))
            effect[members] = ramp((azimuth - WRIST_BACK[0]) / (WRIST_BACK[1] - WRIST_BACK[0]))
    zone = band & np.isin(segments, ("hand", "lowerArm"))
    smooth = smooth_over_length(weights, verts, edges, zone, WRIST_BACK_REACH * length)
    return weights + (smooth - weights) * effect[:, None]


def split_half(weights, column):
    """Move the meeting of limb and parent through the half bone, so skinning follows an arc."""
    out = weights.copy()
    for side in ("left", "right"):
        for name, parent_name in HALF_BONES.items():
            half = column.get(f"{side}{name}Half")
            if half is None:
                continue
            parent = column[parent_name if parent_name[0].islower() else f"{side}{parent_name}"]
            limb = column[f"{side}{name}"]
            a, b = out[:, parent], out[:, limb]
            total = a + b
            shared = (a > 0) & (b > 0)
            mix = np.divide(b, total, out=np.zeros_like(b), where=total > 0)
            # Mix 0 stays on the parent, 1 on the limb, 0.5 entirely on the half bone.
            out[shared, parent] = (total * np.clip(1 - 2 * mix, 0, 1))[shared]
            out[shared, limb] = (total * np.clip(2 * mix - 1, 0, 1))[shared]
            out[shared, half] = (total * (1 - np.abs(2 * mix - 1)))[shared]
    return out


def split_twist(weights, column, bones, verts):
    """Share each limb's weight between its twist bones by how far along the limb a vertex lies."""
    out = weights.copy()
    by_name = {bone["name"]: bone for bone in bones}
    for side in ("left", "right"):
        for name, spec in TWISTS.items():
            if f"{side}{name}Twist1" not in column:
                continue
            bone = by_name[f"{side}{name}"]
            axis = bone["tail"] - bone["head"]
            t = (verts - bone["head"]) @ axis / (axis @ axis)
            limb = column[f"{side}{name}"]
            stations = [
                (at, column[f"{side}{name}Twist{i}"])
                for i, (at, _share) in enumerate(spec["bones"], 1)
            ]
            if spec["fullAt"] is not None:
                # The limb itself takes over at the far end; before the first station, that bone.
                stations = [(0.0, stations[0][1]), *stations, (spec["fullAt"], limb)]
            else:
                # The limb holds the near end; the last twist bone holds the far end.
                stations = [(0.0, limb), *stations]
            at = np.array([s[0] for s in stations])
            total = out[:, limb].copy()
            out[:, limb] = 0.0
            for i, (_position, target) in enumerate(stations):
                # Each station's tent function over t; the ends stay flat beyond the last station.
                tent = np.zeros(len(stations))
                tent[i] = 1.0
                out[:, target] += total * np.interp(t, at, tent)
    return out


def seat_elbow_tip(base, bones, verts, weights, column):
    """Hand the back of the elbow from the half bone to the two limbs, so the tip stays out.

    Every vertex keeps its distance from the pivot under a rotation about it, so a half bone's arc
    can't reach the corner where the two limbs' back lines meet. The skin riding each limb can: a
    point above the joint that rides the forearm comes down onto the forearm's back line, and the
    upper arm's back line runs straight to it. What rides which limb is decided at rest, so the rest
    pose doesn't move. Returns the weights and how strongly each vertex was handed over.
    """
    out = weights.copy()
    zone = np.zeros(len(verts))
    by_name = {bone["name"]: bone for bone in bones}
    segments = np.asarray(base["segments"])
    template_x = np.asarray(base["verts"])[:, 0]
    ring = verts[base["rings"]["elbow"]]
    for side, on_side in (("left", template_x >= 0), ("right", template_x < 0)):
        lower = by_name[f"{side}LowerArm"]
        upper = by_name[f"{side}UpperArm"]
        centre = np.asarray(lower["head"], float)
        frontal = rest_matrix(lower)[:3, 2]
        down = (lower["tail"] - lower["head"]) / np.linalg.norm(lower["tail"] - lower["head"])
        up = (upper["head"] - lower["head"]) / np.linalg.norm(upper["head"] - lower["head"])
        side_ring = ring * (1.0 if side == "left" else -1.0, 1.0, 1.0)
        radial = side_ring - side_ring.mean(axis=0)
        radius = float(np.linalg.norm(radial - np.outer(radial @ down, down), axis=1).mean())
        axial = np.full(len(verts), np.inf)
        azimuth_cos = np.zeros(len(verts))
        for segment, along, sign in (("upperArm", up, -1.0), ("lowerArm", down, 1.0)):
            members = (segments == segment) & on_side
            offset = verts[members] - centre
            axial[members] = sign * (offset @ along)
            around = offset - np.outer(offset @ along, along)
            front = frontal - along * (frontal @ along)
            front = front / np.linalg.norm(front)
            around /= np.maximum(np.linalg.norm(around, axis=1, keepdims=True), 1e-9)
            azimuth_cos[members] = around @ front
        azimuth = np.degrees(np.arccos(np.clip(azimuth_cos, -1, 1)))
        back = ramp((azimuth - TIP_BACK[0]) / (TIP_BACK[1] - TIP_BACK[0]))
        reach = (TIP_FADE[1] * radius - np.abs(axial)) / ((TIP_FADE[1] - TIP_FADE[0]) * radius)
        effect = back * ramp(reach)
        moved = np.flatnonzero(effect > 0)
        if not len(moved):
            continue
        boundary = -TIP_RISE * radius * back[moved]
        band = TIP_BAND * radius
        forearm = ramp((axial[moved] - boundary + band / 2) / band)
        upper_family = [column[f"{side}UpperArm"], *twist_columns(column, f"{side}UpperArm")]
        lower_family = [column[f"{side}LowerArm"], *twist_columns(column, f"{side}LowerArm")]
        current = out[moved]
        target = np.zeros_like(current)
        for family, share in ((upper_family, 1 - forearm), (lower_family, forearm)):
            held = current[:, family]
            total = held.sum(axis=1, keepdims=True)
            parts = np.divide(held, total, out=np.zeros_like(held), where=total > 0)
            parts[total[:, 0] <= 0, 0] = 1.0
            target[:, family] = parts * share[:, None]
        out[moved] = current + (target - current) * effect[moved, None]
        zone[moved] = effect[moved]
    return out, zone


def twist_columns(column, name):
    return [column[f"{name}Twist{i}"] for i in (1, 2) if f"{name}Twist{i}" in column]


def lower_zone(base, verts, L):
    """How far the fit applies to each vertex: 1 at the groin, fading to 0 at the waist and shin.

    Only skin of the trunk and legs: a hand hanging at the hip's height is not lower body. Height
    decides the rest, over a smooth band, so that no edge of the fit shows on the belly.
    """
    segments = np.asarray(base["segments"])
    at, waist = L["at"], L["waist"]
    knee = verts[base["rings"]["knee"]].mean(0)[2]
    ankle = verts[base["rings"]["ankle"]].mean(0)[2]
    z = verts[:, 2]
    start = at(FIT_FULL_BELOW)
    fade_top = 1 - ramp((z - start) / (waist - start))
    # Below the knee nothing folds, and the ankle has its own blend.
    gone = knee + FIT_SHIN_OUT * (ankle - knee)
    fade_shin = ramp((z - gone) / (FIT_SHIN_OUT * (knee - ankle) / 2))
    return np.where(np.isin(segments, FIT_PARTS), fade_top * fade_shin, 0.0)


def project_simplex_batch(w, allowed):
    """project_simplex for many vertices at once, each over the bones `allowed` to it."""
    count = w.shape[1]
    ordered = -np.sort(-np.where(allowed, w, -np.inf), axis=1)
    finite = np.isfinite(ordered)
    running = np.cumsum(np.where(finite, ordered, 0.0), axis=1) - 1
    positive = finite & (ordered - running / np.arange(1, count + 1) > 0)
    last = count - 1 - np.argmax(positive[:, ::-1], axis=1)
    shift = running[np.arange(len(w)), last] / (last + 1)
    return np.maximum(w - shift[:, None], 0.0) * allowed


def spread_groin_weights(verts, L, weights, zone, edges):
    """Share hip motion across the front fold and the saddle joining the inner thighs."""
    x, y, z = verts.T / L["T"]
    height = z - L["crotch"] / L["T"]
    band = ramp((height + 0.34) / 0.20) * ramp((0.40 - height) / 0.20)
    front = ramp((0.12 - y) / 0.18)
    saddle = np.exp(-(x / 0.14) ** 2)
    blend = band * np.maximum(front, saddle) * zone
    # Include one fixed ring around the moving patch. Diffusion then has the same
    # neighbours as on the full mesh and cannot jump across the gap between legs.
    local_edges = edges[(blend[edges] > 0).any(axis=1)]
    vertices, links = np.unique(local_edges, return_inverse=True)
    if not len(vertices):
        return
    local = weights[vertices]
    columns = np.any(local > 0, axis=0)
    rate = blend[vertices]
    spread = diffuse(local[:, columns], links.reshape(-1, 2), 0.65 * rate, 105)
    local[:, columns] += rate[:, None] * (spread - local[:, columns])
    changed = rate > 0
    weights[vertices[changed]] = limit(local[changed])


def fit_lower_body(base, bones, verts, L, weights, rounds=200, keep_home=0.02):
    """Weights refitted below the waist, for every vertex at once; everything else is untouched.

    Linear skinning folds the groin at a hard line. Delta mush of the same poses doesn't, so each
    vertex's weights are solved to come as close to it as they can (the weight step of SSDR; Le and
    Deng, 2012), which a standard runtime plays with nothing added. Only trunk and leg skin, only
    over trunk and leg bones, and on poses of the legs and spine. A vertex may use the bones it or
    its neighbours do, on its own side unless it already has weight on the other. Where the fit
    fades out it is blended with the weights the vertex had, then cut to four without a jump. A
    vertex outside the zone keeps its weights exactly: the result is checked.
    """
    names = [bone["name"] for bone in bones]
    zone = lower_zone(base, verts, L)
    fitted = np.flatnonzero(zone > 0)

    usable = np.isin([name.removeprefix("left").removeprefix("right") for name in names], FIT_BONES)
    source, sink, _ = adjacency(len(verts), base["faces"])
    reach = weights > 0
    for _ in range(2):
        grown = np.zeros_like(reach)
        np.logical_or.at(grown, source, reach[sink])
        reach = reach | grown
    is_left = np.asarray(base["verts"])[:, 0] >= 0
    far_side = np.array(["right" if side else "left" for side in is_left])
    for column, name in enumerate(names):
        if name.startswith(("left", "right")):
            other = far_side == ("left" if name.startswith("left") else "right")
            reach[other & (weights[:, column] == 0), column] = False
    reach = reach & usable

    # Candidates as a padded table: vertices x slots, with a mask of the slots in use.
    slots = int(reach[fitted].sum(1).max())
    candidates = np.zeros((len(fitted), slots), int)
    allowed = np.zeros((len(fitted), slots), bool)
    for row, vertex in enumerate(fitted):
        found = np.flatnonzero(reach[vertex])
        candidates[row, : len(found)] = found
        allowed[row, : len(found)] = True

    mush = Mush(verts, base["faces"], mush_zone(base))
    home = np.column_stack((verts[fitted], np.ones(len(fitted))))
    moves, goals = [], []
    for pose in FIT_EXAMPLES:
        target = mush(pose_verts(bones, weights, verts, pose))[fitted]
        skins = pose_skins(bones, pose)[candidates]  # vertices x slots x 4 x 4
        moves.append(np.einsum("nkij,nj->nki", skins, home)[:, :, :3])
        goals.append(target)
    # Vertices x slots x equations; and vertices x equations.
    moves = np.concatenate(moves, axis=2)
    goals = np.concatenate(goals, axis=1)
    moves = moves * allowed[:, :, None]
    # With weights summing to one, the part bones agree on changes nothing but the conditioning.
    shared = moves.sum(1) / allowed.sum(1, keepdims=True)
    moves = (moves - shared[:, None, :]) * allowed[:, :, None]
    goals = goals - shared

    home_weights = np.take_along_axis(weights[fitted], candidates, axis=1) * allowed
    normal = moves @ moves.transpose(0, 2, 1) + keep_home * np.eye(slots)
    right = np.einsum("nke,ne->nk", moves, goals) + keep_home * home_weights
    step = 1.0 / np.linalg.norm(normal, 2, axis=(1, 2))
    w = previous = home_weights
    for i in range(rounds):
        ahead = w + (w - previous) * i / (i + 3)
        gradient = np.einsum("nkl,nl->nk", normal, ahead) - right
        previous, w = w, project_simplex_batch(ahead - step[:, None] * gradient, allowed)

    out = weights.copy()
    solved = np.zeros((len(fitted), weights.shape[1]))
    np.add.at(solved, (np.arange(len(fitted))[:, None], candidates), w * allowed)
    out[fitted] = solved
    edges = edges_of(base["faces"])
    out = diffuse(out, edges, np.where(zone > 0, 0.5, 0.0), 2)
    blended = weights + (out - weights) * zone[:, None]
    blended[fitted] = mirrored(blended, verts, names)[fitted]
    result = weights.copy()
    result[fitted] = limit(blended[fitted])
    spread_groin_weights(verts, L, result, zone, edges)
    check_lower_body(result, weights, zone)
    return result


def mirrored(weights, verts, names):
    """The left side's weights on both sides, and the midline's averaged with its own mirror image.

    The fit itself isn't symmetric: a vertex frame takes its first neighbour, whose order doesn't
    mirror, and a midline vertex counted as left could add no right bone. Up to 1.4% apart, and
    about 2 mm in a crouch (docs/systems/rig.md). A body that doesn't mirror keeps its weights.
    """
    from scipy.spatial import cKDTree

    verts = np.asarray(verts, float)
    distance, opposite = cKDTree(verts).query(verts * (-1.0, 1.0, 1.0))
    if len(np.unique(opposite)) != len(verts) or distance.max() > 1e-5:
        return weights
    swap = {"left": "right", "right": "left"}
    columns = [
        names.index(next((swap[s] + n[len(s):] for s in swap if n.startswith(s)), n)) for n in names
    ]
    image = weights[opposite][:, columns]
    out = weights.copy()
    x = np.asarray(verts)[:, 0]
    right = x < -1e-6
    middle = ~right & (x <= 1e-6)
    out[right] = image[right]
    out[middle] = 0.5 * (weights[middle] + image[middle])
    return out


def check_lower_body(result, weights, zone):
    """What the fit promises, as a failure rather than a surprise."""
    outside = zone <= 0
    if not np.array_equal(result[outside], weights[outside]):
        raise ValueError("the fit changed a vertex outside its zone")
    if not np.allclose(result.sum(1), 1.0) or (result < 0).any():
        raise ValueError("fitted weights do not sum to one")
    if (result > 0).sum(1).max() > MAX_INFLUENCES:
        raise ValueError("fitted weights have too many influences")


def save_rig(out, bones, weights):
    """Write skeleton.json and weights.npz; weights has one column per bone, in the same order."""
    rows = [
        {**bone, **{key: bone[key].tolist() for key in ("head", "tail", "align")}} for bone in bones
    ]
    record = {"T": bones.T, "glides": bones.glides, "groin": bones.groin}
    (out / "skeleton.json").write_text(json.dumps({"bones": rows, **record}, indent=1))
    names = np.array([bone["name"] for bone in bones])
    np.savez_compressed(out / "weights.npz", weights=weights, bones=names)


def load_rig(out):
    """The bones and weights save_rig wrote, exactly as they were."""
    saved = json.loads((out / "skeleton.json").read_text())
    bones = Skeleton(
        ({**row, **{key: np.array(row[key]) for key in ("head", "tail", "align")}} for row in saved["bones"]),
        T=saved["T"],
        glides=saved["glides"],
        groin=saved["groin"],
    )
    with np.load(out / "weights.npz") as saved:
        return bones, saved["weights"]
