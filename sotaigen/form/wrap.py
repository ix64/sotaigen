"""Fit the fixed-topology CC0 template onto a body's blended surface and measure the damage.

Step 1: matching radial landmarks on the template and the field drive a thin-plate warp.
Step 2: vertices step onto the zero set while tangential smoothing keeps the edge flow even.

Run through `uv run sotaigen-dev build`, which knows when a wrap is out of date.
"""

import json
from collections import Counter

import numpy as np

from sotaigen.form.abdomen import AbdominalProfile
from sotaigen.form.anatomy import calibrated, waist_band
from sotaigen.form.axilla import axillary_vertices, landmark_rays
from sotaigen.form.field import peeled, unit
from sotaigen.form.foot import sculpted as sculpted_foot
from sotaigen.form.glutes import GluteContours
from sotaigen.form.head import build_head, extended, head_field, insert_head, source_count
from sotaigen.form.thigh import ThighProfile
from sotaigen.mesh import adjacency, edge_owners, edges_of, vertex_neighbours, write_obj
from sotaigen.params.spec import PROFILES
from sotaigen.template import before_loops

TORSO_FULL = (0.02, 0.15, 0.30, 0.42, 0.55, 0.70)
TORSO_MIDLINE = (0.85, 1.0)
ANGLES = (0, 45, 90, 135, 180)
THIGH_LEVELS = (0.12, 0.35, 0.6, 0.85, 1.0)
SHIN_LEVELS = (0.2, 0.4, 0.6, 0.8, 1.0)
# How far a seam's correction reaches into the hand and the foot, in the template's millimetres.
SEAM_REACH = {"hand": 70.0, "foot": 70.0}
# Rows of the limb, counted from the seam, that the correction takes its value and its slope from.
SEAM_ROWS = 3
# The most a seam may miss by, as a share of the limb's radius there (seam_radius). The miss is the
# gap between two sections centred on each other, so it grows with them and not with the stature:
# only the short-limbed bodies, the smallest ones, come near these. See docs/systems/wrap-uv.md.
SEAM_MISS_SHARE = {"hand": 0.65, "foot": 0.52}
# The most a seam may fold. The template's own folds along these rings are 34 and 9 degrees, and the
# wrap used to reach 45 and 20, which showed as a hard line near the joint. A seam that needs more
# than these has lost its shape again; see docs/systems/wrap-uv.md.
SEAM_CREASE_DEG = {"hand": 35.0, "foot": 16.0}
# How far the sole may miss the floor. wrap.json rounds to 0.1 mm and every body measures 0.0 to
# 0.1, so this leaves room for a few rounding steps. Half a millimetre on a body of 1.5 m can't be
# seen; a foot that floats or sinks visibly would be off by several.
FOOT_TO_GROUND_MM = 0.5
ARM_LEVELS = (0.25, 0.5, 0.75, 1.0)
# How many times the neck's rows are spaced along their columns and stepped back onto the surface.
NECK_PASSES = 3


def direction(angle):
    """0 is the front, 90 the wearer's left, 180 the back, in the horizontal plane."""
    a = np.radians(angle)
    return np.array([np.sin(a), -np.cos(a), 0.0])


def sample(verts, center, axis, ray, slab=8.0):
    """Radial distance of the template surface from an axis point along a ray."""
    offset = verts - center
    along = offset @ axis
    near = np.abs(along) < slab
    while near.sum() < 8:
        slab *= 1.6
        near = np.abs(along) < slab
    radial = offset[near] - np.outer(along[near], axis)
    size = np.linalg.norm(radial, axis=1)
    cosine = (radial @ ray) / np.maximum(size, 1e-6)
    best = np.argsort(-cosine)[:4]
    return center + ray * float(np.mean(size[best] * cosine[best]))


def march_many(body, centers, rays, limit=500.0, step=2.0, chunk=48):
    """First surface crossing of each ray going out from an inside point, refined by bisection.

    A ray that never crosses stays at its centre. Rays are walked in chunks so the ones that have
    already crossed cost nothing further; one field call on many points is what makes this fast.
    """
    centers, rays = np.asarray(centers, float), np.asarray(rays, float)
    ts = np.arange(0.0, limit, step)
    crossing = np.full(len(centers), -1)
    walking = np.arange(len(centers))
    for first in range(0, len(ts) - 1, chunk):
        along = ts[first : first + chunk + 1]
        points = centers[walking, None] + rays[walking, None] * along[None, :, None]
        values = body(points.reshape(-1, 3)).reshape(len(walking), -1)
        hit = (values[:, :-1] <= 0) & (values[:, 1:] > 0)
        found = hit.any(1)
        crossing[walking[found]] = first + hit[found].argmax(1)
        walking = walking[~found]
        if not len(walking):
            break
    out = centers.copy()
    crossed = crossing >= 0
    if crossed.any():
        a, b = ts[crossing[crossed]], ts[crossing[crossed] + 1]
        c, r = centers[crossed], rays[crossed]
        for _ in range(12):
            m = (a + b) / 2
            outside = body(c + m[:, None] * r) > 0
            b = np.where(outside, m, b)
            a = np.where(outside, a, m)
        out[crossed] = c + r * ((a + b) / 2)[:, None]
    return out


def march(body, center, ray):
    """march_many for a single ray."""
    return march_many(body, center[None], ray[None])[0]


def perpendicular(axis):
    """Four rays around a limb axis: front, outer, back, inner."""
    front = np.array([0.0, -1.0, 0.0])
    front = front - axis * (front @ axis)
    front /= np.linalg.norm(front)
    side = np.cross(axis, front)
    if side[0] < 0:
        side = -side
    return [front, side, -front, -side]


def raw_center(verts, axis_mask, z):
    band = verts[axis_mask & (np.abs(verts[:, 2] - z) < 8)]
    return np.array(
        [(band[:, 0].min() + band[:, 0].max()) / 2, (band[:, 1].min() + band[:, 1].max()) / 2, z]
    )


def target_centers(body, starts):
    """Centre of the section around each inside point, from rays along x and y."""
    starts = np.asarray(starts, float)
    n = len(starts)
    axes = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0]])
    hits = march_many(body, np.repeat(starts, 4, axis=0), np.tile(axes, (n, 1))).reshape(n, 4, 3)
    return np.column_stack(
        (
            (hits[:, 0, 0] + hits[:, 1, 0]) / 2,
            (hits[:, 2, 1] + hits[:, 3, 1]) / 2,
            starts[:, 2],
        )
    )


def correspondences(base, body, field, L):
    """Pairs of template and target points; the right side is mirrored afterwards."""
    verts = np.array(base["verts"], float)
    regions = np.array(base["regions"])
    knee = verts[base["rings"]["knee"]].mean(0)
    knee[0] = abs(knee[0])
    elbow = verts[base["rings"]["elbow"]].mean(0)
    elbow[0] = abs(elbow[0])
    pivot = np.array(base["landmarks"]["shoulderPivot"], float)
    trunk = regions == "trunk"
    pit_raw = verts[trunk, 2].max()
    crotch_raw = verts[np.abs(verts[:, 0]) < 3, 2].min()
    raw_T = pit_raw - crotch_raw
    # Each pair's target is a point or, while marching is deferred, a (centre, ray) to march along.
    source, target = [], []

    def add(a, b):
        source.append(a)
        target.append(b)

    # Low levels of the torso are shared with the thighs in the template's own regions.
    lower = (trunk | (regions == "leg")) & (verts[:, 0] > -2)
    levels = TORSO_FULL + TORSO_MIDLINE
    level_z = [L["at"](f) for f in levels]
    starts = np.array([[0.0, 0.0, z] for z in level_z])
    ahead = np.array([0.0, 1.0, 0.0])
    ys = march_many(
        body,
        np.vstack((starts, starts)),
        np.vstack((np.tile(ahead, (len(levels), 1)), np.tile(-ahead, (len(levels), 1)))),
    )[:, 1].reshape(2, -1)
    for i, f in enumerate(levels):
        z_raw, z_target = crotch_raw + f * raw_T, level_z[i]
        left_trunk = verts[lower if f < 0.45 else trunk & (verts[:, 0] > -2)]
        band = left_trunk[np.abs(left_trunk[:, 2] - z_raw) < 10]
        raw_c = np.array([0.0, (band[:, 1].min() + band[:, 1].max()) / 2, z_raw])
        target_c = np.array([0.0, (ys[0, i] + ys[1, i]) / 2, z_target])
        upper = (0, 45, 135, 180) if f < 0.9 else (0, 180)
        for angle in ANGLES if f in TORSO_FULL else upper:
            ray = direction(angle)
            add(sample(left_trunk, raw_c, np.array([0.0, 0.0, 1.0]), ray), (target_c, ray))

    # Neck ring: it sits where the neck meets the head, low under the jaw and high at the nape.
    neck_ring = verts[base["openings"]["neck"]]
    neck_c = neck_ring.mean(0)
    heights = np.arange(L["chin"] - 0.05 * L["head"], L["chin"] + 0.6 * L["head"], 2.0)
    axis_points = target_centers(
        body, np.column_stack((0.0 * heights, np.full(len(heights), 0.01 * L["T"]), heights))
    )
    for angle in ANGLES:
        ray = direction(angle)
        around = march_many(body, axis_points, np.tile(ray, (len(heights), 1)))
        joined = field(around) <= 0
        add(
            sample(neck_ring, neck_c, np.array([0.0, 0.0, 1.0]), ray, slab=40.0),
            around[joined.argmax()] if joined.any() else around[-1],
        )

    # Thigh: fractions between the crotch and the knee.
    legs = (regions == "leg") & (verts[:, 0] > 0)
    hip, knee_j = L["hipJoint"], L["kneeJoint"]
    thigh_z = [L["crotch"] + u * (L["knee"] - L["crotch"]) for u in THIGH_LEVELS]
    guesses = [hip + (knee_j - hip) * ((hip[2] - z) / (hip[2] - knee_j[2])) for z in thigh_z]
    for u, target_c in zip(THIGH_LEVELS, target_centers(body, guesses), strict=True):
        z_raw = crotch_raw + u * (knee[2] - crotch_raw) - 12 * (u == 0)
        raw_c = raw_center(verts, legs, z_raw)
        for ray in perpendicular(np.array([0.0, 0.0, 1.0])):
            add(sample(verts[legs], raw_c, np.array([0.0, 0.0, 1.0]), ray), (target_c, ray))

    # Shin: fractions between the knee and the ankle.
    ankle = verts[base["rings"]["ankle"]].mean(0)
    ankle[0] = abs(ankle[0])
    guesses = [L["kneeJoint"] + (L["ankleJoint"] - L["kneeJoint"]) * u for u in SHIN_LEVELS]
    for u, target_c in zip(SHIN_LEVELS, target_centers(body, guesses), strict=True):
        z_raw = knee[2] + u * (ankle[2] - knee[2])
        raw_c = raw_center(verts, legs, z_raw)
        for ray in perpendicular(np.array([0.0, 0.0, 1.0])):
            add(sample(verts[legs], raw_c, np.array([0.0, 0.0, 1.0]), ray), (target_c, ray))

    # Upper arm and forearm, joint to joint.
    arms = (regions == "arm") & (verts[:, 0] > 0)
    raw_axis = (elbow - pivot) / np.linalg.norm(elbow - pivot)
    raw_len = np.linalg.norm(elbow - pivot)
    stub = np.linalg.norm(L["elbow"] - L["shoulderJoint"])
    wrist = verts[base["rings"]["wrist"]].mean(0)
    wrist[0] = abs(wrist[0])
    raw_fore = wrist - elbow
    for u in ARM_LEVELS:
        raw_c = elbow + raw_fore * u
        target_c = L["elbow"] + (L["wrist"] - L["elbow"]) * u
        pairs = zip(perpendicular(unit(raw_fore)), perpendicular(L["foreDir"]), strict=True)
        for ray_raw, ray_t in pairs:
            add(sample(verts[arms], raw_c, unit(raw_fore), ray_raw), (target_c, ray_t))
    for u in ARM_LEVELS:
        raw_c = pivot + raw_axis * raw_len * u
        target_c = L["shoulderJoint"] + L["armDir"] * stub * u
        for ray_raw, ray_t in zip(perpendicular(raw_axis), perpendicular(L["armDir"]), strict=True):
            add(sample(verts[arms], raw_c, raw_axis, ray_raw), (target_c, ray_t))

    # Root of the arm on its free sides, where it leaves the girdle.
    for side in (0, 1, 2):
        ray_raw, ray_t = perpendicular(raw_axis)[side], perpendicular(L["armDir"])[side]
        add(
            sample(verts[arms], pivot + raw_axis * raw_len * 0.06, raw_axis, ray_raw),
            (L["shoulderJoint"] + L["armDir"] * stub * 0.06, ray_t),
        )

    # Keep the anterior fold, suspended floor and posterior fold distinct. Choosing the
    # lowest arm/trunk seam point alone used to pile all three material rows onto one tip.
    for index, ray in zip(axillary_vertices(base), landmark_rays(L), strict=True):
        add(verts[index], (L["shoulderJoint"], ray))

    # The shoulder cap ties the arm to the girdle.
    arm_top = verts[arms][verts[arms][:, 2].argmax()]
    add(
        arm_top,
        L["acromion"]
        + np.array(
            [0.0, 0.0, march(body, L["acromion"], np.array([0.0, 0.0, 1.0]))[2] - L["acromion"][2]]
        ),
    )

    deferred = [i for i, t in enumerate(target) if isinstance(t, tuple)]
    marched = march_many(body, [target[i][0] for i in deferred], [target[i][1] for i in deferred])
    for i, point in zip(deferred, marched, strict=True):
        target[i] = point
    source, target = np.array(source), np.array(target)
    mirrored = source[:, 0] > 1.0
    flip = np.array([-1.0, 1.0, 1.0])
    return (
        np.vstack((source, source[mirrored] * flip)),
        np.vstack((target, target[mirrored] * flip)),
        {
            "pit": pit_raw,
            "crotch": crotch_raw,
            "knee": knee[2],
            "elbow": elbow,
            "wrist": wrist,
            "ankle": ankle,
            "T": raw_T,
        },
    )


def thin_plate(source, target, points, smoothing=0.002, scale=1000.0):
    """3D biharmonic warp with kernel r and an affine part, in metres for conditioning."""
    source_m, points_m = source / scale, points / scale
    n = len(source)
    distance = np.linalg.norm(source_m[:, None] - source_m[None], axis=2)
    system = np.zeros((n + 4, n + 4))
    system[:n, :n] = distance + smoothing * np.eye(n)
    system[:n, n] = system[n, :n] = 1.0
    system[:n, n + 1 : n + 4] = source_m
    system[n + 1 : n + 4, :n] = source_m.T
    rhs = np.zeros((n + 4, 3))
    rhs[:n] = target
    solution = np.linalg.solve(system, rhs)
    kernel = np.linalg.norm(points_m[:, None] - source_m[None], axis=2)
    return kernel @ solution[:n] + solution[n] + points_m @ solution[n + 1 :]


def relax(points, adjacency, normals, rate, pinned):
    source, sink, count = adjacency
    mean = np.zeros_like(points)
    for axis in range(3):
        mean[:, axis] = np.bincount(source, points[sink, axis], len(points)) / count
    move = mean - points
    move -= normals * np.sum(move * normals, axis=1, keepdims=True)
    move[pinned] = 0.0
    return points + rate * move


def project(body, points, adjacency, midline, pinned, iterations=40):
    """Step onto the surface while evening out the edges; pinned vertices only do the first.

    The rings where hands and feet attach have neighbours on one side only, so relaxing them would
    pull them up the limb, away from the joint, and leave the extremity a gap to bridge.
    """
    for i in range(iterations):
        value = body(points)
        grad = body.gradient(points, step=1.5)
        size = np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-6)
        normals = grad / size
        points = points - 0.7 * normals * (value[:, None] / size)
        if i < iterations - 6:
            points = relax(points, adjacency, normals, 0.35, pinned)
        points[midline, 0] = 0.0
    return points


def quad_normals(points, quads):
    """mesh.face_normals for an F x 4 array of quads, without its per-face Python loop."""
    p0 = points[quads[:, 0]]
    a, b, c = points[quads[:, 1]] - p0, points[quads[:, 2]] - p0, points[quads[:, 3]] - p0
    return np.cross(a, b) + np.cross(b, c)


def untangle(body, points, quads, adjacency, midline, rounds=12, free=None):
    """Where faces fold over, smooth fully in a small neighbourhood and step back onto the surface."""
    source, sink, count = adjacency
    for _ in range(rounds):
        normals = quad_normals(points, quads)
        centres = points[quads].mean(1)
        folded = np.sum(normals * body.gradient(centres, step=1.5), axis=1) < 0
        if not folded.any():
            break
        active = np.zeros(len(points), bool)
        active[quads[folded]] = True
        for _ in range(3):
            grown = active.copy()
            grown[source[active[sink]]] = True
            active = grown
        if free is not None:
            active &= free
        for _ in range(12):
            mean = np.zeros_like(points)
            for axis in range(3):
                mean[:, axis] = np.bincount(source, points[sink, axis], len(points)) / count
            points[active] += 0.6 * (mean[active] - points[active])
            value = body(points[active])
            grad = body.gradient(points[active], step=1.5)
            size = np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-6)
            points[active] -= 0.7 * grad / size * (value[:, None] / size)
        points[midline, 0] = 0.0
    return points


def corner_angles(points, quads):
    """Every corner's angle in degrees, as a flat array."""
    p = points[quads]
    a = np.roll(p, 1, axis=1) - p
    b = np.roll(p, -1, axis=1) - p
    size = np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2)
    cosine = np.sum(a * b, axis=2) / np.maximum(size, 1e-9)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1))).ravel()


def flipped_quads(body, points, quads):
    """Which quads face into the surface."""
    outward = body.gradient(points[quads].mean(1), step=1.5)
    return np.sum(quad_normals(points, quads) * outward, axis=1) < 0


# Where the trunk's bands change from one of the editor's parameter groups to the next, in shares of
# the torso above the crotch: the shoulders and chest down to the rib cage's lower edge, the waist, then
# the hip in front and the seat behind.
TRUNK_SHOULDER = 0.6
TRUNK_WAIST = 0.3


def body_part(region, point, L):
    """The parameter group (sotaigen/data/sliders.json) of a template region at a point of a wrapped body."""
    if region == "trunk":
        level = (point[2] - L["crotch"]) / L["T"]
        if level >= TRUNK_SHOULDER:
            return "shoulder"
        if level >= TRUNK_WAIST:
            return "waist"
        return "hip" if point[1] < 0 else "glute"
    return {"hand": "arm", "foot": "leg"}.get(region, region)


def tally_parts(regions, points, L):
    """How many of `points`, each in its template region, lie in each part; most first, then by name."""
    parts = Counter(body_part(region, point, L) for region, point in zip(regions, points, strict=True))
    return dict(sorted(parts.items(), key=lambda row: (-row[1], row[0])))


def metrics(body, raw, warped, points, flipped, quads, edges):
    """What the wrap reports on the fit in `points`; `flipped` says which quads face in, on the finished mesh."""
    value = np.abs(body(points))
    grad = body.gradient(points, step=1.5)
    size = np.maximum(np.linalg.norm(grad, axis=1), 1e-6)
    distance = value / size
    length = lambda p: np.linalg.norm(p[edges[:, 0]] - p[edges[:, 1]], axis=1)
    ratio = length(points) / np.maximum(length(warped), 1e-6)
    raw_angles = corner_angles(raw, quads)
    angles = corner_angles(points, quads)
    return {
        "verts": len(points),
        "faces": len(quads),
        "surfaceErrorMm": {
            "p95": round(float(np.percentile(distance, 95)), 2),
            "max": round(float(distance.max()), 2),
        },
        "flippedFaces": int(flipped.sum()),
        "edgeChangeFromWarp": {
            "p1": round(float(np.percentile(ratio, 1)), 2),
            "p50": round(float(np.percentile(ratio, 50)), 2),
            "p99": round(float(np.percentile(ratio, 99)), 2),
        },
        "shortestEdgeMm": round(float(length(points).min()), 2),
        "cornerAngleDeg": {
            "rawP1": round(float(np.percentile(raw_angles, 1)), 1),
            "p1": round(float(np.percentile(angles, 1)), 1),
            "rawP99": round(float(np.percentile(raw_angles, 99)), 1),
            "p99": round(float(np.percentile(angles, 99)), 1),
        },
    }


def rotation_between(a, b):
    """Smallest rotation taking unit vector a to unit vector b."""
    v = np.cross(a, b)
    c = float(a @ b)
    skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + skew + skew @ skew / (1.0 + c)


def seam_rows(start, allowed, neighbours, rows):
    """Vertices within `rows` edge steps of `start`, walking only through `allowed` ones."""
    seen, front = set(start), set(start)
    for _ in range(rows - 1):
        front = {j for i in front for j in neighbours[i] if allowed[j]} - seen
        seen |= front
    return sorted(seen)


def smooth_correction(raw, neighbours, free, known, values):
    """Biharmonic fill of a displacement over `free` vertices, given its values on `known` ones.

    Matching several rows of `known` on each side carries the slope across, not only the value, so
    no crease forms where the correction meets the limb or dies out. Edges weigh by inverse length,
    because the template's edges in the palm and on the forearm differ a lot in length.
    """
    free, known = np.asarray(free), np.asarray(known)
    order = np.concatenate((free, known))
    column = {int(v): k for k, v in enumerate(order)}
    rows = [v for v in order if all(int(n) in column for n in neighbours[v])]
    laplacian = np.zeros((len(rows), len(order)))
    for r, v in enumerate(rows):
        members = np.array(neighbours[v])
        weight = 1.0 / np.maximum(np.linalg.norm(raw[members] - raw[v], axis=1), 1e-6)
        laplacian[r, [column[int(n)] for n in members]] = weight / weight.sum()
        laplacian[r, column[int(v)]] = -1.0
    a, b = laplacian[:, : len(free)], laplacian[:, len(free) :]
    return np.linalg.solve(a.T @ a, -a.T @ (b @ values))


def seam_crease(faces, points, ring, *, full_precision=False):
    """The sharpest fold along a seam, in degrees: the angle between faces across its ring's edges."""
    on_ring = set(ring)
    across = [
        owners
        for edge, owners in edge_owners(faces).items()
        if len(owners) == 2 and on_ring.issuperset(edge)
    ]
    quads = np.array(faces)[np.array(across)]
    first, second = quad_normals(points, quads[:, 0]), quad_normals(points, quads[:, 1])
    cosine = np.sum(first * second, axis=1) / np.maximum(
        np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1), 1e-9
    )
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).max())
    return angle if full_precision else round(angle, 1)


def attach_extremities(base, facts, L, keep, projected):
    """Place the template hands and feet and carry their tangents into the limbs."""
    raw = sculpted_foot(base)
    regions = np.array(base["regions"])
    full = np.zeros_like(raw)
    full[keep] = projected
    neighbours_of = vertex_neighbours(len(raw), base["faces"])

    left_hand = raw[(regions == "hand") & (raw[:, 0] > 0)]
    hand_scale = L["handLength"] / np.linalg.norm(left_hand - facts["wrist"], axis=1).max()
    left_foot = raw[(regions == "foot") & (raw[:, 0] > 0)]
    foot_scale = L["footLength"] / np.ptp(left_foot[:, 1])
    raw_fore = unit(facts["wrist"] - facts["elbow"])
    turn = rotation_between(raw_fore, L["foreDir"])
    report = {"handScale": round(float(hand_scale), 3), "footScale": round(float(foot_scale), 3)}

    for side in (1.0, -1.0):
        flip = np.array([side, 1.0, 1.0])
        for region, joint_raw, rotate, scale in (
            ("hand", facts["wrist"], turn, hand_scale),
            ("foot", facts["ankle"], np.eye(3), foot_scale),
        ):
            members = np.nonzero((regions == region) & (raw[:, 0] * side > 0))[0]
            ring = sorted(
                {j for i in members for j in neighbours_of[i] if keep[j] and regions[j] != region}
            )
            # The part hangs from the limb it really got, which the landmark only approximates. A
            # foot seats its ankle at the requested height, with its sole on the floor.
            joint_target = full[ring].mean(0) * flip
            height_change = 0.0
            if region == "foot":
                joint_target[2] = L["ankleJoint"][2]
                height_change = L["ankle"] - L["footLength"] * base["landmarks"]["anklePerFoot"]

            def place(
                points,
                flip=flip,
                rotate=rotate,
                scale=scale,
                joint_raw=joint_raw,
                joint_target=joint_target,
                height_change=height_change,
            ):
                # Work in the left side's frame so a rotation means the same on both sides.
                local = (points * flip - joint_raw) @ rotate.T * scale + joint_target
                if height_change:
                    # Keep sole and toe thickness. Ease in between 15% and 35% of the
                    # template ankle height, then carry the same slope up the instep.
                    above_toes = np.maximum(points[:, 2] / joint_raw[2] - 0.15, 0.0)
                    ramp = np.minimum(above_toes, 0.20)
                    height = (above_toes - ramp + ramp**2 / 0.40) / 0.75
                    local[:, 2] += height_change * (height - 1)
                return local * flip

            placed = place(raw[members])
            # Displacement from the placed template to the last few rows of the limb.
            limb_rows = np.array(seam_rows(ring, keep, neighbours_of, SEAM_ROWS))
            miss = full[limb_rows] - place(raw[limb_rows])
            distance = np.linalg.norm(raw[members][:, None] - raw[ring][None], axis=2).min(1)
            reached = distance < SEAM_REACH[region]
            # Past the reach the part stays as it was, over rows of its own for the slope's sake.
            beyond = np.zeros(len(raw), bool)
            beyond[members[~reached]] = True
            border = {j for i in members[reached] for j in neighbours_of[i] if beyond[j]}
            still = np.array(seam_rows(border, beyond, neighbours_of, SEAM_ROWS))
            correction = smooth_correction(
                raw,
                neighbours_of,
                members[reached],
                np.concatenate((limb_rows, still)),
                np.vstack((miss, np.zeros((len(still), 3)))),
            )
            full[members] = placed
            full[members[reached]] = placed[reached] + correction
            crease = seam_crease(base["faces"], full, ring, full_precision=True)
            if region == "foot" and crease > 14:
                # A planted foot and a thin shin can meet at too steep an angle when every
                # shin row is pinned. Share the horizontal correction across the join itself,
                # leaving the ring's height, the next shin row and the arch farther down fixed.
                band = np.r_[ring, members[reached]]
                pinned_rows = ~np.isin(limb_rows, ring)
                fixed = limb_rows[pinned_rows]
                change = smooth_correction(
                    raw,
                    neighbours_of,
                    band,
                    np.r_[fixed, still],
                    np.vstack((miss[pinned_rows, :2], np.zeros((len(still), 2)))),
                )
                spread = place(raw[band])[:, :2] + change
                weight = min(1.0, (crease - 14.0) / 4.0)
                full[band, :2] += weight * (spread - full[band, :2])
                crease = seam_crease(base["faces"], full, ring)
            if side > 0:
                ring_miss = full[ring] - place(raw[ring])
                report[f"{region}SeamMissMm"] = round(
                    float(np.linalg.norm(ring_miss, axis=1).max()), 1
                )
                report[f"{region}Lowest"] = round(float(full[members][:, 2].min()), 1)
                report[f"{region}SeamCreaseDeg"] = round(crease, 1)
    return full, report


def seam_radius(base, points, part):
    """Mean radius of the limb's end where the left hand or foot joins it, in the units of `points`."""
    raw = np.array(base["verts"], float)
    regions = np.array(base["regions"])
    neighbours = vertex_neighbours(len(raw), base["faces"])
    limb = np.isin(regions, ("trunk", "neck", "leg", "arm"))
    members = np.nonzero((regions == part) & (raw[:, 0] > 0))[0]
    ring = points[sorted({j for i in members for j in neighbours[i] if limb[j]})]
    return float(np.linalg.norm(ring - ring.mean(0), axis=1).mean())


def neck_columns(base):
    """The neck's vertices as ring vertices x rows from the ring down, and what lies under each column."""
    neighbours = vertex_neighbours(len(base["verts"]), base["faces"])
    regions = np.asarray(base["regions"])
    rows = [list(base["openings"]["neck"])]
    seen = set(rows[0])
    while True:
        below = [[w for w in neighbours[v] if w not in seen] for v in rows[-1]]
        if regions[below[0][0]] != "neck":
            return np.array(rows).T, below
        rows.append([w for (w,) in below])
        seen.update(rows[-1])


def even_columns(points, columns, below, straight=False):
    """Space each column's vertices evenly along it, between its ring vertex and the rows under the neck.

    `straight` lays them on the straight line between those two instead, which also takes out whatever
    the column has slid sideways.
    """
    ends = np.array([points[under].mean(0) for under in below])
    share = np.arange(columns.shape[1]) / columns.shape[1]
    if straight:
        ring = points[columns[:, 0]]
        points[columns] = ring[:, None] + (ends - ring)[:, None] * share[None, :, None]
        return
    line = np.concatenate((points[columns], ends[:, None]), axis=1)
    along = np.linalg.norm(np.diff(line, axis=1), axis=2).cumsum(1)
    along = np.column_stack((np.zeros(len(line)), along))
    for k, column in enumerate(columns):
        for axis in range(3):
            points[column, axis] = np.interp(along[k, -1] * share, along[k], line[k, :, axis])


def seat_neck_ring(base, body, head, points, L):
    """Slide the neck ring up the neck to where it enters the head, and even out the neck below it.

    Relaxing a vertex toward all its neighbours pulls it toward the narrow end of a surface that flares,
    since those along its row lie inside the flare. The neck's rows, whose ring has no row above it,
    come out of project spaced in a series that runs to nothing at the ring, the worse the faster the
    neck widens over the trapezius. So here they are spaced along their columns instead.
    """
    ring = np.asarray(base["openings"]["neck"])
    neck = np.nonzero(np.asarray(base["regions"]) == "neck")[0]
    # Every ring vertex lifted by every height at once, stepped onto the body, then tried in the head.
    heights = np.arange(0.0, 0.5 * L["head"], 1.0)
    lifted = points[ring, None] + heights[None, :, None] * np.array([0.0, 0.0, 1.0])
    lifted = lifted.reshape(-1, 3)
    value = body(lifted)
    grad = body.gradient(lifted, step=1.0)
    size = np.maximum(np.sum(grad * grad, axis=1, keepdims=True), 1e-6)
    inside = (head(lifted - grad / size * value[:, None]) <= 0).reshape(len(ring), -1)
    rise = np.where(inside.any(1), heights[inside.argmax(1)], 0.0)
    points[ring, 2] += rise
    columns, below = neck_columns(base)
    for i in range(NECK_PASSES):
        even_columns(points, columns, below, straight=i == 0)
        for _ in range(4):
            value = body(points[neck])
            grad = body.gradient(points[neck], step=1.0)
            size = np.maximum(np.sum(grad * grad, axis=1, keepdims=True), 1e-6)
            points[neck] -= 0.8 * grad * value[:, None] / size
    return rise


def fit_waist(body, points, quads, links, midline, pinned, trunk, L, p):
    """Refit the waist band while holding its exterior and pinned limb seams in place."""
    level = (points[:, 2] - L["crotch"]) / L["T"]
    band = waist_band(level, p["waist"])
    # Fade across five neighbouring arm rings so the region boundary cannot split the surface.
    influence = trunk.astype(float)
    source, sink, _ = links
    for _ in range(5):
        grown = influence.copy()
        np.maximum.at(grown, source, influence[sink] - 1 / 6)
        influence = grown
    active = (band > 0) & (influence > 0) & ~pinned
    fitted = points.copy()
    for iteration in range(30):
        value = body(fitted[active])
        gradient = body.gradient(fitted[active], step=1.5)
        size = np.maximum(np.linalg.norm(gradient, axis=1, keepdims=True), 1e-6)
        normals = np.zeros_like(fitted)
        normals[active] = gradient / size
        fitted[active] -= 0.7 * normals[active] * (value[:, None] / size)
        if iteration < 24:
            relaxed = relax(fitted, links, normals, 0.35, ~active)
            fitted += (relaxed - fitted) * influence[:, None]
        fitted[midline, 0] = 0.0
    return untangle(body, fitted, quads, links, midline, free=active)


# hip_front_lead's bands of height, in shares of the torso above the crotch, and how far either side of
# the midline it reads the side outline: past the breasts' apexes, short of the arms.
CHEST_BAND = (0.6, 0.8)
HIP_BAND = (0.0, 0.3)
OUTLINE_HALF_WIDTH = 0.25


def hip_front_lead(field, L):
    """How far, in mm, the hip's front stands ahead of the chest's in the side outline.

    Each front is the most forward surface over its band of heights within OUTLINE_HALF_WIDTH of the
    midline, which is what a side view shows: on a breasted body the chest's is the breasts' apexes,
    wherever their spacing puts them. Reported, not held: a child's belly or an older woman's stands
    ahead of her chest, and bodies past it wrap and rig as well as any (docs/DECISIONS.md). Negative is
    behind the chest.
    """
    T, at = L["T"], L["at"]
    step = 4.0
    xs = np.arange(0.0, OUTLINE_HALF_WIDTH * T, step)
    ahead = np.arange(0.5 * T, 0.0, -step)  # from well in front of the body back to the midplane

    def front(band):
        z = at(np.linspace(*band, 9))
        rays = np.stack(np.meshgrid(xs, z, indexing="ij"), axis=-1).reshape(-1, 2)
        points = np.insert(np.repeat(rays, len(ahead), axis=0), 1, np.tile(-ahead, len(rays)), axis=1)
        inside = (field(points) <= 0).reshape(len(rays), len(ahead))
        reached = inside.any(axis=1)
        coarse = np.where(reached, ahead[inside.argmax(axis=1)], 0.0)
        # Only the rays that might hold the front are refined, between their last step outside and first in.
        near = reached & (coarse >= coarse.max() - step)
        low, high = coarse[near], coarse[near] + step
        for _ in range(8):
            middle = (low + high) / 2
            into = field(np.insert(rays[near], 1, -middle, axis=1)) <= 0
            low, high = np.where(into, middle, low), np.where(into, high, middle)
        return float(low.max())

    return round(front(HIP_BAND) - front(CHEST_BAND), 1)


def place_loops(full, coarse, body, keep, raw, warped, points, quads):
    """The loops template/loops.py added, put onto a template wrapped without them.

    Each of their vertices starts at the middle of the rung it halves and goes onto the surface at
    the crossing nearest to it along the rung's normal, the mean of its ends'. There is no relaxing,
    as the rest of the wrap does: relaxing by edge counts would pull the vertices round the new
    loops toward them, and every vertex the template had before would move. A step along the
    field's gradient instead can cross the thin wall between the armpit and the front fold's
    crease, so it is only for where the normal finds no surface. Returns `keep`, `quads`, `raw`, `warped` and `points` for `full`.
    """
    count = len(coarse["verts"])
    kept = np.isin(np.array(full["regions"]), ("trunk", "neck", "leg", "arm"))
    if not kept[count:].all() or not np.array_equal(kept[:count], keep):
        raise ValueError("an added loop reaches a hand, a foot or the head")
    index = -np.ones(len(kept), int)
    index[kept] = np.arange(kept.sum())
    first = full["refined"]["from"]
    parents = full["refined"]["parents"]
    normals = np.zeros((kept.sum(), 3))
    corners = quad_normals(points, quads)
    for corner in quads.T:
        np.add.at(normals, corner, corners)

    def grown(values, unit=False):
        out = np.zeros((kept.sum(), 3))
        out[: len(values)] = values[: len(values)]
        for v in range(count, len(kept)):
            a, b = parents[v - first]
            out[index[v]] = out[index[a]] + out[index[b]]
            out[index[v]] /= np.linalg.norm(out[index[v]]) if unit else 2
        return out

    raw, warped, points = grown(raw), grown(warped), grown(points)
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    normals = grown(normals[: len(keep[keep])], unit=True)
    new = index[count:]
    reach = np.array([np.linalg.norm(points[index[a]] - points[index[b]]) for a, b in parents[count - first :]])
    steps = np.linspace(-1.0, 1.0, 41)
    along = points[new, None, :] + (steps[None, :, None] * reach[:, None, None]) * normals[new, None, :]
    values = body(along.reshape(-1, 3)).reshape(len(new), len(steps))
    placed = points[new].copy()
    for i in range(len(new)):
        crossings = np.flatnonzero(np.sign(values[i, :-1]) != np.sign(values[i, 1:]))
        if not len(crossings):
            # In a fold the rung's ends face apart and their mean runs along the skin: step down
            # the field's gradient there instead.
            point = points[new[i]][None]
            for _ in range(20):
                grad = body.gradient(point, step=1.5)
                point = point - 0.7 * grad * (body(point)[:, None] / max(np.sum(grad * grad), 1e-12))
            placed[i] = point[0]
            continue
        k = crossings[np.argmin(np.abs(steps[crossings] + steps[crossings + 1]))]
        lo, hi = steps[k], steps[k + 1]
        f_lo = values[i, k]
        for _ in range(30):
            mid = (lo + hi) / 2
            f_mid = body((points[new[i]] + mid * reach[i] * normals[new[i]])[None])[0]
            if np.sign(f_mid) == np.sign(f_lo):
                lo, f_lo = mid, f_mid
            else:
                hi = mid
        placed[i] = points[new[i]] + (lo + hi) / 2 * reach[i] * normals[new[i]]
    placed[np.abs(raw[new, 0]) < 0.5, 0] = 0.0
    points[new] = placed
    template_faces = np.array(full["faces"])
    quads = index[template_faces[kept[template_faces].all(1)]]
    return kept, quads, raw, warped, points


def wrap_body(base, p, out):
    """Wrap the template onto one body's surface; write the result to `out` and return wrap.json."""
    full_base, base = base, before_loops(base)
    body, L = calibrated(p)
    lead = hip_front_lead(body, L)
    # The measured profiles come off for the fit and go back on after it, innermost first.
    body, profiles = peeled(body, (ThighProfile, AbdominalProfile))
    reference_p = {k: v for k, v in p.items() if k not in PROFILES}
    offset = p.get("waistContourShift", 0.0)
    neutral_p = reference_p | {"waistContourShift": 0.0} if offset else reference_p
    neutral = calibrated(neutral_p)[0] if offset else body
    fitter = neutral.reference if isinstance(neutral, GluteContours) else neutral
    shaped = body.reference if isinstance(body, GluteContours) else body
    fitter.clips = shaped.clips = body.clips = []
    for group in [*body.groups, *fitter.groups]:
        group.pop("clips", None)
    field = head_field(p, L)
    source, target, facts = correspondences(base, fitter, field, L)
    keep = np.isin(np.array(base["regions"]), ("trunk", "neck", "leg", "arm"))
    index = -np.ones(len(keep), int)
    index[keep] = np.arange(keep.sum())
    template_faces = np.array(base["faces"])
    quads = index[template_faces[keep[template_faces].all(1)]]
    raw = np.array(base["verts"], float)[keep]
    warped = thin_plate(source, target, raw)
    midline = np.abs(raw[:, 0]) < 0.5
    warped[midline, 0] = 0.0
    attached = np.isin(np.array(base["regions"]), ("hand", "foot"))
    pinned = np.zeros(len(keep), bool)
    pinned[template_faces[attached[template_faces].any(1)]] = True
    pinned = pinned[keep]
    faces = quads.tolist()
    links = adjacency(len(raw), faces)
    points = project(fitter, warped.copy(), links, midline, pinned)
    points = untangle(fitter, points, quads, links, midline)
    if offset:
        trunk = np.array(base["regions"])[keep] == "trunk"
        points = fit_waist(shaped, points, quads, links, midline, pinned, trunk, L, p)
    if base is not full_base:
        keep, quads, raw, warped, points = place_loops(
            full_base, base, fitter, keep, raw, warped, points, quads
        )
        base, faces = full_base, quads.tolist()
    out.mkdir(parents=True, exist_ok=True)
    full, seam = attach_extremities(base, facts, L, keep, points)
    if isinstance(body, GluteContours):
        full = body.from_reference(full)
        warped = body.from_reference(warped)
    for layer in reversed(profiles):
        full = layer.from_reference(full)
        warped = layer.from_reference(warped)
        body = layer
    fitted = full[keep]
    seat_neck_ring(base, body, field, full, L)
    # Whether the mesh is broken is asked of the mesh as it is written; the other metrics are of the fit.
    kept = full[keep]
    flipped = flipped_quads(body, kept, quads)
    result = metrics(body, raw, warped, fitted, flipped, quads, edges_of(faces))
    result["landmarks"] = len(source)
    result.update(seam)
    template = np.array(base["verts"], float)
    off_midline = np.abs(template[:, 0]) > 1.0
    crossed = np.sign(full[:, 0]) != np.sign(template[:, 0])
    result["crossedMidline"] = int((crossed & off_midline).sum())
    # Only for a broken mesh, whose wrap.json nothing downstream builds on: a whole body's stays as it was.
    if result["flippedFaces"] or result["crossedMidline"]:
        regions = np.array(base["regions"])
        faced = quads[flipped]
        past = np.nonzero(crossed & off_midline)[0]
        result["brokenAt"] = {
            "flippedFaces": tally_parts(regions[keep][faced[:, 0]], kept[faced].mean(1), L),
            "crossedMidline": tally_parts(regions[past], full[past], L),
        }
    result["hipFrontLead"] = lead
    head = build_head(base, full, p, L)
    everything = insert_head(full, head, source_count(base))
    write_obj(out / "wrapped.obj", everything, extended(base)["faces"])
    np.save(out / "wrapped.npy", everything)
    (out / "wrap.json").write_text(json.dumps(result, indent=1))
    return result
