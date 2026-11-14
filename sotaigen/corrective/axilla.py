"""Local shoulder correctives over forward elevation and side elevation.

The reconstructed axillary pocket supplies the anatomical shape. Rotation-aware smoothing
redistributes the residual skinning deformation, retaining the rest-surface detail.
"""


import numpy as np

from sotaigen.corrective import pose_grid
from sotaigen.corrective.drive import bound, weights
from sotaigen.form.axilla import patch_weights
from sotaigen.form.field import ramp
from sotaigen.mesh import adjacency, triangulate, vertex_normals
from sotaigen.rig.skinning import (
    pose_skins,
    pose_verts,
    smoothed,
    swing_euler,
    torso_length,
)
from sotaigen.template import before_loops, onto_loops

KEYS = (0, 20, 35, 60, 75, 95)
SPREAD_KEYS = (-24, 0, 35, 72, 110)
# A half turn: the twist bones cancel most of a roll by the shoulder, so a target made without one
# still fits there, and a hand on top of the head (raised, rolled out 90) needs them.
TWIST_LIMIT = 180
# How much of the smoothing a side raise takes, by its key: up to about 72 skinning alone is close, and
# the full smoothing draws the posterior fold's edge into one long line. Mixed with a forward raise this
# fades out by STRENGTH_FORWARD degrees forward, where the keys clear the front fold.
SPREAD_STRENGTH = {35: 0.3, 72: 0.5}
STRENGTH_FORWARD = 60.0
# And a forward raise's own keys: skinning alone folds the armpit there, but all of the smoothing breaks
# the toon shadow on the ribs under the arm into teeth. Past the side keys above it applies in full.
FORWARD_STRENGTH = {35: 0.7, 60: 0.6, 75: 0.6, 95: 0.7}
SAMPLES = tuple((x, z) for x in KEYS for z in SPREAD_KEYS if (x, z) != (0, 0))
NAMES = {
    side: tuple(
        f"axilla{side.title()}{x}" + (f"Spread{z}" if z else "") for x, z in SAMPLES
    ) for side in ("left", "right")
}


def driver_metadata(targets=None):
    if hasattr(targets, "driver"):
        return targets.driver
    return {
        "kind": "grid",
        "keys": list(KEYS),
        "spreadKeys": list(SPREAD_KEYS),
        "twistLimit": TWIST_LIMIT,
        "samples": [list(p) for p in SAMPLES],
        "targets": {side: list(names) for side, names in NAMES.items()},
    }


def key_strength(angle, spread):
    """How much of its smoothing the key at this forward angle and side raise keeps."""
    if not 0 <= spread <= max(SPREAD_STRENGTH):
        return 1.0
    side_only = float(ramp(1 - angle / STRENGTH_FORWARD))
    side = 1.0 - (1.0 - SPREAD_STRENGTH.get(spread, 1.0)) * side_only
    return min(side, FORWARD_STRENGTH.get(angle, 1.0))


def sample_pose(side, angle, twist, spread):
    """A table point as a pose: the left arm's swing (angle, spread) and twist, mirrored on the right."""
    sign = 1 if side == "left" else -1
    return {side + "UpperArm": swing_euler(angle, sign * twist, sign * spread)}


def groups(table=None, source="sculpted"):
    """The table as correctives groups (sotaigen.corrective.drive), a side each: what files carry."""
    table = table or driver_metadata()
    if table.get("kind") == "simplices":
        return pose_grid.groups(table, source)
    return [
        {
            "name": f"axilla.{side}",
            "source": source,
            "axes": [
                {"bone": f"{side}UpperArm", "read": "swingX", "scale": 1, "keys": list(table["keys"])},
                # The right side's side raise read as the left's.
                {"bone": f"{side}UpperArm", "read": "swingZ", "scale": 1 if side == "left" else -1,
                 "keys": list(table["spreadKeys"])},
            ],
            "fades": [{"bone": f"{side}UpperArm", "read": "twist", "limit": table["twistLimit"]}],
            "samples": [list(p) for p in table["samples"]],
            "targets": list(table["targets"][side]),
        }
        for side in ("left", "right")
    ]


def pose_weights(pose, table=None):
    """Tensor-product interpolation, with zero rest correction and mirrored lateral inputs.

    The arm is read by its swing and twist (skinning.swing_twist) and the targets are made at
    twist-free swings (sample_pose), so a turn about one axis reads as it would by Euler angles.
    """
    return weights(pose, groups(table))


def bind_verts(verts, targets, pose):
    """The rest skin with the targets the pose drives added; off poses share the input."""
    return bound(verts, targets, pose_weights(pose, driver_metadata(targets)))


def _retain_surface(rest, goal, polygons, rotations, mobility):
    """Keep the local target's oriented material area from flattening or turning inside out.

    A positive area constraint in the bone-rotated tangent plane retains a fold's two sides.
    Boundary vertices are fixed. This geometric constraint belongs to the target construction,
    not the test's collapse threshold; the final interpolated mesh is checked independently.
    """
    original = rest[polygons]
    first = (polygons[:, :, None] == polygons[:, None, :]).argmax(axis=2)
    unique = first == np.arange(polygons.shape[1])
    material_rotation = ((rotations[polygons] * unique[:, :, None, None]).sum(1)
                         / unique.sum(1)[:, None, None])
    normal = np.cross(original, np.roll(original, -1, axis=1)).sum(1)
    area = np.linalg.norm(normal, axis=1) * 0.5
    normal /= np.maximum(2 * area[:, None], 1e-12)
    normal = (material_rotation @ normal[:, :, None])[:, :, 0]
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    free = mobility[polygons]
    # Triangles are padded to four slots. They still have only three material
    # vertices: merge their gradients before computing the PBD denominator and
    # count each affected vertex once. Otherwise cyclic corner order changes the
    # correction and mirrored patches can deform differently.
    repeated = [(previous, corner, np.flatnonzero(first[:, corner] == previous))
                for corner in range(polygons.shape[1]) for previous in range(corner)
                if np.any(first[:, corner] == previous)]
    # A nearly fixed boundary vertex must not absorb a whole face constraint.
    # Bound the cumulative solve displacement by its local material edge scale,
    # fading with the same support as the corrective itself.
    edge = np.linalg.norm(original - np.roll(original, -1, axis=1), axis=2)
    radius = np.zeros(len(rest))
    counts = np.zeros(len(rest))
    for corner in range(polygons.shape[1]):
        # Count both ends of each material edge, independent of face winding.
        for endpoint in (corner, (corner + 1) % polygons.shape[1]):
            np.add.at(radius, polygons[:, endpoint], edge[:, corner])
            np.add.at(counts, polygons[:, endpoint], edge[:, corner] > 0)
    radius = radius / np.maximum(counts, 1) * mobility
    result = goal.copy()
    for _ in range(192):
        points = result[polygons]
        cross = np.cross(points, np.roll(points, -1, axis=1)).sum(1)
        signed = np.sum(cross * normal, axis=1) * 0.5
        missing = np.maximum(0.3 * area - signed, 0)
        if not np.any(missing > 0.03 * area + 1e-8):
            break
        gradient = 0.5 * np.cross(
            np.roll(points, -1, axis=1) - np.roll(points, 1, axis=1), normal[:, None, :]
        )
        for previous, corner, rows in repeated:
            gradient[rows, previous] += gradient[rows, corner]
            gradient[rows, corner] = 0
        denominator = np.sum(free * np.sum(gradient**2, axis=2), axis=1)
        step = missing / np.maximum(denominator, 1e-8)
        delta = gradient * (step[:, None] * free)[:, :, None]
        movement = np.zeros_like(result)
        count = np.zeros(len(result))
        for corner in range(polygons.shape[1]):
            np.add.at(movement, polygons[:, corner], delta[:, corner])
            np.add.at(count, polygons[:, corner], (missing > 0) & unique[:, corner])
        result += movement / np.maximum(count[:, None], 1)
        offset = result - goal
        distance = np.linalg.norm(offset, axis=1)
        result = goal + offset * np.minimum(1, radius / np.maximum(distance, 1e-12))[:, None]
    return result


def _patch_topology(faces, raw, support):
    """Include whole quads before extracting the collar and remapping constraints.

    Selecting triangles first can drop the fourth corner of a boundary quad.
    searchsorted would then silently map that absent corner to an unrelated vertex.
    """
    selected = [f for f in faces if (support[f] > 0).any()]
    triangles = triangulate(selected, raw)
    ids, local = np.unique(triangles, return_inverse=True)
    local = local.reshape(-1, 3)
    quads = np.asarray([f for f in selected if len(f) == 4], int).reshape(-1, 4)
    polygons = np.vstack((np.column_stack((local, local[:, -1])), np.searchsorted(ids, quads)))
    return ids, local, polygons


def surface_goal(rest, posed, polygons, rotations, mobility, links, detail):
    """Evaluate the material surface constraint at an actual pose, including twist."""
    goal = smoothed(posed, *links, 18, free=mobility)
    goal += (rotations @ detail[:, :, None])[:, :, 0]
    goal = posed + (goal - posed) * mobility[:, None]
    return _retain_surface(rest, goal, polygons, rotations, mobility)


def targets(base, verts, bones, weights):
    """Bake local shape-preserving corrections; no radius-based transport of chest skin.

    Found on the template before template/loops.py's loops and carried onto them: the smoothing
    counts edges, so on the denser loops it would reach fewer millimetres and squeeze the front
    fold, and as it is every vertex the template had before gets the same correction.
    """
    small = before_loops(base)
    if small is not base:
        count = len(small["verts"])
        found = targets(small, np.asarray(verts)[:count], bones, np.asarray(weights)[:count])
        return {name: onto_loops(base, delta) for name, delta in found.items()}
    raw = np.asarray(base["verts"])
    lookup = {bone["name"]: bone for bone in bones}
    torso = torso_length(bones)
    result = {}
    for side, names in NAMES.items():
        sign = 1 if side == "left" else -1
        support = patch_weights(verts, lookup[side + "UpperArm"]["head"], torso)
        support *= (raw[:, 0] * sign > 0)
        support *= ~np.isin(base["segments"], ("head", "neck"))
        ids, faces, polygons = _patch_topology(base["faces"], raw, support)
        local = verts[ids]
        local_weights = weights[ids]
        free = support[ids]
        links = adjacency(len(local), faces.tolist())
        detail = local - smoothed(local, *links, 18, free=free)
        for (angle, spread), name in zip(SAMPLES, names, strict=True):
            pose = sample_pose(side, angle, 0, spread)
            posed = pose_verts(bones, local_weights, local, pose)
            skins = pose_skins(bones, pose)[:, :3, :3]
            blend = np.einsum("vb,bij->vij", local_weights, skins)
            # The polar rotation follows the bones continuously. A surface-normal/tangent
            # frame is unstable on the narrow material rows around a deeply bent axilla.
            u, _, vt = np.linalg.svd(blend)
            orientation = np.linalg.det(u @ vt)
            u[:, :, -1] *= orientation[:, None]
            rotation = u @ vt
            goal = surface_goal(local, posed, polygons, rotation, free, links, detail)
            correction = goal - posed
            delta = np.zeros_like(verts)
            delta[ids] = np.linalg.solve(blend, correction[:, :, None])[:, :, 0]
            result[name] = delta * key_strength(angle, spread)
    return result


def normal_targets(base, verts, bones, weights, targets, normals, *, triangles=None):
    """Pose-aware bind-space unit-normal deltas for the existing position targets.

    The runtime first adds weighted normal offsets and then applies its blended skin
    matrix. Invert that matrix at each key to reproduce the geometric surface normal.
    Intermediate reaches, twist and spread use the same interpolation as positions.
    """
    if triangles is None:
        triangles = triangulate(base["faces"], np.asarray(base["verts"]))
    triangles = np.asarray(triangles)
    normals = np.asarray(normals)
    head = np.asarray(base["segments"]) == "head"
    torso = torso_length(bones)
    result = {}
    table = driver_metadata(targets)
    sides = ({side: local["targets"] for side, local in table["sides"].items()}
             if table.get("kind") == "simplices" else NAMES)
    for side, all_names in sides.items():
        if table.get("kind") == "simplices":
            samples = []
            for name, point in zip(all_names, table["sides"][side]["points"], strict=True):
                if name:
                    angles = np.zeros(3)
                    angles[table.get("axes", [0, 1, 2])] = point
                    samples.append((name, angles))
        else:
            samples = [(name, (angle, 0, spread))
                       for name, (angle, spread) in zip(all_names, SAMPLES, strict=True)]
        names = [name for name, _ in samples]
        # One ring around every possibly moved triangle needs new shading. Accumulating
        # complete face fans there requires one further ring of geometry.
        travel = np.array([np.linalg.norm(targets[name], axis=1) for name in names])
        moved = travel.max(0) > 1e-6
        touched_tris = triangles[moved[triangles].any(1)]
        shading_ids = np.unique(touched_tris)
        shading_ids = shading_ids[~head[shading_ids]]
        shading = np.zeros(len(verts), bool)
        shading[shading_ids] = True
        fans = triangles[shading[triangles].any(1)]
        ids, local_tris = np.unique(fans, return_inverse=True)
        local_tris = local_tris.reshape(-1, 3)
        shaded = np.searchsorted(ids, shading_ids)
        local_weights = weights[ids]
        used = local_weights.any(0)
        packed = local_weights[:, used]
        original = normals[shading_ids].astype(float)
        original /= np.linalg.norm(original, axis=1, keepdims=True)
        for (name, (angle, twist, spread)), distance in zip(samples, travel, strict=True):
            skins = pose_skins(bones, sample_pose(side, angle, twist, spread))[used, :3, :]
            blend = np.einsum("vb,bij->vij", packed, skins)
            bound = verts[ids] + targets[name][ids]
            posed = (blend[:, :, :3] @ bound[:, :, None])[:, :, 0] + blend[:, :, 3]
            geometric = vertex_normals(posed, local_tris)[shaded]
            bind = np.linalg.solve(blend[shaded, :, :3], geometric[:, :, None])[:, :, 0]
            bind /= np.linalg.norm(bind, axis=1, keepdims=True)

            # Fade over the outer collar. An unmoved point next to a bent triangle can
            # need a new normal as well, so use the largest displacement in its incident faces.
            peak = np.zeros(len(verts))
            face_travel = distance[touched_tris].max(axis=1)
            for corner in touched_tris.T:
                np.maximum.at(peak, corner, face_travel)
            gain = ramp(peak[shading_ids] / (0.006 * torso))[:, None]
            bind = original + gain * (bind - original)
            bind /= np.linalg.norm(bind, axis=1, keepdims=True)
            delta = np.zeros_like(verts)
            live = gain[:, 0] > 0
            delta[shading_ids[live]] = bind[live] - original[live]
            result[name] = delta
    if table.get("kind") == "simplices":
        return pose_grid.Correctives(result, table)
    return result
