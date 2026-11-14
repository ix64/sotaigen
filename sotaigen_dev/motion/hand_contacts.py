"""Measure padPinch and cylinderGrip against the posed skin, in palm lengths.

Run with one or more body names. The handle axis, centre and radius are expressed in the
wrist-to-middle-knuckle frame, so a shorter body holds a correspondingly smaller prop.
"""

import argparse

import numpy as np

from sotaigen.corrective.crossings import delivered_skin, edges_through_triangles
from sotaigen.mesh import unit_face_normals
from sotaigen.pipeline import rigged
from sotaigen.rig.poses import POSES
from sotaigen.rig.skinning import pose_verts, rest_matrix

FINGERS = ("thumb", "index", "middle", "ring", "little")
HANDLE_CENTER = np.array([0.0, 0.80, 0.4274])
HANDLE_AXIS = np.array([1.0, -0.19, 0.0])
HANDLE_AXIS /= np.linalg.norm(HANDLE_AXIS)
HANDLE_RADIUS = 0.258
THIN_HANDLE_RADIUS = 0.18
BALL_CENTER = np.array([0.05, 0.90, 0.64])
BALL_RADIUS = 0.40


def handle_contact(points, triangles):
    """Closest radial vector per triangle in a cross-section of the infinite cylinder."""
    height = points - HANDLE_CENTER
    lateral = np.cross((0, 0, 1), HANDLE_AXIS)
    flat = np.column_stack((height @ lateral, height[:, 2]))
    corners = flat[triangles]
    following = np.roll(corners, -1, axis=1)
    edge = following - corners
    parameter = -np.sum(corners * edge, axis=2) / np.maximum(np.sum(edge**2, axis=2), 1e-20)
    nearest = corners + np.clip(parameter, 0, 1)[:, :, None] * edge
    nearest_edge = np.argmin(np.sum(nearest**2, axis=2), axis=1)
    closest = nearest[np.arange(len(triangles)), nearest_edge]
    # In cross-section, the origin can also fall inside the triangle.
    cross = corners[:, :, 0] * following[:, :, 1] - corners[:, :, 1] * following[:, :, 0]
    closest[(cross > 0).all(axis=1) | (cross < 0).all(axis=1)] = 0
    return closest


def sphere_clearances(points, triangles):
    """Distance from the sphere to the closest point on each filled triangle."""
    corners = points[triangles] - BALL_CENTER
    edges = np.roll(corners, -1, axis=1) - corners
    t = -np.sum(corners * edges, axis=2) / np.maximum(np.sum(edges**2, axis=2), 1e-20)
    edge_distance = np.linalg.norm(corners + np.clip(t, 0, 1)[:, :, None] * edges, axis=2).min(1)
    normal = np.cross(edges[:, 0], edges[:, 1])
    size = np.linalg.norm(normal, axis=1)
    normal /= np.maximum(size[:, None], 1e-20)
    inside = (np.sum(np.cross(edges, -corners) * normal[:, None, :], axis=2) >= 0).all(axis=1)
    plane_distance = np.abs(np.sum(normal * corners[:, 0], axis=1))
    return np.where(inside & (size > 1e-12), plane_distance, edge_distance) - BALL_RADIUS


def pad_faces(finger, side, faces, segments, verts, bone):
    """Outward-facing pulp in the distal half of a phalanx, excluding the nail and tip cap."""
    distal = [f for f in faces if np.all(segments[f] == f"{finger}Distal")]
    matrix = rest_matrix(bone)
    centers = np.array([verts[f].mean(0) for f in distal])
    length = np.linalg.norm(bone["tail"] - bone["head"])
    along = (centers - bone["head"]) @ matrix[:3, 1] / length
    transverse = matrix[:3, 0] * side if finger == "thumb" else matrix[:3, 2]
    facing = unit_face_normals(verts, distal) @ transverse
    return [f for f, t, n in zip(distal, along, facing) if 0.45 < t < 0.95 and n > 0.50]


def pinch_contact(verts, patches, palm_length):
    thumb, index = (np.array([verts[f].mean(0) for f in faces]) for faces in patches)
    tn, fn = (unit_face_normals(verts, faces) for faces in patches)
    separation = np.linalg.norm(thumb[:, None] - index, axis=2) / palm_length
    alignment = tn @ fn.T
    closest = np.unravel_index(np.argmin(separation), separation.shape)
    offset = (thumb[closest[0]] - index[closest[1]]) @ fn[closest[1]] / palm_length
    return float(separation[closest]), float(alignment[closest]), float(offset)


def measure(preset):
    rig = rigged(preset)
    base, rest, bones, weights = rig.base, rig.verts, rig.bones, rig.weights
    segments = np.asarray(base["segments"])
    by_name = {b["name"]: b for b in bones}
    mask = np.asarray(base["regions"]) == "hand"
    posed = {
        name: pose_verts(bones, weights, rest, POSES[name]) for name in ("padPinch", "cylinderGrip")
    }
    rows = {}
    for side, sign in (("left", 1), ("right", -1)):
        skin = mask & (sign * np.asarray(base["verts"])[:, 0] > 0)
        faces = [f for f in base["faces"] if np.all(skin[f])]
        triangles, edges = delivered_skin(faces, base["verts"])
        frame = rest_matrix(by_name[f"{side}Hand"])
        palm_length = np.linalg.norm(by_name[f"{side}Hand"]["tail"] - frame[:3, 3])
        pinch, grip = posed["padPinch"], posed["cylinderGrip"]
        patches = [
            pad_faces(finger, sign, faces, segments, rest, by_name[f"{side}{finger.title()}Distal"])
            for finger in ("thumb", "index")
        ]
        gap, alignment, offset = pinch_contact(pinch, patches, palm_length)
        local = (grip - frame[:3, 3]) @ frame[:3, :3] * (sign, 1, 1) / palm_length
        radial = handle_contact(local, triangles)
        clearances = np.linalg.norm(radial, axis=1) - HANDLE_RADIUS
        directions = {}
        distal = {}
        for finger in FINGERS:
            members = np.flatnonzero((segments[triangles] == f"{finger}Distal").all(axis=1))
            nearest = members[np.argmin(clearances[members])]
            distal[finger] = float(clearances[nearest])
            directions[finger] = radial[nearest] / max(np.linalg.norm(radial[nearest]), 1e-12)
        rows[side] = {
            "palmMm": palm_length,
            "padGap": gap,
            "padNormalDot": alignment,
            "padNormalOffset": offset,
            "handleMin": float(clearances.min()),
            "distalGaps": distal,
            "gripOpposition": max(
                directions["thumb"] @ directions[finger] for finger in FINGERS[1:]
            ),
            "crossings": {
                name: len(edges_through_triangles(points, edges, triangles))
                for name, points in posed.items()
            },
        }
    return rows


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("presets", nargs="+")
    for preset in parser.parse_args().presets:
        print(json.dumps({preset: measure(preset)}))
