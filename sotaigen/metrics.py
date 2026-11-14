"""What a pose does to the mesh: how far faces and edges deform, where, and how limbs keep girth."""

import numpy as np

from sotaigen.corrective.crossings import SkinParts
from sotaigen.corrective.recipe import Correctives
from sotaigen.mesh import COLLAPSED, edges_of, face_areas, face_pairs, unit_face_normals
from sotaigen.rig.skinning import pose_verts


class RestMesh:
    """The rest-pose measures that every posed copy of a mesh is compared with."""

    def __init__(self, verts, faces):
        self.verts = verts
        self.faces = faces
        self.edges = edges_of(faces)
        self.area = np.maximum(face_areas(verts, faces), 1e-6)
        self.edge = self.edge_lengths(verts)

    def edge_lengths(self, points):
        return np.linalg.norm(points[self.edges[:, 0]] - points[self.edges[:, 1]], axis=1)

    def area_ratio(self, posed):
        return face_areas(posed, self.faces) / self.area


def pose_row(rest, posed):
    """One pose's row in poses.json."""
    area = rest.area_ratio(posed)
    edge = rest.edge_lengths(posed) / rest.edge
    return {
        "collapsedFaces": int((area < COLLAPSED).sum()),
        "areaP1": round(float(np.percentile(area, 1)), 2),
        "edgeP1": round(float(np.percentile(edge, 1)), 2),
        "edgeP99": round(float(np.percentile(edge, 99)), 2),
    }


def folded_edges(points, faces, pairs):
    """Which face pairs fold back on each other by more than 90 degrees."""
    normals = unit_face_normals(points, faces)
    return np.sum(normals[pairs[:, 0]] * normals[pairs[:, 1]], axis=1) < 0


def girth(points, members, start, end):
    """Mean distance of a limb slice from the limb's own axis."""
    axis = end - start
    axis = axis / np.linalg.norm(axis)
    offset = points[members] - start
    radial = offset - np.outer(offset @ axis, axis)
    return float(np.linalg.norm(radial, axis=1).mean())


def count_by(labels):
    """How often each label occurs, sorted by label so that reports compare line by line."""
    return {str(s): int((labels == s).sum()) for s in sorted(set(labels.tolist()))}


def posed_meshes(base, verts, bones, weights, poses, field):
    """Evaluate the same corrective shapes and skinning as the exported body, of this calibrated field."""
    correctives = Correctives.production(base, verts, bones, weights, field=field)
    for name, pose in poses.items():
        yield name, pose_verts(bones, weights, correctives.bind(verts, pose), pose)


def joint_report(base, verts, bones, weights, shoulder, poses, field):
    """Per pose, where the damage sits by template part and how the arm keeps its girth.

    `base` is the template with the head, `shoulder` the rest position of the shoulder joint and
    `poses` the poses to measure, `field` the body's calibrated field. Girths are shares of the rest girth, on the wearer's left arm, in
    every pose: no pose has to be known by name.
    """
    return posed_joint_report(
        base, verts, shoulder, posed_meshes(base, verts, bones, weights, poses, field)
    )


def posed_joint_report(base, verts, shoulder, meshes):
    """Measure each already-posed mesh against the same rest surface."""
    faces = base["faces"]
    segments = np.asarray(base["segments"])
    raw = np.asarray(base["verts"], float)
    face_segment = np.array([segments[f[0]] for f in faces])
    rest = RestMesh(verts, faces)
    pairs = face_pairs(faces)

    def ring(points, key):
        return points[base["rings"][key]].mean(axis=0)

    left = raw[:, 0] > 0
    rest_elbow, rest_wrist = ring(verts, "elbow"), ring(verts, "wrist")
    t = (verts - rest_elbow) @ (rest_wrist - rest_elbow) / np.sum((rest_wrist - rest_elbow) ** 2)
    fore_mid = (segments == "lowerArm") & left & (np.abs(t - 0.5) < 0.12)
    fore_end = (segments == "lowerArm") & left & (t > 0.88)
    u = (verts - shoulder) @ (rest_elbow - shoulder)
    u = u / np.sum((rest_elbow - shoulder) ** 2)
    arm_root = (segments == "upperArm") & left & (u > 0.12) & (u < 0.3)

    parts = SkinParts(base)
    report = {
        "restCreases": int(folded_edges(verts, faces, pairs).sum()),
        "restSelfIntersections": parts.intersections(verts)["selfIntersections"],
        "poses": {},
    }
    for pose, posed in meshes:
        area = rest.area_ratio(posed)
        bad = area < COLLAPSED
        folded = folded_edges(posed, faces, pairs)
        row = {
            "creases": int(folded.sum()),
            "collapsed": int(bad.sum()),
            "areaP1": float(np.percentile(area, 1)),
            "collapsedBySegment": count_by(face_segment[bad]),
            "foldsBySegment": count_by(face_segment[pairs[folded, 0]]),
            "legsPastMidline": parts.legs_past_midline(posed),
            **parts.intersections(posed),
        }
        e, w = ring(posed, "elbow"), ring(posed, "wrist")
        root = girth(posed, arm_root, e, posed[arm_root].mean(0))
        row["girth"] = {
            "foreMid": girth(posed, fore_mid, e, w)
            / girth(verts, fore_mid, rest_elbow, rest_wrist),
            "wrist": girth(posed, fore_end, e, w) / girth(verts, fore_end, rest_elbow, rest_wrist),
            "armRoot": root / girth(verts, arm_root, rest_elbow, verts[arm_root].mean(0)),
        }
        report["poses"][pose] = row
    return report
