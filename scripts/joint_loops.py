"""Is there enough edge flow to bend each joint? Loops around it, and the folds its check pose leaves.

For each joint: how many edge loops lie within one limb radius on each side, how far apart they are,
and the sharpest fold between neighbouring faces on the outside of the bend, at rest and posed. The
outside is what a silhouette shows: the inside presses into itself however dense the mesh is.

uv run scripts/joint_loops.py [body ...]
"""

import sys

import numpy as np

from sotaigen.form.bodies import load_bodies
from sotaigen.form.head import template_rows
from sotaigen.form.wrap import quad_normals
from sotaigen.mesh import edge_owners, vertex_neighbours
from sotaigen.pipeline import BUILD, ensure_built
from sotaigen.template import load_template

# Ring name or the two segments meeting there, the pose bending it, its angle, and which way the
# outside of the bend faces along Y in the rest pose (+1 is the back).
JOINTS = {
    "shoulder": (("trunk", "upperArm"), "armsUp", 72, 1),
    "elbow": ("elbow", "elbowBend", 120, 1),
    "wrist": ("wrist", "reach", 0, 1),
    "hip": (("trunk", "upperLeg"), "crouch", 85, 1),
    "knee": ("knee", "crouch", 115, -1),
    "ankle": ("ankle", "crouch", 28, 1),
    "indexMiddle": ("indexIntermediate", "fist", 90, 1),
}


def read_points(path):
    with open(path) as f:
        rows = [line.split()[1:4] for line in f if line.startswith("v ")]
    return np.array(rows, float)


def ring_of(base, spec):
    if isinstance(spec, str):
        return list(base["rings"][spec])
    first, second = spec
    segments = base["segments"]
    raw = base["verts"]
    found = set()
    for face in base["faces"]:
        names = {segments[i] for i in face}
        if first in names and second in names:
            found.update(i for i in face if raw[i][0] > 0)
    return sorted(found)


def rows_on_side(ring, neighbours, height, sign, reach, limit=12):
    """Rows of vertices going away from the ring on one side, while they stay within `reach`."""
    rows, seen = [], set(ring)
    front = set(ring)
    for _ in range(limit):
        front = {j for i in front for j in neighbours[i] if j not in seen and sign * height[j] > 0}
        if not front:
            break
        seen |= front
        members = sorted(front)
        if np.median(np.abs(height[members])) > reach:
            break
        rows.append(members)
    return rows


def fold_angles(points, faces, pairs):
    normals = quad_normals(points, np.array(faces))
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    return np.degrees(
        np.arccos(np.clip(np.sum(normals[pairs[:, 0]] * normals[pairs[:, 1]], axis=1), -1, 1))
    )


def main(names):
    base = load_template()
    raw = np.array(base["verts"], float)
    faces = base["faces"]
    neighbours = vertex_neighbours(len(raw), faces)
    closed = [(edge, o) for edge, o in edge_owners(faces).items() if len(o) == 2]
    edges = np.array([edge for edge, _ in closed])
    pairs = np.array([o for _, o in closed])
    for name in names:
        ensure_built(name, ("pose",))
        rest = template_rows(np.load(BUILD / "bodies" / name / "wrapped.npy"), base)
        rest_folds = fold_angles(rest, faces, pairs)
        print(name)
        print(
            f"  {'joint':12s} {'radius':>6s} {'loops':>5s} {'spacing':>7s} "
            f"{'rest p95/max':>13s} {'posed p95/max':>14s}  pose"
        )
        for joint, (spec, pose, angle, outside) in JOINTS.items():
            ring = ring_of(base, spec)
            centre = rest[ring].mean(0)
            radius = np.linalg.norm(rest[ring] - centre, axis=1).mean()
            axis = np.linalg.svd(rest[ring] - centre)[2][2]
            height = (rest - centre) @ axis
            near = np.linalg.norm(rest - centre, axis=1) < 3 * radius
            height = np.where(near, height, 0.0)
            sides = [
                rows_on_side(ring, neighbours, height, sign, radius) for sign in (1.0, -1.0)
            ]
            rows = [row for side in sides for row in side]
            levels = sorted(float(np.median(height[row])) for row in rows)
            spacing = float(np.median(np.diff(levels))) if len(levels) > 1 else 0.0
            zone = np.zeros(len(raw), bool)
            zone[ring] = True
            for row in rows:
                zone[row] = True
            on_outside = outside * (rest[:, 1] - centre[1]) >= 0
            mask = (zone & on_outside)[edges].all(1)
            posed = template_rows(read_points(BUILD / "bodies" / name / "poses" / f"{pose}.obj"), base)
            posed_folds = fold_angles(posed, faces, pairs)
            print(
                f"  {joint:12s} {radius:6.1f} {len(rows) + 1:5d} {spacing:7.1f} "
                f"{np.percentile(rest_folds[mask], 95):6.1f}/{rest_folds[mask].max():6.1f} "
                f"{np.percentile(posed_folds[mask], 95):7.1f}/{posed_folds[mask].max():6.1f}"
                f"  {pose} {angle} deg"
            )


if __name__ == "__main__":
    main(sys.argv[1:] or list(load_bodies()))
