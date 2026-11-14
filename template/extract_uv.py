# SPDX-License-Identifier: GPL-3.0-or-later
"""Lay the whole skin out in one 0-1 atlas and write sotaigen/data/template/uv.npz; see docs/systems/wrap-uv.md.

The body keeps the islands its authors drew in the CC0 source. The source spreads them over several
UV tiles, which glTF and VRM can't address, so they are brought to one texel density and packed. The
head is generated here, so it has no UVs in the source: the neck and the head are unwrapped together
as one island, cut only up the back, so a texture can cross the neck ring.

Only rerun when the template's faces or the head's topology change; everything else reads the npz.
The source is an external file, so always pass --disable-autoexec:
    Blender -b --factory-startup --disable-autoexec --python template/extract_uv.py \\
        -- <human_base_meshes_bundle.blend> <shape.npy>

The second file is the shape densities are measured on and the head is unwrapped on. UVs are shared
by every body; use the mean of the typical bodies as described in template/PROVENANCE.md.
`tests/test_uv.py` checks the atlas on every body.
"""

import hashlib
import itertools
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sotaigen.form.head import ROWS, SIDES, extended, neck_ring_order
from sotaigen.template import load_template

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract

OUT = Path(__file__).resolve().parents[1] / "sotaigen" / "data" / "template" / "uv.npz"
OBJECT = "GEO-body_female_stylized"
# Texels per millimetre relative to the body. A face needs them most; hands are seen close up.
DENSITY = {"head": 1.5, "hand": 1.25}
# Gap around each island, as a share of the atlas: 8 texels at 2048 between neighbours.
MARGIN = 0.002
# The ways of packing to try. How tight a packing comes out swings by several percent with small changes in
# the islands' shapes, so one setting isn't enough.
PACKINGS = tuple(
    itertools.product(
        ("CONCAVE", "CONVEX", "AABB"),
        ("ANY", "CARDINAL", "AXIS_ALIGNED", "AXIS_ALIGNED_X", "AXIS_ALIGNED_Y"),
    )
)


def coverage(mesh):
    """How much of the atlas the islands fill, from the mesh's UVs as they stand in object mode."""
    corners = np.empty((len(mesh.loops), 2))
    mesh.uv_layers[0].data.foreach_get("uv", corners.ravel())
    total = 0.0
    for poly in mesh.polygons:
        points = corners[list(poly.loop_indices)]
        x, y = points[:, 0], points[:, 1]
        total += 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    return total, corners


def pack_tightest(obj):
    """Pack every way in PACKINGS from the same layout, and keep whichever fills the atlas most."""
    mesh = obj.data
    bpy.ops.object.mode_set(mode="OBJECT")
    _, start = coverage(mesh)
    best = None
    for shape, rotate in PACKINGS:
        mesh.uv_layers[0].data.foreach_set("uv", start.ravel())
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.uv.select_all(action="SELECT")
        bpy.ops.uv.pack_islands(
            rotate=True,
            rotate_method=rotate,
            shape_method=shape,
            margin_method="ADD",
            margin=MARGIN,
            udim_source="CLOSEST_UDIM",
        )
        bpy.ops.object.mode_set(mode="OBJECT")
        filled, corners = coverage(mesh)
        print("UV_PACK", shape, rotate, round(filled, 4), flush=True)
        if best is None or filled > best[0]:
            best = (filled, corners.copy())
    mesh.uv_layers[0].data.foreach_set("uv", best[1].ravel())
    bpy.ops.object.mode_set(mode="EDIT")
    print("UV_PACKED", round(best[0], 4), flush=True)


def source_uvs(blend, base):
    """The source's UV for every corner of the template's faces, in the template's order.

    The template is cut out again with extract.py's loops, whose UVs BMesh interpolates, and has to
    come out as base-mesh.json holds it.
    """
    extract.REFINE = base.get("refined", {}).get("besideRings", {})
    cut, uvs = extract.template(Path(blend))
    if cut["verts"] != base["verts"] or cut["faces"] != base["faces"]:
        raise SystemExit("the source doesn't cut to base-mesh.json: rerun template/extract.py")
    return uvs


def build_mesh(base, points, uvs):
    """The whole skin as a Blender object with the source's UVs on the template's faces."""
    full = extended(base)
    mesh = bpy.data.meshes.new("skin")
    mesh.from_pydata(points.tolist(), [], full["faces"])
    mesh.validate()
    mesh.uv_layers.new(name="UVMap")
    obj = bpy.data.objects.new("skin", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    for poly, corners in zip(mesh.polygons, uvs, strict=False):
        for loop, uv in zip(poly.loop_indices, corners, strict=True):
            mesh.uv_layers[0].data[loop].uv = uv
    return obj, full


def head_cut(base):
    """The head's column of edges up the back, from the neck ring's midline vertex to the crown."""
    raw = np.asarray(base["verts"], float)
    ring = neck_ring_order(base)
    back = int(np.argmax(raw[ring, 1] - 1e3 * (np.abs(raw[ring, 0]) > 0.5)))
    first = len(raw)
    column = [int(ring[back])]
    column += [first + i * SIDES + 2 * back for i in range(ROWS)]
    column.append(first + ROWS * SIDES)
    return list(itertools.pairwise(column))


def islands_of(bm, uv_layer):
    """Faces joined through edges that aren't seams, as lists of faces."""
    seen, found = set(), []
    for start in bm.faces:
        if start.index in seen:
            continue
        seen.add(start.index)
        stack, members = [start], []
        while stack:
            face = stack.pop()
            members.append(face)
            for edge in face.edges:
                if edge.seam:
                    continue
                for other in edge.link_faces:
                    if other.index not in seen:
                        seen.add(other.index)
                        stack.append(other)
        found.append(members)
    return found


def corners_match(edge, uv_layer):
    """Whether the two faces of an edge agree on its UVs."""
    first, second = edge.link_faces[0], edge.link_faces[1]

    def ends(face):
        return {
            loop.vert.index: tuple(round(c, 6) for c in loop[uv_layer].uv)
            for loop in face.loops
            if loop.vert in edge.verts
        }

    return ends(first) == ends(second)


def face_area_uv(face, uv_layer):
    """A face's area in UV space, by the shoelace formula."""
    p = [loop[uv_layer].uv for loop in face.loops]
    twice = sum(a.x * b.y - b.x * a.y for a, b in zip(p, p[1:] + p[:1], strict=True))
    return abs(twice) / 2


def main():
    args = sys.argv[sys.argv.index("--") + 1 :]
    blend, shape = Path(args[0]), Path(args[1])
    base = load_template()
    digest = hashlib.sha256(blend.read_bytes()).hexdigest()
    if digest != base["source"]["blendSha256"]:
        raise SystemExit(f"{blend} is not the template's source: {digest}")
    points = np.load(shape)

    obj, full = build_mesh(base, points, source_uvs(blend, base))
    regions = full["regions"]
    template_faces = len(base["faces"])

    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv[0]

    # Faces keep their polygon order through validate(); the head comes after the template.
    is_head = {f.index: f.index >= template_faces for f in bm.faces}
    for edge in bm.edges:
        faces = edge.link_faces
        edge.seam = (
            len(faces) == 2
            and not any(is_head[f.index] for f in faces)
            and not corners_match(edge, uv_layer)
        )
    cut = {tuple(sorted(pair)) for pair in head_cut(base)}
    for edge in bm.edges:
        if tuple(sorted(v.index for v in edge.verts)) in cut:
            edge.seam = True

    # Neck and head: the islands holding head faces, once the ring between them is no seam.
    together = next(
        members
        for members in islands_of(bm, uv_layer)
        if any(is_head[f.index] for f in members)
    )
    together_ids = {f.index for f in together}
    for face in bm.faces:
        face.select_set(face.index in together_ids)
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.uv.unwrap(method="MINIMUM_STRETCH", margin=0.0)

    # One texel density across islands, measured on the reference shape, then each region's share.
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv[0]
    for members in islands_of(bm, uv_layer):
        area_3d = sum(f.calc_area() for f in members)
        area_uv = sum(face_area_uv(f, uv_layer) for f in members)
        counts = {}
        for face in members:
            region = regions[face.loops[0].vert.index]
            counts[region] = counts.get(region, 0) + 1
        boost = DENSITY.get(max(counts, key=counts.get), 1.0)
        scale = boost * (area_3d / area_uv) ** 0.5
        for face in members:
            for loop in face.loops:
                loop[uv_layer].uv *= scale
    for face in bm.faces:
        face.select_set(True)
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.uv.select_all(action="SELECT")
    pack_tightest(obj)
    bpy.ops.object.mode_set(mode="OBJECT")

    mesh = obj.data
    corners = np.empty((len(mesh.loops), 2))
    mesh.uv_layers[0].data.foreach_get("uv", corners.ravel())
    sizes = np.array([len(f) for f in full["faces"]])
    order = [loop for poly in mesh.polygons for loop in poly.loop_indices]
    if [len(p.vertices) for p in mesh.polygons] != sizes.tolist():
        raise SystemExit("Blender changed the faces")
    # The loops follow each polygon's own vertices, so check them against the face list too.
    for poly, face in zip(mesh.polygons, full["faces"], strict=True):
        if list(poly.vertices) != list(face):
            raise SystemExit(f"polygon {poly.index} changed its corner order")
    # The packer works up to the atlas's edge; leave the same gap there as between islands.
    corners = 0.5 + (corners[order] - 0.5) * (1 - 2 * MARGIN)
    print("UV_RANGE", corners.min(0), corners.max(0))
    np.savez_compressed(
        OUT,
        corners=np.round(np.clip(corners, 0, 1) * 65535).astype(np.uint16),
        sizes=sizes.astype(np.uint16),
    )
    print("UV_EXTRACT", OUT, len(corners), "corners")


if __name__ == "__main__":
    main()
