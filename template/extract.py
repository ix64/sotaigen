# SPDX-License-Identifier: GPL-3.0-or-later
"""Cut the body below the neck out of the CC0 base mesh into sotaigen/data/template/base-mesh.json.

Only rerun when the cut itself changes; everything else reads the JSON, never the .blend. The source
is an external file, so always pass --disable-autoexec:
    Blender -b --factory-startup --disable-autoexec --python template/extract.py \\
        -- <human_base_meshes_bundle.blend>
"""

import hashlib
import json
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector

OUT = Path(__file__).resolve().parents[1] / "sotaigen" / "data" / "template" / "base-mesh.json"
OBJECT = "GEO-body_female_stylized"
# Face sets of the source mesh, read from the file. Everything not listed is left behind.
FACE_SETS = {
    "trunk": {258, 259, 260},
    "neck": {6},
    "arm": {263, 262},
    "leg": {256, 257},
    "hand": {*range(272, 287), 359},
    "foot": {288, *range(316, 323)},
}
# The neck has no face-set edge low enough, so it is cut along a loop near a seed, in metres.
NECK_SEED = ((0.0, 0.03, 1.352), (1, 0, 0))
# Named anatomical parts, which the skin weights start from. Sets not listed count as trunk or neck.
SEGMENTS = {
    256: "upperLeg",
    257: "lowerLeg",
    263: "upperArm",
    262: "lowerArm",
    273: "hand",
    359: "thumbMetacarpal",
    285: "thumbProximal",
    284: "thumbDistal",
    278: "indexProximal",
    283: "indexIntermediate",
    282: "indexDistal",
    277: "middleProximal",
    279: "middleIntermediate",
    272: "middleDistal",
    280: "ringProximal",
    274: "ringIntermediate",
    275: "ringDistal",
    281: "littleProximal",
    286: "littleIntermediate",
    276: "littleDistal",
    288: "foot",
    316: "foot",
    **dict.fromkeys(range(317, 323), "toes"),
}
FORMAT = "sotaigen/template"
VERSION = 1
# What the code needs of the template that its topology doesn't say, in the template's own numbering
# and millimetres, on the wearer's left. Written to the JSON as they are (template/PROVENANCE.md).
LANDMARKS = {
    # The centre the template's upper arm turns about, which the wrap maps onto the body's shoulder.
    "shoulderPivot": [128.0, 15.0, 1215.0],
    "vertices": {
        # The axilla: the crest of the anterior fold, the suspended floor and the crest of the
        # posterior fold, which the wrap keeps apart.
        "axillaFrontFold": 1437,
        "axillaFloor": 1441,
        "axillaBackFold": 1447,
        # Where the folds' supports run to on the chest wall and on the upper arm: skin handles,
        # not the tendons' footprints (sotaigen.contact.axilla_ribbons).
        "axillaFrontChest": 2031,
        "axillaFrontArm": 2255,
        "axillaBackChest": 1781,
        "axillaBackArm": 2248,
    },
    # The ankle's height above the sole, and how far the foot reaches inward of the ankle, as
    # shares of the foot's length.
    "anklePerFoot": 0.471,
    "footInnerPerFoot": 0.186,
}
# Joints are recorded where two face sets meet.
SET_RINGS = {
    "elbow": (263, 262),
    "wrist": (262, 273),
    "knee": (256, 257),
    "ankle": (257, 288),
    "toes": (316, 317),
    "thumbMetacarpal": (273, 359),
    "thumbProximal": (359, 285),
    "thumbDistal": (285, 284),
    "indexProximal": (273, 278),
    "indexIntermediate": (278, 283),
    "indexDistal": (283, 282),
    "middleProximal": (273, 277),
    "middleIntermediate": (277, 279),
    "middleDistal": (279, 272),
    "ringProximal": (273, 280),
    "ringIntermediate": (280, 274),
    "ringDistal": (274, 275),
    "littleProximal": (273, 281),
    "littleIntermediate": (281, 286),
    "littleDistal": (286, 276),
}


# Edge loops added around a joint ring: {ring: strips on each side}, or {ring: {face set: strips}}
# for one side only. One loop goes through the middle
# of each of the nearest strips of quads on either side, so 2 adds four loops to each such joint.
# The source's vertices keep their order and the added ones follow them; `refined` in the JSON says
# which vertices they are, their two parents and the faces before the cut (docs/parts/limbs.md).
REFINE = {"elbow": 2, "knee": 2}
# Rings loops can be cut beside that aren't joints of the template: the shoulder's, where the upper
# arm meets the trunk. They aren't written to `rings`.
LOOP_RINGS = {"shoulder": (260, 263)}


def edge_loop(start_edge, start_vert):
    """Walk a closed edge loop through valence-4 vertices, or return None if it hits a pole."""
    loop, edge, vert = [start_edge], start_edge, start_vert
    while True:
        if len(vert.link_edges) != 4:
            return None
        faces = set(edge.link_faces)
        following = [e for e in vert.link_edges if e is not edge and not faces & set(e.link_faces)]
        if len(following) != 1:
            return None
        edge = following[0]
        if edge is start_edge:
            return loop
        loop.append(edge)
        vert = edge.other_vert(vert)
        if len(loop) > 400:
            return None


def find_cut(bm, seed, across):
    """The closed loop nearest the seed whose start edge runs most nearly along `across`."""
    seed, across = Vector(seed), Vector(across).normalized()
    for vert in sorted(bm.verts, key=lambda v: (v.co - seed).length)[:60]:
        edges = sorted(
            vert.link_edges,
            key=lambda e: -abs((e.other_vert(vert).co - vert.co).normalized().dot(across)),
        )
        loop = edge_loop(edges[0], edges[0].other_vert(vert))
        if loop and len(loop) >= 8:
            return loop
    raise SystemExit(f"no closed loop near {tuple(seed)}")


def between_sets(bm, layer, pair):
    """The edges where two face sets meet, split by side as (left, right)."""
    edges = [e for e in bm.edges if {f[layer] for f in e.link_faces} == set(pair)]
    return [e for e in edges if e.verts[0].co.x > 0], [e for e in edges if e.verts[0].co.x < 0]


def strip_rungs(ring, first_set, count, layer):
    """The edges across the `count` strips of quads beside a ring, starting on `first_set`'s side."""
    loop = {v for e in ring for v in e.verts}
    seen, rungs = set(), set()
    for step in range(count):
        faces = {
            f
            for v in loop
            for f in v.link_faces
            if f not in seen
            and sum(u in loop for u in f.verts) == 2
            and (step or f[layer] == first_set)
        }
        across = {f: [e for e in f.edges if sum(v in loop for v in e.verts) == 1] for f in faces}
        if len(faces) != len(loop) or any(len(f.verts) != 4 or len(across[f]) != 2 for f in faces):
            raise SystemExit(f"strip {step} beside a ring of {len(loop)} isn't a ring of quads")
        strip = {e for edges in across.values() for e in edges}
        if len(strip) != len(loop):
            raise SystemExit(f"strip {step} beside a ring of {len(loop)} isn't closed")
        rungs |= strip
        seen |= faces
        loop = {v for f in faces for v in f.verts} - loop
    return rungs


def kept_faces(bm, barrier, layer, key):
    """The faces reached from the trunk without crossing the neck cut, in the mesh's order, each as
    its vertices' keys, its face set and its corners' UVs. `barrier` holds the cut's edges as pairs
    of keys."""
    uv = bm.loops.layers.uv.active
    seed = next(f for f in bm.faces if f[layer] == 258)
    kept, stack = {seed}, [seed]
    while stack:
        for edge in stack.pop().edges:
            if frozenset(key(v) for v in edge.verts) in barrier:
                continue
            for other in edge.link_faces:
                if other not in kept:
                    kept.add(other)
                    stack.append(other)
    bm.faces.index_update()
    return [
        (tuple(key(v) for v in f.verts), f[layer], [tuple(c[uv].uv) for c in f.loops])
        for f in sorted(kept, key=lambda f: f.index)
    ]


def source(blend):
    """The faces before and after REFINE's loops, the vertices' positions, the cut and the rings.

    Vertices are keyed by their index in the source; one an added loop made is keyed by the source
    vertices it lies between. BMesh drops its Python handles in the cut, so nothing else is kept.
    """
    with bpy.data.libraries.load(str(blend), link=False) as (_, data):
        data.objects = [OBJECT]
    bm = bmesh.new()
    bm.from_mesh(data.objects[0].data)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    layer = bm.faces.layers.int[".sculpt_face_set"]
    tag = bm.verts.layers.int.new("source")
    for vert in bm.verts:
        vert[tag] = vert.index + 1

    def pairs(edges):
        return {frozenset(v.index for v in e.verts) for e in edges}

    cut = {"neck": pairs(find_cut(bm, *NECK_SEED))}
    rings = {name: between_sets(bm, layer, pair) for name, pair in SET_RINGS.items()}
    beside = {**SET_RINGS, **LOOP_RINGS}
    barrier = set().union(*cut.values())
    before = kept_faces(bm, barrier, layer, lambda v: v.index)
    rungs = set()
    for name, count in REFINE.items():
        # A count of strips on both sides, or {face set: count} for one side only.
        counts = count if isinstance(count, dict) else dict.fromkeys(map(str, beside[name]), count)
        for ring in rings.get(name) or between_sets(bm, layer, beside[name]):
            for first_set, strips in counts.items():
                rungs |= strip_rungs(ring, int(first_set), strips, layer)
    rings = {name: pairs(pair[0]) for name, pair in rings.items()}
    position = {v.index: v.co.copy() for v in bm.verts}
    if rungs:
        result = bmesh.ops.subdivide_edges(bm, edges=list(rungs), cuts=1, use_grid_fill=False)
        for vert in result["geom_inner"]:
            if isinstance(vert, bmesh.types.BMVert):
                vert[tag] = 0

    def key(vert):
        if vert[tag]:
            return vert[tag] - 1
        ends = sorted(u[tag] - 1 for u in (e.other_vert(vert) for e in vert.link_edges) if u[tag])
        if len(ends) != 2:
            raise SystemExit("an added vertex doesn't halve one rung")
        position[tuple(ends)] = vert.co.copy()
        return tuple(ends)

    after = kept_faces(bm, barrier, layer, key)
    return before, after, position, cut, rings


def template(blend):
    """The template as base-mesh.json holds it, and the source's UV of every corner of its faces."""
    before, after, position, cut, rings = source(blend)
    region_of = {s: name for name, group in FACE_SETS.items() for s in group}
    stray = sorted({face_set for _, face_set, _ in before} - set(region_of))
    if stray:
        raise SystemExit(f"face sets without a region: {stray}")

    # The source's vertices are numbered on the faces before the loops, so the loops can't move them.
    index, verts, parts = {}, [], []

    def number(face):
        corners, face_set, _ = face
        row = []
        for vert in corners:
            if vert not in index:
                index[vert] = len(verts)
                verts.append([round(c * 1000, 2) for c in position[vert]])
                parts.append(SEGMENTS.get(face_set, region_of[face_set]))
            row.append(index[vert])
        return row

    coarse = [number(face) for face in before]
    first = len(verts)
    faces = [number(face) for face in after]
    if first != len(verts) and faces == coarse:
        raise SystemExit("the loops added vertices but no faces")

    def ids(edges):
        return sorted({index[v] for pair in edges for v in pair if v in index})

    out = {
        "format": FORMAT,
        "version": VERSION,
        "note": "Body of a CC0 base mesh below the neck, in mm in the source's own proportions. See template/PROVENANCE.md.",
        "source": {
            "title": "Blender Studio Human Base Meshes v1.4.1",
            "object": OBJECT,
            "license": "CC0",
            "url": "https://www.blender.org/download/demo-files/",
            "blendSha256": hashlib.sha256(blend.read_bytes()).hexdigest(),
        },
        "landmarks": LANDMARKS,
        "openings": {name: ids(loop) for name, loop in cut.items()},
        "rings": {name: ids(ring) for name, ring in rings.items()},
        "segments": parts,
        "verts": verts,
        "faces": faces,
    }
    if REFINE:
        added = sorted((v for v in index if index[v] >= first), key=index.get)
        out["refined"] = {
            "besideRings": REFINE,
            "from": first,
            # Each added vertex halves a rung of the source, between these two template vertices.
            "parents": [sorted(index[u] for u in v) for v in added],
            "faces": coarse,
        }
    return out, [uvs for _, _, uvs in after]


def main():
    blend = Path(sys.argv[sys.argv.index("--") + 1])
    out, _ = template(blend)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    added = len(out["refined"]["parents"]) if REFINE else 0
    loops = {k: len(v) for k, v in out["openings"].items()}
    print("BODY_EXTRACT", len(out["verts"]), "verts", len(out["faces"]), "faces", loops, "added", added)


if __name__ == "__main__":
    main()
