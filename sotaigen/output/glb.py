"""Write a body's finished rig as a binary glTF, to build/bodies/<name>/body.glb.

uv run sotaigen-dev export [name ...] [--sex female,male] [--group long,balanced] [--no-parts] [--contact]

The file has the head's parts and smooth normals from sotaigen.output.face, or with --no-parts the skin alone.
Its pose correctives are the sculpted ones; --contact bakes the IPC contact tables in too, minutes per
body, and writes build/bodies/<name>/body-ipc.glb instead.
A body whose wrap is broken (flipped faces, vertices past the midline, an open skin) is refused, before
any bake, with where it broke and, by sotaigen.mend, a parameter that ends it, and nothing is written; the
command then exits 1. Quality problems only warn and travel with the file, in extras.sotaigen.quality.
Both levels are sotaigen/output/blend.py's.
This module only reads what the rig stage finished; nothing in the pipeline imports it, so no
stage of sotaigen.pipeline can reach it. The file is deterministic: the same rig gives the same bytes.

What a reader can rely on, beyond plain glTF:

- The first len(rest.npy) vertices are rest.npy's, in the fixed template order. A UV seam only adds copies
  after them, listed in extras.sotaigen.splitFrom.
- The triangle list depends on the template alone, so every body of a template shares it.
- skin.joints follows the skeleton's order, so a joint index is a column of weights.npz.
- A node's local axes are the rig's: local Y along the bone, local Z to the front, local X the bend
  axis. A rig rotation written after a node's rest rotation therefore means what it means in
  sotaigen.rig.skeleton; only the root's world frame and the unit differ, and the root node stays at identity.
- Each Part is a node of its own after the body's, with its own mesh and material, on the one skin.
  Its morph targets are on its own mesh, sparse where that is smaller.
- extras.sotaigen says what the file is (format "sotaigen/glb", version) and what made it
  (sotaigen.output.metadata), and holds what a plain glTF player can't do: extras.sotaigen.drivers drives the
  helper bones and extras.sotaigen.correctives the pose correctives (web/runtime/driver.mjs, and
  tests/test_export.py for the check that the two agree). Its fields are docs/systems/export.md's GLB 约定.
  extras.sotaigen.checkPoses, with this rig's grounded crouch, isn't part of the contract.
"""

import argparse
import json
import os
import struct
import sys
import time
from functools import cache
from pathlib import Path

import numpy as np

from sotaigen import mend
from sotaigen.corrective.recipe import Correctives
from sotaigen.form.anatomy import calibrated
from sotaigen.form.head import neck_ring_order
from sotaigen.mesh import skin_normals, triangulate, vertex_normals
from sotaigen.output.face import EXPRESSIONS, HeadShape, blended_normals, bound_parts, head_parts
from sotaigen.output.head_interface import eye_place
from sotaigen.output.metadata import asset_generator, file_info
from sotaigen.output.quality import BrokenBody, body_notes, broken, glb_quality
from sotaigen.pipeline import BUILD, comma_list, rigged, select, template
from sotaigen.rig.poses import check_poses
from sotaigen.rig.skinning import (
    FORWARD_LIFT,
    HALF_BONES,
    HALF_ROLLS,
    SHOULDER_RHYTHM,
    TWISTS,
    bone_local,
    check_pose,
    glides_of,
    rest_matrix,
    sided,
)
from sotaigen.uv import export_fields

GLB_NAME = "body.glb"
CONTACT_GLB_NAME = "body-ipc.glb"
GLB_MAGIC = 0x46546C67
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963
COMPONENT = {"uint8": 5121, "uint16": 5123, "uint32": 5125, "float32": 5126}
SHAPE = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4", 16: "MAT4"}
# The rig's millimetres with Z up and -Y to the front, as glTF's metres with Y up and +Z to the
# front. X stays: glTF's +X is the wearer's left, as here. It is a rotation, so windings survive.
TO_GLTF = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
MM_PER_M = 1000.0
WEIGHT_STEPS = 65535
# A vertex moved less than this by a morph target counts as not moved, so noise costs no bytes.
MORPH_EPSILON_MM = 1e-4
# What a sparse accessor adds to the JSON over a dense one: its "sparse" object and a second view.
SPARSE_JSON_BYTES = 140


def to_gltf(points):
    """Points in the rig's frame as glTF metres."""
    return np.asarray(points) @ TO_GLTF.T / MM_PER_M


def stored_normals(normals):
    """Normals of any length in the rig's frame as glTF's unit float32 ones."""
    normals = np.asarray(normals, float)
    sizes = np.linalg.norm(normals, axis=1, keepdims=True)
    if not np.isfinite(sizes).all() or (sizes == 0).any():
        raise ValueError("normals must be finite and have a length")
    turned = (normals / sizes @ TO_GLTF.T).astype(np.float32)
    return turned / np.linalg.norm(turned, axis=1, keepdims=True)


def joints_and_weights(weights):
    """A V x B array as the four strongest (joint, weight), strongest first, in 16-bit steps.

    Each vertex's stored weights sum to exactly one, and a weight that rounds to nothing gives up
    its joint, as glTF requires.
    """
    order = np.argsort(-weights, axis=1, kind="stable")
    if np.take_along_axis(weights, order[:, 4:], axis=1).any():
        raise ValueError("a vertex has more than four influences")
    joints = order[:, :4]
    wanted = np.take_along_axis(weights, joints, axis=1) * WEIGHT_STEPS
    wanted = wanted * (WEIGHT_STEPS / wanted.sum(axis=1, keepdims=True))
    steps = np.floor(wanted).astype(np.int64)
    # Largest remainder: the steps lost to rounding go back to the weights that lost the most.
    missing = WEIGHT_STEPS - steps.sum(axis=1)
    ranked = np.argsort(steps - wanted, axis=1, kind="stable")
    for count in range(1, 5):
        rows = np.flatnonzero(missing == count)
        np.add.at(steps, (np.repeat(rows, count), ranked[rows, :count].ravel()), 1)
    joints = np.where(steps > 0, joints, 0)
    kind = np.uint8 if weights.shape[1] <= 256 else np.uint16
    return joints.astype(kind), steps.astype(np.uint16)


def quaternion(rotation):
    """A rotation matrix as glTF's (x, y, z, w), unit length with w not negative."""
    m = rotation
    trace = np.trace(m)
    if trace > 0:
        q = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1], 1 + trace])
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        q = np.zeros(4)
        q[i] = 1 + m[i, i] - m[j, j] - m[k, k]
        q[j] = m[i, j] + m[j, i]
        q[k] = m[k, i] + m[i, k]
        q[3] = m[k, j] - m[j, k]
    q = q / np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def rest_world(bone):
    """A bone's rest matrix in glTF's frame and unit, keeping its local axes."""
    rest = rest_matrix(bone)
    world = np.eye(4)
    world[:3, :3] = TO_GLTF @ rest[:3, :3]
    world[:3, 3] = TO_GLTF @ rest[:3, 3] / MM_PER_M
    return world


def driver_table(bones):
    """What sotaigen.rig.skeleton.sided needs beyond the pose, for a runtime to do the same work: each
    helper bone by its full name, and the bone it follows, so that no runtime builds a name."""
    names = {b["name"] for b in bones}
    unsided = [b["name"] for b in bones if not b["name"].startswith(("left", "right"))]
    resolved = sided(dict.fromkeys(unsided, (0.0, 0.0, 0.0)))
    sides = (("left", 1), ("right", -1))
    glides = glides_of(bones)
    return {
        # Names that sided passes through; any other unsided name means both sides, the right mirrored.
        "central": [name for name in unsided if name in resolved],
        # A share of the upper arm's raise is the girdle's, and a forward raise lifts it a little more:
        # `side` is the sign of a side raise's Euler Z on that side.
        "rhythm": {
            "share": SHOULDER_RHYTHM,
            "forwardLift": FORWARD_LIFT,
            "arms": [
                {"arm": f"{side}UpperArm", "girdle": f"{side}Shoulder", "side": sign}
                for side, sign in sides
                if f"{side}UpperArm" in names
            ],
        },
        # A half bone takes half of its limb's swing, and some of its true twist where `roll` says.
        "half": [
            {"bone": f"{side}{limb}Half", "of": f"{side}{limb}", **({"roll": HALF_ROLLS[limb]} if limb in HALF_ROLLS else {})}
            for side, _ in sides
            for limb in HALF_BONES
            if f"{side}{limb}Half" in names
        ],
        # A twist bone takes this share of its driver's true twist about local Y.
        "twist": [
            {"bone": f"{side}{limb}Twist{i}", "of": f"{side}{spec['driver']}", "share": share}
            for side, _ in sides
            for limb, spec in TWISTS.items()
            for i, (_at, share) in enumerate(spec["bones"], 1)
            if f"{side}{limb}Twist{i}" in names
        ],
        # Metres per degree of opening, in the bone's rest frame: sotaigen.rig.skinning.bone_local.
        "glide": {name: [v / MM_PER_M for v in glide] for name, glide in glides.items()},
    }


def head_attachments(base, verts, bones):
    """What a head swap may rely on, by index and name: not by position, which follows the body.

    `region` is the skin's head by vertex, first index and count, a block in every body; `neckRing`
    the ring it joins, by azimuth. Its weights have no jaw in them (sotaigen.rig.skeleton.jaw_share). The
    crown is the skin's; `eyeLine` and `eyeX` place the left eye bone by sotaigen.output.head_interface,
    the rule a head swap writes them by too.
    """
    segments = np.asarray(base["segments"])
    members = np.flatnonzero(segments == "head")
    ring = neck_ring_order(base)
    crown = int(members[np.argmax(verts[members, 2])])
    eye = next(b for b in bones if b["name"] == "leftEye")["head"] if any(b["name"] == "leftEye" for b in bones) else None
    line, across = (None, None) if eye is None else eye_place(verts[crown], verts[ring], eye)[:2]
    return {
        "bones": [name for name in ("head", "jaw", "leftEye", "rightEye") if any(b["name"] == name for b in bones)],
        "region": [int(members[0]), len(members)],
        "neckRing": [int(i) for i in ring],
        "crownVertex": crown,
        "eyeLine": line,
        "eyeX": across,
    }


def split_uv_seams(faces, vertex_count, uvs, face_uvs):
    """Copy a vertex for each further UV it has: (source of each vertex, its uv, the new faces)."""
    uvs = np.asarray(uvs, float)
    first = {}
    copies = {}
    out = []
    for face, corners in zip(faces, face_uvs, strict=True):
        row = []
        for vertex, corner in zip(face, corners, strict=True):
            if first.setdefault(vertex, corner) == corner:
                row.append(vertex)
            else:
                row.append(copies.setdefault((vertex, corner), vertex_count + len(copies)))
        out.append(row)
    source = np.concatenate((np.arange(vertex_count), [v for v, _ in copies])).astype(int)
    uv = np.zeros((len(source), 2))
    for vertex, corner in first.items():
        uv[vertex] = uvs[corner]
    for (_vertex, corner), index in copies.items():
        uv[index] = uvs[corner]
    return source, uv, out


class Chunks:
    """The binary chunk under construction, with the buffer views and accessors that describe it."""
    def __init__(self):
        self.data = bytearray()
        self.views = []
        self.accessors = []

    def add_view(self, array, target=None):
        """Store an array's bytes on a four-byte boundary; returns its buffer view's index."""
        view = {"buffer": 0, "byteOffset": len(self.data), "byteLength": array.nbytes}
        if target:
            view["target"] = target
        self.data += array.astype(array.dtype.newbyteorder("<")).tobytes()
        self.data += bytes(-len(self.data) % 4)
        self.views.append(view)
        return len(self.views) - 1

    def add(self, array, target=None, normalized=False, bounds=False):
        """Store an array with one row per element; returns its accessor index."""
        array = np.asarray(array)
        width = int(np.prod(array.shape[1:], dtype=int))
        accessor = {
            "bufferView": self.add_view(array, target),
            "componentType": COMPONENT[array.dtype.name],
            "count": len(array),
            "type": SHAPE[width],
        }
        if normalized:
            accessor["normalized"] = True
        if bounds:
            flat = array.reshape(len(array), width)
            accessor["min"] = flat.min(axis=0).tolist()
            accessor["max"] = flat.max(axis=0).tolist()
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def add_offsets(self, offsets):
        """Store a morph target's V x 3 float32 offsets, in whichever form takes fewer bytes.

        Dense stores 12 bytes a vertex. Sparse stores an index and 12 bytes for each moved vertex,
        over zeros. With none moved there is nothing to store: an accessor without a view is zeros.
        """
        moved = np.flatnonzero(offsets.any(axis=1))
        index_kind = np.uint16 if len(offsets) < 65535 else np.uint32
        sparse_bytes = moved.size * (12 + index_kind(0).nbytes) + SPARSE_JSON_BYTES
        if moved.size and sparse_bytes >= offsets.nbytes:
            return self.add(offsets, ARRAY_BUFFER, bounds=True)
        accessor = {
            "componentType": COMPONENT["float32"],
            "count": len(offsets),
            "type": "VEC3",
            "min": offsets.min(axis=0).tolist(),
            "max": offsets.max(axis=0).tolist(),
        }
        if moved.size:
            # Neither view takes a target: the spec reserves those for what a GPU reads directly.
            accessor["sparse"] = {
                "count": int(moved.size),
                "indices": {
                    "bufferView": self.add_view(moved.astype(index_kind)),
                    "componentType": COMPONENT[index_kind.__name__],
                },
                "values": {"bufferView": self.add_view(offsets[moved])},
            }
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def add_targets(self, morphs, source=None, *, normal_deltas=None):
        """Store morph targets, each going through `source` when a seam copied vertices."""
        targets = []
        for name, offsets in morphs.items():
            offsets = np.array(offsets, float)
            offsets[np.linalg.norm(offsets, axis=1) <= MORPH_EPSILON_MM] = 0.0
            if source is not None:
                offsets = offsets[source]
            # Adding zero turns the -0.0 of a negated axis into 0.0.
            target = {"POSITION": self.add_offsets(to_gltf(offsets).astype(np.float32) + 0.0)}
            if normal_deltas is not None:
                normal = normal_deltas.get(name)
                if normal is None:
                    # One mesh has the same morph attributes for each key; an empty accessor is zero.
                    normal = np.zeros_like(offsets, dtype=np.float32)
                else:
                    normal = normal[source] if source is not None else normal
                    # Directions turn to glTF axes, without the mm-to-m scaling used for positions.
                    normal = (normal @ TO_GLTF.T).astype(np.float32) + 0.0
                target["NORMAL"] = self.add_offsets(normal)
            targets.append(target)
        return targets


def pack_glb(document, data):
    """A glTF document and its binary chunk as the bytes of a GLB."""
    text = json.dumps(document, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    data = bytes(data) + bytes(-len(data) % 4)
    total = 12 + 8 + len(text) + 8 + len(data)
    return b"".join(
        (
            struct.pack("<III", GLB_MAGIC, 2, total),
            struct.pack("<II", len(text), JSON_CHUNK),
            text,
            struct.pack("<II", len(data), BIN_CHUNK),
            data,
        )
    )


def read_glb(blob):
    """A GLB's bytes as (document, binary chunk); raises ValueError on anything malformed."""
    magic, version, total = struct.unpack_from("<III", blob)
    if magic != GLB_MAGIC or version != 2 or total != len(blob):
        raise ValueError("not a version 2 GLB of this length")
    size, kind = struct.unpack_from("<II", blob, 12)
    if kind != JSON_CHUNK:
        raise ValueError("the first chunk is not JSON")
    document = json.loads(blob[20 : 20 + size])
    start = 20 + size
    size, kind = struct.unpack_from("<II", blob, start)
    if kind != BIN_CHUNK:
        raise ValueError("the second chunk is not binary")
    return document, blob[start + 8 : start + 8 + size]


def material_row(name, material):
    """A Part's Material as glTF's, lit like the skin."""
    row = {
        "name": name,
        "pbrMetallicRoughness": {
            "baseColorFactor": [float(c) for c in material.base_color],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.75,
        },
    }
    if material.alpha != "OPAQUE":
        row["alphaMode"] = material.alpha
    if material.alpha == "MASK":
        row["alphaCutoff"] = float(material.alpha_cutoff)
    if material.double_sided:
        row["doubleSided"] = True
    # glTF has neither. A material's draw order is where VRM's renderQueueOffsetNumber lives too.
    info = {}
    if material.render_order:
        info["renderOrder"] = int(material.render_order)
    if material.outline:
        info["outline"] = True
    if info:
        row["extras"] = {"sotaigen": info}
    return row


def part_weights(part, names):
    """A part's V x B weights: its own, or all of them on the head bone."""
    if part.weights is not None:
        if part.weights.shape[1] != len(names):
            raise ValueError(f"part {part.name}: weights must have {len(names)} columns")
        return part.weights
    if "head" not in names:
        raise ValueError(f"part {part.name}: no weights, and the rig has no head bone")
    weights = np.zeros((len(part.verts), len(names)))
    weights[:, names.index("head")] = 1.0
    return weights


def part_mesh(chunks, part, names, material):
    """Store a part, returning its glTF mesh. Its triangle list doesn't depend on its shape."""
    points = to_gltf(part.verts)
    # A list of zeros makes every quad tie, so each one is split from its first corner.
    triangles = triangulate(part.faces, np.zeros_like(points))
    if part.normals is not None:
        normals = stored_normals(part.normals)
    else:
        normals = vertex_normals(points, triangles)
    joints, steps = joints_and_weights(part_weights(part, names))
    attributes = {
        "POSITION": chunks.add(points.astype(np.float32), ARRAY_BUFFER, bounds=True),
        "NORMAL": chunks.add(normals, ARRAY_BUFFER),
    }
    if part.uvs is not None:
        attributes["TEXCOORD_0"] = chunks.add(part.uvs.astype(np.float32), ARRAY_BUFFER)
    attributes["JOINTS_0"] = chunks.add(joints, ARRAY_BUFFER)
    attributes["WEIGHTS_0"] = chunks.add(steps, ARRAY_BUFFER, normalized=True)
    index_kind = np.uint16 if len(points) < 65535 else np.uint32
    primitive = {
        "attributes": attributes,
        "indices": chunks.add(
            triangles.astype(index_kind).ravel()[:, None], ELEMENT_ARRAY_BUFFER, bounds=True
        ),
        "material": material,
    }
    mesh = {"name": part.name, "primitives": [primitive]}
    if part.morphs:
        primitive["targets"] = chunks.add_targets(part.morphs)
        mesh["weights"] = [0.0] * len(part.morphs)
        mesh["extras"] = {"targetNames": list(part.morphs)}
    return mesh


def baked_motion(chunks, bones, name, fps, poses, *, body_morph_names=(), correctives=None):
    """Absolute local glTF transforms at the sample times, including helper bone rotations."""
    if not poses or not np.isfinite(fps) or fps <= 0:
        raise ValueError("a motion needs at least one frame and a positive frame rate")
    times = chunks.add((np.arange(len(poses)) / fps).astype(np.float32), bounds=True)
    channels, samplers = [], []

    def channel(index, path, values):
        sampler = len(samplers)
        output = chunks.add(np.asarray(values, dtype=np.float32))
        samplers.append({"input": times, "output": output, "interpolation": "LINEAR"})
        channels.append({"sampler": sampler, "target": {"node": index, "path": path}})

    frames = []
    for pose in poses:
        check_pose(pose, bones)
        frames.append(sided(pose))
    world = {b["name"]: rest_world(b) for b in bones}
    for index, bone in enumerate(bones):
        key, parent = bone["name"], bone["parent"]
        rest = world[key]
        if parent is None:
            track = to_gltf([pose.get("rootTranslation", (0, 0, 0)) for pose in poses])
            channel(index, "translation", rest[:3, 3] + track)
        else:
            rest = np.linalg.inv(world[parent]) @ rest
        if not any(key in frame for frame in frames):
            continue
        rotations, places = [], []
        for frame in frames:
            local = bone_local(bone, frame.get(key, (0.0, 0.0, 0.0)), glides_of(bones).get(key))
            q = quaternion(rest[:3, :3] @ local[:3, :3])
            if rotations and q @ rotations[-1] < 0:
                q = -q
            rotations.append(q)
            places.append(rest[:3, 3] + rest[:3, :3] @ local[:3, 3] / MM_PER_M)
        channel(index, "rotation", rotations)
        if key in glides_of(bones):
            channel(index, "translation", places)
    if body_morph_names:
        # glTF requires one SCALAR per morph target per frame, in mesh order.
        # These skeletal poses leave unrelated body expressions at their neutral weights.
        values = []
        for pose in poses:
            driven = correctives.pose_weights(pose)
            values.extend(driven.get(key, 0.0) for key in body_morph_names)
        channel(len(bones), "weights", values)
    return {"name": name, "samplers": samplers, "channels": channels}


SKIN_MATERIAL = {
    "name": "skin",
    "pbrMetallicRoughness": {
        "baseColorFactor": [0.8, 0.72, 0.68, 1.0],
        "metallicFactor": 0.0,
        "roughnessFactor": 0.75,
    },
}


def skin_layout(base, verts, normals):
    """(which stored vertex each original is, UVs or None, stored triangles, stored unit normals)."""
    if normals is None:
        normals = skin_normals(base, verts)
    if np.shape(normals) != (len(verts), 3):
        raise ValueError(f"normals must be {len(verts)} x 3, not {np.shape(normals)}")
    faces = base["faces"]
    template_verts = np.asarray(base["verts"], float)
    uv = None
    source = np.arange(len(verts))
    if "uvs" in base:
        source, uv, faces = split_uv_seams(faces, len(verts), base["uvs"], base["faceUvs"])
        template_verts = template_verts[source]
    return source, uv, triangulate(faces, template_verts), stored_normals(normals)


def corrective_normal_inputs(base, verts, normals=None):
    """(normals, triangles) as body_mesh gives them to Correctives.normal_targets, for making
    those normal deltas elsewhere."""
    source, _, triangles, unit_normals = skin_layout(base, verts, normals)
    return unit_normals @ TO_GLTF, source[triangles]


def body_mesh(chunks, base, verts, bones, weights, normals, morphs, correctives):
    """The skin's mesh, with its morph targets and the correctives' among them.

    Returns the mesh, which stored vertex each original is (seam copies follow the originals) and
    the names of the body's morph targets in their stored order.
    """
    source, uv, triangles, unit_normals = skin_layout(base, verts, normals)
    points = to_gltf(verts)[source]
    joints, steps = joints_and_weights(np.asarray(weights)[source])
    attributes = {
        "POSITION": chunks.add(points.astype(np.float32), ARRAY_BUFFER, bounds=True),
        "NORMAL": chunks.add(unit_normals[source], ARRAY_BUFFER),
    }
    if uv is not None:
        attributes["TEXCOORD_0"] = chunks.add(uv.astype(np.float32), ARRAY_BUFFER)
    attributes["JOINTS_0"] = chunks.add(joints, ARRAY_BUFFER)
    attributes["WEIGHTS_0"] = chunks.add(steps, ARRAY_BUFFER, normalized=True)
    index_kind = np.uint16 if len(source) < 65535 else np.uint32
    primitive = {
        "attributes": attributes,
        "indices": chunks.add(
            triangles.astype(index_kind).ravel()[:, None], ELEMENT_ARRAY_BUFFER, bounds=True
        ),
        "material": 0,
    }
    mesh = {"name": "body", "primitives": [primitive]}
    normal_deltas = None
    if correctives is not None:
        driven = correctives.targets
        collisions = driven.keys() & (morphs or {}).keys()
        if collisions:
            raise ValueError(f"body morph names reserved for correctives: {sorted(collisions)}")
        normal_deltas = correctives.normal_targets(
            base, verts, bones, weights, unit_normals @ TO_GLTF, triangles=source[triangles]
        )
        morphs = {**(morphs or {}), **driven}
    if morphs:
        primitive["targets"] = chunks.add_targets(morphs, source, normal_deltas=normal_deltas)
        mesh["weights"] = [0.0] * len(morphs)
        mesh["extras"] = {"targetNames": list(morphs)}
    return mesh, source, tuple(morphs or ())


def skeleton_nodes(chunks, bones):
    """A node per bone in the bones' order, the roots among them and the inverse bind matrices."""
    world = {bone["name"]: rest_world(bone) for bone in bones}
    nodes = []
    for bone in bones:
        local = world[bone["name"]]
        if bone["parent"] is not None:
            local = np.linalg.inv(world[bone["parent"]]) @ local
        children = [i for i, other in enumerate(bones) if other["parent"] == bone["name"]]
        node = {
            "name": bone["name"],
            "translation": local[:3, 3].tolist(),
            "rotation": quaternion(local[:3, :3]).tolist(),
        }
        if children:
            node["children"] = children
        nodes.append(node)
    roots = [i for i, bone in enumerate(bones) if bone["parent"] is None]
    inverse = np.array([np.linalg.inv(world[bone["name"]]) for bone in bones])
    inverse[:, 3] = (0.0, 0.0, 0.0, 1.0)
    # glTF stores matrices by column.
    return nodes, roots, chunks.add(inverse.transpose(0, 2, 1).astype(np.float32))


def glb_bytes(
    base,
    verts,
    bones,
    weights,
    extras=None,
    morphs=None,
    parts=(),
    normals=None,
    motions=(),
    *,
    check_pose_map=None,
    correctives=None,
):
    """The GLB of one rig, with its parts if it has any.

    `base` is the extended template. UVs come along when it has `uvs` (U x 2, in glTF's convention
    with V down) and `faceUvs` (a list of UV indices per corner, parallel to `faces`). `morphs` maps
    a name to a V x 3 array of vertex offsets in the rig's frame, in mm. `parts` is a list of Part.
    `normals` replaces the skin's skin_normals: V x 3 in the rig's frame, in the order of `verts`.
    Either way each seam copy takes its original's. A rig with its own pose encoder may
    supply the already encoded semantic check poses, independently of baked animation channels.
    `correctives` (sotaigen.corrective.recipe.Correctives) adds the pose correctives as morph targets, with
    their driver tables and their weights in each baked motion; a generated body takes
    Correctives.production.
    """
    names = [bone["name"] for bone in bones]
    taken = {"body", *names}
    for part in parts:
        if part.name in taken:
            raise ValueError(f"part name {part.name!r} is already taken")
        taken.add(part.name)
    chunks = Chunks()
    mesh, source, morph_names = body_mesh(
        chunks, base, verts, bones, weights, normals, morphs, correctives
    )
    nodes, roots, inverse_bind = skeleton_nodes(chunks, bones)
    nodes.append({"name": "body", "mesh": 0, "skin": 0})
    body_node = len(nodes) - 1

    # After everything the body stores, so that adding parts leaves the body's bytes where they were.
    meshes = [mesh]
    materials = [SKIN_MATERIAL]
    for part in parts:
        meshes.append(part_mesh(chunks, part, names, len(materials)))
        materials.append(material_row(part.name, part.material))
        nodes.append({"name": part.name, "mesh": len(meshes) - 1, "skin": 0})

    info = {**(extras or {}), "drivers": driver_table(bones)}
    if correctives is not None:
        info |= correctives.metadata()
    # The expressions the parts carry, by their VRM 1.0 preset names: the morph names that are a contract.
    carried = {name for part in parts for name in (part.morphs or {})}
    info["expressions"] = [name for name in EXPRESSIONS if name in carried]
    if "jaw" in names:
        info["head"] = head_attachments(base, verts, bones)
    if len(source) > len(verts):
        info["splitFrom"] = source[len(verts) :].tolist()
    # Not part of the contract: what this version checks a body in, for a viewer to show.
    info["checkPoses"] = check_poses(base, verts, bones, weights) if check_pose_map is None else check_pose_map
    animations = [
        baked_motion(
            chunks,
            bones,
            *clip,
            body_morph_names=morph_names if correctives is not None else (),
            correctives=correctives,
        )
        for clip in motions
    ]
    document = {
        "asset": {"version": "2.0", "generator": asset_generator()},
        "scene": 0,
        "scenes": [{"nodes": [*roots, *range(body_node, len(nodes))]}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "skins": [
            {
                "inverseBindMatrices": inverse_bind,
                "skeleton": roots[0],
                "joints": list(range(len(bones))),
            }
        ],
        "accessors": chunks.accessors,
        "bufferViews": chunks.views,
        "buffers": [{"byteLength": len(chunks.data)}],
        "extras": {"sotaigen": info},
    }
    if animations:
        document["animations"] = animations
    return pack_glb(document, chunks.data)


def glb_path(name, contact=False):
    return BUILD / "bodies" / name / (CONTACT_GLB_NAME if contact else GLB_NAME)


@cache
def export_base():
    """The template with the head and the UVs the exporter takes. Shared: don't change it."""
    base = template()
    return {**base, **export_fields(base["faces"])}


def write_glb(body, path, *, contact=False, parts=None, normals=None):
    """Write a Body's GLB at `path` and return the path; BrokenBody, before anything else, for a broken mesh.

    By default the file has the head's placeholder parts and smooth normals (sotaigen.output.face);
    `parts` gives others (a list of Part, empty for none) with `normals` for the skin, or its own when
    None. Its pose correctives are the sculpted ones, or with `contact` the IPC contact tables, which
    need sotaigen[contact] and take minutes a body (Correctives.production). A body read from the build
    (body.name) says its name and rig's fingerprint, and its quality list counts the check poses' collapsed
    faces; any other says the parameters it started from. The bytes depend on the body alone.
    """
    path = Path(path)
    base = export_base()
    stopped = broken(body.wrap, base["faces"])
    if stopped:
        raise BrokenBody(body.name or path.stem, stopped)
    if parts is None:
        head = HeadShape(body.controls)
        parts = bound_parts(head, head_parts(head), body)
        normals = blended_normals(base, body.verts, head)
    correctives = Correctives.production(
        base, body.verts, body.bones, body.weights, field=calibrated(body.controls)[0], contact=contact
    )
    extras = file_info(body.spec)
    notes = body.notes if body.notes is not None else body_notes(body.spec, body.controls)
    # A typical or test body's file has its check poses' collapsed faces, whichever way it was made.
    extras["quality"] = glb_quality(
        body.wrap, base, body.verts, body.bones, body.weights, correctives, posed="name" in extras, found=notes
    )
    blob = glb_bytes(
        base, body.verts, body.bones, body.weights, extras, parts=parts, normals=normals, correctives=correctives
    )
    scratch = path.with_suffix(".tmp")
    scratch.write_bytes(blob)
    os.replace(scratch, path)
    return path


def export_body(name, parts=(), normals=None, *, contact=False):
    """Write a body's GLB from the build, by default the skin alone, to glb_path(name, contact)."""
    return write_glb(rigged(name), glb_path(name, contact), contact=contact, parts=list(parts), normals=normals)


def export_with_face(name, contact=False):
    """Write a body's whole GLB from the build, with the head's parts and smooth normals."""
    return write_glb(rigged(name), glb_path(name, contact), contact=contact)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sotaigen-dev export", description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", metavar="name", help="default: all of them")
    parser.add_argument("--sex", type=comma_list, help="only these sexes, as female,male")
    parser.add_argument("--group", type=comma_list, help="only these groups, as long,balanced")
    parser.add_argument(
        "--no-parts",
        dest="write",
        action="store_const",
        const=export_body,
        default=export_with_face,
        help="the skin alone, without the head's parts and smooth normals",
    )
    parser.add_argument(
        "--contact",
        action="store_true",
        help="bake the IPC contact tables in, minutes per body, and write body-ipc.glb",
    )
    args = parser.parse_args(argv)
    try:
        names = select(args.names, args.sex, args.group)
    except ValueError as error:
        parser.error(str(error))

    failed = 0
    for name in names:
        start = time.perf_counter()
        try:
            path = args.write(name, contact=args.contact)
        except BrokenBody as error:
            print(error, file=sys.stderr)
            mend.tell_body(name)
            failed += 1
            continue
        seconds = time.perf_counter() - start
        print(f"{path}: {path.stat().st_size} bytes, {seconds:.2f} s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
