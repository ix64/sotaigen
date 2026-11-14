"""Add the measured reference handle as a rigid, animated scene node to the pickup sample."""

import numpy as np

from sotaigen.output.glb import (
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    MM_PER_M,
    TO_GLTF,
    Chunks,
    pack_glb,
    quaternion,
    read_glb,
    to_gltf,
)
from sotaigen.rig.skinning import pose_skins, rest_matrix
from sotaigen_dev.motion.hand_contacts import HANDLE_AXIS, HANDLE_CENTER, HANDLE_RADIUS
from sotaigen_dev.motion.hand_motion import handle_pose


def cylinder_mesh():
    """Capped transverse handle in palm lengths; separate cap rims keep the ends flat."""
    around = np.arange(32) * (2 * np.pi / 32)
    radial = np.cos(around)[:, None] * np.cross((0, 0, 1), HANDLE_AXIS) + np.sin(around)[
        :, None
    ] * np.array((0, 0, 1))
    n = len(radial)
    lo, hi = (
        HANDLE_CENTER + sign * 0.92 * HANDLE_AXIS + HANDLE_RADIUS * radial for sign in (-1, 1)
    )
    verts = np.concatenate((lo, hi, lo, hi))
    normals = np.concatenate(
        (radial, radial, np.tile(-HANDLE_AXIS, (n, 1)), np.tile(HANDLE_AXIS, (n, 1)))
    )
    triangles = []
    for i in range(n):
        j = (i + 1) % n
        triangles.extend(((i, j, n + i), (j, n + j, n + i)))
    for i in range(1, n - 1):
        triangles.extend(((2 * n, 2 * n + i + 1, 2 * n + i), (3 * n, 3 * n + i, 3 * n + i + 1)))
    return verts, normals, np.array(triangles)


def add_pickup_handle(blob, bones, sample_count):
    doc, binary = read_glb(blob)
    chunks = Chunks()
    chunks.data = bytearray(binary)
    chunks.accessors = doc["accessors"]
    chunks.views = doc["bufferViews"]
    points, normals, triangles = cylinder_mesh()
    i = next(i for i, b in enumerate(bones) if b["name"] == "leftHand")
    wrist = rest_matrix(bones[i])
    length = np.linalg.norm(bones[i]["tail"] - bones[i]["head"])
    attributes = {
        "POSITION": chunks.add(
            (points * length / MM_PER_M).astype("f4"), ARRAY_BUFFER, bounds=True
        ),
        "NORMAL": chunks.add(normals.astype("f4"), ARRAY_BUFFER),
    }
    primitive = {
        "attributes": attributes,
        "indices": chunks.add(triangles.ravel().astype("u2"), ELEMENT_ARRAY_BUFFER),
        "material": len(doc["materials"]),
    }
    doc["materials"].append(
        {
            "name": "handle",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.06, 0.34, 0.32, 1],
                "metallicFactor": 0,
                "roughnessFactor": 0.8,
            },
        }
    )
    node = len(doc["nodes"])
    doc["scenes"][doc["scene"]]["nodes"].append(node)
    prop = {"name": "referenceHandle", "mesh": len(doc["meshes"])}
    doc["nodes"].append(prop)
    doc["meshes"].append({"name": "referenceHandle", "primitives": [primitive]})
    poses = [pose_skins(bones, handle_pose(t))[i] @ wrist for t in np.linspace(0, 1, sample_count)]
    animation = doc["animations"][0]
    rotations = [quaternion(TO_GLTF @ m[:3, :3]) for m in poses]
    for j in range(1, len(rotations)):
        if rotations[j] @ rotations[j - 1] < 0:
            rotations[j] = -rotations[j]
    for path, values in (
        ("translation", to_gltf([m[:3, 3] for m in poses])),
        ("rotation", rotations),
    ):
        prop[path] = values[0].tolist()
        animation["channels"].append(
            {"sampler": len(animation["samplers"]), "target": {"node": node, "path": path}}
        )
        animation["samplers"].append(
            {
                "input": animation["samplers"][0]["input"],
                "output": chunks.add(np.asarray(values, "f4")),
                "interpolation": "LINEAR",
            }
        )
    doc["buffers"][0]["byteLength"] = len(chunks.data)
    return pack_glb(doc, chunks.data)
