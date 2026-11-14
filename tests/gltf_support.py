"""Reading a GLB back by the glTF rules, and running the page's runtime under Node: what several test
files check exported files with."""

import json
import subprocess
from pathlib import Path

import numpy as np

from sotaigen.output.glb import COMPONENT, SHAPE, WEIGHT_STEPS
from sotaigen.rig.skinning import euler, sided

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "web" / "runtime" / "driver.mjs"
# Poses with a bone turned to a fade's limit: the hanging arm's twist at 90 and the leg's at 120. Read
# through the swing and twist split they came out a hair short of it, and drove weights of 1e-32.
LIMIT_POSES = [
    {"upperArm": [0.0, 90.0, -26.0]},
    {"upperArm": [5.0, 90.0, -24.0]},
    {"upperArm": [0.0, 90.0, -30.31]},
    {"upperLeg": [0.0, 120.0, -30.0]},
    {"upperLeg": [40.0, 120.0, 0.0]},
]


def accessor(document, data, index):
    """An accessor's elements as one row each, by the spec's layout, sparse ones included."""
    row = document["accessors"][index]
    kinds = {code: name for name, code in COMPONENT.items()}
    width = {name: size for size, name in SHAPE.items()}[row["type"]]

    def stored(view_index, kind, count, size):
        view = document["bufferViews"][view_index]
        return np.frombuffer(
            data,
            dtype=np.dtype(kinds[kind]).newbyteorder("<"),
            count=count * size,
            offset=view["byteOffset"],
        )

    if "bufferView" in row:
        values = stored(row["bufferView"], row["componentType"], row["count"], width)
        values = values.reshape(row["count"], width)
    else:
        values = np.zeros((row["count"], width), np.dtype(kinds[row["componentType"]]))
    if "sparse" in row:
        sparse = row["sparse"]
        count = sparse["count"]
        at = stored(sparse["indices"]["bufferView"], sparse["indices"]["componentType"], count, 1)
        assert (np.diff(at.astype(int)) > 0).all(), "sparse indices must strictly increase"
        values = values.copy()
        values[at] = stored(
            sparse["values"]["bufferView"], row["componentType"], count, width
        ).reshape(count, width)
    return values


def primitive(document, mesh=0):
    return document["meshes"][mesh]["primitives"][0]


def trs(node):
    x, y, z, w = node["rotation"]
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_of([x, y, z, w])
    matrix[:3, 3] = node["translation"]
    return matrix


def rotation_of(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def local_matrix(value, glide=None):
    """A value of sotaigen.rig.skeleton.sided as a 4 x 4 matrix: Euler degrees, or a matrix already. `glide` is the
    driver table's glide, for the jaw."""
    if isinstance(value, np.ndarray):
        return value
    matrix = euler(value)
    if glide is not None:
        matrix[:3, 3] = np.asarray(glide) * max(value[0], 0.0)
    return matrix


def skin_stored(document, data, pose, mesh=0, morph_weights=None):
    """A stored mesh skinned under `pose`, in mm in glTF's axes, by the glTF rules."""
    rotations = sided(pose)
    table_glide = document["extras"]["sotaigen"]["drivers"]["glide"].get("jaw")
    joints_list = document["skins"][0]["joints"]
    nodes = document["nodes"]
    parent = {child: i for i, node in enumerate(nodes) for child in node.get("children", ())}
    world = {}
    for index in joints_list:
        node = nodes[index]
        glide = table_glide if node["name"] == "jaw" else None
        local = trs(node) @ local_matrix(rotations.get(node["name"], (0.0, 0.0, 0.0)), glide)
        world[index] = world[parent[index]] @ local if index in parent else local
    inverse = accessor(document, data, document["skins"][0]["inverseBindMatrices"])
    inverse = inverse.reshape(-1, 4, 4).transpose(0, 2, 1).astype(float)
    skins = np.array([world[j] @ inverse[k] for k, j in enumerate(joints_list)])

    attributes = primitive(document, mesh)["attributes"]
    points = accessor(document, data, attributes["POSITION"]).astype(float)
    names = document["meshes"][mesh].get("extras", {}).get("targetNames", [])
    for name, weight in (morph_weights or {}).items():
        target = primitive(document, mesh)["targets"][names.index(name)]["POSITION"]
        points = points + weight * accessor(document, data, target)
    joints = accessor(document, data, attributes["JOINTS_0"]).astype(int)
    weights = accessor(document, data, attributes["WEIGHTS_0"]) / WEIGHT_STEPS
    homogeneous = np.column_stack((points, np.ones(len(points))))
    moved = np.einsum("vkij,vj->vki", skins[joints], homogeneous)
    return np.einsum("vk,vki->vi", weights, moved)[:, :3] * 1000.0


def run_driver(body, payload):
    """Run a module body under Node with `driver` imported and `input` parsed from `payload`."""
    code = (
        f"import * as driver from {json.dumps(DRIVER.as_uri())};\n"
        'let text = ""; for await (const piece of process.stdin) text += piece;\n'
        f"const input = JSON.parse(text);\n{body}"
    )
    done = subprocess.run(
        ["node", "--input-type=module", "-e", code],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)
