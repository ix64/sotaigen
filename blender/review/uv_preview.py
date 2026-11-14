# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender side of the UV sample: wrapped bodies under a colour grid, to see how the atlas lands.

Each body stands twice in one picture, front and back; a second picture looks close at the seams.

Blender -b --factory-startup --python-exit-code 1 --python blender/review/uv_preview.py \\
    -- <out prefix> <build dir> <body> [body ...]
"""

import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Euler, Vector

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sotaigen.form.head import extended
from sotaigen.template import load_template
from sotaigen.uv import load_uv

MM = 0.001
# Where the second picture looks, on the last body: name, centre as a share of height, width in mm.
CLOSE = {
    "headBack": ((0.0, 0.0, 0.90), (0, 0, 180), 340),
    "neck": ((0.0, 0.0, 0.84), (0, 0, 0), 180),
    "hand": (None, (0, 0, 0), 200),
    "foot": ((0.09, -0.04, 0.05), (0, 0, 0), 240),
}


def grid_material():
    image = bpy.data.images.new("grid", 2048, 2048)
    image.generated_type = "COLOR_GRID"
    material = bpy.data.materials.new("grid")
    material.use_nodes = True
    node = material.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = image
    material.node_tree.nodes.active = node
    return material


def body_object(name, points, faces, uv, material):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata((points * MM).tolist(), [], faces)
    mesh.polygons.foreach_set("use_smooth", [True] * len(faces))
    mesh.uv_layers.new(name="UVMap")
    mesh.uv_layers[0].data.foreach_set("uv", uv.ravel())
    mesh.materials.append(material)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def shoot(scene, camera, centre, width_mm, size, path):
    scene.render.resolution_x, scene.render.resolution_y = size
    camera.data.ortho_scale = width_mm * MM
    camera.location = Vector(centre) + Vector((0.0, -4.0, 0.0))
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main():
    args = sys.argv[sys.argv.index("--") + 1 :]
    prefix, build, names = args[0], Path(args[1]), args[2:]
    base = load_template()
    full = extended(base)
    faces = full["faces"]
    uv = load_uv(faces)

    scene = bpy.context.scene
    for leftover in list(scene.objects):
        bpy.data.objects.remove(leftover)
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.display.render_aa = "16"
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("world")
    scene.world.color = (1, 1, 1)
    material = grid_material()
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    camera.data.type = "ORTHO"
    camera.data.clip_end = 20
    camera.rotation_euler = Euler((np.pi / 2, 0, 0))
    scene.collection.objects.link(camera)
    scene.camera = camera

    spacing = 1.0
    x = 0.0
    last = None
    tallest = 0.0
    for name in names:
        points = np.load(build / name / "wrapped.npy")
        tallest = max(tallest, points[:, 2].max() * MM)
        for turn in (0.0, np.pi):
            obj = body_object(f"{name}{turn}", points, faces, uv, material)
            obj.rotation_euler = Euler((0, 0, turn))
            obj.location = (x, 0, 0)
            x += spacing
            last = obj, points, turn
    scene.view_layers[0].update()
    shoot(
        scene,
        camera,
        ((x - spacing) / 2, 0, tallest / 2),
        (x + 0.2) / MM,
        (1800, int(1800 * (tallest + 0.1) / (x + 0.2))),
        f"{prefix}_bodies.png",
    )

    obj, points, _ = last
    height = points[:, 2].max()
    regions = np.array(full["regions"])
    for shot, (share, rotation, width) in CLOSE.items():
        obj.rotation_euler = Euler(np.radians(rotation))
        if share is None:
            hand = points[(regions == "hand") & (points[:, 0] > 0)].mean(0) * MM
            offset = hand
        else:
            offset = Vector((share[0] * height, share[1] * height, share[2] * height)) * MM
        scene.view_layers[0].update()
        target = Vector(obj.location) + obj.matrix_world.to_3x3() @ Vector(offset)
        shoot(scene, camera, target, width, (800, 800), f"{prefix}_{shot}.png")


if __name__ == "__main__":
    main()
