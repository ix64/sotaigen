# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender background render of scripts/abdomen_study.py's actual meshes, at one scale per body.

blender -b --factory-startup -t 2 --python blender/review/abdomen_render.py
"""

import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build/abdomen-study"

for source in sorted(OUT.glob("*.npz")):
    data = np.load(source)
    vertices, faces = data["v"] / 1000, data["f"]
    torso, crotch = float(data["T"]) / 1000, float(data["crotch"]) / 1000
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.studio_light = "paint.sl"
    shading.color_type = "MATERIAL"
    shading.show_shadows = False
    shading.show_cavity = True
    shading.cavity_type = "WORLD"
    shading.cavity_valley_factor = 0.6
    shading.show_specular_highlight = False
    shading.show_object_outline = False
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    scene.display.render_aa = "16"
    mesh = bpy.data.meshes.new("Study skin")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new("Study skin", mesh)
    scene.collection.objects.link(obj)
    for face in mesh.polygons:
        face.use_smooth = True
    views = [("side", math.pi / 2)]
    if source.stem.startswith("female-long"):
        views.append(("quarter", math.pi / 4))
    for label, angle in views:
        scene.render.resolution_x, scene.render.resolution_y = 400, 600
        scene.render.resolution_percentage = 100
        cam = bpy.data.objects.new("Study camera", bpy.data.cameras.new("Study camera"))
        scene.collection.objects.link(cam)
        scene.camera = cam
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = 1.10 * torso
        direction = Vector((math.sin(angle), -math.cos(angle), 0))
        cam.location = Vector((0, 0, crotch + 0.47 * torso)) + 5 * direction
        cam.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(OUT / f"{source.stem}-{label}.png")
        bpy.ops.render.render(write_still=True)
