# SPDX-License-Identifier: GPL-3.0-or-later
"""Pictures of a body's .blend as it opens, posed by its controls, for scripts/head_swap_study.py.

Blender -b --factory-startup FILE.blend --python-exit-code 1 --python blender/review/head_swap_render.py -- REQUEST.json

REQUEST.json is {"central": [...], "size": px, "shots": [{"pose": {...}, "view": "front", "centre": [m, m, m],
"width": m, "hide": [object name, ...], "out": path}]}. The workbench engine, in each material's colour.
"""

import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sotaigen_blender import rig as R

# Where the camera sits from the centre: the wearer's front is -Y, their left +X.
VIEWS = {
    "front": (0, -1, 0),
    "side": (1, 0, 0),
    "threeQuarter": (0.7, -0.7, 0.1),
    "back": (0, 1, 0),
    "under": (0.5, -0.75, -0.45),
}


def main():
    request = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_cavity = False
    scene.display.render_aa = "16"
    scene.render.resolution_x = scene.render.resolution_y = request["size"]
    scene.render.film_transparent = False
    world = bpy.data.worlds.new("backdrop")
    scene.world = world
    world.color = (0.9, 0.9, 0.9)

    camera_data = bpy.data.cameras.new("camera")
    camera_data.type = "ORTHO"
    camera = bpy.data.objects.new("camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    rig = bpy.data.objects[R.RIG]
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.parent == rig]
    for shot in request["shots"]:
        R.set_pose(rig, shot["pose"], request["central"])
        for obj in meshes:
            obj.hide_render = obj.name in shot.get("hide", ())
        direction = Vector(VIEWS[shot["view"]]).normalized()
        centre = Vector(shot["centre"])
        camera.location = centre + direction * 3.0
        camera.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
        camera_data.ortho_scale = shot["width"]
        scene.render.filepath = shot["out"]
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=True)


main()
