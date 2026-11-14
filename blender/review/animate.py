# SPDX-License-Identifier: GPL-3.0-or-later
"""Key a clip onto a body's .blend by its controls, read the keyed file back, film it. Runs inside
Blender, with --factory-startup, so with Auto Run Python Scripts off.

Blender -b --factory-startup FILE.blend --python-exit-code 1 --python blender/review/animate.py -- key REQUEST.json OUT.blend
Blender -b --factory-startup OUT.blend --python-exit-code 1 --python blender/review/animate.py -- read REQUEST.json OUT.npz
Blender -b --factory-startup OUT.blend --python-exit-code 1 --python blender/review/animate.py -- film REQUEST.json OUT.mp4

REQUEST.json is {"poses": [pose, ...], "central": [bone, ...], "fps": 60, "sample": [frame, ...]}, each
pose as sotaigen.rig.poses names one, a frame being its index in "poses" (Blender's frame is that plus one).
"""

import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sotaigen_blender import rig as R


def key(request, out):
    """Every driven control keyed on every frame, constant-free: nothing here interpolates a sample."""
    rig = bpy.data.objects[R.RIG]
    scene = bpy.context.scene
    poses = request["poses"]
    scene.render.fps = round(request["fps"])
    scene.frame_start, scene.frame_end = 1, len(poses)
    driven = sorted({name for pose in poses for name in pose if name != "rootTranslation"})
    root = rig.pose.bones[R.control("root")]
    # A captured motion is what a person did: the limits are for posing by hand, and hold it back.
    rig[R.LIMITS_SWITCH] = False
    for index, pose in enumerate(poses):
        R.set_pose(rig, pose, request["central"])
        for name in driven:
            rig.pose.bones[R.control(name)].keyframe_insert("rotation_euler", frame=index + 1)
        root.keyframe_insert("location", frame=index + 1)
    scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    print(f"keyed {len(driven)} controls on {len(poses)} frames")


def read(request, out):
    """The skin before the subdivision on the sampled frames, the limits off and on."""
    rig = bpy.data.objects[R.RIG]
    body = bpy.data.objects[R.BODY]
    scene = bpy.context.scene
    for mod in body.modifiers:
        if mod.type == "SUBSURF":
            mod.show_viewport = False
    keys = body.data.shape_keys
    result = {}
    saved = bool(rig[R.LIMITS_SWITCH])
    for on in (False, True):
        rig[R.LIMITS_SWITCH] = on
        for frame in request["sample"]:
            # Changing a custom property alone does not reach the drivers that read it.
            rig.update_tag()
            scene.frame_set(frame + 1)
            bpy.context.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            prefix = f"{'on' if on else 'off'}:{frame}:"
            result[prefix + "skin"] = R.evaluated_verts(body, depsgraph).astype(np.float32)
            evaluated = keys.evaluated_get(depsgraph)
            result[prefix + "keys"] = np.array([kb.value for kb in evaluated.key_blocks[1:]], np.float32)
    np.savez(out, **result)
    info = {
        "blender": bpy.app.version_string,
        "autoexecFail": bool(bpy.app.autoexec_fail),
        "texts": len(bpy.data.texts),
        "frames": [scene.frame_start, scene.frame_end],
        "fps": scene.render.fps,
        "limitsAsSaved": saved,
    }
    Path(out).with_suffix(".json").write_text(json.dumps(info, indent=1))


def film(request, out):
    """The file as saved, or with the limits as asked, from a camera that walks along with the hips."""
    rig = bpy.data.objects[R.RIG]
    scene = bpy.context.scene
    settings = request["film"]
    if "limits" in settings:
        rig[R.LIMITS_SWITCH] = bool(settings["limits"])
        rig.update_tag()
    height = settings["height"]

    follow = bpy.data.objects.new("follow", None)
    scene.collection.objects.link(follow)
    copy = follow.constraints.new("COPY_LOCATION")
    copy.target, copy.subtarget, copy.use_z = rig, "hips", False

    data = bpy.data.cameras.new("camera")
    data.lens = 50
    camera = bpy.data.objects.new("camera", data)
    scene.collection.objects.link(camera)
    camera.parent = follow
    distance = 2.1 * height
    angle = math.radians(settings.get("azimuth", 30))
    camera.location = (distance * math.sin(angle), -distance * math.cos(angle), 0.55 * height)
    camera.rotation_euler = (math.radians(90), 0, angle)
    scene.camera = camera

    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
    floor = bpy.context.object
    material = bpy.data.materials.new("floor")
    material.diffuse_color = (0.62, 0.62, 0.62, 1)
    floor.data.materials.append(material)

    scene.world = scene.world or bpy.data.worlds.new("world")
    scene.world.color = (0.85, 0.85, 0.85)
    scene.render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "MATERIAL"
    shading.show_shadows = True
    scene.display.render_aa = "8"
    scene.render.resolution_x, scene.render.resolution_y = 720, 900
    scene.frame_step = settings.get("step", 2)
    scene.render.fps = round(request["fps"] / scene.frame_step)
    scene.render.image_settings.media_type = "VIDEO"
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.filepath = str(out)
    bpy.ops.render.render(animation=True)


def main():
    args = sys.argv[sys.argv.index("--") + 1 :]
    mode, request, out = args[0], json.loads(Path(args[1]).read_text()), Path(args[2])
    {"key": key, "read": read, "film": film}[mode](request, out)


main()
