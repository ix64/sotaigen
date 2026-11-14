# SPDX-License-Identifier: GPL-3.0-or-later
"""Pose a body's .blend by its controls and read it back, as it opens: run with --factory-startup,
so with Auto Run Python Scripts off. Runs inside Blender.

Blender -b --factory-startup FILE.blend --python-exit-code 1 --python sotaigen_sotaigen_blender/pose.py -- POSES.json OUT

POSES.json is {"poses": {name: pose}, "central": [bone, ...], "limits": [true, false]}, each pose as
sotaigen.rig.poses names one; "actions": true poses the file's own actions too, each by its name, as playing
it on the rest pose does. Writes OUT.npz, under "<limits on or off>:<pose>:", every object's evaluated
vertices before the subdivision (mm), the skin's shape key values and the deform bones' pose
matrices; under "rest:", the same objects' own vertices and the deform bones' and controls' rest
matrices; and OUT.json, what the file holds: its drivers' flags, constraints, texts and whether
Blender refused to run anything.
"""

import json
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sotaigen_blender import require_blender
from sotaigen_blender import rig as R


def main():
    require_blender()
    args = sys.argv[sys.argv.index("--") + 1 :]
    request = json.loads(Path(args[0]).read_text())
    out = Path(args[1])
    rig = bpy.data.objects[R.RIG]
    body = bpy.data.objects[R.BODY]
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.parent == rig]
    for obj in meshes:
        for mod in obj.modifiers:
            if mod.type == "SUBSURF":
                mod.show_viewport = False
    deform = [b.name for b in rig.data.bones if b.use_deform]
    keys = body.data.shape_keys
    result = {}
    for obj in meshes:
        co = np.empty(len(obj.data.vertices) * 3, np.float32)
        obj.data.vertices.foreach_get("co", co)
        result["rest:" + obj.name] = co.reshape(-1, 3) * 1000
    result["rest:bones"] = np.array([np.array(rig.data.bones[b].matrix_local) for b in deform])
    result["rest:controls"] = np.array([np.array(rig.data.bones[R.control(b)].matrix_local) if R.control(b) in rig.data.bones else np.array(rig.data.bones[b].matrix_local) for b in deform])
    actions = sorted(bpy.data.actions, key=lambda action: action.name) if request.get("actions") else []
    for on in request["limits"]:
        rig[R.LIMITS_SWITCH] = bool(on)
        for name, pose in [*request["poses"].items(), *((action.name, action) for action in actions)]:
            if isinstance(pose, bpy.types.Action):
                R.set_pose(rig, {}, request["central"])
                played = rig.animation_data_create()
                played.action = pose
                played.action_slot = pose.slots[0]
                bpy.context.scene.frame_set(1)
            else:
                if rig.animation_data:
                    rig.animation_data.action = None
                R.set_pose(rig, pose, request["central"])
            bpy.context.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            prefix = f"{'on' if on else 'off'}:{name}:"
            for obj in meshes:
                result[prefix + obj.name] = R.evaluated_verts(obj, depsgraph).astype(np.float32)
            evaluated = keys.evaluated_get(depsgraph)
            result[prefix + "keys"] = np.array([kb.value for kb in evaluated.key_blocks[1:]], np.float32)
            result[prefix + "bones"] = np.array([np.array(rig.pose.bones[b].matrix) for b in deform])
            # The driver chain settles in one update: a second changes nothing.
            bpy.context.view_layer.update()
            again = R.evaluated_verts(body, bpy.context.evaluated_depsgraph_get()).astype(np.float32)
            assert np.array_equal(again, result[prefix + body.name]), name
    flags = R.driver_flags([rig, bpy.data.objects[R.HUB], keys, *(o.data.shape_keys for o in meshes)])
    info = {
        "blender": bpy.app.version_string,
        "autoexec": bool(bpy.context.preferences.filepaths.use_scripts_auto_execute),
        "autoexecFail": bool(bpy.app.autoexec_fail),
        "texts": len(bpy.data.texts),
        "drivers": len(flags),
        "notSimple": [f for f in flags if not f[2] or not f[3]],
        "expressions": sorted({f[4] for f in flags}),
        "constraints": {
            kind: sum(1 for pb in rig.pose.bones for c in pb.constraints if c.type == kind)
            for kind in sorted({c.type for pb in rig.pose.bones for c in pb.constraints})
        },
        "objects": sorted(o.name for o in bpy.data.objects),
        "meshes": [o.name for o in meshes],
        "deform": deform,
        "bones": [b.name for b in rig.data.bones],
        "keyNames": [kb.name for kb in keys.key_blocks[1:]],
        "poseAssets": sorted(action.name for action in bpy.data.actions if action.asset_data),
        "info": rig[R.INFO].to_dict(),
    }
    np.savez(out.with_suffix(".npz"), **result)
    out.with_suffix(".json").write_text(json.dumps(info, indent=1))


main()
