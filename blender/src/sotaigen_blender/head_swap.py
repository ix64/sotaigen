# SPDX-License-Identifier: GPL-3.0-or-later
"""The Blender half of the head swap: read a file's meshes out, write the swapped file. Runs inside
Blender, which has numpy but not scipy, so the geometry and the weights are worked out by sotaigen/output/head_swap.py
between the two.

Blender -b --factory-startup FILE.blend --python-exit-code 1 --python sotaigen_blender/head_swap.py -- read OUT [--keys]
Blender -b --factory-startup BODY.blend --python-exit-code 1 --python sotaigen_blender/head_swap.py -- apply PLAN HEAD.blend OUT.blend
Blender -b --factory-startup OUT.blend --python-exit-code 1 --python sotaigen_blender/head_swap.py -- note INFO.json OUT.blend

`read` writes OUT.json and OUT.npz: every mesh in the view layer in world space, its faces, active UV
map, vertex groups, corner normals and (with --keys) shape keys, every armature's bones and every empty.
`apply` takes the plan (PLAN.json, PLAN.npz) and changes the body's mesh in place, so its shape keys'
drivers, vertex groups and modifiers stay what they are; nothing it adds to the file is Python.
`note` adds what the checks found to the rig's custom property and saves.
"""

import json
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sotaigen_blender import require_blender
from sotaigen_blender import rig as R


def arguments():
    return sys.argv[sys.argv.index("--") + 1 :]


# Reading -----------------------------------------------------------------------------------------


def corner_uv(me):
    """The active UV map by corner, or None."""
    layer = me.uv_layers.active
    if layer is None:
        return None
    uv = np.empty(len(me.loops) * 2, np.float32)
    layer.data.foreach_get("uv", uv)
    return uv.reshape(-1, 2)


def vertex_custom_normals(me):
    """Each vertex's authored normal, the mean of its corners', or None when the mesh has none."""
    if not me.has_custom_normals:
        return None
    corners = np.empty(len(me.loops) * 3, np.float32)
    me.corner_normals.foreach_get("vector", corners)
    verts = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", verts)
    out = np.zeros((len(me.vertices), 3))
    np.add.at(out, verts, corners.reshape(-1, 3))
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def read_mesh(obj, index, arrays, with_keys):
    """One mesh object's arrays under `obj<index>_`, its row in OUT.json as the return value."""
    me = obj.data
    prefix = f"obj{index}_"
    world = np.array(obj.matrix_world)
    co = np.empty(len(me.vertices) * 3, np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    arrays[prefix + "verts"] = (co @ world[:3, :3].T + world[:3, 3]) * 1000
    counts = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("loop_total", counts)
    corners = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", corners)
    arrays[prefix + "counts"], arrays[prefix + "corners"] = counts, corners
    material = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("material_index", material)
    arrays[prefix + "material"] = material
    uv = corner_uv(me)
    if uv is not None:
        arrays[prefix + "uv"] = uv
    authored = vertex_custom_normals(me)
    if authored is not None:
        turned = authored @ np.linalg.inv(world[:3, :3])
        arrays[prefix + "normals"] = turned / np.maximum(np.linalg.norm(turned, axis=1, keepdims=True), 1e-9)
    normals = np.empty(len(me.loops) * 3, np.float32)
    me.corner_normals.foreach_get("vector", normals)
    arrays[prefix + "cornerNormals"] = normals.reshape(-1, 3)
    rows, columns, values = [], [], []
    for vertex in me.vertices:
        for slot in vertex.groups:
            rows.append(vertex.index)
            columns.append(slot.group)
            values.append(slot.weight)
    arrays[prefix + "groupRows"] = np.array(rows, np.int32)
    arrays[prefix + "groupColumns"] = np.array(columns, np.int32)
    arrays[prefix + "groupValues"] = np.array(values, np.float64)
    keys = []
    if me.shape_keys:
        keys = [block.name for block in me.shape_keys.key_blocks[1:]]
        if with_keys:
            basis = np.empty(len(me.vertices) * 3, np.float64)
            me.shape_keys.key_blocks[0].data.foreach_get("co", basis)
            for j, block in enumerate(me.shape_keys.key_blocks[1:]):
                delta = np.empty(len(me.vertices) * 3, np.float64)
                block.data.foreach_get("co", delta)
                arrays[f"{prefix}key{j}"] = (delta - basis).reshape(-1, 3) @ world[:3, :3].T * 1000
    return {
        "name": obj.name,
        "groups": [group.name for group in obj.vertex_groups],
        "materials": [slot.material.name if slot.material else None for slot in obj.material_slots],
        "keys": keys,
        "matrixWorld": world.tolist(),
        "hasSubdivision": any(mod.type == "SUBSURF" for mod in obj.modifiers),
    }


def read(out, with_keys):
    arrays = {}
    meshes, bones, empties = [], {}, {}
    objects = list(bpy.context.view_layer.objects)
    for obj in objects:
        if obj.type == "MESH":
            meshes.append(read_mesh(obj, len(meshes), arrays, with_keys))
        elif obj.type == "ARMATURE":
            world = np.array(obj.matrix_world)
            for bone in obj.data.bones:
                # The first armature that names a bone wins; a head file should have one.
                bones.setdefault(
                    bone.name,
                    {
                        "matrix": (world @ np.array(bone.matrix_local)).tolist(),
                        "head": (world[:3, :3] @ np.array(bone.head_local) + world[:3, 3]).tolist(),
                        "tail": (world[:3, :3] @ np.array(bone.tail_local) + world[:3, 3]).tolist(),
                        "deform": bool(bone.use_deform),
                    },
                )
        elif obj.type == "EMPTY":
            empties[obj.name] = np.array(obj.matrix_world)[:3, 3].tolist()
    rig = bpy.data.objects.get(R.RIG)
    info = rig[R.INFO].to_dict() if rig is not None and R.INFO in rig else None
    other = sorted(o.name for o in objects if o.type not in ("MESH", "ARMATURE", "EMPTY"))
    summary = {"meshes": meshes, "bones": bones, "empties": empties, "info": info, "ignored": other, "texts": len(bpy.data.texts)}
    Path(out).with_suffix(".json").write_text(json.dumps(summary))
    np.savez(Path(out).with_suffix(".npz"), **arrays)


# The plan's body mesh ----------------------------------------------------------------------------


def merge_skin(body, plan, data, materials):
    """Take the head's vertices out of the body's mesh and put the new ones in, keeping the mesh's
    shape keys, whose drivers hang on it. The survivors keep their order, the new ones come last."""
    me = body.data
    removed = data["removed"]
    new_verts = data["skin_verts"] / 1000
    survivors = len(me.vertices) - len(removed)
    keys_before = [block.name for block in me.shape_keys.key_blocks] if me.shape_keys else []
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.verts[int(i)] for i in removed], context="VERTS")
    for i, vertex in enumerate(bm.verts):
        vertex.index = i
    added = [bm.verts.new(tuple(co)) for co in new_verts]
    for i, vertex in enumerate(added):
        vertex.index = survivors + i
    # A vertex takes a freed slot in the pool, so the iteration order is not the creation order.
    bm.verts.sort()
    bm.verts.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.get("UVMap") or bm.loops.layers.uv.new("UVMap")
    corners = data["face_corners"]
    uvs = data["face_uv"]
    start = 0
    for count, slot in zip(data["face_counts"].tolist(), data["face_material"].tolist(), strict=True):
        face = bm.faces.new([bm.verts[int(i)] for i in corners[start : start + count]])
        face.smooth = True
        face.material_index = materials[slot]
        for loop, uv in zip(face.loops, uvs[start : start + count], strict=True):
            loop[uv_layer].uv = tuple(uv)
        start += count
    bm.to_mesh(me)
    bm.free()
    assert [block.name for block in me.shape_keys.key_blocks] == keys_before
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    # The new vertices are at their rest place in every key: no corrective moves a new head.
    for block in me.shape_keys.key_blocks:
        moved = np.empty(len(me.vertices) * 3, np.float32)
        block.data.foreach_get("co", moved)
        moved[survivors * 3 :] = co[survivors * 3 :]
        block.data.foreach_set("co", moved)
    for j, name in enumerate(plan["skinKeys"]):
        block = body.shape_key_add(name=name, from_mix=False)
        moved = co.copy().reshape(-1, 3)
        moved[survivors:] += data[f"skin_key{j}"] / 1000
        block.data.foreach_set("co", moved.astype(np.float32).ravel())
        block.value = 0.0
    me.update()
    return survivors


def set_groups(obj, names, weights, base=0):
    """Vertex groups by bone, on the vertices from `base` on. The same value goes in as one call."""
    for j, name in enumerate(names):
        column = weights[:, j]
        rows = np.flatnonzero(column > 0)
        if not len(rows):
            continue
        group = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
        for value in np.unique(column[rows]):
            group.add((rows[column[rows] == value] + base).tolist(), float(value), "REPLACE")


def set_body_normals(me, rows):
    """Authored normals on the rows that have one; a zero row keeps Blender's own."""
    me.normals_split_custom_set_from_vertices([tuple(n) for n in rows])


# The parts ---------------------------------------------------------------------------------------


def keep_vertices(obj, keep):
    """Leave only the kept vertices; their order stays. bmesh takes the shape keys along."""
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.verts[int(i)] for i in np.flatnonzero(~keep)], context="VERTS")
    bm.to_mesh(me)
    bm.free()
    me.update()


def adopt_part(obj, row, data, rig, collection, names):
    """An appended object as a part of the body: its world transform in its mesh, parented to the rig,
    bound by the plan's weights, its other vertex groups and modifiers left alone."""
    me = obj.data = obj.data.copy()
    matrix = Matrix(row["matrix"])
    me.transform(matrix, shape_keys=True)
    if matrix.to_3x3().determinant() < 0:
        me.flip_normals()
    keep = data.get(row["prefix"] + "keep")
    if keep is not None and not keep.all():
        keep_vertices(obj, keep)
    obj.animation_data_clear()
    for constraint in list(obj.constraints):
        obj.constraints.remove(constraint)
    for modifier in list(obj.modifiers):
        if modifier.type == "ARMATURE":
            obj.modifiers.remove(modifier)
    for name in names:
        group = obj.vertex_groups.get(name)
        if group is not None:
            obj.vertex_groups.remove(group)
    obj.parent = None
    obj.matrix_world = Matrix.Identity(4)
    obj.parent = rig
    obj.matrix_parent_inverse = Matrix.Identity(4)
    for owner in list(obj.users_collection):
        owner.objects.unlink(obj)
    collection.objects.link(obj)
    set_groups(obj, names, data[row["prefix"] + "weights"])
    modifier = obj.modifiers.new("armature", "ARMATURE")
    modifier.object = rig
    modifier.use_vertex_groups = True
    modifier.use_bone_envelopes = False
    obj.modifiers.move(len(obj.modifiers) - 1, 0)
    for block in me.shape_keys.key_blocks[1:] if me.shape_keys else ():
        block.value = 0.0


# The bones ---------------------------------------------------------------------------------------


def move_bones(rig, moves):
    """The attachment bones to the new head's places. Each keeps its rest rotation, so its local axes
    stay what the contract says (docs/systems/rig.md#下颌和眼骨), and so does its control."""
    if not moves:
        return
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    for name, row in moves.items():
        for bone_name in (name, R.control(name)):
            bone = rig.data.edit_bones[bone_name]
            bone.translate(Vector(row["head"]) - bone.head)
            bone.length = row["length"]
    bpy.ops.object.mode_set(mode="OBJECT")


def scale_glide(rig, ratio):
    """The jaw's slide is a share of its bone's length (sotaigen.output.glb.driver_table)."""
    glide = rig.pose.bones["jaw"].constraints.get(R.GLIDE)
    if glide is None or ratio == 1.0:
        return
    glide.to_max_x *= ratio
    glide.to_max_y *= ratio
    glide.to_max_z *= ratio


# The file ----------------------------------------------------------------------------------------


def apply_plan(plan_path, head_path, out):
    plan = json.loads(Path(plan_path).with_suffix(".json").read_text())
    data = dict(np.load(Path(plan_path).with_suffix(".npz")))
    rig = bpy.data.objects[R.RIG]
    body = bpy.data.objects[R.BODY]
    parts = bpy.data.collections[R.PARTS]
    for name in plan["removeParts"]:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    before = set(bpy.data.objects)
    names = plan["appendOrder"]
    with bpy.data.libraries.load(str(head_path), link=False) as (_source, target):
        target.objects = list(names)
    # After the load the list holds the objects, in the order they were named.
    appended = dict(zip(names, target.objects, strict=True))

    if plan["mode"] == "whole":
        slots = {}
        skin_object = appended[plan["skinObject"]]
        for i, slot in enumerate(skin_object.material_slots):
            if slot.material is not None:
                body.data.materials.append(slot.material)
                slots[i] = len(body.data.materials) - 1
        # Row 0 of the plan's materials is the body's own skin, for the seam's strip.
        materials = [0, *[slots.get(i, 0) for i in range(len(skin_object.material_slots) or 1)]]
        survivors = merge_skin(body, plan, data, materials)
        region = body.vertex_groups[R.HEAD_REGION]
        added = list(range(survivors, len(body.data.vertices)))
        region.add(added, 1.0, "REPLACE")
        set_groups(body, plan["boneNames"], data["skin_weights"], survivors)
        set_body_normals(body.data, data["normals"])
        body.data.update()
    for row in plan["parts"]:
        adopt_part(appended[row["source"]], row, data, rig, parts, plan["boneNames"])
    if plan["mode"] == "whole" and not any(row["source"] == plan["skinObject"] for row in plan["parts"]):
        bpy.data.objects.remove(appended[plan["skinObject"]], do_unlink=True)
    # What came along with the head: its armature, as a parent, and whatever it pointed at.
    kept = {appended[row["source"]] for row in plan["parts"]}
    for obj in set(bpy.data.objects) - before - kept:
        bpy.data.objects.remove(obj, do_unlink=True)
    move_bones(rig, plan["bones"])
    scale_glide(rig, plan["glideScale"])
    info = rig[R.INFO].to_dict()
    info["head"] = plan["head"]
    info["quality"] = with_findings(info.get("quality"), plan["quality"])
    info["headSwap"] = plan["swap"]
    rig[R.INFO] = info
    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    bpy.context.view_layer.update()
    save(out)


def with_findings(quality, found):
    """A file's quality record with the head swap's findings added, and the swap among its checks."""
    quality = dict(quality or {"checked": [], "found": []})
    checked = list(quality.get("checked", []))
    return {"checked": checked + ([] if "headSwap" in checked else ["headSwap"]), "found": [*quality.get("found", []), *found]}


def note(info_path, out):
    rig = bpy.data.objects[R.RIG]
    found = json.loads(Path(info_path).read_text())
    info = rig[R.INFO].to_dict()
    info["quality"] = with_findings(info.get("quality"), found["quality"])
    info["headSwap"] = {**info.get("headSwap", {}), **found["swap"]}
    rig[R.INFO] = info
    save(out)


def save(out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scratch = out.with_suffix(".tmp.blend")
    bpy.ops.wm.save_as_mainfile(filepath=str(scratch), compress=True, check_existing=False)
    scratch.replace(out)


def main():
    require_blender()
    args = arguments()
    if args[0] == "read":
        read(args[1], "--keys" in args)
    elif args[0] == "apply":
        apply_plan(args[1], args[2], args[3])
    elif args[0] == "note":
        note(args[1], args[2])
    else:
        raise SystemExit(f"unknown step {args[0]}")


main()
