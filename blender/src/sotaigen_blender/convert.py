# SPDX-License-Identifier: GPL-3.0-or-later
"""Build a body's rigged .blend from the bundle sotaigen/output/blend.py writes. Runs inside Blender.

Blender -b --factory-startup --python-exit-code 1 --python sotaigen_blender/convert.py -- BUNDLE OUT.blend

BUNDLE holds input.json and input.npz. The file has the armature `rig` with its controls and deform
bones, the skin `body` with its corrective shape keys and a subdivision surface after the armature,
the head's placeholder parts as objects of their own, the controls' widgets, and what the generator
says about the body in the rig's custom property `sotaigen`. No Python goes into the file.
"""

import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sotaigen_blender import rebase, require_blender
from sotaigen_blender import rig as R

# Bone collections under the controls, by the deform bone's name without its side.
GROUPS = {
    "torso": ("root", "hips", "spine", "chest", "upperChest"),
    "head": ("neck", "head", "jaw", "Eye"),
    "arms": ("Shoulder", "UpperArm", "LowerArm", "Hand"),
    "legs": ("UpperLeg", "LowerLeg", "Foot", "Toes"),
}
FINGERS = "fingers"
# Blender's theme bone colours: the wearer's left blue, right red, the midline yellow.
COLORS = {"left": "THEME04", "right": "THEME01", "": "THEME09"}
RING_SEGMENTS = 32


def arguments():
    args = sys.argv[sys.argv.index("--") + 1 :]
    return Path(args[0]), Path(args[1])


def unsided(name):
    for side in ("left", "right"):
        if name.startswith(side):
            return side, name[len(side) :]
    return "", name


def move(obj, collection):
    for owner in list(obj.users_collection):
        owner.objects.unlink(obj)
    collection.objects.link(obj)


def faces_of(data, prefix):
    return data[f"{prefix}counts"], data[f"{prefix}corners"]


# Materials ---------------------------------------------------------------------------------------


def material(name, spec):
    """A Principled material from sotaigen.output.parts.Material's fields; what Blender has no field for stays
    in the material's custom properties."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    shader = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = spec["baseColor"]
    shader.inputs["Roughness"].default_value = spec.get("roughness", 0.75)
    shader.inputs["Metallic"].default_value = 0.0
    mat.diffuse_color = spec["baseColor"]
    alpha = spec.get("alpha", "OPAQUE")
    if alpha != "OPAQUE":
        shader.inputs["Alpha"].default_value = spec["baseColor"][3]
        mat.surface_render_method = "BLENDED" if alpha == "BLEND" else "DITHERED"
        if alpha == "MASK":
            mat.alpha_threshold = spec.get("alphaCutoff", 0.5)
    mat.use_backface_culling = not spec.get("doubleSided", False)
    for key in ("outline", "renderOrder", "alpha"):
        if key in spec:
            mat[key] = spec[key]
    return mat


# Meshes ------------------------------------------------------------------------------------------


def finish_mesh(obj, uv=None, normals=None):
    """Smooth shading, the UV map and custom normals (zero rows keep Blender's own)."""
    me = obj.data
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    if uv is not None:
        layer = me.uv_layers.new(name="UVMap")
        layer.data.foreach_set("uv", np.asarray(uv, np.float32).ravel())
    if normals is not None and np.abs(normals).sum() > 0:
        me.normals_split_custom_set_from_vertices([tuple(n) for n in np.asarray(normals, float)])
    me.update()


# Controls ----------------------------------------------------------------------------------------


def ring(name, points=None):
    """A widget mesh: a unit ring in the bone's XZ plane, round its Y axis, with extra edges."""
    me = bpy.data.meshes.new(name)
    t = np.linspace(0, 2 * math.pi, RING_SEGMENTS, endpoint=False)
    verts = [(math.cos(a), 0.0, math.sin(a)) for a in t]
    edges = [(i, (i + 1) % RING_SEGMENTS) for i in range(RING_SEGMENTS)]
    for a, b in points or ():
        edges.append((len(verts), len(verts) + 1))
        verts += [a, b]
    me.from_pydata(verts, edges, [])
    return bpy.data.objects.new(name, me)


def controls(rig, spec, collection):
    """Bone collections and colours, the widgets drawn on the deform bones, and the joint limits."""
    data = rig.data
    top = data.collections.new(R.CONTROLS)
    groups = {name: data.collections.new(name, parent=top) for name in (*GROUPS, FINGERS)}
    deform = data.collections.new(R.DEFORM)
    deform.is_visible = False
    widgets = bpy.data.collections.new(R.WIDGETS)
    collection.children.link(widgets)
    shapes = {
        "ring": ring(R.WIDGET_PREFIX + "ring"),
        # The root's ring on the ground, with an arrow to the front (the root's local Z).
        "root": ring(R.WIDGET_PREFIX + "root", [((0, 0, 1), (0, 0, 1.35)), ((0, 0, 1.35), (0.12, 0, 1.2)),
                                                 ((0, 0, 1.35), (-0.12, 0, 1.2))]),
    }
    for shape in shapes.values():
        widgets.objects.link(shape)
    rig[R.LIMITS_SWITCH] = True
    rig.property_overridable_library_set(f'["{R.LIMITS_SWITCH}"]', True)
    ui = rig.id_properties_ui(R.LIMITS_SWITCH)
    ui.update(description="The controls' range-of-motion limits (docs/motion/range-of-motion.md); off for exaggerated poses")
    for bone in data.bones:
        if bone.name.endswith(R.CONTROL_SUFFIX):
            continue
        deform.assign(bone)
    for bone_spec in spec["bones"]:
        name = bone_spec["name"]
        if not R.is_main(name):
            continue
        side, limb = unsided(name)
        group = next((g for g, members in GROUPS.items() if limb in members), FINGERS)
        bone = data.bones[R.control(name)]
        groups[group].assign(bone)
        bone.color.palette = COLORS[side]
        pb = rig.pose.bones[R.control(name)]
        pb.color.palette = COLORS[side]
        widget = spec["widgets"][name]
        pb.custom_shape = shapes["root" if name == "root" else "ring"]
        pb.custom_shape_transform = rig.pose.bones[name]
        pb.use_transform_at_custom_shape = True
        pb.use_custom_shape_bone_size = False
        radius = widget["radius"] / 1000
        pb.custom_shape_scale_xyz = (radius, radius, radius)
        pb.custom_shape_translation = (0.0, widget["along"] / 1000, 0.0)
        pb.custom_shape_wire_width = 2.0
        limits = spec["limits"].get(name)
        if limits and any(limits):
            c = pb.constraints.new("LIMIT_ROTATION")
            c.name = R.LIMIT
            c.owner_space = "LOCAL"
            c.euler_order = spec["limitOrders"][name]
            for axis, row in zip("xyz", limits, strict=True):
                if row is None:
                    continue
                setattr(c, f"use_limit_{axis}", True)
                setattr(c, f"min_{axis}", math.radians(row[0]))
                setattr(c, f"max_{axis}", math.radians(row[1]))
            fc = c.driver_add("influence")
            R.clean(fc)
            fc.driver.type = "AVERAGE"
            v = fc.driver.variables.new()
            v.name = "on"
            v.type = "SINGLE_PROP"
            v.targets[0].id_type = "OBJECT"
            v.targets[0].id = rig
            v.targets[0].data_path = f'["{R.LIMITS_SWITCH}"]'
    data.display_type = "OCTAHEDRAL"
    rig.show_in_front = True


# The file ----------------------------------------------------------------------------------------


# The bundle this package reads: sotaigen.output.blender's BUNDLE_FORMAT and BUNDLE_VERSION.
BUNDLE_FORMAT, BUNDLE_VERSION = "sotaigen/blender-bundle", 1


def check_bundle(spec):
    """Refuse a bundle of another sotaigen: another format or version, or names other than these scripts'."""
    if (spec.get("format"), spec.get("version")) != (BUNDLE_FORMAT, BUNDLE_VERSION):
        raise SystemExit(
            f"sotaigen-blender reads {BUNDLE_FORMAT} version {BUNDLE_VERSION}, not {spec.get('format')} "
            f"version {spec.get('version')}: install the sotaigen-blender that goes with this sotaigen"
        )
    if spec.get("names") != R.NAMES:
        raise SystemExit(f"sotaigen-blender names things {R.NAMES}, the bundle {spec.get('names')}: install matching packages")


def build(bundle):
    spec = json.loads((bundle / "input.json").read_text())
    check_bundle(spec)
    data = np.load(bundle / "input.npz")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    collection = bpy.data.collections.new(spec["name"])
    scene.collection.children.link(collection)
    bones = spec["bones"]
    names = [b["name"] for b in bones]
    rig, frame_error = R.armature(bones)
    move(rig, collection)
    R.couple(rig, bones, spec["drivers"])
    controls(rig, spec, collection)
    rig[R.INFO] = {**spec["info"], "restFrameError": frame_error}

    counts, corners = faces_of(data, "skin_")
    body = R.mesh_object(R.BODY, data["skin_verts"], counts, corners, rig)
    move(body, collection)
    finish_mesh(body, data["skin_uv"], data["skin_normals"])
    body.data.materials.append(material("skin", spec["skin"]["material"]))
    R.skin(body, rig, data["weights"], names)
    # Not bones, so the armature leaves them alone: where a head swap cuts and joins the skin.
    for group_name, rows in ((R.HEAD_REGION, data["head_region"]), (R.NECK_RING, data["neck_ring"])):
        body.vertex_groups.new(name=group_name).add([int(i) for i in rows], 1.0, "REPLACE")
    tables = spec["correctives"]["groups"]
    hub, _ = R.correctives(
        body,
        rig,
        tables,
        spec["correctives"]["names"],
        data["deltas"],
        lambda b, a: R.control_angle(rig, b, a, "SWING_TWIST_Y"),
    )
    mechanism = bpy.data.collections.new(R.MECHANISM)
    collection.children.link(mechanism)
    move(hub, mechanism)
    hub.hide_viewport = hub.hide_render = True
    sub = body.modifiers.new("subdivision", "SUBSURF")
    sub.levels = spec["subdivision"]["viewport"]
    sub.render_levels = spec["subdivision"]["render"]
    sub.use_custom_normals = True

    parts = bpy.data.collections.new(R.PARTS)
    collection.children.link(parts)
    for i, part in enumerate(spec["parts"]):
        prefix = f"part{i}_"
        counts, corners = faces_of(data, prefix)
        obj = R.mesh_object(part["name"], data[prefix + "verts"], counts, corners, rig)
        move(obj, parts)
        finish_mesh(
            obj,
            data[prefix + "uv"] if part["uvs"] else None,
            data[prefix + "normals"] if part["normals"] else None,
        )
        obj.data.materials.append(material(part["name"], part["material"]))
        if part["weights"]:
            R.skin(obj, rig, data[prefix + "weights"], names)
        else:
            group = obj.vertex_groups.new(name="head")
            group.add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")
            mod = obj.modifiers.new("armature", "ARMATURE")
            mod.object = rig
        if part["morphs"]:
            obj.shape_key_add(name="Basis", from_mix=False)
            basis = data[prefix + "verts"] / 1000
            for j, morph in enumerate(part["morphs"]):
                key = obj.shape_key_add(name=morph, from_mix=False)
                key.data.foreach_set("co", (basis + data[prefix + "morphs"][j] / 1000).astype(np.float32).ravel())
                # Blender gives a new key the value 1; an expression starts neutral.
                key.value = 0.0
    bpy.context.view_layer.layer_collection.children[spec["name"]].children[R.WIDGETS].exclude = True
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    if "restPose" in spec:
        rebase.apply(rig, spec["restPose"], spec["drivers"]["central"])
    # Pose with the controls at rest, so the file opens ready to animate.
    R.set_pose(rig, {}, ())
    central = spec["drivers"]["central"]
    for name, pose in spec["touchPoses"].items():
        R.pose_asset(rig, name, rebase.asset_pose(rig, pose, central) if "restPose" in spec else pose, central)
    return rig


def main():
    require_blender()
    bundle, out = arguments()
    build(bundle)
    out.parent.mkdir(parents=True, exist_ok=True)
    scratch = out.with_suffix(".tmp.blend")
    bpy.ops.wm.save_as_mainfile(filepath=str(scratch), compress=True, check_existing=False)
    scratch.replace(out)


main()
