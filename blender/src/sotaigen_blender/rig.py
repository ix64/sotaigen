# SPDX-License-Identifier: GPL-3.0-or-later
"""The downstream rig's parts in Blender: its names, the armature's two layers, the coupling, the skin
and the corrective shape keys. Runs inside Blender, which has numpy but not body.

sotaigen_blender/convert.py builds a body's .blend from these and sotaigen_blender/pose.py poses one. Everything here is
constraints and simple-expression drivers, so a saved file poses with Auto Run Python Scripts off and carries no
Python of its own (docs/systems/blend.md).

Units are Blender's metres; the generator's millimetres are divided by 1000 on the way in. The
armature object stays at the identity, so its space is the generator's frame: X to the wearer's left,
-Y to the front, Z up.
"""

import math

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

SIDES = (("left", 1.0), ("right", -1.0))

# The names a downstream script can rely on: docs/systems/blend.md lists them for the user to decide.
CONTROL_SUFFIX = "_ctrl"
RIG = "rig"
BODY = "body"
HUB = "MCH-correctives"
WIDGET_PREFIX = "WGT-"
# Degrees: a corrective's fade is 0 from this far below its limit on, as sotaigen/corrective/drive.py's
# LIMIT_SLACK, so a control turned to the limit, read in float32, drives no weight at all.
LIMIT_SLACK = 1e-4
# Constraints, by what they do.
FOLLOW = "followControl"
RHYTHM = "shoulderRhythm"
LIFT = "forwardLift"
GLIDE = "jawGlide"
HALF = "halfSwing"
TWIST = "twistShare"
LIMIT = "jointLimit"
# Vertex groups on the skin for downstream's head swap, apart from the bones' own.
HEAD_REGION = "headRegion"
NECK_RING = "neckRing"
# Custom properties on the rig object.
LIMITS_SWITCH = "jointLimits"
INFO = "sotaigen"
# What the names above are called by in the bundle (sotaigen.output.blender.NAMES), which convert.py checks.
NAMES = {"rig": RIG, "body": BODY, "headRegion": HEAD_REGION, "neckRing": NECK_RING, "info": INFO}
# Bone collections.
CONTROLS = "controls"
DEFORM = "deform"
# Object collections, under the body's own.
PARTS = "parts"
MECHANISM = "mechanism"
WIDGETS = "widgets"


def control(name):
    """A deform bone's control. The side stays first, so that Blender's flip_name pairs left and right."""
    return name + CONTROL_SUFFIX


def deform_of(name):
    return name.removesuffix(CONTROL_SUFFIX)


def capital(name):
    return name[0].upper() + name[1:]


def is_main(name):
    """A bone a pose names: not a half bone, not a twist bone. Each has a control."""
    return not name.endswith("Half") and "Twist" not in name


def rest_matrix(bone):
    """sotaigen.rig.skinning.rest_matrix, in metres."""
    head, tail, align = (np.asarray(bone[k], float) for k in ("head", "tail", "align"))
    y = (tail - head) / np.linalg.norm(tail - head)
    z = align - y * (align @ y)
    z = z / np.linalg.norm(z)
    x = np.cross(y, z)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, head / 1000
    return m


# Drivers -----------------------------------------------------------------------------------------


def clean(fcurve):
    """A driver F-curve that passes the driver's value through untouched."""
    while fcurve.keyframe_points:
        fcurve.keyframe_points.remove(fcurve.keyframe_points[0])
    while fcurve.modifiers:
        fcurve.modifiers.remove(fcurve.modifiers[0])
    return fcurve


def driver(owner, path, index, expression, variables, kind="SCRIPTED"):
    """A driver on `owner`'s `path`; `variables` maps a name to ('prop', id, path) or
    ('transform', id, bone, channel, rotation mode). A SCRIPTED one must be a simple expression."""
    fcurve = clean(owner.driver_add(path, index) if index is not None else owner.driver_add(path))
    d = fcurve.driver
    d.type = kind
    for name, spec in variables.items():
        v = d.variables.new()
        v.name = name
        if spec[0] == "prop":
            v.type = "SINGLE_PROP"
            v.targets[0].id_type = "OBJECT"
            v.targets[0].id = spec[1]
            v.targets[0].data_path = spec[2]
        else:
            v.type = "TRANSFORMS"
            t = v.targets[0]
            t.id = spec[1]
            t.bone_target = spec[2]
            t.transform_type = spec[3]
            t.transform_space = "LOCAL_SPACE"
            t.rotation_mode = spec[4]
    if kind == "SCRIPTED":
        d.expression = expression
    return fcurve


def control_angle(rig, bone, axis, mode="XYZ"):
    """A driver variable reading a control's angle (radians) after its own constraints, the joint
    limit among them. Never the raw rotation property, which would read past the limit."""
    return ("transform", rig, control(bone), "ROT_" + axis, mode)


# The armature ------------------------------------------------------------------------------------


def armature(bones, name=RIG):
    """The rig object: every deform bone, and a control per bone a pose names, in the same hierarchy
    and rest frames. Returns the object and how far Blender's rest frames are from the runtime's."""
    data = bpy.data.armatures.new(name)
    rig = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    frames = {bone["name"]: rest_matrix(bone) for bone in bones}
    for layer in ("deform", "control"):
        for bone in bones:
            name = bone["name"]
            if layer == "control" and not is_main(name):
                continue
            eb = data.edit_bones.new(name if layer == "deform" else control(name))
            eb.head = (0, 0, 0)
            eb.tail = (0, 1, 0)
            eb.matrix = Matrix(frames[name].tolist())
            eb.length = float(np.linalg.norm(np.subtract(bone["tail"], bone["head"]))) / 1000
            eb.use_deform = layer == "deform"
            eb.use_connect = False
    for bone in bones:
        if bone["parent"]:
            data.edit_bones[bone["name"]].parent = data.edit_bones[bone["parent"]]
            if is_main(bone["name"]):
                data.edit_bones[control(bone["name"])].parent = data.edit_bones[control(bone["parent"])]
    bpy.ops.object.mode_set(mode="OBJECT")
    error = max(float(np.abs(np.array(data.bones[n].matrix_local) - m).max()) for n, m in frames.items())
    for pb in rig.pose.bones:
        pb.rotation_mode = "XYZ"
    return rig, error


def forward_lift(rig, table):
    """A forward raise also elevates the shoulder, and the arm takes as much off: control arm's Euler X,
    clipped at zero, adding to each one's Euler Z, the channel the side raise's share goes through."""
    share = table["rhythm"]["forwardLift"]
    for row in table["rhythm"]["arms"]:
        sign = row["side"]
        for owner, scale in ((row["girdle"], sign * share), (row["arm"], -sign * share)):
            c = rig.pose.bones[owner].constraints.new("TRANSFORM")
            c.name = LIFT
            c.target, c.subtarget = rig, control(row["arm"])
            c.owner_space = c.target_space = "LOCAL"
            c.map_from = c.map_to = "ROTATION"
            c.from_rotation_mode = "XYZ"
            c.to_euler_order = "XYZ"
            c.mix_mode_rot = "ADD"
            c.use_motion_extrapolate = False
            c.map_to_x_from, c.map_to_y_from, c.map_to_z_from = "Y", "Y", "X"
            c.from_min_x_rot, c.from_max_x_rot = 0.0, math.pi
            c.from_min_y_rot, c.from_max_y_rot = -math.pi, math.pi
            c.to_min_x_rot = c.to_max_x_rot = c.to_min_y_rot = c.to_max_y_rot = 0.0
            c.to_min_z_rot, c.to_max_z_rot = 0.0, scale * math.pi


def jaw_glide(rig, table):
    """A bone slides as it opens, only the jaw: the control's Euler X, clipped at zero, adding to the deform
    bone's place along sotaigen.output.glb.driver_table's glide, which is metres per degree in its rest frame."""
    for bone, glide in table["glide"].items():
        slide(rig, bone, glide)


def slide(rig, bone, glide):
    c = rig.pose.bones[bone].constraints.new("TRANSFORM")
    c.name = GLIDE
    c.target, c.subtarget = rig, control(bone)
    c.owner_space = c.target_space = "LOCAL"
    c.map_from = "ROTATION"
    c.map_to = "LOCATION"
    c.from_rotation_mode = "XYZ"
    c.mix_mode = "ADD"
    c.use_motion_extrapolate = False
    c.map_to_x_from = c.map_to_y_from = c.map_to_z_from = "X"
    c.from_min_x_rot, c.from_max_x_rot = 0.0, math.pi
    c.to_min_x = c.to_min_y = c.to_min_z = 0.0
    c.to_max_x, c.to_max_y, c.to_max_z = (v * 180.0 for v in glide)


def twist_share(rig, owner, reads, share):
    """The owner turns about its own Y by this share of the true twist of the bone it reads, by the
    Swing and Y Twist mode, and nothing else."""
    c = rig.pose.bones[owner].constraints.new("TRANSFORM")
    c.name = TWIST
    c.target, c.subtarget = rig, reads
    c.owner_space = c.target_space = "LOCAL"
    c.map_from = c.map_to = "ROTATION"
    c.from_rotation_mode = "SWING_TWIST_Y"
    c.to_euler_order = "XYZ"
    c.mix_mode_rot = "REPLACE"
    c.map_to_x_from, c.map_to_y_from, c.map_to_z_from = "X", "Y", "Z"
    c.from_min_y_rot, c.from_max_y_rot = -math.pi, math.pi
    c.to_min_y_rot, c.to_max_y_rot = -share * math.pi, share * math.pi
    c.from_min_x_rot, c.from_max_x_rot = -math.pi, math.pi
    c.from_min_z_rot, c.from_max_z_rot = -math.pi, math.pi
    c.to_min_x_rot = c.to_max_x_rot = c.to_min_z_rot = c.to_max_z_rot = 0.0
    c.use_motion_extrapolate = True


def couple(rig, bones, table):
    """The runtime's coupling (sotaigen.rig.skinning.sided) with constraints only.

    Each deform bone with a control copies it in local space. The shoulder rhythm adds the clipped
    share of the control arm's raise to the shoulder and takes it off the arm, and so does the forward
    lift (forward_lift above); the jaw slides as it opens (jaw_glide above); a half bone damped-
    tracks its limb's tail at half influence, the hand's after taking its share of the hand's roll; a
    twist bone takes its share of its driver's true twist, read by the Swing and Y Twist mode: the
    deform upper arm's, after the rhythm, and the hand control's. `table` is sotaigen.output.glb.driver_table's.
    """
    for bone in bones:
        name = bone["name"]
        if not is_main(name):
            continue
        c = rig.pose.bones[name].constraints.new("COPY_TRANSFORMS")
        c.name = FOLLOW
        c.target, c.subtarget = rig, control(name)
        c.owner_space = c.target_space = "LOCAL"
    share = table["rhythm"]["share"]
    for row in table["rhythm"]["arms"]:
        shoulder, arm, sign = row["girdle"], row["arm"], row["side"]
        # Euler X raises forward, Z raises to the side; the right side mirrors Z.
        for owner, scale in ((shoulder, share), (arm, -share)):
            c = rig.pose.bones[owner].constraints.new("TRANSFORM")
            c.name = RHYTHM
            c.target, c.subtarget = rig, control(arm)
            c.owner_space = c.target_space = "LOCAL"
            c.map_from = c.map_to = "ROTATION"
            c.from_rotation_mode = "XYZ"
            c.to_euler_order = "XYZ"
            c.mix_mode_rot = "ADD"
            c.use_motion_extrapolate = False
            c.map_to_x_from, c.map_to_y_from, c.map_to_z_from = "X", "Y", "Z"
            c.from_min_x_rot, c.from_max_x_rot = 0.0, math.pi
            c.to_min_x_rot, c.to_max_x_rot = 0.0, scale * math.pi
            c.from_min_y_rot, c.from_max_y_rot = -math.pi, math.pi
            c.to_min_y_rot = c.to_max_y_rot = 0.0
            if sign > 0:
                c.from_min_z_rot, c.from_max_z_rot = 0.0, math.pi
                c.to_min_z_rot, c.to_max_z_rot = 0.0, scale * math.pi
            else:
                c.from_min_z_rot, c.from_max_z_rot = -math.pi, 0.0
                c.to_min_z_rot, c.to_max_z_rot = -scale * math.pi, 0.0
    for row in table["half"]:
        # Its share of the roll first: the damped track then swings it from there.
        if "roll" in row:
            twist_share(rig, row["bone"], control(row["of"]), row["roll"])
        c = rig.pose.bones[row["bone"]].constraints.new("DAMPED_TRACK")
        c.name = HALF
        c.target, c.subtarget = rig, row["of"]
        c.head_tail = 1.0
        c.track_axis = "TRACK_Y"
        c.influence = 0.5
    for row in table["twist"]:
        # A twist bone of the upper arm reads it after the shoulder rhythm, as sotaigen.rig.skinning.sided
        # does; it is their parent, so this makes no cycle. One that follows another bone reads its control.
        parent = rig.data.bones[row["bone"]].parent
        reads = row["of"] if parent is not None and parent.name == row["of"] else control(row["of"])
        twist_share(rig, row["bone"], reads, row["share"])
    forward_lift(rig, table)
    jaw_glide(rig, table)


# Meshes ------------------------------------------------------------------------------------------


def mesh_object(name, verts, counts, corners, rig=None):
    """A mesh object from mm vertices and faces given as counts and corners, parented to `rig`."""
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", (np.asarray(verts, float) / 1000).astype(np.float32).ravel())
    me.loops.add(len(corners))
    me.loops.foreach_set("vertex_index", np.asarray(corners, np.int32))
    me.polygons.add(len(counts))
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    me.polygons.foreach_set("loop_start", starts.astype(np.int32))
    me.update(calc_edges=True)
    if me.validate():
        raise ValueError(f"the mesh {name} needed fixing, so its vertices might have moved")
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    if rig is not None:
        obj.parent = rig
    return obj


def skin(obj, rig, weights, names):
    """Vertex groups from a V x B weight array, columns named `names`, and the armature modifier."""
    for j, name in enumerate(names):
        column = weights[:, j]
        rows = np.flatnonzero(column)
        if not len(rows):
            continue
        group = obj.vertex_groups.new(name=name)
        for value in np.unique(column[rows]):
            group.add(rows[column[rows] == value].tolist(), float(value), "REPLACE")
    mod = obj.modifiers.new("armature", "ARMATURE")
    mod.object = rig
    mod.use_vertex_groups = True
    mod.use_bone_envelopes = False
    mod.use_deform_preserve_volume = False
    return mod


# Correctives -------------------------------------------------------------------------------------


def correctives(body, rig, groups, names, deltas, angle, hub_name=HUB):
    """Shape keys for the pose correctives, each driven by the product of its axes' tents and its gain.

    The tents and gains are custom properties on an empty, the hub: the controls' swing and twist in
    degrees as the runtime reads them (sotaigen.rig.skinning.swing_twist, through
    sotaigen.corrective.drive), each key's tent weight of the smoothstepped interval interpolation
    (sotaigen.corrective.drive.interval) and each group's gain, its fades multiplied. `groups` is
    extras.sotaigen.correctives.groups (sotaigen.corrective.recipe.Correctives.groups), `names` the
    targets in order and `deltas` their K x V x 3 offsets in mm. `angle(bone, axis)` gives the driver
    variable that reads a control's Swing and Y Twist channel, in radians (control_angle with
    SWING_TWIST_Y). Returns the hub and the keys' F-curves.
    """
    body.shape_key_add(name="Basis", from_mix=False)
    hub = bpy.data.objects.new(hub_name, None)
    bpy.context.scene.collection.objects.link(hub)
    hub.parent = rig

    def prop(name, expression, variables):
        hub[name] = 0.0
        driver(hub, f'["{name}"]', None, expression, variables)
        return ("prop", hub, f'["{name}"]')

    def tents(prefix, keys, source):
        out = {}
        for i, k in enumerate(keys):
            if i == 0:
                expression = f"1 - smoothstep({keys[0]:g}, {keys[1]:g}, a)"
            elif i == len(keys) - 1:
                expression = f"smoothstep({keys[i - 1]:g}, {k:g}, a)"
            else:
                expression = f"smoothstep({keys[i - 1]:g}, {k:g}, a) - smoothstep({k:g}, {keys[i + 1]:g}, a)"
            out[float(k)] = prop(f"{prefix}.{k:g}", expression, {"a": source})
        return out

    reads = {}

    def swing_twist(bone):
        """The control's swing and twist in degrees, as sotaigen.rig.skinning.swing_twist reads them: the Y
        twist channel as it is, the swing as its rotation vector (x, 0, z). Blender's swing channels
        are 2 asin of the swing quaternion's X and Z; `n` is the sine of half the swing angle and `k`
        the swing angle over it, in degrees."""
        if bone not in reads:
            x, z = angle(bone, "X"), angle(bone, "Z")
            n = prop(f"angle.{bone}.n", "sqrt(sin(x / 2) * sin(x / 2) + sin(z / 2) * sin(z / 2))", {"x": x, "z": z})
            k = prop(f"angle.{bone}.k", "degrees(2 * asin(n) / n if n > 1e-9 else 2)", {"n": n})
            reads[bone] = {
                "swingX": prop(f"angle.{bone}.x", "k * sin(x / 2)", {"k": k, "x": x}),
                "twist": prop(f"angle.{bone}.y", "degrees(t)", {"t": angle(bone, "Y")}),
                "swingZ": prop(f"angle.{bone}.z", "k * sin(z / 2)", {"k": k, "z": z}),
            }
        return reads[bone]

    def reading(group, index, row):
        """One axis's or fade's reading, times its scale: a right-side bone read as the left."""
        read = swing_twist(row["bone"])
        if row["read"] == "largest":
            return prop(f"{group}.{index}.largest", "max(abs(x), abs(y), abs(z))",
                        {"x": read["swingX"], "y": read["twist"], "z": read["swingZ"]})
        if row.get("scale", 1) == 1:
            return read[row["read"]]
        return prop(f"{group}.{index}.read", f"{row['scale']:g} * v", {"v": read[row["read"]]})

    expressions = {}
    for group in groups:
        if group.get("kind"):
            raise ValueError(f"a .blend can't drive the {group['kind']} table of {group['name']}")
        name = group["name"]
        axes = [tents(f"{name}.axis{i}", axis["keys"], reading(name, f"axis{i}", axis)) for i, axis in enumerate(group["axes"])]
        fades = {}
        for i, fade in enumerate(group["fades"]):
            fades[f"f{i}"] = reading(name, f"fade{i}", {**fade, "scale": 1})
        gain = prop(
            f"{name}.gain",
            " * ".join(
                f"(abs(f{i}) < {fade['limit'] - LIMIT_SLACK!r}) * (1 - smoothstep(0, {fade['limit']:g}, abs(f{i})))"
                for i, fade in enumerate(group["fades"])
            ) or "1",
            fades,
        )
        for sample, target in zip(group["samples"], group["targets"], strict=True):
            expressions[target] = (*(tent[float(key)] for tent, key in zip(axes, sample, strict=True)), gain)
    basis = np.array([v.co for v in body.data.vertices], float)
    fcurves = []
    for i, name in enumerate(names):
        key = body.shape_key_add(name=name, from_mix=False)
        key.data.foreach_set("co", (basis + np.asarray(deltas[i], float) / 1000).astype(np.float32).ravel())
        *tents_of, g = expressions[name]
        variables = {f"a{i}": tent for i, tent in enumerate(tents_of)} | {"g": g}
        fcurves.append(driver(key, "value", None, " * ".join(variables), variables))
    return hub, fcurves


# Posing and reading back -------------------------------------------------------------------------


def set_pose(rig, pose, central):
    """Turn the controls to a pose as sotaigen.rig.skinning.sided names it: a limb without a side poses both,
    the right mirrored. Every other control goes back to rest."""
    for pb in rig.pose.bones:
        if pb.name.endswith(CONTROL_SUFFIX):
            pb.rotation_mode = "XYZ"
            pb.rotation_euler = (0.0, 0.0, 0.0)
            pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            pb.location = (0.0, 0.0, 0.0)
    for name, value in pose.items():
        if name == "rootTranslation":
            continue
        one = name.startswith(("left", "right")) or name in central
        sided = (
            [(name, value)]
            if one
            else [(f"left{capital(name)}", value), (f"right{capital(name)}", (value[0], -value[1], -value[2]))]
        )
        for bone, (x, y, z) in sided:
            rig.pose.bones[control(bone)].rotation_euler = Euler(
                (math.radians(x), math.radians(y), math.radians(z)), "XYZ"
            )
    offset = Vector(pose.get("rootTranslation", (0.0, 0.0, 0.0))) / 1000
    root = rig.pose.bones[control("root")]
    root.location = root.bone.matrix_local.to_3x3().transposed() @ offset


def pose_asset(rig, name, pose, central):
    """A pose as an action marked as an asset, so that Blender's pose library shows it: one key on each
    control the pose turns. Applying it leaves the other controls as they are."""
    action = bpy.data.actions.new(name)
    action.use_fake_user = True
    slot = action.slots.new("OBJECT", rig.name)
    strip = action.layers.new("pose").strips.new(type="KEYFRAME")
    curves = strip.channelbag(slot, ensure=True).fcurves
    for bone, value in pose.items():
        one = bone.startswith(("left", "right")) or bone in central
        sided = (
            [(bone, value)]
            if one
            else [(f"left{capital(bone)}", value), (f"right{capital(bone)}", (value[0], -value[1], -value[2]))]
        )
        for each, angles in sided:
            path = f'pose.bones["{control(each)}"].rotation_euler'
            for axis, angle in enumerate(angles):
                curves.new(path, index=axis).keyframe_points.insert(1.0, math.radians(angle))
    action.asset_mark()
    return action


def evaluated_verts(obj, depsgraph):
    """The object's evaluated vertices in mm, in the rig's frame (its world matrix is the identity)."""
    evaluated = obj.evaluated_get(depsgraph)
    me = evaluated.to_mesh()
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    evaluated.to_mesh_clear()
    return co.reshape(-1, 3).astype(float) * 1000


def driver_flags(owners):
    """(data path, index, simple, valid, expression) of every driver on these IDs."""
    out = []
    for owner in owners:
        if owner is not None and owner.animation_data:
            for fc in owner.animation_data.drivers:
                d = fc.driver
                simple = d.type != "SCRIPTED" or d.is_simple_expression
                out.append((fc.data_path, fc.array_index, bool(simple), bool(d.is_valid), d.expression))
    return out
