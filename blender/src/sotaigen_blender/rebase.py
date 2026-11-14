# SPDX-License-Identifier: GPL-3.0-or-later
"""Make another pose a built file's rest pose. Runs inside Blender, on what sotaigen_blender/convert.py built.

The rig's bones are renamed as mechanism and go on working as they did, in the A pose's frames: every
constraint, driver and joint limit keeps reading the angles it was made to read. Above them come new
controls with the pose as their rest, each adding its turn to its mechanism control's held pose, and
below them new deform bones with the pose as their rest, each copying its mechanism bone. So the bones
reach what they reached before for any pose, zero on the new controls being the new rest; only the
skin needs thought, because a vertex between two bones can't be skinned back the way it came.

The skin's new basis is the posed skin. Linear blend skinning from it differs from skinning from the A
pose on every vertex with more than one bone, by a term that vanishes at the new rest (docs/systems/
blend.md). So the correctives are carried into the new basis by the skin's own blend at the pose, and
each key driven at the pose has that value taken off its driver, baked into the basis instead. One new
key per turned control, `rest.<bone>`, then takes the difference out: in full when the control is back
at its A pose, none at the rest, and linearly along the way and past it. Together they give the A pose
back exactly; poses away from both rests differ by millimetres near joints, measured in that document.

A pose asset comes out of the A pose's angles the same way: asset_pose turns them into the new controls'.

Nothing here names a bone: it reads the armature and the rest pose it is given.
"""

import math

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

from sotaigen_blender import rig as R

MECHANISM_PREFIX = "MCH-"
# The mechanism's own control is the deform bone's mechanism name with this after it, so that
# R.set_pose, which goes by the controls' suffix, leaves it holding the pose.
MECHANISM_CONTROL = ".control"
BONES_MECHANISM = "mechanism"
# Constraints this step adds.
HOLD = "restOffset"
HOLD_LOCATION = "restOffsetLocation"
COPY = "followMechanism"
KEY_PREFIX = "rest."
# How far past each rest a `rest.<bone>` key's reading goes: three times the way between them.
KEY_RANGE = 3.0
# A key's driven value at the pose under this isn't worth an expression: 1e-9 of a 50 mm delta.
QUIET = 1e-9
# A key that moves nothing further than this many metres isn't written.
STILL = 1e-9
# The skin's and parts' error at either rest that this step refuses to save, in mm: Blender's precision
# leaves about 0.002 (tests/test_rest_pose.py).
REFUSE_MM = 0.003


def mechanism(name):
    return MECHANISM_PREFIX + name


def mechanism_control(name):
    return MECHANISM_PREFIX + name + MECHANISM_CONTROL


def held_pose(rig, rest, central):
    """Turn the controls to the rest pose, its translation on the children of the control with no parent,
    so that the root itself stays on the ground. Returns ({control's deform name: its Euler angles}, for
    those it turned, and {the same: its location}, for those it moved)."""
    pose = dict(rest["pose"])
    offset = Vector(pose.pop("rootTranslation", (0.0, 0.0, 0.0))) / 1000
    R.set_pose(rig, pose, central)
    moved = {}
    if offset.length:
        for pb in rig.pose.bones:
            if pb.name.endswith(R.CONTROL_SUFFIX) and pb.parent and pb.parent.parent is None:
                pb.location = pb.bone.matrix_local.to_3x3().transposed() @ offset
                moved[R.deform_of(pb.name)] = pb.location.copy()
    turned = {}
    for bone in rig.data.bones:
        if bone.name.endswith(R.CONTROL_SUFFIX):
            angles = rig.pose.bones[bone.name].rotation_euler
            if any(abs(a) > 0 for a in angles):
                turned[R.deform_of(bone.name)] = tuple(angles)
    return turned, moved


def group_weights(obj, deform):
    """The object's V x B weights by its vertex groups, normalised as the armature modifier does."""
    column = {g.index: deform.index(g.name) for g in obj.vertex_groups if g.name in deform}
    out = np.zeros((len(obj.data.vertices), len(deform)))
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group in column:
                out[v.index, column[g.group]] = g.weight
    return out / np.maximum(out.sum(1, keepdims=True), 1e-12)


class Snapshot:
    """What the rig gives at the controls' current values: every mesh's vertices in metres, the skin's
    key values and the deform bones' matrices."""

    def __init__(self, rig, meshes, body, deform):
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        self.verts = {obj.name: R.evaluated_verts(obj, depsgraph) / 1000 for obj in meshes}
        keys = body.data.shape_keys.evaluated_get(depsgraph)
        self.keys = np.array([kb.value for kb in keys.key_blocks[1:]], float)
        self.bones = np.array([np.array(rig.pose.bones[name].matrix, float) for name in deform])


def orthonormal(matrix):
    """The rotation nearest a posed frame, which float products have left a millionth off one."""
    u, _, vt = np.linalg.svd(matrix[:3, :3])
    out = matrix.copy()
    out[:3, :3] = u @ vt
    return out


def blended(weights, skins):
    """Each vertex's 4 x 4 skin from its bones' (V x B, B x 4 x 4)."""
    return np.einsum("vb,bij->vij", weights, skins)


def skinned(blend, verts):
    return (blend[:, :3, :3] @ verts[:, :, None])[:, :, 0] + blend[:, :3, 3]


def unskinned(blend, posed):
    """The vertices that `blend` takes to `posed`."""
    return np.linalg.solve(blend[:, :3, :3], (posed - blend[:, :3, 3])[:, :, None])[:, :, 0]


def key_coordinates(key):
    co = np.empty(len(key.data) * 3, np.float32)
    key.data.foreach_get("co", co)
    return co.reshape(-1, 3).astype(float)


def set_key(key, co):
    key.data.foreach_set("co", np.asarray(co, np.float32).ravel())


def rename_as_mechanism(rig, meshes, deform, controls):
    """The bones keep working under their new names: Blender carries the constraints' and drivers' targets
    and the vertex groups along, and the groups go back to the names the new deform bones take."""
    for name in deform:
        rig.data.bones[name].name = mechanism(name)
    for name in controls:
        rig.data.bones[R.control(name)].name = mechanism_control(name)
    for obj in meshes:
        for group in obj.vertex_groups:
            if group.name.startswith(MECHANISM_PREFIX):
                group.name = group.name.removeprefix(MECHANISM_PREFIX)


def new_layers(rig, deform, controls, frames):
    """The new deform bones and controls with the pose as their rest, in the same hierarchy. A control
    sits on its deform bone, as it does at the A pose."""
    data = rig.data
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    edit = data.edit_bones
    parents = {name: (edit[mechanism(name)].parent.name if edit[mechanism(name)].parent else None) for name in deform}
    lengths = {name: edit[mechanism(name)].length for name in deform}
    for name in deform:
        edit[mechanism(name)].use_deform = False
    for own, keep in ((lambda n: n, deform), (R.control, controls)):
        for name in keep:
            bone = edit.new(own(name))
            bone.head, bone.tail = (0, 0, 0), (0, 1, 0)
            bone.matrix = Matrix(frames[name].tolist())
            bone.length = lengths[name]
            bone.use_deform = own(name) == name
            bone.use_connect = False
    for name in deform:
        if parents[name]:
            parent = parents[name].removeprefix(MECHANISM_PREFIX)
            edit[name].parent = edit[parent]
            if name in controls:
                edit[R.control(name)].parent = edit[R.control(parent)]
    bpy.ops.object.mode_set(mode="OBJECT")
    for name in (*deform, *map(R.control, controls)):
        rig.pose.bones[name].rotation_mode = "XYZ"


def wire_layers(rig, deform, controls, rest_angles):
    """Deform bones copy their mechanism in world space; a mechanism control adds its new control's turn
    after the pose it holds, before its joint limit, and the root and what hangs from it the location as
    well. Widgets, colours and collections go to the new bones."""
    data = rig.data
    collections = {c.name: c for c in data.collections_all}
    hidden = data.collections.new(BONES_MECHANISM)
    hidden.is_visible = False
    for name in deform:
        c = rig.pose.bones[name].constraints.new("COPY_TRANSFORMS")
        c.name = COPY
        c.target, c.subtarget = rig, mechanism(name)
        c.owner_space = c.target_space = "WORLD"
        old = data.bones[mechanism(name)]
        collections[R.DEFORM].unassign(old)
        collections[R.DEFORM].assign(data.bones[name])
        hidden.assign(old)
    for name in controls:
        pb, held = rig.pose.bones[R.control(name)], rig.pose.bones[mechanism_control(name)]
        c = held.constraints.new("COPY_ROTATION")
        c.name = HOLD
        c.target, c.subtarget = rig, R.control(name)
        c.owner_space = c.target_space = "LOCAL"
        c.euler_order = "XYZ"
        c.mix_mode = "AFTER"
        held.constraints.move(len(held.constraints) - 1, 0)
        # Only where a pose moves a bone: every constraint is a round trip through local space, and the
        # chain's float drift grows with them, to 0.005 mm at the fingers of a tall body with one on each.
        if held.parent is None or held.parent.parent is None:
            c = held.constraints.new("COPY_LOCATION")
            c.name = HOLD_LOCATION
            c.target, c.subtarget = rig, R.control(name)
            c.owner_space = c.target_space = "LOCAL"
            c.use_offset = True
            held.constraints.move(len(held.constraints) - 1, 1)
        limit = held.constraints.get(R.LIMIT)
        if limit is not None:
            widen_limit(limit, rest_angles.get(name))
        old = data.bones[mechanism_control(name)]
        for collection in list(old.collections):
            collection.unassign(old)
            collection.assign(data.bones[R.control(name)])
        hidden.assign(old)
        data.bones[R.control(name)].color.palette = old.color.palette
        pb.color.palette = held.color.palette
        pb.custom_shape, held.custom_shape = held.custom_shape, None
        pb.custom_shape_transform, held.custom_shape_transform = rig.pose.bones[name], None
        pb.use_transform_at_custom_shape = held.use_transform_at_custom_shape
        pb.use_custom_shape_bone_size = held.use_custom_shape_bone_size
        pb.custom_shape_scale_xyz = held.custom_shape_scale_xyz
        pb.custom_shape_translation = held.custom_shape_translation
        pb.custom_shape_wire_width = held.custom_shape_wire_width


def widen_limit(limit, angles):
    """A rest pose let past its joint limit stays reachable: the limit there stretches to it, or the rest
    pose wouldn't be the rest."""
    if angles is None:
        return
    # A limit splits the rotation in its own order, which needn't be the control's.
    angles = Euler(angles, "XYZ").to_matrix().to_euler(limit.euler_order)
    for axis, angle in zip("xyz", angles, strict=True):
        if getattr(limit, f"use_limit_{axis}"):
            setattr(limit, f"min_{axis}", min(getattr(limit, f"min_{axis}"), angle))
            setattr(limit, f"max_{axis}", max(getattr(limit, f"max_{axis}"), angle))


def rest_key(body, rig, name, angles, co):
    """The `rest.<bone>` key: 0 at the control's rest angles, 1 at its A pose, read off the mechanism's
    control as the correctives are."""
    key = body.shape_key_add(name=KEY_PREFIX + name, from_mix=False)
    set_key(key, co)
    key.slider_min, key.slider_max = -KEY_RANGE, KEY_RANGE
    a, b, c = angles
    size = a * a + b * b + c * c
    channel = {
        v: ("transform", rig, mechanism_control(name), f"ROT_{v.upper()}", "XYZ") for v in "xyz"
    }
    R.driver(
        key,
        "value",
        None,
        f"clamp(1 - (x * {a:.10f} + y * {b:.10f} + z * {c:.10f}) / {size:.12f}, {-KEY_RANGE:g}, {KEY_RANGE:g})",
        channel,
    )
    return key


def apply(rig, rest, central):
    """Make `rest` the file's rest pose: {"pose": ..., ...} of sotaigen.rig.rest_pose.solve. `central` is the rig's
    bones without a side (sotaigen.output.glb.driver_table's). Raises if either rest doesn't come back."""
    body = bpy.data.objects[R.BODY]
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.parent == rig]
    subdivisions = [mod for o in meshes for mod in o.modifiers if mod.type == "SUBSURF"]
    for mod in subdivisions:
        mod.show_viewport = False
    rig[R.LIMITS_SWITCH] = False
    deform = [b.name for b in rig.data.bones if b.use_deform]
    controls = [R.deform_of(b.name) for b in rig.data.bones if b.name.endswith(R.CONTROL_SUFFIX)]
    weights = {obj.name: group_weights(obj, deform) for obj in meshes}
    keys = body.data.shape_keys.key_blocks

    # Read what the file does now: the A pose, the rest pose, and the way back one control at a time.
    turned, moved = held_pose(rig, rest, central)
    order = list(turned)
    if not order:
        raise ValueError("the rest pose turns no control")
    at_rest = Snapshot(rig, meshes, body, deform)
    states = []
    for name in order:
        rig.pose.bones[R.control(name)].rotation_euler = (0.0, 0.0, 0.0)
        states.append((name, Snapshot(rig, meshes, body, deform)))
    R.set_pose(rig, {}, central)
    at_a = Snapshot(rig, meshes, body, deform)
    frames = {name: orthonormal(at_rest.bones[i]) for i, name in enumerate(deform)}
    inverse = np.linalg.inv(at_rest.bones)
    a_to_rest = np.einsum("bij,bjk->bik", at_rest.bones, np.linalg.inv(at_a.bones))

    def rest_to(snapshot):
        return np.einsum("bij,bjk->bik", snapshot.bones, inverse)

    # The skin in the new rest. A corrective's delta rides the skin's blend at the pose.
    skin = weights[body.name]
    base = at_rest.verts[body.name]
    held = blended(skin, a_to_rest)
    a_basis = key_coordinates(keys[0])
    carried = np.array(
        [(held[:, :3, :3] @ (key_coordinates(kb) - a_basis)[:, :, None])[:, :, 0] for kb in keys[1:]]
    )
    driven = at_rest.keys

    def wanted(snapshot):
        """What the new basis's keys must add, beyond the carried correctives, at this snapshot."""
        blend = blended(skin, rest_to(snapshot))
        return unskinned(blend, snapshot.verts[body.name]) - base - np.einsum(
            "k,kvi->vi", snapshot.keys - driven, carried
        )

    moves, reached = {}, np.zeros_like(base)
    for name, snapshot in states:
        now = wanted(snapshot)
        moves[name] = now - reached
        reached = now
    # The translation, last off, has its share in the last control's key: it only matters where a vertex
    # hangs from the root as well as what moved, and the root has no weights.
    moves[order[-1]] += wanted(at_a) - reached

    rename_as_mechanism(rig, meshes, deform, controls)
    new_layers(rig, deform, controls, frames)
    wire_layers(rig, deform, controls, turned)
    # The mechanism's controls hold the pose again; the new ones stay at zero.
    for name, angles in turned.items():
        rig.pose.bones[mechanism_control(name)].rotation_euler = angles
    for name, location in moved.items():
        rig.pose.bones[mechanism_control(name)].location = location

    # The meshes in the new rest.
    for obj in meshes:
        me = obj.data
        verts = at_rest.verts[obj.name]
        me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
        if obj is body or not me.shape_keys:
            continue
        blocks = me.shape_keys.key_blocks
        blend = blended(weights[obj.name], a_to_rest)
        rest_basis = key_coordinates(blocks[0])
        for block in blocks[1:]:
            delta = key_coordinates(block) - rest_basis
            set_key(block, verts + (blend[:, :3, :3] @ delta[:, :, None])[:, :, 0])
        set_key(blocks[0], verts)
    me = body.data
    set_key(keys[0], base)
    for block, delta in zip(keys[1:], carried, strict=True):
        set_key(block, base + delta)
    fcurves = {fc.data_path: fc for fc in me.shape_keys.animation_data.drivers}
    for block, value in zip(keys[1:], driven, strict=True):
        if value > QUIET:
            block.slider_min = -1.0
            d = fcurves[f'key_blocks["{block.name}"].value'].driver
            d.expression = f"{d.expression} - {value:.12f}"
    extra = {}
    for name in order:
        # The last one stays whatever it moves: it settles the A pose below.
        if name == order[-1] or np.abs(moves[name]).max() > STILL:
            extra[name] = rest_key(body, rig, name, turned[name], base + moves[name])

    # Blender's rest frames and float products leave a trace at each rest: settle it there, on the
    # basis and on the last key, so that neither rest carries more than rounding. The joint limits' own
    # rounding differs by a little, so the key takes the middle of what it is with them on and off.
    rig[R.LIMITS_SWITCH] = True
    R.set_pose(rig, {}, central)
    rests = np.linalg.inv(np.array([np.array(rig.data.bones[n].matrix_local, float) for n in deform]))

    def blend_of(snapshot, obj=body):
        return blended(weights[obj.name], np.einsum("bij,bjk->bik", snapshot.bones, rests))

    def apart(snapshot, reference):
        return {
            obj.name: float(np.linalg.norm(snapshot.verts[obj.name] - reference.verts[obj.name], axis=1).max() * 1000)
            for obj in meshes
        }

    # Every mesh: a new rest frame's rounding is up to 5e-6 rad, 0.016 mm at a fingertip. Taking it out of
    # the coordinates takes it out of both rests, since only the pose matrices are then left.
    now = Snapshot(rig, meshes, body, deform)
    for obj in meshes:
        verts = at_rest.verts[obj.name]
        settle = unskinned(blend_of(now, obj), verts) - verts
        if obj.data.shape_keys:
            for block in obj.data.shape_keys.key_blocks:
                set_key(block, key_coordinates(block) + settle)
        obj.data.vertices.foreach_set("co", (verts + settle).astype(np.float32).ravel())
    a_pose = to_a_pose(rig, controls)
    steps = []
    for on in (False, True):
        rig[R.LIMITS_SWITCH] = on
        now = Snapshot(rig, meshes, body, deform)
        steps.append(unskinned(blend_of(now), at_a.verts[body.name]) - keyed_shape(body, now))
    set_key(extra[order[-1]], key_coordinates(extra[order[-1]]) + (steps[0] + steps[1]) / 2)
    errors = {"a": apart(Snapshot(rig, meshes, body, deform), at_a)}
    R.set_pose(rig, {}, central)
    errors["rest"] = apart(Snapshot(rig, meshes, body, deform), at_rest)
    for where, row in errors.items():
        worst = max(row.items(), key=lambda item: item[1])
        print(f"rest pose: at the {where} pose, {worst[0]} is {worst[1]:.5f} mm from where it was")
        if worst[1] > REFUSE_MM:
            raise ValueError(f"at the {where} pose, {worst[0]} is {worst[1]:.5f} mm from where it was")
    for mod in subdivisions:
        mod.show_viewport = True
    info = rig[R.INFO]
    info["restPose"] = {
        "request": {k: v for k, v in rest["request"].items() if v is not None},
        "pose": rest["pose"],
        "measured": rest["measured"],
        "pastLimits": rest["pastLimits"],
        "unclean": rest["unclean"],
        "aPose": a_pose,
        "errorMm": errors,
    }
    return errors


def asset_pose(rig, pose, central):
    """A pose named by the A pose's angles, as the new controls name it, side by side: each mechanism control
    does R Q, so Q = R^-1 P. The bones then reach exactly what they do in the A pose's file."""
    out = {}
    for bone, value in pose.items():
        one = bone.startswith(("left", "right")) or bone in central
        sided = (
            [(bone, value)]
            if one
            else [(f"left{R.capital(bone)}", value), (f"right{R.capital(bone)}", (value[0], -value[1], -value[2]))]
        )
        for each, angles in sided:
            held = rig.pose.bones[mechanism_control(each)].rotation_euler.to_matrix()
            wanted = Euler([math.radians(angle) for angle in angles], "XYZ").to_matrix()
            out[each] = [math.degrees(v) for v in (held.inverted() @ wanted).to_euler("XYZ")]
    return out


def to_a_pose(rig, controls):
    """Turn the new controls back to the A pose; returns what each one that moves had to do, as the file
    tells it to whoever needs to get back: {bone: {"rotation": degrees, "location": mm}}."""
    out = {}
    for name in controls:
        held = rig.pose.bones[mechanism_control(name)]
        new = rig.pose.bones[R.control(name)]
        new.matrix_basis = held.matrix_basis.inverted()
        if any(abs(v) > 0 for v in (*held.rotation_euler, *held.location)):
            out[name] = {
                "rotation": [float(np.degrees(v)) for v in new.rotation_euler],
                "location": [float(v * 1000) for v in new.location],
            }
    return out


def keyed_shape(body, snapshot):
    """The skin in the new rest as `snapshot` has it before its skinning: the keys' sum at their values."""
    keys = body.data.shape_keys.key_blocks
    basis = key_coordinates(keys[0])
    out = basis.copy()
    values = snapshot.keys
    for block, value in zip(keys[1:], values, strict=True):
        if value:
            out += value * (key_coordinates(block) - basis)
    return out
