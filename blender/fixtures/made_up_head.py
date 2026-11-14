# SPDX-License-Identifier: GPL-3.0-or-later
"""A made-up head to swap onto a body: the body's own head, reshaped, with pieces no skin has.

Blender -b --factory-startup BODY.blend --python-exit-code 1 --python blender/fixtures/made_up_head.py -- OUT.blend
    [--coarse] [--no-skin] [--closed] [--weighted] [--on-ring] [--pinched-hair] [--marked]

Comes from a body's file and from nothing else, so no character is in it. The shape is plainly another
head: a higher, wider skull, a narrower jaw, a chin further forward, a thicker neck stub. It carries
what a real head brings that ours doesn't:

- teeth and a tongue as separate pieces of the skin's own object, bound by vertex groups `jaw` and `head`;
- eyeballs in an object of their own, bound by `leftEye`, `rightEye`;
- hair in an object with no groups at all: a cap over the skull with its faces flipped, and a strand that
  no skin is near;
- its own armature with `head`, `jaw`, `leftEye`, `rightEye`, all moved with the shape;
- a transform on every object, so the file's own frame isn't the body's;
- a shape key and custom normals on the skin.

`--coarse` keeps every other row and column, so the neck's opening has as many vertices as the ring (36)
and welds to it, instead of zipping 72 to 36 as the full one does. `--no-skin` leaves the skin out and
everything else in its own place, for `swap-head --parts-only`. `--closed` leaves out the hair, the one
piece that is open, so that with `--no-skin` the file has nothing to join to the ring. `--weighted` gives the
skin's upper half weights of its own, half `head` and half `neck`, which nothing else would give it.
`--on-ring` moves the neck's opening onto the body's ring, where a fit of 1 / FILE_SCALE leaves it.
`--pinched-hair` adds to the hair two triangles meeting at a vertex, a boundary that touches itself.
`--marked` marks the skin with the group headRegion.
"""

import math
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import itertools

from sotaigen_blender import rig as R

# The head's rows in a body's file: sotaigen.form.head.topology.
SIDES, ROWS = 72, 46
# The whole head is this much bigger in its file and this far off, for the tool to undo.
FILE_SCALE = 1.2
FILE_SHIFT = (0.03, 0.05, -0.2)
# Every object has this transform, and its mesh is whatever then puts it there.
OBJECT_SCALE = 0.5
OBJECT_SHIFT = (0.1, 0.0, 0.05)
# The eyeball's radius, as a share of the head's height.
EYE = 0.07


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def reshaped(points, crown, ring_z, centre):
    """Another head's shape, by height: t is 0 at the crown and 1 at the ring."""
    t = (crown - points[:, 2]) / (crown - ring_z)
    out = points.copy()
    x, y = points[:, 0] - centre[0], points[:, 1] - centre[1]
    skull = 1 - smooth(0.25, 0.65, t)
    jaw = smooth(0.5, 0.92, t) * (1 - smooth(0.92, 1.0, t))
    neck = smooth(0.92, 1.0, t)
    wide = 1 + 0.10 * skull - 0.16 * jaw + 0.08 * neck
    deep = 1 + 0.08 * skull - 0.04 * jaw + 0.08 * neck
    out[:, 0] = centre[0] + x * wide
    out[:, 1] = centre[1] + y * deep - 0.09 * (crown - ring_z) * jaw * (y < 0)
    out[:, 2] = points[:, 2] + 0.08 * (crown - ring_z) * skull
    return out


def on_ring(row, ring):
    """A row of twice the ring's count on the ring: its vertices and the middles of its edges, with the
    same centre as the ring's vertices, so that the fit leaves them there. The first is nearest the row's."""
    middles = (ring + np.roll(ring, -1, axis=0)) / 2
    target = np.column_stack((ring, middles)).reshape(-1, 3)
    return np.roll(target, -int(np.argmin(np.linalg.norm(target - row[0], axis=1))), axis=0)


def add_material(me, name, color):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = next(n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = color
    material.diffuse_color = color
    me.materials.append(material)
    return len(me.materials) - 1


def sphere(bm, centre, radius, scale=(1.0, 1.0, 1.0), segments=16, rings=10):
    """A UV sphere's new vertices in `bm`, in mm around `centre`."""
    made = bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings, radius=radius)["verts"]
    for vertex in made:
        vertex.co = [c + v * s for c, v, s in zip(centre, vertex.co, scale, strict=True)]
    return made


def arc(bm, centre, radius, height, depth, segments=14):
    """A horseshoe of teeth, opening backward: a closed band around `centre`, its vertices in `bm`."""
    made = []
    angles = np.linspace(-0.45 * math.pi, 0.45 * math.pi, segments)
    columns = []
    for angle in angles:
        x, y = centre[0] + radius * math.sin(angle), centre[1] - radius * math.cos(angle)
        out = np.array([math.sin(angle), -math.cos(angle)]) * depth / 2
        column = [
            bm.verts.new((x + out[0] * a, y + out[1] * a, centre[2] + b * height / 2))
            for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))
        ]
        made += column
        columns.append(column)
    for left, right in itertools.pairwise(columns):
        for k in range(4):
            bm.faces.new((left[k], right[k], right[(k + 1) % 4], left[(k + 1) % 4]))
    bm.faces.new(columns[0][::-1])
    bm.faces.new(columns[-1])
    return made


def finish(obj, groups):
    for name, rows in groups.items():
        obj.vertex_groups.new(name=name).add(rows, 1.0, "REPLACE")


def main():
    args = sys.argv[sys.argv.index("--") + 1 :]
    out = Path(args[0])
    coarse = "--coarse" in args
    no_skin = "--no-skin" in args
    closed = "--closed" in args
    weighted = "--weighted" in args
    body = bpy.data.objects[R.BODY]
    rig = bpy.data.objects[R.RIG]
    info = rig[R.INFO].to_dict()["head"]
    first, count = info["region"]
    assert count == SIDES * ROWS + 1
    co = np.empty(len(body.data.vertices) * 3)
    body.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3) * 1000
    points = co[first : first + count]
    ring = co[info["neckRing"]]
    crown, ring_z = float(co[info["crownVertex"], 2]), float(ring[:, 2].mean())
    centre = ring.mean(0)
    bones = {
        name: (np.array(rig.data.bones[name].head_local) * 1000, np.array(rig.data.bones[name].tail_local) * 1000)
        for name in ("head", "jaw", "leftEye", "rightEye")
    }
    head_h = crown - ring_z

    def place(p):
        return reshaped(np.atleast_2d(p), crown, ring_z, centre)

    new_points = reshaped(points, crown, ring_z, centre)

    def eye_centre(name):
        """Pushed about as the skin is, and forward until the ball shows through the skin by 3 mm."""
        at = place(bones[name][0])[0]
        near = np.linalg.norm((new_points - at)[:, [0, 2]], axis=1) < EYE * head_h
        at[1] = new_points[near & (new_points[:, 1] < at[1]), 1].min() + EYE * head_h - 3.0
        return at

    step = 2 if coarse else 1
    rows = list(range(0, ROWS, step))
    sides = list(range(0, SIDES, step))
    if rows[-1] != ROWS - 1:
        rows.append(ROWS - 1)
    grid = np.array([[new_points[i * SIDES + k] for k in sides] for i in rows])
    if "--on-ring" in args:
        grid[0] = on_ring(grid[0], ring)
    verts = [*grid.reshape(-1, 3).tolist(), new_points[-1].tolist()]
    around = len(sides)
    faces = []
    for i in range(len(rows) - 1):
        for k in range(around):
            a, b = i * around + k, i * around + (k + 1) % around
            faces.append((a, b, b + around, a + around))
    pole = len(verts) - 1
    last = (len(rows) - 1) * around
    faces += [(last + k, last + (k + 1) % around, pole) for k in range(around)]

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    # With --no-skin the pieces stay where the body's own head has them: they were modelled on it.
    in_file = Matrix.Identity(4)
    if not no_skin:
        in_file = Matrix.Diagonal((FILE_SCALE, FILE_SCALE, FILE_SCALE, 1.0))
        in_file.translation = FILE_SHIFT
    place_file = Matrix.Diagonal((OBJECT_SCALE, OBJECT_SCALE, OBJECT_SCALE, 1.0))
    place_file.translation = OBJECT_SHIFT
    # What's made in the body's frame in mm goes to its place in the file, as a vertex of an object there.
    undo = np.array(place_file.inverted() @ in_file)

    def into_file(array):
        p = np.asarray(array, float) / 1000
        return p @ undo[:3, :3].T + undo[:3, 3]

    armature = bpy.data.armatures.new("headRig")
    head_rig = bpy.data.objects.new("headRig", armature)
    scene.collection.objects.link(head_rig)
    head_rig.matrix_world = place_file
    bpy.context.view_layer.objects.active = head_rig
    bpy.ops.object.mode_set(mode="EDIT")
    for name, (at, tail) in bones.items():
        bone = armature.edit_bones.new(name)
        moved = eye_centre(name) if name.endswith("Eye") else place(at)[0]
        # A little off what the shape alone gives, as a real head is.
        if name == "jaw":
            moved = moved + np.array([0.0, -0.01, -0.005]) * head_h
        bone.head = into_file(moved)
        bone.tail = into_file(moved + (tail - at) * 1.1)
    bpy.ops.object.mode_set(mode="OBJECT")

    # The skin, and the teeth and the tongue in the same object, not connected to it.
    me = bpy.data.meshes.new("headSkin")
    me.from_pydata(into_file(verts).tolist(), [], faces)
    me.update()
    skin = bpy.data.objects.new("headSkin", me)
    scene.collection.objects.link(skin)
    skin.matrix_world = place_file
    add_material(me, "testSkin", (0.8, 0.72, 0.68, 1.0))
    mouth = add_material(me, "testMouth", (0.72, 0.30, 0.32, 1.0))
    uv = me.uv_layers.new(name="testUV")
    corners = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", corners)
    flat = np.array(verts)[corners]
    uv.data.foreach_set("uv", ((flat[:, :2] - flat[:, :2].min(0)) / np.ptp(flat[:, :2], axis=0)).astype(np.float32).ravel())
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    skin_count = len(me.vertices)
    jaw_at, _ = bones["jaw"]
    # Inside the mouth: under the eyes, in front of the jaw's pivot.
    middle = place(np.array([0.0, jaw_at[1] - 0.20 * head_h, jaw_at[2] - 0.13 * head_h]))[0]
    bm = bmesh.new()
    bm.from_mesh(me)
    upper = arc(bm, middle + np.array([0.0, 0.0, 0.03 * head_h]), 0.10 * head_h, 0.02 * head_h, 0.02 * head_h)
    lower = arc(bm, middle - np.array([0.0, 0.0, 0.01 * head_h]), 0.09 * head_h, 0.02 * head_h, 0.02 * head_h)
    tongue = sphere(bm, middle + np.array([0.0, 0.04 * head_h, -0.012 * head_h]), 0.05 * head_h, (0.9, 1.3, 0.35))
    for vertex in (*upper, *lower, *tongue):
        vertex.co = into_file(vertex.co)
    for face in bm.faces:
        if face.verts[0] in set(upper) or face.verts[0] in set(lower) or face.verts[0] in set(tongue):
            face.material_index = mouth
    bm.verts.index_update()
    groups = {
        "head": [v.index for v in upper],
        "jaw": [v.index for v in (*lower, *tongue)],
    }
    bm.to_mesh(me)
    bm.free()
    finish(skin, groups)
    if "--marked" in args:
        finish(skin, {"headRegion": list(range(skin_count))})
    if weighted:
        high = [i for i in range(skin_count) if verts[i][2] > ring_z + 0.5 * head_h]
        skin.vertex_groups["head"].add(high, 0.5, "REPLACE")
        skin.vertex_groups.new(name="neck").add(high, 0.5, "REPLACE")
    # A blink of sorts, on the skin only, and normals somebody chose.
    skin.shape_key_add(name="Basis", from_mix=False)
    key = skin.shape_key_add(name="blink", from_mix=False)
    moved = np.empty(len(me.vertices) * 3)
    key.data.foreach_get("co", moved)
    moved = moved.reshape(-1, 3)
    moved[: skin_count // 2, 2] -= 0.002
    key.data.foreach_set("co", moved.astype(np.float32).ravel())
    key.value = 0.0
    me.update()
    normals = np.empty(len(me.vertices) * 3)
    me.vertex_normals.foreach_get("vector", normals)
    normals = normals.reshape(-1, 3)
    points_file = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", points_file)
    points_file = points_file.reshape(-1, 3)
    radial = points_file - points_file[:skin_count].mean(0)
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    chosen = normals + 0.5 * radial
    chosen /= np.linalg.norm(chosen, axis=1, keepdims=True)
    chosen[skin_count:] = 0.0
    me.normals_split_custom_set_from_vertices([tuple(n) for n in chosen])

    # The eyes, on their own: two pieces of one object.
    eye_me = bpy.data.meshes.new("eyes")
    eyes = bpy.data.objects.new("eyes", eye_me)
    scene.collection.objects.link(eyes)
    eyes.matrix_world = place_file
    add_material(eye_me, "testEye", (0.95, 0.95, 0.97, 1.0))
    bm = bmesh.new()
    pieces = {}
    for name in ("leftEye", "rightEye"):
        made = sphere(bm, eye_centre(name), EYE * head_h)
        for vertex in made:
            vertex.co = into_file(vertex.co)
        pieces[name] = made
    bm.verts.index_update()
    groups = {name: [v.index for v in made] for name, made in pieces.items()}
    bm.to_mesh(eye_me)
    bm.free()
    eye_me.polygons.foreach_set("use_smooth", np.ones(len(eye_me.polygons), bool))
    finish(eyes, groups)

    # The hair: no groups. A cap with its faces pointing in, and a strand that reaches no skin.
    hair_me = bpy.data.meshes.new("hair")
    hair = bpy.data.objects.new("hair", hair_me)
    scene.collection.objects.link(hair)
    hair.matrix_world = place_file
    add_material(hair_me, "testHair", (0.25, 0.20, 0.42, 1.0))
    bm = bmesh.new()
    # The skull's upper rows, a little outside it, as a separate shell.
    heights = (crown - points[:-1].reshape(ROWS, SIDES, 3)[:, :, 2].mean(1)) / head_h
    high = np.flatnonzero(heights < 0.30)[::2]
    shell = new_points[:-1].reshape(ROWS, SIDES, 3)[high][:, ::2]
    middle_of_skull = shell.mean((0, 1))
    shell = middle_of_skull + (shell - middle_of_skull) * 1.04
    grid_hair = [[bm.verts.new(tuple(p)) for p in row] for row in shell]
    for lower, upper in itertools.pairwise(grid_hair):
        for k in range(len(lower)):
            bm.faces.new((lower[k], lower[(k + 1) % len(lower)], upper[(k + 1) % len(lower)], upper[k]))
    bm.faces.new(grid_hair[-1])
    # Its faces point in, as a shell modelled from inside would.
    bmesh.ops.reverse_faces(bm, faces=list(bm.faces))
    cap = list(bm.verts)
    strand = sphere(bm, np.array([centre[0] + 0.05 * head_h, centre[1] + 1.1 * head_h, ring_z - 0.25 * head_h]), 0.05 * head_h, (0.7, 0.7, 2.4), 10, 8)
    for vertex in (*cap, *strand):
        vertex.co = into_file(vertex.co)
    if "--pinched-hair" in args:
        # Beside the cap, the size of a lock.
        at = middle_of_skull + np.array([0.15, 0.0, 0.3]) * head_h
        made = [bm.verts.new(tuple(into_file(at + np.array(p) * 0.03 * head_h))) for p in ((0, 0, 0), (1, 0, 0), (1, 0, 1), (-1, 0, 0), (-1, 0, -1))]
        bm.faces.new((made[0], made[1], made[2]))
        bm.faces.new((made[0], made[3], made[4]))
    bm.to_mesh(hair_me)
    bm.free()
    hair_me.polygons.foreach_set("use_smooth", np.ones(len(hair_me.polygons), bool))

    if closed:
        bpy.data.objects.remove(hair, do_unlink=True)
    if no_skin:
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[bm.verts[i] for i in range(skin_count)], context="VERTS")
        bm.to_mesh(me)
        bm.free()
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out), compress=True, check_existing=False)


main()
