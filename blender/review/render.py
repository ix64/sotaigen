# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender side of sotaigen_dev/review.py: renders of OBJ files in a look and a camera, from a manifest.

Blender -b --factory-startup --python-exit-code 1 --python blender/review/render.py -- <manifest.json>

The manifest names each OBJ once with every shot wanted from it, so one Blender process can serve a
whole sheet. All framing is decided by review.py; this file only places the camera it is told to.
A shot's fields replace what the RENDER_VIEWS, RENDER_LOOK and RENDER_CAMERA variables used to set:

look    clay: grey studio light, shows every bump in the surface
        toon: two flat tones and an outline, how an animation body is actually read
        silhouette: one flat dark grey, for proportion and the gaps between limbs and trunk
camera  ortho: true proportions
        persp: through a 50 mm lens, how a body appears on screen
"""

import json
import os
import re
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

BODY_COLOR = (0.48, 0.50, 0.55)
# The editor's silhouette, 0x2f3136, in linear colour; shown as it is through the Standard transform.
SILHOUETTE_COLOR = (0.0284, 0.0307, 0.0369)
WIRE_COLOR = (0.02, 0.02, 0.04)
# Metres from the framed point to an orthographic camera; OBJ millimetres are scaled to metres.
CAMERA_DISTANCE = 3.0
MM = 0.001

# Toon tones in linear colour.
LIT = (0.86, 0.84, 0.82, 1.0)
SHADE = (0.40, 0.40, 0.48, 1.0)
# Above this N.L a surface is lit. It sets where the terminator falls on the form.
TERMINATOR = 0.35
# The outline's width in pixels of the image, whatever the sheet's scale.
OUTLINE_PX = 2.5
# How far the outline falls behind the body along the view, in outline widths. Without it the
# pushed-out copy pokes through tight folds, such as between toes, as black specks.
OUTLINE_BEHIND = 4.0
# Direction to the key light in the camera's frame: left of it, above it and behind it.
KEY = Vector((-0.45, 0.55, 0.70)).normalized()


def available_engines(scene):
    """The render engines, read from the error: the enum's own items under-report them."""
    current = scene.render.engine
    try:
        scene.render.engine = ""
    except TypeError as error:
        return re.findall(r"'(\w+)'", str(error).split("not found in")[-1])
    scene.render.engine = current
    raise RuntimeError("the render engine enum accepted an empty name")


def enum_or_current(owner, name, wanted):
    """Use the wanted value when this Blender offers it, otherwise leave the setting alone."""
    items = {item.identifier for item in owner.bl_rna.properties[name].enum_items}
    if wanted in items:
        setattr(owner, name, wanted)
    else:
        print(f"render.py: {name}={wanted} is not offered ({sorted(items)})", file=sys.stderr)


def set_up_scene(scene):
    shading = scene.display.shading
    enum_or_current(shading, "light", "STUDIO")
    enum_or_current(shading, "color_type", "SINGLE")
    shading.single_color = BODY_COLOR
    shading.show_cavity = False
    shading.show_object_outline = False
    enum_or_current(scene.display, "render_aa", "16")
    scene.render.film_transparent = True
    scene.render.resolution_percentage = 100

    camera_data = bpy.data.cameras.new("camera")
    camera_data.clip_start, camera_data.clip_end = 0.1, 8
    camera = bpy.data.objects.new("camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    return camera


class Looks:
    """Switches the scene between looks, restoring what toon changes when going back to clay."""

    def __init__(self, scene):
        self.scene = scene
        self.engines = available_engines(scene)
        self.studio_light = scene.display.shading.studio_light
        self.clay_transform = scene.view_settings.view_transform

    def engine(self, word):
        found = [engine for engine in self.engines if word in engine]
        if not found:
            raise RuntimeError(f"no {word} render engine among {self.engines}")
        return found[0]

    def use(self, look):
        scene = self.scene
        shading = scene.display.shading
        flat = look == "silhouette"
        enum_or_current(shading, "light", "FLAT" if flat else "STUDIO")
        shading.single_color = SILHOUETTE_COLOR if flat else BODY_COLOR
        if look == "toon":
            scene.render.engine = self.engine("EEVEE")
            scene.eevee.taa_render_samples = 16
            # Flat tones have to reach the image as they are, without a filmic curve. This enum's
            # items are dynamic and read as empty, and without it the tones are wrong, so a Blender
            # lacking it should fail here: the TypeError lists what it offers.
            scene.view_settings.view_transform = "Standard"
        else:
            scene.render.engine = self.engine("WORKBENCH")
            # The silhouette's one colour has to reach the image as it is, as toon's tones do.
            scene.view_settings.view_transform = "Standard" if flat else self.clay_transform


class Toon:
    """A body's MToon-style shading and outline, added the first time a shot wants them."""

    def __init__(self, obj, line_mm):
        self.line_mm = line_mm
        self.light_node, self.hull = add_toon(obj, line_mm)


def add_toon(obj, line_mm):
    """Shade like MToon, the VRM runtime's shader: N.L through a hard step, with no cast shadows.

    Returns the node holding the light's direction and the outline, which follow the camera.
    """
    material = bpy.data.materials.new("toon")
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    geometry = nodes.new("ShaderNodeNewGeometry")
    dot = nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].color = SHADE
    ramp.color_ramp.elements[1].position = TERMINATOR
    ramp.color_ramp.elements[1].color = LIT
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(geometry.outputs["Normal"], dot.inputs[0])
    links.new(dot.outputs["Value"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs[0], output.inputs["Surface"])
    obj.data.materials.append(material)

    # Inverted hull, the outline VRM runtimes draw: a pushed-out copy showing only its inner side.
    mesh = obj.data.copy()
    mesh.materials.clear()
    points = np.empty(len(mesh.vertices) * 3)
    normals = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", points)
    mesh.vertex_normals.foreach_get("vector", normals)
    mesh.vertices.foreach_set("co", points + normals * line_mm)
    mesh.flip_normals()
    mesh.update()
    line = bpy.data.materials.new("outline")
    line.use_backface_culling = True
    line.node_tree.nodes.clear()
    ink = line.node_tree.nodes.new("ShaderNodeEmission")
    ink.inputs["Color"].default_value = (0.02, 0.02, 0.03, 1.0)
    line_output = line.node_tree.nodes.new("ShaderNodeOutputMaterial")
    line.node_tree.links.new(ink.outputs[0], line_output.inputs["Surface"])
    mesh.materials.append(line)
    hull = bpy.data.objects.new("outline", mesh)
    hull.scale = obj.scale
    bpy.context.scene.collection.objects.link(hull)
    return dot, hull


def import_body(scene, source, wants_wire):
    """The OBJ as a smooth mesh, and its wireframe twin when a shot wants one."""
    bpy.ops.wm.obj_import(filepath=source, forward_axis="Y", up_axis="Z")
    body = bpy.context.selected_objects[0]
    body.scale = (MM, MM, MM)
    for polygon in body.data.polygons:
        polygon.use_smooth = True
    if not wants_wire:
        return body, None, None
    body.color = (*BODY_COLOR, 1)
    wire = body.copy()
    scene.collection.objects.link(wire)
    modifier = wire.modifiers.new("wire", "WIREFRAME")
    modifier.use_relative_offset = False
    # Outward, so the wires sit on the surface instead of inside the solid body.
    modifier.offset = 1.0
    wire.color = (*WIRE_COLOR, 1)
    return body, wire, modifier


def remove_body(body, wire, toon):
    mesh = body.data
    if toon is not None:
        hull_mesh = toon.hull.data
        bpy.data.objects.remove(toon.hull)
        bpy.data.meshes.remove(hull_mesh)
    if wire is not None:
        bpy.data.objects.remove(wire)
    bpy.data.objects.remove(body)
    bpy.data.meshes.remove(mesh)


def place_camera(scene, camera, shot):
    width, height = shot["width"], shot["height"]
    scene.render.resolution_x, scene.render.resolution_y = width, height
    data = camera.data
    if shot["camera"] == "persp":
        data.type = "PERSP"
        data.sensor_width = shot["sensor_mm"]
        data.lens = shot["lens_mm"]
        data.shift_x, data.shift_y = shot["shift"]
        camera.location = Vector(shot["eye"]) * MM
    else:
        data.type = "ORTHO"
        data.shift_x = data.shift_y = 0.0
        # An orthographic camera's scale spans the longer side of the frame.
        data.ortho_scale = max(width, height) * shot["mm_per_px"] * MM
        toward = Vector(shot["toward"])
        camera.location = Vector(shot["center"]) * MM + toward * CAMERA_DISTANCE
    right, up, toward = map(Vector, (shot["right"], shot["up"], shot["toward"]))
    camera.rotation_euler = Matrix((right, up, toward)).transposed().to_euler()


def aim_toon(toon, body, camera, shot, line_mm):
    """Keep the key light on the camera, and the outline behind the body along the view."""
    toon.light_node.inputs[1].default_value = camera.rotation_euler.to_matrix() @ KEY
    behind = OUTLINE_BEHIND * line_mm * MM
    if shot["camera"] == "persp":
        # Along each point's own ray, so that the outline stays centred on the silhouette.
        grow = 1 + behind / (shot["distance_mm"] * MM)
        toon.hull.scale = body.scale * grow
        toon.hull.location = camera.location * (1 - grow)
    else:
        toon.hull.scale = body.scale
        toon.hull.location = -Vector(shot["toward"]) * behind


def shoot(scene, camera, looks, shot, body, wire, modifier, toon):
    """Render one shot. Returns the body's Toon, made here if this is the first shot to want it."""
    shading = scene.display.shading
    looks.use(shot["look"])
    if shot["look"] == "clay":
        shading.studio_light = shot.get("studio_light", looks.studio_light)
    place_camera(scene, camera, shot)
    if shot["look"] == "toon":
        line_mm = OUTLINE_PX * shot["mm_per_px"]
        if toon is None:
            toon = Toon(body, line_mm)
        elif toon.line_mm != line_mm:
            raise RuntimeError("one manifest has one scale per mesh, so one outline width")
        toon.hull.hide_render = False
        aim_toon(toon, body, camera, shot, line_mm)
    elif toon is not None:
        toon.hull.hide_render = True

    wired = shot["wire"] and wire is not None
    if wire is not None:
        wire.hide_render = not wired
        enum_or_current(shading, "color_type", "OBJECT" if wired else "SINGLE")
    if wired:
        # About half a pixel wide at this sheet's scale, in the mesh's own millimetres.
        modifier.thickness = 0.5 * shot["mm_per_px"]

    # Written aside and moved, so an interrupted run cannot leave a partial file in the cache.
    partial = shot["out"] + ".partial.png"
    scene.render.filepath = partial
    bpy.ops.render.render(write_still=True)
    os.replace(partial, shot["out"])
    return toon


def main():
    manifest = sys.argv[sys.argv.index("--") + 1]
    with open(manifest) as file:
        meshes = json.load(file)["meshes"]
    scene = bpy.context.scene
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    camera = set_up_scene(scene)
    looks = Looks(scene)
    for mesh in meshes:
        wants_wire = any(shot["wire"] for shot in mesh["shots"])
        body, wire, modifier = import_body(scene, mesh["source"], wants_wire)
        toon = None
        for shot in mesh["shots"]:
            toon = shoot(scene, camera, looks, shot, body, wire, modifier, toon)
        remove_body(body, wire, toon)


main()
