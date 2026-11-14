# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender side of scripts/head_review.py: coloured parts in toon and clay, from a manifest.

Blender -b --factory-startup --python-exit-code 1 --python blender/review/face_render.py -- <manifest>

Driven by scripts/head_review.py, with the parts as they export. Not part of the pipeline. A scene is an .npz of parts, each with its own vertex normals,
because the toon look has to shade with the normals the export carries, not the mesh's.

toon  N.L through the hard step blender/review/render.py uses, on the part's normals, with an inverted hull
      for the parts asked to have one.
clay  Workbench, on the mesh's real normals: shows the forms themselves.
"""

import json
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

MM = 0.001
CAMERA_DISTANCE = 3.0
TERMINATOR = 0.35
SHADE_TINT = (0.72, 0.70, 0.80, 1.0)
OUTLINE_BEHIND = 4.0
OUTLINE_MM = 1.0
KEY = Vector((-0.45, 0.55, 0.70)).normalized()


def make_mesh(name, verts, faces, normals, colors):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    mesh.update()
    attribute = mesh.attributes.new("sn", "FLOAT_VECTOR", "POINT")
    attribute.data.foreach_set("vector", normals.ravel())
    corner = mesh.attributes.new("col", "FLOAT_COLOR", "POINT")
    corner.data.foreach_set("color", colors.ravel())
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    obj.scale = (MM, MM, MM)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def toon_material(name, image):
    """Colour (and a decal where one is given) through the hard step, on the part's own normals."""
    material = bpy.data.materials.new(name)
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    normal = nodes.new("ShaderNodeAttribute")
    normal.attribute_name = "sn"
    unit = nodes.new("ShaderNodeVectorMath")
    unit.operation = "NORMALIZE"
    dot = nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].color = SHADE_TINT
    ramp.color_ramp.elements[1].position = TERMINATOR
    ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    color = nodes.new("ShaderNodeAttribute")
    color.attribute_name = "col"
    multiply = nodes.new("ShaderNodeVectorMath")
    multiply.operation = "MULTIPLY"
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(normal.outputs["Vector"], unit.inputs[0])
    links.new(unit.outputs[0], dot.inputs[0])
    links.new(dot.outputs["Value"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], multiply.inputs[0])
    if image:
        uv = nodes.new("ShaderNodeUVMap")
        uv.uv_map = "UVMap"
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = bpy.data.images.load(image)
        texture.image.alpha_mode = "STRAIGHT"
        texture.extension = "CLIP"
        links.new(uv.outputs["UV"], texture.inputs["Vector"])
        mask = nodes.new("ShaderNodeMath")
        mask.operation = "MULTIPLY"
        links.new(texture.outputs["Alpha"], mask.inputs[0])
        links.new(color.outputs["Alpha"], mask.inputs[1])
        mix = nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        links.new(mask.outputs[0], mix.inputs["Factor"])
        links.new(color.outputs["Color"], mix.inputs["A"])
        links.new(texture.outputs["Color"], mix.inputs["B"])
        links.new(mix.outputs["Result"], multiply.inputs[1])
    else:
        links.new(color.outputs["Color"], multiply.inputs[1])
    links.new(multiply.outputs[0], emission.inputs["Color"])
    links.new(emission.outputs[0], output.inputs["Surface"])
    return material, dot


def add_outline(obj, line_mm):
    """The inverted hull, on the mesh's real normals, as in blender/review/render.py."""
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
    line.diffuse_color = (0, 0, 0, 0)
    line.node_tree.nodes.clear()
    ink = line.node_tree.nodes.new("ShaderNodeEmission")
    ink.inputs["Color"].default_value = (0.02, 0.02, 0.03, 1.0)
    out = line.node_tree.nodes.new("ShaderNodeOutputMaterial")
    line.node_tree.links.new(ink.outputs[0], out.inputs["Surface"])
    mesh.materials.append(line)
    hull = bpy.data.objects.new("hull", mesh)
    hull.scale = obj.scale
    bpy.context.scene.collection.objects.link(hull)
    return hull


def load_parts(scene_file):
    """Every part of one scene as objects. Returns the light's nodes, parts and hulls."""
    data = np.load(scene_file["npz"])
    line_mm = float(data["line_mm"]) if "line_mm" in data else OUTLINE_MM
    lights, objects, hulls = [], [], []
    for index in range(int(data["parts"])):
        verts = data[f"verts{index}"]
        normals = data[f"normals{index}"]
        if f"colors{index}" in data:
            colors = data[f"colors{index}"]
        else:
            colors = np.tile(data[f"color{index}"], (len(verts), 1))
        obj = make_mesh(f"part{index}", verts, data[f"faces{index}"], normals, colors)
        image = scene_file.get("images", {}).get(str(index))
        if image:
            layer = obj.data.uv_layers.new(name="UVMap")
            uvs = data[f"uvs{index}"]
            loops = np.empty(len(obj.data.loops), int)
            obj.data.loops.foreach_get("vertex_index", loops)
            layer.data.foreach_set("uv", uvs[loops].ravel())
        material, dot = toon_material(f"toon{index}", image)
        if f"clay{index}" in data:
            shade = float(data[f"clay{index}"])
            material.diffuse_color = (shade, shade, shade, 1.0)
        obj.data.materials.append(material)
        lights.append(dot)
        objects.append(obj)
        wanted = bool(data[f"outline{index}"]) if f"outline{index}" in data else index == 0
        if wanted:
            hulls.append(add_outline(obj, line_mm))
    return lights, objects, hulls, line_mm


def set_engine(scene, word):
    try:
        scene.render.engine = ""
    except TypeError as error:
        offered = [part.strip(" '()") for part in str(error).split("not found in")[-1].split(",")]
    scene.render.engine = next(engine for engine in offered if word in engine)


def use_look(scene, look):
    if look == "clay":
        set_engine(scene, "WORKBENCH")
        shading = scene.display.shading
        shading.light = "STUDIO"
        shading.color_type = "MATERIAL"
        shading.show_cavity = False
        shading.show_object_outline = False
    else:
        set_engine(scene, "EEVEE")
        scene.eevee.taa_render_samples = 16


def main():
    manifest = sys.argv[sys.argv.index("--") + 1]
    with open(manifest) as file:
        job = json.load(file)
    scene = bpy.context.scene
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    camera_data = bpy.data.cameras.new("camera")
    camera_data.type = "ORTHO"
    camera_data.clip_start, camera_data.clip_end = 0.1, 8
    camera = bpy.data.objects.new("camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    scene.render.film_transparent = job.get("transparent", False)
    world = bpy.data.worlds.new("white")
    world.use_nodes = True
    world.node_tree.nodes[0].inputs[0].default_value = (1, 1, 1, 1)
    scene.world = world
    scene.view_settings.view_transform = "Standard"
    scene.render.resolution_percentage = 100
    for item in job["scenes"]:
        lights, objects, hulls, line_mm = load_parts(item)
        for shot in item["shots"]:
            look = shot.get("look", "toon")
            use_look(scene, look)
            width = shot.get("width", shot.get("px"))
            height = shot.get("height", shot.get("px"))
            scene.render.resolution_x, scene.render.resolution_y = width, height
            camera_data.ortho_scale = max(width, height) * shot["mm_per_px"] * MM
            right, up, toward = map(Vector, (shot["right"], shot["up"], shot["toward"]))
            camera.location = Vector(shot["center"]) * MM + toward * CAMERA_DISTANCE
            rotation = Matrix((right, up, toward)).transposed()
            camera.rotation_euler = rotation.to_euler()
            key = rotation @ KEY
            tilt = Matrix.Rotation(np.radians(shot.get("light_tilt", 0.0)), 3, "X")
            for dot in lights:
                dot.inputs[1].default_value = tilt @ key
            for hull in hulls:
                hull.hide_render = look != "toon"
                hull.location = -toward * (OUTLINE_BEHIND * line_mm * MM)
            partial = shot["out"] + ".partial.png"
            scene.render.filepath = partial
            bpy.ops.render.render(write_still=True)
            os.replace(partial, shot["out"])
        for obj in objects + hulls:
            mesh = obj.data
            bpy.data.objects.remove(obj)
            bpy.data.meshes.remove(mesh)


main()
