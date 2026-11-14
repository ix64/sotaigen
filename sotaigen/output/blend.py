"""Write a body's rigged .blend, to build/bodies/<body>/body.blend, by running Blender on sotaigen-blender's convert.py.

uv run sotaigen-dev blend [name ...] [--sex female,male] [--group long,balanced]
    [--t-pose] [--arm-abduction DEG] [--ankle-spacing MM] [--palms-down] [--straight-arms] [--past-limits]
    [--out FILE]

The file is what docs/systems/blend.md describes: controls the animator turns and the runtime's deform
bones following them by constraints, the corrective shape keys driven by simple expressions, the
controls' range-of-motion limits, the head's placeholder parts and a subdivision surface. It poses
as sotaigen.rig.skinning.pose_verts does with Correctives.production's targets bound, to float precision;
tests/test_blend.py checks that on the typical bodies.

Like sotaigen/output/glb.py this only reads what the rig stage finished; no stage reaches it. Code that
imports Blender's Python modules is GPL and lives in blender/; this module only writes its input and
runs it. A body whose wrap is broken (flipped faces, vertices past the midline, an open skin) is
refused, with where it broke and, by sotaigen.mend, a parameter that ends it; quality problems only warn
and go into the file, under the rig's `sotaigen`. The GLB's
export takes both levels from here too.
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from sotaigen import mend
from sotaigen.corrective.recipe import Correctives
from sotaigen.diagnostics import Diagnostic
from sotaigen.errors import BlenderMissing
from sotaigen.form.anatomy import calibrated
from sotaigen.form.head import neck_ring_order
from sotaigen.mesh import skin_normals
from sotaigen.output.blender import (
    BUNDLE_FORMAT,
    BUNDLE_VERSION,
    NAMES,
    can_write_blend,
    run_blender,
    script,
)
from sotaigen.output.face import HeadShape, blended_normals, bound_parts, head_parts
from sotaigen.output.glb import driver_table, export_base, head_attachments
from sotaigen.output.metadata import BLEND_FORMAT, file_info
from sotaigen.output.metadata import VERSION as METADATA_VERSION
from sotaigen.output.quality import BrokenBody, body_notes, broken, quality, report
from sotaigen.pipeline import (
    BUILD,
    comma_list,
    ensure_built,
    rigged,
    select,
)
from sotaigen.rig.limits import joint_limits, limit_order
from sotaigen.rig.rest_pose import PastLimits, add_arguments, request_of, solve
from sotaigen.rig.touch import blend_poses, solve_all, unreached
from sotaigen.uv import load_uv

BLEND_NAME = "body.blend"
# A file with another rest pose, beside the A pose's (docs/systems/blend.md).
REST_NAME = "body-rest.blend"
# The rig's custom property sotaigen: the GLB's metadata (sotaigen.output.metadata) for a .blend.
FORMAT, VERSION = BLEND_FORMAT, METADATA_VERSION
# Decided with the fingers left as they are (docs/DECISIONS.md, 手指不加密).
SUBDIVISION = {"viewport": 1, "render": 2}
# The skin's material, as sotaigen/output/glb.py's SKIN_MATERIAL.
SKIN = {"baseColor": [0.8, 0.72, 0.68, 1.0], "roughness": 0.75}
# A ring widget sits this much outside the skin round its bone.
WIDGET_MARGIN = 1.2


# Widgets ----------------------------------------------------------------------------------------


def widgets(bones, weights, verts):
    """{bone: {"radius", "along"}} in mm: a ring round each control's deform bone, outside the skin
    it moves, halfway along the bone."""
    weights, verts = np.asarray(weights), np.asarray(verts, float)
    names = [bone["name"] for bone in bones]
    height = verts[:, 2].max() - verts[:, 2].min()
    out = {}
    for bone in bones:
        name = bone["name"]
        if "Twist" in name or name.endswith("Half"):
            continue
        head, tail = np.asarray(bone["head"], float), np.asarray(bone["tail"], float)
        length = float(np.linalg.norm(tail - head))
        if name == "root":
            out[name] = {"radius": 0.22 * height, "along": 0.0}
            continue
        # The bone's own skin, with its half and twist bones.
        columns = [j for j, other in enumerate(names) if other == name or other.startswith(name + "Twist")
                   or other == name + "Half"]
        mine = weights[:, columns].sum(1) >= 0.5
        axis = (tail - head) / length
        rel = verts[mine] - head
        t = rel @ axis
        middle = (t > 0.25 * length) & (t < 0.75 * length)
        if middle.sum() >= 8:
            distance = np.linalg.norm(rel[middle] - np.outer(t[middle], axis), axis=1)
            radius = float(np.percentile(distance, 90)) * WIDGET_MARGIN
        else:
            radius = 0.3 * length
        out[name] = {"radius": round(radius, 3), "along": round(0.5 * length, 3)}
    return out


# The bundle Blender reads ------------------------------------------------------------------------


def counts_corners(faces):
    return np.array([len(f) for f in faces]), np.concatenate([np.asarray(f) for f in faces])


def material_row(material):
    return {
        "baseColor": list(material.base_color),
        "roughness": 0.75,
        "alpha": material.alpha,
        "alphaCutoff": material.alpha_cutoff,
        "renderOrder": material.render_order,
        "doubleSided": material.double_sided,
        "outline": material.outline,
    }


def write_bundle(
    folder, name, base, verts, bones, weights, correctives, parts=(), normals=None, info=None, touch=None, rest=None
):
    """input.json and input.npz for sotaigen-blender's convert.py. `touch` is sotaigen.rig.touch.blend_poses's pose assets; `rest`
    is what sotaigen.rig.rest_pose.solve gives, for a file with another rest pose, without which the bundle is what it
    always was."""
    faces = base["faces"]
    names = [bone["name"] for bone in bones]
    counts, corners = counts_corners(faces)
    custom = np.zeros((len(verts), 3))
    if normals is not None:
        own = skin_normals(base, verts)
        # Only where the given normals differ from the skin's own; elsewhere Blender's stay.
        moved = np.linalg.norm(np.asarray(normals) - own, axis=1) > 1e-9
        custom[moved] = np.asarray(normals)[moved]
    targets = correctives.targets
    arrays = {
        "skin_verts": np.asarray(verts, float),
        "skin_counts": counts,
        "skin_corners": corners,
        "skin_uv": load_uv(faces),
        "skin_normals": custom,
        "head_region": np.flatnonzero(np.asarray(base["segments"]) == "head"),
        "neck_ring": neck_ring_order(base),
        "weights": np.asarray(weights, float),
        "deltas": np.array([targets[n] for n in targets], np.float32),
    }
    rows = []
    for i, part in enumerate(parts):
        prefix = f"part{i}_"
        counts, corners = counts_corners(part.faces)
        arrays[prefix + "verts"] = np.asarray(part.verts, float)
        arrays[prefix + "counts"], arrays[prefix + "corners"] = counts, corners
        if part.weights is not None:
            arrays[prefix + "weights"] = np.asarray(part.weights, float)
        if part.uvs is not None:
            # glTF's V runs down, Blender's up.
            uv = np.asarray(part.uvs, float)[corners] * (1, -1) + (0, 1)
            arrays[prefix + "uv"] = uv
        if part.normals is not None:
            arrays[prefix + "normals"] = np.asarray(part.normals, float)
        if part.morphs:
            arrays[prefix + "morphs"] = np.array([part.morphs[m] for m in part.morphs], float)
        rows.append(
            {
                "name": part.name,
                "material": material_row(part.material),
                "weights": part.weights is not None,
                "uvs": part.uvs is not None,
                "normals": part.normals is not None,
                "morphs": list(part.morphs),
            }
        )
    spec = {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "names": NAMES,
        "name": name,
        "info": {"format": FORMAT, "version": VERSION, **(info or {}), "head": head_attachments(base, verts, bones)},
        "bones": [
            {k: (np.asarray(v, float).tolist() if k in ("head", "tail", "align") else v) for k, v in b.items()}
            for b in bones
        ],
        "boneNames": names,
        "drivers": driver_table(bones),
        "correctives": {"groups": correctives.groups(), "names": list(targets)},
        "limits": joint_limits(bones),
        "limitOrders": {b["name"]: limit_order(b["name"]) for b in bones},
        "widgets": widgets(bones, weights, verts),
        "skin": {"material": SKIN},
        "parts": rows,
        "subdivision": SUBDIVISION,
        "touchPoses": touch or {},
    }
    if rest is not None:
        spec["restPose"] = rest
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "input.json").write_text(json.dumps(spec))
    np.savez(folder / "input.npz", **arrays)
    return spec


def make_blend(path, name, base, verts, bones, weights, correctives, *, parts=(), normals=None, info=None,
               broken_by=(), warnings=(), checked=("wrap", "uv"), rest_request=None):
    """Write the .blend at `path`, or raise BrokenBody when `broken_by` says the mesh is broken.
    `warnings` (diagnostics, from the checks `checked` names) go into the file with `info`, under the
    rig's custom property sotaigen, with one for each hand-to-head contact the body's arm doesn't
    reach; those it reaches go in as pose assets.
    `rest_request` is sotaigen.rig.rest_pose.solve's arguments for a file with another rest pose: PastLimits
    when it needs a control past its joint limit and doesn't allow that."""
    if broken_by:
        raise BrokenBody(name, broken_by)
    rest = None if rest_request is None else solve(bones, joint_limits(bones), **rest_request)
    checked = [*checked, "touch"]
    if rest is not None:
        warnings = [*warnings, *map(Diagnostic.from_json, (*rest["pastLimits"], *rest["unclean"]))]
        checked.append("rest")
    solved = solve_all(base, verts, bones, weights)
    info = {**(info or {}), "quality": report([*warnings, *unreached(solved)], checked)}
    with tempfile.TemporaryDirectory(prefix="blend-", dir=path.parent if path.parent.exists() else None) as scratch:
        write_bundle(
            Path(scratch), name, base, verts, bones, weights, correctives, parts, normals, info, blend_poses(solved), rest
        )
        run_blender(script("convert"), scratch, path)
    return path


def blend_path(name):
    return BUILD / "bodies" / name / BLEND_NAME


def sources_of(correctives):
    """Where each group of the file's correctives came from, by group: the .blend has them as drivers."""
    return {group["name"]: group["source"] for group in correctives.groups()}


def write_blend(body, path, *, rest=None, name=None):
    """Write a Body's rigged .blend at `path` and return the path, as docs/systems/blend.md describes it.

    Needs Blender and sotaigen-blender (BlenderMissing without either), and raises BrokenBody for a broken
    mesh and PastLimits for a rest pose past the joint limits, before Blender runs. `rest` asks for
    another rest pose, as sotaigen.rig.rest_pose.solve takes it. The collection is `name`, else the
    body's own name, else the file's. A body read from the build (body.name) says its name and rig's
    fingerprint and lists the check poses' collapsed faces; any other says what made it, and what checking
    its parameters found.
    """
    path = Path(path)
    why = can_write_blend()
    if why:
        raise BlenderMissing(why)
    base = export_base()
    name = name or body.name or path.stem
    stopped = broken(body.wrap, base["faces"])
    if stopped:
        raise BrokenBody(name, stopped)
    head = HeadShape(body.controls)
    correctives = Correctives.production(
        base, body.verts, body.bones, body.weights, field=calibrated(body.controls)[0]
    )
    if body.name:
        ensure_built(body.name, ("wrap", "rig", "pose", "metrics"))
        out = BUILD / "bodies" / body.name
        info = {**file_info(body.spec, "blend"), "correctives": sources_of(correctives)}
        poses = json.loads((out / "poses.json").read_text())
        notes = body.notes if body.notes is not None else body_notes(body.spec, body.controls)
        warnings = [*quality(body.wrap, body.verts, base["faces"], poses), *notes]
        checked = ("wrap", "uv", "parameters", "poses")
    else:
        info = {**file_info(body.spec, "blend"), "correctives": sources_of(correctives)}
        notes = body.notes if body.notes is not None else body_notes(body.spec, body.controls)
        warnings = [*quality(body.wrap, body.verts, base["faces"]), *notes]
        checked = ("wrap", "uv", "parameters")
    return make_blend(
        path, name, base, body.verts, body.bones, body.weights, correctives,
        parts=bound_parts(head, head_parts(head), body),
        normals=blended_normals(base, body.verts, head),
        info=info, warnings=warnings, checked=checked, rest_request=rest,
    )


def blend_body(name, path=None, rest_request=None):
    """A body's .blend from the build; with `rest_request`, with that rest pose instead of the A pose,
    and by default under its own file name."""
    out = path or (blend_path(name) if rest_request is None else BUILD / "bodies" / name / REST_NAME)
    return write_blend(rigged(name), out, rest=rest_request)


def main(argv=None):

    parser = argparse.ArgumentParser(prog="sotaigen-dev blend", description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", metavar="name", help="default: all of them")
    parser.add_argument("--sex", type=comma_list, help="only these sexes, as female,male")
    parser.add_argument("--group", type=comma_list, help="only these groups, as long,balanced")
    add_arguments(parser)
    parser.add_argument("--out", type=Path, help="the file to write, for one body")
    args = parser.parse_args(argv)
    try:
        names = select(args.names, args.sex, args.group)
    except ValueError as error:
        parser.error(str(error))
    if args.out and len(names) != 1:
        parser.error("--out names one file, so it takes one body")
    rest_request = request_of(args)
    why = can_write_blend()
    if why:
        print(why, file=sys.stderr)
        return 3
    failed = 0
    for name in names:
        start = time.perf_counter()
        try:
            path = blend_body(name, args.out, rest_request)
        except (BrokenBody, PastLimits) as error:
            print(error, file=sys.stderr)
            mend.tell_body(name)
            failed += 1
            continue
        print(f"{path}: {path.stat().st_size} bytes, {time.perf_counter() - start:.1f} s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
