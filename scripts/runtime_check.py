"""Export bodies and check them in a real browser: the glTF Validator, three.js against NumPy.

Needs Google Chrome and the page's npm packages, so it is not part of `uv run sotaigen-dev test`:

    pnpm install
    pnpm run runtime-check [body ...]

The default is all bodies and all check poses. Each body is checked as `uv run sotaigen-dev export`
writes it, with the head's real parts and expressions: every mesh under every check pose with morph
weights, the normals three.js holds, whether they reach the image, and the page driven by hand. The
skin alone, as --no-parts writes it, goes to the Validator and is what the normals are told from.

It exits with 1 if the Validator reports an error, the page logs one, a vertex is further than 1 mm
from sotaigen.rig.skinning.pose_verts, a normal is not the one given, or a control on the page does nothing. The
full numbers land in build/reports/runtime-check/report.json and the screenshots in
build/reports/runtime-check/shots/.
"""

import argparse
import gzip
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sotaigen.corrective.recipe import Correctives
from sotaigen.output.face import face_parts, head_normals
from sotaigen.output.glb import (
    export_body,
    export_with_face,
    glb_bytes,
    joints_and_weights,
    read_glb,
    split_uv_seams,
    stored_normals,
    to_gltf,
    vertex_normals,
)
from sotaigen.output.metadata import file_info
from sotaigen.pipeline import BUILD, bodies, rigged, surface
from sotaigen.rig.poses import POSES, check_poses
from sotaigen.rig.skinning import pose_verts
from sotaigen.uv import export_fields

OUT = BUILD / "reports" / "runtime-check"
RUNTIME = ROOT / "web" / "runtime"
LIMIT_MM = 1.0
READINGS = ("gpu", "cpu", "clip")
# Morph weights to check under: none, and every expression at once, at different weights.
SETTINGS = {
    "zero": {},
    "mix": {"blink": 0.3, "happy": 0.4, "aa": 1.0},
}
# A reading with the bones left still, or the morph weights left out, must miss by this much, in mm,
# or the check has not looked at what it claims.
BLIND_MM = 5.0
UNMORPHED_BLIND_MM = 0.5
# Given normals must match what the file was given this closely, in degrees, and differ from the
# plain file's by far more.
NORMAL_LIMIT_DEGREES = 0.01
GIVEN_NORMALS_MUST_MOVE_DEGREES = 5.0


def write_references(name):
    """Each check pose as NumPy skins it, in glTF's axes and mm, as little-endian float64."""
    rig = rigged(name)
    faces = rig.base["faces"]
    fields = export_fields(faces)
    # The GLB copies a vertex at each UV seam, and every copy must follow its original.
    source, _, _ = split_uv_seams(faces, len(rig.verts), fields["uvs"], fields["faceUvs"])
    folder = OUT / "reference" / name
    folder.mkdir(parents=True, exist_ok=True)
    correctives = Correctives.production(
        rig.base, rig.verts, rig.bones, rig.weights, field=surface(name)[0]
    )
    for pose, rotations in check_poses(rig.base, rig.verts, rig.bones, rig.weights).items():
        posed = pose_verts(rig.bones, rig.weights, correctives.bind(rig.verts, rotations), rotations)
        posed = posed[source]
        (folder / f"{pose}.f64").write_bytes((to_gltf(posed) * 1000.0).astype("<f8").tobytes())


def part_weights(part, names):
    if part.weights is not None:
        return part.weights
    weights = np.zeros((len(part.verts), len(names)))
    weights[:, names.index("head")] = 1.0
    return weights


def part_normals(part):
    if part.normals is not None:
        return stored_normals(part.normals)
    triangles = np.array(
        [(face[0], face[i], face[i + 1]) for face in part.faces for i in range(1, len(face) - 1)]
    )
    return vertex_normals(to_gltf(part.verts), triangles)


def export_both(name):
    """Write what the command writes, and the skin alone beside it for telling the normals apart."""
    plain = OUT / "plain" / f"{name}.glb"
    plain.parent.mkdir(parents=True, exist_ok=True)
    os.replace(export_body(name), plain)
    export_with_face(name)
    return plain


def write_part_references(name):
    """Each mesh of the exported file as NumPy skins it, pose after pose, for each morph setting."""
    rig = rigged(name)
    parts = face_parts(name)
    names = [bone["name"] for bone in rig.bones]
    faces = rig.base["faces"]
    fields = export_fields(faces)
    source, _, _ = split_uv_seams(faces, len(rig.verts), fields["uvs"], fields["faceUvs"])
    correctives = Correctives.production(
        rig.base, rig.verts, rig.bones, rig.weights, field=surface(name)[0]
    )
    meshes = [("body", rig.verts, np.asarray(rig.weights), correctives.targets, source)]
    meshes += [
        (part.name, part.verts, part_weights(part, names), part.morphs, np.arange(len(part.verts)))
        for part in parts
    ]
    folder = OUT / "reference" / "parts" / name
    folder.mkdir(parents=True, exist_ok=True)
    poses = check_poses(rig.base, rig.verts, rig.bones, rig.weights)
    for setting, weights in SETTINGS.items():
        rows = []
        for pose in poses.values():
            combined = weights | correctives.pose_weights(pose)
            for _, verts, skin, morphs, order in meshes:
                moved = verts + sum(
                    (combined.get(target, 0.0) * offsets for target, offsets in morphs.items()),
                    start=np.zeros_like(verts),
                )
                posed = pose_verts(rig.bones, skin, moved, pose)[order]
                rows.append((to_gltf(posed) * 1000.0).astype("<f8"))
        (folder / f"{setting}.f64").write_bytes(np.concatenate(rows).tobytes())
    normals = [stored_normals(head_normals(name))[source], *map(part_normals, parts)]
    (folder / "normals.f32").write_bytes(np.concatenate(normals).astype("<f4").tobytes())
    manifest = {
        "settings": SETTINGS,
        "poses": list(POSES),
        "meshes": [
            {"name": row[0], "count": len(row[4]), "targets": list(row[3])} for row in meshes
        ],
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))


def stored_weight_error(name):
    """The worst vertex shift from storing weights in 16 bits, by pose, with NumPy on both sides."""
    rig = rigged(name)
    joints, steps = joints_and_weights(np.asarray(rig.weights))
    stored = np.zeros_like(rig.weights)
    np.add.at(stored, (np.arange(len(stored))[:, None], joints), steps / 65535.0)
    return {
        pose: float(
            np.linalg.norm(
                pose_verts(rig.bones, stored, rig.verts, rotations)
                - pose_verts(rig.bones, rig.weights, rig.verts, rotations),
                axis=1,
            ).max()
        )
        for pose, rotations in POSES.items()
    }


def future_fields_glb(name):
    """A GLB with stand-in UVs (with a seam) and a morph target, so the Validator sees those paths."""
    rig = rigged(name)
    base = dict(rig.base)
    faces = base["faces"]
    base["uvs"] = np.column_stack(
        (np.linspace(0, 1, len(rig.verts)), np.linspace(1, 0, len(rig.verts)))
    ).tolist() + [[0.5, 0.5]]
    seam = len(rig.verts)
    base["faceUvs"] = [list(face) for face in faces]
    base["faceUvs"][0] = [seam, *faces[0][1:]]
    offsets = np.zeros_like(rig.verts)
    offsets[:, 2] = np.linspace(0, 1, len(rig.verts))
    blob = glb_bytes(
        base, rig.verts, rig.bones, rig.weights, file_info(rig.spec), {"standIn": offsets}
    )
    path = OUT / "future-fields.glb"
    path.write_bytes(blob)
    return path


def file_costs(name):
    """The file without the parts, with them, and what each expression adds, in bytes."""
    rig = rigged(name)
    base = {**rig.base, **export_fields(rig.base["faces"])}
    parts = face_parts(name)
    normals = head_normals(name)
    correctives = Correctives.production(
        rig.base, rig.verts, rig.bones, rig.weights, field=surface(name)[0]
    )

    def size(these, given=None):
        return len(
            glb_bytes(
                base,
                rig.verts,
                rig.bones,
                rig.weights,
                parts=these,
                normals=given,
                correctives=correctives,
            )
        )

    without_targets = size([replace(part, morphs={}) for part in parts], normals)
    targets = {}
    for target in dict.fromkeys(target for part in parts for target in part.morphs):
        only = [
            replace(part, morphs={k: v for k, v in part.morphs.items() if k == target})
            for part in parts
        ]
        moved = sum(
            int((np.abs(part.morphs[target]) > 1e-4).any(axis=1).sum())
            for part in parts
            if target in part.morphs
        )
        targets[target] = {"bytes": size(only, normals) - without_targets, "movedVerts": moved}
    return {
        "plainBytes": size(()),
        "withPartsNoTargetsBytes": without_targets,
        "withPartsBytes": size(parts, normals),
        "partVerts": {part.name: len(part.verts) for part in parts},
        "targets": targets,
    }


def size_table(names):
    rows = []
    for name in names:
        blob = (BUILD / "bodies" / name / "body.glb").read_bytes()
        document, data = read_glb(blob)
        views = document["bufferViews"]
        accessors = document["accessors"]
        attributes = document["meshes"][0]["primitives"][0]["attributes"]
        parts = {
            key: views[accessors[i]["bufferView"]]["byteLength"] for key, i in attributes.items()
        }
        parts["indices"] = views[
            accessors[document["meshes"][0]["primitives"][0]["indices"]]["bufferView"]
        ]["byteLength"]
        parts["inverseBind"] = views[
            accessors[document["skins"][0]["inverseBindMatrices"]]["bufferView"]
        ]["byteLength"]
        rows.append(
            {
                "preset": name,
                "bytes": len(blob),
                "gzipBytes": len(gzip.compress(blob, 6, mtime=0)),
                "jsonBytes": len(blob) - len(data) - 28,
                "parts": parts,
                **file_costs(name),
            }
        )
    return rows


def part_problems(report):
    """Print the parts' section of the report and return what fails."""
    problems = []
    print("\nEvery mesh: worst vertex in mm, three.js against NumPy, all poses and morph settings")
    print(
        "  body           mesh          verts   gpu     cpu     no weights  bones still  normal"
    )
    for row in report["parts"]:
        for mesh, values in row["meshes"].items():
            normal = row["normals"][mesh]["maxDegrees"]
            print(
                f"  {row['preset']:16} {mesh:13} {values['count']:5} {values['gpu']:.4f}  "
                f"{values['cpu']:.4f}  {values['unmorphed']:10.2f}  {values['frozen']:11.2f}  "
                f"{normal:.4f}"
            )
            for reading in ("gpu", "cpu"):
                if values[reading] > LIMIT_MM:
                    problems.append(
                        f"{row['preset']} {mesh} {reading}: {values[reading]:.3f} mm "
                        f"at {values['worstAt']}"
                    )
            if values["frozen"] < BLIND_MM:
                problems.append(f"{row['preset']} {mesh}: holding the bones still barely differs")
            if normal > NORMAL_LIMIT_DEGREES:
                problems.append(f"{row['preset']} {mesh}: normals off by {normal:.3f} degrees")
            if values["hasTargets"] and values["unmorphed"] < UNMORPHED_BLIND_MM:
                problems.append(f"{row['preset']} {mesh}: leaving out the weights barely differs")
        moved = row["normals"]["body"]["fromPlainDegrees"]
        pixels = row["pixels"]
        print(
            f"  {row['preset']:16} head normals are up to {moved:.1f} degrees from the plain file's; "
            f"{pixels['givenAgainstPlain']} of {pixels['size'] ** 2} pixels differ, "
            f"{pixels['swappedAgainstPlain']} with the plain normals copied back"
        )
        if moved < GIVEN_NORMALS_MUST_MOVE_DEGREES:
            problems.append(f"{row['preset']}: given normals are the plain ones")
        if pixels["givenAgainstPlain"] < 1000:
            problems.append(f"{row['preset']}: given normals do not reach the image")
        if pixels["swappedAgainstPlain"] > 0:
            problems.append(f"{row['preset']}: the image differs for more than the normals")
    return problems


def look_problems(report, preset):
    """Print what each look costs and return what fails. What the page's controls reach is for
    pnpm run web-check."""
    problems = []
    # In the toon look too, the skin and what shares its colour have the file's one colour.
    document, _ = read_glb((BUILD / "bodies" / preset / "body.glb").read_bytes())
    skin = document["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"][:3]
    for name, row in report["lookCost"].items():
        if row["look"] != name:
            problems.append(f"look: asked for {name} and got {row['look']}")
    for mesh in ("body", "nose", "ears"):
        if report["lookCost"]["toon"]["colors"][mesh] != skin:
            problems.append(f"look: the file's skin colour isn't on {mesh}")
    print("Each look per frame, drawn and waited for:")
    for name, row in report["lookCost"].items():
        print(
            f"  {name:18} {row['ms']:.2f} ms, {row['calls']} draw calls, {row['triangles']} "
            f"triangles, first frame {row['firstFrameMs']:.0f} ms "
            f"(slowest {row['firstFrameMsMax']:.0f})"
        )
    return problems


def segment_of(name, vertex):
    return rigged(name).base["segments"][vertex]


def print_validation(report):
    print("glTF Validator")
    for row in report["validation"]:
        print(
            f"  {Path(row['path']).parent.name}/{Path(row['path']).name}: {row['numErrors']} errors, "
            f"{row['numWarnings']} warnings, {row['numInfos']} infos, {row['numHints']} hints"
        )
        for message in row["messages"]:
            print(f"    [{message['severity']}] {message['code']} {message['pointer']}")
    print(f"  version {report['validation'][0]['validatorVersion']}")


def print_consistency(report, weight_errors):
    print(f"\nChrome {report['browser']}, WebGL: {report['consistency'][0]['glRenderer']}")
    print(
        "Worst vertex in mm, three.js against NumPy, by body (gpu cpu clip; 16-bit weights alone)"
    )
    worst = {reading: (0.0, None) for reading in READINGS}
    for row in report["consistency"]:
        name = row["preset"]
        tops = {}
        for reading in READINGS:
            pose, values = max(row["poses"].items(), key=lambda item: item[1][reading]["maxMm"])
            tops[reading] = values[reading]["maxMm"]
            if values[reading]["maxMm"] > worst[reading][0]:
                worst[reading] = (
                    values[reading]["maxMm"],
                    (name, pose, segment_of(name, values[reading]["worstVertex"])),
                )
        print(
            f"  {name:16} "
            + " ".join(f"{tops[reading]:.4f}" for reading in READINGS)
            + f"; {max(weight_errors[name].values()):.4f}"
        )
    print("Worst by reading:")
    for reading, (size, where) in worst.items():
        print(f"  {reading}: {size:.4f} mm at {where}")
    print("Worst vertex by pose over all bodies, in mm (gpu cpu clip; 16-bit weights alone):")
    for pose in POSES:
        cells = [
            max(row["poses"][pose][reading]["maxMm"] for row in report["consistency"])
            for reading in READINGS
        ]
        alone = max(errors[pose] for errors in weight_errors.values())
        print(f"  {pose:12} " + " ".join(f"{cell:.4f}" for cell in cells) + f"; {alone:.4f}")
    undriven = {
        pose: max(row["poses"][pose]["undriven"]["maxMm"] for row in report["consistency"])
        for pose in POSES
    }
    print("Without the driver, the worst vertex by pose in mm:")
    print("  " + ", ".join(f"{pose} {size:.1f}" for pose, size in undriven.items()))
    print(
        "setPose per frame, ms: "
        + ", ".join(f"{row['preset']} {row['setPoseMs']:.3f}" for row in report["consistency"][:3])
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", metavar="name", help="default: all of them")
    args = parser.parse_args(argv)
    names = args.names or list(bodies())
    if not (ROOT / "web" / "node_modules").is_dir():
        print("Install the pages' packages first: pnpm install")
        return 2
    if not (BUILD / "web" / "app" / "index.html").is_file():
        print("Build the pages first: pnpm run web-build, which pnpm run runtime-check does itself")
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    plain_files = []
    for name in names:
        plain_files.append(export_both(name))
        write_references(name)
        write_part_references(name)
    weight_errors = {name: stored_weight_error(name) for name in names}
    extras = [future_fields_glb(names[0]), *plain_files]

    report_path = OUT / "report.json"
    report_path.unlink(missing_ok=True)
    command = [
        "node",
        str(RUNTIME / "check.mjs"),
        "--names",
        ",".join(names),
        *(arg for path in extras for arg in ("--validate", str(path))),
        "--out",
        str(report_path),
        "--shots",
        str(OUT / "shots"),
    ]
    subprocess.run(command, check=True, cwd=ROOT)
    report = json.loads(report_path.read_text())
    report["sizes"] = size_table(names)
    report["weightQuantisationMm"] = weight_errors
    report_path.write_text(json.dumps(report, indent=1))

    print_validation(report)
    print_consistency(report, weight_errors)
    parts = part_problems(report)
    page = look_problems(report, "female-long" if "female-long" in names else names[0])
    print("\nSize of each GLB")
    for row in report["sizes"]:
        print(
            f"  {row['preset']:16} {row['bytes'] / 1e6:.3f} MB, gzip {row['gzipBytes'] / 1e6:.3f} MB, "
            f"JSON {row['jsonBytes'] / 1e3:.1f} kB"
        )
    first = report["sizes"][0]
    print(
        f"  {first['preset']}: skin alone {first['plainBytes']}, with parts and no targets "
        f"{first['withPartsNoTargetsBytes']}, with targets {first['withPartsBytes']}"
    )
    print(f"  the skin's mesh of {first['preset']}: {first['parts']}")
    for target, cost in first["targets"].items():
        print(f"  {target:10} {cost['bytes']:6} bytes, {cost['movedVerts']:4} vertices move")
    print(
        f"\nLoad, median of {report['loads'][0]['runs'] if report['loads'] else 0} new browsers (ms)"
    )
    for row in report["loads"]:
        print(
            f"  {row['preset']:16} fetch {row['fetchMs']:.1f}, parse {row['parseMs']:.1f}, "
            f"first frame {row['firstFrameMs']:.1f}, total {row['totalMs']:.1f} "
            f"(slowest {row['totalMsMax']:.1f})"
        )
    print("\nScreenshots:")
    for path in report["shots"]:
        print(f"  {path}")
    print(f"Full report: {report_path}")

    problems = parts + page + [f"page: {text}" for text in report["pageProblems"]]
    problems += [
        f"validator: {row['path']} has {row['numErrors']} errors"
        for row in report["validation"]
        if row["numErrors"]
    ]
    for row in report["consistency"]:
        for pose, values in row["poses"].items():
            problems += [
                f"{row['preset']} {pose} {reading}: {values[reading]['maxMm']:.3f} mm"
                for reading in READINGS
                if values[reading]["maxMm"] > LIMIT_MM
            ]
    # A check that can't tell driven from undriven has not looked at the skin.
    for row in report["consistency"]:
        if max(values["undriven"]["maxMm"] for values in row["poses"].values()) < 10 * LIMIT_MM:
            problems.append(f"{row['preset']}: the undriven reading barely differs")
    for problem in problems:
        print(f"FAILED {problem}")
    print("FAILED" if problems else "PASSED")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
