"""Measure and exercise physical abdominal profile targets through the production pipeline.

Run: uv run scripts/abdomen_study.py [--meshes]
Results: build/reports/abdomen-study. Local profile controls inherit binding from their reference skin.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sotaigen.corrective.crossings import SkinParts
from sotaigen.form.abdomen import (
    AbdominalProfile,
    mesh_profile_measures,
    profile_measures,
    sagittal_depths,
)
from sotaigen.form.anatomy import calibrated
from sotaigen.form.bodies import load_bodies
from sotaigen.form.wrap import hip_front_lead, quad_normals
from sotaigen.generation import full_template, glb_blob, made
from sotaigen.mesh import triangulate
from sotaigen.metrics import RestMesh, pose_row, posed_meshes
from sotaigen.output import face
from sotaigen.rig.poses import check_poses
from sotaigen.uv import corner_triangles, load_uv, stretch, stretch_report


def run(meshes=False):
    out = ROOT / "build/abdomen-study"
    out.mkdir(parents=True, exist_ok=True)
    presets, rows = load_bodies(), {}
    levels = np.linspace(0.20, 0.78, 100)
    for name, p in presets.items():
        body, L = calibrated(p)
        measures = profile_measures(body, L)
        rows[name] = {"torsoMm": L["T"], "base": measures, "targets": []}
        for front in (-0.03, 0.0, 0.02):
            for back in (0.0, 0.03, 0.06):
                shaped = AbdominalProfile(body, L, p, front_arc=front, back_hollow=back)
                actual = profile_measures(shaped, L)
                rows[name]["targets"].append({"asked": [front, back], "actual": actual})
    (out / "measurements.json").write_text(json.dumps(rows, indent=2))
    print("Measured 12 bodies and 108 target pairs", flush=True)
    if not meshes:
        return
    template = full_template()
    faces = template["faces"]
    triangles = triangulate(faces, np.array(template["verts"]))
    quads = np.array(
        [
            f
            for f, part in zip(
                faces, np.array(template["regions"])[[f[0] for f in faces]], strict=True
            )
            if len(f) == 4 and part == "trunk"
        ]
    )
    uv, (uv_tri, _), cv = load_uv(faces), corner_triangles(faces), np.concatenate(faces)
    parts = SkinParts(template)
    results = {}
    for name in ("female-chibi", "female-long", "female-balanced"):
        p = presets[name]
        body, L = calibrated(p)
        original = made(p)
        cases = [
            ("base", {}),
            ("front-in", {"front_arc": -0.03}),
            ("front-out", {"front_arc": 0.02}),
            ("back-flat", {"back_hollow": 0.0}),
            ("back-hollow", {"back_hollow": 0.06}),
        ]
        for label, targets in cases:
            shape = AbdominalProfile(body, L, p, **targets)
            names = {"front_arc": "abdomenFrontArc", "back_hollow": "lumbarHollow"}
            adjusted = p | {names[key]: value for key, value in targets.items()}
            exact = made(adjusted) if targets else original
            vertices, bones, weights = exact.verts, exact.bones, exact.weights
            poses = check_poses(template, vertices, bones, weights)
            rest = RestMesh(vertices, faces)
            pose_results = {
                key: {
                    "collapsedFaces": pose_row(rest, posed)["collapsedFaces"],
                    "legsPastMidline": parts.legs_past_midline(posed),
                }
                for key, posed in posed_meshes(
                    template,
                    vertices,
                    bones,
                    weights,
                    {k: poses[k] for k in ("crouch", "twist", "reach", "stride")},
                    calibrated(adjusted)[0],
                )
            }
            normals = quad_normals(vertices, quads)
            outward = shape.gradient(vertices[quads].mean(1), step=1.5)
            report = stretch_report(*stretch(vertices, uv, cv, uv_tri))
            key = f"{name}-{label}"
            results[key] = {
                "targets": targets,
                "measured": profile_measures(shape, L),
                "meshMeasured": mesh_profile_measures(vertices, triangles, L),
                "hipFrontExcess": hip_front_lead(shape, L) / L["T"],
                "trunkFlipped": int((np.sum(normals * outward, axis=1) < 0).sum()),
                "uv": {k: report[k] for k in ("l2", "anisotropyP95", "densityP5", "densityP95")},
                "poses": pose_results,
            }
            np.savez(
                out / f"{key}.npz",
                v=vertices,
                f=triangles,
                levels=levels,
                profile=sagittal_depths(shape, L, levels),
                T=L["T"],
                crotch=L["crotch"],
            )
            if name == "female-long":
                head = face.HeadShape(adjusted)
                parts = face.bound_parts(head, face.head_parts(head), exact)
                blob = glb_blob(exact, calibrated(adjusted)[0], head, parts, {"experimentalAbdomenTargets": targets})
                (out / f"{key}.glb").write_bytes(blob)
            print(key, json.dumps(results[key]), flush=True)
            (out / "meshes.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", action="store_true")
    run(parser.parse_args().meshes)
