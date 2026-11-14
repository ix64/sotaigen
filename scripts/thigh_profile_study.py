"""Exercise the outer thigh arc at the ends this body's profile reaches.

Uses all body/style pairs by default. A subset may be passed as positional names.
Outputs build/reports/thigh-profile-front-study/report.json and production mesh snapshots.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from sotaigen.corrective.crossings import SkinParts
from sotaigen.form.anatomy import calibrated
from sotaigen.form.bodies import load_bodies
from sotaigen.form.thigh import ThighProfile, mesh_measures
from sotaigen.generation import full_template, made, wrap_failures
from sotaigen.mesh import triangulate
from sotaigen.metrics import RestMesh, pose_row
from sotaigen.rig.poses import check_poses
from sotaigen.rig.skinning import pose_verts
from sotaigen.uv import corner_triangles, load_uv, stretch, stretch_report

OUT = Path(__file__).resolve().parents[1] / "build/thigh-profile-front-study"


def study(names):
    OUT.mkdir(parents=True, exist_ok=True)
    template = full_template()
    faces = template["faces"]
    parts = SkinParts(template)
    uv = load_uv(faces)
    uv_tri, _ = corner_triangles(faces)
    cv = np.concatenate(faces)
    rows = {}
    for name in names:
        p = load_bodies()[name]
        body, L = calibrated(p)
        profile = ThighProfile(body, L)
        original = made(p)
        variants = {"base": {}}
        for a in profile._arc_domain(0.0):
            variants[f"a{a:+}"] = {"outerThighArc": profile._measure(a, 0.0)["outerArc"]}
        for case, controls in variants.items():
            exact = made(p | controls) if controls else original
            vertices = exact.verts
            tri = triangulate(faces, vertices)
            actual = mesh_measures(vertices, tri, L)
            rest = RestMesh(vertices, faces)
            poses = check_poses(template, vertices, exact.bones, exact.weights)
            poses["lift"] = {"leftUpperLeg": (65, 0, 0), "leftLowerLeg": (-75, 0, 0)}
            result = {
                "targets": controls,
                "actual": actual,
                "wrap": wrap_failures(exact),
                "uv": stretch_report(*stretch(vertices, uv, cv, uv_tri)),
                "poses": {},
            }
            if controls:
                result["errorsMm"] = {
                    "arc": (actual["outerArc"] - controls["outerThighArc"]) * L["T"],
                }
                result["boneDeltaMm"] = max(
                    float(np.linalg.norm(a["head"] - b["head"]))
                    for a, b in zip(exact.bones, original.bones, strict=True)
                )
                result["weightDelta"] = float(np.abs(exact.weights - original.weights).max())
            for pose in ("crouch", "stride", "lift"):
                v = pose_verts(exact.bones, exact.weights, vertices, poses[pose])
                result["poses"][pose] = {
                    "collapsedFaces": pose_row(rest, v)["collapsedFaces"],
                    "legsPastMidline": parts.legs_past_midline(v),
                }
            rows[f"{name}/{case}"] = result
            if name == "female-long":
                np.savez(
                    OUT / f"{name}-{case}.npz", v=vertices, f=tri, T=L["T"], crotch=L["crotch"]
                )
            print(
                name,
                case,
                result["wrap"],
                result["poses"],
                "uv",
                result["uv"]["anisotropyP95"],
                flush=True,
            )
            (OUT / "report.json").write_text(json.dumps(rows, indent=2) + "\n")

    verify(rows)


def verify(rows):
    failures = []
    for name, result in rows.items():
        baseline = rows[name.split("/")[0] + "/base"]
        if result["wrap"]:
            failures.append(f"{name}: {result['wrap']}")
        uv = result["uv"]
        if not (
            uv["l2"] <= 1.15
            and uv["anisotropyP95"] <= 2.2
            and uv["densityP5"] >= 0.6
            and uv["densityP95"] <= 1.8
        ):
            failures.append(f"{name}: UV stretch {uv}")
        if result.get("boneDeltaMm", 0) > 1e-7 or result.get("weightDelta", 0) > 1e-7:
            failures.append(f"{name}: reference binding changed")
        for pose, counts in result["poses"].items():
            if (
                counts["legsPastMidline"]
                or counts["collapsedFaces"] > baseline["poses"][pose]["collapsedFaces"]
            ):
                failures.append(f"{name}/{pose}: {counts}")
    if failures:
        raise AssertionError("\n".join(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*")
    args = parser.parse_args()
    study(args.names or list(load_bodies()))
