"""Export levels: what blocks a body's export, and what only warns and travels with the file.

A body whose wrap is broken (flipped faces, vertices past the midline, an open skin) is refused, with
where it broke; quality problems go into the GLB's extras.sotaigen.quality and the .blend's rig
property, as docs/systems/export.md's 导出分级 says.
"""

import numpy as np

from sotaigen.diagnostics import Diagnostic, as_json
from sotaigen.errors import SotaigenError
from sotaigen.form.checks import field_failures, field_measures
from sotaigen.form.solve import target_misses
from sotaigen.mesh import edge_owners
from sotaigen.metrics import RestMesh, pose_row
from sotaigen.params.rules import seam_problems
from sotaigen.rig.poses import check_poses
from sotaigen.rig.skinning import pose_verts
from sotaigen.uv import corner_triangles, load_uv, stretch, stretch_report

# The UV atlas's limits, as tests/test_uv.py holds every body to; a body past them only warns.
UV_L2_STRETCH, UV_ANISOTROPY_P95 = 1.15, 2.2


class BrokenBody(SotaigenError, ValueError):
    """A body whose mesh is broken, which no one downstream can mend in Blender; the message says why,
    and `reasons` holds each of broken's diagnostics, for a page to put into its own words."""

    def __init__(self, name, reasons):
        super().__init__(f"{name}: not exported, the mesh is broken: " + "; ".join(reasons))
        self.reasons = list(reasons)


def broken_at(wrap, metric):
    """Which parts of the body a wrap's flips or crossings are in: {part: count}."""
    return dict(wrap.get("brokenAt", {}).get(metric, {}))


def wrap_broken(wrap):
    """The wrap's own breakage: flipped faces and vertices past the midline, each with where."""
    out = []
    if wrap["flippedFaces"]:
        out.append(Diagnostic("wrap.flipped", "broken", {"count": wrap["flippedFaces"], "at": broken_at(wrap, "flippedFaces")}))
    if wrap["crossedMidline"]:
        out.append(Diagnostic("wrap.midline", "broken", {"count": wrap["crossedMidline"], "at": broken_at(wrap, "crossedMidline")}))
    return out


def broken(wrap, faces):
    """What blocks the export: the mesh itself is broken (docs/DECISIONS.md, 结果违反不变量时). Each says
    where; an open skin is the template's own faces, which no parameter changes."""
    out = wrap_broken(wrap)
    open_edges = sum(1 for owners in edge_owners(faces).values() if len(owners) != 2)
    if open_edges:
        out.append(Diagnostic("skin.open", "broken", {"edges": open_edges}))
    return out


def wrap_warnings(wrap, verts):
    """What a finished wrap says that only warns: the feet off the ground and the hand and foot seams."""
    from sotaigen.form.wrap import FOOT_TO_GROUND_MM, SEAM_CREASE_DEG, SEAM_MISS_SHARE, seam_radius
    from sotaigen.template import load_template

    out = []
    if abs(wrap["footLowest"]) > FOOT_TO_GROUND_MM:
        out.append(Diagnostic("foot.off_ground", "warning", {"mm": wrap["footLowest"], "limit": FOOT_TO_GROUND_MM}))
    raw = load_template()
    for part, limit in SEAM_CREASE_DEG.items():
        if wrap[f"{part}SeamCreaseDeg"] > limit:
            out.append(Diagnostic("seam.crease", "warning", {"part": part, "degrees": wrap[f"{part}SeamCreaseDeg"], "limit": limit}))
    for part, limit in SEAM_MISS_SHARE.items():
        share = wrap[f"{part}SeamMissMm"] / seam_radius(raw, verts, part)
        if share > limit:
            out.append(Diagnostic("seam.miss", "warning", {"part": part, "share": round(float(share), 4), "limit": limit}))
    return out


def quality(wrap, verts, faces, poses=None):
    """What only warns and travels with the file: the feet on the ground, seams, collapsed faces in the
    check poses (`poses`, a build's poses.json, when there is one) and the UV atlas's stretch."""
    out = wrap_warnings(wrap, verts)
    for pose, row in (poses or {}).items():
        if row.get("collapsedFaces"):
            out.append(Diagnostic("pose.collapsed", "warning", {"pose": pose, "count": row["collapsedFaces"]}))
    triangles, _ = corner_triangles(faces)
    corner_verts = np.concatenate([np.asarray(face) for face in faces])
    report = stretch_report(*stretch(np.asarray(verts, float), load_uv(faces), corner_verts, triangles))
    if report["l2"] > UV_L2_STRETCH:
        out.append(Diagnostic("uv.stretch", "warning", {"value": round(float(report["l2"]), 4), "limit": UV_L2_STRETCH}))
    if report["anisotropyP95"] > UV_ANISOTROPY_P95:
        out.append(Diagnostic("uv.anisotropy", "warning", {"value": round(float(report["anisotropyP95"]), 4), "limit": UV_ANISOTROPY_P95}))
    return out


def parameter_notes(seams, misses, surface):
    """What checking a body's parameters found, as `sotaigen check` says it and files carry it:
    the seam rules' warnings, the targets missed ({key: (asked, reached)}) and the surface's failures."""
    return [
        *seams,
        *(
            Diagnostic("target.missed", "warning", {"key": key, "asked": round(float(asked), 3), "reached": round(float(actual), 3)})
            for key, (asked, actual) in misses.items()
        ),
        *(Diagnostic(f"surface.{name}", "warning") for name in surface),
    ]


def body_notes(spec, controls):
    """What checking a body's parameters finds beyond the rules, as parameter_notes says it: the seam
    rules' warnings, the targets missed and the surface's failures. A few seconds: it reads the field."""
    misses = target_misses(spec.body)
    surface = field_failures(field_measures(controls))
    return tuple(parameter_notes(seam_problems(spec.body, spec.sex), misses, surface))


def contact_failures(base, verts, bones, weights):
    """Which contact bakes of this rig failed. It still has a body without that protection; only for a
    body asked for with its contact tables, because without them nothing was baked."""
    from sotaigen.contact.axilla_cache import failure as shoulder_failure
    from sotaigen.contact.groin import failure as pelvic_failure
    from sotaigen.contact.hang import failure as hang_failure

    out = []
    for group, why in (
        ("axilla", shoulder_failure(base, verts, bones, weights)),
        ("groin", pelvic_failure(base, verts, bones, weights)),
        ("hang", hang_failure(base, verts, bones, weights)),
    ):
        if why:
            out.append(Diagnostic("contact.failed", "warning", {"group": group, "detail": str(why)}))
    return out


def report(found, checked):
    """A file's quality: which checks ran and what they found, as the file stores it."""
    return {"checked": list(checked), "found": as_json(found)}


def glb_quality(wrap, base, verts, bones, weights, correctives, *, posed=False, found=()):
    """What a GLB's extras.sotaigen.quality holds: `quality`, then `found` (parameter_notes), then
    any failed contact bake, with the checks that ran.

    The collapsed faces take a posing of every check pose, too long for a slider's every release, so
    only with `posed`, and always for the IPC contact tables, whose bake takes far longer anyway. They
    are measured on the file's own `correctives`, which is what poses.json measures for the default ones.
    """
    baked = correctives.contact == "ipc"
    poses = None
    if posed or baked:
        rest = RestMesh(verts, base["faces"])
        poses = {
            name: pose_row(rest, pose_verts(bones, weights, correctives.bind(verts, pose), pose))
            for name, pose in check_poses(base, verts, bones, weights).items()
        }
    failed = contact_failures(base, verts, bones, weights) if baked else []
    checked = ["wrap", "uv", "parameters", *(["poses"] if poses is not None else []), *(["contact"] if baked else [])]
    return report([*quality(wrap, verts, base["faces"], poses), *found, *failed], checked)
