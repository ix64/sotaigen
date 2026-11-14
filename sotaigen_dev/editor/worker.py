"""The editor's local worker: the body for one set of parameters, made as sotaigen.generation makes it.

It says {"ready": true, "version": "..."} once it can work, then reads one JSON request
per line on stdin and answers one JSON line on stdout. The page reaches it through
web/serve.mjs, which kills the worker to cancel a body nobody wants any more:

    {"id": 1, "body": {...}, "out": "path/name.glb"}
        -> {"id": 1, "ok": true, "seconds": {...}, "wrap": {...}, "failures": [...], "glbBytes": N, ...}

Every finding in an answer ("problems", "failures", "broken", "seamWarnings") is a diagnostic,
{"code", "level", "values"} (sotaigen/diagnostics.py), which the page words by its code.

The answer's "broken" lists why the mesh may not be exported (sotaigen.output.quality.broken), empty when it may. A
broken body still gets its GLB, for the page to show; the page refuses to save it. With "contact": true
the file can only be for saving, so a broken body gets {"id": 1, "ok": false, "broken": [...]} and no bake.

With "contact": true the GLB has the IPC contact tables too, which takes minutes for a body not baked
before; on the way it says {"id": 1, "stage": "shoulder"}, then "pelvic" and "hang", as each starts.

`version` changes with anything that can change a GLB, so a result of an older pipeline is never
taken for this one's. Parameters that are a typical or test body's get the bytes uv run sotaigen-dev export writes.

With "blend": true it writes the body's rigged .blend instead, as `sotaigen blend` does, and
answers {"id": 1, "ok": true, "blendBytes": N, "seconds": S}; or, with "ok": false, says why not:
"noBlender": true, or "broken": [...] for a mesh too broken to export, or "problems" or "error" as
for a GLB. The ready line also says "blendVersion", which is `version` and the code that runs inside
Blender, and "blender": whether this machine has one.

With "mend": true it looks for what ends a broken body's breakage (sotaigen.mend), a wrap at a time, which
is why only a refused export asks: on the way {"id": 1, "stage": "3/35"}, then {"id": 1, "ok": true,
"mend": {...}} as sotaigen.mend.search answers, null when the body isn't flipped or crossed after all.

Run as: uv run -m sotaigen_dev.editor.worker
"""

import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import cache
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from sotaigen.corrective import axilla, deltoid, groin, hang_field, pectoral
from sotaigen.corrective import hang as hang_driver
from sotaigen.corrective.recipe import Correctives
from sotaigen.diagnostics import plain
from sotaigen.form.abdomen import AbdominalProfile, mesh_profile_measures, profile_measures
from sotaigen.form.anatomy import calibrated
from sotaigen.form.checks import field_failures, field_measures
from sotaigen.form.solve import delivered_front_arc, solved_controls, target_misses
from sotaigen.form.thigh import ThighProfile
from sotaigen.form.thigh import mesh_measures as thigh_mesh_measures
from sotaigen.generation import (
    contact_failures,
    examined,
    full_template,
    glb_blob,
    made,
    raw_template,
    wrap_failures,
)
from sotaigen.mend import search as search_mend
from sotaigen.mesh import triangulate
from sotaigen.output import face
from sotaigen.output.blend import write_blend
from sotaigen.output.blender import can_write_blend, script
from sotaigen.output.glb import corrective_normal_inputs, export_base
from sotaigen.output.metadata import body_name, file_info
from sotaigen.output.quality import BrokenBody, broken, parameter_notes
from sotaigen.params.file import Spec
from sotaigen.params.rules import problems, seam_problems
from sotaigen.pipeline import files_below, rig_fingerprint
from sotaigen.rig.touch import pose_table, solve_all
from sotaigen.sources import code_key

ROOT = Path(__file__).resolve().parents[2]


# The hanging arm's default table is solved in two parts, these forward keys in the second process
# and the others here: interleaved, so that each part has its share of the poses with contact.
SECOND_FORWARDS = hang_driver.FORWARD[1::2]


def packed(deltas):
    """{name: (rows, values)} of whole-mesh deltas, which are mostly zero, to send between processes.

    A row counts as zero only if all its bits are, so a mirrored -0.0 comes back as it went.
    """
    out = {}
    for name, delta in deltas.items():
        rows = np.flatnonzero(delta.view(np.uint8).reshape(len(delta), -1).any(1))
        out[name] = (len(delta), rows, delta[rows])
    return out


def unpacked(parts):
    out = {}
    for name, (count, rows, values) in parts.items():
        delta = np.zeros((count, 3), values.dtype)
        delta[rows] = values
        out[name] = delta
    return out


def sculpted_shoulder(verts, bones, weights):
    """In the second process: the shoulder's sculpted correctives."""
    return packed(axilla.targets(export_base(), verts, bones, weights))


def second_part(p, verts, bones, weights, shoulder, normals, triangles):
    """In the second process: its part of the hanging arm's table, with their normal deltas and the
    shoulder's."""
    base = export_base()
    shoulder = unpacked(shoulder)
    table = hang_field.Table(base, verts, bones, weights, calibrated(p)[0])
    hang = table.solve(shoulder, SECOND_FORWARDS)
    made = axilla.normal_targets(base, verts, bones, weights, shoulder, normals, triangles=triangles)
    if hang:
        made |= hang_driver.normal_targets(
            base, verts, bones, weights, shoulder, hang, table.metadata(), normals,
            triangles=triangles, names=set(hang),
        )
    return packed(hang), packed(made)


def split_correctives(p, exact, base, field, normals):
    """Correctives.production's default, made in this process and the second at once.

    The shoulder's correctives come first, from the second process; meanwhile this one sets up the
    hanging arm's table and makes the pelvis's. Then each process solves part of the table, this one
    with the pelvis's normal deltas, the second with the shoulder's.
    """
    verts, bones, weights = exact.verts, exact.bones, exact.weights
    second = second_process()
    shouldered = second.submit(sculpted_shoulder, verts, bones, weights)
    table = hang_field.Table(base, verts, bones, weights, field)
    pelvic = groin.sculpted_targets(base, verts, bones, weights)
    unit_normals, triangles = corrective_normal_inputs(base, verts, normals)
    shoulder = pectoral.follow(unpacked(shouldered.result()), base, verts, bones, field)
    shoulder = deltoid.gather(shoulder, base, verts, bones)
    packed_shoulder = packed(shoulder)
    theirs = second.submit(
        second_part, p, verts, bones, weights, packed_shoulder, unit_normals, triangles
    )
    # The second process sets up its own table first, so the pelvis's normals wait until now.
    made = groin.normal_targets(
        base, verts, bones, weights, pelvic, unit_normals, triangles=triangles
    )
    mine = table.solve(shoulder, [f for f in hang_driver.FORWARD if f not in SECOND_FORWARDS])
    if mine:
        made |= hang_driver.normal_targets(
            base, verts, bones, weights, shoulder, mine, table.metadata(), unit_normals,
            triangles=triangles, names=set(mine),
        )
    their_hang, their_normals = (unpacked(part) for part in theirs.result())
    hang = {**mine, **their_hang}
    rows = hang_driver.sample_names(table.keys) if table.keys else {}
    hang = {name: hang[name] for side in rows for _, _, name in rows[side]}
    return Correctives(
        shoulder, pelvic, hang, table.metadata(), contact="sculpted", normals=made | their_normals
    )


def generation_version():
    """A digest of everything that can change a GLB: the code this script can run, face and export
    included, which the rig's fingerprint leaves out, and every data file, the UV atlas's too."""
    # The package's data but the page's sliders, which make no GLB: the UV atlas too, which the
    # build's stages don't read but the export does.
    data = [path for name, path in files_below(ROOT / "sotaigen").items() if path.suffix != ".py" and name != "data/sliders.json"]
    digest = hashlib.sha256(code_key(["sotaigen_dev/editor/worker.py"], ROOT).encode())
    for path in sorted((*data, ROOT / "uv.lock")):
        digest.update(f"{path.relative_to(ROOT)}\n".encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def blend_version(version):
    """A .blend's digest: all that can change the body, given as `version`, and the code run in Blender:
    sotaigen-blender's scripts, not the review and test tools beside them in blender/.

    Blender's own version isn't in it, as it isn't in the build's fingerprints.
    """
    digest = hashlib.sha256(version.encode())
    for path in sorted(script("convert").parent.glob("*.py")):
        digest.update(f"sotaigen_blender/{path.name}\n".encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@cache
def second_process():
    """Where the wrap and the rig run, because the head and its parts don't wait for them.

    A process, not a thread: most of this is small NumPy calls, and two threads came out slower than
    one. web/serve.mjs kills the worker by process group, so this one goes with it.
    """
    return ProcessPoolExecutor(1, mp_context=get_context("spawn"))


def answer(request):
    """One worker request: what the rules refuse, or the exact GLB on disk and its report."""
    body, sex = request["body"], request["sex"]
    invalid = problems(body, sex)
    if invalid:
        return {"id": request.get("id"), "ok": False, "problems": invalid}
    start = time.perf_counter()
    p = solved_controls(body)
    solved = time.perf_counter()
    wrapped = second_process().submit(made, p)
    head = face.HeadShape(p)
    headed = time.perf_counter()
    parts = face.head_parts(head)
    parted = time.perf_counter()
    surface = field_failures(field_measures(p))
    checked = time.perf_counter()
    # While the wrap and the rig run: the hanging arm's contact reads this field, and so does the
    # report's profile below.
    body_field, L = calibrated(p)
    fielded = time.perf_counter()
    exact = wrapped.result()
    parts = face.bound_parts(head, parts, exact)
    joined = time.perf_counter()
    contact = bool(request.get("contact"))
    stopped = broken(exact.wrap, full_template()["faces"])
    # A file only for saving, which this one may not be, and a bake of minutes isn't started for it.
    if contact and stopped:
        return {"id": request.get("id"), "ok": False, "broken": stopped}

    def progress(stage):
        print(json.dumps({"id": request.get("id"), "stage": stage}), flush=True)

    misses = target_misses(body)
    seams = seam_problems(body, sex)
    extras = file_info(Spec(sex, body))
    # A typical or test body's bytes are what uv run sotaigen-dev export writes, so it says what that says. Any other body has no
    # poses measured: that would take the slider's release past its speed target.
    # The default correctives in two processes at once, to the bytes Correctives.production makes.
    normals = face.blended_normals(export_base(), exact.verts, head)
    correctives = None if contact else split_correctives(p, exact, export_base(), body_field, normals)
    blob = glb_blob(
        exact, body_field, head, parts, extras, contact, progress,
        posed="name" in extras, found=[*stopped, *parameter_notes(seams, misses, surface)],
        normals=normals, correctives=correctives,
    )
    Path(request["out"]).write_bytes(blob)
    done = time.perf_counter()
    seconds = {
        **exact.seconds,
        "solve": solved - start,
        "head": headed - solved,
        "parts": parted - headed,
        "surface": checked - parted,
        "field": fielded - checked,
        "export": done - joined,
        "total": done - start,
    }
    thigh = body_field if isinstance(body_field, ThighProfile) else ThighProfile(body_field, L)
    triangles = triangulate(full_template()["faces"], exact.verts)
    thigh_profile = {
        "reference": thigh_mesh_measures(thigh.to_reference(exact.verts), triangles, L),
        "measured": thigh_mesh_measures(exact.verts, triangles, L),
        "limits": thigh.limits(),
        "torsoMm": L["T"],
    }
    if isinstance(body_field, ThighProfile):
        body_field = body_field.reference
    reference = body_field.reference if isinstance(body_field, AbdominalProfile) else body_field
    profile = {
        # A slider's start, so on the skin its target is measured on, as the solve aims it.
        "reference": profile_measures(reference, L) | {"frontArc": delivered_front_arc(p)},
        "measured": mesh_profile_measures(
            exact.verts, triangulate(full_template()["faces"], exact.verts), L
        ),
        "torsoMm": L["T"],
    }
    return {
        "id": request.get("id"),
        "ok": True,
        "abdomen": profile,
        "thigh": thigh_profile,
        "seconds": {key: round(value, 3) for key, value in seconds.items()},
        "wrap": exact.wrap,
        "targets": misses,
        "failures": [
            *parameter_notes([], misses, surface),
            *wrap_failures(exact),
            *(contact_failures(exact) if contact else []),
        ],
        "broken": stopped,
        "seamWarnings": seams,
        "glbBytes": len(blob),
        "contact": "ipc" if contact else "sculpted",
    }


def answer_blend(request):
    """One worker request for a .blend: what `sotaigen blend` writes for this body.

    The collection's name follows from the parameters, so that one cached file is right for whoever
    asks: the typical or test body's name for its parameters, "body" for any others.
    """
    body, sex, asked = request["body"], request["sex"], request.get("id")
    invalid = problems(body, sex)
    if invalid:
        return {"id": asked, "ok": False, "problems": invalid}
    if can_write_blend():
        return {"id": asked, "ok": False, "noBlender": True}
    start = time.perf_counter()
    made_body, _ = examined(Spec(sex, body))
    out = Path(request["out"])
    try:
        write_blend(made_body, out, name=body_name(Spec(sex, body)) or "body")
    except BrokenBody as error:
        return {"id": asked, "ok": False, "broken": error.reasons}
    except RuntimeError as error:
        # Blender's own output, which is long, goes to the terminal; the page gets the first line.
        print(error, file=sys.stderr, flush=True)
        return {"id": asked, "ok": False, "error": str(error).splitlines()[0]}
    return {
        "id": asked,
        "ok": True,
        "blendBytes": out.stat().st_size,
        "seconds": round(time.perf_counter() - start, 3),
    }


def answer_touch(request):
    """One worker request for a body's contact poses: the angles sotaigen/rig/touch.py solves, as a JSON file."""
    body, sex, asked = request["body"], request["sex"], request.get("id")
    invalid = problems(body, sex)
    if invalid:
        return {"id": asked, "ok": False, "problems": invalid}
    start = time.perf_counter()
    made_body, _ = examined(Spec(sex, body))
    solved = solve_all(export_base(), made_body.verts, made_body.bones, made_body.weights)
    Path(request["out"]).write_text(json.dumps(pose_table(solved)))
    return {"id": asked, "ok": True, "seconds": round(time.perf_counter() - start, 3)}


def answer_mend(request):
    """One worker request for what ends a broken body's breakage. Like any, the server can kill it."""
    body, sex, asked = request["body"], request["sex"], request.get("id")
    invalid = problems(body, sex)
    if invalid:
        return {"id": asked, "ok": False, "problems": invalid}

    def progress(tried, total):
        print(json.dumps({"id": asked, "stage": f"{tried}/{total}"}), flush=True)

    return {"id": asked, "ok": True, "mend": search_mend(body, sex, progress=progress)}


def serve():
    version = generation_version()
    rig_fingerprint()
    second_process().submit(raw_template).result()
    ready = {
        "ready": True,
        "version": version,
        "blendVersion": blend_version(version),
        "blender": can_write_blend() is None,
    }
    print(json.dumps(ready), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        try:
            if request.get("touch"):
                row = answer_touch(request)
            elif request.get("mend"):
                row = answer_mend(request)
            else:
                row = answer_blend(request) if request.get("blend") else answer(request)
        except Exception as error:  # noqa: BLE001 - the page shows it, and the worker stays up
            row = {
                "id": request.get("id"),
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
            if isinstance(error, BrokenProcessPool):
                second_process.cache_clear()
        print(json.dumps(plain(row)), flush=True)


if __name__ == "__main__":
    serve()
