"""Bodies made from parameters in memory, as the build makes a body's, and what checking them finds.

generate(spec) is the library's way in (sotaigen.api): the rules, the solve, the wrap and the rig.
`sotaigen check`, `glb` and `blend` run it, and so does the editor's worker
(sotaigen_dev/editor/worker.py) for the page. A typical body's parameters give the bytes
uv run sotaigen-dev export writes.
"""

import os
import tempfile
import time
from functools import cache
from pathlib import Path

# One thread per process, so that a body made here doesn't take over a machine, and its arithmetic is
# the same as the build's, whose processes run single-threaded too.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np

from sotaigen.body import Body
from sotaigen.corrective.recipe import Correctives
from sotaigen.diagnostics import Diagnostic
from sotaigen.errors import InvalidSpec
from sotaigen.form.anatomy import landmarks
from sotaigen.form.solve import solved_controls
from sotaigen.form.wrap import wrap_body
from sotaigen.output import face
from sotaigen.output.glb import export_base, glb_bytes
from sotaigen.output.quality import body_notes, glb_quality, wrap_broken, wrap_warnings
from sotaigen.output.quality import contact_failures as baked_failures
from sotaigen.params.rules import NECK_THIN_FOR_HEAD, problems
from sotaigen.pipeline import rig_body, template
from sotaigen.template import load_template


@cache
def raw_template():
    return load_template()


def full_template():
    return template()


def made(p):
    """Wrap and rig internal controls, as the build does for a body, keeping nothing on disk: a Body
    of what a second process can send back (the editor's worker makes it in one), without its base,
    landmarks, spec or notes, which generate adds."""
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch)
        wrap = wrap_body(raw_template(), p, out)
        verts = np.load(out / "wrapped.npy")
    wrapped = time.perf_counter()
    bones, weights, verts = rig_body(p, verts)
    rigged = time.perf_counter()
    seconds = {"wrap": wrapped - start, "rig": rigged - wrapped}
    return Body(verts, bones, weights, wrap, seconds, controls=p)


def generate(spec, *, for_head_swap=False):
    """The Body of a sotaigen.params.file.Spec, made in memory: some seconds.

    InvalidSpec when the parameters break a range or a rule. A target past what this body can reach,
    solved or a local contour's, goes as far as it can, and the body's notes say how far. `for_head_swap` lets a neck through that is too thin for this tool's head,
    for a head that replaces it; the body's notes say so. A body whose mesh came out broken is still
    made; writing it is refused (sotaigen.output.quality.broken), as `wrap_failures` says why.
    """
    invalid = problems(spec.body, spec.sex)
    let_through = [d for d in invalid if for_head_swap and d.code == f"rule.{NECK_THIN_FOR_HEAD}"]
    invalid = [d for d in invalid if d not in let_through]
    if invalid:
        raise InvalidSpec(invalid)
    try:
        p = solved_controls(spec.body)
        notes = body_notes(spec, p)
        body = made(p)
    except ValueError as error:
        # A measurement the body can't be read at, such as a station outside the torso.
        raise InvalidSpec([Diagnostic("target.unreachable", "error", {"detail": str(error)})]) from error
    warned = tuple(Diagnostic(d.code, "warning", {**d.values, "forHeadSwap": True}) for d in let_through)
    return body._replace(base=template(), L=landmarks(p), spec=spec, notes=warned + notes)


def examined(spec, *, for_head_swap=False):
    """(the spec's Body, what `check` finds about it): its notes, then its wrap's failures."""
    body = generate(spec, for_head_swap=for_head_swap)
    return body, [*body.notes, *wrap_failures(body)]


def wrap_failures(body):
    """The invariants a finished wrap can answer at once. Posed collapses are not among them."""
    return [*wrap_broken(body.wrap), *wrap_warnings(body.wrap, body.verts)]


def glb_blob(body, field, head, parts, extras=None, contact=False, progress=None, posed=False, found=(),
             normals=None, correctives=None):
    """The GLB through the exporter's own function: skin, parts and normals, as a body's.

    `field` is the body's calibrated field, which the hanging arm's default contact reads. The
    correctives are Correctives.production's unless given, with `normals` as face.blended_normals
    makes them (the editor's worker makes both in two processes, to the same bytes). `posed` and
    `found` say what its quality list has beyond the wrap's and the atlas's, as
    sotaigen.output.quality.glb_quality takes them.
    """
    base = export_base()
    if normals is None:
        normals = face.blended_normals(base, body.verts, head)
    if correctives is None:
        correctives = Correctives.production(
            base,
            body.verts,
            body.bones,
            body.weights,
            field=field,
            contact=contact,
            progress=progress,
        )
    warnings = glb_quality(
        body.wrap, base, body.verts, body.bones, body.weights, correctives, posed=posed, found=found
    )
    return glb_bytes(
        base,
        body.verts,
        body.bones,
        body.weights,
        {**(extras or {}), "quality": warnings},
        parts=parts,
        normals=normals,
        correctives=correctives,
    )


def contact_failures(body):
    """A failed contact bake still gives a body, without that protection; the page has to say so.

    Only for a body asked for with its contact tables: without them nothing was baked.
    """
    return baked_failures(full_template(), body.verts, body.bones, body.weights)
