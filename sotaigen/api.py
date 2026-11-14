"""The tool as a library: a body's parameters in, its rigged GLB or .blend out.

    import sotaigen

    spec = sotaigen.read_spec("my.json")        # or sotaigen.typical("female-long")
    found = sotaigen.check(spec)                # what is wrong or worth knowing, empty when nothing
    body = sotaigen.generate(spec)              # InvalidSpec when the parameters can't be made
    sotaigen.write_glb(body, "out.glb")         # contact=True needs sotaigen[contact]
    sotaigen.write_blend(body, "out.blend")     # needs Blender and sotaigen-blender

The parameter file and its meaning are docs/systems/editor.md's and docs/PARAMETERS.md's; what the
GLB and the .blend hold, docs/systems/export.md's and docs/systems/blend.md's. Everything this raises
for something it can't do is a SotaigenError, with a message that says what.
"""

from sotaigen.body import Body
from sotaigen.diagnostics import Diagnostic
from sotaigen.errors import BlenderMissing, InvalidSpec, SotaigenError
from sotaigen.form.bodies import body_specs
from sotaigen.generation import examined, generate
from sotaigen.output.blend import write_blend
from sotaigen.output.glb import write_glb
from sotaigen.output.quality import BrokenBody
from sotaigen.params.file import FileError, Spec, read_spec
from sotaigen.params.rules import sex_of
from sotaigen.rig.rest_pose import PastLimits

__all__ = [
    "BlenderMissing",
    "Body",
    "BrokenBody",
    "Diagnostic",
    "FileError",
    "InvalidSpec",
    "PastLimits",
    "SotaigenError",
    "Spec",
    "check",
    "generate",
    "read_spec",
    "typical",
    "typical_names",
    "write_blend",
    "write_glb",
]


def typical_names():
    """The names of the bodies this tool comes with, typical and test, as <sex>-<group>."""
    return list(body_specs())


def typical(name):
    """One of the bodies this tool comes with as a Spec, by name, as female-long."""
    specs = body_specs()
    if name not in specs:
        raise InvalidSpec([Diagnostic("body.unknown", "error", {"name": name, "known": list(specs)})])
    return Spec(sex_of(name), dict(specs[name]))


def check(spec, *, for_head_swap=False):
    """What is wrong with a Spec, or worth knowing, as diagnostics; empty when nothing.

    The ranges and rules first, at once; when those hold, the body is made (seconds) and what its surface
    and its wrap say follows. Each is a sotaigen.diagnostics.Diagnostic: its English sentence as a str,
    with `code`, `level` and `values`. Level "warning" doesn't stop a file being written; "error" (the
    ranges, the rules, a target past what the body can reach) stops the body and "broken" (flipped faces,
    vertices past the midline) its file. `for_head_swap` as generate's.
    """
    try:
        _, found = examined(spec, for_head_swap=for_head_swap)
    except InvalidSpec as error:
        return error.problems
    return found
