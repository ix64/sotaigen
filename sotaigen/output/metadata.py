"""What a GLB's extras.sotaigen and a .blend's rig["sotaigen"] say about the file itself, from one place.

Every way of making a file (the build, `sotaigen glb` and `blend`, the API and the editor) writes the
same metadata for the same parameters, so the same body gives the same bytes whichever made it; the
field table is docs/systems/export.md's (GLB 约定). Nothing here says which way it was made.
"""

import json
from functools import cache
from importlib import metadata as package_metadata

from sotaigen.params.file import write_parameter_file
from sotaigen.params.rules import sex_of

GLB_FORMAT = "sotaigen/glb"
BLEND_FORMAT = "sotaigen/blend"
# Raised when a field's meaning changes or one is taken away, the driver tables' among them; a reader
# refuses a version it doesn't know.
VERSION = 1
NAME = "sotaigen"


@cache
def package_version():
    try:
        return package_metadata.version(NAME)
    except package_metadata.PackageNotFoundError:
        return "0+unknown"


def generator():
    """What made the file: the package and its version, and the rig stage's fingerprint, which tells
    apart two files of one version whose generating code differed. Not which entry point made it."""
    from sotaigen.pipeline import rig_fingerprint

    return {"name": NAME, "version": package_version(), "fingerprint": rig_fingerprint()}


def asset_generator():
    """glTF's asset.generator: the tool and its version."""
    return f"{NAME} {package_version()}"


def body_name(spec):
    """The name of the typical or test body whose parameters these are, exactly, or None."""
    from sotaigen.form.bodies import body_specs

    for name, body in body_specs().items():
        if sex_of(name) == spec.sex and body == spec.body:
            return name
    return None


def parameters(spec):
    """The parameter file of a Spec, as an object: what the body was made from, to make it again.

    Its numbers are written as floats: JSON doesn't tell 124 from 124.0, so the page sends one where a
    body's file holds the other, and the same parameters have to give the same bytes.
    """
    body = {key: float(value) if isinstance(value, int) and not isinstance(value, bool) else value
            for key, value in spec.body.items()}
    return json.loads(write_parameter_file(body, spec.sex, spec.base, spec.unknown))


def file_info(spec, kind="glb"):
    """The fields that say what the file is and what made it: format, version, generator, the
    parameters, and the body's name when the parameters are a typical or test body's."""
    out = {
        "format": GLB_FORMAT if kind == "glb" else BLEND_FORMAT,
        "version": VERSION,
        "generator": generator(),
    }
    name = body_name(spec)
    if name is not None:
        out["name"] = name
    out["parameters"] = parameters(spec)
    return out
