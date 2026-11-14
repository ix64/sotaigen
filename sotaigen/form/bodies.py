"""The bodies this tool comes with, typical and test, read from their files and checked.

Typical bodies are what the page starts from; test bodies only test (docs/systems/presets.md).
"""

from functools import cache

from sotaigen.form.solve import SolvedBodies
from sotaigen.params.file import read_spec
from sotaigen.params.rules import check_bodies
from sotaigen.paths import DATA

# By sex and role: sotaigen/data/bodies/<sex>/<role>/<group>.json, each a parameter file
# (sotaigen.params.file) as a downstream tool would write it.
ROLES = ("typical", "test")
# The typical bodies' order, youngest first: the page lists them so. Test bodies follow by name.
TYPICAL_GROUPS = ("chibi", "compact", "slim", "long", "balanced", "rounded")


def body_files(role=None):
    """Each body's file as (name, path), in sex order, typical before test, typical by age."""
    found = []
    for folder in sorted(path for path in (DATA / "bodies").iterdir() if path.is_dir()):
        for kind in ROLES:
            if role not in (None, kind):
                continue
            files = sorted((folder / kind).glob("*.json"))
            if kind == "typical":
                files.sort(key=lambda path: TYPICAL_GROUPS.index(path.stem))
            found += [(f"{folder.name}-{path.stem}", path) for path in files]
    return found


def load_specs(role=None):
    """Every body as written, named <sex>-<group>: its measurements in millimetres."""
    specs = {}
    for name, path in body_files(role):
        spec = read_spec(path)
        if spec.sex != name.split("-", 1)[0]:
            raise ValueError(f"{path}: a {spec.sex} body in the {name.split('-', 1)[0]} folder")
        specs[name] = spec.body
    check_bodies(specs)
    return specs


def load_bodies():
    """Every body as the masses read it: its internal controls, solved when one is read."""

    return SolvedBodies(load_specs())


@cache
def body_specs():
    """Every body's measurements, typical and test, by name. Shared: don't change them."""
    return load_specs()
