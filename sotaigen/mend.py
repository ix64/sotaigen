"""Which parameter, moved back to a typical body's, ends a broken mesh.

Only trying says so: where a mesh breaks often isn't the parameter's group (an arm's flips come from
the shoulders or the waist), and a parameter near the end of its range says little, because a few flipped
faces can be ended by a fair share of all parameters. So when a body has been refused this wraps it again
with one parameter at a time back at the balanced body's share of its unit, until a wrap holds. The
places the mesh broke go first, then each parameter by how near an end of its range it stands. A body
usually has several such parameters, so what this finds is one that ends it, not the cause. Moves the
rules or the masses would refuse aren't tried. Each try is a solve and a wrap, some seconds, and every
parameter may be tried, so nothing runs this but a refusal (docs/systems/export.md, 导出分级).
"""

import json
import sys
import tempfile
from functools import cache
from pathlib import Path

from sotaigen.form.bodies import load_specs
from sotaigen.form.solve import solved_controls
from sotaigen.form.wrap import wrap_body
from sotaigen.params.rules import problems, sex_of, spec_bounds
from sotaigen.params.spec import MILLIMETRES, SEXES, units
from sotaigen.paths import DATA
from sotaigen.pipeline import BUILD
from sotaigen.template import load_template

TYPICAL_GROUP = "balanced"
SLIDERS = DATA / "sliders.json"
# Parameters no slider sets by their own name, under the group whose sliders move them or where they
# work: the two last through the group of macro sliders.
UNSHOWN = {
    "waistRise": "waist",
    "waistDepth": "waist",
    "definition": "quick",
    "softness": "quick",
}


@cache
def parameter_parts():
    """Each parameter's group on the page, as sotaigen/data/sliders.json has them."""
    groups = json.loads(SLIDERS.read_text())["groups"]
    return {**UNSHOWN, **{key: group["id"] for group in groups for key in group["sliders"]}}


@cache
def typical_bodies():
    return {name: spec for name, spec in load_specs("typical").items() if name.endswith(f"-{TYPICAL_GROUP}")}


def moves(spec, sex, broken_parts):
    """Each parameter's move back to the typical body's share, as (key, value), in the order to try."""
    typical = typical_bodies()[f"{sex}-{TYPICAL_GROUP}"]
    unit, unit_typical, table = units(spec), units(typical), SEXES[sex]
    bounds, parts = spec_bounds(spec, sex), parameter_parts()
    found = []
    for key in spec.keys() & typical.keys():
        name = table[key].unit
        value = typical[key] / unit_typical[name] * unit[name]
        if abs(value - spec[key]) < 1e-9 or problems({**spec, key: value}, sex):
            continue
        low, high = bounds[key]
        from_middle = abs((spec[key] - low) / (high - low) - 0.5)
        rank = broken_parts.index(parts[key]) if parts[key] in broken_parts else len(broken_parts)
        found.append((rank, -from_middle, key, value))
    return [(key, value) for _, _, key, value in sorted(found)]


def wrap_of(spec):
    with tempfile.TemporaryDirectory() as scratch:
        return wrap_body(load_template(), solved_controls(spec), Path(scratch))


def is_whole(wrap):
    return not wrap["flippedFaces"] and not wrap["crossedMidline"]


def holds(spec):
    """Whether this body's wrap comes out unbroken. A target its masses can't reach is no answer."""
    try:
        return is_whole(wrap_of(spec))
    except ValueError:
        return False


def broken_parts(wrap):
    """The parts a wrap broke in, most broken first."""
    places = wrap.get("brokenAt", {})
    counts = {}
    for tally in places.values():
        for part, count in tally.items():
            counts[part] = counts.get(part, 0) + count
    return sorted(counts, key=lambda part: (-counts[part], part))


def search(spec, sex, wrap=None, progress=None):
    """What ends this body's breakage, or None when flips and crossings aren't what is wrong with it.

    `wrap` is its wrap.json if at hand; `progress(tried, total)` hears of each try. The answer has `tried`
    and `total`, and, when one was found, the parameter's `key` and `part`, its value `before` and
    `after`, and `unit`.
    """
    wrap = wrap_of(spec) if wrap is None else wrap
    if is_whole(wrap):
        return None
    todo = moves(spec, sex, broken_parts(wrap))
    for tried, (key, value) in enumerate(todo, 1):
        if progress:
            progress(tried, len(todo))
        if holds({**spec, key: value}):
            return {
                "key": key,
                "part": parameter_parts()[key],
                "before": round(spec[key], 2),
                "after": round(value, 2),
                "unit": "mm" if key in MILLIMETRES else "",
                "tried": tried,
                "total": len(todo),
            }
    return {"tried": len(todo), "total": len(todo)}


def sentence(found):
    """What `search` found, for a command's output."""
    if "key" not in found:
        return f"no one parameter back at the typical body's share ends it: {found['total']} tried"
    unit = f" {found['unit']}" if found["unit"] else ""
    return (
        f"{found['key']} from {found['before']}{unit} back to the typical body's share, {found['after']}{unit}, "
        f"ends it, as may others ({found['part']}; try {found['tried']} of {found['total']})"
    )


def tell(spec, sex, wrap, out=None):
    """Search for a refused body from a command, saying that it is, and what it found."""
    out = out or sys.stderr
    if is_whole(wrap):
        return
    print("  looking for a parameter that ends it, a wrap at a time...", file=out, flush=True)
    print(f"  {sentence(search(spec, sex, wrap))}", file=out)


def tell_body(name, out=None):
    """As `tell`, for a body refused after its build."""
    spec = load_specs()[name]
    tell(spec, sex_of(name), json.loads((BUILD / "bodies" / name / "wrap.json").read_text()), out)
