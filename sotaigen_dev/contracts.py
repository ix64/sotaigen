"""The frozen contracts as this code has them, for tests/test_contracts.py to hold against the golden
copies in tests/contracts/.

A golden copy changes only with a version that breaks the contract (docs/reference/README.md), so
writing them again is a decision, not a fix for a failing test:

    uv run sotaigen-dev contracts --write [--blend]

`--blend` writes a typical body's .blend in Blender to read what the file itself holds.
"""

import argparse
import ast
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "contracts"
SCRIPTS = ROOT / "blender" / "src" / "sotaigen_blender"
# What isn't the contract in extras.sotaigen: the parameter file has its own golden copy, the check
# poses may change with any version, a diagnostic's values are held by code in diagnostics.json.
GLB_LEFT_OUT = {"parameters", "checkPoses"}
# The bodies whose GLB and .blend the contract is read from: one of each sex.
READ_FROM = ("female-long", "male-long")


def parameters():
    """The parameter file: its fields, and each sex's parameters with the unit a file holds them in and
    what each is measured on (sotaigen.params.spec.BASES). Ranges are not the contract."""
    from sotaigen.params import file
    from sotaigen.params.spec import SEXES

    return {
        "format": file.FORMAT,
        "version": file.VERSION,
        "fields": sorted(file.KNOWN_FIELDS),
        "sexes": {
            sex: {
                key: {"unit": "ratio" if row.unit == "one" else "mm", "optional": row.optional, "basis": row.basis}
                for key, row in table.items()
            }
            for sex, table in SEXES.items()
        },
    }


def bodies():
    """The typical bodies' names. The test bodies are not the contract."""
    from sotaigen.form.bodies import load_specs

    return {"typical": sorted(load_specs("typical"))}


def _arguments(parser):
    out = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction | argparse._SubParsersAction):
            continue
        out.append(action.dest if not action.option_strings else sorted(action.option_strings))
    return out


def command_line():
    """`sotaigen`'s subcommands, each one's arguments (positionals by name, options by their flags), and
    the exit codes. Not the output's wording."""
    from sotaigen import cli

    top = cli.parser()
    commands = next(a for a in top._actions if isinstance(a, argparse._SubParsersAction))
    return {
        "commands": {name: _arguments(sub) for name, sub in sorted(commands.choices.items())},
        "exitCodes": {"done": cli.DONE, "refused": cli.REFUSED, "usage": cli.USAGE, "missing": cli.MISSING},
    }


def _keys_read(function):
    """The values a sentence reads: v["key"], v.get("key") and the `at` that _where reads."""
    found = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            found.add(node.slice.value)
        is_get = isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "get" and node.args
        if is_get and isinstance(node.args[0], ast.Constant):
            found.add(node.args[0].value)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_where":
            found.add("at")
    return sorted(found)


def diagnostics():
    """Every diagnostic's code with the values it carries, and the levels. The rules and the surface's
    invariants may change with any version, so theirs are two patterns: rule.<name> with its
    shortfall, surface.<name> with nothing."""
    from sotaigen import diagnostics as d

    tree = ast.parse(inspect.getsource(d))
    table = next(
        node.value for node in tree.body if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "SENTENCES"
    )
    codes = {key.value: _keys_read(value) for key, value in zip(table.keys, table.values, strict=True)}
    rule = inspect.signature(d.rule)
    assert list(rule.parameters)[:2] == ["name", "short"], "rule.<name> carries its shortfall"
    codes |= {"rule.<name>": ["short"], "surface.<name>": []}
    return {"levels": list(d.LEVELS), "codes": dict(sorted(codes.items()))}


def shape(value):
    """A JSON value's shape: an object's keys with their shapes, a list's items' shapes merged, and a
    leaf's kind. Numbers are one kind: JSON doesn't tell 1 from 1.0."""
    if isinstance(value, dict):
        return {key: shape(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        merged = None
        for item in value:
            merged = _merge(merged, shape(item))
        return [merged] if merged is not None else []
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    return "string" if isinstance(value, str) else "null"


def _merge(a, b):
    """Two shapes as one. A key only some objects have is marked optional with a trailing "?"."""
    if a is None:
        return b
    if isinstance(a, dict) and isinstance(b, dict):
        split = lambda d: {key.rstrip("?"): (value, key.endswith("?")) for key, value in d.items()}
        left, right = split(a), split(b)
        out = {}
        for key in sorted(left.keys() | right.keys()):
            if key in left and key in right:
                value, optional = _merge(left[key][0], right[key][0]), left[key][1] or right[key][1]
            else:
                value, optional = (left.get(key) or right.get(key))[0], True
            out[key + "?" if optional else key] = value
        return out
    if isinstance(a, list) and isinstance(b, list):
        return [_merge(a[0] if a else None, b[0] if b else None)] if a or b else []
    if a != b:
        raise ValueError(f"one place holds both {a} and {b}")
    return a


def glb(extras):
    """extras.sotaigen's contract from one file's: format, version, every field's shape but those
    GLB_LEFT_OUT, the corrective groups' names and sources, the expressions. A diagnostic's values
    are by code (diagnostics), so `quality` keeps its own fields only."""
    held = {key: value for key, value in extras.items() if key not in GLB_LEFT_OUT}
    held["quality"] = {**held["quality"], "found": [{**f, "values": {}} for f in held["quality"]["found"]]}
    out = shape(held)
    out["quality"]["found"] = [{"code": "string", "level": "string", "values": {}}]
    groups = extras["correctives"]["groups"]
    return {
        "format": extras["format"],
        "version": extras["version"],
        "fields": out,
        "correctiveGroups": sorted({group["name"] for group in groups}),
        "qualityChecks": sorted(extras["quality"]["checked"]),
        "expressions": extras["expressions"],
    }


def glb_of(name, folder):
    """The GLB of a typical body, made as a downstream would (sotaigen.generate, sotaigen.write_glb)."""
    import sotaigen
    from sotaigen.output.glb import read_glb

    path = sotaigen.write_glb(sotaigen.generate(sotaigen.typical(name)), Path(folder) / f"{name}.glb")
    return read_glb(path.read_bytes())[0]["extras"]["sotaigen"]


def merged_glb(extras_of_each):
    found = [glb(extras) for extras in extras_of_each]
    out = found[0]
    for other in found[1:]:
        out["fields"] = _merge(out["fields"], other["fields"])
        out["correctiveGroups"] = sorted(set(out["correctiveGroups"]) | set(other["correctiveGroups"]))
        out["qualityChecks"] = sorted(set(out["qualityChecks"]) | set(other["qualityChecks"]))
        if other["expressions"] != out["expressions"]:
            raise ValueError("two bodies list different expressions")
    return out


def _constants(path):
    return {
        target.id: node.value.value
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def blend_names():
    """The .blend's names as sotaigen-blender's scripts define them (docs/systems/blend.md, 命名)."""
    rig = _constants(SCRIPTS / "rig.py")
    convert = ast.parse((SCRIPTS / "convert.py").read_text())
    groups = next(
        ast.literal_eval(node.value) for node in convert.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "GROUPS"
    )
    fingers = _constants(SCRIPTS / "convert.py")["FINGERS"]
    return {
        "objects": {"rig": rig["RIG"], "body": rig["BODY"], "mechanism": rig["HUB"]},
        "controlSuffix": rig["CONTROL_SUFFIX"],
        "widgetPrefix": rig["WIDGET_PREFIX"],
        "constraints": sorted(rig[key] for key in ("FOLLOW", "RHYTHM", "LIFT", "GLIDE", "HALF", "TWIST", "LIMIT")),
        "vertexGroups": {"headRegion": rig["HEAD_REGION"], "neckRing": rig["NECK_RING"]},
        "properties": {"jointLimits": rig["LIMITS_SWITCH"], "info": rig["INFO"]},
        "collections": {"controls": rig["CONTROLS"], "deform": rig["DEFORM"], "parts": rig["PARTS"],
                        "mechanism": rig["MECHANISM"], "widgets": rig["WIDGETS"]},
        "boneCollections": sorted([*groups, fingers]),
    }


def blend_file(posed_info):
    """What a written .blend holds, from sotaigen-blender's pose.py report: rig["sotaigen"]'s keys, the
    bones' names and the pose assets. The keys' values are the GLB's (glb)."""
    return {
        "info": sorted(posed_info["info"]),
        "bones": sorted(posed_info["bones"]),
        "poseAssets": sorted(posed_info["poseAssets"]),
    }


def read_blend(name, folder):
    """A typical body's .blend written and read back in Blender: pose.py's report with no poses."""
    import sotaigen
    from sotaigen.output.blender import run_blender, script

    folder = Path(folder)
    blend = sotaigen.write_blend(sotaigen.generate(sotaigen.typical(name)), folder / f"{name}.blend")
    request = folder / f"{name}-read.json"
    request.write_text(json.dumps({"poses": {}, "central": [], "limits": [False], "actions": False}))
    out = folder / f"{name}-read"
    run_blender(script("pose"), request, out, blend=blend)
    return json.loads(out.with_suffix(".json").read_text())


def current(blend=False, folder=None):
    """Every contract this code has, by its golden file's name."""
    out = {
        "parameters": parameters(),
        "bodies": bodies(),
        "cli": command_line(),
        "diagnostics": diagnostics(),
        "glb": merged_glb([glb_of(name, folder) for name in READ_FROM]),
        "blend": {"names": blend_names()},
    }
    if blend:
        files = [blend_file(read_blend(name, folder)) for name in READ_FROM]
        out["blend"]["file"] = {
            "info": sorted(set().union(*(f["info"] for f in files))),
            "bones": sorted(set().union(*(f["bones"] for f in files))),
            "poseAssets": sorted(set().union(*(f["poseAssets"] for f in files))),
        }
    return out


def main(argv=None):
    import tempfile

    parser = argparse.ArgumentParser(prog="sotaigen-dev contracts", description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="write the golden copies from this code")
    parser.add_argument("--blend", action="store_true", help="also read a written .blend, in Blender")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as folder:
        found = current(args.blend, folder)
    if not args.blend and (GOLDEN / "blend.json").exists():
        found["blend"]["file"] = json.loads((GOLDEN / "blend.json").read_text()).get("file")
    if not args.write:
        print(json.dumps(found, indent=1))
        return 0
    for name, value in found.items():
        (GOLDEN / f"{name}.json").write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")
    print("wrote", ", ".join(f"{name}.json" for name in found))
    return 0


if __name__ == "__main__":
    sys.exit(main())
