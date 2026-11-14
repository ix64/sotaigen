"""What the page must know about parameters, generated from the Python that defines it.

sotaigen/params/rules.py and sotaigen/params/spec.py stay the only source. This script reads their tables and rules through
Python's own parser and writes them as an ES module, so a change to a number or a rule reaches the page
by rerunning the command, and nobody keeps a second copy by hand:

    uv run sotaigen-dev schema            # build/editor/params.generated.mjs
    uv run sotaigen-dev schema --publish  # sotaigen/data/params.schema.json

Only plain arithmetic translates; a numeric constant of sotaigen.params.rules is written as its number. Anything
else raises Untranslatable, loudly and at generation time.
tests/test_editor.py runs the generated module under Node against the Python on random bodies. The
fourth layer of validation, field_failures, needs the implicit surface and stays on the Python side.
"""

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import sotaigen.params.rules as rules_module
from sotaigen.output.metadata import asset_generator
from sotaigen.params.file import FORMAT, VERSION
from sotaigen.params.rules import HEADS_RANGE, NUMERIC_SLACK, RANGES, STATURE_RANGE
from sotaigen.params.spec import (
    ALL_SPEC,
    BASES,
    CHAIN,
    MILLIMETRES,
    SEXES,
    TORSO_SHARES,
    required,
)

OUT = ROOT / "build" / "editor" / "params.generated.mjs"
# The parameter file's JSON Schema, shipped with the package: `schema.py --publish` writes it.
PUBLISHED = ROOT / "sotaigen" / "data" / "params.schema.json"
MATH = {
    "math.sqrt": "Math.sqrt",
    "math.sin": "Math.sin",
    "math.cos": "Math.cos",
}
OPERATORS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}
# The module's numeric constants, written into the rules as their numbers, as sotaigen.params.rules.ARM_REST_DEG.
CONSTANTS = {
    name: value
    for name, value in vars(rules_module).items()
    if name.isupper() and isinstance(value, int | float) and not isinstance(value, bool)
}


class Untranslatable(ValueError):
    """Python this translator doesn't know. Say what it is rather than guess."""


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{dotted(node.value)}.{node.attr}"
    raise Untranslatable(ast.dump(node))


def expression(node, helpers, names):
    """A Python expression as JavaScript, fully bracketed, so precedence can't differ. `names` are the
    bare names it may read: the function's arguments and locals; any other is a module constant or
    untranslatable, here rather than in the page."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return repr(node.value)
    if isinstance(node, ast.Name):
        if node.id in CONSTANTS:
            return repr(CONSTANTS[node.id])
        if node.id in names:
            return node.id
        raise Untranslatable(f"{node.id}: not an argument, a local or a numeric constant of sotaigen.params.rules")
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left = expression(node.left, helpers, names)
        right = expression(node.right, helpers, names)
        return f"({left} {OPERATORS[type(node.op)]} {right})"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f"(-{expression(node.operand, helpers, names)})"
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        return f"{expression(node.value, helpers, names)}[{json.dumps(node.slice.value)}]"
    if isinstance(node, ast.Call) and not node.keywords:
        name = dotted(node.func)
        if name == "p.get" and len(node.args) == 2:
            key, default = node.args
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                return f"(p[{json.dumps(key.value)}] ?? {expression(default, helpers, names)})"
        args = [expression(a, helpers, names) for a in node.args]
        if name == "math.radians":
            return f"({args[0]} * Math.PI / 180)"
        if name in MATH:
            return f"{MATH[name]}({', '.join(args)})"
        if name in helpers:
            return f"{name}({', '.join(args)})"
    raise Untranslatable(ast.unparse(node))


def tree_of(path):
    return ast.parse(path.read_text())


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def assigned(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return node.value
    raise Untranslatable(f"no assignment to {name}")


def helper(tree, name, helpers):
    """A function that is one return statement, after its docstring."""
    node = function(tree, name)
    body = [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    names = {a.arg for a in node.args.args}
    if len(body) == 1 and isinstance(body[0], ast.Return):
        args = ", ".join(a.arg for a in node.args.args)
        return f"function {name}({args}) {{\n  return {expression(body[0].value, helpers, names)};\n}}"
    # Local names then a return: torso-style helpers written over several lines.
    lines = []
    for statement in body:
        if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name):
            lines.append(
                f"  const {statement.targets[0].id} = {expression(statement.value, helpers, names)};"
            )
            names = names | {statement.targets[0].id}
        elif isinstance(statement, ast.Return):
            lines.append(f"  return {expression(statement.value, helpers, names)};")
        else:
            raise Untranslatable(ast.unparse(statement))
    args = ", ".join(a.arg for a in node.args.args)
    return f"function {name}({args}) {{\n" + "\n".join(lines) + "\n}"


def called_names(node):
    """The plain names called anywhere below a node."""
    return {
        call.func.id
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }


def sex_tables(tree, name):
    """A {sex: (table, ...)} assignment as {sex: [table name, ...]}."""
    value = assigned(tree, name)
    if not isinstance(value, ast.Dict):
        raise Untranslatable(ast.unparse(value))
    out = {}
    for key, tables in zip(value.keys, value.values, strict=True):
        if not isinstance(tables, ast.Tuple) or not all(isinstance(t, ast.Name) for t in tables.elts):
            raise Untranslatable(ast.unparse(tables))
        out[key.value] = [t.id for t in tables.elts]
    return out


def rule_table_names(tree):
    """Every rule table a sex meets, in the order they first appear."""
    names = []
    for kind in ("SEX_RULES", "SEX_SEAM_RULES"):
        for tables in sex_tables(tree, kind).values():
            names += [name for name in tables if name not in names]
    return names


def helper_names(tree):
    """The functions of the module that the rules call, the ones they call first."""
    defined = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    ordered = []

    def visit(name):
        if name in ordered:
            return
        for inner in sorted(called_names(defined[name]) & defined.keys()):
            visit(inner)
        ordered.append(name)

    for table in rule_table_names(tree):
        for name in sorted(called_names(assigned(tree, table)) & defined.keys()):
            visit(name)
    return ordered


def rule_table(tree, name, helpers):
    """A tuple of (name, sentence, lambda p: room) as a JavaScript array of [name, (p) => room]: the page
    words a rule by its name, so the English sentence stays here."""
    rows = []
    for row in assigned(tree, name).elts:
        rule_name, _sentence, room = row.elts
        if not isinstance(room, ast.Lambda) or room.args.args[0].arg != "p":
            raise Untranslatable(ast.unparse(row))
        rows.append(f"  [{json.dumps(rule_name.value)}, (p) => {expression(room.body, helpers, {"p"})}],")
    return f"export const {name} = [\n" + "\n".join(rows) + "\n];"


SPEC_FUNCTIONS = """
export function units(spec) {
  const head = spec.headHeight;
  const stature = head + spec.neckLength + spec.torsoLength + spec.crotchHeight;
  return { head, stature, torso: spec.torsoLength, crotch: spec.crotchHeight, hip: spec.hipBreadth, one: 1 };
}

export function proportions(spec) {
  const q = {};
  for (const [key, value] of Object.entries(spec)) q[key] = MILLIMETRES.includes(key) ? value / spec.torsoLength : value;
  q.stature = units(spec).stature / spec.torsoLength;
  return q;
}

// One sex's view: its own parameters, ranges and rules, and what every sex shares, so the
// page's code takes it as it would one table.
function template(ALL_SPEC, ALL_SPEC_RANGES, REQUIRED, RULES, SEAM_RULES) {
  const specBounds = (spec) => {
    const u = units(spec);
    const bounds = {};
    for (const [key, [low, high]] of Object.entries(ALL_SPEC_RANGES)) {
      const unit = u[ALL_SPEC[key][1]];
      bounds[key] = [low * unit, high * unit];
    }
    return bounds;
  };
  return {
    ALL_SPEC, ALL_SPEC_RANGES, REQUIRED, RULES, SEAM_RULES, MILLIMETRES, CHAIN, TORSO_SHARES, NUMERIC_SLACK,
    STATURE_RANGE, HEADS_RANGE, units, proportions, specBounds,
  };
}
"""


UNIT_WORDS = {
    "stature": "the stature",
    "head": "the head's height",
    "torso": "the torso's length",
    "crotch": "the crotch's height",
    "hip": "the hip's breadth",
}


def parameter_schema(key, row):
    """One parameter's JSON Schema: a number, said in words. Its range is in its unit on the body's own
    lengths, which JSON Schema can't compute, so the description says it."""
    low, high = row.range
    if row.unit == "one":
        unit, scope = "a plain number", f"{low:g} to {high:g}"
    else:
        unit, scope = "millimetres", f"{low:g} to {high:g} of {UNIT_WORDS[row.unit]}"
    words = [f"{unit}; {BASES[row.basis]}", f"range {scope}"]
    if row.optional:
        words.append(
            "optional: without it the body keeps its own shape"
            if row.default is None
            else f"optional: without it, {row.default:g}" + ("" if row.unit == "one" else f" of {UNIT_WORDS[row.unit]}")
        )
    return {"type": "number", "description": "; ".join(words)}


def json_schema():
    """A parameter file's shape as JSON Schema, no more.

    A range depends on the body's own lengths and a rule relates several parameters, which JSON Schema
    has no way to say, so both stay in the generated module. Fields it doesn't know at the top level are
    allowed, as the readers keep them; a parameter its sex doesn't know is not, as
    sotaigen.params.rules.problems refuses it.
    """
    per_template = [
        {
            "if": {"properties": {"sex": {"const": name}}},
            "then": {
                "properties": {
                    "body": {
                        "additionalProperties": False,
                        "required": sorted(required(name)),
                        "properties": {key: parameter_schema(key, row) for key, row in sorted(table.items())},
                    }
                }
            },
        }
        for name, table in SEXES.items()
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"sotaigen/params/v{VERSION}",
        "title": "sotaigen parameters",
        "type": "object",
        "required": ["format", "version", "sex", "body"],
        "additionalProperties": True,
        "properties": {
            "format": {"const": FORMAT},
            "version": {"const": VERSION},
            "sex": {"enum": sorted(SEXES)},
            "generator": {"type": "string"},
            "base": {"type": ["object", "null"]},
            "body": {"type": "object"},
        },
        "allOf": per_template,
    }


def page_rows(table=ALL_SPEC):
    """A sotaigen.params.spec table as the page reads it: control, unit, offset, gain and default."""
    return {key: [row.control, row.unit, row.offset, row.gain, row.default] for key, row in table.items()}


def sex_views(params):
    """export const SEXES: each sex's view, its rules as the tables above, joined."""
    rules = sex_tables(params, "SEX_RULES")
    seams = sex_tables(params, "SEX_SEAM_RULES")
    joined = lambda names: "[" + ", ".join(f"...{name}" for name in names) + "]"
    rows = [
        f"  {name}: template({json.dumps(page_rows(table))}, {json.dumps(RANGES[name])}, "
        f"{json.dumps(list(required(name)))}, {joined(rules[name])}, {joined(seams[name])}),"
        for name, table in SEXES.items()
    ]
    return "export const SEXES = {\n" + "\n".join(rows) + "\n};"


def typedef(name, required, optional=()):
    fields = ", ".join([f"{key}: number" for key in sorted(required)] + [f"{key}?: number" for key in sorted(optional)])
    return f"/** @typedef {{{{ {fields} }}}} {name} */"


def module_text():
    params = tree_of(ROOT / "sotaigen" / "params" / "rules.py")
    helpers = set()
    # What every sex's bodies have; the rest is optional for the type.
    shared = set.intersection(*(set(required(name)) for name in SEXES))
    parts = [
        "// Generated by sotaigen_dev/editor/schema.py from sotaigen/params/rules.py and sotaigen/params/spec.py. Do not edit.",
        typedef("BodyParameters", shared, ALL_SPEC.keys() - shared),
        "// The parameter file's, which the page writes: sotaigen.params.file.",
        f"export const FILE_FORMAT = {json.dumps(FORMAT)};",
        f"export const FILE_VERSION = {VERSION};",
        f"export const GENERATOR = {json.dumps(asset_generator())};",
        f"export const ALL_SPEC = {json.dumps(page_rows())};",
        f"export const MILLIMETRES = {json.dumps(list(MILLIMETRES))};",
        f"export const CHAIN = {json.dumps(list(CHAIN))};",
        f"export const TORSO_SHARES = {json.dumps(list(TORSO_SHARES))};",
        f"export const NUMERIC_SLACK = {NUMERIC_SLACK!r};",
        f"export const STATURE_RANGE = {json.dumps(STATURE_RANGE)};",
        f"export const HEADS_RANGE = {json.dumps(HEADS_RANGE)};",
    ]
    for name in helper_names(params):
        parts.append(helper(params, name, helpers))
        helpers.add(name)
    parts += [rule_table(params, name, helpers) for name in rule_table_names(params)]
    parts += [SPEC_FUNCTIONS.strip(), sex_views(params)]
    return "\n\n".join(parts) + "\n"


def schema_text():
    return json.dumps(json_schema(), indent=1) + "\n"


def write(path=OUT):
    """The module at `path` and the JSON Schema beside it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(module_text())
    (path.parent / "params.schema.json").write_text(schema_text())
    return path


def publish():
    """The JSON Schema the package ships, which tests/test_editor.py holds to this generator."""
    PUBLISHED.write_text(schema_text())
    return PUBLISHED


if __name__ == "__main__":
    print(publish() if "--publish" in sys.argv else write())
