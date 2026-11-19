"""The editor's page code against the Python it copies. Under plain Node, no browser, no npm package.

What the page does with parameters comes in two kinds: generated from sotaigen/params/rules.py and sotaigen/params/spec.py
by sotaigen_dev/editor/schema.py, and written by hand in web/editor/*.ts. Both are run here against the
Python on the same inputs, so neither can drift without a failure here.
"""

import json
import math
import os
import re
import subprocess
import sys
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from sotaigen import cli, generation, mend
from sotaigen.body import Body
from sotaigen.diagnostics import Diagnostic, as_json, rule
from sotaigen.form.anatomy import calibrated, width_at
from sotaigen.form.bodies import body_specs
from sotaigen.form.measures import measure
from sotaigen.form.solve import solved_controls, target_misses
from sotaigen.output import blend, face, glb, metadata, quality
from sotaigen.output.face import face_parts, head_normals
from sotaigen.output.glb import export_base
from sotaigen.params import file as params_file
from sotaigen.params.file import FileError, Spec
from sotaigen.params.rules import (
    RANGES,
    problems,
    random_spec,
    rules,
    seam_problems,
    seam_rules,
    sex_of,
    spec_bounds,
    spec_problems,
)
from sotaigen.params.spec import (
    ALL_SPEC,
    ARM_REST_DEG,
    MILLIMETRES,
    SEXES,
    TORSO_SHARES,
    estimated_controls,
    heads,
    proportions,
    required,
    stature,
)
from sotaigen.pipeline import rigged
from sotaigen_dev.editor import schema as editor_schema
from sotaigen_dev.editor import sliders as editor_sliders
from sotaigen_dev.editor import worker

WEB = ROOT / "web"
EDITOR = WEB / "editor"
RANDOM_BODIES = 200


def run_node(body, payload, imports):
    """Run a module body under Node with `imports` {name: path} and `input` from `payload`."""
    header = "".join(f"import * as {name} from {json.dumps(path.as_uri())};\n" for name, path in imports.items())
    code = (
        f"{header}"
        'let text = ""; for await (const piece of process.stdin) text += piece;\n'
        f"const input = JSON.parse(text);\n{body}"
    )
    done = subprocess.run(
        ["node", "--input-type=module", "-e", code],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.fixture(scope="module")
def schema_module(tmp_path_factory):
    path = tmp_path_factory.mktemp("editor") / "params.generated.mjs"
    return editor_schema.write(path)


def random_specs(count, sex, seed=3, optional=False):
    """Bodies of a sex from its ranges, valid or not: the rules must agree on the broken ones too."""
    rng = np.random.default_rng(seed)
    return [random_spec(rng, sex, optional) for _ in range(count)]


@pytest.mark.node
@pytest.mark.parametrize("sex", SEXES)
def test_the_generated_proportions_and_rules_match_the_python(schema_module, sex):
    specs = random_specs(RANDOM_BODIES, sex, optional=True)
    out = run_node(
        """
        const rooms = (table, p) => table.map(([name, room]) => [name, room(p)]);
        const s = schema.SEXES[input.sex];
        console.log(JSON.stringify(input.specs.map((body) => {
          const p = s.proportions(body);
          return { p, bounds: s.specBounds(body), rules: rooms(s.RULES, p), seams: rooms(s.SEAM_RULES, p) };
        })));
        """,
        {"sex": sex, "specs": specs},
        {"schema": schema_module},
    )
    for spec, row in zip(specs, out, strict=True):
        p = proportions(spec)
        assert row["p"].keys() == p.keys()
        for key in p:
            assert row["p"][key] == pytest.approx(p[key], rel=1e-12, abs=1e-12)
        assert row["bounds"].keys() == spec_bounds(spec, sex).keys()
        for key, bounds in spec_bounds(spec, sex).items():
            assert row["bounds"][key] == pytest.approx(bounds, rel=1e-12, abs=1e-12)
        for table, rows in ((rules(sex), row["rules"]), (seam_rules(sex), row["seams"])):
            assert [name for name, _ in rows] == [name for name, _, _ in table]
            for (_, _, room), (_, value) in zip(table, rows, strict=True):
                assert value == pytest.approx(room(p), rel=1e-12, abs=1e-12)


@pytest.mark.node
@pytest.mark.parametrize("sex", SEXES)
def test_the_generated_tables_are_the_tables(schema_module, sex):
    out = run_node(
        """
        const s = schema.SEXES[input];
        console.log(JSON.stringify({ spec: s.ALL_SPEC, ranges: s.ALL_SPEC_RANGES, required: s.REQUIRED, mm: s.MILLIMETRES }));
        """,
        sex,
        {"schema": schema_module},
    )
    table = SEXES[sex]
    assert out["spec"] == {k: [r.control, r.unit, r.offset, r.gain, r.default] for k, r in table.items()}
    assert {k: tuple(v) for k, v in out["ranges"].items()} == RANGES[sex]
    assert tuple(out["required"]) == required(sex)
    assert tuple(out["mm"]) == MILLIMETRES


@pytest.mark.node
def test_the_generated_module_has_every_template_and_no_other(schema_module):
    out = run_node("console.log(JSON.stringify(Object.keys(schema.SEXES)));", {}, {"schema": schema_module})
    assert out == list(SEXES)


def test_the_json_schema_names_every_parameter_of_each_template():
    schema = editor_schema.json_schema()
    assert schema["properties"]["sex"] == {"enum": sorted(SEXES)}
    assert "sex" in schema["required"]
    cases = {case["if"]["properties"]["sex"]["const"]: case["then"] for case in schema["allOf"]}
    assert cases.keys() == SEXES.keys()
    for name, then in cases.items():
        row = then["properties"]["body"]
        assert row["required"] == sorted(required(name))
        assert row["additionalProperties"] is False
        assert set(row["properties"]) == set(SEXES[name])
    assert schema["additionalProperties"] is True
    assert schema["properties"]["version"]["const"] == params_file.VERSION


def test_the_shipped_json_schema_is_what_the_generator_writes():
    # Rewrite it with `python sotaigen_dev/editor/schema.py --publish` when the table changes.
    assert editor_schema.PUBLISHED.read_text() == editor_schema.schema_text()


def test_python_it_cannot_translate_is_refused_not_guessed():
    node = editor_schema.ast.parse("max(p['a'], 1)", mode="eval").body
    with pytest.raises(editor_schema.Untranslatable):
        editor_schema.expression(node, set(), {"p"})


def test_a_constant_of_the_rules_is_written_as_its_number():
    # The page has no sotaigen.params.rules to look the name up in.
    node = editor_schema.ast.parse("p['a'] * ARM_REST_DEG", mode="eval").body
    assert editor_schema.expression(node, set(), {"p"}) == f'(p["a"] * {ARM_REST_DEG!r})'


VALIDATE = """
const s = schema.SEXES[input.sex];
console.log(JSON.stringify(input.specs.map((body) => files.validate(body, s))));
"""


def assert_same_findings(page, python):
    """The page's diagnostics are Python's: the same codes, levels and values, numbers to the page's own
    rounding of a rule's shortfall, and a value that isn't a number as either side quotes it."""
    assert [(d["code"], d["level"]) for d in page] == [(d.code, d.level) for d in python]
    for mine, theirs in zip(page, python, strict=True):
        assert mine["values"].keys() == theirs.values.keys(), theirs.code
        for key, value in theirs.values.items():
            if isinstance(value, float):
                tolerance = 1.0001e-3 if key == "short" else 1e-9
                assert mine["values"][key] == pytest.approx(value, rel=1e-12, abs=tolerance), (theirs.code, key)
            elif key == "value" and isinstance(value, str):
                assert mine["values"][key].replace('"', "'") == value, theirs.code
            else:
                assert mine["values"][key] == value, (theirs.code, key)


@pytest.mark.node
@pytest.mark.parametrize("sex", SEXES)
def test_the_page_validates_as_python_does(schema_module, sex):
    specs = random_specs(RANDOM_BODIES, sex, seed=4, optional=True)
    out = run_node(
        VALIDATE,
        {"sex": sex, "specs": specs},
        {"schema": schema_module, "files": EDITOR / "params-file.ts"},
    )
    for spec, row in zip(specs, out, strict=True):
        assert_same_findings(row["errors"], problems(spec, sex))
        assert_same_findings(row["seams"], seam_problems(spec, sex) if not spec_problems(spec, sex) else [])
    # Random bodies from the ranges must break rules often, or this compared nothing.
    assert sum(bool(row["errors"]) for row in out) > RANDOM_BODIES / 4


@pytest.mark.node
def test_optional_body_parameters_survive_the_editors_file_round_trip(schema_module):
    body = dict(body_specs()["female-long"])
    body["waistContourShift"] = 12.5
    file = json.loads(params_file.write_parameter_file(body, "female"))
    out = run_node(
        """
        const { errors } = files.validate(input.body, schema.SEXES[input.sex]);
        const saved = JSON.parse(files.writeParameterFile(input, schema, "test"));
        console.log(JSON.stringify({ errors, body: saved.body }));
        """,
        params_file.read_parameter_file(json.dumps(file)),
        {"schema": schema_module, "files": EDITOR / "params-file.ts"},
    )
    assert not problems(body, "female")
    assert out == {"errors": [], "body": body}


@pytest.mark.node
def test_the_page_reports_ranges_as_python_does(schema_module):
    body = {**random_specs(1, "female")[0], "jawBreadthRatio": 99.0, "made up": 1.0}
    del body["crotchHeight"]
    bad = {**body_specs()["female-long"], "jawBreadthRatio": 99.0, "headHeight": 5000.0}
    text = {**body_specs()["female-long"], "softness": "a lot"}
    out = run_node(
        VALIDATE,
        {"sex": "female", "specs": [body, bad, text]},
        {"schema": schema_module, "files": EDITOR / "params-file.ts"},
    )
    for row, spec in zip(out, (body, bad, text), strict=True):
        assert_same_findings(row["errors"], problems(spec, "female"))
    assert {d["values"]["key"] for d in out[0]["errors"]} == {"crotchHeight", "made up"}


@pytest.mark.node
def test_the_page_holds_a_body_to_its_own_templates_parameters(schema_module):
    male = body_specs()["male-long"]
    female = body_specs()["female-balanced"]
    cases = [
        ("male", {**male, "breastProjection": 0.02 * male["torsoLength"]}),
        ("female", {**female, "pectoralThickness": 0.03 * female["torsoLength"]}),
        ("female", {k: v for k, v in female.items() if k != "chestDepth"}),
    ]
    out = [
        run_node(VALIDATE, {"sex": sex, "specs": [spec]}, {"schema": schema_module, "files": EDITOR / "params-file.ts"})[0]
        for sex, spec in cases
    ]
    for (sex, spec), row in zip(cases, out, strict=True):
        assert_same_findings(row["errors"], problems(spec, sex))
    assert out[0]["errors"] == [{"code": "param.unknown", "level": "error", "values": {"key": "breastProjection"}}]
    assert any(d["code"] == "rule.pectoral_under_breast_thin" for d in out[1]["errors"])
    assert out[2]["errors"] == [{"code": "param.missing", "level": "error", "values": {"key": "chestDepth"}}]


def page_ranges():
    return """
        const tables = (base) => schema.SEXES[base.sex];
        const rangesFor = (base) => ({
          body: tables(base).specBounds(base.body),
          derived: { heads: tables(base).HEADS_RANGE, stature: tables(base).STATURE_RANGE },
        });
    """


@pytest.mark.node
def test_sliders_do_what_the_python_says(schema_module):
    specs = body_specs()
    rng = np.random.default_rng(5)
    cases = []
    for name in sorted(specs)[:6]:
        base = {"sex": sex_of(name), "body": specs[name]}
        values = {
            slider: float(rng.uniform(*editor_sliders.slider_bounds(slider, base)))
            for slider in editor_sliders.SLIDERS["sliders"]
            if rng.random() < 0.5
        }
        cases.append({"base": base, "values": values})
    out = run_node(
        page_ranges() + """
        const { cases, definition } = input;
        console.log(JSON.stringify(cases.map(({ base, values }) =>
          sliders.applySliders(base, values, definition, rangesFor(base), tables(base)))));
        """,
        {"cases": cases, "definition": editor_sliders.SLIDERS},
        {"schema": schema_module, "sliders": EDITOR / "sliders.ts"},
    )
    for case, row in zip(cases, out, strict=True):
        expected = editor_sliders.apply_sliders(case["base"], case["values"])
        for key, value in expected["body"].items():
            assert row["body"][key] == pytest.approx(value, rel=1e-12, abs=1e-12)


def test_a_length_in_the_chain_moves_only_itself_what_it_carries_and_the_results():
    # The teen's waist depth sits on its range's floor: a longer torso alone would take it out, so the
    # torso carries everything measured in its shares.
    base = {"sex": "female", "body": body_specs()["female-slim"]}
    body = base["body"]
    for key in ("headHeight", "neckLength", "torsoLength", "crotchHeight"):
        value = body[key] * 1.05
        moved = editor_sliders.apply_sliders(base, {key: value})["body"]
        # The knee is part of the leg, and keeps its place on it.
        carried = {"kneeHeight": body["kneeHeight"] * 1.05} if key == "crotchHeight" else {}
        if key == "torsoLength":
            carried = {other: body[other] * 1.05 for other in TORSO_SHARES if other in body}
        assert moved == pytest.approx(body | {key: value} | carried)
        assert stature(moved) == pytest.approx(stature(body) + value - body[key])
        found = problems(moved, "female")
        assert all(d.code.startswith("rule.") for d in found)
        if key == "torsoLength":
            assert not spec_problems(moved, "female")


def test_a_torso_moved_alone_keeps_every_typical_body_in_its_ranges():
    # Only what is measured in shares of the stature can leave its range.
    for name, body in body_specs().items():
        sex = sex_of(name)
        base = {"sex": sex, "body": body}
        low, high = editor_sliders.slider_bounds("torsoLength", base)
        for value in np.linspace(low, high, 9):
            moved = editor_sliders.apply_sliders(base, {"torsoLength": float(value)})["body"]
            # The head's and the leg's ranges are shares of the stature, which the torso moves.
            by_stature = [key for key, row in ALL_SPEC.items() if row.unit == "stature"]
            found = [d for d in spec_problems(moved, sex) if d.values.get("key") not in (*by_stature, "stature")]
            assert found == [], name


def test_each_mend_check_offers_meets_its_rule_alone():
    # Arms far too thick for the compact body's shoulders break two arm rules.
    body = body_specs()["female-compact"]
    body = body | {"upperArmGirth": body["upperArmGirth"] * 1.6}
    mends = cli.rule_mends(body, "female")
    assert mends and all(mends.values())
    for found in problems(body, "female"):
        name = found.code.removeprefix("rule.")
        for option in mends[name].split(", "):
            key, change = option.split()[:2]
            # Printed to a tenth, so go the rounding further the same way.
            change = float(change) + math.copysign(0.05, float(change))
            mended = body | {key: body[key] + change}
            assert not any(other.code == found.code for other in problems(mended, "female")), option
            low, high = spec_bounds(body, "female")[key]
            assert low - 0.1 <= mended[key] <= high + 0.1, option


def test_the_head_count_tool_keeps_the_stature_and_writes_real_parameters():
    # A body with fewer heads than 6.5, so the head has to shrink to get there.
    base = {"sex": "female", "body": body_specs()["female-slim"]}
    moved = editor_sliders.apply_sliders(base, {"heads": 6.5})["body"]
    assert heads(moved) == pytest.approx(6.5)
    assert stature(moved) == pytest.approx(stature(base["body"]))
    assert moved["headHeight"] < base["body"]["headHeight"]
    assert moved["chestBreadth"] != base["body"]["chestBreadth"]
    # Nothing about the tool stays behind but the numbers: the file needs no head count.
    assert not spec_problems(moved, "female")


def test_the_stature_tool_scales_and_keeps_every_shape():
    base = {"sex": "female", "body": body_specs()["female-balanced"]}
    moved = editor_sliders.apply_sliders(base, {"stature": 1500.0})["body"]
    assert stature(moved) == pytest.approx(1500.0)
    before, after = estimated_controls(base["body"]), estimated_controls(moved)
    for key, value in before.items():
        if key != "stature":
            assert after[key] == pytest.approx(value, rel=1e-9), key


def test_a_direct_slider_set_after_the_head_count_is_still_its_own_value():
    base = {"sex": "female", "body": body_specs()["female-long"]}
    for values in ({"neckLength": 90.0}, {"neckGirth": 300.0}, {"neckLength": 90.0, "neckGirth": 300.0}):
        assert editor_sliders.apply_sliders(base, values) == {"body": base["body"] | values}
    changed = editor_sliders.apply_sliders(base, {"heads": 6})
    original = base["body"]["neckLength"]
    assert changed["body"]["neckLength"] != original
    restored = editor_sliders.apply_sliders(base, {"heads": 6, "neckLength": original})
    assert restored == {"body": changed["body"] | {"neckLength": original}}


def test_a_slider_at_its_start_changes_nothing():
    base = {"sex": "female", "body": body_specs()["female-long"]}
    starts = {name: editor_sliders.slider_start(name, base) for name in editor_sliders.SLIDERS["sliders"]}
    assert editor_sliders.apply_sliders(base, starts) == {"body": base["body"]}


# What web/editor/text.ts gets from the compiled messages, for Node without the npm packages: a message
# of a catalogue with its inputs, and of a message with variants, its "other", which is all Chinese has.
SAY_JS = """
const sayFrom = (table) => (key, inputs = {}) => {
  let text = table[key];
  if (text === undefined) return `MISSING ${key}`;
  if (typeof text !== "string") {
    const { match } = text[0];
    text = match[Object.keys(match).find((variant) => variant.endsWith("=other"))];
  }
  return text.replace(/\\{(\\w+)\\}/g, (_, name) => String(inputs[name]));
};
"""


def catalogue(locale):
    return json.loads((WEB / "messages" / f"{locale}.json").read_text())


# Every code the page is sent, by the worker or the file reader, with values as Python gives them.
PAGE_FINDINGS = [
    Diagnostic("param.missing", "error", {"key": "crotchHeight"}),
    Diagnostic("param.unknown", "error", {"key": "x"}),
    Diagnostic("param.not_number", "error", {"key": "softness", "value": "'a lot'"}),
    Diagnostic("param.outside", "error", {"key": "stature", "value": 99.0, "low": 1000.0, "high": 1800.0}),
    Diagnostic("param.sex", "error", {"sex": "elf"}),
    Diagnostic("target.missed", "warning", {"key": "waistDepth", "asked": 150.0, "reached": 147.5}),
    Diagnostic("surface.hipTissueMidline", "warning"),
    Diagnostic("wrap.flipped", "broken", {"count": 3, "at": {"hip": 2, "leg": 1}}),
    Diagnostic("wrap.midline", "broken", {"count": 2, "at": {}}),
    Diagnostic("skin.open", "broken", {"edges": 2}),
    Diagnostic("foot.off_ground", "warning", {"mm": 2.4, "limit": 1.0}),
    Diagnostic("seam.crease", "warning", {"part": "hand", "degrees": 40.0, "limit": 35.0}),
    Diagnostic("seam.miss", "warning", {"part": "foot", "share": 0.61, "limit": 0.52}),
    *(Diagnostic("contact.failed", "warning", {"group": group, "detail": "x"}) for group in ("axilla", "groin", "hang")),
    *(FileError(code, values).diagnostic for code, values in (
        ("file.not_json", {"detail": "oops"}), ("file.not_object", {}), ("file.format", {"format": '"glTF"'}),
        ("file.version", {"version": '"1"'}), ("file.newer", {"version": 2, "latest": 1}), ("file.no_sex", {}),
        ("file.not_parameters", {}),
    )),
]


@pytest.mark.node
def test_messages_are_in_words_and_anything_unknown_stays_as_it_was():
    out = run_node(
        SAY_JS
        + """
        const say = sayFrom(input.zh);
        const labels = { crotchHeight: "裆高", stature: "身高（整体缩放）" };
        const [rule, seam, missing, range, failure, crease, file] = input.some;
        console.log(JSON.stringify({
          rule: messages.translate(rule, labels, say),
          seam: messages.translate(seam, labels, say),
          missing: messages.translate(missing, labels, say),
          range: messages.translate(range, labels, say),
          unknown: messages.translate("ValueError: something new", labels, say),
          newCode: messages.translate({ code: "made.up", level: "warning", values: { n: 1 } }, labels, say),
          failure: messages.translate(failure, labels, say),
          crease: messages.translate(crease, labels, say),
          file: messages.translate(file, labels, say).text,
          every: input.rules.map((rule) => messages.translate(rule, {}, say).text),
          page: input.page.map((found) => messages.translate(found, {}, say).text),
          english: messages.translate(rule, {}, sayFrom(input.en)),
        }));
        """,
        {
            "some": [
                rule("neck_thin_head", 0.002).json(),
                rule("wrist_thick_hand", 0.001, "warning").json(),
                PAGE_FINDINGS[0].json(),
                PAGE_FINDINGS[3].json(),
                PAGE_FINDINGS[6].json(),
                PAGE_FINDINGS[11].json(),
                FileError("file.not_json", {"detail": "oops"}).diagnostic.json(),
            ],
            "rules": [rule(name, 0.001).json() for sex in SEXES for name, _, _ in rules(sex) + seam_rules(sex)],
            "page": [d.json() for d in PAGE_FINDINGS],
            "zh": catalogue("zh"),
            "en": catalogue("en"),
        },
        {"messages": EDITOR / "messages.ts"},
    )
    assert out["rule"] == {"text": "颈部相对头部太细（差 0.002）", "panel": "neck"}
    assert out["seam"]["text"] == "注意：腕相对手太粗（差 0.001）"
    assert out["missing"]["text"] == "缺少参数：裆高（crotchHeight）"
    assert out["range"]["text"] == "身高（整体缩放）（stature） = 99，超出范围 1000 到 1800"
    assert out["unknown"] == {"text": "ValueError: something new"}
    assert out["newCode"] == {"text": 'made.up {"n":1}'}
    assert "软组织" in out["failure"]["text"]
    assert out["crease"]["panel"] == "arm" and "40" in out["crease"]["text"] and "35" in out["crease"]["text"]
    assert out["file"].startswith("不是 JSON")
    assert out["english"] == {
        "text": "The neck is too thin for the head (short by 0.002)",
        "panel": "neck",
    }
    # Every rule of today has its words, which a new one can go without: it shows as its code.
    assert all(not text.isascii() for text in out["every"])
    # And so does everything the page is sent: a code it doesn't word would show in English.
    assert all(not text.isascii() for text in out["page"]), out["page"]
    assert not any("MISSING" in str(value) or "undefined" in str(value) for value in out.values())


def slider_level(base, values, name, value, always):
    """What the Python says about a slider at a value, ignoring what's broken wherever it stands."""
    moved = editor_sliders.apply_sliders(base, {**values, name: value})
    errors, seams = rule_reasons(moved, base["sex"])
    if set(errors) - always["errors"]:
        return "blocked"
    return "warned" if set(seams) - always["seams"] else None


def reasons(found):
    """What diagnostics break, as web/editor/limits.ts names it: a rule by its name, else code and parameter."""
    return [d.code[5:] if d.code.startswith("rule.") else f"{d.code} {d.values.get('key', '')}".strip() for d in found]


def rule_reasons(parameters, sex):
    """What the rules break, without how far they're missed by. Only for parameters inside their ranges."""
    body = parameters["body"]
    found = spec_problems(body, sex)
    return reasons(problems(body, sex)), reasons(seam_problems(body, sex) if not found else [])


def run_zones(cases, schema_module):
    return run_node(
        page_ranges() + """
        const { cases, definition } = input;
        console.log(JSON.stringify(cases.map(({ base, values, name }) =>
          limits.sliderZones({ name, base, sliders: values, definition, ranges: rangesFor(base), schema: tables(base) }))));
        """,
        {"cases": cases, "definition": editor_sliders.SLIDERS},
        {"schema": schema_module, "limits": EDITOR / "limits.ts"},
    )


@pytest.mark.node
def test_a_sliders_zones_are_where_the_python_says_it_breaks(schema_module):
    specs = body_specs()
    starts = [
        ("female-long", {}),
        ("female-compact", {}),
        ("female-slim", {}),
        ("female-balanced", {"heads": 6.5, "hipBreadth": 240.0}),
    ]
    cases = [
        {"base": {"sex": sex_of(preset), "body": specs[preset]}, "values": values, "name": name}
        for preset, values in starts
        for name in editor_sliders.SLIDERS["sliders"]
    ]
    out = run_zones(cases, schema_module)
    marked = 0
    for case, zones in zip(cases, out, strict=True):
        base, values, name = case["base"], case["values"], case["name"]
        low, high = editor_sliders.slider_bounds(name, base)
        at = lambda share: low + (high - low) * share  # noqa: B023
        samples = [
            rule_reasons(editor_sliders.apply_sliders(base, {**values, name: at(i / 32)}), base["sex"])
            for i in range(33)
        ]
        always = {
            "errors": set.intersection(*(set(row[0]) for row in samples)),
            "seams": set.intersection(*(set(row[1]) for row in samples)),
        }
        level = lambda share: slider_level(base, values, name, at(share), always)  # noqa: B023
        # Inside each zone, the Python says the same; between them, it says nothing is wrong.
        edges = [0.0, *(edge for zone in zones for edge in (zone["from"], zone["to"])), 1.0]
        for zone in zones:
            marked += 1
            assert level((zone["from"] + zone["to"]) / 2) == zone["level"], (name, zone)
            assert zone["reasons"]
        for gap_from, gap_to in zip(edges[::2], edges[1::2], strict=True):
            if gap_to - gap_from > 0.002:
                assert level((gap_from + gap_to) / 2) is None, (name, gap_from, gap_to, zones)
        # And each edge is where it changes, to within a slider's step.
        for zone in zones:
            if zone["from"] > 0:
                assert level(zone["from"] - 0.003) != zone["level"], (name, zone)
            if zone["to"] < 1:
                assert level(zone["to"] + 0.003) != zone["level"], (name, zone)
    assert marked  # The bodies do come up against some rule, or this tested nothing.


@pytest.mark.node
def test_what_a_slider_cant_mend_doesnt_mark_its_track(schema_module):
    # The toddler's big head keeps 0.45 of it inside the neck girth's range: only the rule breaks.
    spec = body_specs()["female-chibi"]
    # Broken wherever the hand stands, and only by the neck.
    base = {"sex": "female", "body": {**spec, "neckGirth": spec["headHeight"] * 0.45}}
    assert any(d.code == "rule.neck_thin_head" for d in problems(base["body"], "female"))
    cases = [
        {"base": base, "values": {}, "name": "handLength"},
        {"base": base, "values": {}, "name": "neckGirth"},
    ]
    hand, neck = run_zones(cases, schema_module)
    assert not [zone for zone in hand if "neck_thin_head" in zone["reasons"]]
    assert [zone for zone in neck if "neck_thin_head" in zone["reasons"]]


@pytest.mark.node
def test_a_file_missing_a_parameter_marks_no_track(schema_module):
    spec = body_specs()["female-long"]
    body = {key: value for key, value in spec.items() if key != "jawBreadthRatio"}
    base = {"sex": "female", "body": spec | {"jawBreadthRatio": spec["jawBreadthRatio"]}}
    zones = run_zones([{"base": base, "values": {}, "name": "jawBreadthRatio"}], schema_module)
    assert isinstance(zones[0], list)
    out = run_node(
        VALIDATE,
        {"sex": "female", "specs": [body]},
        {"schema": schema_module, "files": EDITOR / "params-file.ts"},
    )
    assert out == [{"errors": [{"code": "param.missing", "level": "error", "values": {"key": "jawBreadthRatio"}}], "seams": []}]


def test_every_slider_is_in_a_group_and_drives_real_parameters():
    definition = editor_sliders.SLIDERS
    grouped = [name for group in definition["groups"] for name in group["sliders"]]
    assert sorted(grouped) == sorted(definition["sliders"])
    for spec in definition["sliders"].values():
        if "param" in spec:
            kind, key = spec["param"].split(".")
            assert key in ({"heads", "stature"} if kind == "derived" else ALL_SPEC)
        for key in (*spec.get("gains", {}), *spec.get("torsoGains", {}), *spec.get("scales", {}), *spec.get("laws", {})):
            assert key in ALL_SPEC


@pytest.mark.parametrize("name", ("gluteFoldLength", "groinDepth", "groinRelief", "breastBaseBreadth", "pectoralThickness"))
def test_an_optional_slider_starts_at_its_parameters_default(name):
    # The default is the parameter table's (sotaigen.params.spec.Param.default), in shares of its unit.
    base = {"sex": "female", "body": {"torsoLength": 400.0, "hipBreadth": 200.0}}
    row = ALL_SPEC[name]
    unit = {"torso": 400.0, "hip": 200.0}[row.unit]
    assert editor_sliders.slider_start(name, base) == pytest.approx(row.default * unit)
    assert "default" not in editor_sliders.SLIDERS["sliders"][name]


@pytest.mark.node
def test_the_page_reads_the_same_defaults(schema_module):
    body = body_specs()["female-chibi"]
    names = [name for name, spec in editor_sliders.SLIDERS["sliders"].items()
             if spec.get("param", "").startswith("body.") and spec["param"][5:] not in body and not spec.get("measured")]
    out = run_node(
        """
        const tables = schema.SEXES.female;
        const definition = input.definition;
        console.log(JSON.stringify(input.names.map((name) => sliders.sliderStart(definition.sliders[name], { body: input.body }, tables) ?? null)));
        """,
        {"names": names, "body": body, "definition": editor_sliders.SLIDERS},
        {"schema": schema_module, "sliders": EDITOR / "sliders.ts"},
    )
    expected = [editor_sliders.slider_start(name, {"sex": "female", "body": body}) for name in names]
    assert out == pytest.approx(expected)


FILES = {
    "good": {"format": params_file.FORMAT, "version": 1, "sex": "female", "body": {"a": 1}},
    "keeps": {
        "format": params_file.FORMAT,
        "version": 1,
        "sex": "male",
        "body": {},
        "base": {"preset": "x"},
        "notes": "later",
        "more": {"deep": [1, 2]},
    },
    # Which sex is known is the rules' to say, as for a parameter: the file opens and says so.
    "a sex there isn't": {"format": params_file.FORMAT, "version": 1, "sex": "elf", "body": {}},
    "newer": {"format": params_file.FORMAT, "version": 2, "sex": "female", "body": {}},
    "version zero": {"format": params_file.FORMAT, "version": 0, "sex": "female", "body": {}},
    "version text": {"format": params_file.FORMAT, "version": "1", "sex": "female", "body": {}},
    "other format": {"format": "glTF", "version": 1, "sex": "female", "body": {}},
    "no format": {"version": 1, "sex": "female", "body": {}},
    "no sex": {"format": params_file.FORMAT, "version": 1, "body": {}},
    "sex not text": {"format": params_file.FORMAT, "version": 1, "sex": 1, "body": {}},
    "no body": {"format": params_file.FORMAT, "version": 1, "sex": "female"},
    "body list": {"format": params_file.FORMAT, "version": 1, "sex": "female", "body": []},
    "a list": [],
}


def read_as_the_page_does(text):
    """What web/serve.mjs answers the page for a file's text: sotaigen.params.file run on it."""
    done = subprocess.run(
        [sys.executable, "-m", "sotaigen.params.file"], input=text, capture_output=True, text=True, check=True, cwd=ROOT
    )
    return json.loads(done.stdout)


def test_the_page_reads_files_with_the_python_reader():
    texts = {name: json.dumps(file) for name, file in FILES.items()}
    texts["not json"] = "{"
    texts["not a number"] = '{"format": "sotaigen/params", "version": 1, "sex": "female", "body": {"a": NaN}}'
    for name, text in texts.items():
        answer = read_as_the_page_does(text)
        try:
            mine = {"ok": True, "file": params_file.read_parameter_file(text)}
        except params_file.FileError as error:
            mine = {"ok": False, "problems": [error.diagnostic.json()]}
        if name != "not a number":
            assert answer == mine, name
    assert answer["file"]["body"] == {"a": "nan"}
    keeps = read_as_the_page_does(texts["keeps"])["file"]
    assert keeps["unknown"] == {"notes": "later", "more": {"deep": [1, 2]}} and keeps["sex"] == "male"
    newer = read_as_the_page_does(texts["newer"])["problems"]
    assert newer == [{"code": "file.newer", "level": "error", "values": {"version": 2, "latest": 1}}]
    assert read_as_the_page_does(texts["no sex"])["problems"][0]["code"] == "file.no_sex"


@pytest.mark.node
def test_a_file_comes_back_with_what_this_reader_did_not_know(schema_module):
    first = params_file.write_parameter_file(
        {"a": 1.5}, "male", base={"preset": "x"}, unknown={"notes": "later"}, generator="me"
    )
    again = run_node(
        """
        console.log(JSON.stringify(files.writeParameterFile(input, schema, "me")));
        """,
        params_file.read_parameter_file(first),
        {"files": EDITOR / "params-file.ts", "schema": schema_module},
    )
    assert json.loads(again) == json.loads(first)
    read = params_file.read_parameter_file(again)
    assert read["unknown"] == {"notes": "later"}
    assert read["sex"] == "male"
    assert read["body"] == {"a": 1.5}


def test_a_head_from_parameters_is_the_presets_own():
    spec = body_specs()["female-long"]
    p = solved_controls(spec)
    head = face.HeadShape(p)
    rig = rigged("female-long")
    mine = face.bound_parts(head, face.head_parts(head), rig)
    theirs = face_parts("female-long")
    assert [part.name for part in mine] == [part.name for part in theirs]
    for a, b in zip(mine, theirs, strict=True):
        assert np.array_equal(a.verts, b.verts), a.name
    normals = face.blended_normals(generation.full_template(), rig.verts, head)
    assert np.array_equal(normals, head_normals("female-long"))


def test_a_body_made_in_memory_is_what_the_build_makes():
    rig = rigged("female-long")
    spec = body_specs()["female-long"]
    exact = generation.made(solved_controls(spec))
    assert np.array_equal(exact.verts, rig.verts)
    assert np.array_equal(exact.weights, rig.weights)
    assert json.dumps(exact.bones, default=list) == json.dumps(rig.bones, default=list)


@pytest.mark.slow
def test_the_worker_writes_the_bytes_the_export_writes(tmp_path, monkeypatch):
    name = "female-long"
    # What the command itself writes, beside build/ so that other tests' files stay as they are.
    monkeypatch.setattr(glb, "glb_path", lambda _name, _contact=False: tmp_path / "exported.glb")
    exported = glb.export_with_face(name).read_bytes()
    # As the page sends them, through JSON, so that nothing rests on Python's own ints and floats.
    spec = json.loads(json.dumps(body_specs()[name]))
    out = tmp_path / "made.glb"
    row = worker.answer({"id": 1, "sex": "female", "body": spec, "out": str(out)})
    assert row["ok"] is True
    assert out.read_bytes() == exported
    assert row["glbBytes"] == len(exported)
    assert row["failures"] == []
    assert row["broken"] == []
    assert row["seconds"]["total"] > 0


class InProcess:
    """Stands in for the worker's second process, so that a test's replacements reach what it runs."""

    def submit(self, work, *args):
        done = Future()
        done.set_result(work(*args))
        return done


def glb_info(path):
    return glb.read_glb(path.read_bytes())[0]["extras"]["sotaigen"]


def no_bake(*_args, **_kwargs):
    raise AssertionError("a broken mesh got as far as making correctives")


def test_a_broken_body_is_shown_but_no_export_writes_it(tmp_path, monkeypatch, capsys):
    """The page's preview still comes; the IPC export and `sotaigen glb` write nothing."""
    spec = body_specs()["female-slim"]
    moved = {**spec, "headHeight": spec["headHeight"] + 1}
    p = solved_controls(moved)
    exact = generation.made(p)
    bad = exact._replace(wrap={**exact.wrap, "flippedFaces": 3, "crossedMidline": 2})
    reasons = quality.broken(bad.wrap, export_base()["faces"])
    assert reasons == ["flipped faces after the wrap: 3", "vertices past the midline: 2"]
    monkeypatch.setattr(worker, "made", lambda _p: bad)
    monkeypatch.setattr(worker, "second_process", InProcess)

    # As the slider asks for it: a file to show, which says what it is, and a report for the page.
    shown = tmp_path / "shown.glb"
    row = worker.answer({"id": 1, "sex": "female", "body": moved, "out": str(shown)})
    assert row["ok"] is True and row["broken"] == reasons
    # The same breakage in the same words wherever it is found.
    assert row["failures"] == reasons
    info = glb_info(shown)
    assert [d for d in info["quality"]["found"] if d["level"] == "broken"] == as_json(reasons)
    # No poses are measured for a body no body has: it would take the release past the speed target.
    assert "poses" not in info["quality"]["checked"]

    # As the IPC export asks for it: nothing to show, so nothing is made, and no table is baked.
    monkeypatch.setattr(generation.Correctives, "production", no_bake)
    capsys.readouterr()
    baked = tmp_path / "baked.glb"
    row = worker.answer({"id": 2, "sex": "female", "body": moved, "contact": True, "out": str(baked)})
    assert row == {"id": 2, "ok": False, "broken": reasons}
    assert not baked.exists()
    assert "stage" not in capsys.readouterr().out

    # As the command asks for it, which then looks for what ends it, tried in test_mend.py.
    monkeypatch.setattr(mend, "holds", lambda _spec: True)
    monkeypatch.setattr(cli, "examined", lambda _spec, **_options: (bad, []))
    file = tmp_path / "body.json"
    file.write_text(params_file.write_parameter_file(moved, "female"))
    out = tmp_path / "out.glb"
    assert cli.main(["glb", str(file), str(out)]) == 1
    assert not out.exists()
    said = capsys.readouterr().out
    assert "not exported, the mesh is broken" in said and reasons[0] in said
    assert "back to the typical body's share" in said


def test_the_command_writes_only_warnings_into_the_file(tmp_path, monkeypatch):
    """`sotaigen glb` of a body with quality problems: written, and they go with it."""
    # Not a typical body's parameters, whose file would have its check poses measured too.
    spec = {**body_specs()["female-slim"], "headHeight": body_specs()["female-slim"]["headHeight"] + 1}
    p = solved_controls(spec)
    found = [
        rule("wrist_thick_hand", 0.01, "warning"),
        Diagnostic("target.missed", "warning", {"key": "headHeight", "asked": 1.0, "reached": 2.0}),
        Diagnostic("surface.legsPart", "warning"),
        Diagnostic("foot.off_ground", "warning", {"mm": 9.0, "limit": 1.0}),
    ]
    exact = generation.made(p)._replace(spec=Spec("female", spec, {"name": "female-slim"}), notes=tuple(found[:3]))
    monkeypatch.setattr(cli, "examined", lambda _spec, **_options: (exact, found))
    file = tmp_path / "body.json"
    file.write_text(params_file.write_parameter_file(spec, "female", base={"name": "female-slim"}))
    out = tmp_path / "out.glb"
    # Written is done: only a broken mesh keeps the file back, and what `check` found goes into it.
    assert cli.main(["glb", str(file), str(out)]) == 0
    info = glb_info(out)
    assert info["parameters"] == json.loads(file.read_text())
    assert "name" not in info
    assert info["quality"] == {
        "checked": ["wrap", "uv", "parameters"],
        "found": as_json([*quality.quality(exact.wrap, exact.verts, export_base()["faces"]), *found[:3]]),
    }
    assert "broken" not in info


def test_the_parameters_a_file_carries_are_the_same_however_a_number_was_spelled():
    # The page's JSON writes 124.0 as 124; the file must not tell them apart.
    spec = body_specs()["female-long"]
    as_ints = {key: int(value) if float(value).is_integer() else value for key, value in spec.items()}
    assert metadata.file_info(Spec("female", as_ints)) == metadata.file_info(Spec("female", spec))
    assert json.dumps(metadata.parameters(Spec("female", as_ints))) == json.dumps(metadata.parameters(Spec("female", spec)))


def test_a_body_no_typical_body_has_says_so_and_a_typical_bodys_says_which():
    spec = body_specs()["female-chibi"]
    named = metadata.file_info(Spec("female", spec))
    assert named["name"] == "female-chibi"
    # The same numbers for the other sex are another body's parameters.
    assert "name" not in metadata.file_info(Spec("male", spec))
    moved = metadata.file_info(Spec("female", {**spec, "headHeight": spec["headHeight"] + 1}))
    assert "name" not in moved
    # Nothing says which way the file was made, so the editor's is the export's.
    assert moved["generator"] == named["generator"] and set(moved["generator"]) == {"name", "version", "fingerprint"}
    assert metadata.file_info(Spec("female", spec), "blend")["format"] == "sotaigen/blend"


def test_the_worker_refuses_parameters_the_rules_refuse(tmp_path):
    spec = body_specs()["female-long"]
    body = {**spec, "crotchHeight": 5000.0}
    out = tmp_path / "x.glb"
    row = worker.answer({"id": 1, "sex": "female", "body": body, "out": str(out)})
    assert row["ok"] is False
    assert any(d.values.get("key") == "stature" for d in row["problems"])
    assert not out.exists()


def captured_bundle(monkeypatch, folder):
    """Instead of Blender, keep what it would have been given, and write a file where it would."""
    kept = []

    def fake_blender(_script, scratch, out):
        kept.append(Path(folder) / str(len(kept)))
        kept[-1].mkdir(parents=True)
        for name in ("input.json", "input.npz"):
            (kept[-1] / name).write_bytes((Path(scratch) / name).read_bytes())
        Path(out).write_bytes(b"blend")

    monkeypatch.setattr(blend, "run_blender", fake_blender)
    return kept


@pytest.mark.slow
@pytest.mark.usefixtures("blender_found")
def test_the_workers_blend_is_given_to_blender_as_the_commands_is(tmp_path, monkeypatch):
    """The page's .blend and `sotaigen blend` write the same file: Blender is given the same."""
    kept = captured_bundle(monkeypatch, tmp_path / "kept")
    name = "female-slim"
    spec = json.loads(json.dumps(body_specs()[name]))
    file = tmp_path / "body.json"
    file.write_text(params_file.write_parameter_file(spec, "female"))
    assert cli.main(["blend", str(file), str(tmp_path / f"{name}.blend")]) == 0
    out = tmp_path / "made.part.blend"
    row = worker.answer_blend({"id": 1, "blend": True, "sex": "female", "body": spec, "out": str(out)})
    assert row["ok"] is True and row["blendBytes"] == len(b"blend") and out.exists()
    command, page = kept
    assert json.loads((page / "input.json").read_text()) == json.loads((command / "input.json").read_text())
    with np.load(command / "input.npz") as theirs, np.load(page / "input.npz") as ours:
        assert sorted(theirs.files) == sorted(ours.files)
        for key in theirs.files:
            assert np.array_equal(theirs[key], ours[key]), key


@pytest.mark.usefixtures("blender_found")
def test_only_a_head_swap_lets_a_neck_too_thin_for_the_head_through(tmp_path, monkeypatch, capsys):
    from sotaigen.params.rules import NECK_THIN_FOR_HEAD, problems

    body = json.loads(json.dumps(body_specs()["female-chibi"]))
    body = {**body, "neckGirth": 0.46 * body["headHeight"]}
    assert [d for d in problems(body, "female") if d.code == f"rule.{NECK_THIN_FOR_HEAD}"]
    file = tmp_path / "thin.json"
    file.write_text(params_file.write_parameter_file(body, "female"))
    out = tmp_path / "thin.blend"
    written = []

    def write(made_body, path, **_options):
        written.append(made_body)
        path.write_bytes(b"x")
        return path

    monkeypatch.setattr(cli, "write_blend", write)
    assert cli.main(["blend", str(file), str(out)]) == 1
    assert not written
    assert cli.main(["blend", str(file), str(out), "--for-head-swap"]) == 0
    # The body's notes go into the file's quality list, and they say what was let through.
    let_through = [d for d in written[0].notes if d.code == f"rule.{NECK_THIN_FOR_HEAD}"]
    assert let_through and let_through[0].level == "warning" and let_through[0].values["forHeadSwap"]
    assert str(let_through[0]).startswith("warning: the neck is too thin for the head")
    with pytest.raises(SystemExit):
        cli.main(["glb", str(file), str(out), "--for-head-swap"])
    capsys.readouterr()


@pytest.mark.usefixtures("blender_found")
def test_the_worker_says_why_it_wrote_no_blend(tmp_path, monkeypatch):
    kept = captured_bundle(monkeypatch, tmp_path / "kept")
    spec = body_specs()["female-slim"]
    out = tmp_path / "made.part.blend"
    asked = {"id": 1, "blend": True, "sex": "female", "body": spec, "out": str(out)}

    bad = worker.answer_blend({**asked, "body": {**spec, "crotchHeight": 5000.0}})
    assert bad["ok"] is False and bad["problems"]

    monkeypatch.setattr(worker, "can_write_blend", lambda: "Blender was not found")
    assert worker.answer_blend(asked) == {"id": 1, "ok": False, "noBlender": True}

    # A broken mesh is refused before anything else is made of it, by the page and the command alike.
    monkeypatch.setattr(worker, "can_write_blend", lambda: None)
    wrap = {"flippedFaces": 3, "crossedMidline": 2}
    exact = Body(None, None, None, wrap, {})
    for module in (worker, cli):
        monkeypatch.setattr(module, "examined", lambda _spec, **_options: (exact, []))
    row = worker.answer_blend(asked)
    assert row == {"id": 1, "ok": False, "broken": quality.broken(wrap, export_base()["faces"])}
    assert row["broken"] == ["flipped faces after the wrap: 3", "vertices past the midline: 2"]
    file = tmp_path / "body.json"
    file.write_text(params_file.write_parameter_file(spec, "female"))
    monkeypatch.setattr(mend, "holds", lambda _spec: True)
    assert cli.main(["blend", str(file), str(out)]) == 1
    assert not out.exists() and not kept

    # Blender failing is a failure the worker survives, with the long part in the terminal.
    def blender_failed(*_args):
        raise RuntimeError("Blender failed: blender -b\nits long output")

    monkeypatch.setattr(worker, "write_blend", lambda *_args, **_options: blender_failed())
    assert worker.answer_blend(asked) == {"id": 1, "ok": False, "error": "Blender failed: blender -b"}


@pytest.mark.node
def test_a_broken_bodys_reasons_are_in_words():
    open_skin = quality.broken({"flippedFaces": 0, "crossedMidline": 0}, [[0, 1, 2]])
    wrap = {"flippedFaces": 3, "crossedMidline": 2}
    located = {"flippedFaces": 3, "crossedMidline": 2, "brokenAt": {"flippedFaces": {"neck": 2, "shoulder": 1}, "crossedMidline": {"leg": 2}}}
    out = run_node(
        SAY_JS
        + """
        const say = sayFrom(input.zh);
        const labels = { kneeGirth: "膝围" };
        const [flipped, midline] = input.located;
        console.log(JSON.stringify({
          lines: input.lines.map((found) => messages.translate(found, {}, say).text),
          located: messages.translate(flipped, {}, say),
          midline: messages.translate(midline, {}, say),
          bare: messages.translate(input.lines[0], {}, say),
          open: messages.translate(input.lines.at(-1), {}, say),
          found: messages.translateMend({ key: "kneeGirth", part: "leg", before: 200, after: 240.5, unit: "mm", tried: 3, total: 35 }, labels, say),
          foundEnglish: messages.translateMend({ key: "definition", part: "quick", before: 0.5, after: 0.25, unit: "", tried: 3, total: 35 }, {}, sayFrom(input.en)),
          none: messages.translateMend({ tried: 35, total: 35 }, labels, say),
        }));
        """,
        {
            "lines": as_json([*quality.broken(wrap, [[0, 1, 2], [2, 1, 0]]), *open_skin]),
            "located": as_json(quality.broken(located, [[0, 1, 2], [2, 1, 0]])),
            "zh": catalogue("zh"),
            "en": catalogue("en"),
        },
        {"messages": EDITOR / "messages.ts"},
    )
    assert "翻转" in out["lines"][0] and "3" in out["lines"][0]
    assert "中线" in out["lines"][1] and "2" in out["lines"][1]
    assert out["lines"][2].startswith("皮肤不闭合：3 条边") and "不是参数" in out["lines"][2]
    # Where it broke goes with the words, and the group that has the most is where to go.
    assert "颈部 2" in out["located"]["text"] and "肩胸 1" in out["located"]["text"]
    assert out["located"]["panel"] == "neck"
    assert out["midline"]["panel"] == "leg" and "腿部 2" in out["midline"]["text"]
    assert "panel" not in out["bare"] and "panel" not in out["open"]
    # A parameter that ends it says so in its own unit, and where to go.
    assert "膝围" in out["found"]["text"] and "200 mm" in out["found"]["text"] and "240.5 mm" in out["found"]["text"]
    assert out["found"]["panel"] == "leg"
    assert "definition" in out["foundEnglish"]["text"] and out["foundEnglish"]["panel"] == "quick"
    assert "35" in out["none"]["text"] and "panel" not in out["none"]


# The page's scheduling, with a clock and a build this test turns by hand.
REBUILD_HARNESS = """
const timers = {
  now: null,
  setTimeout(run) { timers.now = run; return run; },
  clearTimeout(id) { if (timers.now === id) timers.now = null; },
  tick() { const run = timers.now; timers.now = null; if (run) run(); },
};
const asked = [];
const landed = [];
const rebuilder = new rebuild.Rebuilder({
  build: (parameters, signal) => new Promise((done, reject) => {
    const row = { parameters, signal, done, reject };
    signal.addEventListener("abort", () => reject(new Error("aborted")));
    asked.push(row);
  }),
  land: (result) => landed.push(result),
  timers,
});
const settle = () => new Promise((done) => setImmediate(done));
const phases = [];
const see = () => phases.push(rebuilder.phase);
"""


def run_rebuild(script):
    return run_node(REBUILD_HARNESS + script, {}, {"rebuild": EDITOR / "rebuild.ts"})


def test_a_drag_asks_for_nothing_and_a_release_asks_after_the_delay():
    out = run_rebuild(
        """
        rebuilder.hold();
        rebuilder.want({ a: 1 }, true);
        rebuilder.want({ a: 2 }, true);
        see();
        const during = asked.length + (timers.now ? 100 : 0);
        rebuilder.release();
        see();
        const waiting = asked.length;
        timers.tick();
        asked[0].done("body");
        await settle();
        see();
        console.log(JSON.stringify({ during, waiting, asked: asked.map((row) => row.parameters), landed, phases }));
        """
    )
    assert out["during"] == 0
    assert out["waiting"] == 0
    assert out["asked"] == [{"a": 2}]
    assert out["landed"] == ["body"]
    # "generating" from the moment of release, whether or not the delay has run out.
    assert out["phases"] == ["adjusting", "generating", "idle"]


def test_several_sliders_in_a_row_make_one_request():
    out = run_rebuild(
        """
        for (const a of [1, 2, 3]) {
          rebuilder.hold();
          rebuilder.want({ a }, true);
          rebuilder.release();
        }
        timers.tick();
        timers.tick();
        console.log(JSON.stringify(asked.map((row) => row.parameters)));
        """
    )
    assert out == [{"a": 3}]


def test_pressing_the_next_slider_stops_the_delay_without_it_moving():
    out = run_rebuild(
        """
        rebuilder.want({ a: 1 }, true);
        rebuilder.hold();
        const pressed = [Boolean(timers.now), rebuilder.phase];
        rebuilder.release();
        const released = [Boolean(timers.now), rebuilder.phase];
        console.log(JSON.stringify({ pressed, released }));
        """
    )
    assert out["pressed"] == [False, "adjusting"]
    assert out["released"] == [True, "generating"]


def test_only_the_latest_lands_and_the_older_is_cancelled():
    out = run_rebuild(
        """
        rebuilder.want({ a: 1 }, true, { now: true });
        const first = asked[0];
        rebuilder.want({ a: 2 }, true, { now: true });
        const aborted = first.signal.aborted;
        asked[1].done("new");
        await settle();
        console.log(JSON.stringify({ aborted, landed, phase: rebuilder.phase, cancelled: rebuilder.cancelled }));
        """
    )
    assert out == {"aborted": True, "landed": ["new"], "phase": "idle", "cancelled": 1}


@pytest.mark.node
def test_an_answer_to_old_parameters_cannot_replace_new_ones():
    # A build that ignores its signal and answers anyway, before and after the newer one.
    for order in (("old", "new"), ("new", "old")):
        out = run_node(
            """
            const pending = [];
            const landed = [];
            const rebuilder = new rebuild.Rebuilder({
              build: () => new Promise((done) => pending.push(done)),
              land: (result) => landed.push(result),
            });
            rebuilder.want({ a: 1 }, true, { now: true });
            rebuilder.want({ a: 2 }, true, { now: true });
            const answers = { old: pending[0], new: pending[1] };
            for (const name of input) {
              answers[name](name);
              await new Promise((done) => setImmediate(done));
            }
            console.log(JSON.stringify({ landed, phase: rebuilder.phase, dropped: rebuilder.dropped }));
            """,
            order,
            {"rebuild": EDITOR / "rebuild.ts"},
        )
        assert out == {"landed": ["new"], "phase": "idle", "dropped": 1}


def test_parameters_the_rules_refuse_ask_for_nothing_and_cancel_what_was_running():
    out = run_rebuild(
        """
        rebuilder.want({ a: 1 }, true, { now: true });
        rebuilder.want({ a: 2 }, false);
        timers.tick();
        await settle();
        console.log(JSON.stringify({
          count: asked.length, aborted: asked[0].signal.aborted, phase: rebuilder.phase, landed,
        }));
        """
    )
    assert out == {"count": 1, "aborted": True, "phase": "invalid", "landed": []}


def test_a_failed_build_is_not_left_saying_generating():
    out = run_rebuild(
        """
        rebuilder.want({ a: 1 }, true, { now: true });
        asked[0].reject(new Error("boom"));
        await settle();
        console.log(JSON.stringify({ phase: rebuilder.phase, error: rebuilder.error.message }));
        """
    )
    assert out == {"phase": "failed", "error": "boom"}


def test_a_body_seen_before_skips_the_delay_and_the_one_on_screen_asks_for_nothing():
    out = run_rebuild(
        """
        rebuilder.want({ a: 1 }, true, { now: true });
        asked[0].done("one");
        await settle();
        rebuilder.want({ a: 2 }, true, { now: true });
        asked[1].done("two");
        await settle();
        rebuilder.want({ a: 2 }, true);
        const same = [asked.length, rebuilder.phase];
        rebuilder.want({ a: 1 }, true);
        console.log(JSON.stringify({ same, back: [asked.length, Boolean(timers.now)] }));
        """
    )
    assert out == {"same": [2, "idle"], "back": [3, False]}


# A worker that speaks the editor worker's protocol, without Python. A body takes as long as it is told.
FAKE_WORKER = """
import { writeFileSync } from "node:fs";
import { createInterface } from "node:readline";
console.log(JSON.stringify({ ready: true, version: "v1" }));
createInterface({ input: process.stdin }).on("line", (line) => {
  const job = JSON.parse(line);
  if (job.contact) console.log(JSON.stringify({ id: job.id, stage: "shoulder" }));
  setTimeout(() => {
    if (job.body.boom) {
      console.log(JSON.stringify({ id: job.id, ok: false, error: "boom" }));
      return;
    }
    writeFileSync(job.out, `glb ${job.body.a}`);
    console.log(JSON.stringify({ id: job.id, ok: true, pid: process.pid, failures: [] }));
  }, job.body.ms ?? 0);
});
"""

SERVER_HARNESS = """
const { readFileSync } = await import("node:fs");
const worker = new serve.BodyWorker({ command: [process.execPath, input.fake], folder: input.folder });
const alive = (pid) => {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
};
const pause = (ms) => new Promise((done) => setTimeout(done, ms));
"""


def run_server(tmp_path, script):
    fake = tmp_path / "fake.mjs"
    fake.write_text(FAKE_WORKER)
    return run_node(
        SERVER_HARNESS + script + "\nworker.close();\nprocess.exit(0);\n",
        {"fake": str(fake), "folder": str(tmp_path)},
        {"serve": ROOT / "web" / "serve.mjs"},
    )


def test_a_newer_request_kills_the_worker_on_the_older_one(tmp_path):
    out = run_server(
        tmp_path,
        """
        await worker.ready();
        const slowPid = worker.child.pid;
        const slow = worker.request("slow", { body: { a: 1, ms: 60000 } });
        const quick = worker.request("quick", { body: { a: 2 } });
        const first = await slow.done;
        const second = await quick.done;
        await pause(100);
        console.log(JSON.stringify({
          first, ok: second.ok, killed: !alive(slowPid), text: readFileSync(second.out, "utf8"),
          different: second.pid !== slowPid,
        }));
        """,
    )
    assert out == {"first": {"superseded": True}, "ok": True, "killed": True, "text": "glb 2", "different": True}


def test_a_client_that_leaves_cancels_its_body_unless_another_still_wants_it(tmp_path):
    out = run_server(
        tmp_path,
        """
        await worker.ready();
        const payload = { body: { a: 1, ms: 200 } };
        const one = worker.request("same", payload);
        const two = worker.request("same", payload);
        const pid = worker.child.pid;
        one.leave();
        const kept = alive(pid) && worker.job !== null;
        const row = await two.done;
        const three = worker.request("gone", { body: { a: 2, ms: 60000 } });
        const before = worker.child.pid;
        three.leave();
        const left = await three.done;
        await pause(100);
        const warm = await worker.ready();
        console.log(JSON.stringify({
          kept, ok: row.ok, left, killed: !alive(before), warm, restarted: worker.child.pid !== before,
        }));
        """,
    )
    assert out == {
        "kept": True,
        "ok": True,
        "left": {"superseded": True},
        "killed": True,
        "warm": "v1",
        "restarted": True,
    }


def test_a_stage_the_worker_reports_is_kept_on_its_job_and_does_not_finish_it(tmp_path):
    out = run_server(
        tmp_path,
        """
        await worker.ready();
        const job = worker.request("ipc", { body: { a: 1, ms: 300 }, contact: true });
        await pause(150);
        const stage = worker.job?.stage;
        const row = await job.done;
        console.log(JSON.stringify({ stage, ok: row.ok, text: readFileSync(row.out, "utf8") }));
        """,
    )
    assert out == {"stage": "shoulder", "ok": True, "text": "glb 1"}


def test_a_failure_in_the_worker_comes_back_and_it_stays_up(tmp_path):
    out = run_server(
        tmp_path,
        """
        await worker.ready();
        const pid = worker.child.pid;
        const row = await worker.request("boom", { body: { boom: true } }).done;
        console.log(JSON.stringify({ ok: row.ok, error: row.error, same: worker.child.pid === pid }));
        """,
    )
    assert out == {"ok": False, "error": "boom", "same": True}


@pytest.mark.node
def test_no_worker_says_so_rather_than_waiting(tmp_path):
    out = run_node(
        """
        const worker = new serve.BodyWorker({ command: ["/no/such/program"], folder: input.folder });
        let said = null;
        try {
          await worker.ready();
        } catch (error) {
          said = error.message;
        }
        let jobSaid = null;
        try {
          await worker.request("key", { body: {} }).done;
        } catch (error) {
          jobSaid = error.message;
        }
        worker.close();
        console.log(JSON.stringify({ said, jobSaid }));
        """,
        {"folder": str(tmp_path)},
        {"serve": ROOT / "web" / "serve.mjs"},
    )
    assert "not running" in out["said"]
    assert "not running" in out["jobSaid"]


@pytest.mark.node
def test_a_body_is_kept_under_its_parameters_and_the_least_used_goes_first(tmp_path):
    out = run_node(
        """
        const { writeFileSync, existsSync, utimesSync } = await import("node:fs");
        const { join } = await import("node:path");
        const cache = new serve.BodyCache(input.folder, 2);
        let age = 100;
        const put = (key) => {
          const part = join(input.folder, `${key}.part.glb`);
          writeFileSync(part, key);
          cache.store(key, { name: key }, part);
          // Distinct times, whatever the file system's resolution.
          const when = new Date(Date.now() - 1000 * age--);
          utimesSync(cache.filePath(key), when, when);
        };
        put("a");
        put("b");
        const hit = cache.read("a");
        put("c");
        const left = ["a", "b", "c"].map((key) => cache.read(key));
        const same = [
          serve.cacheKey("v", "female", { x: 1, y: 2 }) === serve.cacheKey("v", "female", { y: 2, x: 1 }),
          serve.cacheKey("v", "female", { x: 1 }) === serve.cacheKey("v", "female", { x: 1.5 }),
          serve.cacheKey("v", "female", { x: 1 }) === serve.cacheKey("w", "female", { x: 1 }),
          serve.cacheKey("v", "female", { x: 1 }) === serve.cacheKey("v", "female", { x: 1 }, true),
          serve.cacheKey("v", "female", { x: 1 }) === serve.cacheKey("v", "male", { x: 1 }),
        ];
        const part = join(input.folder, "dead.part.glb");
        writeFileSync(part, "half");
        new serve.BodyCache(input.folder, 2);
        console.log(JSON.stringify({ hit, left, same, swept: !existsSync(part) }));
        """,
        {"folder": str(tmp_path)},
        {"serve": ROOT / "web" / "serve.mjs"},
    )
    assert out["hit"] == {"name": "a"}
    # The read made "a" newer than "b", so "b" is the one that went.
    assert out["left"] == [{"name": "a"}, None, {"name": "c"}]
    assert out["same"] == [True, False, False, False, False]
    assert out["swept"] is True


@pytest.mark.node
def test_blends_are_kept_apart_so_that_they_cant_push_a_glb_out(tmp_path):
    out = run_node(
        """
        const { existsSync, mkdirSync, readdirSync, writeFileSync } = await import("node:fs");
        const { join } = await import("node:path");
        const glbs = new serve.BodyCache(input.folder);
        const part = join(input.folder, "glb.part.glb");
        writeFileSync(part, "glb");
        glbs.store("glb", { name: "glb" }, part);
        const blendFolder = join(input.folder, "blend");
        const blends = new serve.BodyCache(blendFolder, 2, ".blend");
        for (const key of ["a", "b", "c"]) {
          const made = join(blendFolder, `${key}.1.part.blend`);
          writeFileSync(made, key);
          blends.store(key, { name: key }, made);
        }
        const kept = readdirSync(blendFolder).sort();
        // What a job left that was killed in the middle, which a next start sweeps.
        writeFileSync(join(blendFolder, "d.2.part.blend"), "half");
        writeFileSync(join(blendFolder, "d.2.part.tmp.blend"), "half");
        mkdirSync(join(blendFolder, "blend-abc"));
        writeFileSync(join(blendFolder, "blend-abc", "input.npz"), "half");
        new serve.BodyCache(blendFolder, 2, ".blend");
        console.log(JSON.stringify({
          glb: glbs.read("glb"),
          kept,
          swept: readdirSync(blendFolder).sort(),
          path: blends.filePath("a") === join(blendFolder, "a.blend"),
        }));
        """,
        {"folder": str(tmp_path)},
        {"serve": ROOT / "web" / "serve.mjs"},
    )
    assert out["glb"] == {"name": "glb"}
    assert len([name for name in out["kept"] if name.endswith(".blend")]) == 2
    assert len([name for name in out["kept"] if name.endswith(".json")]) == 2
    assert out["swept"] == out["kept"]
    assert out["path"] is True


def test_waist_height_slider_changes_only_its_optional_parameter():
    base = {"sex": "female", "body": body_specs()["female-long"]}
    assert editor_sliders.slider_start("waistContourShift", base) == 0
    changed = editor_sliders.apply_sliders(base, {"waistContourShift": 12.0})
    assert changed["body"] == base["body"] | {"waistContourShift": 12.0}


@pytest.mark.node
def test_abdominal_measured_starts_are_not_zero_defaults_and_zero_is_an_explicit_target(schema_module):
    base = {"sex": "female", "body": body_specs()["female-long"]}
    measured = {"frontArc": -3.1, "backHollow": 10.5}
    assert editor_sliders.slider_start("abdomenFrontArc", base) is None
    assert editor_sliders.slider_start("abdomenFrontArc", base, measured) == -3.1
    changed = editor_sliders.apply_sliders(base, {"abdomenFrontArc": 0, "lumbarHollow": 0})
    assert changed["body"] == base["body"] | {"abdomenFrontArc": 0, "lumbarHollow": 0}
    specs = editor_sliders.SLIDERS["sliders"]
    out = run_node(
        "const t = schema.SEXES.female;"
        "console.log(JSON.stringify([sliders.sliderStart(input.spec, input.base, t) ?? null, "
        "sliders.sliderStart(input.spec, input.base, t, input.measured), "
        "sliders.sliderStart(input.spec, input.changed, t, input.measured)]));",
        {"spec": specs["abdomenFrontArc"], "base": base, "measured": measured, "changed": changed},
        {"sliders": EDITOR / "sliders.ts", "schema": schema_module},
    )
    assert out == [None, -3.1, 0]


@pytest.mark.node
def test_a_breast_drawn_from_nothing_brings_its_shape_and_one_already_there_keeps_it(schema_module):
    child = {"sex": "female", "body": body_specs()["female-compact"]}
    teen = {"sex": "female", "body": body_specs()["female-long"]}
    assert "breastProjection" not in child["body"]
    drawn = editor_sliders.apply_sliders(child, {"breastProjection": 10.0})["body"]
    brings = editor_sliders.SLIDERS["sliders"]["breastProjection"]["brings"]
    torso = child["body"]["torsoLength"]
    for key, amount in brings.items():
        assert drawn[key] == pytest.approx(amount * (torso if key in MILLIMETRES else 1.0)), key
    assert not problems(drawn, "female")
    # The disabled sliders showed this shape, all but the gap: a body with a breast and no gap has
    # none, so that slider starts at 0, but a breast drawn from nothing isn't joined at the midline.
    starts = {key: editor_sliders.slider_start(key, child) for key in brings}
    assert {key: drawn[key] for key in brings if key != "breastGap"} == pytest.approx(
        {key: value for key, value in starts.items() if key != "breastGap"}
    )
    kept = editor_sliders.apply_sliders(teen, {"breastProjection": 30.0})["body"]
    assert kept == teen["body"] | {"breastProjection": 30.0}
    out = run_node(
        page_ranges() + """
        const { cases, definition } = input;
        console.log(JSON.stringify(cases.map(({ base, values }) =>
          sliders.applySliders(base, values, definition, rangesFor(base), tables(base)))));
        """,
        {
            "cases": [
                {"base": child, "values": {"breastProjection": 10.0}},
                {"base": teen, "values": {"breastProjection": 30.0}},
            ],
            "definition": editor_sliders.SLIDERS,
        },
        {"schema": schema_module, "sliders": EDITOR / "sliders.ts"},
    )
    assert out[0]["body"] == pytest.approx(drawn)
    assert out[1]["body"] == pytest.approx(kept)


def test_every_parameter_of_a_template_has_a_slider_or_a_reason():
    # What the head count, the macros and the measured thigh move instead, and the waist landmark,
    # which the waist's transition height moves (docs/systems/editor.md).
    indirect = {"waistDepth", "waistRise", "definition", "softness"}
    moved = {
        spec["param"].split(".")[1]
        for spec in editor_sliders.SLIDERS["sliders"].values()
        if spec.get("param", "").startswith("body.")
    }
    for sex, table in SEXES.items():
        assert set(table) - moved - indirect == set(), sex
    assert moved <= set().union(*(set(table) for table in SEXES.values()))


def test_every_slider_says_what_it_means():
    for name, spec in editor_sliders.SLIDERS["sliders"].items():
        for locale in ("en", "zh"):
            words = catalogue(locale)
            assert words.get(f"slider_{name}") and words.get(f"slider_{name}_help"), (
                f"{locale}: {name} has no name or help"
            )
            assert (f"slider_{name}_low" in words) == (f"slider_{name}_high" in words), (
                f"{locale}: {name}'s ends"
            )
            assert (f"slider_{name}_reading" in words) == ("reading" in spec), (
                f"{locale}: {name}'s reading"
            )
        if spec.get("param", "").startswith("body."):
            in_mm = spec["param"].split(".")[1] in MILLIMETRES
            assert (spec.get("unit") == "mm") == in_mm, name


@pytest.mark.parametrize("name", body_specs())
def test_a_presets_lengths_and_targets_are_the_body_it_makes(name):
    preset = body_specs()[name]
    p = solved_controls(preset)
    body, L = calibrated(p)

    # What is true by definition, or by the calibration.
    exact = {
        "crotchHeight": L["crotch"],
        "neckLength": L["chin"] - L["pit"],
        "headHeight": L["head"],
        "torsoLength": L["T"],
        "upperArmLength": np.linalg.norm(L["elbow"] - L["shoulderJoint"]),
        "lowerArmLength": np.linalg.norm(L["wrist"] - L["elbow"]),
        "handLength": L["handLength"],
        "footLength": L["footLength"],
        "ankleHeight": L["ankle"],
        "kneeHeight": L["knee"],
        "acromionRise": L["acromion"][2] - L["pit"],
        "biacromialBreadth": 2 * L["acromion"][0],
    }
    for key, mm in exact.items():
        assert preset[key] == pytest.approx(mm, rel=1e-9), key
    assert preset["waistBreadth"] == pytest.approx(width_at(body, L["waist"]), rel=0.02)
    hip = max(width_at(body, L["at"](f)) for f in np.linspace(-0.05, 0.25, 13))
    assert preset["hipBreadth"] == pytest.approx(hip, rel=0.02)

    # What is asked for as a measurement is what the surface measures, read afresh.
    assert target_misses(preset) == {}
    for key, mm in measure(p).items():
        assert mm == pytest.approx(preset[key], abs=0.05), key
    assert problems(preset, sex_of(name)) == []


@pytest.mark.node
def test_a_reading_on_the_page_is_the_sum_it_says(schema_module):
    preset = body_specs()["female-balanced"]
    specs = {k: v for k, v in editor_sliders.SLIDERS["sliders"].items() if "reading" in v}
    out = run_node(
        """
        const out = {};
        for (const [name, spec] of Object.entries(input.specs)) {
          out[name] = sliders.readingMm(spec.reading, input.body, sliders.sliderStart(spec, { body: input.body }));
        }
        console.log(JSON.stringify(out));
        """,
        {"specs": specs, "body": preset},
        {"sliders": EDITOR / "sliders.ts"},
    )
    assert out["headWidthRatio"] == pytest.approx(preset["headWidthRatio"] * preset["headHeight"])
    assert out["headDepth"] == pytest.approx(preset["headDepth"] * preset["headHeight"] * 0.92)
    # A control value isn't called a measurement, nor a measurement given an estimate of itself.
    assert not {"chestBreadth", "chestDepth", "neckGirth", "upperArmGirth", "calfGirth"} & specs.keys()


def message_inputs(message):
    """The inputs a message uses: its {name}s, or a message with variants' declared inputs."""
    if isinstance(message, str):
        return set(re.findall(r"\{(\w+)\}", message))
    return {
        d.split()[1] for v in message for d in v.get("declarations", []) if d.startswith("input ")
    }


def test_the_catalogues_have_the_same_messages_and_inputs():
    # A missing translation compiles and falls back to English at run time, so they're compared here.
    en, zh = catalogue("en"), catalogue("zh")
    assert sorted(en) == sorted(zh)
    wrong = [
        key for key in en if key != "$schema" and message_inputs(zh[key]) - message_inputs(en[key])
    ]
    assert wrong == []
    # One selector per variant at most: a wrong match key only shows at run time, as the message's key.
    for key, message in en.items():
        if not isinstance(message, str):
            assert all(len(v["selectors"]) == 1 for v in message), key


@pytest.mark.node
def test_every_name_the_page_looks_up_has_its_message():
    words = catalogue("en")
    definition = editor_sliders.SLIDERS
    wanted = [f"group_{group['id']}" for group in definition["groups"]]
    wanted += sorted(
        {
            f"section_{spec['section']}"
            for spec in definition["sliders"].values()
            if "section" in spec
        }
    )
    wanted += [f"sex_{sex}" for sex in SEXES]
    wanted += sorted({f"figure_{path.stem}" for path in (ROOT / "sotaigen" / "data" / "bodies").glob("*/typical/*.json")})
    wanted += run_node(
        "console.log(JSON.stringify(messages.MESSAGE_KEYS));",
        {},
        {"messages": EDITOR / "messages.ts"},
    )
    page = "\n".join(path.read_text() for path in page_sources())
    wanted += re.findall(r"\bsay\(\"(\w+)\"", page) + re.findall(r"\bm\.(\w+)\(", page)
    assert [key for key in wanted if key not in words] == []


def page_sources():
    """The page's own code: not shadcn's copied-in pieces, nor the compiled messages."""
    for path in sorted([*WEB.glob("editor/**/*.ts*"), *WEB.glob("src/**/*.ts*")]):
        if not {"ui", "paraglide", "node_modules"} & set(path.relative_to(WEB).parts):
            yield path


def test_the_page_says_things_only_through_the_catalogue():
    # Text between tags, outside {expressions}, is a word to translate; so is any Chinese at all. Each
    # language's own name in the language menu is the one exception: it reads the same in either.
    text_node = re.compile(r"(?<![=-])>([^<>{}();=&|?]*[A-Za-z\u4e00-\u9fff][^<>{}();=&|?]*)<")
    own_names = '{ en: "English", zh: "中文" }'
    found = []
    for path in page_sources():
        source = re.sub(r"/\*[\s\S]*?\*/", "", path.read_text())
        source = re.sub(r"^\s*//.*$", "", source, flags=re.MULTILINE)
        if path.suffix == ".tsx":
            found += [f"{path.name}: {t.strip()}" for t in text_node.findall(source) if t.strip()]
        found += [
            f"{path.name}: {line.strip()}"
            for line in source.splitlines()
            if re.search(r"[\u4e00-\u9fff]", line) and own_names not in line
        ]
    assert found == []


def test_scale_rewrites_a_file_by_the_package_tools(tmp_path, capsys):
    body = body_specs()["female-long"]
    source = tmp_path / "long.json"
    source.write_text(params_file.write_parameter_file(body, "female", {"name": "female-long"}))
    out = tmp_path / "out.json"
    assert cli.main(["scale", str(source), str(out), "--torso", str(body["torsoLength"] * 0.95), "--stature", "1500"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "valid"
    written = params_file.read_spec(out)
    assert stature(written.body) == pytest.approx(1500.0)
    assert written.base == {"name": "female-long"}
    # The torso first, its shares carried along, then the whole body scaled: every share of the torso holds.
    for key in (key for key in TORSO_SHARES if key in body):
        assert written.body[key] / written.body["torsoLength"] == pytest.approx(body[key] / body["torsoLength"])
    assert cli.main(["scale", str(source), str(out), "--heads", "6.5"]) == 0
    assert heads(params_file.read_spec(out).body) == pytest.approx(6.5)


@pytest.mark.node
def test_a_thicker_thigh_lifts_the_hip_it_cannot_keep_shallow_and_a_thinner_one_leaves_it(schema_module):
    # The masses can't keep a hip shallow under a thick thigh (sotaigen/params/rules.py), so the
    # slider raises the hip just enough to meet that rule: the whole slider is usable, and never lowers it.
    cases = []
    for name in ("male-slim", "female-chibi", "female-balanced"):
        base = {"sex": name.split("-")[0], "body": body_specs()[name]}
        low, top = editor_sliders.ranges_for(base)["body"]["upperThighGirth"]
        thick = editor_sliders.apply_sliders(base, {"upperThighGirth": top})["body"]
        assert not problems(thick, base["sex"]), name
        assert thick["hipDepth"] >= base["body"]["hipDepth"]
        thin = editor_sliders.apply_sliders(base, {"upperThighGirth": low})["body"]
        assert thin == base["body"] | {"upperThighGirth": low}
        cases += [{"base": base, "values": {"upperThighGirth": top}}, {"base": base, "values": {"upperThighGirth": low}}]
    lifted = editor_sliders.apply_sliders(cases[0]["base"], cases[0]["values"])["body"]
    assert lifted["hipDepth"] > cases[0]["base"]["body"]["hipDepth"] + 10
    out = run_node(
        page_ranges() + """
        const { cases, definition } = input;
        console.log(JSON.stringify(cases.map(({ base, values }) =>
          sliders.applySliders(base, values, definition, rangesFor(base), tables(base)))));
        """,
        {"cases": cases, "definition": editor_sliders.SLIDERS},
        {"schema": schema_module, "sliders": EDITOR / "sliders.ts"},
    )
    for case, row in zip(cases, out, strict=True):
        assert row["body"] == pytest.approx(editor_sliders.apply_sliders(case["base"], case["values"])["body"])


def test_what_a_slider_lifts_is_in_shares_of_the_torso_and_its_rule_is_every_sex_s():
    from sotaigen.params.rules import rules
    from sotaigen.params.spec import ALL_SPEC

    for name, spec in editor_sliders.SLIDERS["sliders"].items():
        for other, rule_name in spec.get("lifts", {}).items():
            assert ALL_SPEC[other].unit == "torso", (name, other)
            for sex in ("female", "male"):
                assert rule_name in {row[0] for row in rules(sex)}, (name, sex)


def test_a_name_the_page_would_not_know_stops_the_translation():
    """Not left for the page to fail on: a bare name is an argument, a local or a module constant."""
    import ast

    read = lambda text, names: editor_schema.expression(ast.parse(text, mode="eval").body, set(), names)
    assert read("p['a'] + q", {"p", "q"}) == '(p["a"] + q)'
    with pytest.raises(editor_schema.Untranslatable, match="nowhere"):
        read("p['a'] + nowhere", {"p"})


@pytest.mark.node
def test_the_server_does_not_start_without_the_pages_rules(tmp_path):
    """A failed generation stops the server rather than leaving a page that checks nothing."""
    serve = ROOT / "web" / "serve.mjs"
    code = f"import {{ serve }} from {json.dumps(serve.as_uri())}; await serve(0, {{ cacheFolder: {json.dumps(str(tmp_path))} }});"
    done = subprocess.run(["node", "--input-type=module", "-e", code], capture_output=True, text=True, cwd=ROOT,
                          env={**os.environ, "SOTAIGEN_UV": "false"}, check=False, timeout=60)
    assert done.returncode != 0
    assert "Could not write the page's rules" in done.stderr


@pytest.mark.parametrize("locale", ["en", "zh"])
def test_every_check_pose_has_a_name_on_the_page(locale):
    """The page lists the GLB's check poses by these messages, and shows a bare key for one without."""
    from sotaigen.rig.poses import POSES

    missing = [name for name in POSES if f"pose_{name}" not in catalogue(locale)]
    assert not missing
