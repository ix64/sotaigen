"""Where a broken mesh broke, and the search for a parameter that ends it (sotaigen/mend.py)."""

import json

import numpy as np
import pytest

from sotaigen import cli, generation, mend
from sotaigen.body import Body
from sotaigen.form import wrap
from sotaigen.form.bodies import load_specs
from sotaigen.form.solve import solved_controls
from sotaigen.output import blend, glb, quality
from sotaigen.output.glb import export_base
from sotaigen.params import file as params_file
from sotaigen.params.rules import problems, sex_of
from sotaigen.params.spec import SEXES
from sotaigen.template import load_template
from sotaigen_dev.editor import worker

GROUPS = {group["id"] for group in json.loads(mend.SLIDERS.read_text())["groups"]}
BROKEN_WRAP = {
    "flippedFaces": 3,
    "crossedMidline": 1,
    "brokenAt": {"flippedFaces": {"neck": 2, "arm": 1}, "crossedMidline": {"neck": 1}},
}
LANDMARKS = {"crotch": 800.0, "T": 500.0}


def typical(sex="female"):
    return mend.typical_bodies()[f"{sex}-{mend.TYPICAL_GROUP}"]


def another_body(sex="female"):
    """A body every move has something to do on: its parameters a little off the typical body's."""
    return next(spec for name, spec in load_specs("typical").items() if name.startswith(sex) and spec != typical(sex))


@pytest.mark.parametrize(
    ("region", "point", "part"),
    [
        ("neck", (0, 0, 1400), "neck"),
        ("arm", (300, 0, 1100), "arm"),
        ("hand", (500, 0, 800), "arm"),
        ("leg", (100, 0, 400), "leg"),
        ("foot", (100, -50, 20), "leg"),
        ("trunk", (100, 0, 800 + 0.7 * 500), "shoulder"),
        ("trunk", (100, 0, 800 + 0.45 * 500), "waist"),
        ("trunk", (100, -90, 800 + 0.1 * 500), "hip"),
        ("trunk", (100, 90, 800 + 0.1 * 500), "glute"),
    ],
)
def test_a_part_is_where_on_the_body_it_is(region, point, part):
    assert wrap.body_part(region, point, LANDMARKS) == part
    assert part in GROUPS


def test_a_tally_has_most_first():
    points = np.array([(0, 0, 1400), (0, 0, 1400), (300, 0, 1100)] * 1 + [(300, 0, 1100)] * 3, float)
    regions = ["neck", "neck", "arm", "arm", "arm", "arm"]
    assert list(wrap.tally_parts(regions, points, LANDMARKS).items()) == [("arm", 4), ("neck", 2)]


def test_a_wrap_says_where_it_flipped_and_a_whole_one_says_nothing(tmp_path, monkeypatch):
    p = solved_controls(typical())
    base = load_template()
    assert "brokenAt" not in wrap.wrap_body(base, p, tmp_path / "whole")

    # Every twentieth face of the body flipped, as the wrap's own test would find them.
    monkeypatch.setattr(wrap, "flipped_quads", lambda _body, _points, quads: np.arange(len(quads)) % 20 == 0)
    report = wrap.wrap_body(base, p, tmp_path / "flipped")
    tally = report["brokenAt"]["flippedFaces"]
    assert sum(tally.values()) == report["flippedFaces"] > 0
    assert list(tally.values()) == sorted(tally.values(), reverse=True)
    assert set(tally) <= GROUPS
    assert report["brokenAt"]["crossedMidline"] == {}
    assert quality.broken(report, base["faces"])[0].startswith(
        f"flipped faces after the wrap: {report['flippedFaces']}, at {next(iter(tally))} "
    )


def test_a_refusal_says_where_and_the_open_skin_says_no_parameter():
    faces = export_base()["faces"]
    assert quality.broken(BROKEN_WRAP, faces) == [
        "flipped faces after the wrap: 3, at neck 2, arm 1",
        "vertices past the midline: 1, at neck 1",
    ]
    assert quality.broken({"flippedFaces": 0, "crossedMidline": 0}, faces[1:])[0].endswith(
        "in the template, not from any parameter"
    )
    seams = {"footLowest": 0, "handSeamCreaseDeg": 0, "footSeamCreaseDeg": 0, "handSeamMissMm": 0, "footSeamMissMm": 0}
    verts = np.array(load_template()["verts"], float)
    # The same breakage in the same words, whether the export refuses it or `check` finds it.
    assert generation.wrap_failures(Body(verts, None, None, {**BROKEN_WRAP, **seams}, {})) == quality.broken(BROKEN_WRAP, faces)


@pytest.mark.parametrize("sex", SEXES)
def test_every_parameter_has_a_group_of_the_page(sex):
    parts = mend.parameter_parts()
    assert not set(SEXES[sex]) - parts.keys()
    assert {parts[key] for key in SEXES[sex]} <= GROUPS


def test_moves_try_the_broken_part_first_and_never_what_the_rules_refuse():
    spec, parts = another_body(), mend.parameter_parts()
    found = mend.moves(spec, "female", ["neck", "arm"])
    assert found
    for key, value in found:
        assert not problems({**spec, key: value}, "female")
        assert value != spec[key]
    ranks = [(["neck", "arm"] + [parts[key]]).index(parts[key]) if parts[key] in ("neck", "arm") else 2 for key, _ in found]
    assert ranks == sorted(ranks)
    # The typical body itself has nothing to move back.
    assert mend.moves(typical(), "female", ["neck"]) == []


def test_the_search_stops_at_the_first_move_that_ends_it(monkeypatch):
    spec = another_body()
    order = mend.moves(spec, "female", ["neck", "arm"])
    wanted = order[2][0]
    tried = []

    def holds(moved):
        key = next(key for key in moved if moved[key] != spec.get(key))
        tried.append(key)
        return key == wanted

    monkeypatch.setattr(mend, "holds", holds)
    heard = []
    found = mend.search(spec, "female", BROKEN_WRAP, lambda *at: heard.append(at))
    assert tried == [row[0] for row in order[:3]]
    assert heard == [(n, len(order)) for n in (1, 2, 3)]
    assert found["key"] == wanted and found["part"] == mend.parameter_parts()[wanted]
    assert found["tried"] == 3 and found["total"] == len(order)
    assert found["before"] == round(spec[wanted], 2) and found["after"] == round(order[2][1], 2)
    assert wanted in mend.sentence(found)


def test_the_search_says_when_no_one_parameter_ends_it(monkeypatch):
    monkeypatch.setattr(mend, "holds", lambda _spec: False)
    found = mend.search(another_body(), "female", BROKEN_WRAP)
    assert "key" not in found and found["tried"] == found["total"] > 0
    assert mend.sentence(found) == f"no one parameter back at the typical body's share ends it: {found['total']} tried"


def test_a_body_not_flipped_or_crossed_is_not_searched(monkeypatch, capsys):
    monkeypatch.setattr(mend, "holds", lambda _spec: pytest.fail("a wrap was tried"))
    whole = {"flippedFaces": 0, "crossedMidline": 0}
    assert mend.search(another_body(), "female", whole) is None
    mend.tell(another_body(), "female", whole)
    assert capsys.readouterr().err == ""


def test_a_real_wrap_answers_a_try():
    assert mend.holds(typical()) is True


def test_a_move_the_masses_cant_make_is_no_answer(monkeypatch):
    def unreachable(_spec):
        raise ValueError("past what this body can reach")

    monkeypatch.setattr(mend, "wrap_of", unreachable)
    assert mend.holds(typical()) is False


def test_a_refused_preset_tells_what_ends_it(tmp_path, monkeypatch, capsys):
    name = f"female-{mend.TYPICAL_GROUP}"
    (tmp_path / "bodies" / name).mkdir(parents=True)
    (tmp_path / "bodies" / name / "wrap.json").write_text(json.dumps(BROKEN_WRAP))
    monkeypatch.setattr(mend, "BUILD", tmp_path)
    monkeypatch.setattr(mend, "load_specs", lambda role=None: {name: another_body()})
    monkeypatch.setattr(mend, "holds", lambda _spec: True)
    mend.tell_body(name)
    said = capsys.readouterr().err
    assert "looking for a parameter" in said and "back to the typical body's share" in said


def test_blend_and_export_say_it_on_refusal(monkeypatch, capsys):
    name = next(iter(load_specs("typical")))
    told = []
    monkeypatch.setattr(mend, "tell_body", told.append)

    def refused(*_args, **_kwargs):
        raise quality.BrokenBody(name, ["flipped faces after the wrap: 3, at neck 3"])

    monkeypatch.setattr(blend, "can_write_blend", lambda: None)
    monkeypatch.setattr(blend, "blend_body", refused)
    assert blend.main([name]) == 1
    monkeypatch.setattr(glb, "export_with_face", refused)
    assert glb.main([name]) == 1
    assert told == [name, name]
    assert capsys.readouterr().err.count("at neck 3") == 2
    assert sex_of(name) in SEXES


def test_the_worker_looks_when_asked_and_refuses_what_the_rules_refuse(monkeypatch, capsys):
    spec = another_body()
    asked = {"id": 7, "mend": True, "sex": "female", "body": spec}
    monkeypatch.setattr(mend, "wrap_of", lambda _spec: BROKEN_WRAP)
    monkeypatch.setattr(mend, "holds", lambda _spec: True)
    row = worker.answer_mend(asked)
    assert row["id"] == 7 and row["ok"] is True and row["mend"]["key"]
    assert json.loads(capsys.readouterr().out) == {"id": 7, "stage": f"1/{row['mend']['total']}"}

    monkeypatch.setattr(mend, "wrap_of", lambda _spec: {"flippedFaces": 0, "crossedMidline": 0})
    assert worker.answer_mend(asked) == {"id": 7, "ok": True, "mend": None}

    bad = worker.answer_mend({**asked, "body": {**spec, "crotchHeight": 5000.0}})
    assert bad["ok"] is False and bad["problems"]


@pytest.mark.usefixtures("blender_found")
def test_the_file_commands_say_it_on_refusal(tmp_path, monkeypatch, capsys):
    spec = another_body()
    exact = Body(None, None, None, BROKEN_WRAP, {})
    monkeypatch.setattr(cli, "examined", lambda _spec, **_options: (exact, []))
    monkeypatch.setattr(mend, "holds", lambda _spec: True)
    file = tmp_path / "body.json"
    file.write_text(params_file.write_parameter_file(spec, "female"))
    for command, out in (("glb", tmp_path / "out.glb"), ("blend", tmp_path / "out.blend")):
        assert cli.main([command, str(file), str(out)]) == 1
        assert not out.exists()
        said = capsys.readouterr().out
        assert "at neck 2, arm 1" in said and "back to the typical body's share" in said
