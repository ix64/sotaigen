"""The range table: what it refuses, that the bodies meet it, and that what it allows holds up.

Random bodies are slow to wrap, so their measures are kept in build/cache/stress/ under the digest of the
sources, the same way a body's results are. A change to sotaigen/, data/ or template/ costs one cold
run. A check that gets stricter still applies, because only the measures are kept, not the verdicts.
"""

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from sotaigen.form.bodies import load_bodies, load_specs
from sotaigen.form.checks import field_failures, field_measures, sample_spec
from sotaigen.form.solve import solved_controls, target_misses
from sotaigen.params.file import write_parameter_file
from sotaigen.params.rules import (
    STATURE_RANGE,
    ParameterError,
    check_bodies,
    problems,
    rules,
    seam_problems,
    seam_rules,
    sex_of,
    spec_bounds,
)
from sotaigen.params.spec import (
    ALL_SPEC,
    CHAIN,
    MILLIMETRES,
    SEXES,
    estimated_controls,
    heads,
    layer,
    optional_parameters,
    proportions,
    required,
    sex_table,
    stature,
)
from sotaigen.pipeline import BUILD, fingerprints

# On the path, not loaded from a file: worker processes look its functions up by name.
from sotaigen_dev import stress

SEED = 20261104
# Few on purpose, so the whole run stays fast: enough to catch a range or a rule gone badly wrong. The
# table lets about 1 body in 75 through, so a new failure here may be that and not a regression; check
# with sotaigen_dev/stress.py random, which runs the same thing in the hundreds. See docs/systems/params.md.
FULL_SAMPLES = 6
WORKERS = 4

SPECS = load_specs()
A_BODY = SPECS["female-chibi"]
# The first body of each sex, to move one parameter of.
BODIES = {
    sex: next(spec for name, spec in SPECS.items() if sex_of(name) == sex)
    for sex in SEXES
}
BREAST = layer("breast")
MUSCLES = layer("muscle")


def measures(function, count, sex):
    """Rows for a sex's first `count` random bodies, from build/cache/stress/ when the sources haven't
    changed."""
    # The script decides what a row holds, and it isn't among the build's inputs.
    script = hashlib.sha256(Path(stress.__file__).read_bytes()).hexdigest()[:8]
    digest = fingerprints()["metrics"][:16]
    path = BUILD / "cache" / "stress" / f"test-{sex}-{SEED}-{count}-{digest}-{script}.json"
    if path.exists():
        return json.loads(path.read_text())
    rng = np.random.default_rng(SEED)
    bodies = [sample_spec(rng, sex) for _ in range(count)]
    with stress.worker_pool(WORKERS) as pool:
        rows = list(pool.map(function, bodies))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return rows


def test_the_stress_pools_workers_are_this_processs_children():
    """exit_with_parent ends a worker whose parent isn't this process, so on any start method this
    has to hold, or every worker ends as it starts. measures() above only starts one on a cold cache."""
    with stress.worker_pool(1) as pool:
        assert pool.submit(os.getppid).result() == os.getpid()


def show(bad):
    return "\n".join(f"{stress.failures(row)}: {row}" for row in bad)


def test_proportions_are_shares_of_the_torso_and_rules_read_them_at_any_size():
    q = proportions(A_BODY)
    assert q["torsoLength"] == 1.0
    assert q["stature"] == pytest.approx(stature(A_BODY) / A_BODY["torsoLength"])
    assert q["headWidthRatio"] == A_BODY["headWidthRatio"]
    bigger = {k: v * 1.3 if k in MILLIMETRES else v for k, v in A_BODY.items()}
    assert proportions(bigger) == pytest.approx(q)


def test_every_preset_meets_the_table():
    check_bodies(SPECS)


@pytest.mark.parametrize("name", SPECS)
def test_presets_sit_inside_the_seam_rules(name):
    assert seam_problems(SPECS[name], sex_of(name)) == []


def test_every_body_belongs_to_a_template_and_every_template_has_a_body():
    assert {sex_of(name) for name in SPECS} == SEXES.keys()


@pytest.mark.parametrize("name", SPECS)
def test_the_table_names_every_parameter_of_every_preset(name):
    sex = sex_of(name)
    assert set(required(sex)) <= SPECS[name].keys() <= SEXES[sex].keys()


def test_a_template_is_the_table_less_what_it_leaves_out():
    for sex, table in SEXES.items():
        assert list(table) == [key for key in ALL_SPEC if key in table]
        for key, row in table.items():
            assert row._replace(range=ALL_SPEC[key].range) == ALL_SPEC[key]
        assert required(sex) + optional_parameters(sex) == tuple(
            key for key in table if not table[key].optional
        ) + tuple(key for key in table if table[key].optional)


def test_breasts_are_female_and_the_muscles_are_both_templates():
    # The muscles are the same in women and men, only their size differs; a breast is a woman's.
    assert set(BREAST) <= SEXES["female"].keys()
    assert not set(BREAST) & SEXES["male"].keys()
    for sex in SEXES:
        assert set(MUSCLES) <= SEXES[sex].keys()


def test_a_template_of_its_own_range_changes_that_range_only():
    own = sex_table(without=("breastGap",), ranges={"biacromialBreadth": (0.4, 0.8)})
    assert own["biacromialBreadth"].range == (0.4, 0.8)
    assert "breastGap" not in own
    assert {k: r for k, r in own.items() if k != "biacromialBreadth"} == {
        k: r for k, r in ALL_SPEC.items() if k not in ("breastGap", "biacromialBreadth")
    }
    with pytest.raises(KeyError):
        sex_table(ranges={"tail": (0.0, 1.0)})


@pytest.mark.parametrize("key", BREAST)
def test_a_male_body_has_no_breast_to_set(key):
    male = BODIES["male"]
    low, high = spec_bounds(BODIES["female"], "female")[key]
    found = problems({**male, key: (low + high) / 2}, "male")
    assert found == [f"body has unknown parameter {key}"]


def test_a_pectoral_plate_under_a_breast_goes_as_far_as_it_was_looked_at():
    # The breast as a thickness field lies on the plate, which hides its lower border under it. A male
    # typical body's 12 mm was looked at on female-balanced (0.0223 T); past 0.0225 T nothing was.
    field = SPECS["female-balanced"]
    assert field["breastProjection"] > 0
    assert problems({**field, "pectoralThickness": 4.3}, "female") == []
    assert problems({**field, "pectoralThickness": 12.0}, "female") == []
    assert any("pectoral plate" in line for line in problems({**field, "pectoralThickness": 13.0}, "female"))
    layered = SPECS["female-layerbig"]
    assert layered["pectoralThickness"] > 0 and layered["breastProjection"] > 0
    assert problems(layered, "female") == []
    # Without a breast the plate is as on a male body; and the other three never meet the breast.
    assert problems({**A_BODY, "pectoralThickness": 12.0}, "female") == []
    for key in ("deltoidThickness", "trapeziusThickness", "latissimusThickness"):
        assert problems({**field, key: 7.0}, "female") == []
    # The young male bodies carry no plate, so take one that does.
    plated = next(
        spec for name, spec in SPECS.items() if sex_of(name) == "male" and spec.get("pectoralThickness")
    )
    assert problems(plated, "male") == []


def test_a_template_meets_only_its_own_rules():
    names = {sex: {name for name, _, _ in rules(sex)} for sex in SEXES}
    assert "pectoral_under_breast_thin" in names["female"] - names["male"]
    # The rest are the female stress test's, shared until male bodies are stress tested.
    assert names["male"] <= names["female"]
    assert seam_rules("male") == seam_rules("female")


def test_a_template_there_isnt_is_refused():
    assert problems(A_BODY, "elf") == ["there is no sex 'elf'"]


def test_controls_are_pure():
    spec = dict(A_BODY)
    estimated_controls(spec)
    assert spec == A_BODY


def test_the_chain_adds_up_to_the_stature_and_the_results_are_not_inputs():
    p = estimated_controls(A_BODY)
    assert p["stature"] == pytest.approx(sum(A_BODY[key] for key in CHAIN))
    assert p["heads"] == pytest.approx(heads(A_BODY))
    assert not {"stature", "heads", "T"} & ALL_SPEC.keys()


@pytest.mark.parametrize("key", [k for k in MILLIMETRES if k in required("female") and k not in CHAIN])
def test_moving_a_length_in_the_chain_moves_no_other_millimetre(key):
    """What the spec is for: a longer leg doesn't narrow the shoulders."""
    from sotaigen.form.anatomy import landmarks

    longer = {**A_BODY, "crotchHeight": A_BODY["crotchHeight"] * 1.02, "headHeight": A_BODY["headHeight"] * 1.01}
    before, after = estimated_controls(A_BODY), estimated_controls(longer)
    control, unit, offset = ALL_SPEC[key].control, ALL_SPEC[key].unit, ALL_SPEC[key].offset
    if unit == "torso":
        T0, T1 = landmarks(before)["T"], landmarks(after)["T"]
        assert (after[control] - offset) * T1 == pytest.approx((before[control] - offset) * T0)


@pytest.mark.parametrize(("sex", "key"), [(t, k) for t in SEXES for k in required(t)])
def test_a_value_outside_its_range_is_refused(sex, key):
    body = BODIES[sex]
    low, high = spec_bounds(body, sex)[key]
    for value in (low - (high - low), high + (high - low)):
        found = problems({**body, key: value}, sex)
        assert found
        # A length others are shares of can take them out of range before itself.
        assert key in CHAIN or key in " ".join(found) or "short by" in " ".join(found)


@pytest.mark.parametrize(("sex", "key"), [(t, k) for t in SEXES for k in optional_parameters(t)])
def test_optional_parameters_reject_invalid_values(sex, key):
    body = BODIES[sex]
    low, high = spec_bounds(body, sex)[key]
    margin = (high - low) * 0.01
    for value in (low - margin, high + margin, math.nan, True):
        assert any(key in p for p in problems(body | {key: value}, sex))


@pytest.mark.parametrize("key", BREAST)
def test_chest_controls_accept_their_full_range(key):
    low, high = spec_bounds(A_BODY, "female")[key]
    for value in (low, (low + high) / 2, high):
        assert not problems(A_BODY | {key: value}, "female")


@pytest.mark.parametrize("bad", [math.nan, math.inf, "tall", None, True])
def test_only_finite_numbers_pass(bad):
    assert problems({**A_BODY, "headHeight": bad}, "female")


def test_thigh_front_is_optional_and_limited():
    assert problems(A_BODY, "female") == []
    torso = A_BODY["torsoLength"]
    for share in (-0.03, 0.0, 0.03):
        body = {**A_BODY, "thighFrontShift": share * torso}
        assert problems(body, "female") == []
        assert estimated_controls(body)["thighFrontShift"] == pytest.approx(share)
    for value in (-0.031 * torso, 0.031 * torso, math.nan):
        assert any("thighFrontShift" in issue for issue in problems({**A_BODY, "thighFrontShift": value}, "female"))


# 0.01 and 0.5 are out of range; 0.25 is in it but over the high rule's 0.55 of this body's foot.
@pytest.mark.parametrize("share", [0.01, 0.5, 0.25])
def test_an_impossible_ankle_height_is_refused(share):
    height = share * A_BODY["torsoLength"]
    assert any("ankleHeight" in issue for issue in problems({**A_BODY, "ankleHeight": height}, "female"))


def test_an_ankle_under_the_low_rule_is_refused_inside_its_range():
    # On a 0.4 T foot the low rule's 0.24 of the foot stands above the range's 0.075 T, so 0.23 of the
    # foot is in range and breaks only the rule.
    T = A_BODY["torsoLength"]
    body = {**A_BODY, "footLength": 0.4 * T, "ankleHeight": 0.23 * 0.4 * T}
    assert any("ankleHeight is too low" in issue for issue in problems(body, "female"))


@pytest.mark.parametrize("height", [math.nan, "low"])
def test_an_ankle_height_that_is_no_number_is_refused(height):
    assert any("ankleHeight" in issue for issue in problems({**A_BODY, "ankleHeight": height}, "female"))


@pytest.mark.parametrize("ratio", [0.53, 0.67, math.nan])
def test_the_knee_height_must_stay_in_the_supported_share_of_the_leg(ratio):
    height = ratio * A_BODY["crotchHeight"]
    assert any("kneeHeight" in p for p in problems({**A_BODY, "kneeHeight": height}, "female"))


def test_a_stature_past_its_range_is_refused_though_nothing_sets_it():
    tall = {key: A_BODY[key] * 2 if key in MILLIMETRES else A_BODY[key] for key in A_BODY}
    assert stature(tall) > STATURE_RANGE[1]
    assert any("stature" in issue for issue in problems(tall, "female"))


@pytest.mark.parametrize("name", list(load_specs()))
@pytest.mark.parametrize("end", [0, 1])
def test_a_body_scaled_to_an_end_of_the_stature_range_is_accepted(name, end):
    """The stature tool lands a rounding error past the end it scales to (1800.0000000000002)."""
    spec = load_specs()[name]
    scale = STATURE_RANGE[end] / stature(spec)
    body = {key: value * scale if key in MILLIMETRES else value for key, value in spec.items()}
    sex = name.split("-")[0]
    assert not [issue for issue in problems(body, sex) if "stature" in issue]


def test_a_missing_or_unknown_parameter_is_refused():
    body = {key: value for key, value in A_BODY.items() if key != "jawBreadthRatio"}
    assert any("missing jaw" in problem for problem in problems(body, "female"))
    assert any("unknown" in problem for problem in problems({**A_BODY, "tail": 1.0}, "female"))


def test_check_presets_names_the_preset_and_every_problem():
    specs = {"female-tall": {**A_BODY, "jawBreadthRatio": 9.0, "headWidthRatio": 9.0}}
    with pytest.raises(ParameterError) as caught:
        check_bodies(specs)
    assert "female-tall" in str(caught.value)
    assert "jawBreadthRatio" in str(caught.value) and "headWidthRatio" in str(caught.value)


def test_check_presets_holds_each_body_to_its_own_template():
    breasted = SPECS["female-long"]
    check_bodies({"female-a": breasted, "male-a": BODIES["male"]})
    with pytest.raises(ParameterError) as caught:
        check_bodies({"male-a": breasted})
    assert "male-a: body has unknown parameter breastProjection" in str(caught.value)
    with pytest.raises(ParameterError, match="elf-a: there is no sex 'elf'"):
        check_bodies({"elf-a": A_BODY})


def write_data(folder, specs):
    """Bodies named <sex>-<group>, as test bodies of their sex: each its parameter file."""
    for name, spec in specs.items():
        sex, group = name.split("-", 1)
        path = folder / "bodies" / sex / "test" / f"{group}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(write_parameter_file(spec, sex))


def test_load_presets_reads_presets_that_meet_the_table(tmp_path, monkeypatch):
    write_data(tmp_path, {"female-tall": A_BODY, "male-tall": BODIES["male"]})
    monkeypatch.setattr("sotaigen.form.bodies.DATA", tmp_path)
    assert list(load_bodies()) == ["female-tall", "male-tall"]


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"headHeight": 900.0}, "head"),
        ({"jawBreadthRatio": 9.0}, "jawBreadthRatio"),
        ({"chestBreadth": 1.088 * 0.56 * A_BODY["torsoLength"], "biacromialBreadth": 0.46 * A_BODY["torsoLength"]}, "chest"),
    ],
)
def test_load_presets_refuses_what_the_table_refuses(tmp_path, monkeypatch, changes, named):
    write_data(tmp_path, {"female-tall": {**A_BODY, **changes}})
    monkeypatch.setattr("sotaigen.form.bodies.DATA", tmp_path)
    with pytest.raises(ParameterError, match=named):
        load_bodies()


@pytest.mark.parametrize("sex", SEXES)
def test_sampling_is_repeatable_and_valid(sex):
    first = sample_spec(np.random.default_rng(SEED), sex)
    assert first == sample_spec(np.random.default_rng(SEED), sex)
    assert first.keys() <= SEXES[sex].keys()
    assert not problems(first, sex)
    assert not seam_problems(first, sex)
    assert not target_misses(first)
    assert not field_failures(field_measures(solved_controls(first)))


def full_of(spec):
    return stress.evaluate(solved_controls(spec))


@pytest.mark.parametrize("sex", SEXES)
def test_random_bodies_in_range_wrap_and_rig_within_the_invariants(sex):
    rows = measures(full_of, FULL_SAMPLES, sex)
    bad = [row for row in rows if stress.failures(row)]
    assert not bad, show(bad)


def test_the_field_check_finds_what_the_surface_cannot_reach():
    p = solved_controls(A_BODY)
    assert field_failures(field_measures(p)) == []
    too_thin = field_measures({**p, "waistBreadth": 0.2})
    assert "waistBreadth" in field_failures(too_thin)
    too_narrow = field_measures({**p, "hipBreadth": 0.3})
    assert "hipBreadth" in field_failures(too_narrow)
    # How far the hip leads the chest is measured and reported, not refused (docs/DECISIONS.md).
    too_deep = field_measures({**p, "hipDepth": 0.7})
    assert too_deep["hipFrontExcess"] > field_measures(p)["hipFrontExcess"]
    assert "hipFront" not in field_failures(too_deep)


@pytest.mark.parametrize("waist_width", (0.32, 0.52))
def test_narrow_hip_cannot_move_paired_tissue_through_the_midline(waist_width):
    from sotaigen.form.anatomy import calibrated

    # The narrow end of the old raw ranges, in the fuller of the styles that were then applied on top.
    p = solved_controls(SPECS["female-compact"]) | {
        "waistBreadth": waist_width * 0.88,
        "waistDepth": 0.33 * 0.88**0.5,
        "hipBreadth": 0.44 * 1.26**0.5 * 0.88,
        "thighRadius": 0.10 * 1.26,
        "waist": 0.38,
    }
    field, L = calibrated(p)
    assert L["hipTissueAtMidline"]
    # Before the limit: the narrow waist passed with the trochanter at -9.6 mm;
    # the wide waist drove the pelvic bowl radius negative.
    trochanter, glute = field.groups[2]["prims"]
    outer_thigh = field.groups[-2]["prims"][-1]
    assert trochanter.center[0] == pytest.approx(0, abs=1e-8)
    assert min(glute.center[0], outer_thigh.center[0], field.groups[0]["prims"][3].radii[0]) > 0
    assert "hipTissueMidline" in field_failures(field_measures(p))
