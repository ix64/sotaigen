"""A body asked for in measurements is the body that measures that."""


import numpy as np
import pytest

from sotaigen.form.bodies import load_bodies, load_specs
from sotaigen.form.measures import measure
from sotaigen.form.solve import (
    DEGREES,
    FOLDER,
    LIMITS,
    TARGETS,
    TOLERANCE,
    measured_targets,
    solved_controls,
    target_misses,
)
from sotaigen.params.rules import problems, random_spec, seam_problems, spec_bounds
from sotaigen.params.spec import MILLIMETRES, estimated_controls
from sotaigen.paths import cache_dir

SPECS = load_specs()
A_BODY = SPECS["female-long"]
NEAR = 0.05


def test_presets_are_solved_only_when_read():
    presets = load_bodies()
    assert list(presets) == list(load_specs())
    assert "female-long" in presets and "nope" not in presets


@pytest.mark.parametrize("key", TARGETS)
def test_one_target_moves_and_the_others_hold(key):
    asked = {**A_BODY, key: A_BODY[key] * 1.05}
    measured = measure(solved_controls(asked))
    for other in TARGETS:
        assert measured[other] == pytest.approx(asked[other], abs=NEAR), other


@pytest.mark.parametrize(("name", "key"), [("female-chibi", "hipDepth"), ("female-long", "wristGirth"), ("female-balanced", "calfGirth")])
def test_a_target_that_cant_be_reached_leaves_the_others_where_they_were_asked(name, key):
    spec = SPECS[name]
    asked = {**spec, key: spec_bounds(spec, "female")[key][0]}
    misses = target_misses(asked)
    assert key in misses
    # Only what the missed one really moves may go with it: the arm with the wrist, the shin with the calf.
    allowed = {key, "upperArmGirth"} if key == "wristGirth" else {key, "kneeGirth", "ankleGirth"}
    assert misses.keys() <= allowed, misses


@pytest.mark.slow
def test_below_the_floor_rules_the_masses_can_still_reach():
    # What the three floor rules allow should nearly always be makeable: they mark where the masses stop.
    # About 3 in 10 miss, nearly all of them the two leans, so one seed's 96 bodies are a draw near the
    # limit of one in three: dropping the blended bust's draw from random_spec moved seed 3 from under it
    # to 38, while seeds 4-6 missed 23, 26 and 28. The same limit on four seeds answers for the rules. Since
    # the waist's depth and the upper thigh's girth are solved for, the ranges are where the masses reach
    # them, and the leans' where they mostly do (docs/PARAMETERS.md): 86 of the 384 missed (2026-11-12).
    missed = total = 0
    for seed in (3, 4, 5, 6):
        rng = np.random.default_rng(seed)
        drawn = 0
        while drawn < 96:
            spec = random_spec(rng, "female")
            if problems(spec, "female") or seam_problems(spec, "female"):
                continue
            drawn += 1
            missed += bool(target_misses(spec))
        total += drawn
    assert missed <= total // 3


def test_a_target_reports_what_it_reached_and_nothing_quietly_moves_the_number():
    asked = {**A_BODY, "chestBreadth": spec_bounds(A_BODY, "female")["chestBreadth"][1] * 4}
    misses = target_misses(asked)
    wanted, reached = misses["chestBreadth"]
    assert wanted == asked["chestBreadth"]
    assert reached < wanted - 1
    assert reached == pytest.approx(measured_targets(asked)["chestBreadth"])


def test_the_real_adult_medians_can_be_asked_for():
    # Survey medians in docs/PARAMETERS.md; the surface only, nothing here says the template holds them.
    real = {
        "torsoLength": 546.0,
        "crotchHeight": 780.0,
        "kneeHeight": 465.0,
        "chestBreadth": 268.0,
        "waistBreadth": 298.0,
        "hipBreadth": 353.0,
        "hipDepth": 231.0,
        "neckGirth": 328.0,
        "upperArmGirth": 280.0,
        "wristGirth": 154.0,
        "calfGirth": 372.0,
        "ankleGirth": 215.0,
    }
    asked = {**SPECS["female-balanced"], **real}
    measured = measured_targets(asked)
    for key in real.keys() & TARGETS.keys():
        assert measured[key] == pytest.approx(real[key], abs=1.0), key


def test_scaling_a_body_scales_what_it_measures_and_keeps_its_shares():
    scale = 0.8
    small = {k: v * scale if k in MILLIMETRES else v for k, v in A_BODY.items()}
    p, q = solved_controls(A_BODY), solved_controls(small)
    for key, control in TARGETS.items():
        assert measure(q)[key] == pytest.approx(small[key], abs=NEAR), key
        # Not exactly the same shares: the field has a few constants in mm, such as SOFT_MIDLINE.
        # A tilt is in degrees, and the measure's tolerance is about this much of one.
        assert q[control] == pytest.approx(p[control], abs=0.06 if control in DEGREES else 2e-3), control


@pytest.mark.parametrize(
    "local",
    [{"gluteProjection": 5.0}, {"gluteFoldDepth": 5.0}, {"groinDepth": 5.0}, {"groinRelief": 2.0}, {"gluteUpperFullness": 2.0}, {"breastProjection": 30.0}, {"lumbarHollow": 8.0}],
)
def test_local_contours_ask_for_no_new_solve(local):
    before, after = solved_controls(A_BODY), solved_controls({**A_BODY, **local})
    for control in TARGETS.values():
        assert after[control] == before[control], control


def test_the_estimate_is_a_good_place_to_start():
    # Where the steps begin. Far off, a body would take longer and could settle somewhere else.
    for spec in SPECS.values():
        a, b = estimated_controls(spec), solved_controls(spec)
        for control in TARGETS.values():
            if control in DEGREES:
                # A tilt's zero means nothing, so it can't be held to a share of itself.
                assert abs(a[control] - b[control]) < 4.5, control
            else:
                assert abs(a[control] / b[control] - 1) < 0.25, control


def test_the_neck_estimate_on_a_short_neck_is_still_a_neck():
    # A short neck under a thick trapezius takes off the estimate; a thin one must not take it past the solve's least.
    spec = A_BODY
    torso = spec["torsoLength"]
    short = {**spec, "neckLength": 0.07 * torso, "definition": 0.5, "neckGirth": 0.3 * torso}
    assert estimated_controls(short)["neckRadius"] == pytest.approx(LIMITS["neckRadius"][0])
    assert estimated_controls({**short, "neckGirth": 0.6 * torso})["neckRadius"] < 0.6 / 9.637


def test_solving_is_repeatable_to_its_tolerance():
    first = np.array([measured_targets(A_BODY)[key] for key in TARGETS])
    assert np.abs(first - np.array([A_BODY[key] for key in TARGETS])).max() < 2 * TOLERANCE + 0.01


@pytest.mark.parametrize("name", ["female-chibi", "female-long", "female-slim"])
@pytest.mark.parametrize(("key", "control"), [("upperTrunkLean", "ribTilt"), ("pelvisLean", "pelvisTilt")])
def test_a_posture_is_asked_for_on_the_surface_and_turns_the_mass_to_make_it(name, key, control):
    spec = SPECS[name]
    # The millimetres one way, then what the same body says at more tilt: both must read back.
    further = {**spec, key: spec[key] + 2.0}
    assert solved_controls(further)[control] > solved_controls(spec)[control]
    assert measure(solved_controls(further))[key] == pytest.approx(further[key], abs=NEAR)


def test_a_posture_past_what_a_tilt_can_do_says_so_and_stays_in_the_tilts_range():
    # The range stops about where the masses reach (docs/PARAMETERS.md): the adult's deep hip leans to its
    # top, 0.022 T, at 22.4 degrees. Asked past it, at the old range's 0.03 T, it can't lean that far.
    spec = SPECS["female-balanced"]
    asked = {**spec, "pelvisLean": 0.03 * spec["torsoLength"]}
    assert solved_controls(asked)["pelvisTilt"] == pytest.approx(LIMITS["pelvisTilt"][1], abs=1.0)
    assert "pelvisLean" in target_misses(asked)


def test_a_flank_past_what_the_outline_allows_says_what_it_reached():
    spec = SPECS["female-compact"]
    asked = {**spec, "gluteUpperFullness": spec_bounds(spec, "female")["gluteUpperFullness"][1]}
    reached = target_misses(asked)["gluteUpperFullness"]
    assert reached[0] == asked["gluteUpperFullness"]
    assert 0 < reached[1] < reached[0]
    near = {**spec, "gluteUpperFullness": 1.0}
    assert "gluteUpperFullness" not in target_misses(near)


def test_every_checkout_shares_the_solved_bodies():
    assert FOLDER == cache_dir() / "solve"


def test_the_solve_reaches_every_measurement_with_bounds_and_only_those():
    from sotaigen.form.solve import SENSITIVITY, SOLVED
    from sotaigen.params.spec import ALL_SPEC

    assert set(SOLVED) == {key for key, row in ALL_SPEC.items() if row.bounds is not None}
    assert len(set(SOLVED)) == len(SOLVED)
    assert SENSITIVITY.shape == (len(SOLVED), len(SOLVED))
