"""The shoulder setback: an optional parameter that moves the girdle and arm back, and nothing else."""

import json
from pathlib import Path

import numpy as np
import pytest

from sotaigen.form.anatomy import build, landmarks, scapular_masses
from sotaigen.form.axilla import POCKET_FOLLOWS, hollow
from sotaigen.form.solve import solved_controls
from sotaigen.generation import made
from sotaigen.params.rules import problems
from sotaigen.params.spec import ALL_SPEC
from sotaigen.rig.skeleton import setback_cap_share

BODIES = Path(__file__).resolve().parents[1] / "sotaigen" / "data" / "bodies"
SETBACK = 0.08


def a_typical_body():
    return json.loads((BODIES / "female" / "typical" / "chibi.json").read_text())["body"]


@pytest.fixture(scope="module")
def spec():
    return a_typical_body()


@pytest.fixture(scope="module")
def controls(spec):
    return solved_controls(spec)


def at(controls, setback):
    return {**controls, "shoulderSetback": setback}


def test_the_range_starts_at_no_setback():
    assert ALL_SPEC["shoulderSetback"].range[0] == 0.0
    assert ALL_SPEC["shoulderSetback"].optional


def test_the_girdle_and_arm_move_back_and_the_trunk_does_not(controls):
    still, moved = landmarks(controls), landmarks(at(controls, SETBACK))
    back = np.array([0.0, SETBACK * still["T"], 0.0])
    for key in ("acromion", "shoulderJoint", "elbow", "wrist"):
        assert moved[key] == pytest.approx(still[key] + back)
    for key in ("hipJoint", "kneeJoint", "ankleJoint"):
        assert moved[key] == pytest.approx(still[key])
    assert moved["pit"] == still["pit"] and moved["T"] == still["T"]


def test_no_setback_is_the_same_field_as_none_at_all(controls):
    points = np.array([[120.0, 40.0, 1250.0], [60.0, -60.0, 1100.0], [200.0, 20.0, 1150.0]])
    assert np.array_equal(build(controls)[0](points), build(at(controls, 0.0))[0](points))
    assert scapular_masses(controls, landmarks(controls)) == []


def test_the_masses_to_the_shoulder_come_in_gradually(controls):
    small = scapular_masses(at(controls, 0.01), landmarks(at(controls, 0.01)))
    large = scapular_masses(at(controls, SETBACK), landmarks(at(controls, SETBACK)))
    assert len(small) == len(large) > 0
    assert all(a.r1 < b.r1 for a, b in zip(small, large, strict=True))


def test_the_armpit_stays_between_the_folds(controls):
    """The pocket follows most of the way, not all: all the way it comes out through the back."""
    moved = at(controls, SETBACK)
    L = landmarks(moved)
    pocket, _ = hollow(L)
    still, _ = hollow(landmarks(controls))
    joint = L["shoulderJoint"]
    down = np.array([joint + np.array([0.0, y, -0.2 * L["T"]]) for y in np.linspace(-120, 120, 481)])
    centre = down[np.argmin(pocket(down)), 1]
    unmoved = np.array([p - (0.0, SETBACK * L["T"], 0.0) for p in down])
    start = unmoved[np.argmin(still(unmoved)), 1]
    assert centre - start == pytest.approx(POCKET_FOLLOWS * SETBACK * L["T"], abs=2.0)
    assert POCKET_FOLLOWS < 1.0


def test_the_cap_leaves_the_skin_far_ahead_of_a_set_back_joint(controls):
    L = landmarks(at(controls, SETBACK))
    joint = L["shoulderJoint"]
    points = np.array([joint, joint - (0.0, 0.2 * L["T"], 0.0)])
    near, far = setback_cap_share(points, L)
    assert near == 1.0
    assert far == 0.0
    assert setback_cap_share(points, landmarks(controls)) == 1.0


def test_the_top_of_the_range_wraps_clean(spec):
    top = ALL_SPEC["shoulderSetback"].range[1] * spec["torsoLength"]
    asked = {**spec, "shoulderSetback": top}
    assert problems(asked, "female") == []
    wrap = made(solved_controls(asked)).wrap
    assert wrap["flippedFaces"] == 0
    assert wrap["crossedMidline"] == 0
