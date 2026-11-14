"""The breast thickness field's footprint and apex."""

import numpy as np
import pytest

from sotaigen.form.bodies import load_bodies
from sotaigen.form.breast import BreastEnvelope
from sotaigen.form.field import Body, soft_fold
from sotaigen.pipeline import rigged, surface

T = 500.0
L = {"T": T, "pit": 1300.0}
BREAST = {
    "breastProjection": 0.1,
    "breastBaseBreadth": 0.26,
    "breastFoldDrop": 0.40,
    "breastUpperPoleRatio": 0.45,
    "breastGap": 0.04,
}


def across(p):
    """x (mm) and the thickness across the chest's front at the apex's height."""
    envelope = BreastEnvelope(Body([], []), L, p)
    x = np.linspace(0.0, 0.4 * T, 2001)
    points = np.column_stack((x, np.full(len(x), -0.2 * T), np.full(len(x), envelope.apex)))
    return x, envelope.thickness(points)


def test_an_apex_in_the_middle_is_a_body_without_the_inner_pole():
    _, without = across(BREAST)
    _, middle = across({**BREAST, "breastApexRatio": 0.5})
    np.testing.assert_allclose(middle, without, atol=1e-9)


def test_the_inner_pole_moves_the_apex_out_and_keeps_the_footprint():
    x, t = across({**BREAST, "breastApexRatio": 0.64})
    inner = BREAST["breastGap"] * T / 2
    width = BREAST["breastBaseBreadth"] * T
    assert abs(x[np.argmax(t)] - (inner + 0.64 * width)) < 0.5
    # Across is read rounded at the midline, as the body's field is.
    standing = soft_fold(x[t > 0])
    assert abs(standing.min() - inner) < 0.5 and abs(standing.max() - (inner + width)) < 0.5
    assert np.isclose(t.max(), BREAST["breastProjection"] * T)


def test_footprints_lapped_over_the_midline_meet_in_a_crease():
    lapped = {**BREAST, "breastGap": -0.056, "breastApexRatio": 0.64}
    x, t = across(lapped)
    width = BREAST["breastBaseBreadth"] * T
    apex = lapped["breastGap"] * T / 2 + 0.64 * width
    assert abs(x[np.argmax(t)] - apex) < 0.5
    # The skin on the midline is lifted and the two sides' fields mirror, so it is the lowest point
    # between the apexes: a crease, not a joined dome.
    assert 0 < t[0] < t.max()
    inside = x <= apex
    assert t[inside].argmin() == 0
    envelope = BreastEnvelope(Body([], []), L, lapped)
    points = np.column_stack((x, np.full(len(x), -0.2 * T), np.full(len(x), envelope.apex)))
    mirrored = points * np.array([-1.0, 1.0, 1.0])
    np.testing.assert_allclose(envelope.thickness(mirrored), t)


BREASTED = [name for name, p in load_bodies().items() if p.get("breastProjection", 0) > 0]


@pytest.mark.parametrize("name", BREASTED)
def test_even_a_small_breast_reads_its_apex_on_the_finished_skin(name):
    # The front-most point of a small breast is the chest wall's, towards the sternum (42 mm inside the
    # apex on female-smallbust), so the apex reads as where the skin stands furthest off the chest wall:
    # within half the footprint's radius of where the parameters put it, and off by most of the projection.
    field, _ = surface(name)
    while not isinstance(field, BreastEnvelope):
        field = field.reference
    verts = rigged(name).verts
    across = field.apex_x if field.inner_pole is not None else field.middle
    near = (np.abs(verts[:, 0] - across) < field.half) & (np.abs(verts[:, 2] - field.apex) < field.half)
    near &= verts[:, 1] < 0
    off = field.reference(verts[near])
    furthest = verts[near][np.argmax(off)]
    assert np.hypot(furthest[0] - across, furthest[2] - field.apex) < 0.5 * field.half
    assert off.max() > 0.75 * field.size
