"""The pectoral plate's lower border: a step on a bare chest, hidden under a breast."""

import numpy as np

from sotaigen.form.field import Body
from sotaigen.form.pectoral import BORDER, DROP, GROOVE_WIDTH, HIDDEN_DROP, PectoralPlate

T, CROTCH = 500.0, 800.0
L = {"T": T, "crotch": CROTCH, "pit": CROTCH + T}
BARE = {"pectoralThickness": 0.024}
BREAST = {
    **BARE,
    "breastProjection": 0.09,
    "breastFoldDrop": 0.40,
    "breastBaseBreadth": 0.24,
    "breastUpperPoleRatio": 0.45,
    "breastGap": 0.09,
}
SIZE = BARE["pectoralThickness"] * T


def thickness(p, x):
    """Heights (torso shares) and the plate's thickness down the chest's front at x (a torso share)."""
    z = np.linspace(0.45, 1.0, 1101)
    points = np.column_stack((np.full(len(z), x * T), np.full(len(z), -0.2 * T), CROTCH + z * T))
    return z, PectoralPlate(Body([], []), L, p).thickness(points)


def steepest(p, x, top=1.0):
    z, t = thickness(p, x)
    return np.abs(np.gradient(t, z * T))[z < top].max()


def test_a_bare_chest_keeps_the_short_drop():
    # Smoothstep's steepest slope is 1.5 times its rise over its run; out of the sternal groove the
    # plate has its full thickness.
    assert np.isclose(steepest(BARE, 0.1), 1.5 * SIZE / (DROP * T), rtol=0.01)


def test_under_a_breast_the_border_runs_down_into_the_fold():
    # Below the border, wherever it is; above it the plate rises as on a bare chest.
    for x in (0.0, 0.05, 0.12, 0.17, 0.21):
        assert steepest(BREAST, x, top=BORDER) <= 1.01 * 1.5 * SIZE / (HIDDEN_DROP * T)
    # Gone by the fold, or further down when the fold is high.
    z, t = thickness(BREAST, 0.05)
    end = min(1 - BREAST["breastFoldDrop"], BORDER - HIDDEN_DROP)
    assert t[z <= end].max() == 0.0
    assert t[(z > end + 0.01) & (z < BORDER)].min() > 0.0


def test_the_border_and_what_lies_above_it_stay_as_they_were():
    # At the sternum the border is at BORDER itself.
    z, bare = thickness(BARE, 0.0)
    _, breast = thickness(BREAST, 0.0)
    np.testing.assert_array_equal(breast[z >= BORDER], bare[z >= BORDER])


def test_a_breast_growing_from_nothing_lengthens_the_drop_smoothly():
    _, bare = thickness(BARE, 0.05)
    _, tiny = thickness({**BREAST, "breastProjection": 1e-6}, 0.05)
    assert np.abs(tiny - bare).max() < 1e-3


def test_between_the_breasts_the_plate_keeps_the_cleavage():
    # Below the apex the groove reaches out to the breasts' inner edges, so the skin between them is
    # lifted no more than the sternum is, and the plate is full again a groove's width past the edge.
    inner = BREAST["breastGap"] / 2
    fold = 1 - BREAST["breastFoldDrop"]
    apex = fold + BREAST["breastBaseBreadth"] * (1 - BREAST["breastUpperPoleRatio"])
    z = np.array([apex - 0.02, apex - 0.05])

    def at(x, p):
        points = np.column_stack((np.full(len(z), x * T), np.full(len(z), -0.2 * T), CROTCH + z * T))
        return PectoralPlate(Body([], []), L, p).thickness(points)
    assert (at(inner, BREAST) <= 1.02 * at(0.0, BREAST)).all()
    # 1 / GROOVE_FLOOR, less what the border rising towards the armpit takes off.
    assert (at(inner + GROOVE_WIDTH + 0.01, BREAST) > 1.8 * at(0.0, BREAST)).all()


def test_footprints_lapped_over_the_midline_keep_the_sternum_groove():
    # A lapped gap has no edge past the midline: the groove is the one of footprints meeting there.
    _, lapped = thickness({**BREAST, "breastGap": -0.056}, 0.0)
    _, met = thickness({**BREAST, "breastGap": 0.0}, 0.0)
    np.testing.assert_allclose(lapped, met)
