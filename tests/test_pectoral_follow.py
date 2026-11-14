"""The pectoral plate following a raised arm, on top of the shoulder's targets."""

from functools import cache

import numpy as np
import pytest

from sotaigen.corrective import axilla, deltoid, pectoral
from sotaigen.corrective.recipe import Correctives
from sotaigen.pipeline import rigged, surface


@cache
def shoulder(name):
    r = rigged(name)
    plain = axilla.targets(r.base, r.verts, r.bones, r.weights)
    followed = pectoral.follow(plain, r.base, r.verts, r.bones, surface(name)[0])
    return r, plain, followed


def test_a_body_without_a_plate_keeps_its_targets():
    _, plain, followed = shoulder("female-balanced")
    assert pectoral.plate_of(surface("female-balanced")[0]) is None
    assert followed is plain


def test_production_carries_the_plate_on_the_shoulder_targets():
    r, _, followed = shoulder("male-balanced")
    made = Correctives.production(r.base, r.verts, r.bones, r.weights, field=surface("male-balanced")[0])
    # The raised shoulder's own shape goes on top of the plate's (sotaigen/corrective/deltoid.py).
    expected = deltoid.gather(followed, r.base, r.verts, r.bones)
    for name, delta in expected.items():
        np.testing.assert_array_equal(made.shoulder[name], delta)


def outward(name, target):
    """The plate's change under a target, along the rest normal: what follow added."""
    _, plain, followed = shoulder(name)
    return np.einsum("vi,vi->v", followed[target] - plain[target], normals(name))


@cache
def normals(name):
    from sotaigen.mesh import triangulate, vertex_normals

    r = rigged(name)
    return vertex_normals(np.asarray(r.verts), triangulate(r.base["faces"], np.asarray(r.base["verts"])))


@pytest.mark.parametrize("target", ["0Spread72", "0Spread110", "95", "60Spread35"])
def test_a_raised_arm_carries_the_plate_up_and_out_on_its_own_side(target):
    r = rigged("male-balanced")
    verts = np.asarray(r.verts)
    centres = {}
    for side, sign in (("left", 1), ("right", -1)):
        change = outward("male-balanced", f"axilla{side.title()}{target}")
        gained, lost = change > 0.5, change < -0.5
        assert gained.any() and lost.any()
        # Only this arm's half of the chest changes.
        assert not (np.abs(change) > 1e-9)[verts[:, 0] * sign < 0].any()
        up = verts[gained, 2].mean() - verts[lost, 2].mean()
        assert up > 0, (side, target)
        centres[side] = verts[gained].mean(0)
    # The two arms carry it alike, mirrored.
    left, right = centres["left"], centres["right"]
    np.testing.assert_allclose([left[0], left[2]], [-right[0], right[2]], atol=2.0)


def test_a_raised_arm_thins_what_it_carries():
    r = rigged("male-balanced")
    before = pectoral.plate_of(surface("male-balanced")[0]).thickness(np.asarray(r.verts))
    x = np.asarray(r.verts)[:, 0]
    for side, sign in (("left", 1), ("right", -1)):
        half = x * sign > 0
        change = outward("male-balanced", f"axilla{side.title()}0Spread110")
        assert (before + change)[half].max() < before[half].max()
