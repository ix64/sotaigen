"""The shoulder card's sweep: the sculpted correctives unless a bake is asked for."""

import numpy as np
import pytest

from sotaigen.contact import axilla_cache
from sotaigen.corrective.recipe import Correctives
from sotaigen.pipeline import rigged, surface
from sotaigen_dev.motion.shoulder_sheet import shoulder_targets

NAME = "male-balanced"


def test_the_default_never_reaches_the_contact_table(monkeypatch):
    def bake(*_args, **_kwargs):
        raise AssertionError("the default sweep reached the contact table")

    monkeypatch.setattr(axilla_cache, "targets", bake)
    r = rigged(NAME)
    found = shoulder_targets(NAME, r.base, r.verts, r.bones, r.weights)
    made = Correctives.production(r.base, r.verts, r.bones, r.weights, field=surface(NAME)[0])
    for name, delta in made.shoulder.items():
        np.testing.assert_array_equal(found[name], delta)
    with pytest.raises(AssertionError, match="contact table"):
        shoulder_targets(NAME, r.base, r.verts, r.bones, r.weights, contact=True)
