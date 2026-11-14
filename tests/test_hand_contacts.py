"""The two hand poses must meet their contact constraints on the final skinned mesh."""


import pytest

from sotaigen.form.bodies import load_bodies
from sotaigen_dev.motion import hand_contacts as contacts


@pytest.mark.parametrize("preset", load_bodies())
def test_grasps_bring_skin_to_the_target_without_passing_through_it(preset):
    for side, row in contacts.measure(preset).items():
        # Face centres of thumb and index pulp, with the pads facing one another.
        assert row["padGap"] < 0.02, side
        assert row["padNormalDot"] < -0.97, side
        assert row["padNormalOffset"] > 0, side
        # Five fingertips near the same cylinder. The thumb is on the opposite side
        # from each finger, and even the interior of each skin triangle clears it.
        assert row["handleMin"] >= 0, side
        assert max(row["distalGaps"].values()) < 0.015, side
        assert row["gripOpposition"] < -0.4, side
        assert row["crossings"] == {"padPinch": 0, "cylinderGrip": 0}, side
