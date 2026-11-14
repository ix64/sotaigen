"""The hand on the head and at the face: solved for each typical body, within the limits, on the skin.

Every typical body takes every contact, posed the way the build poses (the correctives bound, then
pose_verts), and the crossings are counted as the pose metrics count them. Nothing here bakes.
"""

import json
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from sotaigen.corrective.crossings import SkinParts
from sotaigen.corrective.recipe import Correctives
from sotaigen.form.bodies import load_specs
from sotaigen.pipeline import rigged, surface
from sotaigen.rig.limits import clipped, joint_limits, nearest_xyz
from sotaigen.rig.skinning import euler, pose_verts
from sotaigen.rig.touch import (
    ARRIVED_DEG,
    ARRIVED_MM,
    ARRIVED_TIPS_DEG,
    BY_NAME,
    CLEARANCE_MM,
    CONTACTS,
    Arm,
    HeadAnchors,
    blend_poses,
    pose_table,
    solve_all,
    solve_contact,
    touch_name,
    unreached,
)

NAMES = list(load_specs("typical"))
# What must not pass through what. The elbow's own crease (lowerArm|upperArm) is the elbow card's
# business, and the upper arm against the trunk is the shoulder's.
INTO_HEAD = ("hand|head", "hand|neck", "head|lowerArm", "lowerArm|neck", "head|upperArm", "neck|upperArm", "hand|hand")
# The contacts a short arm under a big head only reaches lower down (docs/systems/touch.md).
MOVES = {"crown", "occiput"}


class Case(NamedTuple):
    rig: object
    correctives: Correctives
    parts: SkinParts
    rest: dict
    arm: Arm
    solved: dict


@pytest.fixture(scope="module")
def cases():
    out = {}
    for name in NAMES:
        rig = rigged(name)
        parts = SkinParts(rig.base)
        out[name] = Case(
            rig,
            Correctives.production(rig.base, rig.verts, rig.bones, rig.weights, field=surface(name)[0]),
            parts,
            parts.intersections(rig.verts)["intersectionsByPair"],
            Arm(rig.base, rig.verts, rig.bones, rig.weights),
            solve_all(rig.base, rig.verts, rig.bones, rig.weights),
        )
    return out


def posed(case, pose):
    rig = case.rig
    return pose_verts(rig.bones, rig.weights, case.correctives.bind(rig.verts, pose), pose)


def new_crossings(case, pose):
    found = case.parts.intersections(posed(case, pose))["intersectionsByPair"]
    return {pair: count - case.rest.get(pair, 0) for pair, count in found.items() if count > case.rest.get(pair, 0)}


@pytest.fixture(scope="module")
def crossings(cases):
    return {
        (name, contact.name): new_crossings(case, case.solved[contact.name].pose())
        for name, case in cases.items()
        for contact in CONTACTS
    }


@pytest.mark.slow
@pytest.mark.parametrize("name", NAMES)
def test_anchors_are_read_off_the_skin(cases, name):
    case = cases[name]
    anchors = HeadAnchors.from_skin(case.rig.base, case.rig.verts)
    assert anchors.crown_z == pytest.approx(case.rig.L["H"], abs=0.1)
    assert anchors.height == pytest.approx(case.rig.L["head"], abs=0.5)
    assert 0.25 < anchors.half_width / anchors.height < 0.5


@pytest.mark.parametrize("name", NAMES)
def test_the_hand_arrives_or_the_arm_is_named(cases, name):
    for contact in CONTACTS:
        solved = cases[name].solved[contact.name]
        if solved.reached:
            assert solved.arrivalMm <= ARRIVED_MM, contact.name
            assert solved.palmDeg <= (ARRIVED_DEG if contact.part == "palm" else ARRIVED_TIPS_DEG), contact.name
            assert CLEARANCE_MM - 0.5 <= solved.gapMm <= CLEARANCE_MM + 3.0, contact.name
        else:
            assert solved.notes, contact.name
    # All of these reach on every typical body, at the place they're named for or one further out.
    for contact in CONTACTS:
        solved = cases[name].solved[contact.name]
        assert solved.reached, (contact.name, solved.notes)
        if contact.name not in MOVES:
            assert solved.place == 0, contact.name


@pytest.mark.slow
@pytest.mark.parametrize("name", NAMES)
def test_nothing_goes_through_the_head_or_the_neck(crossings, name):
    for contact in CONTACTS:
        found = crossings[name, contact.name]
        assert not {pair: found[pair] for pair in INTO_HEAD if pair in found}, contact.name


def test_a_pose_solved_for_one_body_does_not_do_for_another(cases, crossings):
    """The reason for all this, and that the check above can see it: another body's angles go wrong."""
    wrong = 0
    for contact in CONTACTS:
        donor = cases[NAMES[0]].solved[contact.name].pose()
        for name in NAMES[1:]:
            case = cases[name]
            found = new_crossings(case, donor)
            skins = case.arm.skins(donor)
            gap = float(case.arm.skin.distance(case.arm.skinned(skins, case.arm.hand)).min())
            if any(pair in found for pair in INTO_HEAD) or gap > CLEARANCE_MM + 3.0:
                wrong += 1
    assert wrong >= len(CONTACTS) * (len(NAMES) - 1) * 3 // 4


@pytest.mark.parametrize("name", NAMES)
def test_within_the_joint_limits(cases, name):
    limits = joint_limits(cases[name].rig.bones)
    for contact in CONTACTS:
        pose = cases[name].solved[contact.name].pose()
        assert clipped(pose, limits) == pose, contact.name


@pytest.mark.parametrize("name", NAMES)
def test_the_upper_arm_is_at_the_angles_a_file_reads(cases, name):
    """A file's shoulder rhythm reads the turn's Euler angles nearest zero, and the runtime reads the
    pose's: told another reading of one turn, the two differ by 100 mm and more."""
    for contact in CONTACTS:
        angles = cases[name].solved[contact.name].pose()["leftUpperArm"]
        assert nearest_xyz(euler(angles)[:3, :3], (0.0, 0.0, 0.0)) == pytest.approx(angles, abs=0.01), contact.name


@pytest.mark.parametrize("name", NAMES)
def test_both_hands_keep_to_their_own_sides(cases, name):
    for contact in CONTACTS:
        if not contact.both:
            continue
        found = new_crossings(cases[name], cases[name].solved[contact.name].pose("both"))
        assert not {pair: n for pair, n in found.items() if "<>" in pair}, contact.name
        assert not {pair: found[pair] for pair in INTO_HEAD if pair in found}, contact.name


def test_the_right_hand_mirrors_the_left(cases):
    name = NAMES[0]
    case = cases[name]
    for contact in CONTACTS:
        solved = case.solved[contact.name]
        left, right = posed(case, solved.pose("left")), posed(case, solved.pose("right"))
        rest = np.asarray(case.rig.verts)
        twin = np.array([np.argmin(np.sum((rest - v * (-1, 1, 1)) ** 2, axis=1)) for v in rest[::50]])
        assert np.abs(left[::50] - right[twin] * (-1, 1, 1)).max() < 0.01, contact.name


def test_a_solve_gives_the_same_angles_twice(cases):
    case = cases[NAMES[0]]
    contact = BY_NAME["mouth"]
    assert solve_contact(case.arm, contact).angles == case.solved[contact.name].angles


def test_the_page_table_and_the_file_name_what_they_hold(cases):
    solved = cases[NAMES[0]].solved
    table = pose_table(solved)
    assert json.loads(json.dumps(table)) == table
    # The page's names are the .blend's: the left hand's and both hands'.
    expected = [touch_name(c.name, "left") for c in CONTACTS] + [touch_name(c.name, "both") for c in CONTACTS if c.both]
    assert sorted(table["poses"]) == sorted(expected) == sorted(table["reports"])
    assets = blend_poses(solved)
    assert set(table["poses"]) <= set(assets)
    for contact in CONTACTS:
        assert f"{touch_name(contact.name)}.left" in assets
        assert f"{touch_name(contact.name)}.right" in assets
        assert (f"{touch_name(contact.name)}.both" in assets) == contact.both
    assert not unreached(solved)


def test_the_page_offers_the_poses_python_names():
    text = Path("web/editor/regions.ts").read_text()
    listed = text[text.index("TOUCH_POSES = [") : text.index("];", text.index("TOUCH_POSES = ["))]
    named = re.findall(r'"([\w.]+)"', listed)
    assert sorted(named) == sorted(
        [touch_name(c.name, "left") for c in CONTACTS] + [touch_name(c.name, "both") for c in CONTACTS if c.both]
    )
    for language in ("en", "zh"):
        words = json.loads(Path(f"web/messages/{language}.json").read_text())
        for name in named:
            # A message's key can't hold a dot (web/editor/text.ts poseLabel).
            assert f"pose_{name.replace('.', '_')}" in words, (language, name)


def test_the_worker_solves_what_the_preset_does(cases, tmp_path):
    """The page's answer, by the worker's own function: the exact body is the body's, to the bit."""
    from sotaigen.params.rules import problems
    from sotaigen_dev.editor.worker import answer_touch

    name = NAMES[0]
    sex, group = name.split("-")
    body = json.loads(Path(f"sotaigen/data/bodies/{sex}/typical/{group}.json").read_text())["body"]
    assert not problems(body, sex)
    out = tmp_path / "touch.json"
    row = answer_touch({"id": 1, "body": body, "sex": sex, "out": str(out)})
    assert row["ok"], row
    assert json.loads(out.read_text()) == json.loads(json.dumps(pose_table(cases[name].solved)))
