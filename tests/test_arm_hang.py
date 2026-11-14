"""A hanging arm stays out of the side, hip and thigh down to the target, and only there."""

import json

import numpy as np
import pytest

from sotaigen.contact import hang as hang_contact
from sotaigen.contact.axilla_cache import targets as shoulder_targets
from sotaigen.corrective import axilla
from sotaigen.corrective import hang as hang_driver
from sotaigen.corrective.crossings import SkinParts, edges_through_triangles
from sotaigen.mesh import opposite_vertices
from sotaigen.pipeline import rigged
from sotaigen.rig.skinning import pose_verts

BODIES = ("female-chibi", "female-long", "female-balanced")
ARM = {"upperArm", "lowerArm", "hand"}


def delivered(name):
    rig = rigged(name)
    hang, table = hang_contact.targets(rig.base, rig.verts, rig.bones, rig.weights)
    return rig, shoulder_targets(rig.base, rig.verts, rig.bones, rig.weights), hang, table


def arm_crossings(rig, shoulder, hang, table, pose):
    parts = SkinParts(rig.base)
    bound = axilla.bind_verts(rig.verts, shoulder, pose)
    bound = hang_driver.bind_verts(bound, hang, pose, table)
    posed = pose_verts(rig.bones, rig.weights, bound, pose)
    count = 0
    for edge, triangle in edges_through_triangles(posed, parts.edges, parts.triangles).tolist():
        pair = {str(parts.groups[parts.edges[edge, 0]]), str(parts.groups[parts.triangles[triangle, 0]])}
        if pair & ARM and pair - {"hand"}:
            count += 1
    return count


@pytest.mark.bake
@pytest.mark.parametrize("name", BODIES)
def test_both_arms_hang_to_the_target_without_crossing(name):
    rig, shoulder, hang, table = delivered(name)
    assert table is not None, hang_contact.failure(rig.base, rig.verts, rig.bones, rig.weights)
    bottom = table["adductionKeys"][-1]
    assert hang_driver.rest_tilt(rig.bones) - bottom == pytest.approx(hang_driver.FROM_VERTICAL, abs=0.01)
    for forward, adduction in ((0, bottom), (7.5, bottom), (12.5, (bottom + 24) / 2), (45, bottom), (0, 27)):
        pose = {"upperArm": (forward, 0.0, -adduction)}
        pose = {"leftUpperArm": pose["upperArm"], "rightUpperArm": (forward, 0.0, adduction)}
        assert arm_crossings(rig, shoulder, hang, table, pose) == 0, (forward, adduction)


@pytest.mark.bake
@pytest.mark.parametrize("name", BODIES)
def test_the_audit_read_both_axes_and_passed(name):
    rig = rigged(name)
    key = hang_contact.fingerprint(rig.base, rig.verts, rig.bones, rig.weights)
    report = json.loads((hang_contact.CACHE / key / "report.json").read_text())
    assert report["passed"]
    for row in report["sides"].values():
        assert not row["crossedPoses"] and not row["unsafeIntervals"] and not row["wholeBodyCrossed"]
    assert report["mirrored"] or set(report["sides"]) == {"left", "right"}


@pytest.mark.bake
def test_the_table_waits_for_the_shoulder_and_leaves_the_standing_body_alone():
    rig, _, hang, table = delivered("female-long")
    assert hang_driver.bind_verts(rig.verts, hang, {}, table) is rig.verts
    for pose in ({"upperArm": (0.0, 0.0, -24.0)}, {"upperArm": (0.0, 0.0, -10.0)}, {"upperArm": (90.0, 0.0, 0.0)}):
        assert not any(hang_driver.pose_weights(pose, table).values()), pose


@pytest.mark.bake
def test_a_bent_elbow_a_turned_hand_or_a_flexed_hip_fades_it():
    _, _, _, table = delivered("female-long")
    hang = {"upperArm": (0.0, 0.0, -table["adductionKeys"][-1])}
    assert max(hang_driver.pose_weights(hang, table).values()) == pytest.approx(1.0)
    for extra in ({"lowerArm": (30.0, 0, 0)}, {"hand": (0, 0, 30.0)}, {"upperLeg": (30.0, 0, 0)},
                  {"upperArm": (0.0, 90.0, -table["adductionKeys"][-1])}):
        assert not any(hang_driver.pose_weights(hang | extra, table).values()), extra
    one = hang_driver.pose_weights(hang | {"leftUpperLeg": (30.0, 0, 0)}, table)
    assert not any(v for k, v in one.items() if k.startswith("hangL"))
    assert any(v for k, v in one.items() if k.startswith("hangR"))


@pytest.mark.bake
def test_the_right_side_mirrors_the_left_on_a_symmetric_body():
    rig, _, hang, table = delivered("female-long")
    left, right = table["sides"]["left"], table["sides"]["right"]
    assert left["samples"] == right["samples"]
    opposite = opposite_vertices(rig.verts)
    for a, b in zip(left["targets"], right["targets"], strict=True):
        np.testing.assert_allclose(hang[b][opposite], hang[a] * hang_driver.FLIP, atol=1e-9)


def test_a_failed_bake_leaves_the_arm_without_it_and_says_why(tmp_path, monkeypatch, capsys):
    rig = rigged("female-long")
    monkeypatch.setattr(hang_contact, "CACHE", tmp_path)
    calls = []

    def refuse(*args):
        calls.append(args)
        raise RuntimeError("the audit said no")

    monkeypatch.setattr(hang_contact, "bake", refuse)
    for _ in range(2):
        assert hang_contact.targets(rig.base, rig.verts, rig.bones, rig.weights) == ({}, None)
    assert len(calls) == 1  # The record stands until someone deletes it: the same input fails the same way.
    assert hang_contact.failure(rig.base, rig.verts, rig.bones, rig.weights) == "the audit said no"
    assert "the audit said no" in capsys.readouterr().err
