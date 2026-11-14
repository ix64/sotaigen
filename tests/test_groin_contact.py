"""The contact table keeps the abdomen and thigh root apart, and only there."""

import json

import numpy as np
import pytest

from sotaigen.contact import groin as groin_contact
from sotaigen.corrective import groin
from sotaigen.corrective.crossings import SkinParts, edges_through_triangles
from sotaigen.corrective.recipe import Correctives
from sotaigen.form.bodies import load_bodies
from sotaigen.pipeline import rigged
from sotaigen.rig.skinning import pose_verts

BODIES = ("female-chibi", "female-long", "female-balanced")


def delivered(name):
    rig = rigged(name)
    return rig, groin_contact.targets(rig.base, rig.verts, rig.bones, rig.weights)


@pytest.mark.bake
@pytest.mark.parametrize("name", BODIES)
def test_the_skin_stays_out_of_itself_between_keys_and_at_uneven_legs(name):
    # Only the midline pocket may cross, as the audit allows and reports; the next test holds its limit.
    rig, targets = delivered(name)
    parts = SkinParts(rig.base)
    pocket = groin_contact.GroinSampler(rig.base, rig.verts, rig.bones, rig.weights).pocket()
    poses = [(70, 70), (85, 85), (100, 100), (115, 115), (120, 120), (100, 0), (0, 115), (35, 65), (45, 80), (70, 110)]
    for a, b in poses:
        pose = groin.sample_pose(a, b)
        posed = pose_verts(rig.bones, rig.weights, groin.bind_verts(rig.verts, targets, pose), pose)
        hits = [
            (edge, triangle)
            for edge, triangle in edges_through_triangles(posed, parts.edges, parts.triangles).tolist()
            if parts.label(parts.edges[edge, 0], parts.triangles[triangle, 0]) in groin_contact.WHOLE_PAIRS
            and not (pocket[parts.edges[edge]].all() and pocket[parts.triangles[triangle]].all())
        ]
        assert not hits, (a, b)


@pytest.mark.bake
@pytest.mark.parametrize("name", BODIES)
def test_the_audit_read_every_left_and_right_angle_and_passed(name):
    rig = rigged(name)
    key = groin_contact.fingerprint(rig.base, rig.verts, rig.bones, rig.weights)
    report = json.loads((groin_contact.CACHE / key / "report.json").read_text())
    angles = groin_contact.KEYS[-1] // groin_contact.AUDIT_STEP + 1
    assert report["poses"] == angles**2
    assert report["passed"] and not report["crossedPoses"] and not report["unsafeIntervals"]
    assert all(row[2] <= groin_contact.POCKET_EDGES for row in report["pocketCrossed"])


@pytest.mark.bake
def test_contact_moves_only_its_patch_and_never_the_standing_body():
    rig, targets = delivered("female-long")
    sculpted = groin.sculpted_targets(rig.base, rig.verts, rig.bones, rig.weights, groin_contact.KEYS)
    ids, _ = groin_contact.corrections(rig.base, rig.verts, rig.bones, rig.weights)
    outside = np.setdiff1d(np.arange(len(rig.verts)), ids)
    assert targets.keys() == sculpted.keys()
    for name in targets:
        np.testing.assert_array_equal(targets[name][outside], sculpted[name][outside])
    assert groin.bind_verts(rig.verts, targets, {}) is rig.verts


@pytest.mark.bake
def test_the_visible_shape_keeps_the_sculpture_until_the_skin_meets():
    # Near the midline the slot between the legs is narrow and may answer early; it doesn't show.
    rig, targets = delivered("female-long")
    sculpted = groin.sculpted_targets(rig.base, rig.verts, rig.bones, rig.weights, groin_contact.KEYS)
    away_from_slot = np.abs(rig.verts[:, 0]) > 0.12 * groin.landmarks(rig.bones)["T"]
    for name in ("pelvisL20R20", "pelvisL40R0", "pelvisL40R40"):
        moved = np.linalg.norm((targets[name] - sculpted[name])[away_from_slot], axis=1)
        assert moved.max() < 1.0, name


@pytest.mark.slow
@pytest.mark.parametrize("name", list(load_bodies()))
def test_the_patch_stays_below_the_loops_round_the_armpit(name):
    # Those loops lie inside the barrier at rest: in the patch, the table pushed them from the first
    # degree of hip flexion, 1-6 mm on every sample.
    rig = rigged(name)
    refined = rig.base["refined"]
    first = refined["from"] + refined["batches"][0]["count"]
    armpit = np.arange(first, first + refined["batches"][1]["count"])
    assert refined["batches"][1]["name"] == "axilla"
    ids = groin_contact.GroinSampler(rig.base, rig.verts, rig.bones, rig.weights).ids
    assert not np.isin(armpit, ids).any()


def test_the_contact_table_keeps_its_own_keys_and_is_driven_on_them():
    # The sculpture is keyed every 20 degrees; the contact table keeps 90 and 110 too, or the belly
    # passes through the thigh root between 100 and 120. A table is driven on the keys its targets name.
    assert groin.KEYS == (0, 20, 40, 60, 80, 100, 120)
    assert set(groin_contact.KEYS) == {*groin.KEYS, 90, 110}
    assert groin.keys_of(dict.fromkeys(groin.NAMES)) == groin.KEYS
    assert groin.keys_of(dict.fromkeys(groin_contact.NAMES)) == groin_contact.KEYS
    rig = rigged("female-long")
    targets = {name: np.zeros_like(rig.verts) for name in groin_contact.NAMES}
    targets["pelvisL90R0"] = np.ones_like(rig.verts)
    (group,) = Correctives(groin=targets).groups()
    assert [axis["keys"] for axis in group["axes"]] == [list(groin_contact.KEYS)] * 2
    assert group["targets"] == list(groin_contact.NAMES)
    moved = groin.bind_verts(rig.verts, targets, groin.sample_pose(90, 0)) - rig.verts
    np.testing.assert_allclose(moved, 1.0, atol=1e-12, rtol=0)


def test_a_sampler_runs_ipctk_on_one_thread():
    # With more threads ipctk sums the barrier over contacts in a varying order: the same bake twice
    # could give tables that differ in the last bits.
    import ipctk

    rig = rigged("female-long")
    ipctk.set_num_threads(4)
    sampler = groin_contact.GroinSampler(rig.base, rig.verts, rig.bones, rig.weights)
    assert ipctk.get_num_threads() == 1
    assert not ipctk.has_intersections(sampler.solver.mesh, sampler.solver.x)


@pytest.mark.bake
def test_mirroring_needs_a_symmetric_patch():
    rig = rigged("female-long")
    ids, _ = groin_contact.corrections(rig.base, rig.verts, rig.bones, rig.weights)
    twin = groin_contact.mirror_of(rig.base, rig.verts, ids)
    assert twin is not None
    np.testing.assert_array_equal(twin[twin], np.arange(len(ids)))
    uneven = rig.verts.copy()
    uneven[ids[0], 1] += 1.0
    assert groin_contact.mirror_of(rig.base, uneven, ids) is None


def test_a_table_crowded_only_in_the_pocket_fails_as_a_table_does(tmp_path, monkeypatch):
    # Raised as anything but RuntimeError, it would leave no failure record and stop the export.
    rig = rigged("female-long")
    report = {"crossedPoses": [], "unsafeIntervals": [], "collapsedPoses": [], "wholeBodyCrossed": [],
              "pocketCrossed": [[0, 40, 3], [70, 80, 18]], "passed": False}
    monkeypatch.setattr(groin_contact, "solve_table", lambda *args: {n: np.zeros((1, 3)) for n in groin_contact.NAMES})
    monkeypatch.setattr(groin_contact, "audit", lambda *args: dict(report))
    with pytest.raises(RuntimeError, match=r"\[70, 80, 18\]"):
        groin_contact.bake(rig.base, rig.verts, rig.bones, rig.weights, tmp_path)


def test_a_failed_bake_keeps_the_sculpture_and_says_why(tmp_path, monkeypatch, capsys):
    rig = rigged("female-long")
    monkeypatch.setattr(groin_contact, "CACHE", tmp_path)
    calls = []

    def refuse(*args):
        calls.append(args)
        raise RuntimeError("the audit said no")

    monkeypatch.setattr(groin_contact, "bake", refuse)
    sculpted = groin.sculpted_targets(rig.base, rig.verts, rig.bones, rig.weights)
    for _ in range(2):
        result = groin_contact.targets(rig.base, rig.verts, rig.bones, rig.weights)
        # Without a table it is the default correctives' sculpture, on the sculpture's own keys.
        assert result.keys() == sculpted.keys()
        for name in sculpted:
            np.testing.assert_array_equal(result[name], sculpted[name])
    assert len(calls) == 1  # The record stands until someone deletes it: the same input fails the same way.
    assert groin_contact.failure(rig.base, rig.verts, rig.bones, rig.weights) == "the audit said no"
    assert "the audit said no" in capsys.readouterr().err
