"""Incomplete bakes must never become cached production shapes."""

import json

import numpy as np
import pytest

from sotaigen.body import Body
from sotaigen.contact import axilla_cache
from sotaigen.corrective.pose_grid import PoseGrid


def inputs():
    base = {"verts": [[0., 0., 0.], [1., 0., 0.]], "faces": [], "segments": ["trunk", "trunk"]}
    return base, np.array(base["verts"]), [], np.ones((2, 1))


def fake_bake(base, verts, bones, weights, output, *, side, input_hashes):
    output.mkdir(parents=True)
    grid = PoseGrid.rectilinear([[0, 95], [-24, 0, 110]])
    np.savez(output / "table.npz", points=grid.points, cells=grid.cells,
             ids=[0 if side == "left" else 1], deltas=np.zeros((len(grid.points), 1, 3)),
             initialComplete=np.ones(len(grid.points), bool))
    (output / "report.json").write_text(json.dumps(
        {"auditPassed": True,
         "rounds": [{"stepDegrees": 1.25, "failed": [], "unsafeIntervals": 0}]}))


def test_successful_bake_is_reused_but_changed_geometry_is_not(tmp_path, monkeypatch):
    from sotaigen.contact import axilla_bake

    monkeypatch.setattr(axilla_cache, "CACHE", tmp_path)
    calls = []
    def bake(*args, **kwargs):
        calls.append(kwargs["side"])
        fake_bake(*args, **kwargs)
    monkeypatch.setattr(axilla_bake, "bake_side", bake)
    base, verts, bones, weights = inputs()
    first = axilla_cache.targets(base, verts, bones, weights)
    second = axilla_cache.targets(base, verts, bones, weights)
    assert calls == ["left", "right"]
    assert first.driver == second.driver
    axilla_cache.targets(base, verts + 1, bones, weights)
    assert calls == ["left", "right", "left", "right"]
    assert axilla_cache.fingerprint(base, verts, bones, weights) != axilla_cache.fingerprint(
        base, verts, bones, weights * .9)


LEGACY = {"axillaLeft20": np.zeros((2, 3))}


def test_failed_side_falls_back_to_legacy_and_says_why(tmp_path, monkeypatch, capsys):
    from sotaigen.contact import axilla_bake

    monkeypatch.setattr(axilla_cache, "CACHE", tmp_path)
    monkeypatch.setattr(axilla_cache.axilla, "targets", lambda *args: LEGACY)
    calls = []
    def fail(*args, **kwargs):
        calls.append(kwargs["side"])
        if kwargs["side"] == "right":
            raise RuntimeError("contact audit failed")
        fake_bake(*args, **kwargs)
    monkeypatch.setattr(axilla_bake, "bake_side", fail)
    args = inputs()
    key = axilla_cache.fingerprint(*args)
    assert axilla_cache.failure(*args) is None
    assert axilla_cache.targets(*args) is LEGACY
    assert "contact audit failed" in capsys.readouterr().err
    assert axilla_cache.failure(*args) == "contact audit failed"
    assert not (tmp_path / key).exists()
    assert list(tmp_path.glob("*.failed/left/report.json"))
    # The same inputs would fail the same way, so they are not baked again.
    assert axilla_cache.targets(*args) is LEGACY
    assert calls == ["left", "right"]
    # Deleting the record, as with anything in this folder, asks for another try.
    (tmp_path / f"{key}.failed.json").unlink()
    monkeypatch.setattr(axilla_bake, "bake_side", fake_bake)
    assert hasattr(axilla_cache.targets(*args), "driver")
    assert (tmp_path / key / "ready.json").exists()
    assert axilla_cache.failure(*args) is None


@pytest.mark.parametrize("error", [KeyboardInterrupt, TypeError])
def test_an_interrupted_or_broken_bake_is_not_recorded_as_this_bodys_failure(
    tmp_path, monkeypatch, error
):
    from sotaigen.contact import axilla_bake

    monkeypatch.setattr(axilla_cache, "CACHE", tmp_path)
    def fail(*args, **kwargs):
        raise error("not about this body")
    monkeypatch.setattr(axilla_bake, "bake_side", fail)
    args = inputs()
    with pytest.raises(error):
        axilla_cache.targets(*args)
    assert axilla_cache.failure(*args) is None


def test_a_failed_bake_still_exports_a_consistent_legacy_glb(tmp_path, monkeypatch, capsys):
    from sotaigen.contact import axilla_bake
    from sotaigen.corrective.recipe import Correctives
    from sotaigen.output.glb import glb_bytes
    from sotaigen.pipeline import rigged
    from sotaigen.uv import export_fields

    monkeypatch.setattr(axilla_cache, "CACHE", tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError("contact audit failed")
    monkeypatch.setattr(axilla_bake, "bake_side", fail)
    rig = rigged("female-chibi")
    base = {**rig.base, **export_fields(rig.base["faces"])}
    shoulder = axilla_cache.targets(base, rig.verts, rig.bones, rig.weights)
    blob = glb_bytes(base, rig.verts, rig.bones, rig.weights,
                     correctives=Correctives(shoulder))
    length = int.from_bytes(blob[12:16], "little")
    document = json.loads(blob[20:20 + length])
    groups = document["extras"]["sotaigen"]["correctives"]["groups"]
    # The sculpted grid it fell back to, not the simplex table of a bake.
    assert [group["name"] for group in groups] == ["axilla.left", "axilla.right"]
    assert not any(group.get("kind") for group in groups)
    names = document["meshes"][0]["extras"]["targetNames"]
    assert [*groups[0]["targets"], *groups[1]["targets"]] == names
    assert "contact audit failed" in capsys.readouterr().err


def test_cache_refuses_a_failed_or_coarser_audit(tmp_path):
    key = "test"
    (tmp_path / "ready.json").write_text(json.dumps(
        {"fingerprint": key, "scope": "twist-zero", "auditStepDegrees": 1.25}))
    args = inputs()
    for side in ("left", "right"):
        fake_bake(*args, tmp_path / side, side=side, input_hashes={})
    report = tmp_path / "left/report.json"
    data = json.loads(report.read_text())
    data["rounds"][-1]["stepDegrees"] = 5
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Incomplete"):
        axilla_cache.load_samples(tmp_path, key)


def test_a_reported_pass_cannot_publish_malformed_morph_data(tmp_path, monkeypatch):
    from sotaigen.contact import axilla_bake

    monkeypatch.setattr(axilla_cache, "CACHE", tmp_path)
    def malformed(*args, **kwargs):
        fake_bake(*args, **kwargs)
        path = args[4] / "table.npz"
        with np.load(path) as data:
            fields = {name: data[name] for name in data.files}
        fields["deltas"][(fields["points"] == 0).all(axis=1)] = 1
        np.savez(path, **fields)
    monkeypatch.setattr(axilla_bake, "bake_side", malformed)
    monkeypatch.setattr(axilla_cache.axilla, "targets", lambda *args: LEGACY)
    args = inputs()
    key = axilla_cache.fingerprint(*args)
    assert axilla_cache.targets(*args) is LEGACY
    assert "moves the rest" in axilla_cache.failure(*args)
    assert not (tmp_path / key).exists()


def test_the_editor_reports_a_failed_bake_among_its_failures(tmp_path, monkeypatch):
    from sotaigen import generation as generate

    monkeypatch.setattr(axilla_cache, "CACHE", tmp_path)
    base = generate.full_template()
    verts, weights = np.asarray(base["verts"]), np.ones((len(base["verts"]), 1))
    exact = Body(verts, [], weights, {}, {})
    assert generate.contact_failures(exact) == []
    key = axilla_cache.fingerprint(base, verts, [], weights)
    (tmp_path / f"{key}.failed.json").write_text(json.dumps({"error": "contact audit failed"}))
    assert generate.contact_failures(exact) == ["shoulder contact bake failed: contact audit failed"]


@pytest.fixture
def symmetric_body():
    from sotaigen.pipeline import rigged

    rig = rigged("female-chibi")
    return rig.base, np.array(rig.verts), rig.bones, np.array(rig.weights)


def left_table(base, verts, bones, weights, output):
    """A left table of a real patch, with made-up shapes: only the mirroring is under test."""
    from sotaigen.contact.axilla import ShoulderSampler

    output.mkdir(parents=True)
    ids = ShoulderSampler(base, verts, bones, weights, "left").ids
    grid = PoseGrid.rectilinear([[0, 95], [-24, 0, 110]])
    deltas = np.random.default_rng(1).normal(size=(len(grid.points), len(ids), 3))
    deltas[(grid.points == 0).all(axis=1)] = 0
    np.savez(output / "table.npz", points=grid.points, cells=grid.cells, ids=ids, deltas=deltas,
             initialComplete=np.ones(len(grid.points), bool))
    (output / "report.json").write_text(json.dumps(
        {"auditPassed": True, "side": "left",
         "rounds": [{"stepDegrees": 1.25, "failed": [], "unsafeIntervals": 0}]}))


def test_a_symmetric_bodys_right_table_is_the_mirror_of_its_left(tmp_path, symmetric_body):
    from sotaigen.contact.axilla import ShoulderSampler
    from sotaigen.contact.axilla_bake import mirror_side
    from sotaigen.mesh import opposite_vertices

    left, right = tmp_path / "left", tmp_path / "right"
    left_table(*symmetric_body, left)
    assert mirror_side(*symmetric_body, left, right)
    report = json.loads((right / "report.json").read_text())
    assert report["side"] == "right" and report["mirroredFromLeft"]
    a, b = np.load(left / "table.npz"), np.load(right / "table.npz")
    verts = symmetric_body[1]
    # Moved by the table, each vertex lands on the mirror image of its opposite number.
    moved = np.array(verts)
    moved[b["ids"]] += b["deltas"][-1]
    other = np.array(verts)
    other[a["ids"]] += a["deltas"][-1]
    opposite = opposite_vertices(verts)
    assert np.abs(moved[b["ids"]] - other[opposite[b["ids"]]] * [-1, 1, 1]).max() < 1e-5
    assert np.array_equal(b["ids"], ShoulderSampler(*symmetric_body, "right").ids)


@pytest.mark.parametrize("segment", ["trunk", "foot"])
def test_a_body_that_is_not_symmetric_gets_no_mirrored_table(tmp_path, symmetric_body, segment):
    from sotaigen.contact.axilla_bake import mirror_side

    base, verts, bones, weights = symmetric_body
    left, right = tmp_path / "left", tmp_path / "right"
    left_table(*symmetric_body, left)
    # Anywhere on the body, and by much less than anyone could see.
    verts[base["segments"].index(segment)] += [1e-3, 0, 0]
    assert not mirror_side(base, verts, bones, weights, left, right)
    assert not right.exists()


def test_a_skin_that_does_not_mirror_where_the_arm_moves_gets_no_mirrored_table(tmp_path, symmetric_body):
    from sotaigen.contact.axilla_bake import mirror_side

    base, verts, bones, weights = symmetric_body
    left, right = tmp_path / "left", tmp_path / "right"
    left_table(*symmetric_body, left)
    table = np.load(left / "table.npz")
    names = [bone["name"] for bone in bones]
    moved = np.flatnonzero(weights[table["ids"], names.index("leftUpperArm")] > .5)[0]
    weights[table["ids"][moved], names.index("leftUpperArm")] -= .01
    assert not mirror_side(base, verts, bones, weights, left, right)
    assert not right.exists()


def test_only_the_left_side_is_baked_on_a_symmetric_body(tmp_path, monkeypatch, symmetric_body):
    from sotaigen.contact import axilla_bake

    monkeypatch.setattr(axilla_cache, "CACHE", tmp_path)
    calls = []
    def bake(base, verts, bones, weights, output, *, side, input_hashes):
        calls.append(side)
        left_table(base, verts, bones, weights, output)
    monkeypatch.setattr(axilla_bake, "bake_side", bake)
    assert hasattr(axilla_cache.targets(*symmetric_body), "driver")
    assert calls == ["left"]
    key = axilla_cache.fingerprint(*symmetric_body)
    assert json.loads((tmp_path / key / "right" / "report.json").read_text())["mirroredFromLeft"]
