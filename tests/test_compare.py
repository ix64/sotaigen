"""sotaigen_dev.compare on made-up results: what it lists, and which exit code each case gets."""

import json

import pytest

from sotaigen import pipeline, sources
from sotaigen.pipeline import fingerprints, write_fingerprint
from sotaigen_dev import compare

NAME = "regular-soft"
WRAP = {
    "flippedFaces": 0,
    "crossedMidline": 0,
    "surfaceErrorMm": {"p95": 0.01, "max": 0.85},
}
POSES = {
    "rest": {"collapsedFaces": 0, "areaP1": 1.0},
    "crouch": {"collapsedFaces": 2, "areaP1": 0.2},
}
JOINTS = {
    "restCreases": 56,
    "poses": {"crouch": {"creases": 90, "collapsedBySegment": {"trunk": 2}}},
}


def results(preset_dir, wrap=WRAP, poses=POSES, joints=JOINTS):
    preset_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("wrap.json", wrap), ("poses.json", poses), ("joints.json", joints)):
        (preset_dir / name).write_text(json.dumps(data))


def edited(data, *path, value):
    """A deep copy of the data with one value replaced."""
    copy = json.loads(json.dumps(data))
    target = copy
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return copy


def vouch(preset):
    write_fingerprint(compare.BUILD / "bodies" / preset, {"metrics": fingerprints()["metrics"]})


@pytest.fixture
def sides(tmp_path, monkeypatch):
    """The two build directories, both holding the same fresh results for one body."""
    # This build in its bodies/, the other laid out as builds were before they had one.
    here, other = tmp_path / "build" / "bodies", tmp_path / "other"
    monkeypatch.setattr(compare, "BUILD", here.parent)
    results(here / NAME)
    results(other / NAME)
    vouch(NAME)
    return here, other


def run(other, *argv):
    return compare.main(["--against", str(other), *argv])


def test_identical_results_report_no_changes(sides, capsys):
    assert run(sides[1]) == 0
    printed = capsys.readouterr()
    assert printed.out == "0 changed metrics in 0 of 1 bodies\n"
    assert printed.err == ""


def test_more_collapsed_faces_is_a_regression_naming_preset_and_pose(sides, capsys):
    here, other = sides
    results(here / NAME, poses=edited(POSES, "crouch", "collapsedFaces", value=3))
    assert run(other) == 1
    out = capsys.readouterr().out
    assert "REGRESSION regular-soft / crouch: poses.json collapsedFaces 2 -> 3" in out


def test_legs_crossing_in_a_pose_is_a_regression(sides, capsys):
    here, other = sides
    joints = edited(JOINTS, "poses", "crouch", "legsPastMidline", value=0)
    results(other / NAME, joints=joints)
    results(here / NAME, joints=edited(joints, "poses", "crouch", "legsPastMidline", value=9))
    assert run(other) == 1
    assert "REGRESSION regular-soft / crouch: joints.json legsPastMidline 0 -> 9" in (
        capsys.readouterr().out
    )


@pytest.mark.parametrize("metric", ["flippedFaces", "crossedMidline"])
def test_a_worse_wrap_is_a_regression(sides, capsys, metric):
    here, other = sides
    results(here / NAME, wrap=edited(WRAP, metric, value=1))
    assert run(other) == 1
    assert f"REGRESSION regular-soft / (whole body): wrap.json {metric} 0 -> 1" in (
        capsys.readouterr().out
    )


def test_a_better_hard_metric_is_listed_but_is_no_regression(sides, capsys):
    here, other = sides
    results(here / NAME, poses=edited(POSES, "crouch", "collapsedFaces", value=1))
    assert run(other) == 0
    out = capsys.readouterr().out
    assert "collapsedFaces" in out and "2 -> 1" in out
    assert "REGRESSION" not in out


@pytest.mark.parametrize("value", [None, float("nan")])
def test_a_hard_metric_that_stops_being_a_number_is_a_regression(sides, value):
    here, other = sides
    results(here / NAME, poses=edited(POSES, "crouch", "collapsedFaces", value=value))
    assert run(other) == 1


def test_soft_changes_exit_0_and_are_listed_by_pose(sides, capsys):
    here, other = sides
    results(
        here / NAME,
        wrap=edited(WRAP, "surfaceErrorMm", "max", value=9.0),
        poses=edited(POSES, "crouch", "areaP1", value=0.1),
        joints=edited(JOINTS, "poses", "crouch", "collapsedBySegment", "trunk", value=5),
    )
    assert run(other) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == NAME
    text = "\n".join(lines)
    assert "surfaceErrorMm.max" in text and "0.85 -> 9.0" in text
    assert "areaP1" in text and "0.2 -> 0.1" in text
    assert "collapsedBySegment.trunk" in text and "2 -> 5" in text
    assert "  crouch\n" in text and "  (whole body)\n" in text
    assert "REGRESSION" not in text
    assert "3 changed metrics in 1 of 1 bodies" in text


def test_only_changes_are_listed_unless_all_is_asked_for(sides, capsys):
    here, other = sides
    results(here / NAME, poses=edited(POSES, "crouch", "areaP1", value=0.1))
    run(other)
    out = capsys.readouterr().out
    assert "areaP1" in out and "creases" not in out and "  rest\n" not in out
    run(other, "--all")
    out = capsys.readouterr().out
    assert "creases" in out and "  rest\n" in out


@pytest.mark.parametrize("file", compare.FILES)
@pytest.mark.parametrize("missing_here", [True, False])
def test_a_missing_file_is_an_error_naming_it(sides, capsys, file, missing_here):
    here, other = sides
    gone = (here if missing_here else other) / NAME / file
    gone.unlink()
    assert run(other) == 2
    err = capsys.readouterr().err
    assert str(gone) in err and "missing" in err


def test_a_preset_on_one_side_only_is_an_error_and_naming_the_others_avoids_it(sides, capsys):
    here, other = sides
    results(here / "medium-soft")
    vouch("medium-soft")
    assert run(other) == 2
    assert str(other / "medium-soft" / "wrap.json") in capsys.readouterr().err
    assert run(other, NAME) == 0


def test_only_the_named_presets_are_compared(sides, capsys):
    here, other = sides
    for side in (here, other):
        results(side / "petite-soft")
    results(here / "petite-soft", wrap=edited(WRAP, "flippedFaces", value=4))
    vouch("petite-soft")
    assert run(other, NAME) == 0
    assert run(other, "petite-soft") == 1


def test_an_unknown_name_is_an_error(sides):
    assert run(sides[1], "nope") == 2


def test_a_regression_still_wins_over_a_missing_file(sides, capsys):
    here, other = sides
    results(here / NAME, poses=edited(POSES, "crouch", "collapsedFaces", value=3))
    (other / NAME / "joints.json").unlink()
    assert run(other) == 1
    printed = capsys.readouterr()
    assert "REGRESSION" in printed.out and "missing" in printed.err


def test_a_hard_metric_gone_from_a_pose_both_sides_have_is_an_error(sides, capsys):
    here, other = sides
    poses = edited(POSES, "crouch", "areaP1", value=0.2)
    del poses["crouch"]["collapsedFaces"]
    results(here / NAME, poses=poses)
    assert run(other) == 2
    assert "regular-soft / crouch: poses.json of this build has no collapsedFaces" in (
        capsys.readouterr().err
    )


def test_a_changed_mesh_with_the_same_metrics_is_listed_but_is_no_regression(sides, capsys):
    (sides[0] / NAME / "wrapped.npy").write_bytes(b"after")
    (sides[1] / NAME / "wrapped.npy").write_bytes(b"before")
    assert run(sides[1]) == 0
    printed = capsys.readouterr().out
    assert "wrapped.npy" in printed and "1 changed metrics in 1 of 1 bodies" in printed


def test_the_same_mesh_is_not_listed(sides, capsys):
    (sides[0] / NAME / "wrapped.npy").write_bytes(b"same")
    (sides[1] / NAME / "wrapped.npy").write_bytes(b"same")
    assert run(sides[1]) == 0
    assert capsys.readouterr().out == "0 changed metrics in 0 of 1 bodies\n"


def test_a_new_soft_metric_is_listed_and_passes(sides, capsys):
    here, other = sides
    results(here / NAME, poses=edited(POSES, "crouch", "shoulderDrop", value=4))
    assert run(other) == 0
    assert "shoulderDrop" in capsys.readouterr().out


def test_a_pose_added_is_listed_with_its_hard_metrics_and_passes(sides, capsys):
    here, other = sides
    results(here / NAME, poses={**POSES, "wave": {"collapsedFaces": 7, "areaP1": 0.5}})
    assert run(other) == 0
    out = capsys.readouterr().out
    assert "  wave\n" in out and "absent -> present" in out and "absent -> 7" in out
    assert "areaP1" not in out


def test_a_pose_removed_is_listed_and_passes(sides, capsys):
    here, other = sides
    results(here / NAME, poses={"rest": POSES["rest"]})
    assert run(other) == 0
    out = capsys.readouterr().out
    assert "  crouch\n" in out and "present -> absent" in out and "2 -> absent" in out


def test_stale_results_here_are_refused_with_the_command_to_fix_them(sides, capsys):
    here, other = sides
    write_fingerprint(here / NAME, {"metrics": "from an older source"})
    assert run(other) == 2
    err = capsys.readouterr().err
    assert "out of date" in err and "uv run sotaigen-dev build regular-soft --stages metrics" in err
    assert run(other, "--allow-stale") == 0
    assert "out-of-date" in capsys.readouterr().err


def test_the_other_side_may_be_older(sides):
    write_fingerprint(sides[1] / NAME, {"metrics": "from an older source"})
    assert run(sides[1]) == 0


def test_a_broken_result_file_is_an_error_naming_it(sides, capsys):
    other = sides[1]
    (other / NAME / "poses.json").write_text("half of a fi")
    assert run(other) == 2
    assert str(other / NAME / "poses.json") in capsys.readouterr().err


def test_folders_without_results_are_not_presets(sides, capsys):
    here, other = sides
    (here / "review" / "cache").mkdir(parents=True)
    assert run(other) == 0
    assert "in 0 of 1 bodies" in capsys.readouterr().out


@pytest.mark.parametrize("target", ["absent", "itself"])
def test_a_useless_other_side_is_refused(sides, capsys, target):
    here, other = sides
    with pytest.raises(SystemExit) as stop:
        run(here if target == "itself" else other / "absent")
    assert stop.value.code == 2
    assert "--against" in capsys.readouterr().err


def test_nothing_on_either_side_is_an_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(compare, "BUILD", tmp_path / "build")
    (tmp_path / "build").mkdir()
    (tmp_path / "other").mkdir()
    assert run(tmp_path / "other") == 2
    assert "no results" in capsys.readouterr().err


def test_a_change_to_the_tool_rebuilds_nothing():
    for stage in pipeline.STAGES:
        assert not any(key.startswith("sotaigen_dev/compare.py:") for key in sources.reached(pipeline.STAGE_CODE[stage]))
