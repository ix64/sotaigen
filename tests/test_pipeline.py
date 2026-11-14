"""sotaigen.pipeline: fingerprints and cascades on a stand-in tree, and the real results' layout."""

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from sotaigen import pipeline, sources
from sotaigen.form.head import extended
from sotaigen.output.blender import find_blender
from sotaigen.pipeline import BUILT, CACHED, SKIPPED, STAGES, ensure_built, main, select
from sotaigen.rig.poses import POSES
from sotaigen.template import load_template

# A stand-in for data/: the logic under test must hold whatever bodies the project defines.
CATALOGUE = {f"{b}-{s}": {} for b in ("petite", "regular", "tall") for s in ("soft", "slender")}
NAME = "regular-soft"
# What the tests of real results build. Any body will do, so none is named.
REAL = next(iter(pipeline.bodies()))
ALL = ("wrap", "rig", "pose", "metrics")
INPUTS = (
    "sotaigen/anatomy.py",
    "sotaigen/wrap.py",
    "sotaigen/template.py",
    "sotaigen/rig.py",
    "sotaigen/poses.py",
    "sotaigen/metrics.py",
    "sotaigen/pipeline.py",
    "sotaigen/data/bodies/female/typical/long.json",
    "sotaigen/data/template/base-mesh.json",
)
# The stand-in modules call each other as the real ones do, since a stage's code is what its
# function in pipeline.py can reach.
CODE = {
    "sotaigen/anatomy.py": "def calibrated():\n    return 1\n",
    "sotaigen/template.py": "def load_template():\n    return 1\n",
    "sotaigen/wrap.py": "from .anatomy import calibrated\nfrom .template import load_template\n\n\n"
                    "def wrap_body():\n    return calibrated() + load_template()\n",
    "sotaigen/rig.py": "def skeleton():\n    return 1\n",
    "sotaigen/poses.py": "def check_poses():\n    return 1\n",
    "sotaigen/metrics.py": "def pose_row():\n    return 1\n",
    "sotaigen/pipeline.py": "from . import metrics, poses, rig, wrap\n\n\n"
                     "def run_wrap(name, out):\n    return wrap.wrap_body()\n\n\n"
                     "def run_rig(name, out):\n    return rig.skeleton()\n\n\n"
                     "def run_pose(name, out):\n    return poses.check_poses()\n\n\n"
                     "def run_metrics(name, out):\n    return metrics.pose_row() + poses.check_poses()\n",
}
# What the summary at the end of a command reads; the stand-in's other results are empty files.
REPORTS = {
    "wrap.json": '{"flippedFaces": 0, "crossedMidline": 0, "footLowest": 0.1}',
    "poses.json": "{}",
}


class Tree:
    """A copy of the inputs' layout, whose stages note that they ran and write results that change
    with their own inputs, unless `same` names the stage: then they always write the same."""

    def __init__(self, root):
        self.root = root
        self.calls = []
        self.same = set()

    def run(self, stage):
        def run_stage(name, out):
            self.calls.append(stage)
            result = "same" if stage in self.same else pipeline.stage_keys()[stage]
            for path in pipeline.artifacts(stage, out):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(REPORTS.get(path.name, result))

        return run_stage

    def edit(self, path, extra=b"\n"):
        with open(self.root / path, "ab") as f:
            f.write(extra)

    def change(self, path, old="return 1", new="return 1 + 1"):
        """Change what a definition does, rather than how the file reads."""
        text = (self.root / path).read_text()
        assert old in text
        (self.root / path).write_text(text.replace(old, new, 1))

    def build(self, stages=ALL, **options):
        """Build, and return the stages that ran and the report."""
        self.calls.clear()
        report = ensure_built(NAME, stages, **options)
        return list(self.calls), report

    @property
    def out(self):
        return self.root / "build" / "bodies" / NAME


@pytest.fixture
def catalogue(monkeypatch):
    monkeypatch.setattr(pipeline, "bodies", lambda: CATALOGUE)
    return CATALOGUE


@pytest.fixture
def tree(tmp_path, monkeypatch, catalogue):
    for path in INPUTS:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(f"# {path}\n{CODE.get(path, '')}")
    stand_in = Tree(tmp_path)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "BUILD", tmp_path / "build")
    monkeypatch.setattr(pipeline, "RUN", {stage: stand_in.run(stage) for stage in STAGES})
    return stand_in


def statuses(report):
    return {stage: row.status for stage, row in report.items()}


def vouched(out):
    """The stages whose results a body's fingerprint vouches for."""
    return set(pipeline.read_fingerprint(out)) & set(STAGES)


def test_the_first_build_runs_every_stage_and_the_second_none(tree):
    ran, report = tree.build()
    assert ran == list(ALL)
    assert [report[stage].status for stage in ALL] == [BUILT] * 4

    ran, report = tree.build()
    assert ran == []
    assert [report[stage].status for stage in ALL] == [CACHED] * 4


def test_a_changed_byte_in_the_data_redoes_everything(tree):
    tree.build()
    tree.edit("sotaigen/data/bodies/female/typical/long.json")
    assert tree.build()[0] == list(ALL)


def test_undoing_a_change_redoes_it_again_rather_than_trusting_older_files(tree):
    tree.build()
    before = (tree.root / "sotaigen/data/bodies/female/typical/long.json").read_bytes()
    tree.edit("sotaigen/data/bodies/female/typical/long.json")
    tree.build()
    (tree.root / "sotaigen/data/bodies/female/typical/long.json").write_bytes(before)
    assert tree.build()[0] == list(ALL)


@pytest.mark.parametrize(
    ("path", "old", "redone"),
    [
        ("sotaigen/anatomy.py", "return 1", ALL),
        ("sotaigen/wrap.py", "load_template()\n", ALL),
        ("sotaigen/data/template/base-mesh.json", None, ALL),
        ("sotaigen/rig.py", "return 1", ("rig", "pose", "metrics")),
        ("sotaigen/pipeline.py", "rig.skeleton()", ("rig", "pose", "metrics")),
        ("sotaigen/poses.py", "return 1", ("pose", "metrics")),
        ("sotaigen/metrics.py", "return 1", ("metrics",)),
    ],
)
def test_a_change_redoes_its_stage_and_what_reads_its_results(tree, path, old, redone):
    tree.build()
    if old is None:
        tree.edit(path)
    else:
        tree.change(path, old, old.rstrip("\n") + " + 1\n" if old.endswith("\n") else old + " + 1")
    assert tree.build()[0] == list(redone)


def test_a_touched_file_with_the_same_content_redoes_nothing(tree):
    tree.build()
    os.utime(tree.root / "sotaigen/data/bodies/female/typical/long.json", (1, 1))
    assert tree.build()[0] == []


@pytest.mark.parametrize(
    "edit",
    [
        b"\n# A comment.\n",
        b"\n\n\n",
        b'\n\ndef unused():\n    """Nothing calls this."""\n    return 2\n',
        b"\nUNUSED = 3\n",
        b"\nif __name__ == '__main__':\n    print(calibrated())\n",
    ],
)
def test_what_no_stage_can_run_redoes_nothing(tree, edit):
    tree.build()
    tree.edit("sotaigen/anatomy.py", edit)
    assert tree.build()[0] == []


def test_a_docstring_or_new_layout_redoes_nothing(tree):
    tree.build()
    (tree.root / "sotaigen/anatomy.py").write_text('"""Bodies."""\n\n\ndef calibrated():\n    """One."""\n\n    return (1)\n')
    assert tree.build()[0] == []


def test_code_a_module_runs_on_import_counts_wherever_the_module_is_imported(tree):
    tree.build()
    tree.edit("sotaigen/anatomy.py", b"\nprint('loaded')\n")
    assert tree.build()[0] == list(ALL)


def test_a_new_module_counts_once_a_stage_reaches_it(tree):
    tree.build()
    (tree.root / "sotaigen" / "new.py").write_text("def x():\n    return 1\n")
    assert tree.build()[0] == []
    tree.change("sotaigen/metrics.py", "return 1", "from .new import x\n    return x()")
    assert tree.build()[0] == ["metrics"]
    tree.change("sotaigen/new.py")
    assert tree.build()[0] == ["metrics"]


def test_a_module_in_a_subpackage_counts_where_it_is_reached(tree):
    (tree.root / "sotaigen" / "sub").mkdir()
    (tree.root / "sotaigen" / "sub" / "__init__.py").write_text("LIFT = 0.0\n")
    tree.change("sotaigen/wrap.py", "from .template", "from .sub import LIFT\nfrom .template")
    tree.change("sotaigen/wrap.py", "load_template()\n", "load_template() + LIFT\n")
    assert tree.build()[0] == list(ALL)
    assert tree.build()[0] == []
    (tree.root / "sotaigen" / "sub" / "__init__.py").write_text("LIFT = 50.0\n")
    assert tree.build()[0] == list(ALL)


def test_a_module_only_the_rig_reaches_counts_for_the_rig(tree):
    (tree.root / "sotaigen" / "late.py").write_text("def x():\n    return 1\n")
    tree.change("sotaigen/pipeline.py", "rig.skeleton()", "rig.skeleton() + late.x()")
    tree.change("sotaigen/pipeline.py", "from . import metrics", "from . import late, metrics")
    tree.build()
    tree.change("sotaigen/late.py")
    assert tree.build()[0] == ["rig", "pose", "metrics"]


def test_a_module_the_wrap_also_reaches_counts_for_the_wrap(tree):
    (tree.root / "sotaigen" / "shared.py").write_text("def x():\n    return 1\n")
    tree.change("sotaigen/wrap.py", "load_template()\n", "load_template() + x()\n")
    tree.change("sotaigen/wrap.py", "from .template", "from .shared import x\nfrom .template")
    tree.build()
    tree.change("sotaigen/shared.py")
    assert tree.build()[0] == list(ALL)


def test_a_namesake_in_a_subpackage_is_a_module_of_its_own(tree):
    (tree.root / "sotaigen" / "sub").mkdir()
    (tree.root / "sotaigen" / "sub" / "__init__.py").write_text("")
    (tree.root / "sotaigen" / "sub" / "rig.py").write_text("def x():\n    return 1\n")
    tree.change("sotaigen/wrap.py", "from .template", "from .sub.rig import x\nfrom .template")
    tree.change("sotaigen/wrap.py", "load_template()\n", "load_template() + x()\n")
    tree.build()
    tree.change("sotaigen/sub/rig.py")
    assert tree.build()[0] == list(ALL)
    tree.change("sotaigen/rig.py")
    assert tree.build()[0] == ["rig", "pose", "metrics"]


def test_a_stage_redone_with_the_same_results_leaves_those_below_alone(tree):
    tree.same.add("wrap")
    tree.build()
    tree.change("sotaigen/anatomy.py")
    ran, report = tree.build()
    assert ran == ["wrap"]
    assert [report[stage].status for stage in ALL] == [BUILT, CACHED, CACHED, CACHED]
    assert vouched(tree.out) == set(ALL)
    assert tree.build()[0] == []


def test_results_left_below_a_redone_stage_are_vouched_for_again_only_if_they_still_hold(tree):
    tree.same.add("wrap")
    tree.build()
    tree.change("sotaigen/anatomy.py")
    assert tree.build(("wrap",))[0] == ["wrap"]
    # Until the stages below are asked for, nothing vouches for what they left.
    assert vouched(tree.out) == {"wrap"}
    assert (tree.out / "weights.npz").exists()
    assert tree.build()[0] == []
    assert vouched(tree.out) == set(ALL)

    tree.same.clear()
    tree.change("sotaigen/anatomy.py")
    assert tree.build(("wrap",))[0] == ["wrap"]
    assert tree.build()[0] == ["rig", "pose", "metrics"]


def test_results_whose_own_code_changed_go_with_the_stage_above(tree):
    tree.same.add("wrap")
    tree.build()
    tree.change("sotaigen/anatomy.py")
    tree.change("sotaigen/rig.py")
    tree.build(("wrap",))
    assert not (tree.out / "weights.npz").exists()
    assert (tree.out / "poses").exists()


@pytest.mark.parametrize(
    "path",
    [
        "sotaigen/__pycache__/anatomy.py",
        "sotaigen/sub/__pycache__/x.py",
        "sotaigen/.hidden.py",
        "template/__pycache__/extract.cpython-314.pyc",
        "template/.DS_Store",
    ],
)
def test_caches_and_hidden_files_do_not_count(tree, path):
    tree.build()
    (tree.root / path).parent.mkdir(parents=True, exist_ok=True)
    (tree.root / path).write_text("x")
    assert tree.build()[0] == []


@pytest.mark.parametrize("path", ["sotaigen/data/template/uv.json", "sotaigen/data/template/deep/uv.bin"])
def test_a_data_file_in_the_template_counts_for_the_wrap(tree, path):
    tree.build()
    (tree.root / path).parent.mkdir(parents=True, exist_ok=True)
    (tree.root / path).write_text("{}")
    assert tree.build()[0] == list(ALL)
    tree.edit(path)
    assert tree.build()[0] == list(ALL)
    (tree.root / path).unlink()
    assert tree.build()[0] == list(ALL)


@pytest.mark.parametrize("path", ["sotaigen/data/deep/table.bin", "sotaigen/data/table.csv", "sotaigen/table.csv"])
def test_a_data_file_anywhere_counts_for_the_wrap(tree, path):
    tree.build()
    (tree.root / path).parent.mkdir(parents=True, exist_ok=True)
    (tree.root / path).write_text("1")
    assert tree.build()[0] == list(ALL)


@pytest.mark.parametrize("name", ["extract.py", "PROVENANCE.md"])
def test_what_only_remakes_or_documents_the_template_does_not_count(tree, name):
    (tree.root / "template").mkdir(exist_ok=True)
    (tree.root / "template" / name).write_text("before")
    tree.build()
    tree.edit(f"template/{name}")
    assert tree.build()[0] == []


def test_asking_for_a_stage_builds_what_it_reads_and_nothing_below(tree):
    ran, report = tree.build(("rig",))
    assert ran == ["wrap", "rig"]
    assert report["pose"].status == SKIPPED
    assert not (tree.out / "poses").exists()


def test_redoing_a_stage_removes_stale_results_below_it(tree):
    tree.build()
    tree.edit("sotaigen/data/bodies/female/typical/long.json")
    tree.build(("wrap",))
    assert not (tree.out / "weights.npz").exists()
    assert not (tree.out / "poses").exists()
    assert vouched(tree.out) == {"wrap"}


def test_an_export_goes_with_the_rig_it_came_from_but_is_never_required(tree):
    tree.build()
    glb = tree.root / "build" / "bodies" / NAME / "body.glb"
    glb.write_text("exported")
    assert tree.build()[0] == []
    assert glb.exists()
    tree.change("sotaigen/rig.py")
    tree.build()
    assert not glb.exists()


def test_a_stale_stage_redoes_when_asked_for_after_an_upstream_change(tree):
    tree.build()
    tree.change("sotaigen/rig.py")
    tree.build(("wrap",))
    assert tree.build(("rig",))[0] == ["rig"]


def test_an_interrupted_stage_leaves_no_matching_fingerprint(tree):
    tree.build()
    tree.edit("sotaigen/data/bodies/female/typical/long.json")

    def interrupted(name, out):
        (out / "skeleton.json").write_text("half of a fi")
        raise KeyboardInterrupt

    pipeline.RUN["rig"] = interrupted
    with pytest.raises(KeyboardInterrupt):
        tree.build()
    assert vouched(tree.out) == {"wrap"}

    pipeline.RUN["rig"] = tree.run("rig")
    assert tree.build()[0] == ["rig", "pose", "metrics"]


def test_a_missing_result_is_rebuilt_even_if_its_fingerprint_matches(tree):
    tree.build()
    (tree.out / "poses" / f"{next(iter(POSES))}.obj").unlink()
    # The poses come back as they were, so the metrics made from them still hold.
    assert tree.build()[0] == ["pose"]
    os.remove(tree.out / "poses.json")
    assert tree.build()[0] == ["metrics"]


def test_force_redoes_what_is_up_to_date(tree):
    tree.build()
    assert tree.build(force=True)[0] == list(ALL)


def test_a_stage_that_writes_nothing_is_a_failure_and_stays_unfingerprinted(tree):
    pipeline.RUN["wrap"] = lambda name, out: None
    with pytest.raises(pipeline.BuildError, match="did not write wrapped"):
        tree.build()
    assert vouched(tree.out) == set()


def test_unknown_preset_and_stage_are_errors(tree):
    with pytest.raises(ValueError, match="unknown body: nope"):
        ensure_built("nope", ("wrap",))
    with pytest.raises(ValueError, match="unknown stage: nope"):
        ensure_built(NAME, ("nope",))


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["nope"], "unknown body: nope"),
        (["--sex", "h99"], "unknown sex: h99"),
        (["--group", "nope"], "unknown group: nope"),
        (["--stages", "wrap,nope"], "unknown stage: nope"),
        (["regular-soft", "--group", "slender"], "no body matches"),
        (["--jobs", "0"], "--jobs must be at least 1"),
    ],
)
def test_the_command_refuses_a_bad_selection(tree, capsys, argv, message):
    with pytest.raises(SystemExit) as stop:
        main(argv)
    assert stop.value.code == 2
    assert message in capsys.readouterr().err
    assert not tree.calls


def test_selection_by_name_body_and_style(catalogue):
    assert select([], None, None) == list(catalogue)
    assert select(["petite-soft"], None, None) == ["petite-soft"]
    assert select([], ["petite", "regular"], ["soft"]) == ["petite-soft", "regular-soft"]
    assert select([], None, ["soft"]) == ["petite-soft", "regular-soft", "tall-soft"]
    assert select(["petite-soft", "tall-soft"], ["tall"], None) == ["tall-soft"]


@pytest.fixture
def at_once(monkeypatch):
    # Threads for processes: a worker process would not see the tree's stand-ins.
    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", ThreadPoolExecutor)
    for variable in pipeline.SINGLE_THREADED:
        monkeypatch.setenv(variable, "1")


def test_presets_built_at_once_all_finish_and_report_in_the_order_asked(tree, at_once, capsys):
    assert main(["--group", "soft", "--stages", "wrap", "--jobs", "2"]) == 0
    soft = ["petite-soft", "regular-soft", "tall-soft"]
    for name in soft:
        assert "wrap" in pipeline.read_fingerprint(tree.root / "build" / "bodies" / name)
    printed = capsys.readouterr().out
    assert sorted(soft, key=printed.rindex) == soft
    assert "petite-slender" not in printed


def test_a_failure_among_presets_at_once_fails_the_build(tree, at_once, monkeypatch, capsys):
    good = pipeline.RUN["wrap"]
    def wrap(name, out):
        if name == "regular-soft":
            raise pipeline.BuildError("regular-soft: no surface")
        return good(name, out)
    monkeypatch.setitem(pipeline.RUN, "wrap", wrap)
    assert main(["--group", "soft", "--stages", "wrap", "--jobs", "2"]) == 1
    assert "BUILD FAILED: regular-soft: no surface" in capsys.readouterr().err
    assert "wrap" not in pipeline.read_fingerprint(tree.root / "build" / "bodies" / "regular-soft")


def test_the_command_reports_status_and_invariants(tree, capsys):
    assert main([NAME, "--stages", "wrap"]) == 0
    assert main([NAME, "--stages", "wrap"]) == 0
    printed = capsys.readouterr().out
    assert "built" in printed
    assert "cached" in printed
    assert "flipped 0, crossedMidline 0, footLowest 0.1" in printed


def modules_of(stage):
    return {key.split(":")[0] for key in sources.reached(pipeline.STAGE_CODE[stage])}


def test_the_named_inputs_are_all_counted_in_the_real_tree():
    wrap = modules_of("wrap")
    for module in ("form/anatomy", "form/field", "form/head", "params/spec", "form/wrap", "template", "mesh"):
        assert f"sotaigen/{module}.py" in wrap
    # The solve reads the pelvic sculpture, so the wrap does too.
    assert "sotaigen/form/groin.py" in wrap
    assert "sotaigen/rig/skeleton.py" in modules_of("rig")
    assert {"sotaigen/rig/poses.py", "sotaigen/metrics.py", "sotaigen/rig/skeleton.py"} <= modules_of("metrics")
    data = {str(p.relative_to(pipeline.ROOT)) for p in pipeline.data_inputs()}
    assert {"sotaigen/data/bodies/female/typical/long.json", "sotaigen/data/template/base-mesh.json"} <= data
    assert not {"template/extract.py", "template/PROVENANCE.md", "sotaigen/data/sliders.json"} & data
    assert "sotaigen/data/template/uv.npz" not in data


def test_what_only_reads_finished_results_feeds_no_stage():
    for stage in pipeline.STAGES:
        only_reads = {
            "sotaigen/export.py",
            "sotaigen/blend.py",
            "sotaigen/face.py",
            "sotaigen/compare.py",
            "sotaigen/displacement.py",
        }
        assert not only_reads & modules_of(stage), stage


def test_the_real_results_follow_the_agreed_layout():
    ensure_built(REAL, ALL)
    out = pipeline.BUILD / "bodies" / REAL
    for name in (
        "wrapped.npy",
        "wrapped.obj",
        "wrap.json",
        "skeleton.json",
        "weights.npz",
        "poses.json",
        "joints.json",
        "fingerprint.json",
    ):
        assert (out / name).is_file(), name
    assert sorted(p.stem for p in (out / "poses").glob("*.obj")) == sorted(POSES)


def test_the_saved_rig_is_what_rigged_reads():
    rig = pipeline.rigged(REAL)
    base, verts, bones, weights = rig.base, rig.verts, rig.bones, rig.weights
    names = [bone["name"] for bone in bones]
    saved = json.loads((pipeline.BUILD / "bodies" / REAL / "skeleton.json").read_text())["bones"]
    assert [row["name"] for row in saved] == names
    assert weights.shape == (len(verts), len(names))
    assert len(verts) == len(base["verts"])
    with np.load(pipeline.BUILD / "bodies" / REAL / "weights.npz") as file:
        assert file["bones"].tolist() == names


def test_every_preset_has_the_same_fixed_topology():
    sizes = set()
    for name in pipeline.bodies():
        ensure_built(name, ("wrap",))
        sizes.add(np.load(pipeline.BUILD / "bodies" / name / "wrapped.npy").shape)
    assert sizes == {(len(extended(load_template())["verts"]), 3)}


def pytest_exit_code(folder, body):
    """Run a test module under this project's conftest.py in a pytest of its own."""
    shutil.copy(Path(__file__).with_name("conftest.py"), folder)
    (folder / "test_case.py").write_text(f"import pytest\n\n{body}")
    command = [sys.executable, "-m", "pytest", "-q", str(folder), "--rootdir", str(folder)]
    return subprocess.run(command, cwd=folder, capture_output=True, check=False).returncode


@pytest.mark.parametrize(
    "body",
    [
        "def test_it():\n    pytest.skip('no')\n",
        "@pytest.mark.skip\ndef test_it():\n    pass\n",
        "pytest.skip('no', allow_module_level=True)\n",
        "@pytest.mark.parametrize('x', [])\ndef test_it(x):\n    pass\n",
    ],
)
def test_a_skip_fails_the_run(tmp_path, body):
    # A skip at module level fails collection, which pytest reports as an interrupted run.
    assert pytest_exit_code(tmp_path, body) in (
        pytest.ExitCode.TESTS_FAILED,
        pytest.ExitCode.INTERRUPTED,
    )


@pytest.mark.parametrize(
    "body",
    [
        "def test_it():\n    pytest.xfail('no')\n",
        "@pytest.fixture\ndef thing():\n    pytest.xfail('no')\n\ndef test_it(thing):\n    pass\n",
        "@pytest.mark.xfail\ndef test_it():\n    assert False\n",
        "@pytest.mark.xfail\ndef test_it():\n    pass\n",
        "@pytest.mark.xfail(strict=False)\ndef test_it():\n    pass\n",
        "@pytest.mark.xfail(run=False)\ndef test_it():\n    pass\n",
        "pytestmark = pytest.mark.xfail\n\ndef test_it():\n    assert False\n",
    ],
)
def test_an_xfail_fails_the_run(tmp_path, body):
    assert pytest_exit_code(tmp_path, body) == pytest.ExitCode.TESTS_FAILED


def test_a_plain_passing_test_still_passes_that_conftest(tmp_path):
    assert pytest_exit_code(tmp_path, "def test_it():\n    pass\n") == pytest.ExitCode.OK


TIERED = """def test_quick():
    pass

@pytest.mark.slow
def test_slow():
    pass

@pytest.mark.bake
def test_bake():
    pass

@pytest.mark.blender
def test_blender():
    pass
"""


def run_tier(folder, tier, *named):
    """The tests a run under this project's conftest.py keeps, and its exit code."""
    shutil.copy(Path(__file__).with_name("conftest.py"), folder)
    (folder / "test_case.py").write_text(f"import pytest\n\n{TIERED}")
    env = {k: v for k, v in os.environ.items() if k != "SOTAIGEN_TESTS"} | ({"SOTAIGEN_TESTS": tier} if tier else {})
    command = [sys.executable, "-m", "pytest", "-q", "--collect-only", "--rootdir", str(folder),
               *(named or [str(folder)])]
    done = subprocess.run(command, cwd=folder, env=env, capture_output=True, text=True, check=False)
    kept = {line.split("::")[-1] for line in done.stdout.splitlines() if "::" in line}
    return kept, done.returncode


def with_blender(*tests):
    return {*tests, "test_blender"} if find_blender() else set(tests)


@pytest.mark.parametrize(
    ("tier", "named", "kept"),
    [
        (None, (), {"test_quick"}),
        ("merge", (), with_blender("test_quick", "test_slow")),
        ("full", (), with_blender("test_quick", "test_slow", "test_bake")),
        # Named on the command line, its slow tests run, and a bake still takes asking for.
        (None, ("test_case.py",), with_blender("test_quick", "test_slow")),
        (None, ("test_case.py::test_slow",), {"test_slow"}),
    ],
)
def test_what_each_tier_keeps(tmp_path, tier, named, kept):
    assert run_tier(tmp_path, tier, *named)[0] == kept


def test_a_mistyped_tier_is_refused(tmp_path):
    # Not taken for the daily run: a merge check that ran a third of the tests would still pass.
    assert run_tier(tmp_path, "merg")[1] == pytest.ExitCode.USAGE_ERROR


def test_without_node_its_tests_leave_the_run_and_say_so(tmp_path):
    """Deselected, not skipped: a skip would fail the run, and the summary says how many left."""
    shutil.copy(Path(__file__).with_name("conftest.py"), tmp_path)
    (tmp_path / "test_case.py").write_text(
        "import pytest\n\ndef test_python():\n    pass\n\n@pytest.mark.node\ndef test_javascript():\n    pass\n"
    )
    python_only = tmp_path / "bin"
    python_only.mkdir()
    (python_only / "python").symlink_to(sys.executable)
    env = {**{k: v for k, v in os.environ.items() if k != "SOTAIGEN_TESTS"}, "PATH": str(python_only)}
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "--rootdir", str(tmp_path), str(tmp_path)],
                          cwd=tmp_path, env=env, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stdout
    assert "1 passed, 1 deselected" in done.stdout
    assert "1 tests that run JavaScript left out" in done.stdout
