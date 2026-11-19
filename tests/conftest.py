"""A test that was skipped or allowed to fail checked nothing, so it fails the run.

That covers pytest.skip, mark.skip, an empty parametrize, pytest.xfail and mark.xfail, whether the
marked test then fails or passes. A known defect goes in its domain's document, not in a test mark.

Three tiers, each a run of the whole folder; what leaves one is deselected, not skipped, and the run
says how many left:

- The daily run leaves out tests marked `slow` and `bake`. Those marked `blender` count as slow,
  since each starts Blender.
- SOTAIGEN_TESTS=merge leaves out only `bake`. A merge runs this tier, so `slow` moves a test to later
  and takes nothing away.
- SOTAIGEN_TESTS=full runs everything. Tests marked `bake` check the IPC contact tables themselves,
  which take minutes per body to bake and which no body has unless asked for.

Naming a file or a test on the command line runs its slow tests in any tier: whoever names it wants it.
Tests marked `blender` need Blender, as rendering does; without it they leave every run the same way.
Tests marked `node` run the page's JavaScript under Node, installed with fnm using `.node-version`;
without `node` on PATH they leave the same way.

`uv run sotaigen-dev test` spreads files over 4 processes, a whole file to each.
Before any starts, this process brings every body's build up to date, so that the tests only read build/: two
processes building the same body would write the same files.
"""

import os
import shutil
from pathlib import Path

import pytest

BAKE = "bake"
BLENDER = "blender"
NODE = "node"
SLOW = "slow"
HERE = Path(__file__).resolve().parent
LEFT_OUT = []
NO_BLENDER = []
NO_NODE = []
SLOW_OUT = []
# How many each worker left out; any one of them will do, since each collects every test.
FROM_WORKERS = {}


def pytest_configure(config):
    config.addinivalue_line(
        "markers", f"{BAKE}: needs real IPC contact bakes, minutes per body; runs with SOTAIGEN_TESTS=full"
    )
    config.addinivalue_line("markers", f"{BLENDER}: needs Blender; left out when there is none")
    config.addinivalue_line("markers", f"{NODE}: runs JavaScript under Node; left out when there is none")
    config.addinivalue_line(
        "markers",
        f"{SLOW}: leaves the daily run; a merge runs it (uv run sotaigen-dev test-merge)",
    )
    if os.environ.get("SOTAIGEN_TESTS") not in (None, "merge", "full"):
        raise pytest.UsageError(f"SOTAIGEN_TESTS is merge, full or unset, not {os.environ['SOTAIGEN_TESTS']}")


def is_worker(config):
    return hasattr(config, "workerinput")


def names_tests(config):
    """Whether the command line names particular tests, not the whole folder."""
    return any(Path(arg.partition("::")[0]).resolve() != HERE for arg in config.args)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    """With workers to come, build every body first in this process."""
    config = session.config
    if is_worker(config) or not getattr(config.option, "numprocesses", None):
        return
    from sotaigen import pipeline

    # Inherited by the workers: a pool of them on top of a full-size BLAS pool in each only fights.
    for variable in pipeline.SINGLE_THREADED:
        os.environ.setdefault(variable, "1")
    pipeline.build_all(list(pipeline.bodies()), pipeline.DEFAULT_STAGES, False, pipeline.DEFAULT_JOBS, {})


def is_slow(item):
    return item.get_closest_marker(SLOW) or item.get_closest_marker(BLENDER)


def pytest_collection_modifyitems(config, items):
    LEFT_OUT.clear()
    NO_BLENDER.clear()
    NO_NODE.clear()
    SLOW_OUT.clear()
    from sotaigen.output.blender import find_blender

    if find_blender() is None:
        NO_BLENDER.extend(item for item in items if item.get_closest_marker(BLENDER))
        if NO_BLENDER:
            config.hook.pytest_deselected(items=list(NO_BLENDER))
            items[:] = [item for item in items if not item.get_closest_marker(BLENDER)]
    if shutil.which("node") is None:
        NO_NODE.extend(item for item in items if item.get_closest_marker(NODE))
        if NO_NODE:
            config.hook.pytest_deselected(items=list(NO_NODE))
            items[:] = [item for item in items if not item.get_closest_marker(NODE)]
    if is_worker(config):
        config.workeroutput["noBlender"] = len(NO_BLENDER)
        config.workeroutput["noNode"] = len(NO_NODE)
    if os.environ.get("SOTAIGEN_TESTS") == "full":
        return
    LEFT_OUT.extend(item for item in items if item.get_closest_marker(BAKE))
    if LEFT_OUT:
        config.hook.pytest_deselected(items=list(LEFT_OUT))
        items[:] = [item for item in items if not item.get_closest_marker(BAKE)]
    if is_worker(config):
        config.workeroutput["leftOut"] = len(LEFT_OUT)
    if os.environ.get("SOTAIGEN_TESTS") == "merge" or names_tests(config):
        return
    SLOW_OUT.extend(item for item in items if is_slow(item))
    if SLOW_OUT:
        config.hook.pytest_deselected(items=list(SLOW_OUT))
        items[:] = [item for item in items if not is_slow(item)]
    if is_worker(config):
        config.workeroutput["slowOut"] = len(SLOW_OUT)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    FROM_WORKERS["leftOut"] = getattr(node, "workeroutput", {}).get("leftOut", 0)
    FROM_WORKERS["noBlender"] = getattr(node, "workeroutput", {}).get("noBlender", 0)
    FROM_WORKERS["noNode"] = getattr(node, "workeroutput", {}).get("noNode", 0)
    FROM_WORKERS["slowOut"] = getattr(node, "workeroutput", {}).get("slowOut", 0)


def pytest_terminal_summary(terminalreporter):
    slow = len(SLOW_OUT) or FROM_WORKERS.get("slowOut", 0)
    if slow:
        terminalreporter.write_line(
            f"{slow} slow tests left out; a merge runs them (uv run sotaigen-dev test-merge)."
        )
    count = len(LEFT_OUT) or FROM_WORKERS.get("leftOut", 0)
    if count:
        terminalreporter.write_line(
            f"{count} tests of real contact bakes left out; run them with "
            "uv run sotaigen-dev test-full."
        )
    missing = len(NO_BLENDER) or FROM_WORKERS.get("noBlender", 0)
    if missing:
        terminalreporter.write_line(
            f"{missing} tests that need Blender left out: Blender was not found "
            "(set BLENDER=/path/to/blender)."
        )
    without_node = len(NO_NODE) or FROM_WORKERS.get("noNode", 0)
    if without_node:
        terminalreporter.write_line(
            f"{without_node} tests that run JavaScript left out: node was not found (run fnm install and fnm use)."
        )


@pytest.fixture
def blender_found(monkeypatch):
    """For a test that stands in for Blender: whatever asks whether this machine can write a .blend
    before computing anything hears yes, so the test runs the same with Blender or without."""
    from sotaigen import cli
    from sotaigen.output import blend
    from sotaigen_dev.editor import worker

    for module in (cli, blend, worker):
        monkeypatch.setattr(module, "can_write_blend", lambda: None)


def fail_unchecked(report):
    if hasattr(report, "wasxfail"):
        report.outcome = "failed"
        report.longrepr = f"An xfail counts as a failure in this project: {report.wasxfail}"
        del report.wasxfail
    elif report.skipped:
        report.outcome = "failed"
        report.longrepr = f"A skip counts as a failure in this project: {report.longrepr}"
    return report


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    return fail_unchecked((yield))


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    return fail_unchecked((yield))
