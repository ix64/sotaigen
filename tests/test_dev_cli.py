"""The installed developer command works from outside the checkout and preserves tool failures."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import sotaigen_dev_cli as cli

ENTRY = Path(sys.executable).parent / "sotaigen-dev"


@pytest.mark.parametrize("command", [None, *cli.MODULES, *cli.TIERS, "lint"])
def test_installed_command_help_is_read_only_and_works_outside_checkout(command, tmp_path):
    arguments = [str(ENTRY), *([command] if command else []), "--help"]
    result = subprocess.run(
        arguments, cwd=tmp_path, text=True, capture_output=True, check=False, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
    assert not list(tmp_path.iterdir())


def test_unknown_command_is_a_usage_error(tmp_path):
    result = subprocess.run(
        [str(ENTRY), "buid"], cwd=tmp_path, text=True, capture_output=True, check=False
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr and "build" in result.stderr


@pytest.mark.parametrize(
    "command,tier", [("test", None), ("test-merge", "merge"), ("test-full", "full")]
)
def test_test_tier_and_pytest_overrides_reach_child_without_changing_parent(
    monkeypatch, command, tier
):
    monkeypatch.setenv("SOTAIGEN_TESTS", "full")
    calls = []

    def invoke(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return SimpleNamespace(returncode=5)

    monkeypatch.setattr(cli.subprocess, "run", invoke)
    assert cli.main([command, "tests/test_dev_cli.py", "-n", "0", "-k", "a or b"]) == 5
    ((arguments, options),) = calls
    assert arguments[:3] == [sys.executable, "-m", "pytest"]
    assert arguments[3:] == [
        "-n",
        "4",
        "--dist",
        "loadfile",
        "tests/test_dev_cli.py",
        "-n",
        "0",
        "-k",
        "a or b",
    ]
    assert options["cwd"] == cli.ROOT
    assert options["env"].get("SOTAIGEN_TESTS") == tier
    assert os.environ["SOTAIGEN_TESTS"] == "full"


def test_module_arguments_and_failure_are_forwarded(monkeypatch, tmp_path):
    calls = []
    out = str(tmp_path / "two words" / "sheet.png")

    def invoke(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return SimpleNamespace(returncode=3)

    monkeypatch.setattr(cli.subprocess, "run", invoke)
    assert cli.main(["review", "sheet", "female-long", "--out", out]) == 3
    assert calls[0][0] == [
        sys.executable,
        "-m",
        "sotaigen_dev.review",
        "sheet",
        "female-long",
        "--out",
        out,
    ]
    assert calls[0][1]["cwd"] == cli.ROOT


@pytest.mark.parametrize(
    "status,expected", [(0, ["ruff", "sotaigen_dev.check_docs"]), (1, ["ruff"])]
)
def test_lint_stops_at_the_first_failure(monkeypatch, status, expected):
    seen = []

    def run(module, arguments, env=None):
        seen.append(module)
        return status

    monkeypatch.setattr(cli, "run_module", run)
    assert cli.main(["lint"]) == status
    assert seen == expected


def test_docs_only_needs_no_ruff(monkeypatch):
    seen = []

    def run(module, arguments, env=None):
        seen.append(module)
        return 1

    monkeypatch.setattr(cli, "run_module", run)
    assert cli.main(["lint", "--docs-only"]) == 1
    assert seen == ["sotaigen_dev.check_docs"]


@pytest.mark.parametrize("command", ["schema", "reference", "lint"])
def test_bad_options_never_generate_files(monkeypatch, command):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid options must not execute a command")

    monkeypatch.setattr(cli, "run_module", unexpected)
    with pytest.raises(SystemExit) as error:
        cli.main([command, "--not-an-option"])
    assert error.value.code == 2


def test_signal_exit_status_is_preserved(monkeypatch):
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=-15))
    assert cli.main(["build", "female-long"]) == 143
