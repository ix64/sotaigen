"""Local development commands; the installed sotaigen CLI remains the downstream interface."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULES = {
    "build": ("sotaigen.pipeline", "build bodies: wrap, rig, pose and measure"),
    "export": ("sotaigen.output.glb", "export built-in bodies as GLB"),
    "blend": ("sotaigen.output.blend", "export built-in bodies as rigged .blend files"),
    "review": ("sotaigen_dev.review", "render, assemble and verify review sheets"),
    "compare": ("sotaigen_dev.compare", "compare build metrics and mesh hashes"),
    "displacement": ("sotaigen_dev.displacement", "measure mesh displacement between builds"),
    "motion": ("sotaigen_dev.motion.sheet", "inspect movement against an anatomy card"),
    "motion-blend": ("sotaigen_dev.motion.blend", "key, check and render real motion in Blender"),
    "stress": ("sotaigen_dev.stress", "exercise parameter ranges and inspect failures"),
    "contracts": ("sotaigen_dev.contracts", "check or regenerate downstream contract snapshots"),
    "reference": ("sotaigen_dev.reference", "generate or check the reference tables"),
    "schema": (
        "sotaigen_dev.editor.schema",
        "generate the editor schema; --publish writes JSON Schema",
    ),
}
TIERS = {"test": None, "test-merge": "merge", "test-full": "full"}
DESCRIPTIONS = {
    **{name: description for name, (_, description) in MODULES.items()},
    "test": "daily Python tests (4 workers; omit slow, Blender and contact bakes)",
    "test-merge": "Python tests including slow and Blender tests; omit contact bakes",
    "test-full": "all Python tests, including expensive contact bakes",
    "lint": "ruff and documentation checks; --docs-only checks just documentation",
}


def run_module(module, arguments, env=None):
    """Use this environment, from this checkout; keep subprocess output and exit status visible."""
    result = subprocess.run(
        [sys.executable, "-m", module, *arguments], cwd=ROOT, env=env, check=False
    )
    return result.returncode if result.returncode >= 0 else 128 - result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="sotaigen-dev",
        description="Development commands for this checkout. Frontend commands use pnpm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Commands:\n"
        + "\n".join(f"  {name:14} {description}" for name, description in DESCRIPTIONS.items())
        + "\n\nRun sotaigen-dev COMMAND --help for options. Paths are relative to the checkout root.",
    )
    parser.add_argument("command", choices=DESCRIPTIONS, metavar="COMMAND")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments for the command")
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        parser.print_help()
        return 0
    selected = parser.parse_args(argv)
    if not (ROOT / "sotaigen" / "pipeline.py").is_file():
        parser.error("install sotaigen-dev from a checkout with uv sync")
    args = selected.args
    if args[:1] == ["--"]:
        args = args[1:]
    try:
        if selected.command == "lint":
            lint = argparse.ArgumentParser(
                prog="sotaigen-dev lint", description=DESCRIPTIONS["lint"]
            )
            lint.add_argument("--docs-only", action="store_true")
            options = lint.parse_args(args)
            if not options.docs_only:
                status = run_module("ruff", ["check"])
                if status:
                    return status
            return run_module("sotaigen_dev.check_docs", [])
        if selected.command in TIERS:
            env = os.environ.copy()
            tier = TIERS[selected.command]
            if tier is None:
                env.pop("SOTAIGEN_TESTS", None)
            else:
                env["SOTAIGEN_TESTS"] = tier
            return run_module("pytest", ["-n", "4", "--dist", "loadfile", *args], env)
        if selected.command == "reference":
            reference = argparse.ArgumentParser(
                prog="sotaigen-dev reference", description=DESCRIPTIONS["reference"]
            )
            reference.add_argument("--check", action="store_true")
            reference.parse_args(args)
        if selected.command == "schema":
            schema = argparse.ArgumentParser(
                prog="sotaigen-dev schema", description=DESCRIPTIONS["schema"]
            )
            schema.add_argument("--publish", action="store_true")
            schema.parse_args(args)
        return run_module(MODULES[selected.command][0], args)
    except KeyboardInterrupt:
        return 130
