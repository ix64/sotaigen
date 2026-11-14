"""The package's public names (sotaigen.__all__, from sotaigen.api) stay what they are whatever else has
been imported: a submodule of the same name would take the name's place once imported, as
sotaigen/generate.py once did to sotaigen.generate. And the README's example runs as written."""

import json
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

import pytest

import sotaigen
from sotaigen.paths import DATA

ROOT = Path(__file__).resolve().parents[1]

# In a fresh interpreter: every submodule imported first, as a long-running process ends up with.
EVERY_MODULE_FIRST = """
import importlib, json, pkgutil, sys
import sotaigen
from sotaigen import api

for module in pkgutil.walk_packages(sotaigen.__path__, "sotaigen."):
    importlib.import_module(module.name)
wrong = [name for name in sotaigen.__all__ if getattr(sotaigen, name) is not getattr(api, name)]
spec = sotaigen.typical(sys.argv[1])
print(json.dumps({"wrong": wrong, "generate": callable(sotaigen.generate), "sex": spec.sex}))
"""


def test_no_submodule_is_named_as_a_public_name():
    modules = {module.name.rsplit(".", 1)[-1] for module in pkgutil.iter_modules(sotaigen.__path__)}
    assert not modules & set(sotaigen.__all__)


def test_the_public_names_are_the_apis_after_every_module_is_imported():
    done = subprocess.run(
        [sys.executable, "-c", EVERY_MODULE_FIRST, "female-long"], capture_output=True, text=True, check=True, cwd=ROOT
    )
    found = json.loads(done.stdout.splitlines()[-1])
    assert found == {"wrong": [], "generate": True, "sex": "female"}


@pytest.mark.slow
def test_the_readmes_example_runs(tmp_path):
    """The README's Python example, on a typical body's file and without the .blend, which needs Blender."""
    text = (ROOT / "README.md").read_text()
    example = re.search(r"```python\n(.*?)```", text, re.DOTALL).group(1)
    assert "sotaigen.generate(spec)" in example
    (tmp_path / "body.json").write_text((DATA / "bodies" / "female" / "typical" / "long.json").read_text())
    runnable = "\n".join(line for line in example.splitlines() if "write_blend" not in line)
    subprocess.run([sys.executable, "-c", runnable], cwd=tmp_path, check=True, capture_output=True, text=True)
    assert (tmp_path / "body.glb").stat().st_size > 1_000_000


def project(path):
    import tomllib

    return tomllib.loads(path.read_text())["project"]


def test_the_two_packages_share_one_version_and_say_where_they_come_from():
    """sotaigen and sotaigen-blender are released together; the version is what the files say made them."""
    from importlib.metadata import version

    main, gpl = project(ROOT / "pyproject.toml"), project(ROOT / "blender" / "pyproject.toml")
    assert main["version"] == gpl["version"] == version("sotaigen") == version("sotaigen-blender")
    for meta in (main, gpl):
        assert meta["description"] and meta["authors"] and meta["classifiers"]
        assert meta["urls"]["Repository"] == "https://github.com/ix64/sotaigen"
        assert (ROOT / ("" if meta is main else "blender") / meta["readme"]).is_file()
