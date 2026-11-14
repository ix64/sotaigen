"""The default path stands apart from the IPC contact stack: what imports and makes a body's default
correctives loads neither ipctk nor any module of sotaigen.contact, and no module of the tool it loads
imports fcntl, which Windows doesn't have (docs/DECISIONS.md, 发布与兼容).

Python's own subprocess imports fcntl on macOS and Linux, so the check is on the tool's modules rather
than on sys.modules.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

import sotaigen

PACKAGE = Path(sotaigen.__file__).resolve().parent
CONTACT_ONLY = {"ipctk", "fcntl"}

# Runs in a fresh interpreter, so that nothing an earlier test imported counts.
DEFAULT = """
import json, sys
import sotaigen.cli, sotaigen.generation, sotaigen.pipeline
from sotaigen.corrective.recipe import Correctives
from sotaigen.output.glb import glb_bytes
from sotaigen.output.quality import glb_quality
from sotaigen.pipeline import bodies, rigged, surface
from sotaigen.uv import export_fields

name = next(iter(bodies()))
rig = rigged(name)
base = {**rig.base, **export_fields(rig.base["faces"])}
made = Correctives.production(base, rig.verts, rig.bones, rig.weights, field=surface(name)[0])
wrap = json.loads((sotaigen.pipeline.BUILD / "bodies" / name / "wrap.json").read_text())
quality = glb_quality(wrap, base, rig.verts, rig.bones, rig.weights, made)
glb_bytes(base, rig.verts, rig.bones, rig.weights, {"quality": quality}, correctives=made)
print(json.dumps({"modules": sorted(sys.modules), "contact": made.contact}))
"""


def imported_names(path):
    """Every module a file imports, at any depth of its code."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


def test_only_the_contact_package_imports_the_contact_stack():
    """Not even inside a function: outside sotaigen/contact/, nothing names ipctk or fcntl."""
    found = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(PACKAGE.parent).as_posix()
        if relative.startswith("sotaigen/contact/"):
            continue
        if used := imported_names(path) & CONTACT_ONLY:
            found[relative] = sorted(used)
    assert not found, found


@pytest.mark.slow
def test_making_the_default_correctives_loads_no_contact_code():
    done = subprocess.run([sys.executable, "-c", DEFAULT], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr[-3000:]
    result = json.loads(done.stdout.splitlines()[-1])
    assert result["contact"] == "sculpted"
    modules = result["modules"]
    assert "ipctk" not in modules
    assert not [name for name in modules if name.startswith("sotaigen.contact.")]
    ours = [name for name in modules if name == "sotaigen" or name.startswith("sotaigen.")]
    importing_fcntl = []
    for name in ours:
        spec_path = PACKAGE.parent / Path(*name.split("."))
        path = spec_path / "__init__.py" if spec_path.is_dir() else spec_path.with_suffix(".py")
        if "fcntl" in imported_names(path):
            importing_fcntl.append(name)
    assert not importing_fcntl
