"""Licenses: code that imports Blender's Python API is GPL, and every license named has its text.

Which license covers what is decided in docs/DECISIONS.md ("开源和授权") and laid out in REUSE.toml.
"""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ("blender", "scripts", "sotaigen", "sotaigen_dev", "template", "tests")
BLENDER_API = re.compile(r"^\s*(?:import|from)\s+(?:bpy|bmesh)\b", re.MULTILINE)
HEADER = re.compile(r"SPDX-License-Identifier:\s*(\S+)")
GPL = "GPL-3.0-or-later"


def annotations():
    return tomllib.loads((ROOT / "REUSE.toml").read_text())["annotations"]


def toml_license(relative):
    """The license REUSE.toml gives a file with no header of its own: the last table matching it."""
    found = None
    for table in annotations():
        paths = table["path"] if isinstance(table["path"], list) else [table["path"]]
        if any(Path(relative).full_match(pattern) for pattern in paths):
            found = table["SPDX-License-Identifier"]
    return found


def header(text):
    """The license a file's first two lines name (a shebang may come first), or None."""
    found = HEADER.search("\n".join(text.splitlines()[:2]))
    return found.group(1) if found else None


def python_files():
    for folder in CODE:
        yield from sorted((ROOT / folder).rglob("*.py"))


def test_blender_api_code_is_gpl():
    wrong = {}
    for path in python_files():
        text = path.read_text()
        if not BLENDER_API.search(text):
            continue
        relative = path.relative_to(ROOT).as_posix()
        license = header(text) or toml_license(relative)
        if license != GPL:
            wrong[relative] = license
    assert not wrong, f"imports bpy or bmesh but is not {GPL}: {wrong}"


def test_blender_api_code_lives_in_blender():
    # template/ keeps its two extraction scripts beside the mesh they remake.
    stray = [
        path.relative_to(ROOT).as_posix()
        for path in python_files()
        if BLENDER_API.search(path.read_text())
        and path.relative_to(ROOT).parts[0] not in ("blender", "template")
    ]
    assert not stray, f"imports bpy or bmesh outside blender/: {stray}"


def test_named_licenses_have_their_text():
    named = {table["SPDX-License-Identifier"] for table in annotations()}
    for path in python_files():
        named.add(header(path.read_text()))
    named.discard(None)
    texts = {path.stem for path in (ROOT / "LICENSES").glob("*.txt")}
    assert named <= texts, f"no text in LICENSES/ for {sorted(named - texts)}"
    assert (ROOT / "LICENSE").read_text() == (ROOT / "LICENSES" / "Apache-2.0.txt").read_text()
