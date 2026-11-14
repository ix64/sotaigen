"""Blender itself: where it is on this machine, the scripts sotaigen-blender gives it, and running one.

The scripts use Blender's Python API and are GPL, in the separate package sotaigen-blender
(blender/ in the checkout). This finds their folder without importing them and runs them in a Blender
process of their own. Apart from the .blend writer so that what only needs Blender (the review
renders, the head swap) doesn't load the exporters with it.
"""

import functools
import importlib.util
import os
import re
import shutil
import subprocess
from pathlib import Path

from sotaigen.errors import BlenderMissing, BlenderTooOld

BLENDER_APP = Path("/Applications/Blender.app/Contents/MacOS/Blender")
# What this package hands sotaigen-blender's scripts, and they refuse another: the two packages are installed
# apart, so a version of one may meet another of the other. Raised when the bundle's meaning changes.
BUNDLE_FORMAT, BUNDLE_VERSION = "sotaigen/blender-bundle", 1
# The names a .blend's readers find things by, defined here once; the bundle carries them, and the
# scripts refuse a bundle whose names aren't theirs.
NAMES = {"rig": "rig", "body": "body", "headRegion": "headRegion", "neckRing": "neckRing", "info": "sotaigen"}
# The oldest Blender sotaigen supports: the .blend's constraints, drivers and left/right pairing were
# made and measured in 5.2, and nothing older has been run. sotaigen-blender holds the same floor.
MINIMUM_BLENDER = (5, 2)
VERSION = re.compile(r"^Blender (\d+)\.(\d+)", re.MULTILINE)
NO_BLENDER = "Blender was not found: install it, or point at it with BLENDER=/path/to/blender"
NO_SCRIPTS = "the .blend needs the package sotaigen-blender (GPL), which isn't installed: pip install sotaigen-blender"


def find_blender():
    """BLENDER, then PATH, then the macOS app. A BLENDER that names nothing is not a fallthrough."""
    named = os.environ.get("BLENDER")
    if named:
        path = Path(named)
        return path if path.is_file() and os.access(path, os.X_OK) else None
    found = shutil.which("blender")
    if found:
        return Path(found)
    return BLENDER_APP if BLENDER_APP.is_file() else None


@functools.cache
def _version(path, modified):
    done = subprocess.run([str(path), "--version"], capture_output=True, text=True, check=False, timeout=120)
    match = VERSION.search(done.stdout)
    return (int(match[1]), int(match[2])) if match else None


def blender_version(path):
    """(major, minor) of the Blender at `path`, from its --version; None when it doesn't say."""
    return _version(path, path.stat().st_mtime_ns)


def too_old(path):
    """Why the Blender at `path` won't do, or None. One that doesn't say its version is let through:
    sotaigen-blender's scripts check it again from inside."""
    version = blender_version(path)
    if version is None or version >= MINIMUM_BLENDER:
        return None
    need = ".".join(map(str, MINIMUM_BLENDER))
    return (
        f"Blender {version[0]}.{version[1]} at {path} is older than {need}, the oldest sotaigen supports: "
        f"install {need} or later, or point at one with BLENDER=/path/to/blender"
    )


def run_blender(*args, blend=None):
    """Blender in the background with factory settings, so with Auto Run Python Scripts off."""
    blender = find_blender()
    if blender is None:
        raise BlenderMissing(NO_BLENDER)
    old = too_old(blender)
    if old:
        raise BlenderTooOld(old)
    command = [str(blender), "-b", "--factory-startup"]
    if blend is not None:
        command.append(str(blend))
    command += ["--python-exit-code", "1", "--python", *map(str, args[:1]), "--", *map(str, args[1:])]
    done = subprocess.run(command, check=False, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"Blender failed: {' '.join(command)}\n{done.stdout[-3000:]}{done.stderr[-3000:]}")
    return done


def script(name):
    """The path of sotaigen-blender's script `name` ("convert", "pose", "head_swap"), for run_blender.
    BlenderMissing when the package isn't installed."""
    spec = importlib.util.find_spec("sotaigen_blender")
    if spec is None or not spec.submodule_search_locations:
        raise BlenderMissing(NO_SCRIPTS)
    return Path(next(iter(spec.submodule_search_locations))) / f"{name}.py"


def can_write_blend():
    """None when this machine can write a .blend, else why not: no Blender, one too old, or no
    sotaigen-blender."""
    if importlib.util.find_spec("sotaigen_blender") is None:
        return NO_SCRIPTS
    blender = find_blender()
    return too_old(blender) if blender else NO_BLENDER
