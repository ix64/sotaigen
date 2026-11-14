"""Where the tool's files are: its data, which ships inside the package, its caches and its build.

The caches (the solves, and the IPC contact tables when asked for) are in the user's cache folder,
shared by every checkout and every installed copy, since their keys cover the code that made them:
SOTAIGEN_CACHE names another. The build is where a checkout's bodies, caches of its own and reports go:
<checkout>/build when the package runs from one, else ./build, and SOTAIGEN_BUILD names another.
macOS and Linux only (docs/DECISIONS.md, 发布与兼容).
"""

import os
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
DATA = PACKAGE / "data"
# The checkout, when the package runs from one; otherwise the folder it is installed in.
ROOT = PACKAGE.parent


def cache_dir():
    """The user's cache folder for this tool, or SOTAIGEN_CACHE."""
    named = os.environ.get("SOTAIGEN_CACHE")
    if named:
        return Path(named)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "sotaigen"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "sotaigen"


def build_dir():
    """SOTAIGEN_BUILD, else the checkout's build/, else ./build."""
    named = os.environ.get("SOTAIGEN_BUILD")
    if named:
        return Path(named).resolve()
    checkout = (ROOT / "pyproject.toml").is_file() and (ROOT / "sotaigen").is_dir()
    return (ROOT if checkout else Path.cwd()) / "build"
