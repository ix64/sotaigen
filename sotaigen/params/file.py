"""Read and write the parameter file.

The only reader: the page sends a file's text here, through web/serve.mjs, which runs this module
(`python -m sotaigen.params.file` reads the text on stdin and answers one JSON line). The format is in
docs/systems/editor.md. Its checks stop at the file's shape: ranges and rules belong to
sotaigen.params.rules.problems, which every entry point calls.
"""

import json
import math
import sys
from pathlib import Path
from typing import NamedTuple

from sotaigen.diagnostics import Diagnostic
from sotaigen.errors import SotaigenError

FORMAT = "sotaigen/params"
VERSION = 1
KNOWN_FIELDS = {"format", "version", "sex", "body", "base", "generator"}


class FileError(SotaigenError, ValueError):
    """Not a parameter file this reader can use. The message says why, and `diagnostic` is it as data."""

    def __init__(self, code, values=None):
        self.diagnostic = Diagnostic(code, "error", values)
        super().__init__(str(self.diagnostic))


class Spec(NamedTuple):
    """One body's parameters: its sex ("female" or "male"), the measurements in mm by name
    (docs/PARAMETERS.md), and the file's `base` (what the parameters started from) and fields this reader
    doesn't know, which travel with it."""

    sex: str
    body: dict
    base: dict | None = None
    unknown: dict | None = None


def read_parameter_file(text):
    """A file's text as {"sex", "body", "base", "unknown"}; raises FileError as the page does."""
    try:
        file = json.loads(text, parse_constant=lambda name: math.nan)
    except json.JSONDecodeError as error:
        raise FileError("file.not_json", {"detail": str(error)}) from error
    if not isinstance(file, dict):
        raise FileError("file.not_object")
    if file.get("format") != FORMAT:
        raise FileError("file.format", {"format": json.dumps(file.get("format"))})
    version = file.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise FileError("file.version", {"version": json.dumps(version)})
    if version > VERSION:
        raise FileError("file.newer", {"version": version, "latest": VERSION})
    if not isinstance(file.get("sex"), str):
        raise FileError("file.no_sex")
    if not isinstance(file.get("body"), dict):
        raise FileError("file.not_parameters")
    unknown = {key: value for key, value in file.items() if key not in KNOWN_FIELDS}
    return {"sex": file["sex"], "body": file["body"], "base": file.get("base"), "unknown": unknown}


def write_parameter_file(body, sex, base=None, unknown=None, generator=None):
    file = {"format": FORMAT, "version": VERSION, "sex": sex}
    if base:
        file["base"] = base
    if generator:
        file["generator"] = generator
    file["body"] = body
    return json.dumps({**(unknown or {}), **file}, indent=2) + "\n"


def read_spec(path):
    """A parameter file as a Spec; FileError when it can't be read as one."""
    try:
        text = Path(path).read_text()
    except (OSError, UnicodeDecodeError) as error:
        raise FileError("file.unreadable", {"detail": str(error.strerror or error)}) from error
    file = read_parameter_file(text)
    return Spec(file["sex"], file["body"], file["base"], file["unknown"])


def plain_numbers(value):
    """A file's values as JSON can carry them: a number that isn't finite as its text, which the
    page's checks then call not a number, as they do here."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: plain_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [plain_numbers(item) for item in value]
    return value


def main():
    """A file's text on stdin, one JSON line out: {"ok": true, "file": ...} or {"ok": false, "problems": [...]}."""
    try:
        answer = {"ok": True, "file": plain_numbers(read_parameter_file(sys.stdin.read()))}
    except FileError as error:
        answer = {"ok": False, "problems": [error.diagnostic.json()]}
    print(json.dumps(answer, allow_nan=False))


if __name__ == "__main__":
    main()
