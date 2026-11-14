"""Compare this checkout's finished metrics with another build directory: the gate for a merge.

uv run sotaigen-dev compare --against DIR [body ...] [--all] [--allow-stale]

Reads wrap.json, poses.json and joints.json on both sides and lists what differs, by body and
pose. Only results are read, nothing is built, and this checkout's results must be up to date; the
other side may be as old as it likes. Metrics don't prove an unchanged mesh, so the mesh and the
rig are compared by hash too, as a change to list and never as a regression. None of it proves a
good look.

Exit codes:
    0  no hard metric got worse; changes in the others are listed
    1  a hard metric got worse: more flipped faces, vertices past the midline, collapsed faces or
       legs through each other in a pose
    2  something could not be compared, such as a missing file on one side
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import NamedTuple

from sotaigen.pipeline import BUILD, bodies_of, fingerprints, read_fingerprint

FILES = ("wrap.json", "poses.json", "joints.json")
# Compared by content, where the metrics above could agree while the geometry differs.
HASHED = ("wrapped.npy", "rest.npy", "skeleton.json", "weights.npz")
HARD = {"flippedFaces", "crossedMidline", "collapsedFaces", "legsPastMidline"}
EXIT_REGRESSION = 1
EXIT_INCOMPLETE = 2
# Where a metric of the whole body goes in place of a pose's name.
WHOLE = "(whole body)"
ABSENT = "absent"
# The row that says a pose exists on one side only.
POSE_ROW = "(pose)"


def is_hard(metric):
    return metric.rsplit(".", 1)[-1] in HARD


class Row(NamedTuple):
    preset: str
    pose: str
    file: str
    metric: str
    before: object
    after: object

    @property
    def changed(self):
        return self.before != self.after

    @property
    def regression(self):
        """A hard metric that is worse, or no longer a number. Bad data must not pass as better."""
        if not is_hard(self.metric) or ABSENT in (self.before, self.after):
            return False
        try:
            return not self.after <= self.before
        except TypeError:
            return True

    def line(self):
        flag = "  REGRESSION" if self.regression else ""
        return f"{self.file:12} {self.metric:32} {self.before} -> {self.after}{flag}"


def flatten(value, prefix=""):
    """Nested dictionaries as one level, with dotted keys."""
    if not isinstance(value, dict):
        return {prefix: value}
    flat = {}
    for key, item in value.items():
        flat.update(flatten(item, f"{prefix}.{key}" if prefix else key))
    return flat


def by_pose(file, data):
    """A result file as {pose: {metric: value}}, with WHOLE for what isn't about one pose."""
    if file == "poses.json":
        posed, whole = data, {}
    elif file == "joints.json":
        posed, whole = data.get("poses", {}), {k: v for k, v in data.items() if k != "poses"}
    else:
        posed, whole = {}, data
    return {WHOLE: flatten(whole), **{pose: flatten(row) for pose, row in posed.items()}}


def one_sided(preset, file, pose, old, new):
    """Rows for a pose only one side has: that fact and its hard metrics. The rest would be noise."""
    present = old if new is None else new
    sides = (ABSENT, "present") if old is None else ("present", ABSENT)
    rows = [Row(preset, pose, file, POSE_ROW, *sides)]
    for metric, value in present.items():
        if is_hard(metric):
            before, after = (ABSENT, value) if old is None else (value, ABSENT)
            rows.append(Row(preset, pose, file, metric, before, after))
    return rows


def compare_file(preset, file, before, after):
    """Every metric of one file on both sides, and what can't be compared, as (rows, problems)."""
    rows, problems = [], []
    for pose in dict.fromkeys([*before, *after]):
        old, new = before.get(pose), after.get(pose)
        if old is None or new is None:
            rows += one_sided(preset, file, pose, old, new)
            continue
        for metric in dict.fromkeys([*old, *new]):
            if is_hard(metric) and (metric not in old or metric not in new):
                side = "--against" if metric not in old else "this build"
                problems.append(f"{preset} / {pose}: {file} of {side} has no {metric}")
                continue
            rows.append(
                Row(preset, pose, file, metric, old.get(metric, ABSENT), new.get(metric, ABSENT))
            )
    return rows, problems


def read_results(folder, file):
    """A result file by pose, or None and what is wrong with it."""
    path = folder / file
    if not path.exists():
        return None, f"{path} is missing"
    try:
        return by_pose(file, json.loads(path.read_text())), None
    except (json.JSONDecodeError, AttributeError) as error:
        return None, f"{path} is not a result file: {error}"


def content_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else ABSENT


def hash_rows(preset, against):
    """A row for each hashed file that either side has."""
    rows = []
    for file in HASHED:
        before, after = content_hash(against / preset / file), content_hash(BUILD / "bodies" / preset / file)
        if (before, after) != (ABSENT, ABSENT):
            rows.append(Row(preset, WHOLE, file, "sha256", before, after))
    return rows


def compare_preset(preset, against):
    """The results of a body and the hashes of its geometry, as (rows, problems)."""
    rows, problems = hash_rows(preset, against), []
    for file in FILES:
        before, before_problem = read_results(against / preset, file)
        after, after_problem = read_results(BUILD / "bodies" / preset, file)
        problems += [problem for problem in (before_problem, after_problem) if problem]
        if before is not None and after is not None:
            found, wrong = compare_file(preset, file, before, after)
            rows += found
            problems += wrong
    return rows, problems


def preset_folders(folder):
    """The folders of bodies that hold any results."""
    if not folder.is_dir():
        return set()
    return {
        path.name
        for path in folder.iterdir()
        if path.is_dir() and any((path / file).exists() for file in FILES)
    }


def stale_presets(presets):
    """Bodies of this checkout whose metrics no finished build vouches for."""
    current = fingerprints()["metrics"]
    return [
        name
        for name in presets
        if (BUILD / "bodies" / name).is_dir() and read_fingerprint(BUILD / "bodies" / name).get("metrics") != current
    ]


def print_rows(rows):
    """Rows grouped by body and then pose, in the order they were found."""
    preset = pose = None
    for row in rows:
        if row.preset != preset:
            preset, pose = row.preset, None
            print(preset)
        if row.pose != pose:
            pose = row.pose
            print(f"  {pose}")
        print(f"    {row.line()}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sotaigen-dev compare", description=__doc__.splitlines()[0])
    parser.add_argument("presets", nargs="*", metavar="name", help="default: all on either side")
    parser.add_argument("--against", required=True, type=Path, help="the other build directory")
    parser.add_argument("--all", action="store_true", help="list unchanged metrics too")
    parser.add_argument("--allow-stale", action="store_true", help="accept out-of-date results")
    args = parser.parse_args(argv)
    if not args.against.is_dir():
        parser.error(f"--against: {args.against} is not a directory")
    against, here = bodies_of(args.against), BUILD / "bodies"
    if against.resolve() == here.resolve():
        parser.error(f"--against: {args.against} is this build, which would always match itself")

    names = args.presets or sorted(preset_folders(against) | preset_folders(here))
    if not names:
        print(f"compare: no results in {against} or {here}", file=sys.stderr)
        return EXIT_INCOMPLETE
    stale = stale_presets(names)
    if stale and not args.allow_stale:
        listed = " ".join(stale)
        print(
            f"compare: results out of date for {listed}: sources changed since they were built, "
            f"or the build did not finish\nrun `uv run sotaigen-dev build {listed} --stages metrics` "
            "first, or pass --allow-stale to compare them anyway",
            file=sys.stderr,
        )
        return EXIT_INCOMPLETE
    if stale:
        print(f"compare: using out-of-date results for {' '.join(stale)}", file=sys.stderr)

    rows, problems = [], []
    for name in names:
        found, wrong = compare_preset(name, against)
        rows += found
        problems += wrong

    changes = [row for row in rows if row.changed]
    regressions = [row for row in rows if row.regression]
    print_rows(rows if args.all else changes)
    touched = len({row.preset for row in changes})
    print(f"{len(changes)} changed metrics in {touched} of {len(names)} bodies")
    for row in regressions:
        print(
            f"REGRESSION {row.preset} / {row.pose}: {row.file} {row.metric} "
            f"{row.before} -> {row.after}"
        )
    if any(row.metric.endswith("collapsedFaces") for row in regressions):
        print(
            "A joint with a card in docs/motion: `sotaigen_dev/motion/sheet.py folds <card> <body> "
            "--against DIR` splits these faces by whether the card expects a fold there."
        )
    for problem in problems:
        print(f"compare: {problem}", file=sys.stderr)
    if regressions:
        return EXIT_REGRESSION
    return EXIT_INCOMPLETE if problems else 0


if __name__ == "__main__":
    sys.exit(main())
