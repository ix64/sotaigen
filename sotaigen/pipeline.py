"""One command from parameters to results: wrap, rig, pose and metrics, for any bodies.

uv run sotaigen-dev build [name ...] [--sex female,male] [--group long,balanced]
    [--stages wrap,rig,pose,metrics] [--force] [--jobs 4]

Everything lands in build/bodies/<body>/ (sotaigen.paths.build_dir says where build/ is):

    wrapped.npy, wrapped.obj, wrap.json        wrap
    skeleton.json, weights.npz                 rig
    poses/<pose>.obj                           pose
    poses.json, joints.json                    metrics
    fingerprint.json                           which inputs each finished stage was built from

Renders are no stage: the review tool renders what it shows and keeps them by their content.

A stage is redone when the SHA-256 of its inputs changes, never by modification time, which means
nothing in a fresh clone or a worktree. Its inputs are the code it can run (sotaigen.sources.code_key,
from the functions in STAGE_CODE), the data, and what the stages above it left: a stage redone with
the same results leaves those below it alone. Read results through ensure_built or rigged, so that
they can't be stale.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

from sotaigen.body import Body
from sotaigen.corrective import groin
from sotaigen.form import groin as groin_rest
from sotaigen.form.anatomy import calibrated, landmarks, skin_reference
from sotaigen.form.bodies import load_bodies
from sotaigen.form.head import extended
from sotaigen.form.wrap import wrap_body
from sotaigen.mesh import write_obj
from sotaigen.metrics import RestMesh, pose_row, posed_joint_report, posed_meshes
from sotaigen.params.file import Spec
from sotaigen.params.rules import sex_of
from sotaigen.paths import build_dir
from sotaigen.rig.poses import POSES, check_poses
from sotaigen.rig.skeleton import load_rig, save_rig, skeleton, skin_weights
from sotaigen.sources import code_key
from sotaigen.template import load_template

ROOT = Path(__file__).resolve().parents[1]
# A build: bodies/<name>/ for each body's results, cache/ for what the tools keep to go faster, and
# reports/ for what they write for people to look at (sotaigen.paths.build_dir).
BUILD = build_dir()
STAGES = ("wrap", "rig", "pose", "metrics")
DEFAULT_STAGES = "wrap,rig,pose,metrics"
# Leaves the rest of the machine to whatever else is running on it.
DEFAULT_JOBS = 4
# One thread for each process: a pool of bodies on top of a full-size BLAS pool in each only fights.
SINGLE_THREADED = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
# What each stage runs, as the functions that run it: its code is what they can reach.
STAGE_CODE = {
    "wrap": ("sotaigen/pipeline.py:run_wrap",),
    "rig": ("sotaigen/pipeline.py:run_rig",),
    "pose": ("sotaigen/pipeline.py:run_pose",),
    "metrics": ("sotaigen/pipeline.py:run_metrics",),
}
# What below sotaigen/ is no module and still no stage reads: the page's sliders, the parameter file's
# JSON Schema and the UV atlas, which only the export takes. Every other such file counts as data.
NOT_INPUTS = {"data/sliders.json", "data/params.schema.json", "data/template/uv.npz"}
# What each stage leaves in a body's folder, removed whole when the stage is redone.
ARTIFACTS = {
    "wrap": ("wrapped.npy", "wrapped.obj", "wrap.json"),
    "rig": ("skeleton.json", "weights.npz", "rest.npy"),
    "pose": ("poses",),
    "metrics": ("poses.json", "joints.json"),
}
# Made outside the build from a stage's results, so never required of it, but stale once it's redone.
DERIVED = {"rig": ("body.glb", "body-ipc.glb", "body.blend", "body-rest.blend")}
BUILT, CACHED, SKIPPED = "built", "cached", "skipped"


class BuildError(Exception):
    """A stage failed, as opposed to being skipped."""


class Stage(NamedTuple):
    """How one stage of one body went."""

    status: str
    seconds: float = 0.0


@cache
def bodies():
    return load_bodies()


@cache
def template():
    """The template with the head, whose vertex order every wrapped.npy follows."""
    return extended(load_template())


@cache
def surface(name):
    """A body's calibrated (field, landmarks). Shared between callers: don't change the field."""
    return calibrated(bodies()[name])


def rig_landmarks(name):
    """The landmarks a body's bones are placed from."""
    return landmarks(bodies()[name])


def bodies_of(build):
    """The folder of a build's bodies: its bodies/, or the build itself, laid out as builds were
    before they had one, so that a build of then can be compared with one of now."""
    build = Path(build)
    return build / "bodies" if (build / "bodies").is_dir() or not build.is_dir() else build


def files_below(folder, pattern="*"):
    """Files at any depth as {path relative to `folder`: path}, without caches or hidden files."""
    found = {}
    for path in folder.rglob(pattern):
        parts = path.relative_to(folder).parts
        if path.is_file() and not any(p == "__pycache__" or p.startswith(".") for p in parts):
            found[path.relative_to(folder).as_posix()] = path
    return found


def data_inputs():
    """The files besides code that any stage may read: whatever below sotaigen/ isn't a module, but
    for NOT_INPUTS. That is the package's data: the template, the bodies, the ranges of motion."""
    found = files_below(ROOT / "sotaigen")
    return sorted(path for name, path in found.items() if path.suffix != ".py" and name not in NOT_INPUTS)


def stage_keys():
    """Each stage's own inputs as a digest: the code it can run, and all the data.

    Every stage counts all the data rather than what it reads, so a change there redoes them all.
    """
    data = hashlib.sha256()
    for path in data_inputs():
        data.update(f"{path.relative_to(ROOT)}\n".encode())
        data.update(path.read_bytes())
    keys = {}
    for stage in STAGES:
        roots = [r for r in STAGE_CODE[stage] if (ROOT / r.partition(":")[0]).is_file()]
        missing = sorted(set(STAGE_CODE[stage]) - set(roots))
        code = code_key(roots, ROOT) if roots else ""
        own = f"{stage}\n{code}\nmissing {missing}\n{data.hexdigest()}\n"
        keys[stage] = hashlib.sha256(own.encode()).hexdigest()
    return keys


def fingerprints(keys=None):
    """Each stage's digest, chained so that any change upstream changes everything below it.

    This is what a finished stage records, and what readers check its results against.
    """
    digests, previous = {}, ""
    for stage, own in (keys or stage_keys()).items():
        previous = digests[stage] = hashlib.sha256(f"{stage}\n{previous}\n{own}\n".encode()).hexdigest()
    return digests


def upstream(stage, out):
    """A digest of what the stages above `stage` left in `out`, which is all it reads of them."""
    digest = hashlib.sha256()
    for above in STAGES[: STAGES.index(stage)]:
        for item in artifacts(above, out):
            files = sorted(item.rglob("*")) if item.is_dir() else [item]
            for path in (f for f in files if f.is_file()):
                digest.update(f"{path.relative_to(out)}\n".encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def read_fingerprint(out):
    path = out / "fingerprint.json"
    return json.loads(path.read_text()) if path.exists() else {}


def write_fingerprint(out, stored):
    """Replace the file whole: an interruption leaves the old one or the new one, never half."""
    path = out / "fingerprint.json"
    scratch = path.with_suffix(".tmp")
    scratch.write_text(json.dumps(stored, indent=1, sort_keys=True))
    os.replace(scratch, path)


def artifacts(stage, out):
    """The paths a finished stage must have left; a cache hit needs all of them."""
    if stage == "pose":
        return [out / "poses" / f"{pose}.obj" for pose in POSES]
    return [out / name for name in ARTIFACTS[stage]]


def invalidate(stage, out, stored, keys):
    """Forget a stage and all below it, and remove the results that can't hold any more.

    Those are the stage's own, and those of any stage below whose own inputs have changed. The rest
    stay without a fingerprint while the stages above them are redone: if those leave the same
    results, still_holds vouches for them again. The fingerprint goes first and comes back only
    after the stage has finished, so an interruption at any point leaves files that no fingerprint
    vouches for.
    """
    below = STAGES[STAGES.index(stage) :]
    basis = stored.setdefault("basis", {})
    gone = [name for name in below if name == stage or basis.get(name, {}).get("own") != keys[name]]
    for name in below:
        stored.pop(name, None)
    for name in gone:
        basis.pop(name, None)
    write_fingerprint(out, stored)
    for name in gone:
        for path in (out / item for item in (*ARTIFACTS[name], *DERIVED.get(name, ()))):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)


def still_holds(stage, out, stored, keys):
    """Whether a stage's results came from the same own inputs and the same results above it."""
    made_from = stored.get("basis", {}).get(stage)
    return made_from == {"own": keys[stage], "upstream": upstream(stage, out)}


def run_wrap(name, out):
    wrap_body(load_template(), bodies()[name], out)


def rig_body(p, verts):
    """The rig stage in memory: bones, weights and the finished rest skin of a wrapped body."""
    base = template()
    L = landmarks(p)
    reference_skin = skin_reference(p, verts)
    bones = skeleton(base, reference_skin, L)
    groin.record_amounts(bones, L)
    weights = skin_weights(base, bones, reference_skin, L)
    return bones, weights, groin_rest.rest_verts(base, verts, L)


def run_rig(name, out):
    bones, weights, rest = rig_body(bodies()[name], np.load(out / "wrapped.npy"))
    save_rig(out, bones, weights)
    np.save(out / "rest.npy", rest)


def run_pose(name, out):
    verts = np.load(out / "rest.npy")
    bones, weights = load_rig(out)
    (out / "poses").mkdir()
    base = template()
    poses = check_poses(base, verts, bones, weights)
    for pose, moved in posed_meshes(base, verts, bones, weights, poses, surface(name)[0]):
        write_obj(out / "poses" / f"{pose}.obj", moved, base["faces"])


def run_metrics(name, out):
    base = template()
    verts = np.load(out / "rest.npy")
    bones, weights = load_rig(out)
    rest = RestMesh(verts, base["faces"])
    poses = check_poses(base, verts, bones, weights)
    meshes = dict(posed_meshes(base, verts, bones, weights, poses, surface(name)[0]))
    table = {name: pose_row(rest, moved) for name, moved in meshes.items()}
    (out / "poses.json").write_text(json.dumps(table, indent=1))
    meshes.pop("rest")
    shoulder = rig_landmarks(name)["shoulderJoint"]
    report = posed_joint_report(base, verts, shoulder, meshes.items())
    (out / "joints.json").write_text(json.dumps(report, indent=1))


RUN = {
    "wrap": run_wrap,
    "rig": run_rig,
    "pose": run_pose,
    "metrics": run_metrics,
}


def parse_stages(stages):
    """The stages asked for, as a set, from an iterable or a comma-separated string."""
    wanted = set(stages.split(",") if isinstance(stages, str) else stages)
    unknown = wanted - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stage: {', '.join(sorted(unknown))}; known: {', '.join(STAGES)}")
    return wanted


def check_name(name):
    if name not in bodies():
        raise ValueError(f"unknown body: {name}; known: {', '.join(bodies())}")


def ensure_built(name, stages=("metrics",), force=False):
    """Bring the given stages of a body up to date, with everything they read from upstream.

    `force` redoes all of those even when their fingerprints match, or only the stages it names when
    it is a collection. Returns {stage: Stage} for all stages; those neither asked for nor needed
    are SKIPPED and left alone.
    """
    check_name(name)
    forced = set(STAGES) if force is True else set(force or ())
    wanted = parse_stages(stages)
    last = max((STAGES.index(stage) for stage in wanted), default=-1)
    out = BUILD / "bodies" / name
    out.mkdir(parents=True, exist_ok=True)
    stored = read_fingerprint(out)
    keys = stage_keys()
    current = fingerprints(keys)
    report = {stage: Stage(SKIPPED) for stage in STAGES}
    for stage in STAGES[: last + 1]:
        if stage not in forced and all(path.exists() for path in artifacts(stage, out)):
            if stored.get(stage) != current[stage] and still_holds(stage, out, stored, keys):
                stored[stage] = current[stage]
                write_fingerprint(out, stored)
            if stored.get(stage) == current[stage]:
                report[stage] = Stage(CACHED)
                continue
        invalidate(stage, out, stored, keys)
        print(f"{name}: {stage}", flush=True)
        start = time.perf_counter()
        status = RUN[stage](name, out) or BUILT
        report[stage] = Stage(status, time.perf_counter() - start)
        if status == BUILT:
            missing = [path for path in artifacts(stage, out) if not path.exists()]
            if missing:
                raise BuildError(f"{name}: {stage} did not write {missing[0].relative_to(out)}")
            stored[stage] = current[stage]
            stored["basis"][stage] = {"own": keys[stage], "upstream": upstream(stage, out)}
            write_fingerprint(out, stored)
    return report


@cache
def rigged(name):
    """A body's Body as the build has it, built first if out of date.

    The result is shared between callers in a process, so the arrays are read-only.
    """
    ensure_built(name, ("rig",))
    out = BUILD / "bodies" / name
    verts = np.load(out / "rest.npy")
    bones, weights = load_rig(out)
    verts.flags.writeable = weights.flags.writeable = False
    spec = Spec(sex_of(name), bodies().specs[name])
    wrap = json.loads((out / "wrap.json").read_text())
    return Body(verts, bones, weights, wrap, None, template(), rig_landmarks(name), spec, bodies()[name], name=name)


@cache
def rig_fingerprint():
    """The rig stage's fingerprint, once: the code this process started with, whatever happens to the files."""
    return fingerprints()["rig"]


def comma_list(value):
    return [item for item in value.split(",") if item]


def select(names, sexes, groups):
    """The bodies asked for by name, then narrowed by sex and group; an unknown one is an error."""
    known = bodies()
    for name in names:
        check_name(name)
    known_sexes = {sex_of(name) for name in known}
    known_groups = {name.split("-", 1)[1] for name in known}
    for kind, asked, valid in (("sex", sexes, known_sexes), ("group", groups, known_groups)):
        unknown = set(asked or ()) - valid
        if unknown:
            raise ValueError(
                f"unknown {kind}: {', '.join(sorted(unknown))}; known: {', '.join(sorted(valid))}"
            )
    chosen = [
        name
        for name in (names or known)
        if (not sexes or sex_of(name) in sexes)
        and (not groups or name.split("-", 1)[1] in groups)
    ]
    if not chosen:
        raise ValueError("no body matches that selection")
    return chosen


def status_table(reports):
    """Each body's stages as a table: built with its time, cached or skipped."""
    width = max(len(name) for name in reports) + 2
    lines = ["".ljust(width) + "".join(stage.ljust(14) for stage in STAGES) + "total"]
    for name, report in reports.items():
        cells = "".join(
            (f"{row.status} {row.seconds:.1f}s" if row.status == BUILT else row.status).ljust(14)
            for row in report.values()
        )
        total = sum(row.seconds for row in report.values())
        lines.append(f"{name.ljust(width)}{cells}{total:.1f}s")
    return "\n".join(lines)


def invariant_summary(name, report):
    """One line of the numbers that must hold, read from the stages that ran or hit the cache."""
    out, parts = BUILD / "bodies" / name, []
    if report["wrap"].status != SKIPPED:
        wrap = json.loads((out / "wrap.json").read_text())
        parts += [
            f"flipped {wrap['flippedFaces']}",
            f"crossedMidline {wrap['crossedMidline']}",
            f"footLowest {wrap['footLowest']}",
            f"hipFrontLead {wrap.get('hipFrontLead', '-')}",
        ]
    if report["metrics"].status != SKIPPED:
        table = json.loads((out / "poses.json").read_text())
        collapsed = {pose: row["collapsedFaces"] for pose, row in table.items()}
        parts.append(f"collapsed { {k: v for k, v in collapsed.items() if v} }")
    return f"{name}: " + ", ".join(parts)


def print_summary(reports):
    print()
    print(status_table(reports))
    print()
    for name, report in reports.items():
        print(invariant_summary(name, report))


def build_at_once(names, stages, force, jobs, reports):
    """Fill reports as bodies finish. A failure stops what hasn't started; what has, finishes.

    A body writes only its own folder, and what bodies share takes a lock or replaces a file.
    """
    for variable in SINGLE_THREADED:
        os.environ.setdefault(variable, "1")
    with ProcessPoolExecutor(jobs) as pool:
        running = {pool.submit(ensure_built, name, stages, force): name for name in names}
        try:
            for future in as_completed(running):
                reports[running[future]] = future.result()
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise


def build_all(names, stages, force, jobs, reports):
    """Bring the bodies up to date, as many at once as jobs says."""
    parse_stages(stages)
    if jobs == 1 or len(names) == 1:
        for name in names:
            reports[name] = ensure_built(name, stages, force=force)
        return
    build_at_once(names, stages, force, min(jobs, len(names)), reports)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sotaigen-dev build", description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", metavar="name", help="default: all of them")
    parser.add_argument("--sex", type=comma_list, help="only these sexes, as female,male")
    parser.add_argument("--group", type=comma_list, help="only these groups, as long,balanced")
    parser.add_argument(
        "--stages", default=DEFAULT_STAGES, help=f"default: {DEFAULT_STAGES}"
    )
    parser.add_argument("--force", action="store_true", help="redo even what is up to date")
    parser.add_argument(
        "--jobs", type=int, default=DEFAULT_JOBS, help=f"bodies at once, default {DEFAULT_JOBS}"
    )
    args = parser.parse_args(argv)
    try:
        names = select(args.names, args.sex, args.group)
        parse_stages(args.stages)
    except ValueError as error:
        parser.error(str(error))
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    reports = {}
    try:
        build_all(names, args.stages, args.force, args.jobs, reports)
    except BuildError as error:
        if reports:
            print_summary({name: reports[name] for name in names if name in reports})
        print(f"\nBUILD FAILED: {error}", file=sys.stderr)
        return 1
    reports = {name: reports[name] for name in names}
    print_summary(reports)
    return 0


if __name__ == "__main__":
    # From the importable module, not __main__: workers send back this module's Stage and BuildError,
    # which the parent could not unpickle under the name __mp_main__ that they have in a worker.
    from sotaigen.pipeline import main as run

    sys.exit(run())
