"""Look at bodies that are not bodies: wrap and rig them in a scratch folder, then draw a sheet.

uv run scripts/sample_sheet.py SPEC.json -o OUT.png -- [review sheet args]

SPEC.json is {"row-col": {"body": {...}}}, measurements as in sotaigen/data/bodies/; a name is split at its last hyphen, as
review.py splits bodies, so its two halves become the sheet's body and style axes. Everything goes
through review.py's own sheet, so scale and floor lines mean what they mean there. data/ and the
bodies' build folders stay untouched.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sotaigen.form.anatomy import landmarks
from sotaigen.form.head import extended
from sotaigen.form.solve import solved_controls
from sotaigen.form.wrap import wrap_body
from sotaigen.mesh import write_obj
from sotaigen.pipeline import fingerprints, write_fingerprint
from sotaigen.rig.poses import POSES
from sotaigen.rig.skeleton import skeleton, skin_weights
from sotaigen.rig.skinning import pose_verts
from sotaigen.template import load_template
from sotaigen_dev.stress import worker_pool

SAMPLES = ROOT / "build" / "reports" / "samples"


def load_review():
    """A copy of the review tool of its own, whose BUILD and POSES this sets."""
    spec = importlib.util.spec_from_file_location("review", importlib.util.find_spec("sotaigen_dev.review").origin)
    review = importlib.util.module_from_spec(spec)
    sys.modules["review"] = review  # dataclasses look their module up while the file executes
    spec.loader.exec_module(review)
    return review


def build_sample(job):
    """Wrap, rig and pose one sample, unless the same parameters and sources already did."""
    name, spec, folder = job
    out = Path(folder) / name
    stamp = fingerprints()["pose"]
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    params = out / "params.sha256"
    finished = out / "fingerprint.json"
    same = params.exists() and params.read_text() == digest and finished.exists()
    if same and json.loads(finished.read_text()).get("pose") == stamp:
        return name, False
    p = solved_controls(spec["body"])
    wrap_body(load_template(), p, out)
    base = extended(load_template())
    verts = np.load(out / "wrapped.npy")
    L = landmarks(p)  # as sotaigen.pipeline.rig_landmarks
    bones = skeleton(base, verts, L)
    weights = skin_weights(base, bones, verts, L)
    (out / "poses").mkdir(exist_ok=True)
    for pose, rotations in POSES.items():
        posed = pose_verts(bones, weights, verts, rotations)
        write_obj(out / "poses" / f"{pose}.obj", posed, base["faces"])
    params.write_text(digest)
    write_fingerprint(out, {"pose": stamp})
    return name, True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("spec")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("sheet", nargs="*", help="after --: arguments for review.py sheet")
    args = parser.parse_args(argv)
    specs = json.loads(Path(args.spec).read_text())
    folder = SAMPLES / Path(args.spec).stem
    folder.mkdir(parents=True, exist_ok=True)
    jobs = [(name, spec, folder) for name, spec in specs.items()]
    with worker_pool(args.workers) as pool:
        made = sum(fresh for _, fresh in pool.map(build_sample, jobs))
    print(f"{made} of {len(jobs)} samples built in {folder}")

    review = load_review()
    catalogue = {name: solved_controls(spec["body"]) for name, spec in specs.items()}
    review.load_bodies = lambda: catalogue
    review.BUILD = folder
    return review.cmd_sheet([*args.sheet, "-o", args.out])


if __name__ == "__main__":
    sys.exit(main())
