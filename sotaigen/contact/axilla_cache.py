"""Content-addressed, validated shoulder bakes shared by generation and export.

Only completed two-axis audits are loaded. Failed/incomplete studies remain in a
private temporary directory and never become a usable cache entry. Twisted
poses keep the twist fade and are explicitly outside this contact validation scope.

A body whose bake fails is still generated: it gets the older axilla correctives, with no
contact protection, and failure() says why so that the page and the build can show it.
"""

import json
import sys

import numpy as np

from sotaigen.contact import cache
from sotaigen.corrective import axilla
from sotaigen.corrective.pose_grid import from_samples
from sotaigen.paths import cache_dir
from sotaigen.sources import bake_key

# Shared by every checkout: a bake takes minutes, and its key covers all that decides the result.
CACHE = cache_dir() / "axilla-contact"
# What decides an entry or a failure record: the bake, the mirror, and the checks before it is kept.
BAKED_BY = ("sotaigen/contact/axilla_bake.py:bake_side", "sotaigen/contact/axilla_bake.py:mirror_side",
            "sotaigen/contact/axilla_cache.py:load_samples", "sotaigen/corrective/pose_grid.py:from_samples")


def fingerprint(base, verts, bones, weights):
    return bake_key(BAKED_BY, base, verts, bones, weights)


def load_samples(folder, key):
    manifest = json.loads((folder / "ready.json").read_text())
    if manifest != {"fingerprint": key, "scope": "twist-zero", "auditStepDegrees": 1.25}:
        raise ValueError("Shoulder cache manifest differs from its inputs")
    samples = {}
    for side in ("left", "right"):
        report = json.loads((folder / side / "report.json").read_text())
        if not report.get("auditPassed"):
            raise ValueError("Shoulder cache did not pass its stated contact audit")
        final = report["rounds"][-1]
        if final["stepDegrees"] != 1.25 or final["failed"] or final["unsafeIntervals"]:
            raise ValueError("Incomplete shoulder contact audit")
        with np.load(folder / side / "table.npz", allow_pickle=False) as data:
            if not data["initialComplete"].all():
                raise ValueError("Shoulder samples are incomplete")
            samples[side] = {name: data[name].copy() for name in ("points", "cells", "ids", "deltas")}
    return samples


def failure(base, verts, bones, weights):
    """Why this input has no contact table, or None. Delete the record to bake it again."""
    return cache.failure(CACHE, fingerprint(base, verts, bones, weights))


def targets(base, verts, bones, weights):
    # Import the solver only on this explicit production path; GLB readers
    # and drivers do not need to import IPC merely to read a file.
    from sotaigen.contact.axilla_bake import bake_side, mirror_side

    key = fingerprint(base, verts, bones, weights)

    def bake(folder):
        hashes = {"inputFingerprint": key}
        bake_side(base, verts, bones, weights, folder / "left", side="left", input_hashes=hashes)
        if not mirror_side(base, verts, bones, weights, folder / "left", folder / "right"):
            bake_side(base, verts, bones, weights, folder / "right", side="right", input_hashes=hashes)
        (folder / "ready.json").write_text(json.dumps(
            {"fingerprint": key, "scope": "twist-zero", "auditStepDegrees": 1.25}))
        # Validate the morph/driver contract too, before publishing ready. No partial table is ever
        # returned: a failed bake's folder is kept apart from the entries, for its reports.
        from_samples(load_samples(folder, key), len(verts), axilla.TWIST_LIMIT)

    samples, why = cache.baked(CACHE, key, bake, lambda folder: load_samples(folder, key))
    if samples is None:
        print(f"shoulder contact bake failed, using older correctives: {why}", file=sys.stderr, flush=True)
        return axilla.targets(base, verts, bones, weights)
    return from_samples(samples, len(verts), axilla.TWIST_LIMIT)