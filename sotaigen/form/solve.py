"""From what a body should measure to the controls that make it.

A spec asks for a chest breadth or a calf girth; the masses take a radius. The two differ, and each
control moves more than its own measurement (see SENSITIVITY). This finds the controls by Newton steps
with one stored table, corrected as it goes, so a body costs three to six looks at the surface, about
0.2 s each. A result is kept in the user's cache (sotaigen.paths.cache_dir) under the digest of the code
_row can run, so every checkout shares them: a body costs little, but a whole test run solves thousands.

A target past what the masses can reach gets as close as it can, and target_misses says by how much:
the page shows it rather than move the number the user asked for.
"""

import hashlib
import json
import os
from collections.abc import Mapping

import numpy as np

from sotaigen.form import groin
from sotaigen.form.abdomen import LOWER, STATION, UPPER, AbdominalProfile, sagittal_depths
from sotaigen.form.anatomy import calibrated, landmarks
from sotaigen.form.field import peeled
from sotaigen.form.measures import measure, neutral
from sotaigen.form.thigh import ThighProfile
from sotaigen.params.spec import ALL_SPEC, PROFILES, estimated_controls
from sotaigen.paths import cache_dir
from sotaigen.sources import code_key
from sotaigen.template import template_landmarks

# The measurements the solve reaches by their controls (sotaigen.params.spec's rows with bounds), in the
# order of SENSITIVITY's rows and columns. Not the table's order: the order is the solve's arithmetic.
SOLVED = (
    "chestBreadth",
    "chestDepth",
    "hipDepth",
    "neckGirth",
    "upperArmGirth",
    "wristGirth",
    "kneeGirth",
    "calfGirth",
    "ankleGirth",
    "upperTrunkLean",
    "pelvisLean",
    "upperThighGirth",
    "waistDepth",
)
# Spec key -> the control that mainly drives it.
TARGETS = {key: ALL_SPEC[key].control for key in SOLVED}
# Controls in degrees and not shares of the torso, as the arc a degree sweeps at one torso.
DEGREES = frozenset(ALL_SPEC[key].control for key in SOLVED if ALL_SPEC[key].degrees)
# What the masses can bear, by control.
LIMITS = {ALL_SPEC[key].control: ALL_SPEC[key].bounds for key in SOLVED}
# Measured mm per control mm: the median over six typical bodies (female and male slim, long, balanced,
# 2026-11-11), rows are TARGETS' measurements, columns their controls. It only has to point the right
# way, since the steps correct it. To redo it, move each control by 4%, or a tilt by 2 degrees, and read.
SENSITIVITY = np.array((
    (0.834, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000),
    (0.000, 0.899, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.029, 0.000, 0.000, 0.000),
    (0.000, 0.000, 0.655, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.026, 0.274, 0.000),
    (0.000, 0.000, 0.000, 10.506, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000),
    (0.000, 0.000, 0.000, 0.000, 4.493, 1.358, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000),
    (0.000, 0.000, 0.000, 0.000, 1.003, 4.641, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000),
    (0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 5.679, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000),
    (0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2.531, 2.635, 0.652, 0.000, 0.000, 0.000, 0.000),
    (0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.267, 0.000, 4.553, 0.000, 0.000, 0.000, 0.000),
    (0.000, -0.041, 0.000, 0.044, 0.000, 0.000, 0.000, 0.000, 0.000, 0.089, 0.021, 0.000, 0.000),
    (0.000, 0.000, -0.058, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.056, 0.000, 0.021),
    (0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.118, 0.000, 0.000, 0.000, 0.000, 2.696, 0.000),
    (0.000, 0.076, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.027, 0.000, 0.651),
))
# Close enough, in mm: well below what the wrap resolves.
TOLERANCE = 0.02
# A miss worth telling anyone about, in mm.
SHOWN_MISS = 1.0
MOST_STEPS = 14

FOLDER = cache_dir() / "solve"
_memory = {}


def _sources():
    """The code a solve can run and the template's landmarks it reads (sotaigen.form.anatomy.landmarks):
    a result stays good until either changes."""
    digest = hashlib.sha256(code_key(["sotaigen/form/solve.py:_row"]).encode())
    digest.update(json.dumps(template_landmarks(), sort_keys=True).encode())
    return digest.hexdigest()[:16]


_digest = None


def _read_disk(key):
    global _digest
    _digest = _digest or _sources()
    path = FOLDER / _digest / f"{key}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _write_disk(key, row):
    path = FOLDER / _digest / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique per process too: two test workers may solve the same body at once, and id() repeats.
    scratch = path.with_suffix(f".{os.getpid()}.{id(row):x}.tmp")
    scratch.write_text(json.dumps(row))
    scratch.replace(path)


def _unit(control, torso):
    """A control's change, in the table's millimetres, for a change of one."""
    return torso * np.pi / 180 if control in DEGREES else torso


def _vector(measured):
    return np.array([measured[key] for key in TARGETS])


def _move(table, miss, p, torso):
    """The step in controls towards `miss`, leaving out any control already pushed against its limit."""
    controls = list(TARGETS.values())
    free = np.ones(len(controls), dtype=bool)
    for _ in controls:
        move = np.zeros(len(controls))
        move[free] = np.linalg.solve(table[free][:, free], miss[free])
        stuck = np.array(
            [
                (p[c] <= LIMITS[c][0] + 1e-9 and m < 0) or (p[c] >= LIMITS[c][1] - 1e-9 and m > 0)
                for c, m in zip(controls, move, strict=True)
            ]
        )
        if not (stuck & free).any():
            break
        free &= ~stuck
        if not free.any():
            return np.zeros(len(controls))
    return move


def _steps(start, target):
    """The solved shares for `target`, from the shares in `start`, and what they measure.

    A target that can't be reached mustn't take the others with it: so the measure of a step is the sum
    of the misses, the table keeps SENSITIVITY's zeros, and a control at its limit drops out of the sums.
    """
    torso = landmarks(start)["T"]
    p = dict(start)
    table = SENSITIVITY.copy()
    unrelated = np.abs(SENSITIVITY) < 0.02
    now = _vector(measure(p))
    best, best_now, best_miss = p, now, np.abs(target - now).sum()
    since_best = 0
    for _ in range(MOST_STEPS):
        if np.abs(target - now).max() < TOLERANCE or since_best >= 3:
            break
        move = _move(table, target - now, p, torso)
        q = dict(p)
        for index, control in enumerate(TARGETS.values()):
            low, high = LIMITS[control]
            q[control] = float(np.clip(p[control] + move[index] / _unit(control, torso), low, high))
        applied = np.array([(q[c] - p[c]) * _unit(c, torso) for c in TARGETS.values()])
        after = _vector(measure(q))
        if applied @ applied > 1e-12:
            table = table + np.outer(after - now - table @ applied, applied) / (applied @ applied)
            table[unrelated] = 0.0
        p, now = q, after
        miss = np.abs(target - now).sum()
        if miss < best_miss:
            best, best_now, best_miss = q, now, miss
            since_best = 0
        else:
            since_best += 1
    return best, best_now


def _solved(spec):
    start = neutral(estimated_controls(spec))
    start |= {c: float(np.clip(start[c], *LIMITS[c])) for c in DEGREES}
    target = np.array([spec[key] for key in TARGETS])
    key = hashlib.sha256(
        json.dumps([sorted(start.items()), target.tolist()]).encode()
    ).hexdigest()[:32]
    if key in _memory:
        return _memory[key]
    row = _read_disk(key)
    if row is None:
        row = _row(start, target)
        _write_disk(key, row)
    _memory[key] = row
    return row


def _row(start, target):
    """What is kept of a solve: the controls it found and what the body then measures."""
    p, measured = _steps(start, target)
    return {
        "controls": {control: p[control] for control in TARGETS.values()},
        "measured": dict(zip(TARGETS, measured.tolist(), strict=True)),
    }


def _profile_free(p):
    return {key: value for key, value in p.items() if key not in PROFILES}


def _sculpted_arc(depths, L):
    """The front arc, as a share of the torso, of these front depths once sotaigen.corrective.groin has shaped them."""
    heights = L["at"](np.array((LOWER, STATION, UPPER)))
    lower, station, upper = groin.sculpted_depths(depths, heights, L) / L["T"]
    fraction = (STATION - LOWER) / (UPPER - LOWER)
    return float(station - (1 - fraction) * lower - fraction * upper)


def delivered_front_arc(p):
    """What the finished skin of this body measures, with no abdominal profile asked for."""
    body, L = calibrated(_profile_free(p))
    return _sculpted_arc(sagittal_depths(body, L)[0], L)


def _reference_front_arc(p):
    """The arc to ask of the reference surface, so that the finished skin measures p's.

    The profile only moves the middle station, and the sculpture hardly depends on how deep it is.
    """
    body, L = calibrated(_profile_free(p))
    lower, _, upper = sagittal_depths(body, L)[0]
    fraction = (STATION - LOWER) / (UPPER - LOWER)
    chord = (1 - fraction) * lower + fraction * upper
    arc = asked = p["abdomenFrontArc"]
    for _ in range(8):
        arc += asked - _sculpted_arc((lower, chord + arc * L["T"], upper), L)
    return arc


def solved_controls(spec):
    """The internal parameters that make this body: what sotaigen.form.anatomy and every later stage read."""
    p = estimated_controls(spec) | _solved(spec)["controls"]
    if "abdomenFrontArc" in p:
        p["abdomenFrontArc"] = _reference_front_arc(p)
    return p


def measured_targets(spec):
    """What the solved body measures, by spec key."""
    return dict(_solved(spec)["measured"])


def _flanks_reached(spec):
    """What the buttock's two flanks reach, in mm, by spec key, if either is asked for."""
    asked = [key for key in ("gluteUpperFullness", "gluteLowerFullness") if spec.get(key)]
    if not asked:
        return {}
    p = solved_controls(spec)
    body, L = calibrated(_profile_free(p))
    reached = {"gluteUpperFullness": body.upper_reached, "gluteLowerFullness": body.lower_reached}
    return {key: reached[key] * L["T"] for key in asked}


def _contours_reached(spec):
    """What the abdomen's and the thigh's profiles reached, in mm, by spec key, where a target lay past
    what this body's profile can bend to: they go as far as they can (sotaigen.form.abdomen,
    sotaigen.form.thigh) rather than refuse the body."""
    if not any(key in spec for key in PROFILES):
        return {}
    p = solved_controls(spec)
    body, L = calibrated(p)
    reached = {}
    for layer in peeled(body, (ThighProfile, AbdominalProfile))[1]:
        reached |= layer.reached
    # The abdomen's arc was asked of the reference surface: the finished skin's misses by as much.
    return {key: spec[key] + (value - p[key]) * L["T"] for key, value in reached.items()}


def target_misses(spec):
    """{key: (asked, measured)} for what the masses, the buttock's flanks or the abdomen's and the thigh's
    profiles couldn't reach."""
    measured = measured_targets(spec) | _flanks_reached(spec) | _contours_reached(spec)
    return {
        key: (spec[key], value) for key, value in measured.items() if abs(value - spec[key]) > SHOWN_MISS
    }


class SolvedBodies(Mapping):
    """Bodies by name, solved when one is read, since most runs want one or two of them."""

    def __init__(self, specs):
        self.specs = specs

    def __getitem__(self, name):
        return solved_controls(self.specs[name])

    def __contains__(self, name):
        return name in self.specs

    def __iter__(self):
        return iter(self.specs)

    def __len__(self):
        return len(self.specs)
