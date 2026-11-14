"""What only the surface can answer about a body: the invariants of ARCHITECTURE 3.5 that the
calibrated field breaks, and random valid bodies, which need it to retry what the masses can't make.

The ranges and rules in sotaigen.params.rules come first and are plain arithmetic; these come after,
at a solve and a field each.
"""

import numpy as np

from sotaigen.form.anatomy import calibrated, width_at
from sotaigen.form.field import Body
from sotaigen.form.solve import solved_controls, target_misses
from sotaigen.form.wrap import hip_front_lead
from sotaigen.params.rules import problems, random_spec, seam_problems

# The invariants of ARCHITECTURE 3.5 that the surface alone can answer.
WIDTH_TOLERANCE = 0.02
WRIST_CENTRE_MM = 1.0
KNEE_CENTRE = 0.012
KNEE_CLEAR = 0.005


def field_measures(p):
    """How far the calibrated surface misses what the internal controls ask for."""
    body, L = calibrated(p)
    T, at = L["T"], L["at"]

    waist = width_at(body, L["waist"])
    hip = max(width_at(body, at(f)) for f in np.linspace(-0.05, 0.25, 13))
    above, below = L["crotch"] + 0.03 * T, L["crotch"] - 0.03 * T

    # The wrist's section either side of its landmark.
    wrist = L["wrist"] - L["foreDir"] * 0.02 * T
    ts = np.arange(0.0, 0.3 * T, 0.25)
    across = np.array([L["foreDir"][2], 0.0, -L["foreDir"][0]])
    across /= np.linalg.norm(across)

    def reach(direction):
        outside = np.nonzero(body(wrist + np.outer(ts, direction)) > 0)[0]
        return float(ts[outside[0]]) if len(outside) else 0.3 * T

    # The knee's section along x, at its landmark.
    knee = L["kneeJoint"]
    xs = np.arange(0.0, 0.5 * T, 0.25)
    inside = body(np.column_stack((xs, np.full_like(xs, knee[1]), np.full_like(xs, knee[2])))) <= 0
    found = xs[np.nonzero(inside)[0]]
    uncarved = Body(body.groups, body.clips)

    # How the lower leg's two outlines lean, high and low on the calf.
    ankle = L["ankleJoint"]
    fractions = np.linspace(0.1, 0.7, 61)
    fine = np.arange(0.0, 0.4 * T, 0.1)
    lean = []
    for fraction in fractions:
        centre = knee + (ankle - knee) * fraction
        points = np.column_stack(
            (fine, np.full_like(fine, centre[1]), np.full_like(fine, centre[2]))
        )
        hit = fine[np.nonzero(body(points) <= 0)[0]]
        lean.append(hit[-1] - centre[0] - (centre[0] - hit[0]) if len(hit) else 0.0)
    return {
        "wristOffCentre": abs(reach(across) - reach(-across)),
        "kneeOffCentre": (abs((found[0] + found[-1]) / 2 - knee[0]) / T) if len(found) else 1.0,
        "kneeInner": float(found[0] / T) if len(found) else 0.0,
        "kneeThroughMidline": bool(uncarved(np.array([[0.0, knee[1], knee[2]]]))[0] <= 0),
        "calfLean": float(np.interp(0.25, fractions, lean) - np.interp(0.5, fractions, lean)),
        "waistError": waist / (p["waistBreadth"] * T) - 1,
        "hipError": hip / (p["hipBreadth"] * T) - 1,
        "hipTissueAtMidline": L["hipTissueAtMidline"],
        "hipFrontExcess": hip_front_lead(body, L) / T,
        "legsPart": bool(
            body(np.array([[0.0, 0.0, above]]))[0] < 0
            and body(np.array([[0.0, 0.0, below]]))[0] > 0
        ),
    }


def field_failures(measures):
    """Which invariants a row of field_measures breaks, by name."""
    bad = []
    if abs(measures["waistError"]) > WIDTH_TOLERANCE:
        bad.append("waistBreadth")
    if abs(measures["hipError"]) > WIDTH_TOLERANCE:
        bad.append("hipBreadth")
    if measures["hipTissueAtMidline"]:
        bad.append("hipTissueMidline")
    if not measures["legsPart"]:
        bad.append("legsPart")
    if measures["wristOffCentre"] > WRIST_CENTRE_MM:
        bad.append("wristCentre")
    if measures["kneeOffCentre"] > KNEE_CENTRE:
        bad.append("kneeCentre")
    if measures["kneeInner"] <= KNEE_CLEAR or measures["kneeThroughMidline"]:
        bad.append("kneeMidline")
    if measures["calfLean"] <= 0:
        bad.append("calfLean")
    return bad


def sample_spec(rng, sex, field=True):
    """A random valid body of a sex: uniform in the ranges, the rest by retry.

    `field` also retries what the surface can't reach or the masses can't make, at a few seconds a try.
    """
    while True:
        spec = random_spec(rng, sex)
        if problems(spec, sex) or seam_problems(spec, sex):
            continue
        if not field or (not target_misses(spec) and not field_failures(field_measures(solved_controls(spec)))):
            return spec
