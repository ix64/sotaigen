"""The range of every parameter, and what a body must meet.

Four layers, cheapest first:

    ranges  each parameter on its own, in millimetres on this body's own lengths
    rules   relations between the internal controls, for what only breaks later, when wrapping or
            posing
    seams   rules that predict the seam limits of tests/test_wrap.py. Those limits need the finished
            wrap, so for anything not yet wrapped this is the only warning there is. It is cautious
            and not exact: a body that passes those tests is fine whatever this says, so
            check_bodies leaves it out
    field   what the calibrated surface really measures against what was asked for: exact, because
            it is the invariant itself, and about 0.2 s, so load_bodies doesn't run it

The numbers come from wrapping and rigging bodies nobody designed, with sotaigen_dev/stress.py; the
table in docs/systems/params.md says what breaks past each one. A valid body is one the sex,
the calibration and the rig hold. That is no promise of a good look.

Each sex has its own: sotaigen.params.spec.SEXES says which parameters its bodies hold and in what
ranges, and SEX_RULES and SEX_SEAM_RULES here which rules they meet. Every check takes the
sex's name, with no default, because a body's numbers alone don't say which it is. Like the rules
here the ranges are provisional: a coarse scan on the masses of that time. They depend on the masses, the neck and
the rig, so they have to be measured again once those settle; docs/systems/params.md has the commands.
"""

import math

from sotaigen.diagnostics import RULE_SENTENCES, Diagnostic, rule
from sotaigen.errors import SotaigenError
from sotaigen.params.spec import (
    ARM_REST_DEG,
    CHEST_BREADTH_GAIN,
    KNEE_GIRTH_GAIN,
    SEXES,
    UPPER_ARM_GIRTH_GAIN,
    optional_parameters,
    proportions,
    required,
    units,
)

# Each sex's ranges, as shares of each parameter's unit, from sotaigen.params.spec.SEXES.
RANGES = {name: {key: row.range for key, row in table.items()} for name, table in SEXES.items()}
# Both sexes, until a stress test finds otherwise.
STATURE_RANGE = (1000.0, 1800.0)
# What the head-count tool can aim at.
HEADS_RANGE = (4.0, 7.95)


# Rounding, so that a body scaled as a whole doesn't fall off a limit it was standing on.
NUMERIC_SLACK = 1e-9


def arm_clearance(p):
    """How far the arm's inner edge stands from the ribs by the elbow, as a share of the torso."""
    ribs = (p["biacromialBreadth"] - p["chestBreadth"] / CHEST_BREADTH_GAIN) / 2
    return ribs + p["upperArmLength"] * math.sin(math.radians(ARM_REST_DEG)) - p["upperArmGirth"] / UPPER_ARM_GIRTH_GAIN


# Rules on what a body asks for, each read as proportions(spec): millimetres as shares of the torso. As
# (what it says, how much room is left); negative room breaks the rule. Each is what first broke past its
# limit, see docs/systems/params.md. docs/PARAMETERS.md says how the limits were found.
# The rule that guards this tool's own head, not the neck: `sotaigen blend --for-head-swap` lets it go.
NECK_THIN_FOR_HEAD = "neck_thin_head"

RULES = (
    # Faces flip at the armpit once the ribs reach the shoulders.
    ("shoulders_clear_chest", "the shoulders must clear the chest", lambda p: p["biacromialBreadth"] - p["chestBreadth"] / CHEST_BREADTH_GAIN + 0.02),
    # Past this the projection lands the inner arm on the ribs: flips, then the midline is crossed.
    ("arm_clear_ribs", "the arm must clear the ribs", lambda p: arm_clearance(p) - 0.196),
    # A trunk this short leaves the sex nothing to land on.
    ("torso_short", "the torso is too short", lambda p: 1 / p["stature"] - 0.2),
    # Past this the neck stands out behind the skull and ahead of the face: on male-balanced's head (0.90
    # deep) the side view of tests/test_head.py fails from 1.62, and the head turn collapses a face at
    # 1.71 whatever the head's depth. The chin stays on its landmark up to 1.80 (male probes, 2026-11-08).
    ("neck_thick_head", "the neck is too thick for the head", lambda p: 1.58 - p["neckGirth"] / p["headHeight"]),
    # A neck far thinner than the head may tear when the head turns or nods. 0.47 is the thinnest checked
    # (the neck probes of 2026-11-08): no collapse at the head turn and nod of the checks, larger angles
    # untried.
    ("neck_thin_head", "the neck is too thin for the head", lambda p: p["neckGirth"] / p["headHeight"] - 0.47),
    # The chin leaves its landmark under a head this small for its torso.
    ("torso_long_head", "the torso is too long for the head", lambda p: 3.0 - 1 / p["headHeight"]),
    # The femur runs down the middle of the half hip, so a thigh that doesn't fill it leaves the knee
    # far inside the hip, and in a crouch the knees could pass the midline. 1.05 is the thinnest the
    # masses make on a wide hip (female-balanced, 2026-11-12): no crossing, though the crouch collapses
    # more faces and the legs part below the crotch; user's choice.
    ("thigh_thin_hip", "the thigh is too thin for the hip", lambda p: p["upperThighGirth"] / p["hipBreadth"] - 1.05),
    # The foot pushes the ankle out to stay on its side. Past the knee by this much the shin leans in
    # towards the knee, and in a crouch the knees pass the midline.
    (
        "foot_long_knee", "the foot is too long for the knee",
        lambda p: 0.01 - (p["footLength"] * 0.186 - p["kneeGirth"] / KNEE_GIRTH_GAIN * 0.9),
    ),
    # Keep enough height for the arch while lowering the ankle on a foot of fixed length. 0.24 is the
    # lowest checked (real ankles, 2026-11-06), which wraps and seams within the limits.
    (
        "ankle_low_foot", "ankleHeight is too low for the foot",
        lambda p: p["ankleHeight"] - 0.24 * p["footLength"],
    ),
    (
        "ankle_high_foot", "ankleHeight is too high for the foot",
        lambda p: 0.55 * p["footLength"] - p["ankleHeight"],
    ),
    # A knee wider than the calf turns the leg's taper over and the projection flips faces there.
    ("knee_wide_calf", "the knee is too wide for the calf", lambda p: 1.06 - p["kneeGirth"] / p["calfGirth"]),
    # Below these the masses can't get any smaller: the thigh and the pelvis hold the hip's depth (in each
    # sex's table below), the trapezius a short neck, the arm's taper the wrist. A little short of
    # where they stop, as these are fits with some spread: the page says what a target just past them
    # comes to.
    # On a short neck the trapezius, higher and thicker with definition, and the blends, wider with
    # softness, reach the plane under the chin. Under every sampled body's floor (2026-11-08).
    (
        "neck_thin_length", "the neck is too thin for its length",
        lambda p: p["neckGirth"]
        - (0.51 - 5.36 * p["neckLength"] + 0.61 * p["definition"] + 0.215 * p["softness"]),
    ),
    (
        "wrist_thin_arm", "the wrist is too thin for the arm",
        lambda p: p["wristGirth"] - (0.175 + 0.23 * p["upperArmGirth"] - 0.22 * p["lowerArmLength"]),
    ),
)

# The rules whose limit differs by sex. Each limit is the furthest that sex's bodies have been
# checked at, so a sentence may stand at another number in the other table.
FEMALE_RULES = (
    # A thick arm on narrow shoulders collapses faces in the armpit when raised: all 8 collapses of a
    # stress test of 240 random bodies, both sexes, were within 0.018 of the rule on high shoulders
    # (each sex's own, below), on shoulders of 0.43-0.61 T (2026-11-07, docs/systems/presets.md). This
    # one rejects them all and 12 of the 232 that held. On shoulders wider than 0.72 T, where the two
    # meet, the other is the tighter.
    (
        "arm_thick_narrow_shoulders", "the arm is too thick for shoulders this narrow",
        lambda p: 0.094
        - (p["upperArmGirth"] / UPPER_ARM_GIRTH_GAIN + 0.83 * p["acromionRise"] - 0.15 * (p["biacromialBreadth"] - 0.7)),
    ),
    # A thick arm on a high shoulder collapses faces in the armpit when raised. 0.097 is the most a body
    # has been checked at (male-rounded at 0.0966, 2026-11-07; real adult arms and shoulders at 0.095,
    # 2026-11-06), with no collapse raising the arms.
    (
        "arm_thick_high_shoulder", "the arm is too thick for a shoulder this high",
        lambda p: 0.097 - (p["upperArmGirth"] / UPPER_ARM_GIRTH_GAIN + 0.83 * p["acromionRise"]),
    ),
    # A thicker thigh carries the hip's depth with it: below this the masses can't make the hip as
    # shallow as asked. The slope is a fit of where they stop over 174 bodies (typical, test and random
    # of both sexes, 2026-11-12; spread 0.028 T), the constant the shallowest shipped hip, female-chibi's.
    # It lets 29 of 75 random bodies ask for up to 0.066 T past their stop, which the page reports. The
    # upper thigh's slider lifts the hip to meet it (sotaigen/data/sliders.json). User's choice.
    (
        "hip_shallow_thigh", "the hip is too shallow for the thigh",
        lambda p: p["hipDepth"] - (0.375 * p["upperThighGirth"] + 0.0318),
    ),
)
MALE_RULES = (
    # The same on male bodies, through the female line's limit on 0.61 T shoulders, where the stress
    # test's collapses stop, and 0.012 past it on 0.72 T: male-long and male-slim on 0.62, 0.67 and
    # 0.72 T shoulders, their arms up to 0.018 past the female line (0.012 on 0.72 T, where 0.018 is past
    # the arm's range), had no flips and no collapse raising the arms (2026-11-12). Tighter than the
    # female line below 0.61 T; past 0.742 T the rule on high shoulders is the tighter. User's choice.
    (
        "arm_thick_narrow_shoulders", "the arm is too thick for shoulders this narrow",
        lambda p: 0.0805
        + 0.259 * (p["biacromialBreadth"] - 0.61)
        - (p["upperArmGirth"] / UPPER_ARM_GIRTH_GAIN + 0.83 * p["acromionRise"]),
    ),
    # The same on male bodies, checked further (2026-11-08): male-rounded's upper arm x1.05-1.2 on its
    # 0.75 T shoulders, male-balanced's x1.1-1.2 on 0.775 T, and the range's thickest arm on 0.85 T, up
    # to 0.1148: no flips, no collapse raising the arms, the axilla, wrap and rig tests passing.
    (
        "arm_thick_high_shoulder", "the arm is too thick for a shoulder this high",
        lambda p: 0.1148 - (p["upperArmGirth"] / UPPER_ARM_GIRTH_GAIN + 0.83 * p["acromionRise"]),
    ),
    # The same fit; the constant the shallowest shipped male hip, male-chibi's. It lets 32 of 73 random
    # bodies ask for up to 0.049 T past their stop, which the page reports.
    (
        "hip_shallow_thigh", "the hip is too shallow for the thigh",
        lambda p: p["hipDepth"] - (0.375 * p["upperThighGirth"] + 0.0262),
    ),
)

# What only a sex with a breast has to meet. The pectoral rule is a limit of the model, not of
# anatomy: a breast lies on the pectoral fascia, so a thicker muscle carries the breast forward and hides
# its own lower border under it, which the plate does under the thickness field (sotaigen/form/pectoral.py).
BREAST_RULES = (
    # The breast as a thickness field (`breastProjection`, sotaigen/form/breast.py) lies on the plate, and under
    # it the plate's lower border runs down into the fold. Built and looked at up to 0.0225 T, about a
    # male typical body's 12 mm, on the four female typical bodies with a breast and on female-layerbig
    # (2026-11-08, docs/parts/chest.md): no border, no new flips or collapsed faces. Nothing thicker
    # has been looked at. The room is scaled by the breast so that a body without one has nothing to
    # meet: the page's copy takes plain arithmetic only.
    (
        "pectoral_under_breast_thin", "the pectoral plate under a breast must stay thin",
        lambda p: p.get("breastProjection", 0.0) * (0.0225 - p.get("pectoralThickness", 0.0)),
    ),
)

# The seam limits are shares and degrees, so a body's size doesn't count and neither do these: every
# one is a pure number. Hands and feet are the sex's own, scaled whole, and the limb's end has to
# meet them. The limits are the most a body has been checked at (2026-11-06: real hands, feet and
# ankles, and styled small ones), with the seams of tests/test_wrap.py still inside their limits.
SEAM_RULES = (
    # A big hand on a thin, short forearm folds at the wrist.
    (
        "hand_big_forearm", "the hand is too big for the forearm",
        lambda p: 0.887 - p["handLength"] ** 2 / (p["wristGirth"] * p["lowerArmLength"]),
    ),
    # The fold at the wrist in degrees, within 0.7 of a fit over 276 built bodies (2026-11-08): a hand
    # long for its forearm, and an arm that tapers hard to the wrist. 3 bodies of the stress tests that
    # every other rule let through passed its 35 degree limit; this blocks them, and 1 of the 224 random
    # bodies whose seams all held (at 35.0 degrees).
    (
        "hand_big_forearm_arm", "the hand is too big for this forearm and arm",
        lambda p: 33.7
        - (
            18.5 * p["handLength"] / p["lowerArmLength"]
            + 16.6 * p["handLength"] ** 2 / (p["wristGirth"] * p["lowerArmLength"])
            + 1.83 * p["upperArmGirth"] / p["wristGirth"]
            + 0.22
        ),
    ),
    # A thick wrist on a small hand misses the hand's seam by more than its limit. Checked on the
    # toddler (wrist 0.33 T, 2026-11-06): at 1.36 every test passes, at 1.42 the seam misses.
    ("wrist_thick_hand", "the wrist is too thick for the hand", lambda p: 1.36 - p["wristGirth"] / p["handLength"]),
    # The miss at the hand's seam as a share of the limb's radius, within 0.023 of a fit over the same
    # bodies: a thick wrist for the hand, a long forearm and a thin upper arm. 5 bodies passed the 0.65
    # limit under the rule above; this blocks them, and 4 of the 224 that held, at 0.636-0.649 (two of
    # them passed the limit as they were built on 2026-11-07).
    (
        "wrist_thick_hand_arm", "the wrist is too thick for this hand and arm",
        lambda p: 0.655
        - (
            0.548 * p["wristGirth"] / p["handLength"]
            + 0.221 * p["lowerArmLength"]
            - 0.146 * p["upperArmGirth"]
            - 0.112
        ),
    ),
    # The miss at the foot's seam, as a share of the limb's radius, rises with this; the checked body
    # stayed inside its 0.52 limit at 0.893, but plenty of random ones don't (the rule below). A wide
    # knee lets the shin taper in time.
    (
        "ankle_thick_foot", "the ankle is too thick for the foot",
        lambda p: 0.893 - (0.92 * p["ankleGirth"] - 0.13 * p["kneeGirth"]) / p["footLength"],
    ),
    # The same miss, within 0.026 of a fit over 276 built bodies (2026-11-08), rising less steeply as the
    # ankle thickens and falling with a thick knee. The rule above let 24 bodies of the stress tests
    # past the 0.52 limit, at 0.86-0.89 on it; this blocks them all, and 7 of the 224 random bodies
    # whose seams held, at 0.459-0.516.
    # A straight line in the same shares fits closer but blocks female-chibi, which misses by 0.476.
    # Past an ankle 1.32 times the foot this would ease off again, and the rule above holds it there.
    (
        "ankle_thick_foot_knee", "the ankle is too thick for this foot and knee",
        lambda p: 0.5
        - (
            3.475 * p["ankleGirth"] / p["footLength"]
            - 1.321 * (p["ankleGirth"] / p["footLength"]) ** 2
            - 0.355 * p["kneeGirth"]
            - 1.572
        ),
    ),
    ("ankle_thin_foot", "the ankle is too thin for the foot", lambda p: p["ankleGirth"] / p["footLength"] - 0.75),
    # The radii of the shin go with the torso and its length with the stature, so a long foot on a
    # long torso folds at the ankle.
    ("foot_long_body", "the foot is too long for the body", lambda p: 0.159 - p["footLength"] / p["stature"]),
)


# Which rules each sex's bodies meet, as the tables above, in this order. The male bodies meet the
# female rules but where male bodies were checked past a limit; what the rest cost a male body is in
# docs/PARAMETERS.md.
SEX_RULES = {"female": (RULES, FEMALE_RULES, BREAST_RULES), "male": (RULES, MALE_RULES)}
SEX_SEAM_RULES = {"female": (SEAM_RULES,), "male": (SEAM_RULES,)}


RULE_SENTENCES.update(
    {name: text for tables in (*SEX_RULES.values(), *SEX_SEAM_RULES.values()) for table in tables for name, text, _ in table}
)


def rules(sex):
    """A sex's rules as (name, sentence, room): room(p) below zero breaks it."""
    return tuple(row for table in SEX_RULES[sex] for row in table)


def seam_rules(sex):
    return tuple(row for table in SEX_SEAM_RULES[sex] for row in table)


class ParameterError(SotaigenError, ValueError):
    """Bodies that break the range table or a rule. The message lists every problem."""


def spec_bounds(spec, sex):
    """Each of a sex's parameters' (low, high) in its own unit, on this body's lengths."""
    u = units(spec)
    table = SEXES[sex]
    return {
        key: (low * u[table[key].unit], high * u[table[key].unit])
        for key, (low, high) in RANGES[sex].items()
    }


def _order(found):
    """Diagnostics in a fixed order, the same as the page's: by code, then by parameter."""
    return sorted(found, key=lambda d: (d.code, str(d.values.get("key", ""))))


def spec_problems(spec, sex):
    """What is missing, unknown, not a finite number or outside its range, in one body of a sex."""
    if sex not in SEXES:
        return [Diagnostic("param.sex", "error", {"sex": sex})]
    table = SEXES[sex]
    found = [Diagnostic("param.missing", "error", {"key": key}) for key in set(required(sex)) - spec.keys()]
    found += [Diagnostic("param.unknown", "error", {"key": key}) for key in spec.keys() - table.keys()]
    numbers = {}
    for key in spec.keys() & table.keys():
        value = spec[key]
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            found.append(Diagnostic("param.not_number", "error", {"key": key, "value": repr(value)}))
        else:
            numbers[key] = value
    if found:
        return _order(found)
    bounds = spec_bounds(spec, sex)
    for key, value in numbers.items():
        low, high = bounds[key]
        slack = NUMERIC_SLACK * max(abs(low), abs(high), 1.0)
        if not low - slack <= value <= high + slack:
            found.append(Diagnostic("param.outside", "error", {"key": key, "value": value, "low": low, "high": high}))
    height = units(spec)["stature"]
    low, high = STATURE_RANGE
    slack = NUMERIC_SLACK * max(abs(low), abs(high), 1.0)
    # The stature tool scales to an end and lands a rounding error past it.
    if not low - slack <= height <= high + slack:
        found.append(Diagnostic("param.outside", "error", {"key": "stature", "value": height, "low": low, "high": high}))
    return _order(found)


def broken(rules, p, level="error"):
    """The rules that the applied parameters break, as diagnostics."""
    return [rule(name, -room(p), level) for name, _, room in rules if room(p) < -NUMERIC_SLACK]


def problems(spec, sex):
    """What is wrong with a body of a sex by its ranges and rules; empty when they are met."""
    found = spec_problems(spec, sex)
    if found:
        return found
    return broken(rules(sex), proportions(spec))


def seam_problems(spec, sex):
    """What a sex's seam rules predict for a body, as warnings; empty when they expect no trouble."""
    return broken(seam_rules(sex), proportions(spec), "warning")


def sex_of(name):
    """The sex of a body named <sex>-<group>, as sotaigen.form.bodies.load_specs names them."""
    return name.split("-", 1)[0]


def check_bodies(specs):
    """Raise ParameterError unless every body, named <sex>-<group>, is valid for its sex.

    load_bodies calls this.
    """
    found = [
        f"{name}: {problem}"
        for name, spec in specs.items()
        for problem in problems(spec, sex_of(name))
    ]
    if found:
        raise ParameterError("\n".join(found))


def random_spec(rng, sex, optional=False):
    """A body of a sex with each parameter uniform in its range, valid or not."""
    ranges, table = RANGES[sex], SEXES[sex]
    pick = lambda key: float(rng.uniform(*ranges[key]))
    height = float(rng.uniform(*STATURE_RANGE))
    head, crotch = pick("headHeight") * height, pick("crotchHeight") * height
    neck = pick("neckLength") * head
    spec = {"headHeight": head, "neckLength": neck, "crotchHeight": crotch, "torsoLength": height - head - neck - crotch}
    u = {"head": head, "stature": height, "torso": spec["torsoLength"], "crotch": crotch, "one": 1.0}
    for key in required(sex):
        spec.setdefault(key, pick(key) * u[table[key].unit])
    u["hip"] = spec["hipBreadth"]
    if optional:
        for key in optional_parameters(sex):
            if rng.random() < 0.5:
                spec[key] = float(rng.uniform(*ranges[key])) * u[table[key].unit]
    return spec
