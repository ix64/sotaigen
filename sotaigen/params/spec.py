"""What a body is: measurements in millimetres, and how the masses' internal controls follow from them.

The head, neck, torso and leg lengths are inputs and add up to the stature, so moving one moves nothing
else. The stature and the head count are results. A breadth or girth is what the finished surface should
measure, by the definitions in sotaigen/form/measures.py; docs/PARAMETERS.md says which few are still control
values.

The masses in sotaigen/form/anatomy.py and everything after them read shares of the torso, as they always have:
sotaigen.form.solve.solved_controls finds them. The rules read `proportions` here, plain arithmetic, because they
run at every slider event on the page, where sotaigen_dev/editor/schema.py writes out the same function.
tests/test_editor.py holds the two together.
"""

from typing import NamedTuple


class Param(NamedTuple):
    """One measurement a body file holds, and how the masses read it.

    control = value / (gain x unit) + offset. A gain is there for what is asked for as a measurement: it is
    that measurement's typical multiple of its control, where the solver in sotaigen/form/solve.py starts, and it
    finds the real control. `range` is in shares of the unit, on the body's own lengths; sotaigen/params/rules.py
    holds a body to it. `layer` names the local contour an optional parameter drives, which measurements
    and bindings have to leave out; the masses' own parameters have none.

    A measurement the solve reaches has `bounds`: what the masses can bear, as the control's shares of
    the torso, or its degrees when `degrees` (past these the thin end of a mass turns over).

    `basis` is what the value is measured on, one of BASES. `default` is what a body without an optional
    parameter has, in shares of its unit like the range; None where that is the body's own shape, which
    no number says.
    """

    control: str
    unit: str
    range: tuple[float, float]
    offset: float = 0.0
    gain: float = 1.0
    optional: bool = False
    layer: str | None = None
    bounds: tuple[float, float] | None = None
    degrees: bool = False
    basis: str = "skeleton"
    default: float | None = None


# What a parameter's value is measured on (Param.basis).
BASES = {
    "skeleton": "a length or height between the landmarks the bones are placed from",
    "surface": "the finished surface's measurement, as sotaigen/form/measures.py and the profiles read it; solved for",
    "calibrated": "the surface's breadth at its landmark, calibrated to within 2%",
    "chord": "the surface's offset from a straight line between two stations on the same body",
    "without": "how far the surface moves from the same body without this parameter",
    "under": "the most the tissue stands off the surface under it",
    "shape": "a dimension of the shape this parameter's part adds",
    "share": "a ratio of two dimensions of the same part",
    "control": "an internal control of the masses, not a measurement: what one value makes may change with the masses",
}


def optional(control, unit, range, layer=None, *, basis, default=0.0):
    """A parameter a body may leave out, which then has `default`: zero leaves the shape alone."""
    return Param(control, unit, range, optional=True, layer=layer, basis=basis, default=default)


# What a body without these has, in shares: the glute fold's span (of the hip's width) and the accepted
# groin's depth and relief (of the torso), which are the top of their ranges, so only lighter is on offer.
FOLD_SPAN = 0.2475
GROIN_DEPTH = 0.027
GROIN_RELIEF = 0.014


# The arms' rest angle from straight down, in degrees: not a parameter. The rest pose is what clothes
# and every correction, contact table and constraint are made on, so it is the same on every body
# (docs/DECISIONS.md); another rest pose is an export's step.
ARM_REST_DEG = 35.0

# The jaw's breadth over the head's where its control is zero: the head's masses (sotaigen/form/head.py)
# widen it by JAW_SPAN for each unit of their control.
JAW_BASE = 0.46
JAW_SPAN = 0.36

# A measurement's typical multiple of its control (Param.gain) where a rule reads the control through it:
# the rules in sotaigen/params/rules.py were found on the controls, so they convert by the same numbers.
CHEST_BREADTH_GAIN = 1.088
UPPER_ARM_GIRTH_GAIN = 5.689
KNEE_GIRTH_GAIN = 5.742
UPPER_THIGH_GIRTH_GAIN = 6.1
WAIST_DEPTH_GAIN = 1.08


# Every parameter, once. The ranges are provisional: a coarse scan on the masses of that time, see
# sotaigen/params/rules.py.
ALL_SPEC = {
    # The small-head end, the neck's long end, the deep head and the shallow waist were checked past the
    # coarse scan on 2026-11-06 with real teenage and toddler measurements, and the shallow chest with a real
    # seven-year-old's: the head, face, wrap and UV tests held.
    "headHeight": Param("headShare", "stature", (1 / 7.95, 1 / 4), basis="skeleton"),
    "neckLength": Param("neck", "head", (0.15, 0.535), basis="skeleton"),
    "torsoLength": Param("torsoShare", "stature", (0.2, 0.45), basis="skeleton"),
    "crotchHeight": Param("leg", "stature", (0.36, 0.575), basis="skeleton"),
    "kneeHeight": Param("kneeHeightRatio", "crotch", (0.54, 0.66), basis="skeleton"),
    "ankleHeight": Param("ankleHeight", "torso", (0.075, 0.25), basis="skeleton"),
    # The acromion stands 0.03 T inside the control's half width.
    "biacromialBreadth": Param("shoulderWidth", "torso", (0.3976, 0.7608), offset=0.06, basis="skeleton"),
    # The control's zero stands 0.015 T below the suprasternal notch.
    "acromionRise": Param("shoulderRise", "torso", (-0.115, 0.015), offset=0.015, basis="skeleton"),
    "waistRise": Param("waist", "torso", (0.38, 0.48), basis="skeleton"),
    "chestBreadth": Param("chestWidth", "torso", (0.435, 0.61), gain=CHEST_BREADTH_GAIN, bounds=(0.3, 0.7), basis="surface"),
    "chestDepth": Param("chestDepth", "torso", (0.312, 0.46), gain=1.054, bounds=(0.22, 0.6), basis="surface"),
    "waistBreadth": Param("waistBreadth", "torso", (0.33, 0.6), basis="calibrated"),
    # The waist's depth at its landmark, front to back on the midline. The range is where the masses
    # reach it on random bodies within the rules (0.30-0.45 T nearly always, none under 0.25 T), widened
    # to take every shipped body (0.279-0.383 T); user's choice, 2026-11-12 (docs/PARAMETERS.md).
    "waistDepth": Param("waistDepth", "torso", (0.27, 0.45), gain=WAIST_DEPTH_GAIN, bounds=(0.16, 0.46), basis="surface"),
    "hipBreadth": Param("hipBreadth", "torso", (0.44, 0.75), basis="calibrated"),
    # Down to 0.30 T, where long and slim bodies of both sexes were built clean (2026-11-12): the real
    # long builds' 0.3177 T fits. Elsewhere the hip rule or the masses stop first; user's choice.
    "hipDepth": Param("hipDepth", "torso", (0.30, 0.48), gain=1.002, bounds=(0.25, 0.7), basis="surface"),
    "upperArmLength": Param("upperArm", "torso", (0.4, 0.64), basis="skeleton"),
    "lowerArmLength": Param("lowerArmLength", "torso", (0.28, 0.52), basis="skeleton"),
    "handLength": Param("hand", "torso", (0.1512, 0.352), basis="skeleton"),
    "footLength": Param("foot", "torso", (0.2352, 0.484), basis="skeleton"),
    "upperArmGirth": Param("armRadius", "torso", (0.256, 0.629), gain=UPPER_ARM_GIRTH_GAIN, bounds=(0.03, 0.14), basis="surface"),
    "wristGirth": Param("wristRadius", "torso", (0.16, 0.365), gain=6.694, bounds=(0.02, 0.08), basis="surface"),
    # The least neck radius the solve takes (0.03 T) measures 0.262 T on a neck long enough to clear the
    # trapezius (2026-11-08).
    "neckGirth": Param("neckRadius", "torso", (0.26, 1.03), gain=9.637, bounds=(0.03, 0.15), basis="surface"),
    # Round one thigh a quarter of the way from the crotch to the knee (sotaigen.form.thigh.stations). The
    # range is where the masses reach it on random bodies within the rules (none past 1.1 T); the shipped
    # bodies are 0.752-0.982 T. User's choice, 2026-11-12.
    "upperThighGirth": Param("thighRadius", "torso", (0.70, 1.00), gain=UPPER_THIGH_GIRTH_GAIN, bounds=(0.07, 0.2), basis="surface"),
    "kneeGirth": Param("kneeRadius", "torso", (0.242, 0.784), gain=KNEE_GIRTH_GAIN, bounds=(0.04, 0.16), basis="surface"),
    "calfGirth": Param("calfRadius", "torso", (0.362, 0.695), gain=5.942, bounds=(0.04, 0.18), basis="surface"),
    "ankleGirth": Param("ankleRadius", "torso", (0.2, 0.471), gain=6.411, bounds=(0.025, 0.12), basis="surface"),
    "legClearance": Param("stance", "torso", (0.01, 0.04), basis="control"),
    # Asked for on the surface and the controls are degrees: a line through the old bodies, to start
    # from. estimated_controls starts pelvisLean's tilt from a fit that reads the hip depth too.
    # The tilts' bounds are what the stress scan of docs/systems/params.md has run: they don't turn over,
    # they are just untried past here. The ranges are where random bodies within the rules mostly reach,
    # around every shipped body (0.0087-0.018 T, 0.0085-0.026 T); user's choice, 2026-11-12.
    "upperTrunkLean": Param(
        "ribTilt", "torso", (0.003, 0.03), offset=-0.65, gain=0.002, bounds=(-5.0, 22.4), degrees=True, basis="surface"
    ),
    "pelvisLean": Param(
        "pelvisTilt", "torso", (0.004, 0.022), offset=-3.4, gain=0.00096, bounds=(-5.0, 25.2), degrees=True, basis="surface"
    ),
    "headWidthRatio": Param("headWidthRatio", "one", (0.66, 0.94), basis="share"),
    "headDepth": Param("headDepth", "one", (0.74, 0.99), basis="control"),
    # The jaw's breadth over the head's, at the jaw's widest.
    "jawBreadthRatio": Param("jawBreadthRatio", "one", (0.604, 0.82), basis="share"),
    "definition": Param("definition", "one", (0.0, 0.5), basis="control"),
    "softness": Param("softness", "one", (0.9, 1.2), basis="control"),
    # File envelopes; the profile supplies tighter geometric limits.
    "outerThighArc": optional("outerThighArc", "torso", (-0.05, 0.10), "thigh", basis="chord", default=None),
    "abdomenFrontArc": optional("abdomenFrontArc", "torso", (-0.03, 0.02), "abdomen", basis="chord", default=None),
    "lumbarHollow": optional("lumbarHollow", "torso", (0.0, 0.06), "abdomen", basis="chord", default=None),
    "waistContourShift": optional("waistContourShift", "torso", (-0.06, 0.16), basis="control"),
    # How far the shoulder girdle and the arm stand behind where they do by default. Further back, the arc over
    # a raised arm's shoulder grows; bodies held to 0.12 T (docs/parts/shoulder-neck-set.md).
    "shoulderSetback": optional("shoulderSetback", "torso", (0.0, 0.10), basis="without"),
    "thighFrontShift": optional("thighFrontShift", "torso", (-0.03, 0.03), basis="without"),
    # The two flanks' ranges are the most any body reaches: a body with a flatter outline reaches less,
    # and the solve says how much.
    "gluteProjection": optional("gluteProjection", "torso", (-0.03, 0.03), "glute", basis="without"),
    "glutePeakShift": optional("glutePeakShift", "torso", (-0.042, 0.042), "glute", basis="without"),
    "gluteUpperFullness": optional("gluteUpperFullness", "torso", (-0.021, 0.017), "glute", basis="without"),
    "gluteLowerFullness": optional("gluteLowerFullness", "torso", (-0.02, 0.013), "glute", basis="without"),
    "outerHipShift": optional("outerHipShift", "torso", (-0.055, 0.055), "glute", basis="without"),
    "gluteFoldLength": optional("gluteFoldLength", "hip", (0.1675, 0.3275), "gluteFold", basis="shape", default=FOLD_SPAN),
    "gluteFoldDepth": optional("gluteFoldDepth", "torso", (0.0, 0.023), "gluteFold", basis="without"),
    # A body without these has the accepted groin, which is the top of each: only lighter is on offer.
    "groinDepth": optional("groinDepth", "torso", (0.0, 0.027), "groin", basis="without", default=GROIN_DEPTH),
    "groinRelief": optional("groinRelief", "torso", (0.0, 0.014), "groin", basis="without", default=GROIN_RELIEF),
    # Muscle masses a male body carries, in mm of mass added to its block (a study, not finished-surface
    # measurements; docs/parts/male-muscle.md). A body without them keeps its shape. They are left out of
    # the measured neutral surface, like the breast, so the solve doesn't shrink the ribs to undo them.
    "pectoralThickness": optional("pectoralThickness", "torso", (0.0, 0.05), "muscle", basis="control"),
    "deltoidThickness": optional("deltoidThickness", "torso", (0.0, 0.03), "muscle", basis="control"),
    "trapeziusThickness": optional("trapeziusThickness", "torso", (0.0, 0.03), "muscle", basis="control"),
    "latissimusThickness": optional("latissimusThickness", "torso", (0.0, 0.03), "muscle", basis="control"),
    # The breast as a thickness field on the chest wall (sotaigen/form/breast.py, docs/parts/chest.md).
    # Lengths in mm; the upper pole is a share of the footprint's height.
    "breastProjection": optional("breastProjection", "torso", (0.0, 0.14), "breast", basis="under"),
    "breastBaseBreadth": optional("breastBaseBreadth", "torso", (0.12, 0.32), "breast", basis="shape", default=0.1386),
    "breastFoldDrop": optional("breastFoldDrop", "torso", (0.2, 0.55), "breast", basis="shape", default=0.3027),
    "breastUpperPoleRatio": optional("breastUpperPoleRatio", "one", (0.3, 0.9), "breast", basis="share", default=0.45),
    # The two footprints may lap over the midline, so the skin between them lifts and they close up:
    # looked at to -30 mm on an adult body (docs/parts/chest.md).
    "breastGap": optional("breastGap", "torso", (-0.056, 0.2), "breast", basis="shape"),
    # The apex's place across the footprint, a share of its width from the inner edge; checked from the
    # middle out to 0.7 (docs/parts/chest.md).
    "breastApexRatio": optional("breastApexRatio", "one", (0.5, 0.7), "breast", basis="share", default=0.5),
}

def layer(*names):
    """The parameters of these local contours, in table order."""
    return tuple(key for key, row in ALL_SPEC.items() if row.layer in names)


ABDOMEN_CONTROLS = layer("abdomen")
THIGH_CONTROLS = layer("thigh")
GLUTE_CONTROLS = layer("glute")
GROIN_CONTROLS = layer("groin")
# What a breast can't be made without: a body that leaves them out has their defaults.
BREAST_FOOTPRINT = ("breastBaseBreadth", "breastFoldDrop", "breastUpperPoleRatio")
# What reshapes the calibrated surface where it stands. Calibration starts from a body without these.
CONTOURS = layer("glute", "gluteFold", "groin", "abdomen", "thigh")
# The contours bound to a reference skin on the same skeleton: sotaigen.form.anatomy.skin_reference undoes them.
PROFILES = layer("abdomen", "thigh")
# What a scale of the whole body changes, and what scales with the torso when the head count is set.
MILLIMETRES = tuple(key for key, row in ALL_SPEC.items() if row.unit != "one")
CHAIN = ("headHeight", "neckLength", "torsoLength", "crotchHeight")
# What is measured in shares of the torso, or of a parameter that is: these keep their share when only
# the torso's length changes (sotaigen.params.tools.with_torso_length).
TORSO_SHARES = tuple(key for key, row in ALL_SPEC.items() if row.unit in ("torso", "hip"))


def sex_table(without=(), ranges=None):
    """A sex's table: ALL_SPEC less `without`, with `ranges` {key: (low, high)} in place of its own."""
    ranges = ranges or {}
    unknown = (set(without) | ranges.keys()) - ALL_SPEC.keys()
    if unknown:
        raise KeyError(f"no such parameters: {sorted(unknown)}")
    return {
        key: row._replace(range=ranges.get(key, row.range))
        for key, row in ALL_SPEC.items()
        if key not in without
    }


# What each sex's bodies may hold, named as the folders of sotaigen/data/bodies/ are: every parameter but
# those it leaves out, each with the sex's own range where it has one. A range of its own comes
# from a stress test on that sex's bodies (docs/systems/params.md); until then it keeps the range in
# the table above, which the stress tests of female bodies found. So the male ranges are the female ones
# but where a male body was checked past them, and docs/PARAMETERS.md lists the ones known to stop a
# male body. Breasts are a female body's. The four muscles are both sexes': the muscles are the same
# in both, only their size differs.
MALE_RANGES = {
    # Built at 0.78-0.86 T on male-long, its chest from x1.1 down to the range's 0.435 T, 2026-11-07:
    # no flips, no collapse raising the arms, and the axilla, wrap and rig tests pass.
    "biacromialBreadth": (ALL_SPEC["biacromialBreadth"].range[0], 0.86),
    # The real rounded man's waist is 0.439 T deep (ANSUR II, 35 and over). Built with the ceiling lifted
    # to take it, 2026-11-07: no flips, 14 collapsed faces crouching, every test passing.
    "waistDepth": (ALL_SPEC["waistDepth"].range[0], 0.44),
}
SEXES = {
    "female": sex_table(),
    "male": sex_table(without=layer("breast"), ranges=MALE_RANGES),
}


def required(sex):
    """The parameters every body of a sex must have, in table order."""
    return tuple(key for key, row in SEXES[sex].items() if not row.optional)


def optional_parameters(sex):
    """The parameters a body of a sex may leave out, in table order."""
    return tuple(key for key, row in SEXES[sex].items() if row.optional)


def units(spec):
    """The lengths the controls are shares of."""
    head = spec["headHeight"]
    stature = head + spec["neckLength"] + spec["torsoLength"] + spec["crotchHeight"]
    return {
        "head": head,
        "stature": stature,
        "torso": spec["torsoLength"],
        "crotch": spec["crotchHeight"],
        "hip": spec["hipBreadth"],
        "one": 1.0,
    }


def unit_length(spec, unit):
    """One of `units`, from only the parameters it needs."""
    if unit == "stature":
        return stature(spec)
    if unit == "one":
        return 1.0
    return spec[{"head": "headHeight", "torso": "torsoLength", "crotch": "crotchHeight", "hip": "hipBreadth"}[unit]]


def default_of(spec, key):
    """What a body without optional parameter `key` has, in its unit on this body; None for its own shape."""
    row = ALL_SPEC[key]
    return None if row.default is None else row.default * unit_length(spec, row.unit)


def stature(spec):
    return units(spec)["stature"]


def heads(spec):
    """How many head heights make the stature. A result, never an input."""
    return stature(spec) / spec["headHeight"]


def proportions(spec):
    """Every length as a share of the torso, and the stature with them, which is what the rules read."""
    torso = spec["torsoLength"]
    q = {key: value / torso if key in MILLIMETRES else value for key, value in spec.items()}
    q["stature"] = units(spec)["stature"] / torso
    return q


def estimated_controls(spec):
    """The internal parameters by plain arithmetic: exact for lengths, near for measured targets."""
    u = units(spec)
    p = {}
    for key, value in spec.items():
        row = ALL_SPEC[key]
        p[row.control] = value / (row.gain * u[row.unit]) + row.offset
    del p["headShare"], p["torsoShare"]
    if "breastProjection" in spec:
        # A breast's footprint, where the body leaves it out (Param.default); the gap and the inner pole
        # the masses read as their defaults already.
        for key in BREAST_FOOTPRINT:
            p.setdefault(ALL_SPEC[key].control, ALL_SPEC[key].default)
    if "pelvisLean" in spec:
        # The pelvis turns further for the same lean on a deeper pelvis: a least-squares fit through
        # every body's solved tilt, female and male, typical and test, within 3.7 degrees of each and
        # 4.3 with each body left out of its own fit (25 bodies, refitted 2026-11-08 once the boys'
        # waist level came from boys' data and the older men's hips from real ratios). The fit before
        # read the lean and the hip width; it missed male-balanced by 7.6 degrees once that body's hips
        # narrowed on a deep pelvis. Adding the hip width, the waist width, the waist level or the
        # depth over the width fit as close or closer in-sample, but each did worse left out.
        lean = spec["pelvisLean"]
        p["pelvisTilt"] = 3786 * lean / u["torso"] - 1010 * lean / spec["hipDepth"] - 5.29
    if "neckGirth" in spec:
        # The girth is read just under the chin. Where the top of the trapezius's neck end
        # (sotaigen.form.anatomy.trapezius_masses) comes up near there, the section is mostly the trapezius's blend,
        # so the same girth takes a thinner neck. A fit through 137 solved bodies, random and from the
        # files, 24 of them under the threshold (2026-11-09): off by over 25% at 10 of them without it and
        # at 1 with it, a body just over the threshold (37%). Every typical body is over it, by 0.001 at
        # the least, so it starts where it did. It stops at the least radius the solve takes.
        clearance = spec["neckLength"] / u["torso"] - 0.055 - 0.145 * p["definition"]
        radius = p["neckRadius"]
        p["neckRadius"] = max(radius - 0.72 * max(0.015 - clearance, 0.0), min(radius, 0.03))
    p["stature"] = u["stature"]
    p["heads"] = u["stature"] / u["head"]
    return p
