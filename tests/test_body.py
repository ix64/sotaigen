"""What a body asks for is what the surface measures, for every body."""

import json

import numpy as np
import pytest

from sotaigen.form.anatomy import calibrated, landmarks, width_at
from sotaigen.form.bodies import load_bodies, load_specs
from sotaigen.form.field import SOFT_MIDLINE, Body, Ellipsoid, soft_fold
from sotaigen.form.wrap import hip_front_lead, march_many
from sotaigen.params.spec import ALL_SPEC, default_of, estimated_controls
from sotaigen.pipeline import BUILD, ensure_built, surface

PRESETS = load_bodies()


@pytest.mark.parametrize("offset", [0.0, 0.16])
def test_taking_the_clips_off_one_calibrated_field_leaves_the_next_whole(offset):
    p = PRESETS["female-long"] | {"waistContourShift": offset}
    points = np.random.default_rng(3).uniform((-500, -300, 0), (500, 300, 1800), (4000, 3))
    first, _ = calibrated(p)
    expected = first(points)
    # What wrap_body does to the field it gets.
    first.clips = []
    for group in first.groups:
        group.pop("clips", None)
    assert not np.array_equal(first(points), expected)
    second, _ = calibrated(p)
    assert np.array_equal(second(points), expected)


@pytest.mark.parametrize("name", PRESETS)
def test_waist_and_hip_land_on_their_parameters(name):
    p = PRESETS[name]
    body, L = surface(name)
    T, at = L["T"], L["at"]
    assert width_at(body, L["waist"]) == pytest.approx(p["waistBreadth"] * T, rel=0.02)
    hip = max(width_at(body, at(f)) for f in np.linspace(-0.05, 0.25, 13))
    assert hip == pytest.approx(p["hipBreadth"] * T, rel=0.02)


# The hip's front ahead of the chest's is reported, not held (docs/DECISIONS.md): a child's belly or an
# older woman's stands there, and bodies past it wrap and rig as well as any. What is held is that the
# report says what the surface does.
def test_wrap_reports_how_far_the_hip_front_leads_the_chest():
    name = next(iter(PRESETS))
    ensure_built(name, ("wrap",))
    reported = json.loads((BUILD / "bodies" / name / "wrap.json").read_text())["hipFrontLead"]
    assert reported == hip_front_lead(*surface(name))


@pytest.mark.parametrize("name", PRESETS)
def test_lower_abdomen_spans_the_fronts_of_the_thigh_roots(name):
    body, L = surface(name)
    height = L["at"](0.10)
    side = 0.15 * PRESETS[name]["hipBreadth"] * L["T"]
    starts = np.array([[0.0, 0.0, height], [side, 0.0, height]])
    assert np.all(body(starts) < 0)
    front = march_many(body, starts, np.tile([0.0, -1.0, 0.0], (2, 1)))[:, 1]
    assert front[0] < front[1] < 0


@pytest.mark.parametrize("name", PRESETS)
def test_field_is_mirror_symmetric(name):
    body, L = surface(name)
    # Over the whole height and width, so the chest's layers, the shoulders and the head count too.
    rng = np.random.default_rng(1)
    points = rng.uniform([-0.6 * L["T"], -0.4 * L["T"], 0.0], [0.6 * L["T"], 0.4 * L["T"], L["H"]], (2000, 3))
    assert np.allclose(body(points), body(points * [-1, 1, 1]))


@pytest.mark.parametrize("name", PRESETS)
def test_legs_part_at_the_crotch(name):
    body, L = surface(name)
    T = L["T"]
    above, below = L["crotch"] + 0.03 * T, L["crotch"] - 0.03 * T
    assert body(np.array([[0.0, 0.0, above]]))[0] < 0
    assert body(np.array([[0.0, 0.0, below]]))[0] > 0


def test_the_midline_fold_is_smooth_and_leaves_the_sides_alone():
    xs = np.linspace(-200.0, 200.0, 4001)
    folded = soft_fold(xs)
    assert np.allclose(folded, folded[::-1])
    assert np.gradient(folded, xs)[2000] == pytest.approx(0.0, abs=1e-3)
    far = np.abs(xs) > 6 * SOFT_MIDLINE
    assert np.allclose(folded[far], np.abs(xs[far]), atol=0.01)
    assert np.abs(folded - np.abs(xs)).max() < 0.3 * SOFT_MIDLINE


@pytest.mark.parametrize("name", PRESETS)
def test_the_wrist_is_centred_on_its_landmark(name):
    """A paired mass sits where its landmark says: the hand and the bones are placed from there."""
    body, L = surface(name)
    centre = L["wrist"] - L["foreDir"] * 0.02 * L["T"]
    ts = np.arange(0.0, 0.3 * L["T"], 0.25)
    across = np.array([L["foreDir"][2], 0.0, -L["foreDir"][0]])
    across /= np.linalg.norm(across)

    def reach(direction):
        values = body(centre + np.outer(ts, direction))
        return ts[np.nonzero(values > 0)[0][0]]

    assert reach(across) == pytest.approx(reach(-across), abs=1.0)


def test_a_tapered_mass_has_no_crease_past_its_top():
    """Sampled across the plane of the top, the field's slope must not jump."""
    mass = Ellipsoid((0.0, 0.0, 0.0), (60.0, 40.0, 100.0), taper=0.34)
    zs = np.linspace(95.0, 105.0, 201)
    points = np.column_stack((np.full_like(zs, 70.0), np.zeros_like(zs), zs))
    slope = np.gradient(mass(points), zs)
    assert np.abs(np.diff(slope)).max() < 0.01


@pytest.mark.parametrize("name", PRESETS)
def test_the_knee_is_centred_on_its_landmark_and_clear_of_the_midline(name):
    """The carve at the crotch only parts the legs there: the knee's own section stays whole."""
    body, L = surface(name)
    knee = L["kneeJoint"]
    xs = np.arange(0.0, 0.5 * L["T"], 0.25)
    inside = body(np.column_stack((xs, np.full_like(xs, knee[1]), np.full_like(xs, knee[2])))) <= 0
    found = xs[np.nonzero(inside)[0]]
    assert (found[0] + found[-1]) / 2 == pytest.approx(knee[0], abs=0.012 * L["T"])
    assert found[0] > 0.005 * L["T"]
    uncarved = Body(body.groups, body.clips)
    assert uncarved(np.array([[0.0, knee[1], knee[2]]]))[0] > 0


@pytest.mark.parametrize("name", PRESETS)
def test_calibration_leaves_the_leg_bones_alone(name):
    """The bones come from the uncalibrated landmarks, so calibrating must not move the femur."""
    _, L = surface(name)
    bare = landmarks(PRESETS[name])
    assert np.allclose(L["hipJoint"], bare["hipJoint"])
    assert np.allclose(L["kneeJoint"], bare["kneeJoint"])


@pytest.mark.parametrize("name", list(load_specs("typical")))
@pytest.mark.parametrize(
    ("knee_ratio", "proximal"), [(0.60, "hipJoint"), (0.54, "crotch"), (0.66, "crotch")]
)
def test_thigh_front_transfers_the_mid_thigh_outline_at_the_same_girth(name, knee_ratio, proximal):
    p = {**PRESETS[name], "kneeHeightRatio": knee_ratio}
    neutral = landmarks(p)
    hip, knee = neutral["hipJoint"], neutral["kneeJoint"]
    top = hip[2] if proximal == "hipJoint" else neutral["crotch"]
    height = (top + knee[2]) / 2
    u = (height - knee[2]) / (hip[2] - knee[2])
    center = knee + u * (hip - knee)
    angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    rays = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
    outlines = []
    # A fifth of the thigh's radius either way, in shares of the torso as the control is.
    for bias in (-0.2 * p["thighRadius"], 0.2 * p["thighRadius"]):
        body, L = calibrated({**p, "thighFrontShift": bias})
        for joint in ("hipJoint", "kneeJoint", "ankleJoint"):
            np.testing.assert_array_equal(L[joint], neutral[joint])
        ring = march_many(body, np.tile(center, (len(rays), 1)), rays)
        assert np.all(np.linalg.norm(ring - center, axis=1) > 0)
        girth = np.linalg.norm(ring - np.roll(ring, 1, axis=0), axis=1).sum()
        outlines.append((girth, ring[:, 1].min(), ring[:, 1].max()))
    rear, front = outlines
    shift = 0.4 * p["thighRadius"] * neutral["T"] * (4 * u * (1 - u)) ** 2
    assert front[0] == pytest.approx(rear[0], rel=0.01)
    assert front[1] < rear[1] - 0.75 * shift
    assert front[2] < rear[2] - 0.75 * shift


@pytest.mark.parametrize("name", PRESETS)
def test_the_calf_peaks_higher_outside_than_inside(name):
    """Both outlines of the lower leg swell, but apart: the outer head sits above the inner one."""
    body, L = surface(name)
    knee, ankle = L["kneeJoint"], L["ankleJoint"]
    fractions = np.linspace(0.1, 0.7, 61)
    xs = np.arange(0.0, 0.4 * L["T"], 0.1)
    inner, outer = [], []
    for f in fractions:
        centre = knee + (ankle - knee) * f
        points = np.column_stack((xs, np.full_like(xs, centre[1]), np.full_like(xs, centre[2])))
        found = xs[np.nonzero(body(points) <= 0)[0]]
        inner.append(centre[0] - found[0])
        outer.append(found[-1] - centre[0])
    lean = np.array(outer) - np.array(inner)
    assert np.interp(0.25, fractions, lean) > np.interp(0.5, fractions, lean)


@pytest.mark.parametrize("name", list(load_specs("typical")))
def test_glute_controls_are_distances_keep_the_hip_width_and_shape_separate_flanks(name):
    from sotaigen.params.rules import RANGES, sex_of
    from sotaigen.params.spec import GLUTE_CONTROLS

    p = PRESETS[name]
    neutral, L = calibrated(p)
    T = L["T"]
    # Use the same posterior ray in every comparison. The baseline peak is measured
    # from the fused body, where the pelvis and thigh can shift it above the glute.
    upper, _ = calibrated(p | {"gluteUpperFullness": 0.005})
    lower, _ = calibrated(p | {"gluteLowerFullness": 0.004})
    levels = upper.peak + np.array([-0.10, 0.0, 0.12])
    centers = np.column_stack((np.full(3, 0.11 * T), np.zeros(3), L["at"](levels)))
    rays = np.tile([0, 1, 0], (3, 1))

    def depths(body):
        return march_many(body, centers, rays)[:, 1]

    rest = depths(neutral)
    u, l = depths(upper), depths(lower)
    # What is asked for, at its own station, and nothing at the others. A flat outline can't give what it
    # hasn't got: the strength stops at one and the field says how far it got (flank_strength). So the
    # surface moves by what is reported, and what is reported is what was asked short of that stop. The
    # young boys' buttocks are flat enough to stop there (docs/systems/presets.md).
    for strength, reached, asked, moved in (
        (upper.upper, upper.upper_reached, 0.005, u[2] - rest[2]),
        (lower.lower, lower.lower_reached, 0.004, l[0] - rest[0]),
    ):
        assert moved == pytest.approx(reached * T, abs=0.1)
        if strength < 1.0:
            assert reached * T == pytest.approx(asked * T, abs=0.1)
        else:
            assert reached < asked
    np.testing.assert_allclose(u[:2], rest[:2], atol=0.002)
    np.testing.assert_allclose(l[1:], rest[1:], atol=0.002)
    for key in GLUTE_CONTROLS:
        low, _ = calibrated(p | {key: RANGES[sex_of(name)][key][0]})
        high, _ = calibrated(p | {key: RANGES[sex_of(name)][key][1]})
        for body in (low, high):
            width = max(width_at(body, L["at"](f)) for f in np.linspace(-0.05, 0.25, 13))
            assert width == pytest.approx(p["hipBreadth"] * T, rel=0.02)
        if key == "gluteProjection":
            assert depths(high)[1] - depths(low)[1] == pytest.approx(0.06 * T, abs=0.01)


def test_glute_coordinate_inverse_at_all_slider_corners():
    from itertools import product

    from sotaigen.form.glutes import GluteContours
    from sotaigen.params.rules import RANGES
    from sotaigen.params.spec import GLUTE_CONTROLS

    p = PRESETS["female-long"]
    body, L = calibrated(p)
    points = np.random.default_rng(5402).uniform(
        [-0.40, -0.20, -0.25], [0.40, 0.40, 0.47], (2048, 3)
    ) * L["T"]
    points[:, 2] += L["crotch"]
    for ends in product((0, 1), repeat=len(GLUTE_CONTROLS)):
        corner = {key: RANGES["female"][key][end] for key, end in zip(GLUTE_CONTROLS, ends, strict=True)}
        shape = GluteContours(body, L, p | corner)
        placed = shape.from_reference(points)
        np.testing.assert_array_equal(placed[:, 0], points[:, 0])
        np.testing.assert_allclose(shape.to_reference(placed), points, atol=1e-6, rtol=0)


def test_a_fold_length_is_the_span_between_its_half_depths():
    from sotaigen.form.field import ramp
    from sotaigen.form.glutes import GluteContours

    p = PRESETS["female-long"]
    body, L = calibrated(p)
    half = 0.5 * p["hipBreadth"] * L["T"]
    for span in (0.1675, 0.3275):
        shape = GluteContours(body, L, p | {"gluteFoldDepth": 0.02, "gluteFoldLength": span})
        x = np.linspace(0, 1.2, 4801)
        across = ramp((x - 0.025) / 0.15) * ramp((shape.fold_end - x) / 0.17)
        reached = x[across >= 0.5]
        assert (reached[-1] - reached[0]) * half == pytest.approx(span * p["hipBreadth"] * L["T"], abs=0.2)


def test_fold_length_is_independent_of_medial_depth_and_the_lower_hip_outline():
    p = PRESETS["female-long"]
    neutral, L = calibrated(p)
    T = L["T"]
    half = 0.5 * p["hipBreadth"] * T
    x = np.array([0.28, 0.68, 0.28, 0.28])
    # Probe heights use a reference fold level of -0.029 T. Shift them with the body's
    # glute peak so the test compares the same places relative to its fold.
    fold, _ = calibrated(p | {"gluteFoldLength": 0.3275, "gluteFoldDepth": 0.023})
    shift = fold.fold_level + 0.029
    points = np.column_stack((x * half, np.array([0.20, 0.20, 0.24, -0.13]) * T,
                              L["at"](np.array([-0.038, -0.05, 0.05, -0.038]) + shift)))
    inactive, _ = calibrated(p | {"gluteFoldLength": 0.3275})
    np.testing.assert_array_equal(inactive(points), neutral(points))
    for depth in (0.0069, 0.023):
        cases = []
        for length in (0.1675, 0.3275):
            shape, _ = calibrated(p | {"gluteFoldLength": length, "gluteFoldDepth": depth})
            placed = shape.from_reference(points)
            np.testing.assert_allclose(placed[:, [0, 2]], points[:, [0, 2]], atol=1e-7, rtol=0)
            np.testing.assert_allclose(placed[2:], points[2:], atol=1e-7, rtol=0)
            np.testing.assert_allclose(shape(placed), neutral(points), atol=1e-7, rtol=0)
            cases.append(points[:, 1] - placed[:, 1])
        short, long = cases
        assert short[0] == pytest.approx(depth * T, rel=0.01)
        assert long[0] == pytest.approx(short[0], abs=1e-7)
        assert short[1] == pytest.approx(0, abs=1e-7)
        assert long[1] > short[0] * 0.9


def test_tall_body_fold_follows_the_posterior_thigh_transition():
    p = PRESETS["female-balanced"]
    neutral, L = calibrated(p)
    folded, _ = calibrated(p | {"gluteFoldDepth": 0.015, "gluteFoldLength": 0.2875})
    levels = np.arange(-0.14, 0.05, 0.004)
    half_width = 0.5 * p["hipBreadth"] * L["T"]
    x, z = np.meshgrid(np.array([0.25, 0.35, 0.45]) * half_width, L["at"](levels))
    points = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    directions = np.tile([0, 1, 0], (x.size, 1))

    def rear_profile(surface):
        rear = march_many(surface, points, directions)[:, 1]
        return rear.reshape(x.shape).mean(axis=1)

    smooth = rear_profile(neutral)
    recess = smooth - rear_profile(folded)
    # Compare the added fold with the broadest slope of the untouched skin at the same sites.
    transition = levels[2:-2][(smooth[4:] - smooth[:-4]).argmax()]
    assert recess.max() > 0.01 * L["T"]
    assert abs(levels[recess.argmax()] - transition) < 0.015


# The fold used to stay below the lower-fullness section (peak - 0.10). On the typical bodies the
# thigh transition is higher than that allows, and the user chose the fold at the transition
# (2026-11-06), so the fold now reaches into that section, up to 63% of its depth. What still holds
# is that it stays under its cap and leaves the buttock's prominence where it was.
@pytest.mark.parametrize("name", PRESETS)
@pytest.mark.parametrize("end", (0, 1))
def test_fold_stays_under_its_cap_and_off_the_prominence(name, end):
    from sotaigen.params.rules import RANGES, sex_of
    from sotaigen.params.spec import GLUTE_CONTROLS

    p = PRESETS[name] | {key: RANGES[sex_of(name)][key][end] for key in GLUTE_CONTROLS}
    neutral, L = calibrated(p | {"gluteFoldDepth": 0})
    folded, _ = calibrated(p | {"gluteFoldDepth": 0.023, "gluteFoldLength": 0.3275})
    assert folded.fold_level <= -0.029
    x = np.linspace(0.1, 0.8, 18) * 0.5 * p["hipBreadth"] * L["T"]
    points = np.column_stack((x, np.zeros_like(x), np.full_like(x, L["at"](folded.peak))))
    directions = np.tile([0, 1, 0], (len(x), 1))
    np.testing.assert_allclose(
        march_many(folded, points, directions), march_many(neutral, points, directions), atol=0.002
    )


@pytest.mark.parametrize("name", PRESETS)
@pytest.mark.parametrize("offset", (-0.06, 0.16))
def test_waist_height_preserves_width_targets_and_bone_landmarks(name, offset):
    p = PRESETS[name]
    field, L = calibrated(p | {"waistContourShift": offset})
    neutral = landmarks(p)
    assert L["waist"] == pytest.approx(neutral["waist"] + offset * L["T"])
    assert landmarks(p | {"waistContourShift": offset})["waist"] == neutral["waist"]
    assert width_at(field, L["waist"]) == pytest.approx(p["waistBreadth"] * L["T"], rel=0.02)
    hip = max(width_at(field, L["at"](f)) for f in np.linspace(-0.05, 0.25, 13))
    assert hip == pytest.approx(p["hipBreadth"] * L["T"], rel=0.02)
    for key in ("hipJoint", "kneeJoint", "ankleJoint", "shoulderJoint", "elbow", "wrist"):
        np.testing.assert_array_equal(L[key], neutral[key])


def test_missing_and_zero_waist_offset_keep_the_same_field():
    p = PRESETS["female-long"]
    before, L = calibrated(p)
    after, _ = calibrated(p | {"waistContourShift": 0})
    points = np.random.default_rng(19).uniform([-200, -150, 0], [200, 150, L["H"]], (800, 3))
    np.testing.assert_array_equal(before(points), after(points))


@pytest.mark.parametrize("name", ["female-chibi", "female-long", "female-balanced"])
@pytest.mark.parametrize("offset", [0.0, 0.16])
def test_a_vanishing_breast_does_not_leave_a_shelf_on_the_chest(name, offset):
    p = {key: value for key, value in PRESETS[name].items() if not key.startswith("breast")}
    p |= {"waistContourShift": offset}
    # The footprint of a typical adult breast, in shares of the torso, at next to no projection.
    footprint = {"breastBaseBreadth": 0.234, "breastFoldDrop": 0.404, "breastUpperPoleRatio": 0.45, "breastGap": 0.035}
    bare, L = calibrated(p)
    tiny, _ = calibrated(p | footprint | {"breastProjection": 1e-5})
    x, z = np.meshgrid(np.linspace(0, 0.16, 5) * L["T"], L["at"](np.linspace(0.5, 0.85, 15)))
    starts = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    starts = starts[bare(starts) < 0]
    rays = np.tile([0.0, -1.0, 0.0], (len(starts), 1))
    # Tissue under 0.01 mm thick must not add a pedestal around the footprint.
    np.testing.assert_allclose(
        march_many(tiny, starts, rays), march_many(bare, starts, rays), atol=0.02, rtol=0
    )


DEFAULTED = [key for key, row in ALL_SPEC.items() if row.optional and row.default is not None and row.layer != "groin"]


@pytest.mark.parametrize("key", DEFAULTED)
def test_a_body_without_an_optional_parameter_is_the_body_at_its_default(key):
    # Param.default is what the masses do without the parameter: the public table and the page say so.
    spec = dict(load_specs()["female-long"])
    # What the parameter only acts with: a fold's span needs its depth; the breast's shape, its
    # projection, which this body has.
    spec |= {"gluteFoldDepth": 5.0} if key == "gluteFoldLength" else {}
    spec = {k: v for k, v in spec.items() if k != key}
    points = np.random.default_rng(23).uniform([-250, -200, 0], [250, 200, 1700], (1500, 3))
    without, _ = calibrated(estimated_controls(spec))
    at_default, _ = calibrated(estimated_controls(spec | {key: default_of(spec, key)}))
    np.testing.assert_allclose(at_default(points), without(points), atol=1e-9, rtol=0)


def test_the_groin_defaults_are_the_accepted_shape():
    from sotaigen.form import groin

    assert (ALL_SPEC["groinDepth"].default, ALL_SPEC["groinRelief"].default) == (groin.DEPTH, groin.RELIEF)
