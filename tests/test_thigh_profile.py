"""The thigh's outer arc preserves the skeleton and the girth, and reaches its measured target."""

import numpy as np
import pytest

from sotaigen.form.anatomy import calibrated, landmarks, skin_reference
from sotaigen.form.bodies import load_bodies, load_specs
from sotaigen.form.solve import solved_controls, target_misses
from sotaigen.form.thigh import ThighProfile, field_sections, mesh_measures, section_measures


@pytest.mark.parametrize("name", ["female-chibi", "female-long", "female-balanced", "female-slim"])
def test_field_targets_anchor_sections_and_reference_inverse(name):
    p = load_bodies()[name]
    reference, L = calibrated(p)
    start = section_measures(field_sections(reference, L), L)
    shape = ThighProfile(reference, L, arc=start["outerArc"] + 0.004)
    measured = section_measures(field_sections(shape, L), L)
    assert measured["outerArc"] == pytest.approx(shape.arc_target, abs=0.0001)
    # The girth is the masses' own, which upperThighGirth solves for: the arc shares it out.
    assert measured["thighGirth"] == pytest.approx(start["thighGirth"], abs=0.0002)
    points = np.random.default_rng(83).uniform(
        [-150, -150, L["knee"]], [150, 150, L["waist"]], (600, 3)
    )
    placed = shape.from_reference(points)
    np.testing.assert_allclose(shape.to_reference(placed), points, atol=1e-7, rtol=0)
    np.testing.assert_array_equal(placed[:, 2], points[:, 2])
    outside = (points[:, 2] <= shape.low) | (points[:, 2] >= shape.high)
    np.testing.assert_array_equal(placed[outside], points[outside])
    # The retained section endpoints themselves are exact fixed points.
    for curve in shape.curves[::2]:
        np.testing.assert_array_equal(shape.from_reference(curve), curve)


def test_explicit_zero_missing_targets_and_unreachable_target():
    p = load_bodies()["female-long"]
    reference, L = calibrated(p)
    missing = ThighProfile(reference, L)
    assert missing.a == missing.b == 0
    shape = ThighProfile(reference, L, arc=0)
    assert abs(section_measures(field_sections(shape, L), L)["outerArc"]) < 0.0001
    assert shape.girth_target == shape.origin["thighGirth"]
    # Past what this body can reach, it goes to the nearer end and says what it measures there.
    beyond = ThighProfile(reference, L, arc=0.2)
    reached = beyond.reached["outerThighArc"]
    assert reached < 0.2
    assert section_measures(field_sections(beyond, L), L)["outerArc"] == pytest.approx(reached, abs=1e-4)
    assert "upperThighGirth" not in beyond.reached


@pytest.mark.slow
def test_delivered_mesh_keeps_reference_bones_weights_and_matches_measurements():
    from sotaigen.corrective.crossings import SkinParts
    from sotaigen.form import groin
    from sotaigen.generation import full_template, made, wrap_failures
    from sotaigen.mesh import triangulate
    from sotaigen.metrics import RestMesh, pose_row
    from sotaigen.rig.poses import check_poses
    from sotaigen.rig.skinning import pose_verts

    p = load_bodies()["female-long"] | {
        "abdomenFrontArc": 0.01,
        "lumbarHollow": 0.04,
        "outerHipShift": -0.022,
    }
    reference, L = calibrated(p)
    start = section_measures(field_sections(reference, L), L)
    q = p | {"outerThighArc": start["outerArc"] + 0.006}
    before, after = made(p), made(q)
    assert not wrap_failures(after)
    for a, b in zip(before.bones, after.bones, strict=True):
        np.testing.assert_allclose(a["head"], b["head"], atol=1e-7, rtol=0)
        np.testing.assert_allclose(a["tail"], b["tail"], atol=1e-7, rtol=0)
    np.testing.assert_allclose(after.weights, before.weights, atol=1e-7, rtol=0)
    base = full_template()
    # made delivers the sculpted rest skin. The profile's inverse and the rig are defined on
    # the reference skin, and the sculpture depends on Y, so the two maps don't commute: compare
    # reference skins. The girth stays the masses', read on the reference, as the sculpture adds about
    # 3.7 mm to it; the sculpture leaves the outer arc alone.
    before_reference = groin.reference_verts(base, before.verts, landmarks(p))
    after_reference = groin.reference_verts(base, after.verts, landmarks(q))
    np.testing.assert_allclose(
        skin_reference(q, after_reference), skin_reference(p, before_reference), atol=1e-7, rtol=0
    )
    tri = triangulate(base["faces"], after.verts)
    actual = mesh_measures(after.verts, tri, L)
    assert abs(actual["outerArc"] - q["outerThighArc"]) * L["T"] < 0.6
    solved = mesh_measures(after_reference, triangulate(base["faces"], after_reference), L)
    assert abs(solved["thighGirth"] - start["thighGirth"]) * L["T"] < 2.2
    rows = []
    for exact in (before, after):
        poses = check_poses(base, exact.verts, exact.bones, exact.weights)
        rest = RestMesh(exact.verts, base["faces"])
        parts = SkinParts(base)
        counts = {}
        for name in ("crouch", "stride"):
            v = pose_verts(exact.bones, exact.weights, exact.verts, poses[name])
            assert parts.legs_past_midline(v) == 0
            counts[name] = pose_row(rest, v)["collapsedFaces"]
        rows.append(counts)
    assert all(rows[1][key] <= rows[0][key] for key in rows[0])


def test_profile_limits_are_conditional_and_have_no_silent_clamp():
    p = load_bodies()["female-balanced"]
    body, L = calibrated(p)
    profile = ThighProfile(body, L)
    limits = profile.limits()
    assert list(limits) == ["outerArc"]
    for arc in limits["outerArc"]:
        shape = ThighProfile(body, L, arc=arc)
        assert shape.arc_target == arc and not shape.reached
    past = ThighProfile(body, L, arc=limits["outerArc"][1] + 0.001)
    assert past.reached["outerThighArc"] == pytest.approx(limits["outerArc"][1], abs=1e-6)
    assert limits["outerArc"][1] < profile._arc(1.5)


@pytest.mark.parametrize("name", ["female-chibi", "female-long", "female-balanced", "female-slim"])
def test_arc_redistribution_preserves_anterior_surface_at_fixed_girth(name):
    body, L = calibrated(load_bodies()[name])
    start = ThighProfile(body, L)
    arc = start.arc_target + 0.8 * (start.limits()["outerArc"][1] - start.arc_target)
    changed = ThighProfile(body, L, arc=arc)
    points = np.concatenate(start.curves)
    anterior = points[:, 1] <= start._axis(points[:, 2])
    before, after = start.from_reference(points), changed.from_reference(points)
    np.testing.assert_allclose(after[anterior, 1], before[anterior, 1], atol=1e-8, rtol=0)
    np.testing.assert_allclose(changed.to_reference(after), points, atol=1e-7, rtol=0)
    measured = section_measures(field_sections(changed, L), L)
    assert measured["outerArc"] == pytest.approx(arc, abs=0.0001)
    assert measured["thighGirth"] == pytest.approx(start.girth_target, abs=0.0002)


def test_an_arc_past_what_the_body_reaches_is_reached_and_reported():
    # The file's range is wider than any body's reach (docs/PARAMETERS.md): the body is made with what
    # the profile reaches, and the miss is a note, never a refusal.
    spec = load_specs()["female-long"]
    asked = 0.10 * spec["torsoLength"]
    misses = target_misses(spec | {"outerThighArc": asked})
    wanted, reached = misses["outerThighArc"]
    assert wanted == asked and reached < asked - 10


def test_the_upper_thigh_girth_is_solved_and_moves_the_masses():
    spec = load_specs()["female-long"]
    thicker = spec | {"upperThighGirth": spec["upperThighGirth"] + 20.0}
    assert not target_misses(thicker)
    assert solved_controls(thicker)["thighRadius"] > solved_controls(spec)["thighRadius"]
