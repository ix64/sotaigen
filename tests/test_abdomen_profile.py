"""Measured controls must reach their targets without moving the opposite surface or anchors."""

import numpy as np
import pytest

from sotaigen.form.abdomen import AbdominalProfile, profile_measures, sagittal_depths
from sotaigen.form.anatomy import calibrated, width_at
from sotaigen.form.bodies import load_bodies


@pytest.mark.parametrize("name", ["female-chibi", "female-long", "female-balanced", "female-slim"])
@pytest.mark.parametrize("front,back", [(-0.03, 0.0), (0.02, 0.06)])
def test_absolute_targets_and_fixed_anchor_sections(name, front, back):
    p = load_bodies()[name]
    body, L = calibrated(p)
    shape = AbdominalProfile(body, L, p, front_arc=front, back_hollow=back)
    result = profile_measures(shape, L)
    assert result["frontArc"] == pytest.approx(front, abs=2e-8)
    assert result["backHollow"] == pytest.approx(back, abs=2e-8)
    np.testing.assert_array_equal(
        sagittal_depths(shape, L)[:, [0, 2]], sagittal_depths(body, L)[:, [0, 2]]
    )
    assert width_at(shape, L["waist"]) == width_at(body, L["waist"])


@pytest.mark.parametrize("side", ["front_arc", "back_hollow"])
def test_opposite_surface_and_outside_band_are_unchanged(side):
    p = load_bodies()["female-long"]
    body, L = calibrated(p)
    shape = AbdominalProfile(body, L, p, **{side: 0.0})
    points = np.random.default_rng(14).uniform(
        [-100, -150, L["at"](0.1)], [100, 150, L["at"](0.9)], (1200, 3)
    )
    sign = points[:, 1] >= 0 if side == "front_arc" else points[:, 1] <= 0
    outside = (points[:, 2] <= L["at"](0.25)) | (points[:, 2] >= L["at"](0.70))
    np.testing.assert_array_equal(
        shape.to_reference(points[sign | outside]), points[sign | outside]
    )
    moved = shape.from_reference(points)
    np.testing.assert_allclose(shape.to_reference(moved), points, atol=1e-7, rtol=0)
    np.testing.assert_array_equal(moved[:, [0, 2]], points[:, [0, 2]])
    np.testing.assert_allclose(shape(moved), body(points), atol=1e-7, rtol=0)


@pytest.mark.parametrize("offset", [-0.06, 0, 0.16])
def test_public_targets_apply_after_waist_height_and_breast(offset):
    p = load_bodies()["female-balanced"] | {
        "waistContourShift": offset,
        "abdomenFrontArc": 0.0,
        "lumbarHollow": 0.04,
    }
    body, L = calibrated(p)
    measured = profile_measures(body, L)
    assert measured["frontArc"] == pytest.approx(0, abs=2e-8)
    assert measured["backHollow"] == pytest.approx(0.04, abs=2e-8)


def test_explicit_zero_means_straight_not_unchanged():
    p = load_bodies()["female-long"]
    before, L = calibrated(p)
    after, _ = calibrated(p | {"abdomenFrontArc": 0, "lumbarHollow": 0})
    assert abs(profile_measures(before, L)["backHollow"]) > 0.02
    assert profile_measures(after, L)["backHollow"] == pytest.approx(0, abs=2e-8)


def test_unclipping_a_profile_reaches_the_wrapped_field_without_leaking_to_the_next_call():
    from sotaigen.form.head import _unclipped_body

    p = load_bodies()["female-long"] | {"lumbarHollow": 0.04}
    opened = _unclipped_body(p)
    assert opened.clips == opened.reference.clips == []
    fresh, _ = calibrated(p)
    assert fresh.clips


@pytest.mark.slow
def test_delivered_profile_keeps_reference_binding_and_reaches_mesh_targets():
    from sotaigen.form import groin
    from sotaigen.form.abdomen import mesh_profile_measures
    from sotaigen.form.anatomy import landmarks, skin_reference
    from sotaigen.form.bodies import load_specs
    from sotaigen.form.solve import solved_controls
    from sotaigen.generation import full_template, made
    from sotaigen.mesh import triangulate
    from sotaigen.metrics import RestMesh, pose_row, posed_meshes
    from sotaigen.rig.poses import check_poses

    spec = load_specs()["female-balanced"]
    p = solved_controls(spec)
    original = made(p)
    torso = spec["torsoLength"]
    adjusted = solved_controls(spec | {"abdomenFrontArc": -0.03 * torso, "lumbarHollow": 0.06 * torso})
    actual = made(adjusted)
    base = full_template()
    # made delivers the sculpted rest skin. The profile's inverse and the rig are defined on
    # the reference skin, and the sculpture depends on Y, so the two maps don't commute: compare
    # reference skins.
    original_reference = groin.reference_verts(base, original.verts, landmarks(p))
    actual_reference = groin.reference_verts(base, actual.verts, landmarks(adjusted))
    np.testing.assert_allclose(
        skin_reference(adjusted, actual_reference), original_reference, atol=1e-7, rtol=0
    )
    np.testing.assert_allclose(actual.weights, original.weights, atol=1e-7, rtol=0)
    _, L = calibrated(adjusted)
    triangles = triangulate(base["faces"], actual.verts)
    measured = mesh_profile_measures(actual.verts, triangles, L)
    assert abs(measured["frontArc"] + 0.03) * L["T"] < 1.0
    assert abs(measured["backHollow"] - 0.06) * L["T"] < 1.0
    counts = []
    for exact, params in ((original, p), (actual, adjusted)):
        poses = check_poses(base, exact.verts, exact.bones, exact.weights)
        rest = RestMesh(exact.verts, base["faces"])
        counts.append(
            {
                key: pose_row(rest, posed)["collapsedFaces"]
                for key, posed in posed_meshes(
                    base,
                    exact.verts,
                    exact.bones,
                    exact.weights,
                    {k: poses[k] for k in ("crouch", "twist", "reach", "stride")},
                    calibrated(params)[0],
                )
            }
        )
    assert all(counts[1][key] <= value for key, value in counts[0].items())


@pytest.mark.slow
@pytest.mark.parametrize("name", ["female-chibi", "female-long", "female-balanced", "female-slim"])
@pytest.mark.parametrize("arc", [-0.03, 0.0, 0.02])
def test_the_finished_skin_measures_the_front_arc_asked_for(name, arc):
    from sotaigen.form.abdomen import mesh_profile_measures
    from sotaigen.form.bodies import load_specs
    from sotaigen.form.solve import solved_controls
    from sotaigen.generation import full_template, made
    from sotaigen.mesh import triangulate

    spec = load_specs()[name]
    p = solved_controls(spec | {"abdomenFrontArc": arc * spec["torsoLength"]})
    verts = made(p).verts
    _, L = calibrated(p)
    measured = mesh_profile_measures(verts, triangulate(full_template()["faces"], verts), L)
    assert abs(measured["frontArc"] - arc) * L["T"] < 0.3


def test_a_body_that_asks_for_no_front_arc_is_left_alone():
    from sotaigen.form.bodies import load_specs
    from sotaigen.form.solve import solved_controls

    assert "abdomenFrontArc" not in solved_controls(load_specs()["female-long"])


@pytest.mark.slow
@pytest.mark.parametrize("name", ["female-chibi", "female-long", "female-balanced", "female-slim"])
def test_the_slider_start_is_what_the_untouched_finished_skin_measures(name):
    from sotaigen.form.abdomen import mesh_profile_measures
    from sotaigen.form.solve import delivered_front_arc
    from sotaigen.generation import full_template, made
    from sotaigen.mesh import triangulate

    p = load_bodies()[name]
    verts = made(p).verts
    _, L = calibrated(p)
    measured = mesh_profile_measures(verts, triangulate(full_template()["faces"], verts), L)
    assert abs(measured["frontArc"] - delivered_front_arc(p)) * L["T"] < 0.3
