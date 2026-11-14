"""What the wrap promises in wrap.json holds in every body."""

import json
from functools import cache

import numpy as np
import pytest

from sotaigen.form.anatomy import landmarks
from sotaigen.form.bodies import load_bodies, load_specs
from sotaigen.form.head import template_rows
from sotaigen.form.wrap import (
    FOOT_TO_GROUND_MM,
    SEAM_CREASE_DEG,
    SEAM_MISS_SHARE,
    attach_extremities,
    march_many,
    quad_normals,
    seam_radius,
    seat_neck_ring,
    smooth_correction,
    wrap_body,
)
from sotaigen.mesh import face_normals
from sotaigen.params.rules import seam_problems, sex_of
from sotaigen.params.spec import MILLIMETRES
from sotaigen.pipeline import BUILD, ensure_built
from sotaigen.template import load_template, template_landmarks


def wrap_report(name):
    ensure_built(name, ("wrap",))
    return json.loads((BUILD / "bodies" / name / "wrap.json").read_text())


@pytest.mark.parametrize("name", load_bodies())
def test_no_face_is_flipped(name):
    assert wrap_report(name)["flippedFaces"] == 0


def test_a_short_body_with_a_fuller_chest_finishes_untangling_the_underarms(tmp_path):
    # With six local relaxations two upper-arm quads still face into the surface.
    breast = {"breastProjection": 0.06, "breastBaseBreadth": 0.26, "breastFoldDrop": 0.40, "breastUpperPoleRatio": 0.45, "breastGap": 0.04}
    p = load_bodies()["female-chibi"] | breast
    report = wrap_body(load_template(), p, tmp_path)
    assert report["flippedFaces"] == report["crossedMidline"] == 0


@pytest.mark.parametrize("name", load_bodies())
def test_no_vertex_crosses_the_midline(name):
    assert wrap_report(name)["crossedMidline"] == 0


def off_midline_neck_vertex(base):
    x = np.array(base["verts"])[:, 0]
    return np.nonzero((np.array(base["regions"]) == "neck") & (x > 10.0))[0][0]


def test_the_report_counts_what_seating_the_neck_ring_breaks(tmp_path, monkeypatch):
    base = load_template()
    vertex = off_midline_neck_vertex(base)

    def seat_then_break(base, body, head, points, L):
        seat_neck_ring(base, body, head, points, L)
        points[vertex, 0] *= -1

    monkeypatch.setattr("sotaigen.form.wrap.seat_neck_ring", seat_then_break)
    report = wrap_body(base, load_bodies()["female-balanced"], tmp_path)
    assert report["crossedMidline"] == 1
    assert report["flippedFaces"] > 0
    assert report["brokenAt"]["crossedMidline"] == {"neck": 1}
    assert sum(report["brokenAt"]["flippedFaces"].values()) == report["flippedFaces"]


def test_the_report_does_not_count_what_seating_the_neck_ring_mends(tmp_path, monkeypatch):
    base = load_template()
    vertex = off_midline_neck_vertex(base)

    def attach_broken(base, facts, L, keep, projected):
        full, seam = attach_extremities(base, facts, L, keep, projected)
        full[vertex, 0] *= -1
        return full, seam

    def mend_then_seat(base, body, head, points, L):
        points[vertex, 0] *= -1
        seat_neck_ring(base, body, head, points, L)

    monkeypatch.setattr("sotaigen.form.wrap.attach_extremities", attach_broken)
    monkeypatch.setattr("sotaigen.form.wrap.seat_neck_ring", mend_then_seat)
    report = wrap_body(base, load_bodies()["female-balanced"], tmp_path)
    assert report["flippedFaces"] == report["crossedMidline"] == 0
    assert "brokenAt" not in report


@pytest.mark.parametrize("name", load_bodies())
def test_the_feet_stand_on_the_ground(name):
    assert abs(wrap_report(name)["footLowest"]) <= FOOT_TO_GROUND_MM
    # The finished mesh agrees: nothing that came after the measurement sank below the feet.
    lowest = np.load(BUILD / "bodies" / name / "wrapped.npy")[:, 2].min()
    assert abs(lowest) <= FOOT_TO_GROUND_MM


def miss_share(report, points, part):
    """How far a seam misses its limb, as a share of the limb's radius there."""
    return report[f"{part}SeamMissMm"] / seam_radius(load_template(), points, part)


@pytest.mark.parametrize("name", load_bodies())
@pytest.mark.parametrize("part", SEAM_CREASE_DEG)
def test_hands_and_feet_join_the_limb_without_a_crease(name, part):
    report = wrap_report(name)
    assert report[f"{part}SeamCreaseDeg"] <= SEAM_CREASE_DEG[part]
    points = np.load(BUILD / "bodies" / name / "wrapped.npy")
    assert miss_share(report, points, part) <= SEAM_MISS_SHARE[part]


@pytest.mark.parametrize("ankle_radius", [0.045, 0.056])
@pytest.mark.parametrize("lowered", [False, True])
def test_a_long_foot_keeps_its_length_and_seam_at_two_ankle_heights(
    ankle_radius, lowered, tmp_path, monkeypatch
):
    # In internal controls, as the seam limits are: a long foot, on the radii these cases were written in.
    p = load_bodies()["female-chibi"]
    foot = 0.418
    ankle = (0.32 if lowered else template_landmarks()["anklePerFoot"]) * foot
    p = p | {"foot": foot, "ankleHeight": ankle, "ankleRadius": ankle_radius * 0.92}
    L = landmarks(p)
    base = load_template()
    report = wrap_body(base, p, tmp_path)
    points = np.load(tmp_path / "wrapped.npy")
    raw = np.array(base["verts"])
    is_foot = np.array(base["regions"]) == "foot"
    foot = template_rows(points, base)[is_foot & (raw[:, 0] > 0)]
    expected = ankle * L["T"]
    assert points[base["rings"]["ankle"], 2].mean() == pytest.approx(expected, abs=0.5)
    assert np.ptp(foot[:, 1]) == pytest.approx(L["footLength"], abs=0.1)
    assert abs(foot[:, 2].min()) <= FOOT_TO_GROUND_MM
    assert report["footSeamCreaseDeg"] <= SEAM_CREASE_DEG["foot"]
    assert miss_share(report, points, "foot") <= SEAM_MISS_SHARE["foot"]
    assert report["flippedFaces"] == report["crossedMidline"] == 0

    # The projection report covers the leg. Check both triangles of every attached foot
    # quad against the template too, including those that touch the seam.
    faces = np.array(base["faces"])
    quads = faces[is_foot[faces].any(axis=1)]
    triangles = np.concatenate((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]))

    def normals(vertices):
        a, b, c = vertices[triangles].transpose(1, 0, 2)
        return np.cross(b - a, c - a)

    assert np.all(np.sum(normals(raw) * normals(points), axis=1) > 0)


@pytest.fixture(scope="module")
def wrapped_body(tmp_path_factory):
    """Wraps parameters no body has, by size and limb thickness, once each: wrap.json and the mesh."""
    p = next(iter(load_bodies().values()))

    @cache
    def wrap(size, limbs=1.0):
        folder = tmp_path_factory.mktemp("body")
        changed = {
            "stature": p["stature"] * size,
            "wristRadius": p["wristRadius"] * limbs,
            "ankleRadius": p["ankleRadius"] * limbs,
        }
        report = wrap_body(load_template(), {**p, **changed}, folder)
        return report, np.load(folder / "wrapped.npy")

    return wrap


@pytest.mark.parametrize("part", SEAM_CREASE_DEG)
def test_the_seam_limit_does_not_depend_on_the_bodys_size(wrapped_body, part):
    # The same proportions, larger: the miss in millimetres grows with the body and the share stays.
    small, grown = wrapped_body(1.0), wrapped_body(1.5)
    assert grown[0][f"{part}SeamMissMm"] > small[0][f"{part}SeamMissMm"] * 1.4
    assert miss_share(*grown, part) == pytest.approx(miss_share(*small, part), abs=0.02)
    assert miss_share(*grown, part) <= SEAM_MISS_SHARE[part]


@pytest.mark.parametrize("size", [0.75, 1.5])
@pytest.mark.parametrize("part", SEAM_CREASE_DEG)
def test_a_limb_too_thick_for_its_part_breaks_the_seam_limit_at_any_size(wrapped_body, size, part):
    assert miss_share(*wrapped_body(size, limbs=1.8), part) > SEAM_MISS_SHARE[part]


@pytest.mark.parametrize("size", [0.6, 1.5])
def test_the_seam_rules_do_not_depend_on_the_bodys_size(size):
    bodies = [(sex_of(name), spec) for name, spec in load_specs().items()]
    first_template, first = bodies[0]
    too_thick = {**first, "ankleGirth": first["footLength"] * 1.6}
    assert seam_problems(too_thick, first_template)
    for template, spec in [*bodies, (first_template, too_thick)]:
        scaled = {key: value * size if key in MILLIMETRES else value for key, value in spec.items()}
        assert seam_problems(scaled, template) == seam_problems(spec, template)


@pytest.mark.parametrize("name", load_bodies())
def test_right_hand_and_foot_mirror_the_left(name):
    ensure_built(name, ("wrap",))
    base = load_template()
    raw = np.array(base["verts"], float)
    regions = np.array(base["regions"])
    points = template_rows(np.load(BUILD / "bodies" / name / "wrapped.npy"), base)
    by_position = {tuple(v): i for i, v in enumerate(raw.tolist())}
    for region in ("hand", "foot"):
        left = np.nonzero((regions == region) & (raw[:, 0] > 0))[0]
        right = [by_position[(-x, y, z)] for x, y, z in raw[left].tolist()]
        assert np.abs(points[left] * (-1, 1, 1) - points[right]).max() < 1e-6


class Sphere:
    def __init__(self, radius):
        self.radius = radius
        self.calls = 0

    def __call__(self, points):
        self.calls += 1
        return np.linalg.norm(points, axis=1) - self.radius


def test_marching_finds_each_rays_first_crossing():
    rays = np.array([[1.0, 0, 0], [0, 0.6, 0.8], [0, 0, -1.0]])
    hits = march_many(Sphere(173.0), np.zeros((3, 3)), rays)
    assert np.linalg.norm(hits, axis=1) == pytest.approx(173.0, abs=0.01)


def test_marching_leaves_a_ray_that_never_crosses_at_its_centre():
    centres = np.array([[0.0, 0, 0], [900.0, 0, 0]])
    hits = march_many(Sphere(50.0), centres, np.array([[1.0, 0, 0], [0, 1.0, 0]]))
    assert hits[0] == pytest.approx([50.0, 0, 0], abs=0.01)
    assert hits[1] == pytest.approx(centres[1])


def test_marching_past_one_chunk_still_finds_the_crossing():
    # The far crossing lies beyond the first chunk of steps, and a near ray must not wait for it.
    rays = np.array([[1.0, 0, 0], [1.0, 0, 0]])
    field = Sphere(400.0)
    hits = march_many(field, np.zeros((2, 3)), rays, chunk=8)
    assert hits[:, 0] == pytest.approx(400.0, abs=0.01)


def grid_graph(width, height):
    """A regular grid's positions and neighbour lists."""
    raw = np.array([[x, y, 0.0] for y in range(height) for x in range(width)])
    neighbours = [[] for _ in range(len(raw))]
    for i, (x, y, _) in enumerate(raw.astype(int).tolist()):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if 0 <= x + dx < width and 0 <= y + dy < height:
                neighbours[i].append(y * width + x + dx + dy * width)
    return raw, neighbours


def test_the_seam_correction_carries_a_slope_straight_across():
    # A correction that keeps growing at a steady rate on both sides is already the smoothest.
    raw, neighbours = grid_graph(12, 5)
    columns = raw[:, 0].astype(int)
    known = np.nonzero((columns < 3) | (columns > 8))[0]
    free = np.nonzero((columns >= 3) & (columns <= 8))[0]
    slope = np.column_stack((raw[:, 0] * 2.0, np.zeros(len(raw)), raw[:, 0] * -0.5))
    solved = smooth_correction(raw, neighbours, free, known, slope[known])
    assert solved == pytest.approx(slope[free], abs=1e-6)


def test_the_seam_correction_dies_out_without_a_step():
    # Full size on the limb's rows, nothing on the far ones: it falls monotonically between them.
    raw, neighbours = grid_graph(20, 3)
    columns = raw[:, 0].astype(int)
    near, far = columns < 3, columns > 16
    known = np.nonzero(near | far)[0]
    free = np.nonzero(~(near | far))[0]
    values = np.zeros((len(known), 3))
    values[near[known]] = (1.0, 0.0, 0.0)
    solved = smooth_correction(raw, neighbours, free, known, values)
    along = solved[raw[free, 1] == 1.0, 0]
    assert along[0] <= 1.0 + 0.2 and along[-1] >= -0.2
    assert np.abs(np.diff(along)).max() < 0.2


def test_quad_normals_agree_with_the_general_version():
    points = np.random.default_rng(3).normal(size=(40, 3))
    quads = np.random.default_rng(4).integers(0, 40, size=(30, 4))
    assert quad_normals(points, quads) == pytest.approx(face_normals(points, quads.tolist()))


def test_glute_projection_preserves_the_wrapped_peak_height_and_width(tmp_path):
    # Re-projecting instead of carrying fitted vertices can make a neighbouring ring
    # become the maximum, changing the measured peak height.
    params = load_bodies()["female-balanced"]
    L = landmarks(params)
    raw = load_template()
    hip = np.isin(raw["segments"], ("trunk", "upperLeg"))
    peaks = []
    for value in (-0.03, 0.03):
        report = wrap_body(raw, params | {"gluteProjection": value}, tmp_path)
        assert report["flippedFaces"] == report["crossedMidline"] == 0
        verts = np.load(tmp_path / "wrapped.npy")[:len(hip)]
        region = verts[hip & (verts[:, 2] >= L["at"](-0.05)) & (verts[:, 2] <= L["at"](0.25))]
        peak = region[region[:, 1].argmax()]
        peaks.append((np.ptp(region[:, 0]), peak[2], peak[1]))
    np.testing.assert_allclose(peaks[0][:2], peaks[1][:2], atol=0.001, rtol=0)
    assert peaks[1][2] - peaks[0][2] == pytest.approx(0.06 * L["T"], abs=0.01)


def test_waist_transition_redistributes_the_trunk_without_moving_the_pelvis(tmp_path):
    base = load_template()
    p = load_bodies()["female-long"]
    L = landmarks(p)
    faces = np.asarray(base["faces"])
    trunk = np.asarray(base["regions"]) == "trunk"
    quads = faces[trunk[faces].all(axis=1)]
    edges = np.stack((quads, np.roll(quads, -1, axis=1)), axis=-1).reshape(-1, 2)
    widths, meshes = [], []
    for offset in (0, 0.16):
        report = wrap_body(base, p | {"waistContourShift": offset}, tmp_path)
        assert report["flippedFaces"] == report["crossedMidline"] == 0
        verts = np.load(tmp_path / "wrapped.npy")
        meshes.append(verts)
        a, b = verts[edges[:, 0]], verts[edges[:, 1]]
        row = []
        for z in L["at"](np.array([0.43, 0.60])):
            hit = (a[:, 2] < z) != (b[:, 2] < z)
            start, end = a[hit], b[hit]
            t = (z - start[:, 2]) / (end[:, 2] - start[:, 2])
            row.append(np.ptp(start[:, 0] + t * (end[:, 0] - start[:, 0])))
        widths.append(row)
    assert widths[1][0] > widths[0][0] + 0.01 * L["T"]
    assert widths[1][1] < widths[0][1] - 0.01 * L["T"]
    pelvis_and_legs = meshes[0][:, 2] <= L["at"](0.25)
    np.testing.assert_array_equal(meshes[0][pelvis_and_legs], meshes[1][pelvis_and_legs])


@pytest.mark.slow
@pytest.mark.parametrize("name, offset", [("female-chibi", -0.06), ("female-chibi", 0.16), ("female-slim", -0.06)])
def test_waist_transition_preserves_lower_body_and_uv_at_its_boundaries(tmp_path, name, offset):
    from sotaigen.form.head import extended
    from sotaigen.uv import corner_triangles, load_uv, stretch, stretch_report

    base = load_template()
    p = load_bodies()[name]
    L = landmarks(p)
    wrap_body(base, p, tmp_path)
    before = np.load(tmp_path / "wrapped.npy")
    report = wrap_body(base, p | {"waistContourShift": offset}, tmp_path)
    after = np.load(tmp_path / "wrapped.npy")
    assert report["flippedFaces"] == report["crossedMidline"] == 0
    lower = before[:, 2] <= L["at"](0.25)
    np.testing.assert_array_equal(after[lower], before[lower])
    full = extended(base)
    unchanged = np.isin(full["regions"], ("hand", "foot", "head"))
    np.testing.assert_array_equal(after[unchanged], before[unchanged])
    uv = load_uv(full["faces"])
    triangles, _ = corner_triangles(full["faces"])
    values = stretch_report(*stretch(after, uv, np.concatenate(full["faces"]), triangles))
    assert values["l2"] <= 1.15
    assert values["anisotropyP95"] <= 2.2
    assert values["densityP5"] >= 0.6
    assert values["densityP95"] <= 1.8
