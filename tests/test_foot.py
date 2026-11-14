import numpy as np
import pytest

from sotaigen.form.foot import sculpted
from sotaigen.mesh import triangulate, unit_face_normals
from sotaigen.template import load_template


@pytest.fixture(scope="module")
def foot():
    base = load_template()
    raw = np.array(base["verts"], float)
    left = (np.array(base["regions"]) == "foot") & (raw[:, 0] > 0)
    return base, raw, sculpted(base), left


def test_only_the_feet_move(foot):
    base, raw, points, _ = foot
    others = np.array(base["regions"]) != "foot"
    assert np.array_equal(points[others], raw[others])


def test_the_foot_keeps_what_placing_it_relies_on(foot):
    _, raw, points, left = foot
    assert np.ptp(points[left, 1]) == pytest.approx(np.ptp(raw[left, 1]))
    assert points[left, 2].min() == pytest.approx(raw[left, 2].min())
    # The top of the foot meets the ankle ring: it stays where the limb expects it.
    top = raw[left, 2] > raw[left, 2].max() - 2.0
    assert np.abs(points[left][top] - raw[left][top]).max() < 0.01


def test_the_arch_rises_on_the_inner_side_and_the_outer_edge_stays_down(foot):
    _, _, points, left = foot
    part = points[left]
    length = np.ptp(part[:, 1])
    floor = part[:, 2].min()
    station = (part[:, 1] - part[:, 1].min()) / length
    middle = part[(station > 0.55) & (station < 0.65)]
    third = np.ptp(middle[:, 0]) / 3
    inner = middle[middle[:, 0] < middle[:, 0].min() + third]
    outer = middle[middle[:, 0] > middle[:, 0].max() - third]
    assert (inner[:, 2].min() - floor) / length > 0.05
    assert (outer[:, 2].min() - floor) / length < 0.005


def test_the_right_foot_mirrors_the_left(foot):
    _, raw, points, left = foot
    by_position = {tuple(v): i for i, v in enumerate(raw.tolist())}
    right = [by_position[(-x, y, z)] for x, y, z in raw[left].tolist()]
    assert np.abs(points[left] * (-1, 1, 1) - points[right]).max() < 1e-9


def test_no_face_of_the_foot_turns_over(foot):
    base, raw, points, left = foot
    triangles = triangulate(base["faces"], raw)
    triangles = triangles[left[triangles].any(1)]
    agreement = np.sum(
        unit_face_normals(raw, triangles) * unit_face_normals(points, triangles), axis=1
    )
    assert agreement.min() > 0.0
