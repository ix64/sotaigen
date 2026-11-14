"""That the pose metrics count what they claim, and that every check pose really does something."""

import numpy as np
import pytest

from sotaigen.metrics import COLLAPSED, RestMesh, pose_row
from sotaigen.pipeline import bodies, rigged
from sotaigen.rig.poses import POSES
from sotaigen.rig.skinning import pose_verts

# Two unit squares side by side in the XY plane, sharing the edge from vertex 1 to vertex 4.
POINTS = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]], float)
FACES = [[0, 1, 4, 3], [1, 2, 5, 4]]
# Vertices 2 and 5 are the outer edge of the second square, whose area follows their x.
OUTER = [2, 5]


def squeezed(share):
    """The two squares with the second one narrowed to this share of its rest area."""
    points = POINTS.copy()
    points[OUTER, 0] = 1 + share
    return points


@pytest.fixture
def rest():
    return RestMesh(POINTS, FACES)


def test_the_collapse_threshold_is_the_agreed_one():
    # Every count compared across builds means this. Moving it would pass as an improvement.
    assert COLLAPSED == 0.15


def test_the_rest_pose_collapses_nothing_and_changes_nothing(rest):
    # By name, so that a new column in poses.json doesn't turn this test red.
    row = pose_row(rest, POINTS)
    assert row["collapsedFaces"] == 0
    assert (row["areaP1"], row["edgeP1"], row["edgeP99"]) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize(
    ("share", "collapsed"),
    [(COLLAPSED / 2, 1), (COLLAPSED * 2, 0), (1.0, 0)],
)
def test_a_face_counts_as_collapsed_only_below_the_threshold(rest, share, collapsed):
    assert pose_row(rest, squeezed(share))["collapsedFaces"] == collapsed


def test_every_collapsed_face_is_counted(rest):
    assert pose_row(rest, POINTS * COLLAPSED / 2)["collapsedFaces"] == len(FACES)


def test_a_face_flattened_to_nothing_counts(rest):
    assert pose_row(rest, squeezed(0.0))["collapsedFaces"] == 1


def test_area_and_edge_percentiles_follow_the_deformation(rest):
    row = pose_row(rest, squeezed(0.5))
    # A percentile of so few values interpolates a little way towards the next one.
    assert row["areaP1"] == pytest.approx(0.5, abs=0.02)
    assert row["edgeP1"] == pytest.approx(0.5, abs=0.02)
    assert row["edgeP99"] == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("pose", [name for name in POSES if name != "rest"])
def test_every_check_pose_moves_vertices(pose):
    # Any body will do: a pose that moves nothing checks nothing, on every body alike.
    rig = rigged(next(iter(bodies())))
    verts, bones, weights = rig.verts, rig.bones, rig.weights
    moved = np.linalg.norm(pose_verts(bones, weights, verts, POSES[pose]) - verts, axis=1)
    assert moved.max() > 1.0
