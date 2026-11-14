"""Adaptive pose refinement must be conforming, local, and affine-exact."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sotaigen.corrective import axilla
from sotaigen.corrective.pose_grid import Correctives, PoseGrid, from_samples, groups, pose_weights
from sotaigen.rig.skinning import MIDLINE_BONES, swing_euler


def square():
    return PoseGrid([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], [[0, 1, 2], [0, 2, 3]])


@pytest.mark.parametrize("point", [[.6, .2], [.5, .5], [1., .5]])
def test_interior_diagonal_and_boundary_insertions_keep_complete_affine_coverage(point):
    grid = square().insert(point)
    for x in np.linspace(0, 1, 17):
        for y in np.linspace(0, 1, 17):
            ids, weights = grid.locate([x, y])
            assert weights.min() >= 0
            np.testing.assert_allclose(weights @ grid.points[ids], [x, y], atol=1e-12)
    # Interior split adds 2 cells; shared diagonal also adds 2; outer edge adds 1.
    assert len(grid.cells) == (3 if point[0] == 1 else 4)


def test_refinement_preserves_unrelated_cells_and_values_along_outer_edges():
    before = square()
    after = before.insert([.8, .2])
    assert [0, 2, 3] in after.cells.tolist()
    original_values = np.array([3., -2., 7., 10.])
    # An intentionally different new value must not leak beyond its parent triangle.
    values = np.r_[original_values, 100.]
    for point in [[.2, .8], [.1, .9], [.4, .4], [.8, .8], [0., 0.], [1., 1.]]:
        ids, weights = before.locate(point)
        other, blend = after.locate(point)
        assert blend @ values[other] == pytest.approx(weights @ original_values[ids])


def test_shared_tetrahedron_face_is_split_conformingly():
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]]
    grid = PoseGrid(points, [[0, 1, 2, 3], [0, 2, 1, 4]]).insert([.2, .3, 0])
    assert len(grid.cells) == 6
    rng = np.random.default_rng(42)
    for cell in [[0, 1, 2, 3], [0, 2, 1, 4]]:
        for weights in rng.dirichlet(np.ones(4), 40):
            point = weights @ np.array(points)[cell]
            ids, blend = grid.locate(point)
            np.testing.assert_allclose(blend @ grid.points[ids], point, atol=1e-12)


def test_unsupported_poses_and_duplicate_samples_are_rejected():
    grid = square()
    with pytest.raises(ValueError, match="outside"):
        grid.locate([1.1, .5])
    with pytest.raises(ValueError, match="outside"):
        grid.insert([-.1, .5])
    with pytest.raises(ValueError, match="already"):
        grid.insert([0., 0.])


def test_rectangular_lattice_has_no_zero_volume_cells_or_gaps():
    grid = PoseGrid.rectilinear([[0, 20, 95], [-90, 0, 90], [-24, 0, 110]])
    assert len(grid.cells) == 8 * 6
    for point in np.random.default_rng(93).uniform([0, -90, -24], [95, 90, 110], (300, 3)):
        ids, weights = grid.locate(point)
        np.testing.assert_allclose(weights @ grid.points[ids], point, atol=1e-11)
    with pytest.raises(ValueError, match="increasing"):
        PoseGrid.rectilinear([[0, 0, 1], [-1, 1]])


def driver_table():
    grid = PoseGrid.rectilinear([[0, 30, 95], [-90, 0, 90], [-24, 0, 110]])
    grid = grid.insert([23., 12., 52.])
    return {"kind": "simplices", "domain": [[0, 95], [-90, 90], [-24, 110]],
            "sides": {side: grid.metadata() | {"targets": [
                None if np.array_equal(point, [0, 0, 0]) else f"{side}{i}"
                for i, point in enumerate(grid.points)]} for side in ("left", "right")}}


def test_adaptive_driver_is_zero_at_rest_and_mirrors_twist_and_lateral_intent():
    table = driver_table()
    assert not any(pose_weights({}, table).values())
    both = pose_weights({"upperArm": [40, 27, -12]}, table)
    explicit = pose_weights({"leftUpperArm": [40, 27, -12],
                             "rightUpperArm": [40, -27, 12]}, table)
    assert explicit == both
    assert sum(both.values()) == pytest.approx(2.)
    assert pose_weights({"rightUpperArm": [50, -20, -35], "upperArm": [40, 27, -12]}, table) == both
    # The table's coordinates are the arm's swing and twist; outside its domain they clamp.
    assert pose_weights({"upperArm": swing_euler(120, -120, 130)}, table) == pytest.approx(
        pose_weights({"upperArm": swing_euler(95, -90, 110)}, table), abs=1e-12)


@pytest.mark.node
@pytest.mark.parametrize("planar", [False, True])
def test_browser_and_python_match_at_vertices_shared_faces_and_mixed_poses(planar):
    table = driver_table()
    if planar:
        grid = PoseGrid.rectilinear([[0, 30, 95], [-24, 0, 110]])
        common = {"points": grid.points, "cells": grid.cells,
                  "deltas": np.zeros((len(grid.points), 1, 3))}
        table = from_samples({"left": common | {"ids": [0]},
                              "right": common | {"ids": [1]}}, 2, axilla.TWIST_LIMIT).driver
    poses = []
    for point in table["sides"]["left"]["points"]:
        angles = np.zeros(3)
        angles[table.get("axes", [0, 1, 2])] = point
        poses.append({"upperArm": angles.tolist()})
    poses += [{"upperArm": point.tolist()} for point in np.random.default_rng(32).uniform(
        [-10, -120, -35], [120, 120, 135], (100, 3))]
    poses += [{"upperArm": [30, 0, 10], "rightUpperArm": [20, -30, -50]},
              {"rightUpperArm": [50, 20, 70], "upperArm": [0, 0, 0]}]
    path = Path(__file__).resolve().parents[1] / "web/runtime/driver.mjs"
    code = f'''import {{correctiveWeights}} from {json.dumps(path.as_uri())};
let input = ''; for await (const chunk of process.stdin) input += chunk;
const {{poses, groups, central}} = JSON.parse(input);
console.log(JSON.stringify(poses.map(pose => correctiveWeights(pose, groups, central))));'''
    done = subprocess.run(["node", "--input-type=module", "-e", code],
                          input=json.dumps({"poses": poses, "groups": groups(table, "ipc"), "central": list(MIDLINE_BONES)}),
                          capture_output=True, text=True, check=True)
    for pose, actual in zip(poses, json.loads(done.stdout), strict=True):
        assert actual == pytest.approx(pose_weights(pose, table), abs=1e-12)


def test_bind_mesh_uses_its_own_body_specific_driver():

    table = driver_table()
    targets = Correctives({name: np.tile(point, (2, 1))
                           for local in table["sides"].values()
                           for name, point in zip(local["targets"], local["points"], strict=True)
                           if name}, table)
    verts = np.zeros((2, 3))
    assert axilla.bind_verts(verts, targets, {}) is verts
    np.testing.assert_allclose(
        axilla.bind_verts(verts, targets, {"leftUpperArm": swing_euler(40, 27, -12)}),
        np.tile([40, 27, -12], (2, 1)), atol=1e-12)
    np.testing.assert_array_equal(verts, np.zeros((2, 3)))


def test_local_samples_expand_without_moving_the_opposite_side_or_unrelated_vertices():

    grid = PoseGrid.rectilinear([[0, 1], [-1, 0, 1], [-1, 0, 1]])
    delta = np.repeat(grid.points[:, None, :], 2, axis=1)
    common = {"points": grid.points, "cells": grid.cells, "deltas": delta}
    samples = {"left": common | {"ids": [0, 1]}, "right": common | {"ids": [4, 5]}}
    targets = from_samples(samples, 6, axilla.TWIST_LIMIT)
    verts = np.zeros((6, 3))
    moved = axilla.bind_verts(verts, targets, {"leftUpperArm": swing_euler(.6, -.3, .2)})
    np.testing.assert_allclose(moved[:2], np.tile([.6, -.3, .2], (2, 1)), atol=1e-12)
    np.testing.assert_array_equal(moved[2:], np.zeros((4, 3)))
    assert set(targets) == {name for names in targets.driver["targets"].values() for name in names}
    corrupted = delta.copy()
    corrupted[(grid.points == 0).all(axis=1)] = 1
    with pytest.raises(ValueError, match="moves the rest"):
        from_samples(samples | {"left": samples["left"] | {"deltas": corrupted}}, 6, axilla.TWIST_LIMIT)


def test_two_axis_contact_table_retains_the_twist_fade():

    grid = PoseGrid.rectilinear([[0, 35, 95], [-24, 0, 110]])
    common = {"points": grid.points, "cells": grid.cells,
              "deltas": np.zeros((len(grid.points), 1, 3))}
    table = from_samples({"left": common | {"ids": [0]},
                          "right": common | {"ids": [1]}}, 2, axilla.TWIST_LIMIT).driver
    for y in [-120, -90, -45, 0, 30, 90, 120]:
        pose = {"leftUpperArm": swing_euler(35, y, 0)}
        assert sum(pose_weights(pose, table).values()) == pytest.approx(
            sum(axilla.pose_weights(pose).values()))
    assert not any(pose_weights({}, table).values())
