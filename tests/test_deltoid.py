"""The raised shoulder's hollow and gathered deltoid, on top of the shoulder's targets."""

from functools import cache

import numpy as np

from sotaigen.corrective import axilla, deltoid
from sotaigen.mesh import triangulate, vertex_normals
from sotaigen.pipeline import rigged

NAME = "male-balanced"


@cache
def made():
    r = rigged(NAME)
    plain = axilla.targets(r.base, r.verts, r.bones, r.weights)
    return r, plain, deltoid.gather(plain, r.base, r.verts, r.bones)


def added(side, x, z):
    _, plain, gathered = made()
    name = axilla.NAMES[side][axilla.SAMPLES.index((x, z))]
    return gathered[name] - plain[name]


def test_it_comes_in_with_the_side_raise_and_only_there():
    assert not added("left", 95, 0).any()
    assert not added("left", 20, -24).any()
    size = [np.abs(added("left", 0, z)).max() for z in (35, 72, 110)]
    assert 0 < size[0] < size[1] < size[2]
    # A forward raise takes it off.
    assert not added("left", 95, 110).any()


def test_it_stays_on_its_own_side_and_off_the_neck_and_head():
    r, _, _ = made()
    template = np.asarray(r.base["verts"])
    segments = np.asarray(r.base["segments"])
    moved = np.linalg.norm(added("left", 0, 110), axis=1) > 0
    assert moved.any()
    assert (template[moved, 0] > 0).all()
    assert set(segments[moved]) <= {"trunk", "upperArm"}


def test_the_two_sides_mirror():
    left, right = added("left", 0, 110), added("right", 0, 110)
    assert np.isclose(np.linalg.norm(left, axis=1).max(), np.linalg.norm(right, axis=1).max())


def test_the_hollow_sinks_along_the_normal():
    r, _, _ = made()
    hollow, _ = deltoid.shapes(r.base, r.verts, r.bones)["left"]
    normals = vertex_normals(r.verts, triangulate(r.base["faces"], np.asarray(r.base["verts"])))
    moved = np.linalg.norm(hollow, axis=1) > 1e-9
    assert moved.any()
    assert (np.einsum("vi,vi->v", hollow[moved], normals[moved]) < 0).all()


def test_a_shape_with_no_target_table_is_given_back():
    r, _, _ = made()
    assert deltoid.gather({}, r.base, r.verts, r.bones) == {}
