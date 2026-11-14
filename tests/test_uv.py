"""The UV atlas is one valid layout for every body, and its checks catch what they claim to."""

import numpy as np
import pytest

from sotaigen.form.bodies import load_bodies
from sotaigen.form.head import neck_ring_order
from sotaigen.output.glb import glb_bytes, read_glb, split_uv_seams
from sotaigen.pipeline import BUILD, ensure_built, rigged, template
from sotaigen.uv import (
    corner_triangles,
    export_fields,
    load_uv,
    overlapping_triangles,
    signed_areas,
    stretch,
    stretch_report,
)

# Limits leave some room over what the nine bodies measure; see docs/systems/wrap-uv.md.
L2_STRETCH = 1.15
ANISOTROPY_P95 = 2.2
# How many times denser or sparser than the mean the densest and sparsest 5% of the skin below the
# head may get. The hands get more texels on purpose, so this is a wide band. The head is left out:
# its island holds a fixed share of the atlas, so the larger the body around it, the denser it reads
# against the mean, and a face wants the texels (docs/DECISIONS.md).
DENSITY_P5, DENSITY_P95 = 0.6, 1.8


@pytest.fixture(scope="module")
def atlas():
    faces = template()["faces"]
    uv = load_uv(faces)
    triangles, _ = corner_triangles(faces)
    corner_verts = np.concatenate([np.asarray(face) for face in faces])
    return faces, uv, triangles, corner_verts


def test_every_corner_lies_inside_the_atlas(atlas):
    _, uv, _, _ = atlas
    assert uv.min() >= 0.001 and uv.max() <= 0.999


def test_no_triangles_overlap(atlas):
    _, uv, triangles, _ = atlas
    assert len(overlapping_triangles(uv, triangles)) == 0


def test_no_triangle_is_flipped_or_without_area(atlas):
    _, uv, triangles, _ = atlas
    areas = signed_areas(uv, triangles)
    assert areas.min() > 1e-9


def test_the_neck_flows_into_the_head_without_a_seam_except_at_the_back(atlas):
    """A corner on the neck ring has one UV on both sides, except at the cut up the back."""
    _, uv, _, corner_verts = atlas
    base = template()
    ring = neck_ring_order(base)
    raw = np.asarray(base["verts"], float)
    split = [v for v in ring if len(np.unique(uv[corner_verts == v].round(5), axis=0)) > 1]
    assert len(split) == 1
    assert abs(raw[split[0], 0]) < 0.5 and raw[split[0], 1] > raw[ring, 1].mean()


@pytest.mark.parametrize("name", load_bodies())
def test_stretch_stays_small_on_every_preset(atlas, name):
    _, uv, triangles, corner_verts = atlas
    ensure_built(name, ("wrap",))
    points = np.load(BUILD / "bodies" / name / "wrapped.npy")
    values, area = stretch(points, uv, corner_verts, triangles)
    report = stretch_report(values, area)
    assert report["l2"] <= L2_STRETCH
    assert report["anisotropyP95"] <= ANISOTROPY_P95
    head = (np.asarray(template()["segments"])[corner_verts[triangles]] == "head").all(axis=1)
    below = stretch_report(values[~head], area[~head])
    assert below["densityP5"] >= DENSITY_P5
    assert below["densityP95"] <= DENSITY_P95


def test_the_exporter_takes_the_atlas(atlas):
    faces, uv, _, _ = atlas
    fields = export_fields(faces)
    corners = np.concatenate(fields["faceUvs"])
    # Back to the corners, with V flipped back: nothing moved on the way.
    assert (fields["uvs"][corners] * (1, -1) + (0, 1)) == pytest.approx(uv, abs=1e-9)
    count = len(template()["verts"])
    source, split_uv, _ = split_uv_seams(faces, count, fields["uvs"], fields["faceUvs"])
    assert len(source) == len(split_uv)
    # Only seams split a vertex, and only a small share of them lie on one.
    assert count < len(source) < 1.2 * count


def test_an_exported_body_carries_its_uvs():
    name = next(iter(load_bodies()))
    rig = rigged(name)
    base = {**rig.base, **export_fields(rig.base["faces"])}
    document, _ = read_glb(glb_bytes(base, rig.verts, rig.bones, rig.weights))
    assert "TEXCOORD_0" in document["meshes"][0]["primitives"][0]["attributes"]
    assert len(document["extras"]["sotaigen"]["splitFrom"]) > 0


def test_a_different_topology_is_refused():
    faces = template()["faces"]
    with pytest.raises(ValueError, match="extract_uv"):
        load_uv(faces[:-1])


# The checks themselves, on layouts small enough to see the answer.

SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def test_neighbours_sharing_an_edge_do_not_overlap():
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    assert len(overlapping_triangles(SQUARE, triangles)) == 0


def test_crossing_triangles_overlap():
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [1.0, 1.0], [0.2, 0.6]])
    pairs = overlapping_triangles(uv, np.array([[0, 1, 2], [3, 4, 5]]))
    assert pairs.tolist() == [[0, 1]]


def test_a_triangle_inside_another_overlaps():
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.1, 0.1], [0.2, 0.1], [0.1, 0.2]])
    assert len(overlapping_triangles(uv, np.array([[0, 1, 2], [3, 4, 5]]))) == 1


def test_a_copy_on_top_overlaps_whichever_way_it_winds():
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    assert len(overlapping_triangles(uv, np.array([[0, 1, 2], [3, 4, 5]]))) == 1


def test_triangles_apart_in_a_far_cell_do_not_overlap():
    uv = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.9, 0.9], [1.0, 0.9], [0.9, 1.0]])
    assert len(overlapping_triangles(uv, np.array([[0, 1, 2], [3, 4, 5]]))) == 0


def flat_square():
    points = np.column_stack((SQUARE * 100.0, np.zeros(4)))
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    return points, np.arange(4), triangles


def test_an_undistorted_map_measures_one():
    points, corners, triangles = flat_square()
    values, area = stretch(points, SQUARE * 0.5, corners, triangles)
    assert values == pytest.approx(np.ones((2, 2)))
    assert area == pytest.approx([5000.0, 5000.0])
    report = stretch_report(values, area)
    assert report["l2"] == pytest.approx(1.0)
    assert report["anisotropyMax"] == pytest.approx(1.0)


def test_a_map_squeezed_one_way_measures_the_squeeze():
    points, corners, triangles = flat_square()
    values, area = stretch(points, SQUARE * (1.0, 0.25), corners, triangles)
    assert values[:, 0] / values[:, 1] == pytest.approx(4.0)
    assert stretch_report(values, area)["anisotropyP95"] == pytest.approx(4.0)
