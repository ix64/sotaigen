"""The shared mesh helpers, on shapes small enough to check by hand, and the checked-in template."""

import numpy as np

from sotaigen.form.head import extended, template_rows
from sotaigen.mesh import (
    adjacency,
    edge_owners,
    edges_of,
    face_areas,
    face_normals,
    face_pairs,
    unit_face_normals,
    vertex_neighbours,
)
from sotaigen.template import load_template

# A 2 x 2 quad in the XY plane with a triangle on top of it, sharing one edge.
POINTS = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0], [1, 3, 0]], float)
FACES = [[0, 1, 2, 3], [3, 2, 4]]


def test_face_normals_are_as_long_as_twice_the_area():
    assert np.allclose(face_normals(POINTS, FACES), [[0, 0, 8], [0, 0, 2]])


def test_unit_face_normals_have_unit_length():
    assert np.allclose(unit_face_normals(POINTS, FACES), [[0, 0, 1], [0, 0, 1]])


def test_face_areas():
    assert np.allclose(face_areas(POINTS, FACES), [4, 1])


def test_edges_are_unique_and_sorted():
    assert edges_of(FACES).tolist() == [[0, 1], [0, 3], [1, 2], [2, 3], [2, 4], [3, 4]]


def test_edge_owners_and_face_pairs():
    assert edge_owners(FACES)[(2, 3)] == [0, 1]
    assert face_pairs(FACES).tolist() == [[0, 1]]


def test_adjacency_runs_both_ways():
    source, sink, count = adjacency(5, FACES)
    assert sorted(zip(source.tolist(), sink.tolist())) == list(zip(source.tolist(), sink.tolist()))
    assert sink[source == 2].tolist() == [1, 3, 4]
    assert count.tolist() == [2, 2, 3, 3, 2]


def test_vertex_neighbours():
    assert sorted(vertex_neighbours(5, FACES)[2]) == [1, 3, 4]


def test_the_template_refers_only_to_its_own_vertices():
    base = load_template()
    count = len(base["verts"])
    assert count and base["faces"]
    assert len(base["regions"]) == len(base["segments"]) == count
    rings = [base["openings"]["neck"], *base["rings"].values()]
    for indices in [*base["faces"], *rings]:
        assert indices and all(0 <= index < count for index in indices)
    assert base["source"]["license"] == "CC0"


def test_the_head_sits_between_the_source_mesh_and_the_added_loops():
    """Numbering only ever appends: the source's vertices, the head's, then each batch of loops."""
    base = load_template()
    full = extended(base)
    first = base["refined"]["from"]
    added = len(full["verts"]) - len(base["verts"])
    assert added > 0
    head = np.flatnonzero(np.array(full["segments"]) == "head")
    assert np.array_equal(head, np.arange(first, first + added))
    assert np.array_equal(template_rows(full["verts"], base), base["verts"])
    shifted = [[i + added if i >= first else i for i in face] for face in base["faces"]]
    assert full["faces"][: len(base["faces"])] == shifted
    assert full["segments"] == base["segments"][:first] + ["head"] * added + base["segments"][first:]


def test_an_added_loop_only_halves_rungs_numbered_before_it():
    base = load_template()
    refined = base["refined"]
    first = refined["from"]
    assert all(max(pair) < first + k for k, pair in enumerate(refined["parents"]))
    assert sum(batch["count"] for batch in refined["batches"]) == len(base["verts"]) - first
    rings = [base["openings"]["neck"], *base["rings"].values()]
    assert all(index < first for ring in rings for index in ring)


def test_face_normals_of_mixed_polygons_match_the_definition():
    points = np.random.default_rng(3).uniform(-50, 50, (40, 3))
    faces = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10, 11], [12, 13, 14], [15, 16, 17, 18]]
    wanted = np.zeros((len(faces), 3))
    for i, face in enumerate(faces):
        p = points[face]
        for j in range(1, len(face) - 1):
            wanted[i] += np.cross(p[j] - p[0], p[j + 1] - p[0])
    assert np.array_equal(face_normals(points, faces), wanted)
    assert np.allclose(face_areas(points, faces), np.linalg.norm(wanted, axis=1) / 2)
    assert face_normals(points, []).shape == (0, 3)
