"""The head on wrapped bodies: closed skin, crown and chin on their landmarks, turns and nods."""

import numpy as np
import pytest

from sotaigen.form.bodies import load_bodies, load_specs
from sotaigen.form.head import extended, neck_ring_order, topology
from sotaigen.mesh import edge_owners, face_areas, face_pairs, unit_face_normals
from sotaigen.pipeline import rigged, template
from sotaigen.rig.skinning import pose_verts
from sotaigen.template import load_template

NAMES = list(load_specs("typical"))
# Anything above this across the seam is a crease in the shading, not a turn in the form.
SEAM_LIMIT_DEG = 45
# Every neck face's area over its longest edge squared, above this. A square has 1; the template's neck
# faces have about 0.8. With the rows run together at the ring, most bodies came to 0.008-0.08; with them
# even, the worst face, at the root of the neck, comes to 0.13-0.25. See docs/systems/wrap-uv.md.
NECK_FACE_WIDTH = 0.12


def test_the_skin_is_closed():
    """With the head on, every edge of the body is shared by exactly two faces."""
    assert {len(owners) for owners in edge_owners(template()["faces"]).values()} == {2}


def _ahead_of_the_neck(base, verts):
    """The head's vertices well in front of the neck, which is where the chin is.

    The head's mesh starts on the neck below the chin level and the throat's blend flows a little
    ahead of the ring, so neither its lowest vertex nor one just ahead of the ring is the chin.
    """
    head = verts[np.asarray(base["segments"]) == "head"]
    ring = verts[neck_ring_order(load_template())]
    return head[head[:, 1] < ring[:, 1].min() - 0.08 * (head[:, 2].max() - head[:, 2].min())]


@pytest.mark.parametrize("name", NAMES)
def test_head_spans_its_landmarks(name):
    rig = rigged(name)
    base, verts, L = rig.base, rig.verts, rig.L
    head = verts[np.asarray(base["segments"]) == "head"]
    assert head[:, 2].max() == pytest.approx(L["H"], abs=0.03 * L["head"])
    chin = _ahead_of_the_neck(base, verts)[:, 2].min()
    assert chin == pytest.approx(L["chin"], abs=0.05 * L["head"])


@pytest.mark.parametrize("name", NAMES)
def test_the_side_view_shows_which_way_it_faces(name):
    """The face stands well ahead of the neck, the chin under it, and the skull hangs behind."""
    rig = rigged(name)
    base, verts, L = rig.base, rig.verts, rig.L
    h = L["head"]
    head = verts[np.asarray(base["segments"]) == "head"]
    ring = verts[neck_ring_order(load_template())]
    front = _ahead_of_the_neck(base, verts)
    chin = front[front[:, 2].argmin()]
    assert head[:, 1].min() < ring[:, 1].min() - 0.25 * h, "the face does not stand ahead"
    assert chin[1] < ring[:, 1].min() - 0.10 * h, "the chin sits on the neck"
    assert chin[1] > head[:, 1].min() + 0.03 * h, "the chin is not set back from the face"
    assert head[:, 1].max() > ring[:, 1].max() + 0.10 * h, "the skull does not hang behind"


@pytest.mark.parametrize("name", load_bodies())
def test_the_neck_runs_into_the_head_without_a_crease(name):
    """No sharp edge across the seam between template and head, in any body."""
    rig = rigged(name)
    verts = rig.verts
    faces, _ = topology(load_template())
    band = 3 * len(neck_ring_order(load_template()))
    all_faces = extended(load_template())["faces"]
    offset = len(all_faces) - len(faces)
    pairs = face_pairs(all_faces)
    normals = unit_face_normals(verts, all_faces)
    touches = np.zeros(len(all_faces), bool)
    touches[offset : offset + band] = True
    across = pairs[touches[pairs[:, 0]] | touches[pairs[:, 1]]]
    cosine = np.sum(normals[across[:, 0]] * normals[across[:, 1]], axis=1)
    assert np.degrees(np.arccos(np.clip(cosine, -1, 1))).max() < SEAM_LIMIT_DEG


@pytest.mark.parametrize("name", load_bodies())
def test_seating_the_ring_folds_no_neck_face(name):
    """The ring's lift must stay even enough that the neck's rows do not pass one another."""
    rig = rigged(name)
    base, verts = rig.base, rig.verts
    segments = np.asarray(base["segments"])
    faces = base["faces"]
    owners = np.array([segments[f[0]] for f in faces]) == "neck"
    pairs = face_pairs(faces)
    across = pairs[owners[pairs[:, 0]] | owners[pairs[:, 1]]]
    normals = unit_face_normals(verts, faces)
    assert (np.sum(normals[across[:, 0]] * normals[across[:, 1]], axis=1) < 0).sum() == 0


@pytest.mark.parametrize("name", load_bodies())
def test_the_necks_rows_stay_apart(name):
    """No neck face closes up into a line, which any weights but rigid ones would crush in a turn."""
    rig = rigged(name)
    verts = rig.verts
    base = load_template()
    regions = np.asarray(base["regions"])
    faces = [f for f in base["faces"] if (regions[f] == "neck").any()]
    longest = [np.linalg.norm(verts[f] - verts[np.roll(f, 1)], axis=1).max() for f in faces]
    assert (face_areas(verts, faces) / np.square(longest)).min() > NECK_FACE_WIDTH


@pytest.mark.parametrize("name", NAMES)
def test_turning_the_head_collapses_no_face(name):
    rig = rigged(name)
    base, verts, bones, weights = rig.base, rig.verts, rig.bones, rig.weights
    rest = np.maximum(face_areas(verts, base["faces"]), 1e-6)
    for pose in (
        {"neck": (0, 15, 0), "head": (0, 30, 0)},
        {"neck": (10, 0, 0), "head": (20, 0, 0)},
    ):
        posed = pose_verts(bones, weights, verts, pose)
        assert ((face_areas(posed, base["faces"]) / rest) < 0.15).sum() == 0
