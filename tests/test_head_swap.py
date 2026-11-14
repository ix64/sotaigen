"""The head swap, walked through on real files with a made-up head (blender/fixtures/made_up_head.py).

The head comes in two kinds: one whose neck has 72 vertices, zipped to the ring's 36, and a coarse one
with 36, welded to it. Each file is opened afresh with factory settings, so with Auto Run Python Scripts
off, and turned to the check poses by its controls. Tests marked `blender` leave the run when there is
no Blender (tests/conftest.py). Nothing here bakes a contact table.
"""

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from scipy.spatial import cKDTree

from sotaigen.cli import main as command
from sotaigen.mesh import edge_owners, unit_face_normals
from sotaigen.metrics import RestMesh, pose_row
from sotaigen.output.blend import blend_body
from sotaigen.output.blender import run_blender, script
from sotaigen.output.glb import driver_table
from sotaigen.output.head_swap import (
    BODY,
    CHECK_POSES,
    HEAD_REGION,
    BrokenSwap,
    Dump,
    FileMesh,
    boundary_loops,
    find_skin,
    join_skin,
    largest_crease,
    read_dump,
    resampled,
    swap_head,
    zip_strip,
)
from sotaigen.pipeline import rigged
from sotaigen.rig.poses import POSES

ROOT = Path(__file__).resolve().parents[1]
TEST_HEAD = ROOT / "blender" / "fixtures" / "made_up_head.py"
# Which body gets which head: 2-3 of the twelve are enough to walk the path (docs/parts/head.md#换头工具).
CASES = {"female-compact": False, "male-balanced": True}
# The skin below the head is the body's: float32 metres through the armature, as tests/test_blend.py says.
SKIN_MM = 0.003
# Poses that drive the correctives: they should still move the body as they did.
CORRECTIVE_POSES = ("armsHigh", "reachHigh", "crouch")
# The test head's chin stands 9 % of its height forward of the neck, a sharper turn than our own head's,
# so the crease is measured against this and not the old head's 15-20 degrees.
CREASE_DEG = 30.0
# Across the ring: the zip's fine rows turn it within 16 degrees (the old head's 11-15); the coarse head's
# first row is two of the fine ones away, so it carries more of the turn (45 on male-balanced).
NORMAL_STEP_DEG = {False: 18.0, True: 50.0}


class Swapped(NamedTuple):
    name: str
    coarse: bool
    report: dict
    old: object
    new: object
    head: object
    old_posed: dict
    new_posed: dict
    pose_report: dict
    old_pose_report: dict
    out: Path


def poses_of(blend, folder, tag, central):
    names = (*CHECK_POSES, *CORRECTIVE_POSES)
    request = folder / f"{tag}-request.json"
    request.write_text(
        json.dumps({"poses": {n: POSES[n] for n in names}, "central": central, "limits": [True], "actions": False})
    )
    run_blender(script("pose"), request, folder / f"{tag}-posed", blend=blend)
    return (
        dict(np.load(folder / f"{tag}-posed.npz")),
        json.loads((folder / f"{tag}-posed.json").read_text()),
    )


def make_head(blend, out, *flags):
    run_blender(TEST_HEAD, out, *flags, blend=blend)
    return out


@pytest.fixture(scope="module", params=list(CASES))
def swapped(request, tmp_path_factory):
    name = request.param
    coarse = CASES[name]
    folder = tmp_path_factory.mktemp(name)
    blend = blend_body(name, folder / "body.blend")
    head = make_head(blend, folder / "head.blend", *(["--coarse"] if coarse else []))
    out = folder / "out.blend"
    report = swap_head(blend, head, out)
    central = driver_table(rigged(name).bones)["central"]
    old_posed, old_report = poses_of(blend, folder, "old", central)
    new_posed, new_report = poses_of(out, folder, "new", central)
    return Swapped(
        name,
        coarse,
        report,
        read_dump(blend, folder / "old"),
        read_dump(out, folder / "new"),
        read_dump(head, folder / "head"),
        old_posed,
        new_posed,
        new_report,
        old_report,
        out,
    )


def attachments(s):
    """The file's head fields, with the numbers it says for the new head."""
    return s.new.info["head"]


def new_rows(s):
    first, count = attachments(s)["region"]
    return np.arange(first, first + count)


def body_rows(s):
    """The surviving vertices' rows in the old file, which are also the first rows of the new one."""
    region = s.old.mesh(BODY).group("headRegion")[0]
    return np.delete(np.arange(len(s.old.mesh(BODY).verts)), region)


def part_rows(s, part):
    """The vertices of a test head's piece in the swapped file, picked by what binds them."""
    mesh = s.new.mesh(part[0])
    rows, values = mesh.group(part[1])
    return mesh, rows[values > 0]


# Without Blender ----------------------------------------------------------------------------------


def circle(count, radius=1.0, z=0.0):
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles), np.full(count, z)))


@pytest.mark.parametrize("other", [24, 72, 50, 37])
def test_the_strip_closes_two_loops_of_any_counts(other):
    ring, loop = circle(36), circle(other, z=1.0)
    faces, nearest = zip_strip(ring, loop)
    owners = edge_owners([list(face) for face in faces])
    open_edges = {edge for edge, row in owners.items() if len(row) == 1}
    # Exactly the two loops' own edges are left for the mesh on each side of it.
    assert open_edges == {
        *((min(i, (i + 1) % 36), max(i, (i + 1) % 36)) for i in range(36)),
        *((36 + min(j, (j + 1) % other), 36 + max(j, (j + 1) % other)) for j in range(other)),
    }
    assert all(len(row) <= 2 for row in owners.values())
    points = np.vstack((ring, loop))
    normals = np.cross(points[[f[1] for f in faces]] - points[[f[0] for f in faces]], points[[f[2] for f in faces]] - points[[f[0] for f in faces]])
    middles = np.array([points[f].mean(0) for f in faces])
    assert (np.sum(normals[:, :2] * middles[:, :2], 1) > 0).all(), "a strip face points inward"
    assert len(nearest) == other


def made_mesh(name, verts, faces, groups=None):
    """A mesh as read from a file, with only what a skin needs."""
    groups = groups or {}
    rows = [np.asarray(row, int) for row in groups.values()]
    return FileMesh(
        name,
        np.asarray(verts, float),
        faces,
        None,
        np.zeros(len(faces), np.int32),
        None,
        np.zeros((sum(len(face) for face in faces), 3)),
        list(groups),
        np.concatenate(rows) if rows else np.empty(0, int),
        np.concatenate([np.full(len(row), j) for j, row in enumerate(rows)]) if rows else np.empty(0, int),
        np.ones(sum(len(row) for row in rows)),
        {},
        np.eye(4).tolist(),
    )


def made_head(*meshes):
    return Dump(list(meshes), {}, {}, None, [], 0)


def tube(rows, top=None):
    """Quads between closed rows of one count, the first row lowest and open, wound outward, with a pole
    on top when `top` says where."""
    count = len(rows[0])
    faces = [
        [i * count + k, i * count + (k + 1) % count, (i + 1) * count + (k + 1) % count, (i + 1) * count + k]
        for i in range(len(rows) - 1)
        for k in range(count)
    ]
    verts = np.vstack(rows)
    if top is not None:
        first = (len(rows) - 1) * count
        faces += [[first + k, first + (k + 1) % count, len(verts)] for k in range(count)]
        verts = np.vstack((verts, top))
    return verts, faces


def bowtie(start=0, z=5.0):
    """Two triangles meeting at one vertex, the kind of boundary generated hair has: vertices and faces."""
    points = [[0, 0, z], [1, 0, z], [1, 1, z], [-1, 0, z], [-1, -1, z]]
    return points, [[start, start + 1, start + 2], [start, start + 3, start + 4]]


def sealed(start=0, z=0.0):
    """A closed tetrahedron: a piece with no opening."""
    points = [[0, 0, z], [1, 0, z], [0, 1, z], [0, 0, z + 1]]
    return points, [[start + 0, start + 2, start + 1], [start + 0, start + 1, start + 3], [start + 1, start + 2, start + 3], [start + 2, start + 0, start + 3]]


def open_skin(rows=3, count=8):
    """A tube with a neck's opening and a closed top, as a skin."""
    return tube([circle(count, z=float(i)) for i in range(rows)], top=[0, 0, float(rows)])


def test_a_part_with_a_touching_boundary_is_not_read_when_the_skin_is_marked():
    verts, faces = open_skin()
    hair = made_mesh("hair", *bowtie())
    skin = find_skin(made_head(hair, made_mesh("headSkin", verts, faces, {HEAD_REGION: range(len(verts))})))
    assert skin.mesh.name == "headSkin"
    assert len(skin.loop) == 8
    assert skin.skipped == []


def test_a_part_of_the_skins_own_mesh_with_a_touching_boundary_is_not_read_either():
    verts, faces = open_skin()
    points, pinched = bowtie(len(verts))
    mesh = made_mesh("head", np.vstack((verts, points)), faces + pinched, {HEAD_REGION: range(len(verts))})
    skin = find_skin(made_head(mesh))
    assert skin.vertices.tolist() == list(range(len(verts)))
    assert skin.skipped == []


def test_a_marked_skin_with_a_touching_boundary_is_refused_by_its_objects_name():
    with pytest.raises(BrokenSwap, match=f"marked {HEAD_REGION} in faceMesh.*twice") as refused:
        find_skin(made_head(made_mesh("faceMesh", *bowtie(), {HEAD_REGION: [0]})))
    assert "skin" not in str(refused.value)


def test_a_touching_boundary_is_passed_over_when_nothing_is_marked():
    """Though the hair is the largest piece: its boundary can't be a neck."""
    verts, faces = open_skin()
    points, pinched = bowtie()
    many = np.vstack([np.asarray(points) + [0, 0, i] for i in range(20)])
    bows = [[v + 5 * i for v in face] for i in range(20) for face in pinched]
    skin = find_skin(made_head(made_mesh("hair", many, bows), made_mesh("headSkin", verts, faces)))
    assert skin.mesh.name == "headSkin"
    assert skin.skipped == ["hair"]


def test_a_head_with_only_touching_boundaries_is_refused_by_their_objects_names():
    with pytest.raises(BrokenSwap, match="no piece with an opening.*hair") as refused:
        find_skin(made_head(made_mesh("hair", *bowtie()), made_mesh("eyes", *sealed())))
    assert "eyes" not in str(refused.value)


def test_a_marked_piece_with_no_opening_is_refused():
    with pytest.raises(BrokenSwap, match=f"marked {HEAD_REGION} in eyes has no opening"):
        find_skin(made_head(made_mesh("eyes", *sealed(), {HEAD_REGION: [0]})))


def sloped_ring(count):
    """A neck's section: an ellipse, lower at the back, its vertices unevenly spaced, as a body's ring is."""
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    angles = angles + 0.06 * np.sin(3 * angles)
    return np.column_stack((55 * np.cos(angles), 62 * np.sin(angles), 12 * np.sin(angles)))


def neck_body():
    """A body's neck with a head above it, as rows of the ring's section: the neck ring, the head's region."""
    shape = sloped_ring(36)
    verts, faces = tube([shape + [0, 0, 10.0 * i] for i in range(-4, 5)])
    ring, region = 4 * 36, 5 * 36
    groups = {HEAD_REGION: range(region, len(verts))}
    return made_mesh(BODY, verts, faces, groups), np.arange(ring, region), np.arange(region, len(verts))


def neck_head(opening):
    """A head going straight up from its opening, in rows of the ring's section, its own neck in 4 mm rows."""
    shape = sloped_ring(len(opening))
    verts, faces = tube([opening, *(shape + [0, 0, 4.0 * i] for i in range(1, 40))], top=[0, 0, 180.0])
    return made_mesh("headSkin", verts, faces)


def joined(opening):
    body, ring, region = neck_body()
    mesh = neck_head(opening)
    return join_skin(body, ring, region, 200.0, find_skin(made_head(mesh)), mesh.verts)


@pytest.mark.parametrize("count", [72, 50, 37, 24])
@pytest.mark.parametrize("off", [0.0, 0.01])
def test_a_neck_lofted_onto_the_ring_joins_whatever_its_count(count, off):
    """The strip between them starts with no height. The fairing has to give it some, and not hold it
    flat for its edges being short: that left a fold of 65 degrees here and over 90 on a body's ring."""
    join = joined(resampled(sloped_ring(36), count) + [0, 0, off])
    assert not join.welded
    assert largest_crease(join.positions, join.near_faces + join.new_faces, join.final_ring) < 30
    strip = join.new_faces[-(36 + count) :]
    areas = np.linalg.norm(np.cross(*[join.positions[[f[k] for f in strip]] - join.positions[[f[0] for f in strip]] for k in (1, 2)]), axis=1) / 2
    assert areas.min() > 0.5, "the strip stayed flat"


def test_a_neck_lofted_onto_the_ring_with_its_count_welds():
    join = joined(sloped_ring(36))
    assert join.welded
    assert np.array_equal(join.positions[join.final_ring], neck_body()[0].verts[neck_body()[1]])


def folded(opening):
    """The first row above the opening is below it, so that fairing has to turn the faces between them."""
    mesh = neck_head(opening)
    verts = mesh.verts.copy()
    verts[len(opening) : 2 * len(opening), 2] -= 12.0
    return mesh._replace(verts=verts)


@pytest.mark.parametrize(
    ("count", "says", "denies"),
    [(72, "weld to the ring vertex by vertex", "too far"), (36, "isn't the cause", "too far")],
)
def test_a_flip_on_the_ring_is_not_called_too_far_apart(count, says, denies):
    body, ring, region = neck_body()
    mesh = folded(resampled(sloped_ring(36), count) if count != 36 else sloped_ring(36))
    with pytest.raises(BrokenSwap, match="flips") as refused:
        join_skin(body, ring, region, 200.0, find_skin(made_head(mesh)), mesh.verts)
    assert says in str(refused.value)
    assert denies not in str(refused.value)


def test_a_flip_far_from_the_ring_is_called_too_far_apart():
    body, ring, region = neck_body()
    mesh = neck_head(sloped_ring(72) + [0, 0, 40.0])
    with pytest.raises(BrokenSwap, match="flips.*too far apart"):
        join_skin(body, ring, region, 200.0, find_skin(made_head(mesh)), mesh.verts)


def test_a_boundary_is_found_and_a_pinched_one_refused():
    square = [[0, 1, 2, 3]]
    assert sorted(boundary_loops(square)[0]) == [0, 1, 2, 3]
    assert boundary_loops([[0, 1, 2], [0, 2, 1]]) == []
    with pytest.raises(BrokenSwap, match="twice"):
        boundary_loops([[0, 1, 2], [0, 3, 4]])


def test_resampling_a_loop_keeps_its_vertices_on_it():
    loop = circle(72)
    assert np.allclose(np.linalg.norm(resampled(loop, 36), axis=1), 1, atol=1e-3)


# The join -----------------------------------------------------------------------------------------


@pytest.mark.blender
class TestJoin:
    def test_the_seam_is_closed(self, swapped):
        faces = swapped.new.mesh(BODY).faces
        assert all(len(row) == 2 for row in edge_owners(faces).values())
        assert swapped.report["checks"]["openEdges"] == 0

    def test_the_two_kinds_of_neck_are_what_they_say(self, swapped):
        assert swapped.report["welded"] is swapped.coarse
        assert swapped.report["neckLoop"] == (36 if swapped.coarse else 72)

    def test_normals_are_continuous_at_the_ring(self, swapped):
        checks = swapped.report["checks"]
        assert checks["cornerSpreadDeg"] < 0.1, "a ring vertex has more than one normal"
        assert checks["normalStepDeg"] < NORMAL_STEP_DEG[swapped.coarse]
        assert checks["creaseDeg"] < CREASE_DEG

    def test_no_face_of_the_new_head_flips(self, swapped):
        """Every new face still points about the way the head's own nearest face does, the strip and
        the faired neck among them."""
        mesh = swapped.new.mesh(BODY)
        rows = set(new_rows(swapped).tolist())
        faces = [f for f in mesh.faces if rows.intersection(f)]
        skin = find_skin(swapped.head)
        fitted = skin.mesh.verts[skin.vertices] * swapped.report["scale"] + np.array(swapped.report["shiftMm"])
        own = np.array([fitted[f].mean(0) for f in skin.faces])
        nearest = cKDTree(own).query(np.array([mesh.verts[f].mean(0) for f in faces]))[1]
        agree = np.sum(unit_face_normals(mesh.verts, faces) * unit_face_normals(fitted, skin.faces)[nearest], 1)
        assert agree.min() > 0

    def test_the_body_below_the_head_has_not_moved(self, swapped):
        keep = body_rows(swapped)
        new, old = swapped.new.mesh(BODY), swapped.old.mesh(BODY)
        assert np.abs(new.verts[: len(keep)] - old.verts[keep]).max() < 1e-3
        assert new.groups[: len(old.groups)] == old.groups

    def test_the_file_says_where_the_new_head_is(self, swapped):
        mesh = swapped.new.mesh(BODY)
        region = np.sort(mesh.group("headRegion")[0])
        assert region.tolist() == new_rows(swapped).tolist()
        ring = np.sort(mesh.group("neckRing")[0])
        assert ring.tolist() == sorted(attachments(swapped)["neckRing"])
        assert attachments(swapped)["crownVertex"] == int(new_rows(swapped)[np.argmax(mesh.verts[new_rows(swapped), 2])])
        # The ring stays the same 36 vertices in the same order: everything below keeps its number.
        assert attachments(swapped)["neckRing"] == swapped.old.info["head"]["neckRing"]
        assert "referenceWidth" not in attachments(swapped)
        assert swapped.new.info["headSwap"]["mode"] == "whole"

    def test_the_placeholders_are_gone(self, swapped):
        names = {mesh.name for mesh in swapped.new.meshes}
        assert names == {BODY, "headSkin", "eyes", "hair"}


# The fit ------------------------------------------------------------------------------------------


@pytest.mark.blender
class TestFit:
    def test_the_crown_and_the_neck_land_on_the_body(self, swapped):
        new, old = swapped.new.mesh(BODY), swapped.old.mesh(BODY)
        crown = old.verts[swapped.old.info["head"]["crownVertex"], 2]
        top = new.verts[new_rows(swapped), 2].max()
        assert top == pytest.approx(crown, abs=0.5)

    def test_the_head_file_transform_is_undone(self, swapped):
        # The test head is 1.2 times its shape in its file, and that shape is about 5 % taller than ours
        # from the neck's opening to the top (blender/fixtures/made_up_head.py).
        assert swapped.report["scale"] == pytest.approx(1 / 1.2 * 0.95, abs=0.01)

    def test_attachments_are_where_the_head_put_them(self, swapped):
        """The jaw's and the eyes' places in the file are the head's, by the same fit as its skin, and
        their controls with them, the rotation and the head's own pivot unchanged."""
        s, shift = swapped.report["scale"], np.array(swapped.report["shiftMm"])
        for name in ("jaw", "leftEye", "rightEye"):
            placed = np.array(swapped.head.bones[name]["head"]) * 1000 * s + shift
            for bone in (name, f"{name}_ctrl"):
                row = np.array(swapped.new.bones[bone]["matrix"])
                old = np.array(swapped.old.bones[bone]["matrix"])
                assert row[:3, 3] * 1000 == pytest.approx(placed, abs=1e-2)
                axes = row[:3, :3] / np.linalg.norm(row[:3, :3], axis=0)
                old_axes = old[:3, :3] / np.linalg.norm(old[:3, :3], axis=0)
                assert axes == pytest.approx(old_axes, abs=1e-4), f"{bone} turned"
        assert np.allclose(swapped.new.bones["head"]["matrix"], swapped.old.bones["head"]["matrix"], atol=1e-6)

    def test_the_jaw_moved_at_all(self, swapped):
        moved = np.subtract(swapped.new.bones["jaw"]["head"], swapped.old.bones["jaw"]["head"]) * 1000
        assert np.linalg.norm(moved) > 3.0, "the test checks nothing if the head puts the jaw where it was"


# The weights --------------------------------------------------------------------------------------


@pytest.mark.blender
class TestWeights:
    def bones(self, s):
        return [name for name, bone in s.new.bones.items() if bone["deform"]]

    def test_rows_sum_to_one_and_keep_to_four_bones(self, swapped):
        weights = swapped.new.mesh(BODY).weights(self.bones(swapped))[new_rows(swapped)]
        assert np.allclose(weights.sum(1), 1, atol=1e-5)
        assert (np.count_nonzero(weights, axis=1) <= 4).all()

    def test_ring_keeps_its_weights_and_no_jaw(self, swapped):
        names = self.bones(swapped)
        ring = attachments(swapped)["neckRing"]
        new = swapped.new.mesh(BODY).weights(names)[ring]
        old = swapped.old.mesh(BODY).weights(names)[ring]
        assert np.abs(new - old).max() < 1e-6
        assert new[:, names.index("jaw")].max() == 0

    def test_the_jaw_has_the_chin_and_not_the_skull(self, swapped):
        mesh = swapped.new.mesh(BODY)
        rows = new_rows(swapped)
        names = self.bones(swapped)
        jaw = mesh.weights(names)[rows, names.index("jaw")]
        head = mesh.weights(names)[rows, names.index("head")]
        z = mesh.verts[rows, 2]
        top, low = z.max(), z.min()
        assert jaw[z > low + 0.6 * (top - low)].max() == 0
        front = mesh.verts[rows, 1] < mesh.verts[rows, 1].mean()
        chin = (z < low + 0.3 * (top - low)) & front
        assert jaw[chin].max() > 0.5
        assert head[z > low + 0.6 * (top - low)].min() == pytest.approx(1)

    def test_groups_bind_the_teeth_the_tongue_and_the_eyes(self, swapped):
        names = self.bones(swapped)
        skin_part = swapped.new.mesh("headSkin").weights(names)
        assert skin_part[:, names.index("jaw")].sum() > 0 and skin_part[:, names.index("head")].sum() > 0
        assert np.allclose(skin_part.sum(1), 1)
        eyes = swapped.new.mesh("eyes").weights(names)
        assert set(np.flatnonzero(eyes.sum(0)).tolist()) == {names.index("leftEye"), names.index("rightEye")}
        assert np.allclose(eyes.max(1), 1)

    def test_hair_with_no_groups_takes_the_skin_it_lies_on(self, swapped):
        """The cap's faces point in, and it still matches; the strand reaches no skin and moves as one."""
        names = self.bones(swapped)
        hair = swapped.new.mesh("hair")
        weights = hair.weights(names)
        assert np.allclose(weights.sum(1), 1, atol=1e-5)
        top = hair.verts[:, 2] > np.percentile(hair.verts[:, 2], 50)
        assert weights[top, names.index("head")].min() > 0.99
        far = hair.verts[:, 2] < swapped.new.mesh(BODY).verts[attachments(swapped)["neckRing"], 2].min()
        assert far.any()
        assert np.ptp(weights[far], axis=0).max() < 1e-3
        assert swapped.report["partsMatched"] > 0.5

    def test_the_skin_is_mostly_matched(self, swapped):
        assert swapped.report["skinMatched"] > 0.9


# The poses ----------------------------------------------------------------------------------------


@pytest.mark.blender
class TestPoses:
    def test_no_python_goes_into_the_file(self, swapped):
        report = swapped.pose_report
        assert report["autoexec"] is False
        assert report["autoexecFail"] is False
        assert report["texts"] == 0
        assert report["notSimple"] == []
        assert report["drivers"] == swapped.old_pose_report["drivers"]
        assert report["constraints"] == swapped.old_pose_report["constraints"]

    def test_turn_nod_and_mouth_collapse_nothing(self, swapped):
        rest = RestMesh(np.asarray(swapped.new_posed[f"rest:{BODY}"], float), swapped.new.mesh(BODY).faces)
        for name in CHECK_POSES:
            row = pose_row(rest, np.asarray(swapped.new_posed[f"on:{name}:{BODY}"], float))
            assert row["collapsedFaces"] == 0, name

    def test_the_body_moves_as_it_did(self, swapped):
        """Below the head the swapped file poses as the original, correctives and all."""
        keep = body_rows(swapped)
        for name in (*CHECK_POSES, *CORRECTIVE_POSES):
            old = swapped.old_posed[f"on:{name}:{BODY}"][keep]
            new = swapped.new_posed[f"on:{name}:{BODY}"][: len(keep)]
            assert np.linalg.norm(old - new, axis=1).max() < SKIN_MM, name
        # The head's own shape key comes after the correctives, which are as they were.
        old_keys = swapped.old_posed["on:crouch:keys"]
        assert np.abs(swapped.new_posed["on:crouch:keys"][: len(old_keys)] - old_keys).max() < 1e-5

    def test_the_lower_teeth_and_tongue_go_with_the_jaw(self, swapped):
        _mesh, rows = part_rows(swapped, ("headSkin", "jaw"))
        rest = swapped.new_posed["rest:headSkin"][rows]
        opened = swapped.new_posed["on:jawOpen:headSkin"][rows]
        assert np.linalg.norm(opened - rest, axis=1).min() > 1.0
        # Rigid: every distance between them stays.
        a, b = rest[:: 7], opened[:: 7]
        distance = lambda p: np.linalg.norm(p[:, None] - p[None], axis=2)
        assert np.abs(distance(a) - distance(b)).max() < 1e-2

    def test_the_upper_teeth_stay_with_the_head(self, swapped):
        _mesh, rows = part_rows(swapped, ("headSkin", "head"))
        assert np.linalg.norm(swapped.new_posed["on:jawOpen:headSkin"][rows] - swapped.new_posed["rest:headSkin"][rows], axis=1).max() < 1e-3
        moved = swapped.new_posed["on:headTurn:headSkin"][rows]
        assert np.linalg.norm(moved - swapped.new_posed["rest:headSkin"][rows], axis=1).min() > 1.0

    def test_the_eyes_turn_about_their_bones(self, swapped):
        """Each eyeball turns on the spot in the head's frame: its distance from its bone doesn't change."""
        names = swapped.pose_report["deform"]
        for bone in ("leftEye", "rightEye"):
            rows = part_rows(swapped, ("eyes", bone))[1]
            rest = swapped.new_posed["rest:eyes"][rows]
            at = swapped.new_posed["on:gaze:bones"][names.index(bone)][:3, 3] * 1000
            posed = swapped.new_posed["on:gaze:eyes"][rows]
            pivot = np.array(swapped.new.bones[bone]["head"]) * 1000
            before, after = np.linalg.norm(rest - pivot, axis=1), np.linalg.norm(posed - at, axis=1)
            assert np.abs(before - after).max() < 1e-2
            assert np.ptp(before) < 1e-2, "the eye isn't a ball around its bone"

    def test_a_shape_key_of_the_head_comes_along(self, swapped):
        report = swapped.pose_report
        assert "blink" in report["keyNames"]
        index = report["keyNames"].index("blink")
        assert swapped.new_posed["on:headTurn:keys"][index] == 0


# The other modes ----------------------------------------------------------------------------------


NAME = "female-compact"


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    folder = tmp_path_factory.mktemp("modes")
    blend = blend_body(NAME, folder / "body.blend")
    return folder, blend


@pytest.mark.blender
def test_parts_only_keeps_the_skin(files):
    """Only the face and the hair change: the skin isn't touched, the parts are bound and the bones move."""
    folder, blend = files
    head = make_head(blend, folder / "parts.blend", "--no-skin")
    out = folder / "parts-out.blend"
    assert command(["swap-head", str(blend), str(head), str(out), "--parts-only", "--keep-parts", "ears"]) == 0
    old, new = read_dump(blend, folder / "o"), read_dump(out, folder / "n")
    assert np.array_equal(new.mesh(BODY).verts, old.mesh(BODY).verts)
    assert new.mesh(BODY).faces == old.mesh(BODY).faces
    assert {m.name for m in new.meshes} == {BODY, "ears", "headSkin", "eyes", "hair"}
    assert new.info["head"] == old.info["head"]
    assert not np.allclose(new.bones["leftEye"]["head"], old.bones["leftEye"]["head"], atol=1e-4)
    assert np.allclose(new.mesh("hair").weights(["head"]).max(), 1)


@pytest.mark.blender
def test_the_command_writes_the_file_and_its_report(files, capsys):
    folder, blend = files
    head = make_head(blend, folder / "head.blend")
    out = folder / "out.blend"
    assert command(["swap-head", str(blend), str(head), str(out)]) == 0
    assert out.exists()
    assert not list(folder.glob("head-swap-*")), "the working folder should be gone"
    report = json.loads(out.with_suffix(".json").read_text())
    assert report["mode"] == "whole"
    assert "zipped" in capsys.readouterr().out
    info = read_dump(out, folder / "read").info
    assert {"code": "head_swap.touch_stale", "level": "warning", "values": {}} in info["quality"]["found"]
    assert "headSwap" in info["quality"]["checked"]
    assert info["headSwap"]["creaseDeg"] == pytest.approx(report["checks"]["creaseDeg"])


@pytest.mark.blender
def test_a_head_with_no_opening_writes_nothing(files, capsys):
    """Hair alone has no neck to join: refused, by name, and no file."""
    folder, blend = files
    head = make_head(blend, folder / "closed.blend", "--no-skin", "--closed")
    out = folder / "refused.blend"
    assert command(["swap-head", str(blend), str(head), str(out)]) == 1
    assert not out.exists()
    assert "not written" in capsys.readouterr().err


@pytest.mark.blender
@pytest.mark.parametrize("marked", [True, False])
def test_hair_with_a_touching_boundary_stays_a_part(files, marked):
    """Generated hair: it goes along as a part whether the skin is marked or has to be found."""
    folder, blend = files
    tag = "marked" if marked else "unmarked"
    head = make_head(blend, folder / f"pinched-{tag}.blend", "--pinched-hair", *(["--marked"] if marked else []))
    report = swap_head(blend, head, folder / f"pinched-{tag}-out.blend")
    assert report["source"] == "headSkin"
    new = read_dump(folder / f"pinched-{tag}-out.blend", folder / f"pinched-{tag}-read")
    assert np.allclose(new.mesh("hair").weights(["head"]).max(), 1)
    assert any(row["code"] == "head_swap.kept_parts" for row in report["quality"]) is not marked


@pytest.mark.blender
def test_a_neck_on_the_ring_joins_on_a_real_file(files):
    """A zipped neck with half its vertices on the ring's: the fit given leaves it there. The fold and the
    step in the normals are what the strip with no height made, 72 and 44 degrees before the fairing gave
    it some. The row above the opening is wider than the ring, so the crease stands above CREASE_DEG."""
    folder, blend = files
    head = make_head(blend, folder / "on-ring.blend", "--on-ring")
    report = swap_head(blend, head, folder / "on-ring-out.blend", scale=1 / 1.2)
    assert report["welded"] is False
    assert report["checks"]["openEdges"] == 0
    assert report["checks"]["creaseDeg"] < 40
    assert report["checks"]["normalStepDeg"] < NORMAL_STEP_DEG[False]


@pytest.mark.blender
def test_what_the_author_painted_on_the_skin_stays(files):
    """Away from the join the skin keeps its author's weights: nothing else would give it these."""
    folder, blend = files
    head = make_head(blend, folder / "weighted.blend", "--weighted")
    out = folder / "weighted-out.blend"
    swap_head(blend, head, out)
    new = read_dump(out, folder / "weighted-read")
    mesh = new.mesh(BODY)
    first, count = new.info["head"]["region"]
    rows = np.arange(first, first + count)
    high = rows[mesh.verts[rows, 2] > mesh.verts[rows, 2].min() + 0.6 * np.ptp(mesh.verts[rows, 2])]
    weights = mesh.weights(["head", "neck"])[high]
    assert np.abs(weights - 0.5).max() < 1e-5


@pytest.mark.blender
def test_a_scale_given_is_the_scale_used(files):
    folder, blend = files
    head = make_head(blend, folder / "scaled.blend")
    out = folder / "scaled-out.blend"
    fitted = swap_head(blend, head, folder / "fitted.blend")["scale"]
    report = swap_head(blend, head, out, scale=fitted * 0.95)
    assert report["scale"] == pytest.approx(fitted * 0.95)
    new, old = read_dump(out, folder / "scaled-read"), read_dump(blend, folder / "scaled-old")
    crown = old.mesh(BODY).verts[old.info["head"]["crownVertex"], 2]
    assert new.mesh(BODY).verts[new.info["head"]["crownVertex"], 2] < crown - 5


@pytest.mark.blender
def test_keep_parts_must_name_something(files):
    folder, blend = files
    head = make_head(blend, folder / "again.blend")
    with pytest.raises(BrokenSwap, match="keep-parts"):
        swap_head(blend, head, folder / "never.blend", keep_parts=("nothing",))
