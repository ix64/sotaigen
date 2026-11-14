"""The head's parts: the same mesh in every body, on the skin, moving with it. No Blender."""

import json
import struct

import numpy as np
import pytest

from sotaigen.form.bodies import load_bodies, load_specs
from sotaigen.mesh import unit_face_normals
from sotaigen.output.face import (
    EXPRESSIONS,
    EYE_SHELLS,
    LAYERS,
    cap_mesh,
    face_parts,
    head_normals,
    head_shape,
    lock_mesh,
    shell_shapes,
    unit_rows,
)
from sotaigen.output.glb import glb_bytes, part_weights
from sotaigen.pipeline import rigged
from sotaigen.rig.skinning import pose_skins, pose_verts, triangle_fan

ALL = list(load_bodies())
# How far an open eye's shells may stand off the skin, as a share of the head.
EYE_DOME = 0.03
SOFT = [n for n in ALL if n in load_specs("typical")]
# How far a shell may stand beyond its layer: it is cast on the smooth surface, and the skin's mesh
# cuts inside that surface across its facets, in mm.
FACET_MM = 0.35
VOLUMES = {"nose", "ears", "hair"}


def parts_of(name):
    return {part.name: part for part in face_parts(name)}


def closest_on_triangles(points, corners, candidates=16):
    """For each point, the closest point among its nearest triangles, and that triangle's index.

    The nearest by centroid are enough on a mesh of near-even triangles. Each is tried at the point's
    projection clipped into it and at the closest point of its three edges.
    """
    centres = corners.mean(axis=1)
    nearest = np.empty((len(points), candidates), int)
    for at in range(0, len(points), 256):
        distance = np.linalg.norm(points[at : at + 256, None] - centres, axis=2)
        nearest[at : at + 256] = np.argpartition(distance, candidates, axis=1)[:, :candidates]
    a, b, c = (corners[nearest, i] for i in range(3))
    p = points[:, None]
    ab, ac = b - a, c - a
    ap = p - a
    d1, d2 = np.sum(ab * ap, axis=2), np.sum(ac * ap, axis=2)
    aa, bb, ab_ac = np.sum(ab * ab, axis=2), np.sum(ac * ac, axis=2), np.sum(ab * ac, axis=2)
    denominator = np.maximum(aa * bb - ab_ac * ab_ac, 1e-12)
    v, w = (bb * d1 - ab_ac * d2) / denominator, (aa * d2 - ab_ac * d1) / denominator
    inside = (v >= 0) & (w >= 0) & (v + w <= 1)
    tries = [np.where(inside[..., None], a + v[..., None] * ab + w[..., None] * ac, np.inf)]
    for start, along in ((a, ab), (a, ac), (b, c - b)):
        length = np.maximum(np.sum(along * along, axis=2), 1e-12)
        share = np.clip(np.sum((p - start) * along, axis=2) / length, 0, 1)
        tries.append(start + share[..., None] * along)
    tries = np.stack(tries, axis=2)
    gap = np.linalg.norm(p[:, :, None] - tries, axis=3)
    gap = np.where(np.isnan(gap), np.inf, gap)
    flat = gap.reshape(len(points), -1).argmin(axis=1)
    which, kind = np.unravel_index(flat, gap.shape[1:])
    rows = np.arange(len(points))
    return tries[rows, which, kind], nearest[rows, which]


def skin_triangles(name):
    """The corners and unit normals of the head's triangles."""
    rig = rigged(name)
    triangles = triangle_fan(rig.base["faces"])
    head = np.asarray(rig.base["segments"]) == "head"
    triangles = triangles[head[triangles].all(axis=1)]
    corners = rig.verts[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return corners, unit_rows(normals)


def height_over_skin(points, name):
    """Signed distance from the head's mesh in mm, positive outside."""
    corners, normals = skin_triangles(name)
    found, index = closest_on_triangles(points, corners)
    offset = points - found
    sign = np.sign(np.sum(offset * normals[index], axis=1))
    return np.linalg.norm(offset, axis=1) * sign


def centroids(verts, faces):
    return np.array([verts[face].mean(axis=0) for face in faces])


@pytest.mark.slow
def test_parts_have_the_same_mesh_in_every_preset():
    first = parts_of(ALL[0])
    for name in ALL[1:]:
        parts = parts_of(name)
        assert list(parts) == list(first)
        for key, part in parts.items():
            assert len(part.verts) == len(first[key].verts), f"{name} {key}"
            assert part.faces == first[key].faces, f"{name} {key}"
            assert set(part.morphs) == set(first[key].morphs), f"{name} {key}"


@pytest.mark.parametrize("name", SOFT)
def test_the_expressions_have_the_right_shape_and_move_what_they_should(name):
    parts = parts_of(name)
    for part in parts.values():
        for target, offsets in part.morphs.items():
            assert target in EXPRESSIONS
            assert offsets.shape == part.verts.shape
    moves = {target: {p.name for p in parts.values() if target in p.morphs} for target in EXPRESSIONS}
    assert moves["blink"] == {"eyeWhite", "iris", "pupil", "eyeLight", "lash"}
    assert moves["happy"] == {"eyeWhite", "iris", "pupil", "eyeLight", "lash", "brow", "mouth"}
    assert moves["aa"] == {"mouth"}
    assert not any(parts[n].morphs for n in VOLUMES), "volumes do not take part in expressions"


@pytest.mark.parametrize("name", SOFT)
def test_looking_aside_moves_both_eyes_the_same_way(name):
    rig = rigged(name)
    bones = [bone["name"] for bone in rig.bones]
    iris = parts_of(name)["iris"]
    half = len(iris.verts) // 2
    for angle, sign in ((12.0, 1), (-12.0, -1)):
        pose = {"leftEye": (0.0, 0.0, angle), "rightEye": (0.0, 0.0, angle)}
        offsets = pose_verts(rig.bones, iris.weights, iris.verts, pose) - iris.verts
        # Toward the wearer's left is +X, on the wearer's left eye and on the right one alike.
        assert sign * offsets[:half, 0].mean() > 0
        assert sign * offsets[half:, 0].mean() > 0
    assert (iris.weights[:half, bones.index("leftEye")] == 1).all()
    assert (iris.weights[half:, bones.index("rightEye")] == 1).all()


@pytest.mark.parametrize("name", SOFT)
def test_an_open_eye_stays_on_its_sphere_as_it_turns_and_shuts_onto_the_skin(name):
    rig = rigged(name)
    eye = next(bone for bone in rig.bones if bone["name"] == "leftEye")
    radius = np.linalg.norm(np.subtract(eye["tail"], eye["head"]))
    iris = parts_of(name)["iris"]
    half = len(iris.verts) // 2
    pose = {"leftEye": (8.0, 0.0, 12.0)}
    posed = pose_verts(rig.bones, iris.weights, iris.verts, pose)[:half]
    assert np.allclose(np.linalg.norm(posed - eye["head"], axis=1), radius, atol=1e-6)
    # The eye's shells are ahead of the skin at rest: the whole iris is seen.
    head = head_shape(name)
    u, t = shell_shapes(None)["iris"]
    skin = head.on_front(u, t)
    assert ((skin - iris.verts)[:, 1] > 0).all()
    # Shut, it lies on the skin at its layer, under the lash.
    shut = (iris.verts + iris.morphs["blink"])
    u, t = shell_shapes("blink")["iris"]
    assert np.allclose(shut, head.on_front(u, t, LAYERS["iris"]), atol=1e-6)


@pytest.mark.parametrize("name", SOFT)
def test_the_mouth_opens_on_the_jaw(name):
    rig = rigged(name)
    mouth = parts_of(name)["mouth"]
    rows = mouth.verts.reshape(3, -1, 3)
    posed = pose_verts(rig.bones, mouth.weights, mouth.verts, {"jaw": (20.0, 0.0, 0.0)}).reshape(3, -1, 3)
    middle = rows.shape[1] // 2
    gap_before = rows[0, middle, 2] - rows[2, middle, 2]
    gap_after = posed[0, middle, 2] - posed[2, middle, 2]
    assert gap_after > gap_before + 0.02 * rig.L["head"]
    assert np.isclose(mouth.weights.sum(axis=1), 1.0).all()


@pytest.mark.parametrize("name", SOFT)
def test_a_blink_closes_the_eye_to_a_line(name):
    white = parts_of(name)["eyeWhite"]
    half = len(white.verts) // 2
    closed = (white.verts + white.morphs["blink"])[:half]
    open_span = np.ptp(white.verts[:half, 2])
    width = np.ptp(white.verts[:half, 0])
    rows = closed.reshape(5, -1, 3)
    assert np.abs(rows[0] - rows[-1]).max() < 0.02 * width
    assert np.ptp(closed[:, 2]) < open_span


@pytest.mark.parametrize("name", SOFT)
@pytest.mark.parametrize("target", (None, *EXPRESSIONS))
def test_the_shells_lie_on_the_skin_at_their_layer(name, target):
    """Not through the skin and not off it, at the vertices and across the faces, in every shape.

    An open eye's shells are the exception to "not off it": they lie on the eye bone's sphere, which
    stands off the skin at its pole, so that a turn keeps them on one surface.
    """
    h = rigged(name).L["head"]
    for part in face_parts(name):
        if part.name in VOLUMES or (target is not None and target not in part.morphs):
            continue
        verts = part.verts + part.morphs.get(target, 0.0)
        layer = LAYERS[part.name]
        if target is None and part.name in EYE_SHELLS:
            layer = EYE_DOME * h
        for points in (verts, centroids(verts, part.faces)):
            heights = height_over_skin(points, name)
            assert heights.min() > 0.0, f"{part.name} goes through the skin: {heights.min():.3f}"
            assert heights.max() < layer + FACET_MM, f"{part.name} is off the skin: {heights.max():.3f}"


@pytest.mark.parametrize("name", SOFT)
def test_the_volumes_are_rooted_in_the_skin_and_stay_near_it(name):
    h = rigged(name).L["head"]
    parts = parts_of(name)
    limits = {"nose": 0.04, "ears": 0.16, "hair": 0.16}
    for key, limit in limits.items():
        heights = height_over_skin(parts[key].verts, name)
        assert heights.min() < 0, f"{key} does not reach the skin"
        assert heights.max() < limit * h, f"{key} flies off: {heights.max() / h:.3f} of the head"


@pytest.mark.parametrize("name", SOFT)
def test_the_cap_clears_the_skin_except_at_its_folded_rim(name):
    verts, _, rim = cap_mesh(head_shape(name))
    heights = height_over_skin(verts, name)
    assert heights[~rim].min() >= 1.0, f"{heights[~rim].min():.2f} mm"
    assert heights[rim].min() < 0, "the rim does not fold under the skin"


@pytest.mark.parametrize("name", ALL)
def test_faces_look_outward(name):
    """Backface culling must keep the side the camera sees."""
    head = head_shape(name)
    for part in face_parts(name):
        if part.name in VOLUMES:
            continue
        normals = unit_face_normals(part.verts, part.faces)
        out = head.skin_normal(centroids(part.verts, part.faces))
        assert (np.sum(normals * out, axis=1) > 0).all(), part.name
    cap, faces, _ = cap_mesh(head)
    outward = unit_rows(centroids(cap, faces) - np.array([0.0, 0.0, head.H - 0.52 * head.h]))
    assert (np.sum(unit_face_normals(cap, faces) * outward, axis=1) > -0.2).mean() > 0.98


def signed_volume(verts, faces):
    total = 0.0
    for face in faces:
        for j in range(1, len(face) - 1):
            total += np.dot(verts[face[0]], np.cross(verts[face[j]], verts[face[j + 1]]))
    return total / 6


def test_a_lock_is_a_solid_turned_outward():
    line = np.column_stack((np.zeros(9), np.zeros(9), np.linspace(100.0, 0.0, 9)))
    widths = np.linspace(5.0, 0.0, 9)
    outward = np.tile((0.0, -1.0, 0.0), (9, 1))
    verts, faces = lock_mesh(line, widths, widths / 2, outward)
    assert signed_volume(verts, faces) > 0


def test_ears_are_solids_turned_outward():
    part = parts_of("female-long")["ears"]
    half = len(part.verts) // 2
    assert signed_volume(part.verts[:half], part.faces[: len(part.faces) // 2]) > 0
    shifted = [[i - half for i in face] for face in part.faces[len(part.faces) // 2 :]]
    assert signed_volume(part.verts[half:], shifted) > 0


@pytest.mark.parametrize("name", SOFT)
@pytest.mark.parametrize("pose", ["headTurn", "headNod"])
def test_the_parts_follow_the_skin_when_the_head_turns_and_nods(name, pose):
    """With the jaw and the eyes at rest, every part keeps its place over the skin that is purely the head's."""
    from sotaigen.rig.poses import POSES

    rig = rigged(name)
    columns = [bone["name"] for bone in rig.bones]
    head = columns.index("head")
    posed_skin = pose_verts(rig.bones, rig.weights, rig.verts, POSES[pose])
    pure = np.flatnonzero(rig.weights[:, head] > 1 - 1e-6)
    assert len(pure) > 1000
    for part in face_parts(name):
        moved = pose_verts(rig.bones, part_weights(part, columns), part.verts, POSES[pose])
        # The same rigid move, measured against the skin around it.
        sample = part.verts[:: max(1, len(part.verts) // 60)]
        sample_moved = moved[:: max(1, len(part.verts) // 60)]
        distance = np.linalg.norm(rig.verts[pure][None] - sample[:, None], axis=2)
        nearest = pure[distance.argmin(axis=1)]
        before = np.linalg.norm(sample - rig.verts[nearest], axis=1)
        after = np.linalg.norm(sample_moved - posed_skin[nearest], axis=1)
        assert np.abs(before - after).max() < 1e-6, part.name


@pytest.mark.parametrize("name", SOFT)
def test_the_hair_stays_out_of_the_neck_in_a_nod(name):
    """The back's locks swing with the head. They may touch the neck, not sink far into it."""
    from sotaigen.rig.poses import POSES

    rig = rigged(name)
    hair = parts_of(name)["hair"]
    head = [bone["name"] for bone in rig.bones].index("head")
    skins = pose_skins(rig.bones, POSES["headNod"])
    moved = (np.column_stack((hair.verts, np.ones(len(hair.verts)))) @ skins[head].T)[:, :3]
    posed = pose_verts(rig.bones, rig.weights, rig.verts, POSES["headNod"])
    triangles = triangle_fan(rig.base["faces"])
    near = np.asarray(rig.base["segments"]) == "head"
    near |= np.isin(np.asarray(rig.base["segments"]), ["neck"])
    triangles = triangles[near[triangles].all(axis=1)]
    corners = posed[triangles]
    normals = unit_rows(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]))
    # Only the locks' lower halves can reach the neck, so only those are asked about.
    t = (rig.L["H"] - hair.verts[:, 2]) / rig.L["head"]
    low = moved[t > 0.85]
    found, index = closest_on_triangles(low, corners)
    depth = -np.sum((low - found) * normals[index], axis=1)
    assert depth.max() < 2.0, f"{depth.max():.1f} mm into the neck"


@pytest.mark.parametrize("name", SOFT)
def test_head_normals_cover_the_skin_and_leave_the_neck_alone(name):
    from sotaigen.mesh import skin_normals

    rig = rigged(name)
    normals = head_normals(name)
    assert normals.shape == rig.verts.shape
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    own = skin_normals(rig.base, rig.verts)
    head = np.asarray(rig.base["segments"]) == "head"
    assert np.array_equal(normals[~head], own[~head]), "only the head's skin changes"
    assert not np.allclose(normals[head], own[head]), "the head's skin does change"


@pytest.mark.parametrize("name", SOFT)
def test_head_normals_turn_gradually_to_the_necks_at_the_ring(name):
    """No jump across the seam with the template's neck, nor anywhere along the turn."""
    from sotaigen.mesh import skin_normals

    rig = rigged(name)
    normals = head_normals(name)
    own = skin_normals(rig.base, rig.verts)
    head = np.flatnonzero(np.asarray(rig.base["segments"]) == "head")
    first_row = head[:72]
    # In float64 and by arctan2: an arccos of float32 normals can only say 0 or 0.0198 degrees down here.
    a, b = normals[first_row].astype(float), own[first_row].astype(float)
    angle = np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a * b, axis=1)))
    assert angle.max() < 0.01, f"{angle.max():.1f} degrees at the row next to the seam"
    # Between neighbours, not much sharper than the skin's own turn under the jaw.
    triangles = triangle_fan(rig.base["faces"])
    mask = np.zeros(len(rig.verts), bool)
    mask[head] = True
    triangles = triangles[mask[triangles].all(axis=1)]
    edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))

    def sharpest(values):
        dots = np.sum(values[edges[:, 0]] * values[edges[:, 1]], axis=1)
        return np.degrees(np.arccos(np.clip(dots, -1, 1))).max()

    assert sharpest(normals) < sharpest(own) + 15


def test_the_glb_with_parts_carries_every_part_and_target():
    name = "female-long"
    rig = rigged(name)
    parts = face_parts(name)
    blob = glb_bytes(rig.base, rig.verts, rig.bones, rig.weights, parts=parts, normals=head_normals(name))
    plain = glb_bytes(rig.base, rig.verts, rig.bones, rig.weights)
    assert len(blob) > len(plain)
    for part in parts:
        assert part.name.encode() in blob
    for target in EXPRESSIONS:
        assert target.encode() in blob


def test_nose_and_ears_are_the_skin_of_the_file():
    """They are separate meshes, but must read as the same skin as the exported body."""
    name = "female-long"
    rig = rigged(name)
    blob = glb_bytes(rig.base, rig.verts, rig.bones, rig.weights, parts=face_parts(name))
    length = struct.unpack_from("<I", blob, 12)[0]
    document = json.loads(blob[20 : 20 + length])
    colors = {
        mesh["name"]: document["materials"][mesh["primitives"][0]["material"]][
            "pbrMetallicRoughness"
        ]["baseColorFactor"]
        for mesh in document["meshes"]
    }
    assert colors["nose"] == pytest.approx(colors["body"])
    assert colors["ears"] == pytest.approx(colors["body"])
