"""The rig on wrapped bodies: weights, helper bones, and what posing does to the arms."""


import numpy as np
import pytest

from sotaigen.corrective.crossings import SkinParts, edges_through_triangles
from sotaigen.form.anatomy import landmarks, skin_reference
from sotaigen.form.bodies import load_bodies
from sotaigen.form.wrap import wrap_body
from sotaigen.mesh import edges_of, face_areas
from sotaigen.metrics import joint_report
from sotaigen.output.glb import glb_bytes, read_glb, to_gltf
from sotaigen.pipeline import BUILD, bodies, rigged, surface
from sotaigen.rig import skinning
from sotaigen.rig.poses import POSES
from sotaigen.rig.skeleton import check_lower_body, diffuse, lower_zone, skeleton, skin_weights
from sotaigen.rig.skinning import (
    FORWARD_LIFT,
    HALF_BONES,
    HALF_ROLLS,
    SHOULDER_RHYTHM,
    TWISTS,
    euler,
    pose_skins,
    pose_verts,
    sided,
)
from sotaigen.template import load_template
from sotaigen_dev.motion import clips as motions

# Every invariant here holds for every body: a style must not break the rig.
NAMES = list(load_bodies())



def test_diffusing_adds_up_edge_by_edge_to_the_last_bit():
    rng = np.random.default_rng(7)
    edges = rng.integers(0, 40, (300, 2))
    edges = edges[edges[:, 0] != edges[:, 1]]
    start = rng.random((40, 5))
    rate = rng.random(40)
    expected = start
    degree = np.bincount(edges.ravel(), minlength=40)[:, None]
    for _ in range(3):
        total = np.zeros_like(expected)
        np.add.at(total, edges[:, 0], expected[edges[:, 1]])
        np.add.at(total, edges[:, 1], expected[edges[:, 0]])
        expected = expected + (total / degree - expected) * rate[:, None]
    assert np.array_equal(diffuse(start, edges, rate, 3), expected)


@pytest.mark.parametrize("name", ["leftUpperArn", "upperArn", "Chest", "rightLeg", "root"])
def test_a_pose_naming_no_bone_is_refused(name):
    rig = rigged(NAMES[0])
    verts, bones, weights = rig.verts, rig.bones, rig.weights
    with pytest.raises(ValueError, match=name):
        pose_verts(bones, weights, verts, {name: (10, 0, 0)})


@pytest.mark.parametrize("name", ["upperArm", "leftUpperArm", "spine"])
def test_a_pose_naming_a_bone_or_a_limb_is_taken(name):
    rig = rigged(NAMES[0])
    verts, bones, weights = rig.verts, rig.bones, rig.weights
    assert not np.allclose(pose_verts(bones, weights, verts, {name: (10, 0, 0)}), verts)


# A triangle in the plane z = 0, and an edge from below to above it.
TRIANGLE = np.array([[0, 1, 2]])
TRIANGLE_POINTS = [[0, 0, 0], [4, 0, 0], [0, 4, 0]]


@pytest.mark.parametrize(
    ("ends", "crossings"),
    [
        ([[1, 1, -1], [1, 1, 1]], 1),  # through the middle
        ([[1, 1, 0.5], [1, 1, 1]], 0),  # stops above the plane
        ([[5, 5, -1], [5, 5, 1]], 0),  # goes past the triangle
        ([[1, 1, 0.5], [3, 3, 0.5]], 0),  # runs parallel
    ],
)
def test_an_edge_counts_only_where_it_passes_through_a_triangle(ends, crossings):
    points = np.array(TRIANGLE_POINTS + ends, float)
    edges = np.array([[3, 4]])
    assert len(edges_through_triangles(points, edges, TRIANGLE)) == crossings


def test_an_edge_sharing_a_corner_with_a_triangle_is_not_a_crossing():
    points = np.array(TRIANGLE_POINTS + [[1, 1, 1]], float)
    assert len(edges_through_triangles(points, np.array([[0, 3]]), TRIANGLE)) == 0


def stub_parts(segments, verts, faces):
    return SkinParts({"segments": segments, "verts": verts, "faces": faces})


def test_opposite_limbs_are_told_from_one_limb_and_labels_are_stable():
    # A square on the left thigh, and a square of the right thigh pushed through it.
    left = [[1, 0, 0], [3, 0, 0], [3, 2, 0], [1, 2, 0]]
    right = [[-1, 1, -1], [-1, 1, 1], [-3, 1, 1], [-3, 1, -1]]
    verts = left + right
    parts = stub_parts(["upperLeg"] * 8, verts, [[0, 1, 2, 3], [4, 5, 6, 7]])
    posed = np.array(verts, float)
    posed[4:, 0] += 3  # the right thigh's square now spans x 0..2, through the left one
    report = parts.intersections(posed)
    assert report["selfIntersections"] > 0
    assert list(report["intersectionsByPair"]) == ["upperLeg<>upperLeg"]


def test_an_edge_is_tested_against_the_quads_exported_short_diagonal():
    # A quad folded along its short diagonal 1-3: a roof over x = 0, with a valley under the long one.
    quad = [[-3, 0, 0], [0, -1, 1], [3, 0, 0], [0, 1, 1]]
    # A probe that reaches only the roof, far above the long diagonal's surface.
    probe = [[0.5, 0, 0.5], [0.5, 0, 1.5], [0.6, 0, 1.5]]
    parts = stub_parts(["upperLeg"] * 7, quad + probe, [[0, 1, 2, 3], [4, 5, 6]])
    assert parts.intersections(np.array(quad + probe, float))["selfIntersections"] > 0
    edges = {tuple(edge) for edge in parts.edges.tolist()}
    assert (1, 3) in edges and (0, 2) not in edges


def test_self_intersections_use_the_triangles_in_the_exported_glb():
    rig = rigged("female-long")
    document, data = read_glb(glb_bytes(rig.base, rig.verts, rig.bones, rig.weights))
    primitive = next(m for m in document["meshes"] if m["name"] == "body")["primitives"][0]

    def stored(index, kind, width):
        item = document["accessors"][index]
        view = document["bufferViews"][item["bufferView"]]
        start = view.get("byteOffset", 0) + item.get("byteOffset", 0)
        return np.frombuffer(data, kind, item["count"] * width, start).reshape(-1, width)

    kinds = {5123: "<u2", 5125: "<u4"}
    indices = primitive["indices"]
    triangles = stored(indices, kinds[document["accessors"][indices]["componentType"]], 1)
    positions = stored(primitive["attributes"]["POSITION"], "<f4", 3)
    exported = positions[triangles.reshape(-1, 3)]
    measured = to_gltf(rig.verts).astype("<f4")[SkinParts(rig.base).triangles]
    assert np.array_equal(exported, measured)


def test_legs_past_the_midline_go_by_the_template_side_and_only_count_legs():
    verts = [[1, 0, 0], [-1, 0, 0], [1, 0, 1], [0, 0, 2]]
    parts = stub_parts(["upperLeg", "lowerLeg", "upperArm", "trunk"], verts, [[0, 1, 2, 3]])
    assert parts.legs_past_midline(np.array(verts, float)) == 0
    # Both legs cross; the arm crossing with them is not a leg.
    crossed = np.array([[-1, 0, 0], [1, 0, 0], [-1, 0, 1], [0, 0, 2]], float)
    assert parts.legs_past_midline(crossed) == 2


@pytest.mark.parametrize("pose", ["a pose of any name", "wristRoll"])
def test_girth_is_measured_whatever_a_pose_is_called(pose):
    rig = rigged(NAMES[0])
    base, verts, L, bones, weights = rig.base, rig.verts, rig.L, rig.bones, rig.weights
    shoulder = L["shoulderJoint"]
    poses = {pose: {"hand": (0, 90, 0)}}
    rows = joint_report(base, verts, bones, weights, shoulder, poses, surface(NAMES[0])[0])
    assert rows["poses"][pose]["girth"]["wrist"] >= 0.9


@pytest.mark.parametrize("name", NAMES)
def test_the_hip_joint_stands_right_above_the_knee_ring(name):
    rig = rigged(name)
    base, verts, bones = rig.base, rig.verts, rig.bones
    by_name = {bone["name"]: bone for bone in bones}
    knee = verts[base["rings"]["knee"]].mean(0)
    assert by_name["leftUpperLeg"]["head"][0] == pytest.approx(knee[0])
    assert by_name["rightUpperLeg"]["head"][0] == pytest.approx(-knee[0])
    assert by_name["leftUpperLegHalf"]["head"][0] == pytest.approx(knee[0])


@pytest.mark.parametrize(("name", "ratio"), [("female-chibi", 0.54), ("female-balanced", 0.66)])
def test_redistributing_leg_length_moves_the_wrapped_knee_and_its_hinge(name, ratio, tmp_path):
    rig = rigged(name)
    base, before, old = rig.base, rig.verts, rig.L
    p = {**load_bodies()[name], "kneeHeightRatio": ratio}
    L = landmarks(p)
    assert L["knee"] == pytest.approx(ratio * old["crotch"])
    for key in ("crotch", "ankle", "footLength", "H"):
        assert L[key] == old[key]

    wrap_body(load_template(), p, tmp_path)
    verts = np.load(tmp_path / "wrapped.npy")
    ring = base["rings"]["knee"]
    knee = verts[ring].mean(0)
    requested = L["knee"] - old["knee"]
    assert knee[2] - before[ring, 2].mean() == pytest.approx(requested, rel=0.10)
    ankle = base["rings"]["ankle"]
    assert abs(verts[ankle, 2].mean() - before[ankle, 2].mean()) < 0.2
    raw, regions = np.asarray(base["verts"]), np.asarray(base["regions"])
    crotch = (np.abs(raw[:, 0]) < 0.5) & np.isin(regions, ("trunk", "leg"))
    foot = (regions == "foot") & (raw[:, 0] > 0)
    assert abs(verts[crotch, 2].min() - before[crotch, 2].min()) < 0.5
    assert np.ptp(verts[foot, 1]) == pytest.approx(np.ptp(before[foot, 1]), abs=0.5)

    bones = skeleton(base, verts, L)
    shin = next(i for i, bone in enumerate(bones) if bone["name"] == "leftLowerLeg")
    assert bones[shin]["head"] == pytest.approx(knee)
    weights = skin_weights(base, bones, verts, L)
    radius = np.linalg.norm(verts[ring] - knee, axis=1).mean()
    for pose in ("crouch", "stride"):
        rotations = POSES[pose]
        posed = pose_verts(bones, weights, verts, rotations)
        hinge = (pose_skins(bones, rotations)[shin] @ np.r_[knee, 1])[:3]
        assert np.linalg.norm(posed[ring].mean(0) - hinge) < 0.15 * radius
        assert SkinParts(base).legs_past_midline(posed) == 0


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("pose", ["crouch", "stride"])
def test_bending_the_legs_keeps_each_on_its_own_side(name, pose):
    rig = rigged(name)
    base, verts, bones, weights = rig.base, rig.verts, rig.bones, rig.weights
    posed = pose_verts(bones, weights, verts, POSES[pose])
    assert SkinParts(base).legs_past_midline(posed) == 0


def test_lower_abdomen_clears_the_thighs_during_the_low_sit():
    rig = rigged("female-long")
    clip = motions.load_clip("sitLow")
    poses = motions.motion_poses(clip, rig)
    parts = SkinParts(rig.base)
    # Descent, deepest compression, then settling with the knees spread.
    for seconds in (2.6, 2.9, 3.5):
        pose = poses[round(seconds * clip["fps"])]
        posed = pose_verts(rig.bones, rig.weights, rig.verts, pose)
        pairs = parts.intersections(posed)["intersectionsByPair"]
        assert pairs.get("trunk|upperLeg", 0) == 0


@pytest.mark.parametrize("name", NAMES)
def test_weights_are_normalised_and_within_the_runtime_limit(name):
    weights = rigged(name).weights
    assert np.allclose(weights.sum(1), 1.0)
    assert (weights > 0).sum(1).max() <= 4


@pytest.mark.parametrize("name", NAMES)
def test_rest_pose_moves_nothing(name):
    rig = rigged(name)
    verts, bones, weights = rig.verts, rig.bones, rig.weights
    assert np.allclose(pose_verts(bones, weights, verts, {}), verts, atol=1e-6)


@pytest.mark.parametrize("name", NAMES)
def test_every_helper_bone_carries_weight(name):
    rig = rigged(name)
    bones, weights = rig.bones, rig.weights
    used = weights.sum(0)
    names = [bone["name"] for bone in bones]
    for side in ("left", "right"):
        for limb in HALF_BONES:
            assert used[names.index(f"{side}{limb}Half")] > 0
        for limb, spec in TWISTS.items():
            for i in range(1, len(spec["bones"]) + 1):
                assert used[names.index(f"{side}{limb}Twist{i}")] > 0


def test_a_forward_raise_elevates_the_girdle_and_the_arm_gives_it_back():
    for arm, sign in (("leftUpperArm", 1.0), ("rightUpperArm", -1.0)):
        side = arm.removesuffix("UpperArm")
        rotations = sided({arm: (60.0, 0.0, 0.0)})
        lift = 60.0 * FORWARD_LIFT * sign
        assert rotations[f"{side}Shoulder"][2] == pytest.approx(lift)
        assert rotations[arm][2] == pytest.approx(-lift)
        # Only a raise: the arm hanging back doesn't lower the girdle.
        assert sided({arm: (-40.0, 0.0, 0.0)})[f"{side}Shoulder"][2] == 0.0
    # A side raise is as it was.
    rotations = sided({"leftUpperArm": (0.0, 0.0, 72.0)})
    assert rotations["leftShoulder"][2] == pytest.approx(72.0 * SHOULDER_RHYTHM)


def test_the_forward_lift_keeps_the_arm_on_its_direction():
    rig = rigged(NAMES[0])
    bones = rig.bones
    names = [bone["name"] for bone in bones]
    arm = names.index("leftUpperArm")
    axis = (bones[arm]["tail"] - bones[arm]["head"])[:3]
    pose = {"upperArm": (95.0, 0.0, 0.0)}
    lifted = pose_skins(bones, pose)[arm][:3, :3] @ axis
    original = FORWARD_LIFT
    try:
        skinning.FORWARD_LIFT = 0.0
        plain = pose_skins(bones, pose)[arm][:3, :3] @ axis
    finally:
        skinning.FORWARD_LIFT = original
    angle = np.degrees(np.arccos(lifted @ plain / (lifted @ lifted)))
    assert 0.0 < angle < 5.0


@pytest.mark.parametrize("name", NAMES)
def test_raised_arms_collapse_no_face(name):
    rig = rigged(name)
    base, verts, bones, weights = rig.base, rig.verts, rig.bones, rig.weights
    rest = np.maximum(face_areas(verts, base["faces"]), 1e-6)
    posed = pose_verts(bones, weights, verts, {"upperArm": (0, 0, 72)})
    assert ((face_areas(posed, base["faces"]) / rest) < 0.15).sum() == 0


@pytest.mark.parametrize("name", NAMES)
def test_a_rolled_wrist_keeps_the_forearm_round(name):
    rig = rigged(name)
    base, verts, bones, weights = rig.base, rig.verts, rig.bones, rig.weights
    segments = np.asarray(base["segments"])
    template = np.asarray(base["verts"])
    ring = lambda p, key: p[base["rings"][key]].mean(0)
    e, w = ring(verts, "elbow"), ring(verts, "wrist")
    t = (verts - e) @ (w - e) / np.sum((w - e) ** 2)
    members = (segments == "lowerArm") & (template[:, 0] > 0) & (t > 0.4)

    def girth(points):
        start, end = ring(points, "elbow"), ring(points, "wrist")
        axis = (end - start) / np.linalg.norm(end - start)
        offset = points[members] - start
        return np.linalg.norm(offset - np.outer(offset @ axis, axis), axis=1).mean()

    posed = pose_verts(bones, weights, verts, {"hand": (0, 90, 0)})
    assert girth(posed) >= 0.95 * girth(verts)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("pose", ["headTurn", "headNod"])
def test_turning_the_head_collapses_no_face(name, pose):
    # tests/test_head.py holds this for the default style; it belongs to the head's domain.
    rig = rigged(name)
    base, verts, bones, weights = rig.base, rig.verts, rig.bones, rig.weights
    rest = np.maximum(face_areas(verts, base["faces"]), 1e-6)
    posed = pose_verts(bones, weights, verts, POSES[pose])
    assert ((face_areas(posed, base["faces"]) / rest) < 0.15).sum() == 0


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("angle", [25, 30])
@pytest.mark.parametrize("axis", [0, 1], ids=["nod", "turn"])
def test_the_head_goes_this_far_on_each_bone_with_no_face_collapsed(name, axis, angle):
    # How far the head moves is not monotonic in what it crushes: a face between two rows too close
    # together collapsed at 25 degrees a bone and opened again by 30.
    rig = rigged(name)
    base, verts, bones, weights = rig.base, rig.verts, rig.bones, rig.weights
    rest = np.maximum(face_areas(verts, base["faces"]), 1e-6)
    rotation = tuple(angle if i == axis else 0 for i in range(3))
    posed = pose_verts(bones, weights, verts, {"neck": rotation, "head": rotation})
    assert ((face_areas(posed, base["faces"]) / rest) < 0.15).sum() == 0


@pytest.mark.parametrize("name", NAMES)
def test_the_jaw_rides_below_the_mouth_and_stays_off_the_neck_and_the_upper_face(name):
    rig = rigged(name)
    names = [bone["name"] for bone in rig.bones]
    segments = np.asarray(rig.base["segments"])
    jaw, head = (rig.weights[:, names.index(bone)] for bone in ("jaw", "head"))
    t = (rig.L["H"] - rig.verts[:, 2]) / rig.L["head"]
    assert not jaw[segments != "head"].any()
    ring = rig.base["openings"]["neck"]
    assert not jaw[ring].any()
    assert not jaw[t < 0.55].any()
    # The chin is all jaw; the rest of the face moves whole with the head, jaw and head together.
    ahead = (segments == "head") & (rig.verts[:, 1] < rig.verts[ring, 1].min() - 0.08 * rig.L["head"])
    chin = np.flatnonzero(ahead)[np.argmin(rig.verts[ahead, 2])]
    assert jaw[chin] > 0.8
    face = (segments == "head") & (t < 0.85)
    assert (jaw + head)[face] == pytest.approx(1.0)


@pytest.mark.parametrize("name", NAMES)
def test_the_head_bones_come_last_and_the_eyes_mirror(name):
    rig = rigged(name)
    assert [bone["name"] for bone in rig.bones[-3:]] == ["jaw", "leftEye", "rightEye"]
    jaw, left, right = rig.bones[-3:]
    assert jaw["head"][0] == 0 and jaw["tail"][0] == 0
    assert jaw["tail"][2] < jaw["head"][2] and jaw["tail"][1] < jaw["head"][1]
    assert right["head"] == pytest.approx(left["head"] * (-1, 1, 1))
    assert left["head"][0] > 0
    # The socket's middle is inside the head and a turn of the eye moves its front in the head's frame.
    h = rig.L["head"]
    assert np.linalg.norm(left["tail"] - left["head"]) == pytest.approx(0.14 * h)
    assert rig.L["H"] - 0.7 * h < left["head"][2] < rig.L["H"] - 0.5 * h


@pytest.mark.parametrize("name", NAMES)
def test_opening_the_jaw_takes_the_chin_down_more_than_back(name):
    rig = rigged(name)
    jaw = next(bone for bone in rig.bones if bone["name"] == "jaw")
    moved = pose_skins(rig.bones, {"jaw": (15, 0, 0)})[[bone["name"] for bone in rig.bones].index("jaw")]
    tail = moved[:3, :3] @ jaw["tail"] + moved[:3, 3] - jaw["tail"]
    assert -tail[2] > 1.5 * abs(tail[1])
    assert tail[2] < -0.05 * rig.L["head"]


@pytest.mark.parametrize("name", NAMES)
def test_opening_the_jaw_collapses_no_face(name):
    rig = rigged(name)
    base, verts, bones, weights = rig.base, rig.verts, rig.bones, rig.weights
    rest = np.maximum(face_areas(verts, base["faces"]), 1e-6)
    for pose in ({"jaw": (20, 0, 0)}, {"neck": (10, 0, 0), "head": (20, 0, 0), "jaw": (20, 0, 0)}):
        posed = pose_verts(bones, weights, verts, pose)
        assert ((face_areas(posed, base["faces"]) / rest) < 0.15).sum() == 0, pose


def test_euler_angles_give_the_rotation_back():
    for angles in ((10, 20, 30), (-70, 45, 120), (0, 0, 0), (160, -80, -20)):
        rotation = euler(angles)[:3, :3]
        assert euler(motions.euler_angles(rotation))[:3, :3] == pytest.approx(rotation, abs=1e-3)


@pytest.mark.parametrize("clip", motions.clip_names())
def test_a_clip_moves_the_rig_through_names_it_has(clip):
    rig = rigged(NAMES[0])
    verts, bones, weights = rig.verts, rig.bones, rig.weights
    data = motions.load_clip(clip)
    pose = motions.retargeted(data, bones, motions.frame_count(data) // 2)
    assert not np.allclose(pose_verts(bones, weights, verts, pose), verts)


@pytest.mark.parametrize("clip", motions.clip_names())
@pytest.mark.parametrize("bone", ["leftUpperLeg", "rightLowerLeg", "leftFoot"])
def test_a_clip_turns_the_leg_bones_as_it_asks(clip, bone):
    # Arms give part of their swing to the shoulder, so only the legs land exactly where asked.
    rig = rigged(NAMES[0])
    bones = rig.bones
    data = motions.load_clip(clip)
    frame = motions.frame_count(data) // 2
    skins = pose_skins(bones, motions.retargeted(data, bones, frame))
    row = [b["name"] for b in bones].index(bone)
    asked = motions.asked_of(data, bones, frame)[bone]
    assert skins[row][:3, :3] == pytest.approx(asked, abs=1e-2)


@pytest.mark.parametrize("pose", motions.HAND_POSES)
def test_hand_poses_name_bones_and_stay_within_a_few_collapsed_faces(pose):
    rig = rigged(NAMES[0])
    base, verts, bones, weights = rig.base, rig.verts, rig.bones, rig.weights
    rest = np.maximum(face_areas(verts, base["faces"]), 1e-6)
    posed = pose_verts(bones, weights, verts, motions.HAND_POSES[pose])
    assert ((face_areas(posed, base["faces"]) / rest) < 0.15).sum() <= 6


def reference_rig(name):
    """Weight fitting uses the reference skin, before the accepted static pelvic sculpture."""
    rig = rigged(name)
    wrapped = np.load(BUILD / "bodies" / name / "wrapped.npy")
    return rig._replace(verts=skin_reference(bodies()[name], wrapped))


def unfitted(name):
    rig = reference_rig(name)
    base, verts, L, bones = rig.base, rig.verts, rig.L, rig.bones
    return skin_weights(base, bones, verts, L, fit_lower=False)


@pytest.mark.slow
@pytest.mark.parametrize("name", NAMES)
def test_the_rig_as_built_has_its_waist_down_fit(name):
    # The build and direct weight fitting must use exactly the same reference skin.
    rig = reference_rig(name)
    base, verts, L, bones, weights = rig.base, rig.verts, rig.L, rig.bones, rig.weights
    assert np.array_equal(weights, skin_weights(base, bones, verts, L))
    assert not np.array_equal(weights, unfitted(name))


@pytest.mark.slow
@pytest.mark.parametrize("name", NAMES)
def test_the_fit_leaves_everything_outside_its_zone_exactly_alone(name):
    rig = reference_rig(name)
    base, verts, L, weights = rig.base, rig.verts, rig.L, rig.weights
    outside = lower_zone(base, verts, L) <= 0
    assert np.array_equal(weights[outside], unfitted(name)[outside])
    assert verts[outside, 2].max() >= L["waist"]
    assert not np.isin(np.asarray(base["segments"])[~outside], ("head", "neck", "upperArm")).any()


@pytest.mark.slow
@pytest.mark.parametrize("name", NAMES)
def test_the_fit_leaves_no_seam_where_it_ends(name):
    rig = reference_rig(name)
    base, verts, L, weights = rig.base, rig.verts, rig.L, rig.weights
    zone = lower_zone(base, verts, L)
    a, b = edges_of(base["faces"]).T
    plain = unfitted(name)
    at_edge = (zone[a] > 0) != (zone[b] > 0)
    assert at_edge.any()
    jump = np.abs(weights[a] - weights[b]).sum(1)
    assert jump[at_edge].max() <= np.abs(plain[a] - plain[b]).sum(1)[at_edge].max() + 0.03


def test_the_fit_refuses_a_change_outside_its_zone():
    rig = reference_rig(NAMES[0])
    base, verts, L, weights = rig.base, rig.verts, rig.L, rig.weights
    zone = lower_zone(base, verts, L)
    broken = weights.copy()
    row = np.flatnonzero(zone <= 0)[0]
    broken[row, np.flatnonzero(broken[row])[0]] -= 0.001
    with pytest.raises(ValueError, match="outside its zone"):
        check_lower_body(broken, weights, zone)


@pytest.mark.parametrize("name", NAMES)
def test_the_weights_mirror_across_the_midline(name):
    # The lower-body fit is not symmetric by itself; its result is mirrored (sotaigen/rig/skeleton.py).
    from sotaigen.mesh import opposite_vertices

    rig = rigged(name)
    names = [bone["name"] for bone in rig.bones]
    swap = {"left": "right", "right": "left"}
    columns = [
        names.index(next((swap[s] + n[len(s):] for s in swap if n.startswith(s)), n)) for n in names
    ]
    opposite = opposite_vertices(rig.verts)
    assert opposite is not None
    assert np.abs(rig.weights[opposite][:, columns] - rig.weights).max() < 1e-9


def test_the_hand_half_bone_takes_half_the_swing_and_most_of_the_roll():
    """Half the flexion, as any half bone; its roll lies between the last twist bone's and the hand's."""
    flexed = sided({"hand": (60, 0, 0)})
    assert np.allclose(flexed["leftHandHalf"], euler((30, 0, 0)))
    rolled = sided({"hand": (0, 80, 0)})
    assert np.allclose(rolled["leftHandHalf"], euler((0, 80 * HALF_ROLLS["Hand"], 0)))
    assert np.allclose(rolled["rightHandHalf"], euler((0, -80 * HALF_ROLLS["Hand"], 0)))
    last_twist = TWISTS["LowerArm"]["bones"][-1][1]
    assert last_twist < HALF_ROLLS["Hand"] < 1
