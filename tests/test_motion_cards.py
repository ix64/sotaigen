"""Dynamic anatomy cards: their structure, and the limb geometry the sweep tool reports with.

Nothing here judges a look. The numbers these functions make have no pass mark by design.
"""

import copy

import numpy as np
import pytest

from sotaigen.form.anatomy import skin_reference
from sotaigen.form.bodies import load_bodies
from sotaigen.mesh import triangulate, unit_normals
from sotaigen.metrics import COLLAPSED, RestMesh
from sotaigen.pipeline import BUILD, bodies, rigged
from sotaigen.rig.poses import POSES
from sotaigen.rig.skeleton import skin_weights
from sotaigen.rig.skinning import pose_skins, pose_verts
from sotaigen.template import coarse
from sotaigen_dev.motion import cards


def all_cards():
    return sorted(path.stem for path in cards.CARDS.glob("*.toml"))


def test_there_is_a_card_to_check():
    assert "elbow" in all_cards()


@pytest.mark.parametrize("name", all_cards())
def test_every_card_is_well_formed(name):
    cards.load_card(name, poses_known=POSES)


@pytest.fixture
def elbow():
    return copy.deepcopy(cards.load_card("elbow"))


@pytest.mark.parametrize(
    "spoil",
    [
        lambda card: card["phenomena"][0].update(kind="looks-nice"),
        lambda card: card["phenomena"][0].update(choice="maybe"),
        lambda card: card["phenomena"][0].update(sweep="nowhere"),
        lambda card: card["phenomena"][0].update(zones=["nowhere"]),
        lambda card: card["phenomena"][0].update(views=["above"]),
        lambda card: card["phenomena"][0].update(evidence=[]),
        lambda card: card["phenomena"][0].update(reading=""),
        lambda card: card["phenomena"][0]["evidence"][0].update(strength="vibes"),
        lambda card: card["phenomena"][0]["evidence"][0].pop("source"),
        lambda card: card["phenomena"].append(card["phenomena"][0]),
        lambda card: card["zones"]["elbow-tip"].update(axial=[0.2, 0.1]),
        lambda card: card["guards"][0].update(expected_zones=["nowhere"]),
    ],
)
def test_a_card_that_cannot_be_acted_on_is_refused(elbow, spoil):
    spoil(elbow)
    with pytest.raises(cards.CardError):
        cards.check_card(elbow)


def test_only_a_fully_decided_card_counts_as_confirmed(elbow):
    elbow["status"] = "confirmed"
    for row in elbow["phenomena"]:
        row["choice"] = "keep"
    cards.check_card(elbow)
    elbow["phenomena"][0]["choice"] = "undecided"
    with pytest.raises(cards.CardError, match="undecided"):
        cards.check_card(elbow)
    elbow["phenomena"][0]["choice"] = "keep"
    elbow["phenomena"][0].pop("current")
    with pytest.raises(cards.CardError, match="how the rig does"):
        cards.check_card(elbow)


def test_a_missing_card_names_those_that_exist():
    with pytest.raises(cards.CardError, match="elbow"):
        cards.load_card("no-such-part")


def test_sweep_poses_scale_the_pose_by_each_angle():
    poses = cards.sweep_poses({"name": "flex", "pose": {"lowerArm": [1, 0, 0]}, "angles": [0, 90]})
    assert poses == {"flex000": {"lowerArm": [0, 0, 0]}, "flex090": {"lowerArm": [90, 0, 0]}}


def test_a_zone_wraps_around_the_back_of_the_limb():
    zone = {"axial": [-1, 1], "azimuth": 180, "spread": 20}
    axial = np.array([0.0, 0.0, 0.0, 2.0, np.nan])
    azimuth = np.array([175.0, -175.0, 150.0, 180.0, 180.0])
    assert cards.zone_mask(zone, axial, azimuth).tolist() == [True, True, False, False, False]


def test_a_plane_section_of_a_triangle():
    points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
    segments = cards.plane_section(points, np.array([[0, 1, 2]]))
    assert len(segments) == 1
    assert sorted(map(tuple, segments[0].tolist())) == [(0.0, 0.0), (1.0, 0.0)]


def test_a_face_counts_as_in_a_zone_when_any_corner_is():
    masks = [np.array([True, False, False, False])]
    assert cards.faces_in_zones([[0, 1, 2], [1, 2, 3]], masks).tolist() == [True, False]


@pytest.fixture(scope="module")
def rig():
    return rigged("female-long")


@pytest.fixture(scope="module")
def setup(rig):
    card = cards.load_card("elbow")
    hinge = cards.hinge_of(card, rig.bones, {"lowerArm": (5, 0, 0)})
    axial, azimuth = cards.limb_coordinates(hinge, card, rig.base, rig.verts)
    return card, hinge, axial, azimuth


def test_the_canonical_frame_is_a_rotation_with_the_upper_limb_up(setup):
    _, hinge, _, _ = setup
    rotation = hinge.rotation
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert (rotation @ hinge.upper)[2] > 0.99
    assert np.allclose(hinge.canonical(hinge.centre), cards.ANCHOR)


def test_every_zone_finds_skin_on_both_arms(rig, setup):
    card, _, axial, azimuth = setup
    left = np.asarray(rig.base["verts"])[:, 0] >= 0
    for zone in card["zones"].values():
        mask = cards.zone_mask(zone, axial, azimuth)
        assert mask[left].sum() == mask[~left].sum() > 0


def test_the_back_of_the_elbow_is_where_azimuth_says(rig, setup):
    """Azimuth 180 faces away from the flexion, so a bend must carry it away from the forearm."""
    card, hinge, axial, azimuth = setup
    left = np.asarray(rig.base["verts"])[:, 0] >= 0
    back = cards.zone_mask(card["zones"]["triceps-distal"], axial, azimuth) & left
    front = cards.zone_mask(card["zones"]["biceps-belly"], axial, azimuth) & left
    canonical = hinge.canonical(rig.verts)
    assert canonical[back, 1].mean() > canonical[front, 1].mean()


def test_skin_at_rest_does_nothing(rig, setup):
    card, hinge, axial, _ = setup
    skins = pose_skins(rig.bones, {})
    skin = cards.skin_behaviour(hinge, card, rig.base, rig.verts, rig.verts, skins, axial)
    assert np.nanmax(np.abs(skin.lift)) < 1e-6
    assert np.nanmax(skin.slide) < 1e-6
    assert np.nanmax(np.abs(skin.radius - 1)) < 1e-6


def test_skin_that_moves_only_with_its_limb_does_nothing(rig, setup):
    """Far from the joint a limb moves whole, and from that limb nothing has happened."""
    card, hinge, axial, azimuth = setup
    pose = {"lowerArm": (90, 0, 0)}
    posed = pose_verts(rig.bones, rig.weights, rig.verts, pose)
    skins = pose_skins(rig.bones, pose)
    skin = cards.skin_behaviour(hinge, card, rig.base, rig.verts, posed, skins, axial)
    far = cards.zone_mask(card["zones"]["biceps-belly"], axial, azimuth)
    assert cards.zone_summary(skin, far)["slideMaxMm"] == pytest.approx(0.0, abs=0.05)


def test_a_push_along_the_normal_reads_as_lift_not_slide(rig, setup):
    card, hinge, axial, azimuth = setup
    rest = np.asarray(rig.verts, float)
    normals = unit_normals(rest, triangulate(rig.base["faces"], rest))
    pushed = rig.verts + 3.0 * normals
    skins = pose_skins(rig.bones, {})
    skin = cards.skin_behaviour(hinge, card, rig.base, rig.verts, pushed, skins, axial)
    summary = cards.zone_summary(skin, cards.zone_mask(card["zones"]["elbow-tip"], axial, azimuth))
    assert summary["liftMeanMm"] == pytest.approx(3.0, abs=0.05)
    assert summary["slideMaxMm"] == pytest.approx(0.0, abs=0.05)


def test_a_card_whose_sweep_bends_the_wrong_way_is_refused(rig):
    card = cards.load_card("elbow")
    with pytest.raises(cards.CardError, match="wrong way"):
        cards.hinge_of(card, rig.bones, {"lowerArm": (-5, 0, 0)})


def test_a_hand_bends_forward_in_its_own_frame(rig):
    """A hand's bend axis runs front to back, so "outwards" cannot pick its sign; the bone's own X does."""
    card = cards.load_card("wrist")
    hinge = cards.hinge_of(card, rig.bones, {"hand": (5, 0, 0)})
    names = [bone["name"] for bone in rig.bones]
    thumb = np.asarray(rig.bones[names.index("leftThumbProximal")]["head"], float)
    assert (thumb - hinge.centre) @ hinge.axis > 0
    with pytest.raises(cards.CardError, match="wrong way"):
        cards.hinge_of(card, rig.bones, {"hand": (-5, 0, 0)})


@pytest.mark.parametrize("preset", sorted(load_bodies()))
def test_a_wrist_flexed_to_its_target_collapses_only_its_crease(preset):
    """The user's choice on the wrist card: an arc on the back and one fold in front, to 60 degrees.

    The fold's flattened faces are the crease the card asks for (female-shortbust has 2), so only
    those outside it count.
    """
    rig = rigged(preset)
    card = cards.load_card("wrist")
    hinge = cards.hinge_of(card, rig.bones, {"hand": (5, 0, 0)})
    axial, azimuth = cards.limb_coordinates(hinge, card, rig.base, rig.verts)
    crease = cards.zone_mask(card["zones"]["palmar-crease"], axial, azimuth)
    posed = pose_verts(rig.bones, rig.weights, rig.verts, {"hand": (60, 0, 0)})
    collapsed = RestMesh(rig.verts, rig.base["faces"]).area_ratio(posed) < COLLAPSED
    assert (collapsed & ~cards.faces_in_zones(rig.base["faces"], [crease])).sum() == 0


@pytest.mark.parametrize("preset", sorted(load_bodies()))
def test_a_rolled_wrist_grows_no_ring(preset):
    """A half bone with none of the hand's roll cut a ring at the wrist: its faces shrank to 0.2."""
    rig = rigged(preset)
    posed = pose_verts(rig.bones, rig.weights, rig.verts, {"hand": (0, 120, 0)})
    assert RestMesh(rig.verts, rig.base["faces"]).area_ratio(posed).min() > 0.5


@pytest.mark.slow
@pytest.mark.parametrize("preset", sorted(load_bodies()))
def test_the_palm_keeps_the_crease_the_back_of_the_wrist_gives_up(preset):
    """Only the back's weights are smoothed further: on the palm the edge-counted blend stays."""
    rig = rigged(preset)
    reference = skin_reference(bodies()[preset], np.load(BUILD / "bodies" / preset / "wrapped.npy"))
    card = cards.load_card("wrist")
    hinge = cards.hinge_of(card, rig.bones, {"hand": (5, 0, 0)})
    axial, azimuth = cards.limb_coordinates(hinge, card, rig.base, rig.verts)
    left = np.asarray(rig.base["verts"])[:, 0] >= 0
    palm = cards.zone_mask(card["zones"]["palmar-crease"], axial, azimuth) & left
    unsmoothed = pytest.MonkeyPatch()
    unsmoothed.setattr("sotaigen.rig.skeleton.round_wrist_back", lambda base, bones, verts, weights, edges, band: weights)
    try:
        without = skin_weights(rig.base, rig.bones, reference, rig.L)
    finally:
        unsmoothed.undo()
    # The palm's middle, well inside the azimuth where the smoothing has no effect.
    middle = palm & (np.abs(azimuth) < 60)
    assert np.abs(rig.weights - without)[middle].max() < 1e-9
    back = cards.zone_mask(card["zones"]["dorsal-arc"], axial, azimuth) & left
    assert np.abs(rig.weights - without)[back].max() > 0.01


@pytest.mark.parametrize("preset", sorted(load_bodies()))
def test_a_bent_elbow_keeps_its_tip_out(preset):
    """The user's choice on the elbow card: the tip stays a corner, so bending must not shrink it.

    A half bone's arc pulled the back of the elbow towards the bend axis, by about 15% at 145
    degrees. Skin riding a limb keeps its distance from the axis. Bends well past the common range
    are included, since this is where the old arc lost the most.

    Only the source mesh's vertices are measured. The loops added beside the elbow ring put
    vertices into the zone halfway across the band where the back passes from the upper arm to the
    forearm; they draw in as that band is meant to, as the source's vertex beyond them always has.
    """
    rig = rigged(preset)
    card = cards.load_card("elbow")
    hinge = cards.hinge_of(card, rig.bones, {"lowerArm": (5, 0, 0)})
    axial, azimuth = cards.limb_coordinates(hinge, card, rig.base, rig.verts)
    left = np.asarray(rig.base["verts"])[:, 0] >= 0
    source = np.isin(np.arange(len(rig.verts)), coarse(rig.base)[1])
    tip = cards.zone_mask(card["zones"]["elbow-tip"], axial, azimuth) & left & source

    def reach(points):
        return np.linalg.norm(hinge.canonical(points)[tip][:, 1:] - cards.ANCHOR[1:], axis=1).max()

    rest = reach(rig.verts)
    for angle in (90, 120, 145):
        bent = pose_verts(rig.bones, rig.weights, rig.verts, {"lowerArm": (angle, 0, 0)})
        assert reach(bent) >= 0.98 * rest, f"{angle} degrees"


@pytest.mark.parametrize("preset", sorted(load_bodies()))
def test_the_flexing_side_of_the_elbow_keeps_its_half_bone(preset):
    """Only the back rides the limbs; the fold in front still bends over the half bone's arc."""
    rig = rigged(preset)
    card = cards.load_card("elbow")
    hinge = cards.hinge_of(card, rig.bones, {"lowerArm": (5, 0, 0)})
    axial, azimuth = cards.limb_coordinates(hinge, card, rig.base, rig.verts)
    left = np.asarray(rig.base["verts"])[:, 0] >= 0
    names = [bone["name"] for bone in rig.bones]
    half = rig.weights[:, names.index("leftLowerArmHalf")]
    fold = cards.zone_mask(card["zones"]["inner-fold"], axial, azimuth) & left
    assert half[fold].max() > 0.5
