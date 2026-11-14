"""Dynamic anatomy cards, and the limb geometry that lets them talk about skin.

A card in docs/motion says what should happen to a body part as it moves. This module reads and
checks cards and turns their zones into vertices. It decides nothing about how anything looks: its
numbers describe skin behaviour for a reader who has the card and the images, and have no pass mark.

It lives in sotaigen_dev/ so that neither the cards nor this code feed the generation stages' fingerprints.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sotaigen.mesh import triangulate, unit_normals
from sotaigen.rig.skinning import pose_skins, rest_matrix

CARDS = Path(__file__).resolve().parents[2] / "docs" / "motion"

KINDS = {
    "bone-motion",
    "bone-reveal",
    "muscle-bulge",
    "stretch-flatten",
    "crease",
    "slide",
    "contact",
}
CHOICES = {"undecided", "keep", "exaggerate", "simplify", "drop"}
# How far a claim can be leaned on: measured in people, in a reference text, in a tertiary page, or
# only as far as a search snippet. An observation is what someone saw and nobody has checked.
STRENGTHS = {"measured", "textbook", "tertiary", "snippet", "observation"}
STATUSES = {"draft", "confirmed"}
VIEWS = {"side", "front", "back", "threequarter", "backquarter"}
# Where a canonical elbow sits, so renders stay well away from the origin.
ANCHOR = np.array((0.0, 0.0, 1000.0))


class CardError(ValueError):
    """A card that says something the tools could not act on or a reader could not trust."""


def check_card(card, poses_known=None):
    """Raise CardError for the first thing wrong with a parsed card."""
    name = card.get("region", "<no region>")

    def fail(message):
        raise CardError(f"card {name}: {message}")

    for field in ("region", "title", "status", "scope", "chain", "joint", "sweeps", "phenomena"):
        if field not in card:
            fail(f"missing {field}")
    if card["status"] not in STATUSES:
        fail(f"status {card['status']!r} is not one of {sorted(STATUSES)}")
    sweeps = {sweep["name"]: sweep for sweep in card["sweeps"]}
    zones = card.get("zones", {})
    for key, zone in zones.items():
        low, high = zone["axial"]
        if low >= high or not 0 < zone["spread"] <= 180:
            fail(f"zone {key} has an empty range")
    seen = set()
    for row in card["phenomena"]:
        label = row.get("id", "<no id>")
        if label in seen:
            fail(f"phenomenon {label} appears twice")
        seen.add(label)
        for field in ("name", "structure", "kind", "sweep", "from", "to", "zones", "views"):
            if field not in row:
                fail(f"phenomenon {label} has no {field}")
        for field in ("reading", "choice", "evidence"):
            if not row.get(field):
                fail(f"phenomenon {label} has no {field}")
        if row["kind"] not in KINDS:
            fail(f"phenomenon {label} has unknown kind {row['kind']!r}")
        if row["choice"] not in CHOICES:
            fail(f"phenomenon {label} has unknown choice {row['choice']!r}")
        if row["sweep"] not in sweeps:
            fail(f"phenomenon {label} names unknown sweep {row['sweep']!r}")
        for zone in row["zones"]:
            if zone not in zones:
                fail(f"phenomenon {label} names unknown zone {zone!r}")
        for view in row["views"]:
            if view not in VIEWS:
                fail(f"phenomenon {label} names unknown view {view!r}")
        for item in row["evidence"]:
            if item.get("strength") not in STRENGTHS or not item.get("claim"):
                fail(f"phenomenon {label} has evidence without a strength and a claim")
            if not item.get("source"):
                fail(f"phenomenon {label} has evidence without a source")
        if card["status"] == "confirmed" and not row.get("current"):
            fail(f"confirmed, yet phenomenon {label} does not say how the rig does today")
        if card["status"] == "confirmed" and row["choice"] == "undecided":
            fail(f"confirmed, yet phenomenon {label} is undecided: the user decides every row")
    for guard in card.get("guards", []):
        for zone in guard["expected_zones"]:
            if zone not in zones:
                fail(f"guard on {guard['pose']} names unknown zone {zone!r}")
        if poses_known is not None and guard["pose"] not in poses_known:
            fail(f"guard names unknown pose {guard['pose']!r}")
    return card


def load_card(region, folder=CARDS, poses_known=None):
    path = Path(folder) / f"{region}.toml"
    if not path.exists():
        known = sorted(item.stem for item in Path(folder).glob("*.toml"))
        raise CardError(f"no card {region!r}; known: {', '.join(known) or 'none'}")
    return check_card(tomllib.loads(path.read_text(encoding="utf-8")), poses_known)


def sweep_poses(sweep):
    """A sweep's poses by name, such as flex090, with its pose's rotations scaled by the angle."""
    poses = {}
    for angle in sweep["angles"]:
        pose = {bone: [axis * angle for axis in axes] for bone, axes in sweep["pose"].items()}
        poses[f"{sweep['name']}{angle:03d}"] = pose
    return poses


@dataclass(frozen=True)
class Hinge:
    """A joint in its own terms: all directions unit vectors, in the rest pose of the left side."""

    centre: np.ndarray
    axis: np.ndarray  # The bend axis, against the lower bone's local X: outwards on the limbs.
    upper: np.ndarray  # From the joint towards the far end of the upper limb.
    lower: np.ndarray  # From the joint towards the far end of the lower limb.
    length: float  # Of the lower limb: the unit of a zone's axial range.
    upper_column: int
    lower_column: int

    @property
    def rotation(self):
        """Rows are the canonical axes: X the bend axis outwards, Z along the upper limb."""
        z = self.upper - self.axis * (self.upper @ self.axis)
        z = z / np.linalg.norm(z)
        return np.array((self.axis, np.cross(z, self.axis), z))

    def canonical(self, points):
        """The joint upright with its bend axis along X, so the review's views mean the limb's."""
        return (np.asarray(points) - self.centre) @ self.rotation.T + ANCHOR


def hinge_of(card, bones, sample_pose):
    """The card's joint on these bones. `sample_pose` bends the joint a little, to find its front."""
    joint = card["joint"]
    names = [bone["name"] for bone in bones]
    upper, lower = (bones[names.index(joint[key])] for key in ("upper_bone", "lower_bone"))
    centre = np.asarray(lower["head"], float)
    # Against local X, so that a positive turn carries the far end towards canonical -Y. On an arm or
    # a leg this points away from the body, and on a hand, whose axis runs front to back, to the thumb.
    axis = -rest_matrix(lower)[:3, 0]
    towards_upper = np.asarray(upper["head"], float) - centre
    fore = np.asarray(lower["tail"], float) - centre
    length = float(np.linalg.norm(fore))
    hinge = Hinge(
        centre,
        axis,
        towards_upper / np.linalg.norm(towards_upper),
        fore / length,
        length,
        names.index(joint["upper_bone"]),
        names.index(joint["lower_bone"]),
    )
    tip = np.append(np.asarray(lower["tail"], float), 1.0)
    moved = (pose_skins(bones, sample_pose)[hinge.lower_column] @ tip)[:3]
    forward = hinge.rotation @ (moved - tip[:3])
    if forward[1] >= 0:
        raise CardError(
            f"card {card['region']}: flexing {joint['lower_bone']} moves its tip backwards "
            "in the canonical frame; the card's sweep bends the joint the wrong way"
        )
    return hinge


def limb_coordinates(hinge, card, base, rest):
    """Each vertex of the limb as (axial, azimuth); NaN elsewhere. The right side mirrors the left.

    Axial runs from the joint, positive down the lower limb, in lower limb lengths. Azimuth is in
    degrees around the limb: 0 faces the way the joint flexes, 180 away, and +90 the bend axis, which
    is outwards on a limb and the thumb's side on a hand.
    """
    joint = card["joint"]
    segments = np.asarray(base["segments"])
    mirrored = np.array(rest, float)
    mirrored[:, 0] = np.abs(mirrored[:, 0])
    axial = np.full(len(rest), np.nan)
    azimuth = np.full(len(rest), np.nan)
    frontal = -hinge.rotation[1]
    for segment, along, sign in (
        (joint["upper_segment"], hinge.upper, -1.0),
        (joint["lower_segment"], hinge.lower, 1.0),
    ):
        members = segments == segment
        offset = mirrored[members] - hinge.centre
        front = frontal - along * (frontal @ along)
        front = front / np.linalg.norm(front)
        side = np.cross(along, front)
        side = side if side @ hinge.axis > 0 else -side
        axial[members] = sign * (offset @ along) / hinge.length
        azimuth[members] = np.degrees(np.arctan2(offset @ side, offset @ front))
    return axial, azimuth


def zone_mask(zone, axial, azimuth):
    """Which vertices a card's zone covers. NaN coordinates are never inside."""
    with np.errstate(invalid="ignore"):
        apart = np.abs((azimuth - zone["azimuth"] + 180.0) % 360.0 - 180.0)
        low, high = zone["axial"]
        return (axial >= low) & (axial <= high) & (apart <= zone["spread"])


@dataclass(frozen=True)
class Skin:
    """What a pose did to each vertex, seen from the limb the vertex belongs to.

    Moving with a limb counts as nothing. `lift` is along the rest normal, outwards positive, and
    `slide` across the surface, both in mm. `radius` is the distance from the limb's own axis as a
    share of the same vertex's at rest. All are NaN off the limb.
    """

    lift: np.ndarray
    slide: np.ndarray
    radius: np.ndarray


def skin_behaviour(hinge, card, base, rest, posed, skins, axial):
    """Skin relative to its own limb under a pose. `skins` is pose_skins' bones x 4 x 4 array."""
    rest = np.asarray(rest, float)
    normals = unit_normals(rest, triangulate(base["faces"], rest))
    lift, slide, radius = (np.full(len(rest), np.nan) for _ in range(3))
    # Only the left limb: the right mirrors it, and its bones are other columns.
    left = np.asarray(base["verts"])[:, 0] >= 0
    for column, along, members in (
        (hinge.upper_column, hinge.upper, axial < 0),
        (hinge.lower_column, hinge.lower, axial >= 0),
    ):
        members = members & left
        inverse = np.linalg.inv(skins[column])
        back = np.column_stack((posed[members], np.ones(members.sum()))) @ inverse.T
        moved = back[:, :3] - rest[members]
        lifted = np.sum(moved * normals[members], axis=1)
        lift[members] = lifted
        slide[members] = np.linalg.norm(moved - lifted[:, None] * normals[members], axis=1)
        before = rest[members] - hinge.centre
        after = back[:, :3] - hinge.centre
        before_radial = np.linalg.norm(before - np.outer(before @ along, along), axis=1)
        after_radial = np.linalg.norm(after - np.outer(after @ along, along), axis=1)
        radius[members] = after_radial / np.maximum(before_radial, 1e-6)
    return Skin(lift, slide, radius)


def zone_summary(skin, mask):
    """A zone's skin behaviour as plain numbers, for a reader. Left side only, like Skin."""
    selected = mask & np.isfinite(skin.lift)
    if not selected.any():
        raise CardError("a zone holds no vertices on this mesh")
    lift, slide, radius = skin.lift[selected], skin.slide[selected], skin.radius[selected]
    return {
        "vertices": int(selected.sum()),
        "liftMeanMm": round(float(lift.mean()), 1),
        "liftMinMm": round(float(lift.min()), 1),
        "liftMaxMm": round(float(lift.max()), 1),
        "slideMeanMm": round(float(slide.mean()), 1),
        "slideMaxMm": round(float(slide.max()), 1),
        "radiusShare": round(float(radius.mean()), 3),
    }


def plane_section(points, triangles, x=0.0):
    """Where triangles cross the plane X = x, as N x 2 x 2 segments of (Y, Z)."""
    corners = points[triangles]
    above = corners[:, :, 0] > x
    crossing = above.any(axis=1) & ~above.all(axis=1)
    segments = []
    for tri in corners[crossing]:
        ends = []
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            if (a[0] > x) != (b[0] > x):
                ends.append(a + (b - a) * (x - a[0]) / (b[0] - a[0]))
        if len(ends) == 2:
            segments.append([ends[0][1:], ends[1][1:]])
    return np.array(segments).reshape(-1, 2, 2)


def faces_in_zones(faces, masks):
    """For each face, whether any of its vertices is in any of the masks."""
    inside = np.zeros(len(masks[0]), bool) if masks else None
    for mask in masks:
        inside |= mask
    if inside is None:
        return np.zeros(len(faces), bool)
    return np.array([bool(inside[list(face)].any()) for face in faces])
