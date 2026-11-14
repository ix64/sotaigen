# SPDX-License-Identifier: Apache-2.0
"""Diagnose the shoulder, axilla and arm at the upper-body targets; nothing here is production.

uv run scripts/upper_body_study.py poses [body ...] [--variant prod]
uv run scripts/upper_body_study.py sheet <name> --poses a,b
    [--variant v | --variants v,w] [--bodies ...] [--view front] [--look clay,toon]
    [--focus shoulder] [--context 2.6] [--rows body|pose|view]
uv run scripts/upper_body_study.py weights <body> [--pose name]
    [--variant v]
uv run scripts/upper_body_study.py section <body> --poses a,b
    [--variants prod,lbs] [--tag name]
uv run scripts/upper_body_study.py profile <name> --poses a,b
    --variants v,w [--bodies ...]

`poses` writes the target and compound poses of docs/systems/rig.md's target table (both arms for
single joints, the left hand on the head and at the face as sotaigen/rig/touch.py solves it for each body, both
for hands at the chest), skinned
the way the build skins its check poses, under build/reports/upper-body/<variant>/, with each pose's
collapsed faces and new crossings by pair in report.json. Nothing here bakes a contact table.

A variant is tokens joined by "+", each isolating a mechanism without touching sotaigen/:

- `prod` (no token): the production correctives bound into the rest skin, then pose_verts.
- `lbs`: no correctives at all, to see what the skinning alone does.
- `rhythm33`, `rhythm0`: the shoulder girdle's share of an arm's raise at 1/3 and 0 (production
  0.2), correctives rebaked under it.
- `nocap`: the acromion cap's skin left to the girdle bone rather than the upper arm.
- `fwdlift`: the girdle also elevates during a forward raise, the upper arm turned back so that
  its direction is unchanged.
- `scapula`: a crude upward rotation of the shoulder blade's skin about the root of its spine,
  applied after skinning. Its share and region are guesses; it cuts new creases at its edge.

`sheet --variants` copies the variants' poses side by side into build/reports/upper-body/mix/. `weights`
paints, per vertex near the shoulder, the bone with the most weight, the girdle's share and the
upper arm's, over front, back and top views. `section` draws the skin's outline in the arm's plane
of raise through the shoulder joint. `profile` overlays the front outline's top from the neck to the
arm, read off the saved poses, one colour per variant. Pictures go to build/reports/upper-body/sheets/ and
plots/. None of this passes or fails anything.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sotaigen.corrective.crossings import SkinParts
from sotaigen.corrective.recipe import Correctives
from sotaigen.mesh import COLLAPSED, write_obj
from sotaigen.metrics import RestMesh
from sotaigen.pipeline import BUILD, fingerprints, rigged, surface
from sotaigen.rig import skinning
from sotaigen.rig.skinning import pose_skins, pose_verts, rest_matrix
from sotaigen.rig.touch import CONTACTS, solve_all, touch_name
from sotaigen_dev.pictures import font

OUT = BUILD / "reports" / "upper-body"
TYPICAL = [
    f"{sex}-{style}"
    for sex in ("female", "male")
    for style in ("chibi", "compact", "slim", "long", "balanced", "rounded")
]


def both(name, rotation):
    return {name: rotation}


# The targets (docs/systems/rig.md "目标") and the compound poses downstream asks for.
POSES = {
    "rest": {},
    "spread30": both("upperArm", (0, 0, 30)),
    "spread45": both("upperArm", (0, 0, 45)),
    "tpose55": both("upperArm", (0, 0, 55)),
    "spread65": both("upperArm", (0, 0, 65)),
    "spread72": both("upperArm", (0, 0, 72)),
    "spread90": both("upperArm", (0, 0, 90)),
    "forward45": both("upperArm", (45, 0, 0)),
    "forward60": both("upperArm", (60, 0, 0)),
    "forward75": both("upperArm", (75, 0, 0)),
    "forward95": both("upperArm", (95, 0, 0)),
    "forward110": both("upperArm", (110, 0, 0)),
    "forward130": both("upperArm", (130, 0, 0)),
    "elbow120": both("lowerArm", (120, 0, 0)),
    "elbow145": both("lowerArm", (145, 0, 0)),
    "wrist45": both("hand", (45, 0, 0)),
    "wrist60": both("hand", (60, 0, 0)),
    # Both hands in front of the chest, nearly meeting: the arms a little forward and in, rolled
    # in 20 degrees (a compound Euler input), elbows bent.
    "chestHands": {"upperArm": (15, 20, -20), "lowerArm": (110, 0, 0)},
}

# The hand on the head and at the face, left arm only, solved for each body (sotaigen/rig/touch.py).
TOUCH_POSES = [touch_name(contact.name, "left") for contact in CONTACTS]
TOUCH_BOTH = [touch_name(contact.name, "both") for contact in CONTACTS if contact.both]

# Variant tokens, joined by "+": lbs, rhythm33, rhythm0, nocap, scapula. "prod" is none of them.
TOKENS = ("prod", "lbs", "rhythm33", "rhythm0", "nocap", "scapula", "fwdlift")

# fwdlift: the girdle also elevates during a forward raise, by this share of the forward input,
# with the upper arm turned back so that its direction in the world is unchanged.
FORWARD_LIFT = 0.12
_BONES = {}


def lifted_sided(pose, _original=skinning.sided):
    out = _original(pose)
    bones = _BONES.get("bones")
    if bones is None:
        return out
    by_name = {bone["name"]: bone for bone in bones}
    for side, sign in (("left", 1.0), ("right", -1.0)):
        arm = out.get(f"{side}UpperArm")
        if arm is None or isinstance(arm, np.ndarray):
            continue
        lift = FORWARD_LIFT * max(arm[0], 0.0)
        if lift == 0:
            continue
        shoulder = out.get(f"{side}Shoulder", (0.0, 0.0, 0.0))
        raised = (shoulder[0], shoulder[1], shoulder[2] + sign * lift)
        rest_sh = rest_matrix(by_name[f"{side}Shoulder"])[:3, :3]
        rest_arm = rest_matrix(by_name[f"{side}UpperArm"])[:3, :3]
        relative = rest_sh.T @ rest_arm
        undo = np.linalg.inv(skinning.euler(raised)[:3, :3]) @ skinning.euler(shoulder)[:3, :3]
        local = relative.T @ undo @ relative @ skinning.euler(arm)[:3, :3]
        matrix = np.eye(4)
        matrix[:3, :3] = local
        out[f"{side}Shoulder"] = raised
        out[f"{side}UpperArm"] = matrix
        out[f"{side}UpperArmHalf"] = half_swing_matrix(local)
    return out


def half_swing_matrix(local):
    moved = local[:, 1]
    axis = np.cross((0.0, 1.0, 0.0), moved)
    size = np.linalg.norm(axis)
    out = np.eye(4)
    if size < 1e-9:
        return out
    axis = axis / size
    angle = np.arctan2(size, moved[1]) / 2
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    out[:3, :3] = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew
    return out


def tokens(variant):
    out = set(variant.split("+")) - {"prod"}
    unknown = out - set(TOKENS)
    if unknown:
        raise SystemExit(f"unknown variant token: {', '.join(sorted(unknown))}")
    return out


class Rhythm:
    """Set the girdle's share of an arm's raise for a variant, and put it back after."""

    def __init__(self, variant):
        found = tokens(variant)
        self.value = 1 / 3 if "rhythm33" in found else 0.0 if "rhythm0" in found else None
        self.lift = "fwdlift" in found

    def __enter__(self):
        self.saved = skinning.SHOULDER_RHYTHM, skinning.sided
        if self.value is not None:
            skinning.SHOULDER_RHYTHM = self.value
        if self.lift:
            skinning.sided = lifted_sided

    def __exit__(self, *exc):
        skinning.SHOULDER_RHYTHM, skinning.sided = self.saved
        _BONES.clear()


def variant_weights(preset, rig):
    """The production weights with the shoulder cap (acromion skin) left to the girdle bone."""
    from sotaigen.form.anatomy import skin_reference
    from sotaigen.pipeline import bodies
    from sotaigen.rig.skeleton import skin_weights

    p = bodies()[preset]
    reference = skin_reference(p, np.load(BUILD / "bodies" / preset / "wrapped.npy"))
    return skin_weights(rig.base, rig.bones, reference, rig.L, cap_z=(5.0, 6.0))


class Body:
    def __init__(self, preset, variant="prod"):
        self.preset, self.variant = preset, variant
        found = tokens(variant)
        rig = rigged(preset)
        self.base, self.verts, self.bones, self.weights = rig.base, rig.verts, rig.bones, rig.weights
        _BONES["bones"] = self.bones
        if "nocap" in found:
            self.weights = variant_weights(preset, rig)
        self.names = [bone["name"] for bone in self.bones]
        self.faces = self.base["faces"]
        self.rest = RestMesh(self.verts, self.faces)
        self.parts = SkinParts(self.base)
        self.rest_pairs = self.parts.intersections(self.verts)["intersectionsByPair"]
        if "lbs" in found:
            self.correctives = None
        else:
            self.correctives = Correctives.production(
                self.base, self.verts, self.bones, self.weights, field=surface(preset)[0]
            )
        self.scapula = scapula_region(self) if "scapula" in found else None
        self.solved = None

    def named(self, name):
        """A pose by name: from the table, or solved for this body."""
        if name in POSES:
            return POSES[name]
        if self.solved is None:
            self.solved = {
                touch_name(contact): solved
                for contact, solved in solve_all(self.base, self.verts, self.bones, self.weights).items()
            }
        contact, side = name.split(".")
        return self.solved[contact].pose(side)

    def posed(self, pose):
        verts = self.verts if self.correctives is None else self.correctives.bind(self.verts, pose)
        out = pose_verts(self.bones, self.weights, verts, pose)
        if self.scapula is not None:
            out = scapula_rotation(self, pose, out)
        return out

    def bone(self, name):
        return self.bones[self.names.index(name)]


def scapula_region(body):
    """Back-of-shoulder skin that a shoulder blade would carry, as a 0..1 share per vertex.

    A prototype guess: skin behind the frontal plane of the shoulder joint, between the spine and
    the posterior axillary fold, from the top of the shoulder to the inferior angle (about 0.3 T
    below the joint), taking the girdle bone's and the upper chest's weight there.
    """
    from sotaigen.form.field import ramp

    T = skinning.torso_length(body.bones)
    joint = body.bone("leftUpperArm")["head"]
    v = body.verts
    side = np.where(v[:, 0] >= 0, 1.0, -1.0)
    local = np.column_stack((v[:, 0] * side, v[:, 1], v[:, 2]))
    segments = np.asarray(body.base["segments"])
    trunk = np.isin(segments, ("trunk",))
    behind = ramp((local[:, 1] - (joint[1] - 0.02 * T)) / (0.06 * T))
    across = ramp((local[:, 0] - 0.04 * T) / (0.06 * T)) * ramp(
        (joint[0] + 0.02 * T - local[:, 0]) / (0.06 * T)
    )
    height = ramp((local[:, 2] - (joint[2] - 0.34 * T)) / (0.10 * T)) * ramp(
        (joint[2] + 0.06 * T - local[:, 2]) / (0.06 * T)
    )
    share = behind * across * height * trunk
    return share


SCAPULA_SHARE = 0.25


def scapula_rotation(body, pose, posed):
    """Swing the scapular skin about the root of its spine by a share of the arm's raise.

    Upward rotation turns the inferior angle out and up and the glenoid up (Ludewig et al. 2009).
    Here the rotation axis is the trunk's front-back axis, through a point on the back at the
    height of the joint and 0.07 T from the midline; it adds to whatever the skinning did.
    """
    T = skinning.torso_length(body.bones)
    out = posed.copy()
    for side, sign in (("left", 1.0), ("right", -1.0)):
        rot = None
        for name, value in pose.items():
            if name in ("upperArm", side + "UpperArm"):
                rot = value
        if rot is None:
            continue
        spread = max(rot[2], 0.0)
        forward = max(rot[0], 0.0)
        angle = np.radians(SCAPULA_SHARE * (spread + 0.8 * forward)) * sign
        joint = body.bone(side + "UpperArm")["head"]
        pivot = np.array((sign * 0.07 * T, joint[1] + 0.04 * T, joint[2] - 0.02 * T))
        c, s = np.cos(angle), np.sin(angle)
        turn = np.array(((c, 0, -s), (0, 1, 0), (s, 0, c)))  # about Y: x out raises z
        mask = (np.sign(body.verts[:, 0]) == sign) & (body.scapula > 0)
        share = body.scapula[mask][:, None]
        moved = (out[mask] - pivot) @ turn.T + pivot
        out[mask] = out[mask] + share * (moved - out[mask])
    return out


def variant_dir(variant):
    return OUT / variant


def write_pose(variant, preset, name, posed, faces):
    folder = variant_dir(variant) / preset / "poses"
    folder.mkdir(parents=True, exist_ok=True)
    write_obj(folder / f"{name}.obj", posed, faces)
    (variant_dir(variant) / preset / "fingerprint.json").write_text(
        json.dumps({"pose": fingerprints()["pose"]})
    )


def cmd_poses(args):
    presets = args.presets or TYPICAL
    names = args.only.split(",") if args.only else [*POSES, *TOUCH_POSES, *TOUCH_BOTH]
    report = {}
    with Rhythm(args.variant):
        for preset in presets:
            body = Body(preset, args.variant)
            rows = report[preset] = {}
            for name in names:
                posed = body.posed(body.named(name))
                area = body.rest.area_ratio(posed)
                crossing = body.parts.intersections(posed)["intersectionsByPair"]
                new = {k: v - body.rest_pairs.get(k, 0) for k, v in crossing.items()}
                rows[name] = {
                    "collapsed": int((area < COLLAPSED).sum()),
                    "newByPair": {k: v for k, v in sorted(new.items()) if v > 0},
                }
                if name in TOUCH_POSES:
                    rows[name]["solve"] = body.solved[name].report()
                    rows[name]["angles"] = body.solved[name].angles
                write_pose(args.variant, preset, name, posed, body.faces)
            print(preset, flush=True)
    path = variant_dir(args.variant) / "report.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(report)
    path.write_text(json.dumps(old, indent=1))
    return 0


def cmd_sheet(args):
    from dataclasses import replace

    from sotaigen_dev import review

    review.BUILD = variant_dir(args.variant)
    if args.focus and args.context:
        review.FOCUS[args.focus] = replace(review.FOCUS[args.focus], context=args.context)
    presets = args.presets.split(",") if args.presets else TYPICAL
    names = args.poses.split(",")
    if args.variants:
        # Side by side: copy each variant's pose into one tree as "<variant>~<pose>".
        import shutil

        review.BUILD = OUT / "mix"
        mixed = []
        for name in names:
            for variant in args.variants.split(","):
                label = f"{variant.replace('+', '_')}~{name}"
                mixed.append(label)
                for preset in presets:
                    folder = OUT / "mix" / preset / "poses"
                    folder.mkdir(parents=True, exist_ok=True)
                    shutil.copy(variant_dir(variant) / preset / "poses" / f"{name}.obj",
                                folder / f"{label}.obj")
                    shutil.copy(variant_dir(variant) / preset / "fingerprint.json",
                                OUT / "mix" / preset / "fingerprint.json")
        names = mixed
    review.POSES = {name: {} for name in names}
    axes = review.select_axes(
        presets, pose=",".join(names), view=args.view, focus=args.focus, look=args.look
    )
    rows, cols = {
        "preset": (("look", "preset"), ("pose", "view")),
        "pose": (("look", "pose"), ("preset", "view")),
        "view": (("look", "preset", "view"), ("pose",)),
    }[args.rows]
    out = OUT / "sheets" / f"{args.name}.png"
    review.make_sheet(axes, rows=rows, cols=cols, cell_px=args.cell, out=out)
    print(out)
    return 0


# --- Plots, drawn with Pillow ---

PALETTE = {
    "trunk": (150, 150, 150),
    "Shoulder": (220, 120, 40),
    "UpperArmHalf": (60, 160, 220),
    "UpperArm": (40, 80, 200),
    "UpperArmTwist": (120, 60, 200),
    "other": (90, 180, 90),
}


def bone_family(name):
    for side in ("left", "right"):
        if name.startswith(side):
            rest = name[len(side):]
            if rest.startswith("UpperArmTwist"):
                return "UpperArmTwist"
            if rest in ("Shoulder", "UpperArm", "UpperArmHalf"):
                return rest
            return "other"
    return "trunk"


def scatter(points, colors, size=420, title="", box=None):
    """An orthographic splat of 2D points: x right, y up, equal scale, with a title."""
    image = Image.new("RGB", (size, size + 24), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    lo, hi = box if box is not None else (points.min(0), points.max(0))
    scale = (size - 20) / max(hi - lo)
    order = np.argsort(points[:, 2]) if points.shape[1] > 2 else np.arange(len(points))
    for i in order:
        x = 10 + (points[i, 0] - lo[0]) * scale
        y = 24 + size - 10 - (points[i, 1] - lo[1]) * scale
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=tuple(int(c) for c in colors[i]))
    draw.text((6, 4), title, fill=(0, 0, 0), font=font())
    return image


def cmd_weights(args):
    rig = rigged(args.preset)
    verts, weights, bones = rig.verts, rig.weights, rig.bones
    names = [b["name"] for b in bones]
    if "nocap" in tokens(args.variant):
        weights = variant_weights(args.preset, rig)
    if args.pose:
        with Rhythm(args.variant):
            body = Body(args.preset, args.variant)
            verts = body.posed(POSES[args.pose])
    T = skinning.torso_length(bones)
    joint = bones[names.index("leftUpperArm")]["head"]
    near = (np.linalg.norm(rig.verts - joint, axis=1) < 0.45 * T) & (rig.verts[:, 0] > 0)
    families = np.array([bone_family(n) for n in names])
    top = np.array([families[i] for i in np.argmax(weights, axis=1)])
    dominant = np.array([PALETTE[f] for f in top], float)
    girdle = sum(weights[:, names.index(f"left{n}")] for n in ("Shoulder",))
    arm = sum(
        weights[:, i] for i, n in enumerate(names)
        if n.startswith("leftUpperArm")
    )
    def ramp_color(value, hue):
        return np.array(hue) * value[:, None] + 235 * (1 - value[:, None])
    pts = verts[near]
    panels = []
    views = {
        "front": (pts[:, [0, 2, 1]] * (1, 1, -1)),
        "back": (pts[:, [0, 2, 1]] * (-1, 1, 1)),
        "top": (pts[:, [0, 1, 2]] * (1, -1, 1)),
    }
    for title, colors in (
        ("strongest bone", dominant[near]),
        ("Shoulder (girdle) weight", ramp_color(girdle[near], (220, 100, 20))),
        ("UpperArm + half + twist", ramp_color(arm[near], (30, 70, 210))),
    ):
        for view, xy in views.items():
            panels.append(scatter(xy, colors, title=f"{args.preset} {args.pose or 'rest'} "
                                  f"{view}: {title}"))
    width = panels[0].width
    sheet = Image.new("RGB", (width * 3, panels[0].height * 3), (255, 255, 255))
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % 3) * width, (i // 3) * panel.height))
    draw = ImageDraw.Draw(sheet)
    x = 8
    for family, color in PALETTE.items():
        draw.rectangle((x, sheet.height - 18, x + 12, sheet.height - 6), fill=color)
        draw.text((x + 16, sheet.height - 20), family, fill=(0, 0, 0), font=font(12))
        x += 120
    out = OUT / "plots" / f"weights-{args.preset}-{args.variant}-{args.pose or 'rest'}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(out)
    return 0


def section_points(verts, faces, origin, normal):
    """Where the mesh's edges cross the plane, as 3D points."""
    d = (verts - origin) @ normal
    pts = []
    for face in faces:
        for a, b in zip(face, (*face[1:], face[0]), strict=True):
            if d[a] * d[b] < 0:
                t = d[a] / (d[a] - d[b])
                pts.append(verts[a] + t * (verts[b] - verts[a]))
    return np.array(pts)


def cmd_section(args):
    """Frontal section through the left shoulder joint, for each pose and variant."""
    images = []
    for variant in args.variants.split(","):
        with Rhythm(variant):
            body = Body(args.preset, variant)
            for name in args.poses.split(","):
                images.append(section_image(body, name, variant))
    width, height = images[0].size
    columns = len(args.poses.split(","))
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (width * columns, height * rows), (255, 255, 255))
    for i, image in enumerate(images):
        sheet.paste(image, ((i % columns) * width, (i // columns) * height))
    out = OUT / "plots" / f"section-{args.preset}-{args.tag or args.poses.replace(',', '_')}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(out)
    return 0


def section_image(body, name, variant, size=460):
    pose = POSES[name]
    posed = body.posed(pose)
    skins = pose_skins(body.bones, pose)
    T = skinning.torso_length(body.bones)
    arm = body.names.index("leftUpperArm")
    chest = body.names.index("upperChest")
    joint = skins[arm] @ np.append(body.bone("leftUpperArm")["head"], 1)
    # The plane holds the joint, the world's up and the arm's posed direction (its raise plane).
    tail = skins[arm] @ np.append(body.bone("leftUpperArm")["tail"], 1)
    along = (tail - joint)[:3]
    normal = np.cross(along, (0, 0, 1.0))
    if np.linalg.norm(normal) < 1e-6:
        normal = np.array((0, 1.0, 0))
    normal /= np.linalg.norm(normal)
    origin = joint[:3]
    u = np.cross((0, 0, 1.0), normal)
    u /= np.linalg.norm(u)
    if u[0] < 0:
        u = -u
    v = np.array((0, 0, 1.0))
    faces = body.faces
    skin = section_points(posed, faces, origin, normal)
    rigid_trunk = (np.column_stack((body.verts, np.ones(len(body.verts)))) @ skins[chest].T)[:, :3]
    rigid_arm = (np.column_stack((body.verts, np.ones(len(body.verts)))) @ skins[arm].T)[:, :3]
    segments = np.asarray(body.base["segments"])
    trunk_faces = [f for f in faces if all(segments[i] == "trunk" for i in f)]
    arm_faces = [f for f in faces if all(segments[i] == "upperArm" for i in f)]
    layers = [
        (section_points(rigid_trunk, trunk_faces, origin, normal), (160, 190, 230)),
        (section_points(rigid_arm, arm_faces, origin, normal), (230, 170, 150)),
        (skin, (0, 0, 0)),
    ]
    image = Image.new("RGB", (size, size + 24), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    half = 0.42 * T
    centre = origin
    scale = (size - 20) / (2 * half)
    for pts, color in layers:
        if len(pts) == 0:
            continue
        rel = pts - centre
        keep = (np.abs(rel @ u) < half) & (np.abs(rel @ v) < half) & (
            np.linalg.norm(rel, axis=1) < 1.5 * half
        )
        for p in rel[keep]:
            x = size / 2 + (p @ u) * scale
            y = 24 + size / 2 - (p @ v) * scale
            r = 1.6 if color == (0, 0, 0) else 1.2
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    jx, jy = size / 2, 24 + size / 2
    draw.line((jx - 6, jy, jx + 6, jy), fill=(200, 0, 0))
    draw.line((jx, jy - 6, jx, jy + 6), fill=(200, 0, 0))
    grid = 0.1 * T * scale
    draw.line((10, size + 18, 10 + grid, size + 18), fill=(0, 0, 0), width=2)
    draw.text((14 + grid, size + 10), "0.1 T", fill=(0, 0, 0), font=font(11))
    draw.text((6, 4), f"{body.preset} {variant} {name}", fill=(0, 0, 0), font=font())
    return image


def read_obj_verts(path):
    with open(path) as handle:
        return np.array([line.split()[1:4] for line in handle if line.startswith("v ")], float)


def top_outline(verts, edges, keep, xs):
    """The highest z where the mesh's edges cross each x, over the edges `keep` allows."""
    a, b = verts[edges[:, 0]], verts[edges[:, 1]]
    out = np.full(len(xs), np.nan)
    for i, x in enumerate(xs):
        cross = keep & ((a[:, 0] - x) * (b[:, 0] - x) < 0)
        if cross.any():
            t = (x - a[cross, 0]) / (b[cross, 0] - a[cross, 0])
            out[i] = (a[cross, 2] + t * (b[cross, 2] - a[cross, 2])).max()
    return out


def cmd_profile(args):
    from sotaigen.mesh import edges_of

    colors = [(120, 120, 120), (200, 40, 40), (40, 90, 210), (30, 160, 80), (210, 140, 20)]
    presets = args.presets.split(",") if args.presets else TYPICAL
    names, variants = args.poses.split(","), args.variants.split(",")
    size, pad = 420, 8
    sheet = Image.new("RGB", (size * len(names), (size + 24) * len(presets)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for row, preset in enumerate(presets):
        rig = rigged(preset)
        edges = edges_of(rig.base["faces"])
        segments = np.asarray(rig.base["segments"])
        keep = (~np.isin(segments[edges], ("head", "lowerArm", "hand"))).all(axis=1)
        keep &= (np.asarray(rig.base["verts"])[edges, 0] > 0).all(axis=1)
        T = skinning.torso_length(rig.bones)
        joint = rig.bones[[b["name"] for b in rig.bones].index("leftUpperArm")]["head"]
        xs = np.linspace(0.02 * T, joint[0] + 0.45 * T, 90)
        for column, name in enumerate(names):
            x0, y0 = column * size, row * (size + 24)
            draw.text((x0 + 6, y0 + 4), f"{preset} {name}", fill=(0, 0, 0), font=font())
            scale = (size - 2 * pad) / (xs[-1] - xs[0])
            for variant, color in zip(variants, colors, strict=False):
                path = variant_dir(variant) / preset / "poses" / f"{name}.obj"
                z = top_outline(read_obj_verts(path), edges, keep, xs)
                points = [
                    (x0 + pad + (x - xs[0]) * scale,
                     y0 + 24 + size / 2 - (h - joint[2] - 0.15 * T) * scale)
                    for x, h in zip(xs, z, strict=True) if np.isfinite(h)
                ]
                draw.line(points, fill=color, width=2)
    x = 8
    for variant, color in zip(variants, colors, strict=False):
        draw.text((x, sheet.height - 16), variant, fill=color, font=font(12))
        x += 160
    out = OUT / "plots" / f"profile-{args.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(out)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    poses = commands.add_parser("poses")
    poses.add_argument("presets", nargs="*")
    poses.add_argument("--variant", default="prod", help="tokens joined by +")
    poses.add_argument("--only")
    sheet = commands.add_parser("sheet")
    sheet.add_argument("name")
    sheet.add_argument("--poses", required=True)
    sheet.add_argument("--variant", default="prod")
    sheet.add_argument("--variants", help="side by side, comma list")
    sheet.add_argument("--presets")
    sheet.add_argument("--view", default="front")
    sheet.add_argument("--look", default="clay,toon")
    sheet.add_argument("--focus")
    sheet.add_argument("--context", type=float)
    sheet.add_argument("--rows", choices=("preset", "pose", "view"), default="preset")
    sheet.add_argument("--cell", type=int, default=240)
    weights = commands.add_parser("weights")
    weights.add_argument("preset", metavar="name")
    weights.add_argument("--pose")
    weights.add_argument("--variant", default="prod")
    section = commands.add_parser("section")
    section.add_argument("preset", metavar="name")
    section.add_argument("--poses", required=True)
    section.add_argument("--variants", default="prod")
    section.add_argument("--tag")
    profile = commands.add_parser("profile")
    profile.add_argument("name")
    profile.add_argument("--poses", required=True)
    profile.add_argument("--variants", required=True)
    profile.add_argument("--presets")
    args = parser.parse_args(argv)
    run = {"poses": cmd_poses, "sheet": cmd_sheet, "weights": cmd_weights,
           "section": cmd_section, "profile": cmd_profile}
    return run[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
