"""Real motion clips on the rig: fetch them, retarget them, measure where each pose breaks, look.

uv run sotaigen_dev/motion/clips.py fetch
uv run sotaigen_dev/motion/clips.py glb female-long --clip walk -o build/reports/motions/walk.glb
uv run sotaigen_dev/motion/clips.py glb female-long --clip cylinderPickup -o build/reports/motions/pickup.glb
uv run sotaigen_dev/motion/clips.py contacts female-long --clip walk
uv run sotaigen_dev/motion/clips.py measure [body ...] [--clip walk,run] [--out table.json]
uv run sotaigen_dev/motion/clips.py sheet <body ...> --clip walk [--frames worst|N,N] [--focus hip]
    [--view front,side] [--look clay,toon] [--against DIR] [-o out.png]

`fetch` downloads what motions/SOURCES.md lists to build/reports/motions/raw/, checks each file's digest and
reduces it to motions/clips/<clip>.json, which is committed and which everything else reads. A clip
holds no angles of any one rig: per bone, world-space rotations away from the source's rest pose and
the direction the source's bone had at rest, so that it fits a body of any proportions.

The clips are this tool's test data, not a format for others: CLIP_FORMAT and CLIP_VERSION only let
load_clip refuse a file written another way.
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sotaigen.corrective.crossings import SkinParts
from sotaigen.corrective.recipe import Correctives
from sotaigen.form.bodies import load_specs
from sotaigen.metrics import COLLAPSED, RestMesh, count_by, pose_row
from sotaigen.pipeline import BUILD, fingerprints, rigged, surface
from sotaigen.rig.poses import POSES
from sotaigen.rig.skinning import pose_verts, rest_matrix
from sotaigen_dev.motion.foot_contact import (
    aligning,
    contact_errors,
    contact_tracks,
    euler_angles,
    ground_motion,
)
from sotaigen_dev.motion.hand_motion import PERFORMANCES, hand_sample

CLIPS_DIR = ROOT / "motions" / "clips"
CLIP_FORMAT, CLIP_VERSION = "sotaigen/motion-clip", 1
RAW = BUILD / "reports" / "motions" / "raw"
CMU = "http://mocap.cs.cmu.edu/subjects"
# The files, by digest, that motions/SOURCES.md names; tests/test_motion_clips.py holds the two tables together.
FILES = {
    "35.asf": "2a8e2eda3c0d7d828566b2a9a8ab36b2b8b3864110574e8b73c8f069fded416c",
    "35_01.amc": "0743f4ea48e7e199cd56b2810b5ce81f8ede08d32ff79aa4e363c44cc4fe33aa",
    "35_17.amc": "8096cd53878242dbc35e987288cef6ae81d4e1c0e207086639bbed5cc06c8fee",
    "13.asf": "eb33390bcbccb739d7622bd583fd9be55020868f68689aa6f2060ad688df10dc",
    "13_01.amc": "e65c743d86ea97a6c1b27d38a5ad95529c08042075eb490e54953030b2b230be",
    "13_04.amc": "bc5f5e5b6378413457f782e96c2beedf70c5abc6bdcabca513f8f5b0d966d7e9",
    "13_27.amc": "8679214b96f59a8c49210131373bca9303767e6780073ef35a3b36045955ccfe",
}
# Clip name: subject's skeleton, motion file, what it is, and seconds to keep. 120 frames a second.
CLIP_SPECS = {
    "walk": ("35.asf", "35_01.amc", "walk", (0.0, None)),
    "run": ("35.asf", "35_17.amc", "run/jog", (0.0, None)),
    "sitStool": ("13.asf", "13_01.amc", "sit on high stool, stand up", (0.0, 6.0)),
    "sitLow": ("13.asf", "13_04.amc", "sit on stepstool, chin in hand", (0.0, 8.0)),
    "wave": ("13.asf", "13_27.amc", "direct traffic, wave, point", (0.0, 12.0)),
}
SOURCE_FPS = 120.0
# Half of the original 120 Hz for playback. The diagnostic frame indices remain at 10 Hz.
PLAYBACK_FPS = 60.0
REVIEW_FPS = 10.0
# A target bone and the source bone whose motion it takes. The root drives the hips.
SOURCE_BONE = {
    "hips": "root",
    "spine": "lowerback",
    "chest": "upperback",
    "upperChest": "thorax",
    "neck": "upperneck",
    "head": "head",
    "Shoulder": "clavicle",
    "UpperArm": "humerus",
    "LowerArm": "radius",
    "Hand": "hand",
    "UpperLeg": "femur",
    "LowerLeg": "tibia",
    "Foot": "foot",
    "Toes": "toes",
}
# Source axes (left +X, up +Y, front +Z) in this project's (left +X, front -Y, up +Z).
FROM_SOURCE = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
FRONT = (0.0, -1.0, 0.0)
LEG_BONES = ("UpperLeg", "LowerLeg", "Foot", "Toes")
LIMBS_ONTO_SOURCE = ("UpperArm", "LowerArm", "Hand", *LEG_BONES)


def source_name(target):
    """The source bone for a target bone, or None for one a clip doesn't drive."""
    if target in SOURCE_BONE:
        return SOURCE_BONE[target]
    for side, letter in (("left", "l"), ("right", "r")):
        if target.startswith(side):
            limb = SOURCE_BONE.get(target[len(side) :])
            return f"{letter}{limb}" if limb else None
    return None


# --- Reading the source format ---


def rotation(degrees):
    """Rz Ry Rx in degrees, as ASF and AMC mean it."""
    x, y, z = np.radians(degrees)
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def read_asf(path):
    """The skeleton: per bone its rest direction, its axis matrix, its degrees of freedom, and the
    bones in parents-first order as (name, parent)."""
    bones, order, section = {}, [], None
    current = None
    for line in Path(path).read_text().splitlines():
        words = line.split()
        if not words or words[0].startswith("#"):
            continue
        if words[0].startswith(":"):
            section = words[0]
            continue
        if section == ":bonedata":
            if words[0] == "begin":
                current = {"dof": []}
            elif words[0] == "end":
                bones[current["name"]] = current
            elif words[0] == "name":
                current["name"] = words[1]
            elif words[0] == "direction":
                current["direction"] = np.array(words[1:4], float)
            elif words[0] == "length":
                current["length"] = float(words[1])
            elif words[0] == "axis":
                current["axis"] = rotation(np.array(words[1:4], float))
            elif words[0] == "dof":
                current["dof"] = words[1:]
        elif section == ":hierarchy" and words[0] != "begin" and words[0] != "end":
            order += [(child, words[0]) for child in words[1:]]
    return bones, order


def read_amc(path):
    """Frames as a list of {bone: values}, in file order."""
    frames = []
    for line in Path(path).read_text().splitlines():
        words = line.split()
        if not words or line[0] in "#:":
            continue
        if words[0].isdigit() and len(words) == 1:
            frames.append({})
        elif frames:
            frames[-1][words[0]] = np.array(words[1:], float)
    return frames


def world_rotations(bones, order, frame):
    """Each source bone's rotation away from the rest pose, in world axes, for one frame."""
    world = {"root": rotation(frame["root"][3:6])}
    for child, parent in order:
        bone = bones[child]
        values = frame.get(child, ())
        angles = np.zeros(3)
        for dof, value in zip(bone["dof"], values, strict=True):
            angles["xyz".index(dof[1])] = value
        axis = bone["axis"]
        world[child] = world[parent] @ axis @ rotation(angles) @ axis.T
    return world


def quaternion(matrix):
    """A rotation matrix as (w, x, y, z)."""
    m = matrix
    trace = np.trace(m)
    if trace > 0:
        s = 2 * np.sqrt(1 + trace)
        q = (s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(max(1 + m[i, i] - m[j, j] - m[k, k], 1e-12))
        q = np.zeros(4)
        q[0] = (m[k, j] - m[j, k]) / s
        q[i + 1] = s / 4
        q[j + 1] = (m[j, i] + m[i, j]) / s
        q[k + 1] = (m[k, i] + m[i, k]) / s
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def matrix_from(q):
    # Capture quaternions are rounded for storage. Restore unit length before using their inverse.
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def make_clip(name):
    """One clip as stored: rotations in this project's axes, at PLAYBACK_FPS."""
    skeleton, motion, what, (start, stop) = CLIP_SPECS[name]
    bones, order = read_asf(RAW / skeleton)
    frames = read_amc(RAW / motion)
    first = int(start * SOURCE_FPS)
    last = int(stop * SOURCE_FPS) if stop else len(frames)
    kept = frames[first : last : int(SOURCE_FPS / PLAYBACK_FPS)]
    used = sorted({value for key, value in iter_source_bones()})
    rotations = {bone: [] for bone in used}
    length = bones["lfemur"]["length"] + bones["ltibia"]["length"]
    root_path = []
    tracks = {side: [] for side in ("left", "right")}
    for frame in kept:
        world = world_rotations(bones, order, frame)
        root = frame["root"][:3]
        root_path.append(FROM_SOURCE @ (root - kept[0]["root"][:3]) / length)
        for side, prefix in (("left", "l"), ("right", "r")):
            ankle = root.copy()
            for limb in ("hipjoint", "femur", "tibia"):
                key = prefix + limb
                ankle += world[key] @ bones[key]["direction"] * bones[key]["length"]
            key = prefix + "foot"
            ball = ankle + world[key] @ bones[key]["direction"] * bones[key]["length"]
            tracks[side].append([FROM_SOURCE @ point / length for point in (ankle, ball)])
        for bone in used:
            turned = FROM_SOURCE @ world[bone] @ FROM_SOURCE.T
            rotations[bone].append([round(float(v), 4) for v in quaternion(turned)])
    rest = {
        bone: (FROM_SOURCE @ bones[bone]["direction"]).round(5).tolist()
        for bone in used
        if bone != "root"
    }
    return {
        "format": CLIP_FORMAT,
        "version": CLIP_VERSION,
        "name": name,
        "source": f"CMU Graphics Lab Motion Capture Database, {motion}: {what}",
        "fps": PLAYBACK_FPS,
        "hipShare": round(hip_share(bones), 4),
        "rootPath": np.round(root_path, 6).tolist(),
        "footContacts": {key: contact_tracks(track, PLAYBACK_FPS) for key, track in tracks.items()},
        "rest": rest,
        "rotations": rotations,
    }


def hip_share(bones):
    """The source's hip joint out from the midline, as a share of its leg's length."""
    hip = bones["lhipjoint"]
    out = hip["direction"][0] * hip["length"]
    return float(out / (bones["lfemur"]["length"] + bones["ltibia"]["length"]))


def iter_source_bones():
    for key, value in SOURCE_BONE.items():
        if key[0].islower():
            yield key, value
        else:
            for side, letter in (("left", "l"), ("right", "r")):
                yield f"{side}{key}", f"{letter}{value}"


def cmd_fetch(_argv):
    RAW.mkdir(parents=True, exist_ok=True)
    for file, digest in FILES.items():
        path = RAW / file
        if not path.exists():
            url = f"{CMU}/{file.split('_')[0].split('.')[0]}/{file}"
            print(f"downloading {url}")
            with urllib.request.urlopen(url, timeout=120) as reply:
                path.write_bytes(reply.read())
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise SystemExit(f"{path} is not the file motions/SOURCES.md names")
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    for name in CLIP_SPECS:
        clip = make_clip(name)
        (CLIPS_DIR / f"{name}.json").write_text(json.dumps(clip, separators=(",", ":")))
        print(f"{name}: {len(clip['rotations']['root'])} frames")
    return 0


# --- Retargeting to a rig ---


def load_clip(name):
    """The 10 Hz review frames and their original full-rate capture for solvers and export."""
    recorded = json.loads((CLIPS_DIR / f"{name}.json").read_text())
    if (recorded.get("format"), recorded.get("version")) != (CLIP_FORMAT, CLIP_VERSION):
        raise ValueError(f"{name}: not a {CLIP_FORMAT} clip of version {CLIP_VERSION}: run fetch again")
    stride = int(recorded["fps"] / REVIEW_FPS)
    if stride <= 1:
        return recorded
    return {
        **recorded,
        "fps": recorded["fps"] / stride,
        "rotations": {name: values[::stride] for name, values in recorded["rotations"].items()},
        "rootPath": recorded["rootPath"][::stride],
        "footContacts": {side: track[::stride] for side, track in recorded["footContacts"].items()},
        "capture": recorded,
    }


def clip_names():
    return sorted(path.stem for path in CLIPS_DIR.glob("*.json"))


def hip_share_of(by_name):
    """A rig's hip joint out from the midline, as a share of its leg's length."""
    thigh, shin = by_name["leftUpperLeg"], by_name["leftLowerLeg"]
    length = np.linalg.norm(thigh["tail"] - thigh["head"]) + np.linalg.norm(
        shin["tail"] - shin["head"]
    )
    return float(thigh["head"][0] / length)


def about_front(angle):
    """A rotation about the front axis that takes straight down (-Z) towards the wearer's left."""
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, 0.0, -sine], [0.0, 1.0, 0.0], [sine, 0.0, cosine]])


def heading_frame(clip, frame):
    """Face forward at the clip origin, retaining turns throughout a travelling clip."""
    front = matrix_from(clip["rotations"]["root"][frame]) @ np.array(FRONT)
    heading = np.arctan2(front[0], -front[1])
    cosine, sine = np.cos(heading), np.sin(heading)
    return np.array([[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def asked_of(clip, bones, frame):
    """What a clip wants of each driven bone in one frame: its rotation away from rest, in world axes.

    A limb at rest may point elsewhere than the source's did (arms hang lower here), so the source's
    rotation starts from the source's rest pose, after turning the target bone onto it. The trunk
    and girdle already match closely enough at rest for that to do only harm. Travelling clips
    face forward in the first frame and retain the captured turns.

    A person's legs lean in from wide hips so that the feet land near the midline. Copied onto
    narrower hips the same lean crosses the legs, so each leg is turned out by the difference in
    hip width, as a share of leg length. Footfalls keep their place, not the leg's exact angle.
    """
    delta = {}
    straighten = heading_frame(clip, 0)
    hips = straighten @ matrix_from(clip["rotations"]["root"][frame])
    by_name = {bone["name"]: bone for bone in bones}
    spread = np.arctan(clip["hipShare"] - hip_share_of(by_name))
    for bone in bones:
        name = bone["name"]
        source = source_name(name)
        if source is None or source not in clip["rotations"]:
            continue
        turned = straighten @ matrix_from(clip["rotations"][source][frame])
        if name.endswith(LEG_BONES):
            # About the front axis: to the wearer's left for the left leg, and the right for the right.
            angle = spread if name.startswith("left") else -spread
            turned = hips @ about_front(angle) @ hips.T @ turned
        if name.endswith(LIMBS_ONTO_SOURCE):
            direction = rest_matrix(bone)[:3, 1]
            turned = turned @ aligning(direction, np.array(clip["rest"][source]))
        delta[name] = turned
    return delta


def retargeted(clip, bones, frame):
    """A clip's frame as a pose for these bones: for each driven bone, its local Euler angles."""
    asked = asked_of(clip, bones, frame)
    delta = {}
    pose = {}
    for bone in bones:
        name, parent = bone["name"], bone["parent"]
        inherited = delta[parent] if parent else np.eye(3)
        if name not in asked:
            delta[name] = inherited
            continue
        rest = rest_matrix(bone)[:3, :3]
        delta[name] = asked[name]
        pose[name] = euler_angles(rest.T @ inherited.T @ asked[name] @ rest)
    thigh, shin = (
        next(b for b in bones if b["name"] == "left" + part)
        for part in ("UpperLeg", "LowerLeg")
    )
    length = sum(np.linalg.norm(b["tail"] - b["head"]) for b in (thigh, shin))
    pose["rootTranslation"] = (
        heading_frame(clip, 0) @ clip["rootPath"][frame] * length
    ).tolist()
    return pose


def playback_samples(clip, rig):
    """Solve contact on the captured 60 Hz trajectory before any diagnostic downsampling."""
    clip = clip.get("capture", clip)
    poses = [retargeted(clip, rig.bones, frame) for frame in range(frame_count(clip))]
    ground_motion(rig, poses, clip["footContacts"], clip["fps"])
    return clip["fps"], poses


def motion_poses(clip, rig):
    """The solved poses at the original capture indices used by measure and sheet."""
    fps, poses = playback_samples(clip, rig)
    return poses[:: round(fps / clip["fps"])]


def frame_count(clip):
    return len(clip["rotations"]["root"])


# --- Measuring ---


def evaluate(preset, clip, parts=None):
    """Per frame of a clip on a body: collapsed faces, areas, skin passing through skin."""
    rig = rigged(preset)
    base, verts, bones, weights = rig.base, rig.verts, rig.bones, rig.weights
    parts = parts or SkinParts(base)
    rest = RestMesh(verts, base["faces"])
    resting = parts.intersections(verts)["intersectionsByPair"]
    segments = np.asarray(base["segments"])
    face_segment = np.array([segments[face[0]] for face in base["faces"]])
    rows = []
    correctives = Correctives.production(base, verts, bones, weights, field=surface(preset)[0])
    for frame, pose in enumerate(motion_poses(clip, rigged(preset))):
        posed = pose_verts(bones, weights, correctives.bind(verts, pose), pose)
        row = pose_row(rest, posed)
        row["collapsedBy"] = count_by(face_segment[rest.area_ratio(posed) < COLLAPSED])
        crossing = parts.intersections(posed)["intersectionsByPair"]
        # What the rest pose already has is the template's, not the motion's.
        pairs = {k: v - resting.get(k, 0) for k, v in crossing.items()}
        row["pairs"] = {k: v for k, v in pairs.items() if v > 0}
        body_frame = posed
        if "rootTranslation" in pose:
            unturn = heading_frame(clip, frame) @ heading_frame(clip, 0).T
            body_frame = (posed - pose["rootTranslation"]) @ unturn.T
        row["legsPastMidline"] = parts.legs_past_midline(body_frame)
        rows.append(row)
    return rows


def summary(rows):
    """A clip's worst frames: the most collapsed, the most skin passing through, by pair."""
    pairs, collapsed = {}, {}
    for row in rows:
        for key, value in row["pairs"].items():
            pairs[key] = max(pairs.get(key, 0), value)
        for key, value in row["collapsedBy"].items():
            collapsed[key] = max(collapsed.get(key, 0), value)
    return {
        "frames": len(rows),
        "maxCollapsed": max(row["collapsedFaces"] for row in rows),
        "collapsedBySegment": collapsed,
        "worstAreaP1": min(row["areaP1"] for row in rows),
        "maxLegsPastMidline": max(row["legsPastMidline"] for row in rows),
        "maxByPair": dict(sorted(pairs.items(), key=lambda item: -item[1])),
        "worstFrame": int(np.argmax([sum(row["pairs"].values()) for row in rows])),
    }


def cmd_measure(argv):
    parser = argparse.ArgumentParser(prog="clips.py measure")
    parser.add_argument("presets", nargs="*")
    parser.add_argument("--clip", default="")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    presets = args.presets or list(load_specs("typical"))
    names = [name for name in args.clip.split(",") if name] or clip_names()
    results = {}
    for preset in presets:
        base = rigged(preset).base
        parts = SkinParts(base)
        for name in names:
            row = summary(evaluate(preset, load_clip(name), parts))
            results.setdefault(preset, {})[name] = row
            print(preset, name, json.dumps(row))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
    return 0


# --- Looking ---


def write_frames(preset, clip, frames, root):
    """Write the frames as a build tree that sotaigen_dev/review.py can read, vouched for as current."""
    from sotaigen.mesh import write_obj

    rig = rigged(preset)
    base, verts, bones, weights = rig.base, rig.verts, rig.bones, rig.weights
    out = root / preset / "poses"
    out.mkdir(parents=True, exist_ok=True)
    names = []
    samples = motion_poses(clip, rigged(preset))
    correctives = Correctives.production(base, verts, bones, weights, field=surface(preset)[0])
    for frame in frames:
        pose = f"{clip['name']}{frame:03d}"
        posed = pose_verts(bones, weights, correctives.bind(verts, samples[frame]), samples[frame])
        write_obj(out / f"{pose}.obj", posed, base["faces"])
        names.append(pose)
    (root / preset / "fingerprint.json").write_text(json.dumps({"pose": fingerprints()["pose"]}))
    return names


def review_module():
    """A copy of the review tool of its own, whose BUILD and POSES this sets without touching the one
    everyone else imports."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("review", importlib.util.find_spec("sotaigen_dev.review").origin)
    module = importlib.util.module_from_spec(spec)
    sys.modules["review"] = module
    spec.loader.exec_module(module)
    return module


def cmd_contacts(argv):
    """Measure the contact trajectory on the actual skinned sole, in millimetres."""
    parser = argparse.ArgumentParser(prog="clips.py contacts")
    parser.add_argument("preset", metavar="name")
    parser.add_argument("--clip", choices=clip_names(), required=True)
    args = parser.parse_args(argv)
    rig = rigged(args.preset)
    clip = load_clip(args.clip)
    capture = clip.get("capture", clip)
    poses = [retargeted(capture, rig.bones, i) for i in range(frame_count(capture))]

    contacts = capture["footContacts"]
    before = contact_errors(rig, poses, contacts)
    ground_motion(rig, poses, contacts, capture["fps"])
    after = contact_errors(rig, poses, contacts)
    print(json.dumps({"fps": capture["fps"], "rootOnly": before, "contactIK": after}, indent=2))
    return 0


def cmd_glb(argv):
    """A standard glTF animation player can replay the same feet, travel and helper bones."""
    from sotaigen.output.face import face_parts, head_normals
    from sotaigen.output.glb import glb_bytes
    from sotaigen.uv import export_fields

    parser = argparse.ArgumentParser(prog="clips.py glb")
    parser.add_argument("preset", metavar="name")
    parser.add_argument("--clip", choices=[*clip_names(), *PERFORMANCES], required=True)
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rig = rigged(args.preset)
    if args.clip in PERFORMANCES:
        fps = 30
        frames = 3 * fps
        samples = fps, [hand_sample(args.clip, i / frames) for i in range(frames + 1)]
    else:
        samples = playback_samples(load_clip(args.clip), rig)
    base = {**rig.base, **export_fields(rig.base["faces"])}
    blob = glb_bytes(
        base,
        rig.verts,
        rig.bones,
        rig.weights,
        parts=face_parts(args.preset),
        normals=head_normals(args.preset),
        motions=[(args.clip, *samples)],
        correctives=Correctives.production(
            base, rig.verts, rig.bones, rig.weights, field=surface(args.preset)[0]
        ),
    )
    if args.clip == "cylinderPickup":
        from sotaigen_dev.motion.hand_prop import add_pickup_handle

        blob = add_pickup_handle(blob, rig.bones, len(samples[1]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(blob)
    print(args.out)
    return 0


def cmd_sheet(argv):
    parser = argparse.ArgumentParser(prog="clips.py sheet")
    parser.add_argument("presets", nargs="+")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--frames", default="worst")
    parser.add_argument("--focus")
    parser.add_argument("--view", default="threequarter")
    parser.add_argument("--look", default="clay,toon")
    parser.add_argument("--mm-per-px", type=float)
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args(argv)
    clip = load_clip(args.clip)
    if args.frames == "worst":
        rows = evaluate(args.presets[0], clip)
        scores = [sum(row["pairs"].values()) + row["collapsedFaces"] for row in rows]
        frames = sorted(np.argsort(scores)[-3:].tolist())
    else:
        frames = [int(frame) for frame in args.frames.split(",")]
    root = BUILD / "reports" / "motions" / "poses"
    poses = []
    for preset in args.presets:
        poses = write_frames(preset, clip, frames, root)
    review = review_module()
    review.BUILD = root
    review.POSES = {pose: {} for pose in poses}
    axes = review.select_axes(
        args.presets, pose=",".join(poses), view=args.view, focus=args.focus, look=args.look
    )
    review.make_sheet(axes, mm_per_px=args.mm_per_px, out=args.out)
    return 0


def curled(fingers, proximal, intermediate, distal):
    joints = (("Proximal", proximal), ("Intermediate", intermediate), ("Distal", distal))
    return {f"{finger}{joint}": (angle, 0, 0) for finger in fingers for joint, angle in joints}


LONG_FINGERS = ("index", "middle", "ring", "little")
# A finger bends about local X, towards the palm. A thumb has the palm's axes too, so its own bend,
# across the palm, is about local Z, negative on either hand once sided has mirrored it.
THUMB_ACROSS = {
    "thumbMetacarpal": (20, 0, -30),
    "thumbProximal": (10, 0, -30),
    "thumbDistal": (10, 0, -40),
}
# What a character's hands are asked to do; the source data has no fingers to show it.
HAND_POSES = {
    **{
        name: POSES[name]
        for name in (
            "padPinch",
            "cylinderGrip",
            "thinCylinderGrip",
            "sphereGrip",
            "littleOpposition",
            "handRelaxed",
            "handPoint",
            "handWave",
            "handLightTouch",
        )
    },
    "handRest": {},
    "handGrip": {**curled(LONG_FINGERS, 45, 55, 30), **THUMB_ACROSS},
    "handFist": {**curled(LONG_FINGERS, 80, 95, 60), **THUMB_ACROSS},
    "handVee": {
        **curled(LONG_FINGERS[2:], 80, 95, 60),
        **THUMB_ACROSS,
        "indexProximal": (0, 0, 10),
        "middleProximal": (0, 0, -10),
    },
    "handSpread": {
        "indexProximal": (0, 0, 15),
        "middleProximal": (0, 0, 4),
        "ringProximal": (0, 0, -6),
        "littleProximal": (0, 0, -18),
        "thumbMetacarpal": (0, 0, 20),
    },
    "handThumbUp": {**curled(LONG_FINGERS, 80, 95, 60), "thumbMetacarpal": (0, 0, 15)},
}


def cmd_hands(argv):
    parser = argparse.ArgumentParser(prog="clips.py hands")
    parser.add_argument("presets", nargs="+")
    parser.add_argument("--view", default="front,side")
    parser.add_argument("--look", default="clay,toon")
    parser.add_argument("--mm-per-px", type=float)
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args(argv)
    from sotaigen.mesh import write_obj

    root = BUILD / "reports" / "motions" / "poses"
    for preset in args.presets:
        rig = rigged(preset)
        base, verts, bones, weights = rig.base, rig.verts, rig.bones, rig.weights
        parts = SkinParts(base)
        rest = RestMesh(verts, base["faces"])
        out = root / preset / "poses"
        out.mkdir(parents=True, exist_ok=True)
        resting = parts.intersections(verts)["selfIntersections"]
        for pose, rotations in HAND_POSES.items():
            posed = pose_verts(bones, weights, verts, rotations)
            write_obj(out / f"{pose}.obj", posed, base["faces"])
            row = pose_row(rest, posed)
            crossing = parts.intersections(posed)["selfIntersections"] - resting
            print(preset, pose, row["collapsedFaces"], row["areaP1"], crossing)
        (root / preset / "fingerprint.json").write_text(
            json.dumps({"pose": fingerprints()["pose"]})
        )
    review = review_module()
    review.BUILD = root
    review.POSES = {pose: {} for pose in HAND_POSES}
    axes = review.select_axes(
        args.presets, pose=",".join(HAND_POSES), view=args.view, focus="hand", look=args.look
    )
    review.make_sheet(axes, mm_per_px=args.mm_per_px, out=args.out)
    return 0


COMMANDS = {
    "fetch": cmd_fetch,
    "measure": cmd_measure,
    "sheet": cmd_sheet,
    "glb": cmd_glb,
    "contacts": cmd_contacts,
    "hands": cmd_hands,
}


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] not in COMMANDS:
        print(__doc__, file=sys.stderr)
        return 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
