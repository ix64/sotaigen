"""A real clip through the delivered .blend: retarget it, key it on the controls, read the file back.

uv run sotaigen-dev motion-blend run <body ...> --clip wave,walk,sitLow
uv run sotaigen-dev motion-blend sheet <body ...> --clip wave --frames 24,48 [--limits on,off]
uv run sotaigen-dev motion-blend film <body> --clip wave [--limits on|off]

Everything goes to build/reports/motion-blend/<body>/<clip>/: motion.blend (the keyed file, which opens with
Auto Run off and with the joint limits off), read.npz (the skin Blender evaluates on every tenth of a
second, limits off and on), report.json. No contact table is baked: the correctives are the file's
own, the default sculpted ones.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scipy.ndimage import gaussian_filter1d

from sotaigen.corrective.recipe import Correctives
from sotaigen.output.blend import blend_body, blend_path
from sotaigen.output.blender import run_blender
from sotaigen.output.glb import driver_table
from sotaigen.pipeline import BUILD, fingerprints, rigged, surface
from sotaigen.rig.limits import clipped, joint_limits
from sotaigen.rig.skinning import euler, pose_verts, rest_matrix, swing_twist
from sotaigen_dev.motion.clips import load_clip, playback_samples, review_module
from sotaigen_dev.motion.foot_contact import aligning, contact_errors, euler_angles

ANIMATE = ROOT / "blender" / "review" / "animate.py"
OUT = BUILD / "reports" / "motion-blend"
SAMPLE_EVERY = 6  # of 60 Hz: the 10 Hz frame numbers sotaigen_dev/motion/clips.py and the docs use.
# What docs/DECISIONS.md and docs/systems/rig.md say the mesh is clean to, as it reads the control.
CLEAN = {"forward": 95.0, "side": 55.0}


def folder(preset, clip):
    path = OUT / preset / clip
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- Fitting a retargeted clip to the controls ---
#
# sotaigen_dev/motion/clips.py turns a target limb onto the source's rest direction by the smallest rotation,
# which leaves the roll free. The runtime takes any local rotation, so that goes unseen there; the
# delivered controls have hinges at the elbow and knee and carry the forearm's turn on the hand, so
# they refuse it. What the source's hinge did has to come back as hinges: the error about the bone
# becomes upper-limb twist, the limb keeps its directions and the hand or foot its world rotation.

SMOOTH_SIGMA = 4.0  # frames at 60 Hz: the twist is ill-defined while the limb is nearly straight.
CHAINS = [
    (f"{side}{upper}", f"{side}{lower}", f"{side}{end}")
    for side in ("left", "right")
    for upper, lower, end in (("UpperArm", "LowerArm", "Hand"), ("UpperLeg", "LowerLeg", "Foot"))
]


def turn(degrees, axis):
    """A rotation about a bone's local axis as the upper-left of sotaigen.rig.skinning.euler."""
    return euler(tuple(degrees if a == axis else 0.0 for a in "xyz"))[:3, :3]


def palm_frame(side):
    """A hand's rest frame in a source T pose: along the arm, palm down, thumb to the front."""
    sign = 1.0 if side == "left" else -1.0
    y = np.array([sign, 0.0, 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack((np.cross(y, z), y, z))


def world_deltas(pose, bones, rests):
    """Each bone's rotation away from rest, in world axes, for a pose in local Euler angles."""
    delta = {}
    for bone in bones:
        name, parent = bone["name"], bone["parent"]
        inherited = delta[parent] if parent else np.eye(3)
        if name in pose:
            delta[name] = inherited @ rests[name] @ euler(pose[name])[:3, :3] @ rests[name].T
        else:
            delta[name] = inherited
    return delta


def controls_ready(poses, bones, limits, source_rest):
    """A retargeted clip as the delivered controls can take it. See the note above."""
    rests = {bone["name"]: rest_matrix(bone)[:3, :3] for bone in bones}
    deltas = [world_deltas(pose, bones, rests) for pose in poses]
    for side in ("left", "right"):
        hand = f"{side}Hand"
        wanted = rests[hand][:3, 1]
        source = np.array(source_rest["lhand" if side == "left" else "rhand"])
        # Undo the smallest rotation; the source's palm is what the hand's own frame goes onto.
        onto = aligning(wanted, source).T @ palm_frame(side) @ rests[hand].T
        for delta in deltas:
            delta[hand] = delta[hand] @ onto
    out = [dict(pose) for pose in poses]
    for upper, lower, end in CHAINS:
        y_u = rests[upper][:, 1]
        _, y_l, z_l = rests[lower].T
        low, high = limits[lower][0] if lower in limits and limits[lower][0] else (-180.0, 180.0)
        in_upper = np.array([delta[upper].T @ delta[lower] @ y_l for delta in deltas])
        # The twist about the upper limb that brings the lower one's hinge plane onto its direction.
        a, b = y_u @ y_l, y_u @ z_l
        radius, phase = np.hypot(a, b), np.arctan2(b, a)
        vectors = np.zeros((len(deltas), 2))
        for frame, target in enumerate(in_upper):
            spread = np.arccos(np.clip((y_u @ target) / radius, -1, 1))
            old = np.radians(poses[frame].get(lower, (0, 0, 0))[0])
            flex = min(phase + spread, phase - spread, key=lambda v: abs(v - old))
            limb = y_l * np.cos(flex) + z_l * np.sin(flex)
            limb_flat, target_flat = limb - y_u * (y_u @ limb), target - y_u * (y_u @ target)
            angle = np.arctan2(y_u @ np.cross(limb_flat, target_flat), limb_flat @ target_flat)
            weight = (np.linalg.norm(limb_flat) * np.linalg.norm(target_flat)) ** 2
            vectors[frame] = weight * np.cos(angle), weight * np.sin(angle)
        smooth = gaussian_filter1d(vectors, SMOOTH_SIGMA, axis=0, mode="nearest")
        twists = np.degrees(np.arctan2(smooth[:, 1], smooth[:, 0]))
        for frame, delta in enumerate(deltas):
            twist = twists[frame]
            parent_delta = delta[upper] @ rests[upper] @ turn(twist, "y") @ rests[upper].T
            parent = next(bone["parent"] for bone in bones if bone["name"] == upper)
            inherited = delta[parent] if parent else np.eye(3)
            base = rests[upper].T @ inherited.T @ parent_delta @ rests[upper]
            out[frame][upper] = euler_angles(base)
            # In the frame of the twisted upper limb, what's left of the limb's direction is a hinge.
            target = parent_delta.T @ delta[upper] @ in_upper[frame]
            flex = float(np.degrees(np.arctan2(z_l @ target, y_l @ target)))
            flex = min(max(flex, low), high)
            out[frame][lower] = (round(flex, 3), 0.0, 0.0)
            lower_delta = parent_delta @ rests[lower] @ turn(flex, "x") @ rests[lower].T
            out[frame][end] = euler_angles(rests[end].T @ lower_delta.T @ delta[end] @ rests[end])
    # CMU's hands carry a constant bias of the markers: walking, the same subject's symmetric wrists
    # read -49 and -24 degrees. Each wrist's own median over the clip is taken as its neutral; what the
    # gesture does around it stays. This is for this data, not for a source with real wrists.
    for side in ("left", "right"):
        hand = f"{side}Hand"
        angles = np.array([pose[hand] for pose in out])
        neutral = np.median(angles, axis=0)
        neutral[1] = 0.0
        for frame, pose in enumerate(out):
            pose[hand] = tuple(round(float(v), 3) for v in angles[frame] - neutral)
    return out


def plain(pose):
    return {name: [float(v) for v in value] for name, value in pose.items()}


def request_file(path, rig, fps, poses, **extra):
    request = {
        "fps": fps,
        "central": driver_table(rig.bones)["central"],
        "poses": [plain(pose) for pose in poses],
        **extra,
    }
    file = path / "request.json"
    file.write_text(json.dumps(request))
    return file


def clip_limits(poses, limits):
    """Per control and axis, how often the limits hold the clip back and by how much at most."""
    found = {}
    for pose in poses:
        held = clipped(pose, limits)
        for name, value in pose.items():
            if name == "rootTranslation":
                continue
            for axis, wanted, got in zip("xyz", value, held[name], strict=True):
                if abs(wanted - got) > 1e-6:
                    row = found.setdefault(f"{name}.{axis}", {"frames": 0, "worst": 0.0})
                    row["frames"] += 1
                    row["worst"] = max(row["worst"], round(abs(wanted - got), 1))
    return {
        key: {**row, "share": round(row["frames"] / len(poses), 3)}
        for key, row in sorted(found.items(), key=lambda item: -item[1]["frames"] * item[1]["worst"])
    }


def euler_jumps(poses):
    """Frame-to-frame turns of a keyed Euler channel that are a branch jump, not motion."""
    found = {}
    for before, after in itertools.pairwise(poses):
        for name, value in after.items():
            if name == "rootTranslation" or name not in before:
                continue
            for axis, a, b in zip("xyz", before[name], value, strict=True):
                if abs(b - a) > 60:
                    row = found.setdefault(f"{name}.{axis}", {"count": 0, "worst": 0.0})
                    row["count"] += 1
                    row["worst"] = max(row["worst"], round(abs(b - a), 1))
    return found


def range_used(poses):
    """The largest arm and leg angles the clip asks, as the correctives read them, and the share of
    frames an arm is past what the mesh is clean to."""
    rows = {"forward": [], "side": [], "twist": [], "elbow": [], "hipFlexion": [], "knee": []}
    past = 0
    for pose in poses:
        over = False
        for side, sign in (("left", 1), ("right", -1)):
            x, t, z = swing_twist(pose.get(f"{side}UpperArm", (0, 0, 0)))
            rows["forward"].append(x)
            rows["side"].append(z * sign)
            rows["twist"].append(abs(t))
            over = over or x > CLEAN["forward"] or z * sign > CLEAN["side"]
            rows["elbow"].append(pose.get(f"{side}LowerArm", (0, 0, 0))[0])
            rows["hipFlexion"].append(swing_twist(pose.get(f"{side}UpperLeg", (0, 0, 0)))[0])
            rows["knee"].append(pose.get(f"{side}LowerLeg", (0, 0, 0))[0])
        past += over
    return {
        **{key: round(float(max(values)), 1) for key, values in rows.items()},
        "armPastCleanShare": round(past / len(poses), 3),
    }


def distances(a, b):
    return np.linalg.norm(a - b, axis=1)


def key_and_read(blend, rig, fps, poses, sample, path):
    """Key the poses on a copy of the body's file, reopen that and read it back."""
    request = request_file(path, rig, fps, poses, sample=sample)
    run_blender(ANIMATE, "key", request, path / "motion.blend", blend=blend)
    run_blender(ANIMATE, "read", request, path / "read.npz", blend=path / "motion.blend")
    return json.loads((path / "read.json").read_text()), np.load(path / "read.npz")


def cmd_run(argv):
    parser = argparse.ArgumentParser(prog="sotaigen-dev motion-blend run")
    parser.add_argument("presets", nargs="+")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--raw", action="store_true", help="as sotaigen_dev/motion/clips.py retargets, unfitted")
    args = parser.parse_args(argv)
    for preset in args.presets:
        # Always written anew: body.blend isn't part of the build, so nothing else notices it is stale.
        blend_body(preset)
        rig = rigged(preset)
        limits = joint_limits(rig.bones)
        correctives = Correctives.production(
            rig.base, rig.verts, rig.bones, rig.weights, field=surface(preset)[0]
        )
        for name in args.clip.split(","):
            path = folder(preset, f"{name}-raw" if args.raw else name)
            clip = load_clip(name)
            fps, poses = playback_samples(clip, rig)
            capture = clip.get("capture", clip)
            contacts = capture.get("footContacts")
            feet = {"retargeted": contact_errors(rig, poses, contacts)} if contacts else {}
            if not args.raw:
                poses = controls_ready(poses, rig.bones, limits, capture["rest"])
                if contacts:
                    feet["fitted"] = contact_errors(rig, poses, contacts)
                    feet["fittedHeld"] = contact_errors(rig, [clipped(pose, limits) for pose in poses], contacts)
            sample = list(range(0, len(poses), SAMPLE_EVERY))
            info, data = key_and_read(blend_path(preset), rig, fps, poses, sample, path)
            rows = []
            for frame in sample:
                pose = poses[frame]
                free = pose_verts(rig.bones, rig.weights, correctives.bind(rig.verts, pose), pose)
                held_pose = clipped(pose, limits)
                held = pose_verts(rig.bones, rig.weights, correctives.bind(rig.verts, held_pose), held_pose)
                off, on = data[f"off:{frame}:skin"], data[f"on:{frame}:skin"]
                rows.append(
                    {
                        "frame": frame,
                        "fileVsRuntimeOff": float(distances(off, free).max()),
                        "fileVsRuntimeOn": float(distances(on, held).max()),
                        "limitsMove": float(distances(on, off).max()),
                    }
                )
            worst_on = max(rows, key=lambda row: row["limitsMove"])
            report = {
                "preset": preset,
                "clip": name,
                "blender": info,
                "frames": len(poses),
                "fileVsRuntimeOffMax": round(max(row["fileVsRuntimeOff"] for row in rows), 4),
                "fileVsRuntimeOnMax": round(max(row["fileVsRuntimeOn"] for row in rows), 4),
                "limitsMoveMax": round(worst_on["limitsMove"], 1),
                "limitsMoveWorstFrame10Hz": worst_on["frame"] // SAMPLE_EVERY,
                "limitsMoveMedian": round(float(np.median([row["limitsMove"] for row in rows])), 1),
                "feet": feet,
                "range": range_used(poses),
                "limitsHold": clip_limits(poses, limits),
                "eulerJumps": euler_jumps(poses),
                "perFrame": rows,
            }
            (path / "report.json").write_text(json.dumps(report, indent=1))
            print(json.dumps({k: v for k, v in report.items() if k not in ("perFrame", "blender")}))
    return 0


def cmd_sheet(argv):
    from sotaigen.mesh import write_obj

    parser = argparse.ArgumentParser(prog="sotaigen-dev motion-blend sheet")
    parser.add_argument("presets", nargs="+")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--frames", required=True, help="at 10 Hz, as the docs number them")
    parser.add_argument("--limits", default="on,off")
    parser.add_argument("--focus")
    parser.add_argument("--view", default="front,threequarter")
    parser.add_argument("--look", default="toon")
    parser.add_argument("--mm-per-px", type=float)
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args(argv)
    frames = [int(frame) for frame in args.frames.split(",")]
    root = OUT / "poses"
    names = []
    for preset in args.presets:
        faces = rigged(preset).base["faces"]
        data = np.load(OUT / preset / args.clip / "read.npz")
        poses = root / preset / "poses"
        poses.mkdir(parents=True, exist_ok=True)
        names = []
        for frame in frames:
            for limit in args.limits.split(","):
                name = f"{args.clip}{frame:03d}{limit}"
                skin = data[f"{limit}:{frame * SAMPLE_EVERY}:skin"].astype(float)
                # A travelling clip walks into the review camera's near plane, which cuts the body.
                centre = (skin.min(axis=0) + skin.max(axis=0)) / 2
                skin[:, :2] -= centre[:2]
                write_obj(poses / f"{name}.obj", skin, faces)
                names.append(name)
        (root / preset / "fingerprint.json").write_text(json.dumps({"pose": fingerprints()["pose"]}))
    review = review_module()
    review.BUILD = root
    review.POSES = {name: {} for name in names}
    axes = review.select_axes(args.presets, pose=",".join(names), view=args.view, focus=args.focus, look=args.look)
    review.make_sheet(axes, mm_per_px=args.mm_per_px, out=args.out)
    print(args.out)
    return 0


def cmd_film(argv):
    parser = argparse.ArgumentParser(prog="sotaigen-dev motion-blend film")
    parser.add_argument("preset", metavar="name")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--azimuth", type=float, default=30)
    parser.add_argument("--limits", default="off", choices=["on", "off"], help="off is as the file is saved")
    args = parser.parse_args(argv)
    rig = rigged(args.preset)
    path = folder(args.preset, args.clip)
    fps, poses = playback_samples(load_clip(args.clip), rig)
    height = float(rig.verts[:, 2].max() - rig.verts[:, 2].min()) / 1000
    request = request_file(path, rig, fps, [], film={"height": height, "azimuth": args.azimuth, "step": 2, "limits": args.limits == "on"})
    out = path / f"motion-limits-{args.limits}.mp4"
    run_blender(ANIMATE, "film", request, out, blend=path / "motion.blend")
    print(out, f"{len(poses)} frames at {fps} Hz")
    return 0


COMMANDS = {"run": cmd_run, "sheet": cmd_sheet, "film": cmd_film}


def main():
    argv = sys.argv[1:]
    if argv in (["--help"], ["-h"]):
        print(__doc__)
        return 0
    if not argv or argv[0] not in COMMANDS:
        print(__doc__, file=sys.stderr)
        return 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
