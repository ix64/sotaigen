"""See a head with its features, ears and hair: clay and toon, every view, expressions, turn and nod.

    uv run scripts/head_review.py [--only heads,bodies,faces,motion]
        [--bodies female-chibi,female-long,female-balanced,female-rounded] [--out build/reports/review/head]
    --only also takes jaw (the opening in steps, clay and toon) and gaze (the eyes each way).

Renders through blender/review/face_render.py in Blender, then lays the images out in sheets. The toon
look shades with the normals sotaigen.output.face gives the export, and every part with its own. Not part of the
pipeline.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sotaigen.mesh import face_normals
from sotaigen.output.blender import find_blender
from sotaigen.output.face import EXPRESSIONS, SKIN_COLOR, face_parts, head_normals
from sotaigen.output.glb import part_weights
from sotaigen.pipeline import rigged
from sotaigen.rig.poses import POSES
from sotaigen.rig.skinning import pose_skins, pose_verts, triangle_fan

PRESETS = ["female-chibi", "female-long", "female-balanced", "female-rounded"]
VIEWS = {
    "front": {"right": (1, 0, 0), "up": (0, 0, 1), "toward": (0, -1, 0)},
    "threequarter": {"right": (0.7071, 0.7071, 0), "up": (0, 0, 1), "toward": (0.7071, -0.7071, 0)},
    "side": {"right": (0, 1, 0), "up": (0, 0, 1), "toward": (1, 0, 0)},
    "back": {"right": (-1, 0, 0), "up": (0, 0, 1), "toward": (0, 1, 0)},
}
HEAD_PX = 480
BODY_PX = (360, 640)
# Clay is grey whatever the part's colour, darker for hair and a dark mark for features.
CLAY = {"nose": 0.78, "ears": 0.78, "hair": 0.42}
CLAY_SHELL = 0.12
SHELLS_LIGHT = {"eyeWhite", "eyeLight"}


def require_blender():
    """Blender as body.blend finds it: BLENDER, then PATH, then the macOS app."""
    blender = find_blender()
    if blender is None:
        sys.exit("Blender was not found: put blender on PATH or set BLENDER=/path/to/blender")
    return blender


def named_pose(pose):
    """A pose given by its name in sotaigen.rig.poses, or the pose itself."""
    return POSES[pose] if isinstance(pose, str) else pose


def scene_arrays(name, pose, weights, line_mm):
    """A scene of one body posed, with these morph target weights, as arrays for Blender."""
    rig = rigged(name)
    pose_table = named_pose(pose)
    parts = face_parts(name)
    skin_verts = pose_verts(rig.bones, rig.weights, rig.verts, pose_table)
    skins = pose_skins(rig.bones, pose_table)
    names = [bone["name"] for bone in rig.bones]

    def turned(normals, bone_weights):
        linear = np.einsum("vb,bij->vij", bone_weights, skins[:, :3, :3])
        return np.einsum("vij,vj->vi", linear, normals)

    arrays = {"line_mm": np.array(line_mm)}
    arrays["verts0"] = skin_verts
    arrays["faces0"] = triangle_fan(rig.base["faces"])
    arrays["normals0"] = turned(head_normals(name), rig.weights)
    arrays["color0"] = np.array(SKIN_COLOR)
    arrays["clay0"] = np.array(0.78)
    arrays["outline0"] = np.array(True)
    index = 1
    for part in parts:
        verts = part.verts.copy()
        for target, weight in weights.items():
            if target in part.morphs:
                verts = verts + part.morphs[target] * weight
        bound = part_weights(part, names)
        arrays[f"verts{index}"] = pose_verts(rig.bones, bound, verts, pose_table)
        arrays[f"faces{index}"] = triangle_fan(part.faces)
        normals = part.normals
        if normals is None:
            normals = np.zeros_like(verts)
            for face, normal in zip(part.faces, face_normals(verts, part.faces), strict=True):
                normals[face] += normal
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
        arrays[f"normals{index}"] = turned(normals, bound)
        arrays[f"color{index}"] = np.array(part.material.base_color)
        shade = CLAY.get(part.name, 0.95 if part.name in SHELLS_LIGHT else CLAY_SHELL)
        arrays[f"clay{index}"] = np.array(shade)
        arrays[f"outline{index}"] = np.array(part.material.outline)
        index += 1
    arrays["parts"] = np.array(index)
    return arrays


class Job:
    def __init__(self, out):
        self.out = out
        self.work = Path(tempfile.mkdtemp())
        self.scenes = []

    def add(self, name, pose, weights, tag, shots, mm_per_px):
        npz = self.work / f"{name}-{tag}.npz"
        np.savez(npz, **scene_arrays(name, pose, weights, 2.5 * mm_per_px))
        self.scenes.append({"npz": str(npz), "shots": shots})

    def run(self):
        manifest = self.work / "manifest.json"
        manifest.write_text(json.dumps({"scenes": self.scenes, "transparent": True}))
        here = Path(__file__).resolve().parent
        subprocess.run(
            [
                str(require_blender()),
                "-b",
                "--factory-startup",
                "--python-exit-code",
                "1",
                "--python",
                str(here.parent / "blender" / "review" / "face_render.py"),
                "--",
                str(manifest),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )


def shot(out, view, look, center, mm_per_px, size):
    return {
        "out": str(out),
        "width": size[0],
        "height": size[1],
        "mm_per_px": mm_per_px,
        "center": list(map(float, center)),
        "look": look,
        **VIEWS[view],
    }


def head_center(name, pose="rest"):
    """The middle of the posed head and hair, for a close view."""
    rig = rigged(name)
    verts = pose_verts(rig.bones, rig.weights, rig.verts, named_pose(pose))
    head = verts[np.asarray(rig.base["segments"]) == "head"]
    h = rig.L["head"]
    return np.array([0.0, (head[:, 1].max() + head[:, 1].min()) / 2, rig.L["H"] - 0.50 * h])


def on_white(path):
    image = Image.open(path).convert("RGBA")
    ground = Image.new("RGBA", image.size, "white")
    ground.alpha_composite(image)
    return ground.convert("RGB")


def sheet(rows, columns, cell, path):
    """Rows of labelled cells: rows is [(label, [png, ...])]."""
    margin, header = 110, 22
    width, height = cell
    image = Image.new(
        "RGB", (margin + width * len(columns), header + height * len(rows)), "white"
    )
    draw = ImageDraw.Draw(image)
    for j, column in enumerate(columns):
        draw.text((margin + j * width + 6, 5), column, fill="black")
    for i, (label, files) in enumerate(rows):
        draw.text((6, header + i * height + height // 2), label, fill="black")
        for j, file in enumerate(files):
            image.paste(on_white(file), (margin + j * width, header + i * height))
    image.save(path)


def head_scale(name):
    return 1.42 * rigged(name).L["head"] / HEAD_PX


def add_heads(job, names, raw, sheets):
    for name in names:
        scale = head_scale(name)
        center = head_center(name)
        shots = [
            shot(raw / f"{name}-head-{look}-{view}.png", view, look, center, scale, (HEAD_PX, HEAD_PX))
            for look in ("toon", "clay")
            for view in VIEWS
        ]
        job.add(name, "rest", {}, "heads", shots, scale)
    for look in ("toon", "clay"):
        sheets.append(
            (
                [(n, [raw / f"{n}-head-{look}-{v}.png" for v in VIEWS]) for n in names],
                list(VIEWS),
                (HEAD_PX, HEAD_PX),
                job.out / f"heads-{look}.png",
            )
        )


def add_bodies(job, names, raw, sheets):
    tall = max(rigged(n).L["H"] for n in names)
    scale = 1.06 * tall / BODY_PX[1]
    for name in names:
        center = (0.0, 0.0, tall / 2 * 1.0)
        shots = [
            shot(raw / f"{name}-body-{look}-{view}.png", view, look, center, scale, BODY_PX)
            for look in ("toon", "clay")
            for view in VIEWS
        ]
        job.add(name, "armsDown", {}, "bodies", shots, scale)
    for look in ("toon", "clay"):
        sheets.append(
            (
                [(v, [raw / f"{n}-body-{look}-{v}.png" for n in names]) for v in VIEWS],
                names,
                BODY_PX,
                job.out / f"bodies-{look}.png",
            )
        )


def add_faces(job, names, raw, sheets):
    size = (HEAD_PX, HEAD_PX)
    for name in names:
        scale = head_scale(name)
        center = head_center(name)
        for expression in EXPRESSIONS:
            shots = [
                shot(raw / f"{name}-{expression}-{view}.png", view, "toon", center, scale, size)
                for view in ("front", "threequarter")
            ]
            job.add(name, "rest", {expression: 1.0}, f"face-{expression}", shots, scale)
    columns = ["neutral", *EXPRESSIONS]
    rows = []
    for name in names:
        neutral = raw / f"{name}-head-toon-front.png"
        rows.append((name, [neutral, *[raw / f"{name}-{e}-front.png" for e in EXPRESSIONS]]))
    sheets.append((rows, columns, size, job.out / "expressions-front.png"))
    rows = []
    for name in names:
        neutral = raw / f"{name}-head-toon-threequarter.png"
        rows.append((name, [neutral, *[raw / f"{name}-{e}-threequarter.png" for e in EXPRESSIONS]]))
    sheets.append((rows, columns, size, job.out / "expressions-threequarter.png"))


def add_motion(job, names, raw, sheets):
    size = (HEAD_PX, HEAD_PX)
    views = ("front", "threequarter", "side", "back")
    for name in names:
        scale = head_scale(name)
        for pose in ("headTurn", "headNod"):
            center = head_center(name, pose)
            shots = [
                shot(raw / f"{name}-{pose}-{look}-{view}.png", view, look, center, scale, size)
                for look in ("toon", "clay")
                for view in views
            ]
            job.add(name, pose, {}, pose, shots, scale)
    for look in ("toon", "clay"):
        rows = []
        for name in names:
            for pose in ("headTurn", "headNod"):
                rows.append(
                    (f"{name} {pose}", [raw / f"{name}-{pose}-{look}-{v}.png" for v in views])
                )
        sheets.append((rows, list(views), size, job.out / f"motion-{look}.png"))


# The opening in steps, then with a turn and a nod; the gaze each way, at the eyes' clean range.
JAW_POSES = {
    **{f"open{angle:02d}": {"jaw": (angle, 0, 0)} for angle in (0, 5, 10, 15, 20, 27)},
    "turn": POSES["jawTurn"],
    "nod": POSES["jawNod"],
}
GAZE_POSES = {
    "gazeLeft": {"leftEye": (0, 0, 12), "rightEye": (0, 0, 12)},
    "gazeRight": {"leftEye": (0, 0, -12), "rightEye": (0, 0, -12)},
    "gazeUp": {"leftEye": (8, 0, 0), "rightEye": (8, 0, 0)},
    "gazeDown": {"leftEye": (-8, 0, 0), "rightEye": (-8, 0, 0)},
}


def add_posed(job, names, raw, sheets, poses, stem, views, looks):
    size = (HEAD_PX, HEAD_PX)
    for name in names:
        scale = head_scale(name)
        for key, pose in poses.items():
            center = head_center(name, pose)
            shots = [
                shot(raw / f"{name}-{stem}-{key}-{look}-{view}.png", view, look, center, scale, size)
                for look in looks
                for view in views
            ]
            job.add(name, pose, {}, f"{stem}-{key}", shots, scale)
    for look in looks:
        for view in views:
            rows = [(name, [raw / f"{name}-{stem}-{key}-{look}-{view}.png" for key in poses]) for name in names]
            sheets.append((rows, list(poses), size, job.out / f"{stem}-{look}-{view}.png"))


def add_jaw(job, names, raw, sheets):
    add_posed(job, names, raw, sheets, JAW_POSES, "jaw", ("front", "threequarter", "side"), ("toon", "clay"))


def add_gaze(job, names, raw, sheets):
    add_posed(job, names, raw, sheets, GAZE_POSES, "gaze", ("front", "threequarter"), ("toon",))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="heads,bodies,faces,motion")
    parser.add_argument("--presets", default=",".join(PRESETS))
    parser.add_argument("--out", default="build/reports/review/head")
    args = parser.parse_args()
    names = args.presets.split(",")
    wanted = args.only.split(",")
    out = Path(args.out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    job = Job(out)
    sheets = []
    adders = {"heads": add_heads, "bodies": add_bodies, "faces": add_faces, "motion": add_motion}
    adders |= {"jaw": add_jaw, "gaze": add_gaze}
    for key in wanted:
        adders[key](job, names, raw, sheets)
    job.run()
    for rows, columns, cell, path in sheets:
        sheet(rows, columns, cell, path)
        print(path)


if __name__ == "__main__":
    main()
