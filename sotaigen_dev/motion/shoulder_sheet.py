"""Observe the shoulder/chest card in the thorax frame, with the trunk kept visible.

Unlike an elbow, this region is not a single upright hinge. Use exported triangles,
current axilla corrections and the same material points through every pose. This
writes observations, not a score or a new deformation algorithm.

The correctives are the default sculpted ones, in seconds and without any cache. `contact=True`
(`--contact`) reads the IPC contact table instead, which bakes a body that has none, for minutes.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sotaigen.body import Body
from sotaigen.corrective import axilla
from sotaigen.corrective.deltoid import gather
from sotaigen.corrective.pectoral import follow
from sotaigen.form.axilla import axillary_vertices
from sotaigen.mesh import triangulate, vertex_normals, write_obj
from sotaigen.pipeline import BUILD, fingerprints, rigged, surface
from sotaigen.rig.skinning import Skeleton, pose_skins, pose_verts
from sotaigen_dev.motion.cards import load_card, plane_section, sweep_poses
from sotaigen_dev.pictures import font


def moved(matrix, points):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def sections(path, frames, triangles, base, verts, bones, torso, centre):
    """One fixed thorax sagittal plane; rigid references are labelled, not targets."""
    segments = np.asarray(base["segments"])
    names = [b["name"] for b in bones]
    chest, arm = names.index("upperChest"), names.index("leftUpperArm")
    left = np.asarray(base["verts"])[:, 0] > 0
    x = float(verts[axillary_vertices(base)[0], 0])
    reference = []
    for name, bone in [("trunk", chest), ("upperArm", arm)]:
        mask = (segments == name) & left
        reference.append((bone, triangles[mask[triangles].all(1)]))
    width = 320
    canvas = Image.new("RGB", (len(frames) * width, 392), "#edf2f5")
    draw = ImageDraw.Draw(canvas)
    for column, (name, frame) in enumerate(frames.items()):
        cell = Image.new("RGB", (width, width), "#edf2f5")
        pen = ImageDraw.Draw(cell)
        lines = [
            (plane_section(moved(frame["skins"][bone], verts), tris, x), color)
            for (bone, tris), color in zip(reference, ["#70a3d5", "#d68a40"], strict=True)
        ]
        lines.append((plane_section(frame["posed"], triangles, x), "#253b48"))
        origin = np.array([centre[1], centre[2] - .12 * torso])
        for line, color in lines:
            pixels = (line - origin) / (.75 * torso) * width
            pixels[:, :, 1] *= -1
            pixels += width / 2
            for pair in pixels:
                pen.line([tuple(p) for p in pair], fill=color, width=2)
        canvas.paste(cell, (column * width, 35))
        draw.text((column * width + 10, 8), name, font=font(18), fill="#253b48")
    draw.text((12, 361), f"固定矢状截面 X={x:.1f} mm；黑：实际皮肤  蓝：胸廓刚性参照  橙：上臂刚性参照",
              font=font(17), fill="#253b48")
    canvas.save(path)


def behaviour(path, frames, base, verts, torso, crotch):
    """Chest skin relative to the thorax, not relative to the rotating upper arm."""
    from sotaigen_dev.motion.cards import ANCHOR
    from sotaigen_dev.motion.sheet import draw_heat, lift_colour, slide_colour

    triangles = triangulate(base["faces"], np.asarray(base["verts"]))
    normals = vertex_normals(verts, triangles)
    trunk = np.asarray(base["segments"]) == "trunk"
    shown = np.isin(base["regions"], ("trunk", "arm", "hand"))
    triangles = triangles[shown[triangles].all(1)]
    centre = np.array([0., 0., crotch + .78 * torso])
    cell = 300
    canvas = Image.new("RGB", (cell * len(frames), 674), "#edf2f5")
    pen = ImageDraw.Draw(canvas)
    for column, (name, frame) in enumerate(frames.items()):
        movement = frame["relative"] - verts
        lift = np.sum(movement * normals, axis=1)
        slide = np.linalg.norm(movement - lift[:, None] * normals, axis=1)
        lift[~trunk] = 0
        slide[~trunk] = 0
        points = frame["posed"] - centre + ANCHOR
        for row, (values, color) in enumerate([(lift, lift_colour), (slide, slide_colour)]):
            draw_heat(canvas, (cell * column, 32 + row * cell, cell), .5 * torso,
                      points, triangles, values, color, "front")
        pen.text((cell * column + 10, 8), name, font=font(17), fill="#253b48")
    pen.text((10, 639), "上：相对胸廓的法向位移（红外移／蓝内移）；下：切向位移（紫）。臂部不着色。",
             font=font(15), fill="#253b48")
    pen.text((10, 657), "色标沿用肘卡：法向 12 mm、切向 25 mm；只是位移描述，不是肌肉体积、合格线或真实组织标定。",
             font=font(15), fill="#253b48")
    canvas.save(path)


def shoulder_targets(preset, base, verts, bones, weights, *, contact=False, frozen=None):
    """The shoulder's targets, as Correctives.production makes them; a frozen mesh has no field."""
    if contact:
        # Imported here so that the default can't reach the contact solver or its cache.
        from sotaigen.contact.axilla_cache import targets as contact_targets

        return contact_targets(base, verts, bones, weights)
    targets = axilla.targets(base, verts, bones, weights)
    if frozen:
        return targets
    return gather(follow(targets, base, verts, bones, surface(preset)[0]), base, verts, bones)


def run(preset, *, render=False, look="clay,toon", against=None, frozen=None, contact=False):
    card = load_card("shoulder-chest")
    if frozen:
        from sotaigen.generation import full_template
        data = np.load(frozen / "mesh.npz")
        rows = json.loads((frozen / "bones.json").read_text())
        # A frozen rig keeps the jaw's glide on its bone, as skeleton.json once did.
        glides = {b["name"]: b.pop("glide") for b in rows if "glide" in b}
        bones = Skeleton([b | {k: np.asarray(b[k]) for k in ("head", "tail", "align")} for b in rows],
                         T=float(data["T"]), glides=glides)
        rig = Body(data["v"], bones, data["weights"], {}, None, full_template(),
                   {"T": float(data["T"]), "crotch": float(data["crotch"])})
    else:
        rig = rigged(preset)
    root_folder = BUILD / "reports" / "motion" / (card["region"] + ("-frozen" if frozen else ""))
    base, verts, bones, weights, torso = rig.base, rig.verts, rig.bones, rig.weights, rig.L["T"]
    names = [b["name"] for b in bones]
    chest, arm = names.index("upperChest"), names.index("leftUpperArm")
    centre = np.asarray(bones[arm]["head"])
    triangles = triangulate(base["faces"], np.asarray(base["verts"]))
    targets = shoulder_targets(preset, base, verts, bones, weights, contact=contact, frozen=frozen)
    folder = root_folder / preset
    folder.mkdir(parents=True, exist_ok=True)
    markers = dict(zip(["anteriorFold", "axillaryFloor", "posteriorFold"],
                       axillary_vertices(base), strict=True))
    # Geometric samples of the rest anterior chest, not clinical anatomical landmarks.
    front = ((np.asarray(base["segments"]) == "trunk") & (verts[:, 0] > .04 * torso)
             & (verts[:, 0] < .28 * torso) & (verts[:, 1] < 0)
             & (verts[:, 2] > rig.L["crotch"] + .56 * torso)
             & (verts[:, 2] < rig.L["crotch"] + .83 * torso))
    ids = np.flatnonzero(front)
    apex = ids[np.argmin(verts[ids, 1])]
    markers["anteriorChestSample"] = int(apex)
    aim = verts[apex] + np.array([.06, .015, .07]) * torso
    markers["upperOuterChestSample"] = int(ids[np.argmin(np.linalg.norm(verts[ids] - aim, axis=1))])
    report = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "preset": preset, "card": card["region"], "fingerprint": fingerprints(),
              "source": str(frozen) if frozen else "current generator",
              "correctives": "ipc" if contact else "sculpted",
              "frozenWeights": bool(frozen),
              "frame": "thorax; world axes unchanged", "markers": markers, "sweeps": {}}
    all_arrays = {"rest": verts}
    for sweep in card["sweeps"]:
        frames, rows = {}, {}
        for name, pose in sweep_poses(sweep).items():
            skins = pose_skins(bones, pose)
            posed = pose_verts(bones, weights, axilla.bind_verts(verts, targets, pose), pose)
            frames[name] = {"posed": posed, "skins": skins,
                            "relative": moved(np.linalg.inv(skins[chest]), posed)}
            all_arrays[name] = posed
            back = moved(np.linalg.inv(skins[chest]), posed)
            rows[name] = {
                "materialPointsRelativeToThoraxMM": {
                    key: (back[index] - verts[index]).tolist() for key, index in markers.items()
                },
                "shoulderCentreWorldMM": moved(skins[arm], centre[None])[0].tolist(),
            }
        report["sweeps"][sweep["name"]] = rows
        sections(folder / f"section-{sweep['name']}.png", frames, triangles, base, verts,
                 bones, torso, centre)
        behaviour(folder / f"behaviour-{sweep['name']}.png", frames, base, verts,
                  torso, rig.L["crotch"])
        if render:
            from sotaigen_dev import review
            root = root_folder / "poses"
            dest = root / preset / "poses"
            dest.mkdir(parents=True, exist_ok=True)
            for name, frame in frames.items():
                write_obj(dest / f"{name}.obj", frame["posed"], base["faces"])
            (dest.parent / "fingerprint.json").write_text(json.dumps({"pose": fingerprints()["pose"]}))
            review.BUILD = root
            review.POSES = {name: {} for name in frames}
            axes = review.select_axes([preset], pose=",".join(frames),
                                      view="front,threequarter,backquarter", focus="shoulder", look=look)
            review.make_sheet(axes, rows=("view",), cols=("pose",),
                              against=against, out=folder / f"sweep-{sweep['name']}.png")
    np.savez_compressed(folder / "poses.npz", **all_arrays)
    np.savez_compressed(folder / "mesh.npz", v=verts, f=triangles, T=torso, crotch=rig.L["crotch"])
    (folder / "report.json").write_text(json.dumps(report, indent=2))
    print(folder, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("presets", nargs="+", default=["female-long"])
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--frozen", type=Path, help="read an explicitly frozen mesh and weights")
    parser.add_argument("--contact", action="store_true", help="the IPC table, baked if missing")
    args = parser.parse_args()
    for preset in args.presets:
        run(preset, render=args.render, frozen=args.frozen, contact=args.contact)
