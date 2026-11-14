"""Stand-in parts for the paths through the contract that the head's real parts don't take.

    uv run scripts/standin_parts.py [body ...]

tests/test_export.py reads this data back. The command writes build/reports/runtime-check/standin/<body>.glb,
which the page offers as "<body> + stand-in parts"; scripts/runtime_check.py checks the real parts
instead. They are plain patches, there to take every path through the contract, and not a design for
anything: an opaque part with targets, a see-through one with a draw order, a part with copied weights
that reaches into the neck's blend, and one with its own UVs and normals. The skin gets a morph target
and normals that are obviously not its own, so that a screenshot can tell.
"""

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sotaigen.mesh import skin_normals
from sotaigen.output.glb import glb_bytes, split_uv_seams
from sotaigen.output.metadata import file_info
from sotaigen.output.parts import Material, Part
from sotaigen.pipeline import BUILD, bodies, rigged
from sotaigen.uv import export_fields

OUT = BUILD / "reports" / "runtime-check" / "standin"
# Morph weights to check under: none, and every target at a different weight, one of them at 1.
SETTINGS = {
    "zero": {},
    "mix": {"blink": 1.0, "happy": 0.65, "surprised": 0.3, "sway": 0.8, "cheek": 0.7},
}


class StandIn(NamedTuple):
    base: dict
    parts: list
    morphs: dict
    normals: np.ndarray


def grid(xs, zs, flip=False):
    """Points on an x by z grid and quads counter-clockwise from the front, or the back if flipped."""
    side = len(xs)
    x, z = np.meshgrid(xs, zs)
    faces = []
    for row in range(len(zs) - 1):
        for col in range(side - 1):
            quad = [
                row * side + col,
                row * side + col + 1,
                (row + 1) * side + col + 1,
                (row + 1) * side + col,
            ]
            faces.append(quad[::-1] if flip else quad)
    return x.ravel(), z.ravel(), faces


def onto(skin, x, z, clear, direction):
    """Points at (x, z) just outside the skin, seen from -Y (direction -1) or +Y (direction 1)."""
    facing = skin[skin[:, 1] * direction > 0]
    gaps = np.hypot(x[:, None] - facing[:, 0], z[:, None] - facing[:, 2])
    nearest = np.argsort(gaps, axis=1)[:, :6]
    closeness = 1.0 / (np.take_along_axis(gaps, nearest, axis=1) + 0.5)
    y = (facing[nearest, 1] * closeness).sum(axis=1) / closeness.sum(axis=1)
    return np.column_stack((x, y + direction * clear, z))


def stand_in(name):
    """The parts, skin morph targets and skin normals for a body's rig."""
    rig = rigged(name)
    names = [bone["name"] for bone in rig.bones]
    head_weight = np.asarray(rig.weights)[:, names.index("head")]
    pure = rig.verts[head_weight >= 0.999]
    top = pure[:, 2].max()
    # The skin is pure head bone down to about 0.85 of the head's height.
    height = (top - pure[:, 2].min()) / 0.85

    def at(t):
        return top - t * height

    # Two patches, with the upper half of each closing on the middle, and the middle looking aside.
    chunks = []
    for centre in (-0.175, 0.175):
        x, z, faces = grid(
            np.linspace(centre - 0.07, centre + 0.07, 13) * height,
            np.linspace(at(0.645), at(0.565), 9),
        )
        chunks.append((x, z, faces))
    x = np.concatenate((chunks[0][0], chunks[1][0]))
    z = np.concatenate((chunks[0][1], chunks[1][1]))
    faces = chunks[0][2] + [[i + len(chunks[0][0]) for i in face] for face in chunks[1][2]]
    eyes = onto(pure, x, z, 3.0, -1)
    middle = at(0.605)
    blink = np.zeros_like(eyes)
    blink[:, 2] = np.where(z > middle, middle - z, 0.0) * 0.9
    happy = np.zeros_like(eyes)
    happy[:, 2] = np.where(z < middle, (middle - z) * 0.7, 0.0)
    look = np.zeros_like(eyes)
    look[np.abs(np.abs(x) - 0.175 * height) < 0.03 * height, 0] = 0.02 * height

    x, z, brow_faces = grid(np.linspace(-0.26, 0.26, 27) * height, np.linspace(at(0.51), at(0.47), 3))
    brows = onto(pure, x, z, 3.0, -1)
    raised = np.zeros_like(brows)
    raised[:, 2] = 0.03 * height

    x, z, badge_faces = grid(np.linspace(-0.05, 0.05, 5) * height, np.linspace(at(0.33), at(0.25), 5))
    badge = onto(pure, x, z, 3.0, -1)
    badge_uvs = np.column_stack(((x - x.min()) / np.ptp(x), (z.max() - z) / np.ptp(z)))

    # From high on the back of the head down into the neck, where the skin stops being pure head.
    upper = rig.verts[rig.verts[:, 2] > at(1.25)]
    x, z, hair_faces = grid(
        np.linspace(-0.2, 0.2, 9) * height, np.linspace(at(1.12), at(0.1), 31), flip=True
    )
    hair = onto(upper, x, z, 4.0, 1)
    upper_index = np.flatnonzero(rig.verts[:, 2] > at(1.25))
    gaps = np.linalg.norm(hair[:, None] - rig.verts[upper_index], axis=2)
    copied = np.asarray(rig.weights)[upper_index[gaps.argmin(axis=1)]]
    sway = np.zeros_like(hair)
    below = np.clip((hair[:, 2] - at(0.5)) / (at(1.12) - at(0.5)), 0.0, 1.0)
    sway[:, 0] = 0.05 * height * below**2

    parts = [
        Part(
            "standInEyes",
            eyes,
            faces,
            Material((0.04, 0.05, 0.09, 1.0)),
            morphs={"blink": blink, "happy": happy, "surprised": look},
        ),
        Part(
            "standInBrows",
            brows,
            brow_faces,
            Material((0.25, 0.08, 0.05, 0.7), "BLEND", render_order=2),
            morphs={"happy": raised},
        ),
        Part(
            "standInBadge",
            badge,
            badge_faces,
            Material((0.75, 0.55, 0.1, 1.0)),
            uvs=badge_uvs,
            normals=np.tile((0.0, -1.0, 0.0), (len(badge), 1)),
        ),
        Part(
            "standInHair",
            hair,
            hair_faces,
            Material((0.12, 0.2, 0.45, 1.0), "MASK", double_sided=True, outline=True),
            weights=copied,
            morphs={"sway": sway},
        ),
    ]

    # The skin: a bulge on each cheek, with a few seam copies inside so that the check covers them.
    base = {**rig.base, **export_fields(rig.base["faces"])}
    source, _, _ = split_uv_seams(base["faces"], len(rig.verts), base["uvs"], base["faceUvs"])
    seam = source[len(rig.verts) :]
    seam = seam[head_weight[seam] >= 0.999][:12]
    cheek = np.zeros_like(rig.verts)
    for side in (-1.0, 1.0):
        wanted = np.array((side * 0.3 * height, -0.3 * height, at(0.72)))
        # On the skin itself, whose width changes between bodies, so that the bulge reaches its peak.
        centre = pure[np.linalg.norm(pure - wanted, axis=1).argmin()]
        gaps = np.linalg.norm(rig.verts - centre, axis=1)
        strength = np.exp(-((gaps / (0.1 * height)) ** 2))
        strength[strength < 0.02] = 0.0
        cheek[:, 1] -= 0.1 * height * strength
    cheek[seam, 1] -= 0.01 * height

    # Normals of a ball around the head, fading into the skin's own as the head bone's share drops.
    ball = rig.verts - np.array((0.0, 0.0, at(0.5)))
    ball /= np.linalg.norm(ball, axis=1, keepdims=True)
    normals = skin_normals(rig.base, rig.verts)
    normals = normals + head_weight[:, None] ** 2 * (ball - normals)
    return StandIn(base, parts, {"cheek": cheek}, normals)


def standin_bytes(name, data):
    rig = rigged(name)
    return glb_bytes(
        data.base,
        rig.verts,
        rig.bones,
        rig.weights,
        {**file_info(rig.spec), "standIn": True},
        morphs=data.morphs,
        parts=data.parts,
        normals=data.normals,
    )


def write_standin(name):
    """Write a body's stand-in GLB, returning its path and the data."""
    data = stand_in(name)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.glb"
    path.write_bytes(standin_bytes(name, data))
    return path, data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", metavar="name", help="default: all of them")
    args = parser.parse_args(argv)
    for name in args.names or bodies():
        path, _ = write_standin(name)
        print(f"{path}: {path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
