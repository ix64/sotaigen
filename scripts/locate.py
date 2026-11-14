"""Where a body's wrap still folds or misses the surface, by template region and height.

Only the projected part of the template is measured: not the hands and feet, which keep the
template's own mesh, and not the head, which has its own field.

uv run scripts/locate.py <body> [body ...]
"""

import sys

import numpy as np

from sotaigen.form.anatomy import calibrated
from sotaigen.form.bodies import load_bodies
from sotaigen.form.head import template_rows
from sotaigen.mesh import face_normals
from sotaigen.pipeline import BUILD, ensure_built
from sotaigen.template import load_template


def main(names):
    base = load_template()
    presets = load_bodies()
    regions = np.array(base["regions"])
    segments = np.array(base["segments"])
    keep = np.isin(regions, ("trunk", "neck", "leg", "arm"))
    index = -np.ones(len(keep), int)
    index[keep] = np.arange(keep.sum())
    faces_full = [f for f in base["faces"] if all(keep[i] for i in f)]
    faces = [[int(index[i]) for i in f] for f in faces_full]
    for name in names:
        ensure_built(name, ("wrap",))
        body, L = calibrated(presets[name])
        body.clips = []
        for group in body.groups:
            group.pop("clips", None)
        full = np.load(BUILD / "bodies" / name / "wrapped.npy")
        points = template_rows(full, base)[keep]
        centres = np.array([points[f].mean(0) for f in faces])
        normals = face_normals(points, faces)
        bad = np.sum(normals * body.gradient(centres, step=1.5), axis=1) < 0
        seg = np.array([segments[f[0]] for f in faces_full])
        d = np.abs(body(points)) / np.maximum(
            np.linalg.norm(body.gradient(points, step=1.5), axis=1), 1e-6
        )
        print(name, "flips by segment", {s: int((bad & (seg == s)).sum()) for s in set(seg[bad])})
        if bad.any():
            print(
                "  flip heights",
                np.round(np.percentile(centres[bad, 2], [0, 50, 100])),
                "x",
                np.round(np.percentile(np.abs(centres[bad, 0]), [0, 50, 100])),
            )
        worst = np.argsort(-d)[:4]
        print(
            "  worst surface distance",
            [(np.round(points[i]).tolist(), round(float(d[i]), 1)) for i in worst],
        )
        print(
            "  landmarks z knee",
            round(L["knee"]),
            "ankle",
            round(L["ankle"]),
            "wrist",
            np.round(L["wrist"]).tolist(),
        )


if __name__ == "__main__":
    main(sys.argv[1:] or load_bodies())
