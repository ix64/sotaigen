"""Print landmarks of the CC0 template mesh."""

import numpy as np

from sotaigen.template import load_template

b = load_template()
v = np.array(b["verts"])
r = np.array(b["regions"])
print("bbox", v.min(0), v.max(0))
for k in ("elbow", "wrist", "knee", "ankle"):
    print(k, v[np.array(b["rings"][k])].mean(0))
print("neck", v[b["openings"]["neck"]].mean(0))
t = v[r == "trunk"]
print(
    "trunk z", t[:, 2].min(), t[:, 2].max(), "x", t[:, 0].max(), "y", t[:, 1].min(), t[:, 1].max()
)
a = v[r == "arm"]
print("arm z", a[:, 2].min(), a[:, 2].max(), "x", np.abs(a[:, 0]).min(), np.abs(a[:, 0]).max())
