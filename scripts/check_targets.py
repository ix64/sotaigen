"""Compare what the parameters ask for with what the surface measures."""

import json

import numpy as np

from sotaigen.form.anatomy import calibrated, width_at
from sotaigen.form.bodies import load_bodies
from sotaigen.pipeline import BUILD, ensure_built

for name, p in load_bodies().items():
    ensure_built(name, ("metrics",))
    body, L = calibrated(p)
    T, at = L["T"], L["at"]
    waist = width_at(body, L["waist"])
    hip = max(width_at(body, at(f)) for f in np.linspace(-0.05, 0.25, 13))
    wrap = json.loads((BUILD / "bodies" / name / "wrap.json").read_text())
    poses = json.loads((BUILD / "bodies" / name / "poses.json").read_text())
    print(name, f"T={T:.0f} H={L['H']} heads={p['heads']}")
    print(f"  waist asked {p['waistBreadth'] * T:.0f} got {waist:.0f}")
    print(f"  hip   asked {p['hipBreadth'] * T:.0f} got {hip:.0f}")
    print(
        f"  wrap flipped {wrap['flippedFaces']} of {wrap['faces']}, surface max {wrap['surfaceErrorMm']['max']}"
    )
    print(
        f"  seams hand {wrap['handSeamMissMm']} foot {wrap['footSeamMissMm']} lowest foot {wrap['footLowest']}"
    )
    print(f"  crossed midline {wrap['crossedMidline']}")
    print("  collapsed", {k: v["collapsedFaces"] for k, v in poses.items() if v["collapsedFaces"]})
