"""The editor's sliders on the Python side: what a slider's value does to a body.

sotaigen/data/sliders.json is the one definition; web/editor/sliders.ts is the same function for the
page, and tests/test_editor.py runs the two against each other.

A slider acts on the parameters it started from, the base. A macro slider's zero is the base, so a
body keeps its exact values until someone moves it, whatever line the macro's gains draw. A slider's
limits come from the base too, so they stay put while the body is being edited.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sotaigen.params.rules import HEADS_RANGE, STATURE_RANGE, rules, spec_bounds
from sotaigen.params.spec import MILLIMETRES, default_of, heads, proportions, stature
from sotaigen.params.tools import scaled, with_head_count, with_torso_length

SLIDERS = json.loads((ROOT / "sotaigen/data/sliders.json").read_text())


def ranges_for(base):
    """Every parameter's limits on this body of its sex, and the two results the tools can aim at."""
    return {
        "body": spec_bounds(base["body"], base["sex"]),
        "derived": {"heads": HEADS_RANGE, "stature": STATURE_RANGE},
    }


def clamped(ranges, kind, key, value):
    low, high = ranges[kind][key]
    return min(max(value, low), high)


def slider_bounds(name, base):
    """A slider's (low, high): its parameter's range on this base, or -1 to 1 for a macro."""
    spec = SLIDERS["sliders"][name]
    if spec["kind"] == "macro":
        return -1.0, 1.0
    kind, key = spec["param"].split(".")
    return ranges_for(base)[kind][key]


def slider_start(name, base, measured=None):
    """A slider's value on untouched parameters."""
    spec = SLIDERS["sliders"][name]
    if spec["kind"] == "macro":
        return 0.0
    kind, key = spec["param"].split(".")
    if kind == "derived":
        return {"heads": heads, "stature": stature}[key](base["body"])
    if key not in base["body"]:
        if spec.get("measured"):
            return (measured or {}).get(spec["measured"])
        return default_of(base["body"], key)
    return base["body"][key]


def apply_sliders(base, values):
    """The {"body"} that `values` {slider: value} make of `base`, {"sex", "body"}, in file order.

    Whatever a moved slider writes stays inside its range, but for the lengths a tool works out, which
    must keep adding up. Direct sliders act on the result of preceding controls.
    """
    ranges = ranges_for(base)
    out = {"body": dict(base["body"])}
    for name, spec in SLIDERS["sliders"].items():
        if name not in values or values[name] == slider_start(name, out):
            continue
        value = values[name]
        if spec["kind"] == "macro":
            for key, gain in spec.get("gains", {}).items():
                out["body"][key] = clamped(ranges, "body", key, out["body"][key] + value * gain)
            for key, gain in spec.get("torsoGains", {}).items():
                moved = out["body"][key] + value * gain * out["body"]["torsoLength"]
                out["body"][key] = clamped(ranges, "body", key, moved)
            for key, gain in spec.get("scales", {}).items():
                out["body"][key] = clamped(ranges, "body", key, out["body"][key] * (1 + value * gain))
        elif spec["kind"] == "covary":
            aim = clamped(ranges, "derived", "heads", value)
            out["body"] = with_head_count(out["body"], aim, base["sex"], spec["laws"])
        elif spec["kind"] == "scale":
            out["body"] = scaled(out["body"], clamped(ranges, "derived", "stature", value))
        else:
            key = spec["param"].split(".")[1]
            aim = clamped(ranges, "body", key, value)
            if spec.get("carriesShares"):
                out["body"] = with_torso_length(out["body"], aim)
            for other in spec.get("carries", ()):
                out["body"][other] = out["body"][other] * aim / out["body"][key]
            out["body"][key] = aim
            # What it needs beside it, where the body has none: a breast drawn from nothing needs its shape.
            for other, amount in spec.get("brings", {}).items():
                if other not in out["body"]:
                    torso = out["body"]["torsoLength"] if other in MILLIMETRES else 1.0
                    out["body"][other] = amount * torso
            # What a rule wants raised with it, just enough, never lowered: the hip with the thigh.
            for other, rule in spec.get("lifts", {}).items():
                room = {name: room for name, _, room in rules(base["sex"])}[rule]
                short = -room(proportions(out["body"]))
                if short > 0:
                    lifted = out["body"][other] + short * out["body"]["torsoLength"]
                    out["body"][other] = clamped(ranges, "body", other, lifted)
    return out
