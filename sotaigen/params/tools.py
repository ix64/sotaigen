"""Tools that rewrite a body's parameters all at once: the stature, the head count and the torso's length.

The stature and the head count are readings of a body, never inputs (docs/PARAMETERS.md), so changing
one is a tool that writes the parameters that make it. The torso's length is an input, but every range
but a handful is a share of it (sotaigen/params/spec.py), so moving it alone takes the rest out of their
ranges; the tool here carries them along at their shares, as the page's slider does. The page runs the
same arithmetic in web/editor/sliders.ts, and tests/test_editor.py holds the two together.
"""

import json

from sotaigen.params.rules import spec_bounds
from sotaigen.params.spec import CHAIN, JAW_BASE, MILLIMETRES, TORSO_SHARES, heads, stature
from sotaigen.paths import DATA

# How each length and girth follows the head count, as an exponent of its ratio: the table of
# docs/systems/params.md, kept in the page's slider definitions where the head-count slider reads it.
HEAD_COUNT_LAWS = json.loads((DATA / "sliders.json").read_text())["sliders"]["heads"]["laws"]


def _clamped(bounds, key, value):
    low, high = bounds[key]
    return min(max(value, low), high)


def scaled(body, to_stature):
    """The whole body scaled to a stature of `to_stature` mm: every millimetre in proportion, the shape the same."""
    scale = to_stature / stature(body)
    return {key: amount * scale if key in MILLIMETRES else amount for key, amount in body.items()}


def with_head_count(body, count, sex, laws=HEAD_COUNT_LAWS):
    """The same stature in `count` head heights.

    The head, neck and leg follow their laws and the torso takes up the rest; every other millimetre keeps
    its share of the torso, times its law, inside its range on the new lengths.
    """
    ratio = count / heads(body)
    height = stature(body)
    head = height / count
    neck = body["neckLength"] / body["headHeight"] * ratio ** laws["neckLength"] * head
    crotch = body["crotchHeight"] * ratio ** laws["crotchHeight"]
    torso = height - head - neck - crotch
    scale = torso / body["torsoLength"]
    out = body | {"headHeight": head, "neckLength": neck, "torsoLength": torso, "crotchHeight": crotch}
    out["kneeHeight"] = body["kneeHeight"] * crotch / body["crotchHeight"]
    bounds = spec_bounds(out, sex)
    for key in MILLIMETRES:
        if key in body and key not in CHAIN and key != "kneeHeight":
            out[key] = _clamped(bounds, key, body[key] * scale * ratio ** laws.get(key, 0))
    out["headWidthRatio"] = _clamped(bounds, "headWidthRatio", body["headWidthRatio"] * ratio ** laws["headWidthRatio"])
    # The jaw's law is on the part of its ratio past the narrowest jaw the head draws (JAW_BASE).
    jaw = JAW_BASE + (body["jawBreadthRatio"] - JAW_BASE) * ratio ** laws["jawBreadthRatio"]
    out["jawBreadthRatio"] = _clamped(bounds, "jawBreadthRatio", jaw)
    return out


def with_torso_length(body, torso):
    """The torso `torso` mm long, with every parameter measured in its shares keeping its share.

    So the body's shape below the neck and above the crotch scales with it, and nothing leaves its range
    for the torso's sake alone; the head, the neck and the legs stay as they are, as does the stature but
    for the torso's own change.
    """
    ratio = torso / body["torsoLength"]
    out = dict(body)
    for key in TORSO_SHARES:
        if key in body:
            out[key] = body[key] * ratio
    out["torsoLength"] = torso
    return out

