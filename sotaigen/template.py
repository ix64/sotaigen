"""The fixed-topology template every body is wrapped from; see template/PROVENANCE.md."""

import json
from functools import cache
from pathlib import Path

import numpy as np

from sotaigen.paths import DATA

TEMPLATE = DATA / "template" / "base-mesh.json"
FORMAT = "sotaigen/template"
VERSION = 1
FINGERS = {
    "thumb": ("Metacarpal", "Proximal", "Distal"),
    **dict.fromkeys(("index", "middle", "ring", "little"), ("Proximal", "Intermediate", "Distal")),
}
# The region of each segment: the coarser parts the wrap and the rig tell apart.
REGION_OF = {
    "trunk": "trunk",
    "neck": "neck",
    "upperArm": "arm",
    "lowerArm": "arm",
    "hand": "hand",
    **{f"{finger}{bone}": "hand" for finger, bones in FINGERS.items() for bone in bones},
    "upperLeg": "leg",
    "lowerLeg": "leg",
    "foot": "foot",
    "toes": "foot",
    "head": "head",
}


def load_template(path=TEMPLATE):
    """The template as plain JSON values, with each vertex's region read from its segment."""
    base = json.loads(Path(path).read_text())
    if (base.get("format"), base.get("version")) != (FORMAT, VERSION):
        raise ValueError(f"{path} is not a {FORMAT} file of version {VERSION}")
    base["regions"] = [REGION_OF[segment] for segment in base["segments"]]
    return base


@cache
def template_landmarks():
    """base-mesh.json's `landmarks`, read once and shared: don't change them."""
    return load_template()["landmarks"]


def coarse(base):
    """The base without the loops `refined` added, and which of the base's vertices it keeps.

    Works on the template or on it with the head in (sotaigen.form.head.extended). Added loops are always
    last, so what stays is a prefix. A base with no loops is its own.
    """
    refined = base.get("refined")
    count = len(base["verts"])
    if not refined:
        return base, np.arange(count)
    first = refined["from"]
    keep = np.arange(first)
    start = refined.get("head", (first, 0))[0]
    head = [face for face in base["faces"] if start <= max(face) < first]
    out = dict(base)
    del out["refined"]
    out["verts"] = np.asarray(base["verts"], float)[keep]
    out["faces"] = refined["faces"] + head
    out["regions"] = list(base["regions"][:first])
    out["segments"] = list(base["segments"][:first])
    return out, keep


def prolonged(base, values, keep):
    """Values on coarse(base)'s vertices carried to all of base's: an added vertex takes the mean of
    the two it halves, so a field keeps its spread in millimetres however many loops are added."""
    out = np.zeros((len(base["verts"]),) + values.shape[1:])
    out[keep] = values
    refined = base.get("refined")
    if refined:
        first = refined["from"]
        # In order: a later loop may halve a rung whose ends an earlier loop added.
        for k, (a, b) in enumerate(refined["parents"]):
            out[first + k] = (out[a] + out[b]) / 2
    return out


def before_loops(base):
    """The template as extract.py cut it, before template/loops.py added its batches.

    The wrap fits this one, and the later loops' vertices go onto the surface afterwards
    (body.wrap.place_loops), so adding a batch never moves a vertex the template had already.
    """
    refined = base.get("refined") or {}
    batches = refined.get("batches", [])
    if len(batches) < 2:
        return base
    faces = [list(face) for face in base["faces"]]
    # With the head in (body.head.extended) its faces come after the template's.
    end = refined.get("templateFaces", len(faces))
    for batch in reversed(batches[1:]):
        for index, start in reversed(batch["splits"]):
            a, _, _, d = faces[index]
            _, b, c, _ = faces.pop(end - 1)
            end -= 1
            turned = [a, b, c, d]
            faces[index] = turned[-start:] + turned[:-start] if start else turned
    count = refined["from"] + batches[0]["count"]
    out = dict(base)
    out["verts"] = base["verts"][:count]
    out["faces"] = faces
    out["regions"] = base["regions"][:count]
    out["segments"] = base["segments"][:count]
    out["refined"] = dict(refined, parents=refined["parents"][: batches[0]["count"]], batches=batches[:1])
    if "templateFaces" in refined:
        out["refined"]["templateFaces"] = end
    return out


def onto_loops(base, values):
    """Values on before_loops(base)'s vertices carried onto all of base's: each later loop's vertex
    takes the mean of the two it halves, in the order they were added."""
    refined = base["refined"]
    count = refined["from"] + refined["batches"][0]["count"]
    out = np.zeros((len(base["verts"]),) + values.shape[1:])
    out[:count] = values
    for v in range(count, len(out)):
        a, b = refined["parents"][v - refined["from"]]
        out[v] = (out[a] + out[b]) / 2
    return out
