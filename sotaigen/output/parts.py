"""What a GLB or .blend holds besides the skin: separate meshes on the same skeleton, and how each is drawn.

The head's placeholder parts (sotaigen.output.face) are made of these, and a downstream project can
add its own; the contract is docs/systems/export.md's.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

ALPHA_MODES = ("OPAQUE", "MASK", "BLEND")
# A part's name becomes a node's name, which three.js would otherwise quietly rewrite.
PART_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


@dataclass(frozen=True)
class Material:
    """How one part is drawn. The fields mean what they mean in glTF's material.

    base_color   linear RGBA, each in 0..1. Not sRGB.
    alpha        "OPAQUE", "MASK" (cut at alpha_cutoff) or "BLEND" (see-through, drawn after every
                 opaque part).
    render_order Parts of one kind (opaque or see-through) are drawn lowest first. Equal ones are
                 left to the runtime, by depth for see-through parts.
    double_sided Draw back faces too, for a card or strip seen from both sides.
    outline      Whether a cel-shaded runtime draws an outline around it, as around the skin. For
                 hair and ears, not for the thin shells of the features.
    """

    base_color: tuple[float, float, float, float]
    alpha: str = "OPAQUE"
    alpha_cutoff: float = 0.5
    render_order: int = 0
    double_sided: bool = False
    outline: bool = False

    def __post_init__(self):
        if len(self.base_color) != 4 or not all(0.0 <= c <= 1.0 for c in self.base_color):
            raise ValueError(f"base_color must be linear RGBA in 0..1, not {self.base_color}")
        if self.alpha not in ALPHA_MODES:
            raise ValueError(f"alpha must be one of {ALPHA_MODES}, not {self.alpha!r}")
        if not 0.0 <= self.alpha_cutoff <= 1.0:
            raise ValueError(f"alpha_cutoff must be in 0..1, not {self.alpha_cutoff}")
        if self.render_order != int(self.render_order):
            raise ValueError(f"render_order must be a whole number, not {self.render_order}")


@dataclass(frozen=True, eq=False)
class Part:
    """A separate mesh that shares the body's skeleton: one node, one material. See docs/systems/export.md.

    name      Letters, digits, _ and -, starting with a letter. Unique in the file, and neither
              "body" nor a bone's name.
    verts     V x 3, in mm, in the same frame as the body: Z up, front -Y, the wearer's left +X.
    faces     Lists of corners, counter-clockwise seen from outside. A quad or longer face is a fan
              from its first corner, so a body's shape never changes the triangle list.
    material  A Material.
    weights   None binds every vertex to the head bone with weight 1. Otherwise V x B with the
              columns of rig.bones, as rig.weights: at most four influences, each row summing to 1.
    uvs       Optional V x 2, in glTF's convention with V down. A part has no seams of its own: copy
              the vertex where you want one.
    normals   Optional V x 3, any non-zero length. Without them, area-weighted from the faces.
    morphs    Name to V x 3 offsets from verts, in mm. Each part stores only what its targets move.
    """

    name: str
    verts: np.ndarray
    faces: list
    material: Material
    weights: np.ndarray | None = None
    uvs: np.ndarray | None = None
    normals: np.ndarray | None = None
    morphs: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        if not PART_NAME.fullmatch(self.name):
            raise ValueError(f"a part's name must match {PART_NAME.pattern}, not {self.name!r}")
        verts = rows_of(self.name, "verts", self.verts, 3)
        count = len(verts)
        if count == 0:
            raise ValueError(f"part {self.name}: no vertices")
        if len(self.faces) == 0:
            raise ValueError(f"part {self.name}: no faces")
        for face in self.faces:
            if len(face) < 3 or min(face) < 0 or max(face) >= count:
                raise ValueError(f"part {self.name}: bad face {list(face)} for {count} vertices")
        values = {"verts": verts}
        if self.weights is not None:
            weights = np.asarray(self.weights, float)
            if weights.ndim != 2 or len(weights) != count:
                raise ValueError(f"part {self.name}: weights must be {count} x bones")
            if (weights < 0).any() or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError(f"part {self.name}: each row of weights must sum to 1")
            if (np.count_nonzero(weights, axis=1) > 4).any():
                raise ValueError(f"part {self.name}: a vertex has more than four influences")
            values["weights"] = weights
        if self.uvs is not None:
            values["uvs"] = rows_of(self.name, "uvs", self.uvs, 2, count)
        if self.normals is not None:
            values["normals"] = rows_of(self.name, "normals", self.normals, 3, count)
            if (np.linalg.norm(values["normals"], axis=1) == 0).any():
                raise ValueError(f"part {self.name}: a normal has no length")
        morphs = {}
        for target, offsets in self.morphs.items():
            if not isinstance(target, str) or not target:
                raise ValueError(f"part {self.name}: a morph target needs a name, not {target!r}")
            morphs[target] = rows_of(self.name, f"morph {target}", offsets, 3, count)
        values["morphs"] = morphs
        for key, value in values.items():
            object.__setattr__(self, key, value)


def rows_of(part, what, values, width, count=None):
    """An array of `width` finite columns, and `count` rows when given, as float."""
    values = np.asarray(values, float)
    if values.ndim != 2 or values.shape[1] != width or (count is not None and len(values) != count):
        wanted = f"{'V' if count is None else count} x {width}"
        raise ValueError(f"part {part}: {what} must be {wanted}, not {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"part {part}: {what} is not finite")
    return values
