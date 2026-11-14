"""A made body: its rest skin, skeleton and skin weights, how its wrap went, and what made it.

sotaigen.generation.generate (sotaigen.generate) makes one from parameters in memory, sotaigen.pipeline.rigged reads a typical or test
body's from the build; sotaigen.output.glb.write_glb and sotaigen.output.blend.write_blend write either.
"""

from typing import NamedTuple

import numpy as np


class Body(NamedTuple):
    """verts      the finished rest skin, V x 3 in mm, in the template's order
    bones      the skeleton, sotaigen.rig.skeleton's dicts
    weights    V x B skin weights, the bones' columns
    wrap       the wrap's report (wrap.json): its invariants and seams
    seconds    how long the wrap and the rig took, when made in memory
    base       the template with the head, whose faces and segments the skin has
    L          the landmarks the bones were placed from
    spec       the sotaigen.params.file.Spec it was made from
    controls   the masses' internal controls the solve found for it
    notes      what checking its parameters found (sotaigen.output.quality.body_notes), None until checked
    name       a typical or test body's name when it was read from the build, else None

    The arrays are shared between callers: don't change them.
    """

    verts: np.ndarray
    bones: list
    weights: np.ndarray
    wrap: dict
    seconds: dict | None = None
    base: dict | None = None
    L: dict | None = None
    spec: object = None
    controls: dict | None = None
    notes: tuple | None = None
    name: str | None = None
