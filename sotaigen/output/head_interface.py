"""Where the eyes are, as the head interface says it (extras.sotaigen.head, docs/parts/head.md).

One rule before and after a head swap, so that the two files mean the same by it: from the crown down,
and out from the midline, as shares of the head's height from the crown to the middle of the neck ring.
Both ends are the interface's own (crownVertex, neckRing), so a reader can measure them on any head.
"""

import numpy as np


def eye_place(crown, ring, eye):
    """(eyeLine, eyeX, height): the left eye bone's head `eye` below the `crown` and out from the midline,
    in shares of `height`, the crown's height over the neck `ring`'s middle. All in the same unit."""
    height = float(np.asarray(crown, float)[2] - np.asarray(ring, float)[:, 2].mean())
    eye = np.asarray(eye, float)
    return float((crown[2] - eye[2]) / height), float(abs(eye[0]) / height), height
