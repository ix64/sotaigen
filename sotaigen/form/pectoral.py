"""The pectoral plate, read through the body's field as a local forward shift of the chest.

A mass blended into the chest comes out as a rounded bump whatever its own shape, because one blend
width rounds both its upper and its lower edge. The chest's side view should instead slope forward from
below the clavicle to the muscle's lower border and step back under it. So the plate is a thickness
profile laid on the chest wall: zero at the clavicle and thinner over the sternum, growing to `pectoralThickness` at the
lower border, then gone within a short drop. The border rises towards the armpit. A stylised shape, not
the muscle's fibres or attachments.

Under a breast (sotaigen/form/breast.py, which lies on the plate) that short drop stood as a ridge across the
breast's upper slope and between the two. The breast lies on the muscle's fascia, so a thicker muscle
carries it forward and its lower border is hidden: there the plate runs on down into the
inframammary fold, its border a long, gentle slope.
"""

import numpy as np

from sotaigen.form.field import Layer, ramp, soft_fold

# In torso shares above the crotch: the lower border beside the sternum and how far it rises by the
# armpit, how far below it the plate is gone, and where it ends under the clavicle.
BORDER, BORDER_RISE, DROP, TOP = 0.77, 0.06, 0.05, 0.95
# Across, in torso shares from the midline: how far out the plate grows from the sternal groove, the
# share of its thickness left at the midline, and its lateral end by the armpit. The groove keeps part
# of the plate so it stays a shallow furrow: one as deep as the plate draws a V beside the sternum.
GROOVE_WIDTH, GROOVE_FLOOR, OUTER, OUTER_FADE = 0.07, 0.4, 0.235, 0.07
# Every slope above spans two or more of the template's edges on the chest (about 0.027 T apart). A
# steeper one, as the 0.03 T drop and groove were, falls between rows of vertices that cross it at a
# slant, and its shading breaks into patches along the mesh instead of drawing one border.
# Under a breast the lower border hides: the plate runs on down at least HIDDEN_DROP below its border
# and as far as the inframammary fold, so it ends in the fold or, between and beside the breasts, in a
# slope too gentle to read as an edge (12 mm over 0.18 T is at most 11 degrees, where the 0.05 T drop
# was 34). A breast this far out (a torso share) hides it all; a smaller one, part of the way, so the
# plate doesn't jump as a breast grows from nothing. Engineering values, chosen on the four female
# typical bodies' sections and renders (docs/parts/chest.md).
HIDDEN_DROP, HIDING_BREAST = 0.18, 0.02
# Between the breasts the plate filled the cleavage: it grows from the sternal groove's floor to its
# full thickness within GROOVE_WIDTH of the midline, which lifts the skin between the breasts' inner
# edges as much as their inner slopes. So under the breasts the groove widens out to their inner
# edges and grows over GROOVE_WIDTH past them. From the apex up to the footprint's top it narrows
# back to the bare chest's groove, so the upper chest keeps its shallow furrow.


class PectoralPlate(Layer):
    """The body with its chest's front read `thickness` further forward."""

    def __init__(self, body, L, p):
        super().__init__(body)
        self.torso = L["T"]
        self.crotch = L["crotch"]
        self.size = p["pectoralThickness"] * L["T"]
        # How far below its border the plate is gone: DROP on a bare chest, longer under a breast.
        self.hidden = 0.0
        if p.get("breastProjection", 0.0) > 0:
            self.hidden = float(ramp(p["breastProjection"] / HIDING_BREAST))
            fold = (L["pit"] - p["breastFoldDrop"] * L["T"] - L["crotch"]) / L["T"]
            self.end = min(fold, BORDER - HIDDEN_DROP)
            width = p["breastBaseBreadth"]
            # Footprints lapped over the midline leave no edge past it for the groove to reach out to.
            self.inner = max(p.get("breastGap", 0.0) / 2, 0.0)
            self.top = fold + width
            self.apex = self.top - p["breastUpperPoleRatio"] * width

    def thickness(self, points):
        T = self.torso
        x = soft_fold(points[:, 0]) / T
        y = points[:, 1] / T
        z = (points[:, 2] - self.crotch) / T
        border = BORDER + BORDER_RISE * ramp((x - 0.03) / 0.16)
        if self.hidden > 0:
            drop = DROP + self.hidden * (border - self.end - DROP)
            below = ramp((z - border + drop) / drop)
        else:
            below = ramp((z - border + DROP) / DROP)
        above = ramp(1 - (z - border) / (TOP - border))
        vertical = np.where(z < border, below, above)
        medial = GROOVE_FLOOR + (1 - GROOVE_FLOOR) * ramp(x / GROOVE_WIDTH)
        if self.hidden > 0:
            cleft = GROOVE_FLOOR + (1 - GROOVE_FLOOR) * ramp((x - self.inner) / GROOVE_WIDTH)
            share = self.hidden * ramp((self.top - z) / (self.top - self.apex))
            medial = medial + share * (cleft - medial)
        across = medial * ramp((OUTER - x) / OUTER_FADE)
        front = ramp(-y / 0.06)
        return self.size * vertical * across * front

    def to_reference(self, points):
        q = np.array(points, dtype=float, copy=True)
        q[:, 1] += self.thickness(q)
        return q
