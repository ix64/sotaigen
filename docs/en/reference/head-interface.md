# Head interface

[中文](../../reference/head-interface.md)

Contract 6: the neck ring and the head's attachments. The contract is indices and names, not
positions: positions change with the body's parameters. In the GLB it is `extras.sotaigen.head`, in the
`.blend` `rig["sotaigen"]["head"]`. `sotaigen swap-head` swaps heads by it, and other tools may rely on it
too.

| Field | What it holds |
|---|---|
| `bones` | the bones the head attaches to: `head`, `jaw`, `leftEye`, `rightEye`, with the local axes in [Skeleton](skeleton.md) |
| `region` | the head's skin as `[first vertex, count]`, one run, the same on every body. In the `.blend` it is the vertex group `headRegion`: the head and the body are one mesh, the normals continuous across the ring |
| `neckRing` | the neck ring's vertex indices, anticlockwise seen from above, the same on every body. In the `.blend` it is the vertex group `neckRing`. The head's first row sits on it |
| `crownVertex` | the index of the vertex at the top of the head |
| `eyeLine`, `eyeX` | how far the left eye bone's head is below the crown and off the midline, both in units of "the crown's height over the neck ring's centre": `eyeLine = (crown Z − eye Z) / height`, `eyeX = abs(eye X) / height`, height = crown Z − the neck ring vertices' mean Z |

<!-- generated: topology -->
| What | Value |
|---|---|
| Vertices | 12676 |
| Faces | 12764 |
| SHA-256 of the face list | `9b9e5718f450e1d5a20b5e9a5e056b9fa04290f7d28fd4b1d5461da0fbd4b48a` |
| Head region | 3313 from 8781 |
| Neck ring | 36 vertices |
| Crown vertex | 12093 |
<!-- end generated -->

- **The ring's weights:** only `neck` and `head` on the neck ring, with a little `upperChest` and each
  side's `Shoulder`; `jaw` is 0 on the ring and the whole neck. So a head swap only has to mind what is
  above the ring.
- **What attaches:** teeth, tongue and the lower face to `jaw`, the eyeballs to the eye bones, the pupils
  turning about the eye bone's head. An eye bone's length is the placeholder eyeball's radius, not
  contract. The jaw's forward glide is part of `jaw` (the driver table's `glide`), so what is bound to
  it moves with it.
- **Expressions:** only their names are contract (VRM 1.0 preset names), not their shapes.
- `eyeLine` and `eyeX` are measured from the interface's own fields alone, so a replacement head
  measures the same way and they mean the same before and after a swap.

How the head swap reads a head file (its coordinates, skin, neck opening, attachments, weights) is the
tool's documentation, not contract.
