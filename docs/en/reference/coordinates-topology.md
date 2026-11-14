# Coordinates and topology

[中文](../../reference/coordinates-topology.md)

Contract 1: every body has the same topology with its vertices in the same order. What is made for
one body (a head, clothes, edited weights) carries over to another by vertex index.

## Coordinates

| Where | Axes | Unit |
|---|---|---|
| Inside this tool, the `.blend`, lengths in the parameter file | Z up (height above the ground), forward is -Y, the wearer's left is +X | mm (Blender metres in the `.blend`, the same numbers as the GLB) |
| The GLB | this tool's `(x, y, z)` written as `(x, z, -y)`: +Y up, the front facing +Z, the wearer's left +X, as in VRM | m |

The GLB's change of axes is a pure rotation, so the faces keep their winding. The body is symmetric:
the right side mirrors the left.

## Topology

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

- **Vertex order:** the template's source mesh first (8781 vertices), then the head (3313, from index
  8781), then the loops added later, batch by batch (elbows and knees 400, shoulders and armpits 182).
  **Indices are only appended:** a loop added later goes last, and no existing index moves.
- **Faces:** quads and triangles, the face list above, the same for every body. The GLB splits each quad
  along its shorter diagonal on the template, so every body shares one triangle list as well.
- **UV seam copies:** the GLB's skin duplicates the vertices on UV seams; the copies come after every
  original vertex, and `extras.sotaigen.splitFrom` gives each copy's original. The vertices before them
  are in the order above. The `.blend` duplicates nothing: its UVs are per face corner.
- **UVs:** the same layout on every body.
- **The head's region and the neck ring** belong to the head interface: see
  [Head interface](head-interface.md).

The golden copy is `tests/contracts/topology.json`.
