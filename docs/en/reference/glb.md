# GLB

[中文](../../reference/glb.md)

The GLB's part of contract 4: the fields of `extras.sotaigen`, the structure of its two driver tables
and the diagnostics' codes. The golden copy is `tests/contracts/glb.json`, read from one typical body's
file of each sex.

## What the file holds

A binary glTF 2.0 file: the skin mesh (with UVs, skin weights, and the correctives as morph targets),
the bones as nodes, the placeholder features, ears and hair (a mesh each, the expressions as their
morph targets), and `extras.sotaigen`. Coordinates and units are in
[Coordinates and topology](coordinates-topology.md).

- **A node's rest rotation** is the rotation between its bone and its parent's, the root's being the
  identity. So the [skeleton](skeleton.md)'s Euler angles apply right after the node's rest rotation and
  mean the same in any runtime.
- **`skin.joints`** is in the skeleton's order. Weights are 16-bit, each vertex's summing to 65535.
- `asset.generator` is `"sotaigen <version>"`.

## `extras.sotaigen`

| Key | What it holds | In the contract |
|---|---|---|
| `format`, `version` | `"sotaigen/glb"`, `1`. Raised when a field's meaning changes, a field goes, or a driver table's structure changes; a reader refuses a version it doesn't know | yes |
| `generator` | `{name: "sotaigen", version: the package's, fingerprint: the rig's}`; the fingerprint is opaque, only telling apart two files of one version whose generating code differed | yes (not the fingerprint's value) |
| `name` | the name of a body this tool comes with, when the parameters are exactly its; otherwise absent | yes |
| `parameters` | the whole [parameter file](parameters.md) it was made from: with the same version, the GLB alone makes the same body again | yes |
| `quality` | `{checked, found}`: `checked` the checks that ran, `found` a list of [diagnostics](diagnostics.md) | yes |
| `drivers` | the helper bones' driver table, below | yes |
| `correctives` | the correctives' driver table, below | the structure, not the target names |
| `expressions` | the expressions in the file, VRM 1.0 preset names, in the feature meshes' `targetNames` | yes |
| `head` | the [head interface](head-interface.md) | yes |
| `splitFrom` | each UV seam copy's original vertex | yes |
| `checkPoses` | the poses that check a body, whose names and angles change with versions | no |

**Morph target names:** the skin's corrective target names are opaque. The contract only promises that
each name listed in `correctives.groups` is a morph target of the skin, weighted as the table says;
the names change with bodies and versions, so don't read meaning into them. The expressions' names are
in the contract.

The values in this version:

<!-- generated: glb-values -->
| What | Value |
|---|---|
| `correctives.groups[].name` | `axilla.left`, `axilla.right`, `groin`, `hang.left`, `hang.right` |
| `quality.checked` | `parameters`, `poses`, `uv`, `wrap` |
| `expressions` | `blink`, `happy`, `aa` |
<!-- end generated -->

A reader must tolerate corrective groups, checks and diagnostic codes it doesn't know: 0.1.x may add
them.

## Helper bone drivers

`drivers` gives each helper bone's full name and the bone it follows, so a reader puts no names
together.

| Key | What it holds |
|---|---|
| `central` | the names that mean one bone (torso, neck, head, jaw); any other name without a side means both sides, the right taking `(x, -y, -z)` |
| `rhythm` | `share`: how much of the upper arm's swing the shoulder girdle takes; `forwardLift`: how much more it lifts when the arm lifts forward; `arms`: each side's `{arm, girdle, side}`, `side` the sign of raising the arm sideways on Euler Z |
| `half` | `{bone, of, roll?}`: a half bone takes half of `of`'s swing, and with `roll` that share of `of`'s real twist as well |
| `twist` | `{bone, of, share}`: a twist bone takes `of`'s real twist about its local Y times `share`; the upper arm's read the upper arm after the girdle's share |
| `glide` | `{bone: [x, y, z]}`: metres of slide per degree of opening, in the bone's rest frame; only `jaw` |

## Corrective drivers

`correctives.groups`, one group for one part on one side. `name` is the group's name; `source` how it
was made (`sculpted`, `field` or `ipc`).

| Key | What it holds |
|---|---|
| `axes` | `{bone, read, scale, keys}`: one component of `bone`'s pose times `scale`, interpolated smoothly between the `keys` |
| `fades` | `{bone, read, limit}`: times `smoothstep(1 − abs(reading) / limit)` |
| `samples` | each target's key on each axis |
| `targets` | the target names, one for each of `samples` |

- **Readings:** the pose is first spread to each bone's intent by `central` (the right mirrored, a later
  entry over an earlier one), then split into swing and twist: `swingX`, `twist`, `swingZ`, or
  `largest` (the greatest of their absolute values). `scale` is only ±1, reading the right side as the
  left. The pose itself is read, not the bone after the girdle's share.
- **Weights:** a target's weight is the product of each axis's interpolation weight in its cell, times
  the fades; the fades multiply bone by bone, in the order each bone first appears.
- A file with the experimental IPC contact tables (which `sotaigen glb` doesn't write, and the
  developer tools' `--contact` does) has groups with a `"kind"`, outside the contract. A reader refuses
  a group of a `kind` it doesn't know.
