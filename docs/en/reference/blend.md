# .blend

[中文](../../reference/blend.md)

The `.blend`'s part of contracts 2 and 4: the names of its objects, control bones, constraints, vertex
groups and custom properties, and the fields of `rig["sotaigen"]`. The golden copy is
`tests/contracts/blend.json`.

A file `sotaigen blend` writes holds no Python: it couples bones with constraints and simple-expression
drivers only, so it opens correctly with Auto Run Python Scripts off. It needs Blender 5.2 or later.

## Names

<!-- generated: blend-names -->
| What | Name |
|---|---|
| The armature object | `rig` |
| The skin mesh object | `body` |
| The empty holding the correctives' intermediate properties | `MCH-correctives` |
| A control bone is its deform bone's name plus this | `_ctrl` |
| The control shapes' prefix | `WGT-` |
| Vertex group: the head's skin | `headRegion` |
| Vertex group: the neck ring | `neckRing` |
| Custom property switching the joint limits | `jointLimits` |
| Custom property on the armature: what the GLB's extras.sotaigen says | `sotaigen` |
| Bone collection of the controls | `controls` |
| Bone collection of the deform bones | `deform` |
| Collection of the placeholder parts | `parts` |
| Collection of the mechanism | `mechanism` |
| Collection of the widgets | `widgets` |
| Constraints, by what they do | `followControl`, `forwardLift`, `halfSwing`, `jawGlide`, `jointLimit`, `shoulderRhythm`, `twistShare` |
| Bone collections under the controls | `arms`, `fingers`, `head`, `legs`, `torso` |
<!-- end generated -->

- **Control bones** are what is posed; the deform bones follow them (`followControl`). The half and twist
  bones have none. When two bodies are appended into one file, Blender renames the second one's objects
  `rig.001` and `body.001`.
- **The correctives and the expressions** are shape keys of `body`, named as in the GLB (the corrective
  target names are opaque: see [GLB](glb.md#extrassotaigen)). The intermediate properties that drive
  them live on `MCH-correctives`, named `<group>.<quantity>[.<key>]`, the group being the GLB driver
  table's group `name`.
- **The touch poses** (hand to head, hand to face) are pose assets, named with `.left` or `.both` last:

<!-- generated: pose-assets -->
`touchCheek.both`, `touchCheek.left`, `touchCheek.right`, `touchCrown.left`, `touchCrown.right`, `touchForehead.left`, `touchForehead.right`, `touchMouth.left`, `touchMouth.right`, `touchOcciput.both`, `touchOcciput.left`, `touchOcciput.right`
<!-- end generated -->

## `rig["sotaigen"]`

A custom property of the armature object, with the names and meanings of the GLB's `extras.sotaigen`
(its `format` being `"sotaigen/blend"`), plus `restFrameError` (the largest difference between a bone's
rest frame in Blender and this tool's, in matrix entries: a check that the local axes came through
unchanged). Its keys in this version:

<!-- generated: blend-info -->
`correctives`, `format`, `generator`, `head`, `name`, `parameters`, `quality`, `restFrameError`, `version`
<!-- end generated -->

`correctives` here only records each group's source: the drivers in the file are the coupling. The
head swap adds `headSwap` here, its report.
