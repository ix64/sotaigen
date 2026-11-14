# Skeleton

[中文](../../reference/skeleton.md)

Contract 2: the bones' names, parents, order, local axes and rest pose. The golden copy is
`tests/contracts/bones.json`.

## Rules

- **Names** are VRM humanoid's, and `left` is the wearer's left. Each right bone matches a left one,
  with `left` in its name replaced by `right`.
- **Order:** parents before children. The GLB's `skin.joints` is in this order, so a bone's index is
  its column of skin weights. `jaw`, `leftEye` and `rightEye` come last.
- **Local axes:** Y along the bone from head to tail, Z forward, X the axis it bends about, a limb
  bending forward being positive. Exceptions: `jaw`'s X is the wearer's left (as `head`'s, so X+ opens
  the mouth) and its Y points to the chin; an eye's Y is its line of sight and its Z up, X+ looking up
  and Z+ looking to the wearer's left.
- **Poses:** a bone's local Euler angles `[x, y, z]` in degrees, the rotation `R = Rz · Ry · Rx`
  (column vectors), after the bone's rest rotation. A name without a side (`upperArm`) means both
  sides, the right one taking `(x, -y, -z)`. The root may carry a `rootTranslation` (mm, this tool's
  axes).
- **Rest pose:** an A pose, the upper arms 35° off straight down. A `.blend` may have another rest pose
  (`sotaigen blend --t-pose` and the like); its bones keep their names and axes.
- **Each vertex has at most 4 bones, its weights summing to 1.** The GLB stores weights in 16 bits,
  each vertex's summing to exactly 65535.
- **Kinds of bones:** every bone but the half and twist bones has a control bone in the `.blend`
  (its name plus `_ctrl`). The half bones (`…Half`) and twist bones (`…Twist1`, `…Twist2`) follow other
  bones through the driver table and are not posed directly: see [GLB](glb.md#helper-bone-drivers).
  `head`, `jaw` and the two eyes are the head's attachments: see [Head interface](head-interface.md).

## Bones

<!-- generated: bones -->
| # | Bone | Parent |
|---|---|---|
| 0 | `root` | — |
| 1 | `hips` | `root` |
| 2 | `spine` | `hips` |
| 3 | `chest` | `spine` |
| 4 | `upperChest` | `chest` |
| 5 | `neck` | `upperChest` |
| 6 | `head` | `neck` |
| 7 | `leftShoulder` | `upperChest` |
| 8 | `leftUpperArm` | `leftShoulder` |
| 9 | `leftLowerArm` | `leftUpperArm` |
| 10 | `leftHand` | `leftLowerArm` |
| 11 | `leftPalmCup` | `leftHand` |
| 12 | `leftThumbMetacarpal` | `leftHand` |
| 13 | `leftThumbProximal` | `leftThumbMetacarpal` |
| 14 | `leftThumbDistal` | `leftThumbProximal` |
| 15 | `leftIndexProximal` | `leftHand` |
| 16 | `leftIndexIntermediate` | `leftIndexProximal` |
| 17 | `leftIndexDistal` | `leftIndexIntermediate` |
| 18 | `leftMiddleProximal` | `leftHand` |
| 19 | `leftMiddleIntermediate` | `leftMiddleProximal` |
| 20 | `leftMiddleDistal` | `leftMiddleIntermediate` |
| 21 | `leftRingProximal` | `leftPalmCup` |
| 22 | `leftRingIntermediate` | `leftRingProximal` |
| 23 | `leftRingDistal` | `leftRingIntermediate` |
| 24 | `leftLittleProximal` | `leftPalmCup` |
| 25 | `leftLittleIntermediate` | `leftLittleProximal` |
| 26 | `leftLittleDistal` | `leftLittleIntermediate` |
| 27 | `leftUpperLeg` | `hips` |
| 28 | `leftLowerLeg` | `leftUpperLeg` |
| 29 | `leftFoot` | `leftLowerLeg` |
| 30 | `leftToes` | `leftFoot` |
| 31 | `leftUpperArmHalf` | `leftShoulder` |
| 32 | `leftLowerArmHalf` | `leftUpperArm` |
| 33 | `leftUpperLegHalf` | `hips` |
| 34 | `leftLowerLegHalf` | `leftUpperLeg` |
| 35 | `leftHandHalf` | `leftLowerArm` |
| 36 | `leftUpperArmTwist1` | `leftUpperArm` |
| 37 | `leftUpperArmTwist2` | `leftUpperArm` |
| 38 | `leftLowerArmTwist1` | `leftLowerArm` |
| 39 | `leftLowerArmTwist2` | `leftLowerArm` |
| 40 | `rightShoulder` | `upperChest` |
| 41 | `rightUpperArm` | `rightShoulder` |
| 42 | `rightLowerArm` | `rightUpperArm` |
| 43 | `rightHand` | `rightLowerArm` |
| 44 | `rightPalmCup` | `rightHand` |
| 45 | `rightThumbMetacarpal` | `rightHand` |
| 46 | `rightThumbProximal` | `rightThumbMetacarpal` |
| 47 | `rightThumbDistal` | `rightThumbProximal` |
| 48 | `rightIndexProximal` | `rightHand` |
| 49 | `rightIndexIntermediate` | `rightIndexProximal` |
| 50 | `rightIndexDistal` | `rightIndexIntermediate` |
| 51 | `rightMiddleProximal` | `rightHand` |
| 52 | `rightMiddleIntermediate` | `rightMiddleProximal` |
| 53 | `rightMiddleDistal` | `rightMiddleIntermediate` |
| 54 | `rightRingProximal` | `rightPalmCup` |
| 55 | `rightRingIntermediate` | `rightRingProximal` |
| 56 | `rightRingDistal` | `rightRingIntermediate` |
| 57 | `rightLittleProximal` | `rightPalmCup` |
| 58 | `rightLittleIntermediate` | `rightLittleProximal` |
| 59 | `rightLittleDistal` | `rightLittleIntermediate` |
| 60 | `rightUpperLeg` | `hips` |
| 61 | `rightLowerLeg` | `rightUpperLeg` |
| 62 | `rightFoot` | `rightLowerLeg` |
| 63 | `rightToes` | `rightFoot` |
| 64 | `rightUpperArmHalf` | `rightShoulder` |
| 65 | `rightLowerArmHalf` | `rightUpperArm` |
| 66 | `rightUpperLegHalf` | `hips` |
| 67 | `rightLowerLegHalf` | `rightUpperLeg` |
| 68 | `rightHandHalf` | `rightLowerArm` |
| 69 | `rightUpperArmTwist1` | `rightUpperArm` |
| 70 | `rightUpperArmTwist2` | `rightUpperArm` |
| 71 | `rightLowerArmTwist1` | `rightLowerArm` |
| 72 | `rightLowerArmTwist2` | `rightLowerArm` |
| 73 | `jaw` | `head` |
| 74 | `leftEye` | `head` |
| 75 | `rightEye` | `head` |
<!-- end generated -->
