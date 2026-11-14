# 骨骼

[English](../en/reference/skeleton.md)

冻结的约定 2：骨的名字、父子关系、顺序、局部轴和静止姿势。金样在 `tests/contracts/bones.json`。

## 规则

- **名字** 沿用 VRM humanoid 的骨名，`left` 指穿着者左侧。右侧和左侧逐根对应，名字的 `left` 换成 `right`。
- **顺序：** 父在子前。GLB 的 `skin.joints` 就是这个顺序，所以序号就是蒙皮权重的列。`jaw`、`leftEye`、`rightEye` 排在最后。
- **局部轴：** Y 沿骨从头到尾，Z 朝前，X 是弯曲轴，肢体向前弯为正。例外：`jaw` 的 X 是穿着者左（和 `head` 相同，X+ 张嘴），Y 指下巴；眼骨的 Y 是视线，Z 朝上，X+ 向上看，Z+ 向穿着者左看。
- **姿势：** 每根骨一组局部欧拉角 `[x, y, z]`（度），旋转是 `R = Rz · Ry · Rx`（列向量），在骨的静止旋转之后。不带侧的名字（`upperArm`）指两侧，右侧取 `(x, -y, -z)`。根可另带 `rootTranslation`（mm，本工具的轴）。
- **静止姿势：** A 姿势，上臂离竖直向下 35°。`.blend` 可以换静止姿势（`sotaigen blend --t-pose` 等），换了的文件骨名和轴不变。
- **每个顶点最多受 4 根骨影响，权重和为 1。** GLB 里权重存成 16 位，每个顶点的和正好是 65535。
- **骨的类别：** 半转骨和扭转骨以外的每根骨在 `.blend` 里各有一根控制骨（名字加 `_ctrl`）。半转骨（`…Half`）和扭转骨（`…Twist1`、`…Twist2`）由驱动表跟随别的骨，不该直接摆，见 [GLB](glb.md#辅助骨的驱动表)。`head`、`jaw`、两根眼骨是头部挂点，见[换头的接口](head-interface.md)。

## 骨骼表

<!-- generated: bones -->
| # | 骨 | 父骨 |
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
