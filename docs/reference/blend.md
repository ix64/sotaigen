# .blend

[English](../en/reference/blend.md)

冻结的约定 2 和 4 的 `.blend` 部分：物体、控制骨、约束、顶点组和自定义属性的名字，以及 `rig["sotaigen"]` 的字段。金样在 `tests/contracts/blend.json`。

`sotaigen blend` 写的文件里没有 Python：联动只用约束和简单表达式驱动器，关掉自动运行脚本也能正确打开。要 Blender 5.2 或更新的版本。

## 名字

<!-- generated: blend-names -->
| 东西 | 名字 |
|---|---|
| 骨架物体 | `rig` |
| 皮肤网格物体 | `body` |
| 放修正形状中间属性的空物体 | `MCH-correctives` |
| 控制骨是变形骨名加这个后缀 | `_ctrl` |
| 控件形状的前缀 | `WGT-` |
| 顶点组：头的皮肤 | `headRegion` |
| 顶点组：颈环 | `neckRing` |
| 开关关节极限的自定义属性 | `jointLimits` |
| 骨架上的自定义属性，内容同 GLB 的 `extras.sotaigen` | `sotaigen` |
| 控制骨的骨骼集合 | `controls` |
| 变形骨的骨骼集合 | `deform` |
| 占位部件的集合 | `parts` |
| 机制的集合 | `mechanism` |
| 控件形状的集合 | `widgets` |
| 约束，按作用命名 | `followControl`, `forwardLift`, `halfSwing`, `jawGlide`, `jointLimit`, `shoulderRhythm`, `twistShare` |
| 控制骨下的骨骼集合 | `arms`, `fingers`, `head`, `legs`, `torso` |
<!-- end generated -->

- **控制骨** 用来摆姿势，变形骨跟随它（`followControl`）。半转骨和扭转骨没有控制骨。两个身体追加进同一个文件时，Blender 会把第二个的物体改名为 `rig.001`、`body.001`。
- **修正形状和表情** 是 `body` 的形状键，名字和 GLB 相同（修正目标名不透明，见 [GLB](glb.md#extrassotaigen)）。驱动它们的中间属性在 `MCH-correctives` 上，名字是 `<组>.<量>[.<键>]`，组名就是 GLB 驱动表里组的 `name`。
- **触点姿势**（手到头、手到脸）存成姿势资产，名字以 `.left`、`.both` 结尾：

<!-- generated: pose-assets -->
`touchCheek.both`, `touchCheek.left`, `touchCheek.right`, `touchCrown.left`, `touchCrown.right`, `touchForehead.left`, `touchForehead.right`, `touchMouth.left`, `touchMouth.right`, `touchOcciput.both`, `touchOcciput.left`, `touchOcciput.right`
<!-- end generated -->

## `rig["sotaigen"]`

骨架物体上的自定义属性，和 GLB 的 `extras.sotaigen` 同名同义（`format` 是 `"sotaigen/blend"`），另加 `restFrameError`（Blender 里骨的静止坐标系和本工具的最大差，矩阵元素之差，用来确认局部轴没有被改动）。这一版的键：

<!-- generated: blend-info -->
`correctives`, `format`, `generator`, `head`, `name`, `parameters`, `quality`, `restFrameError`, `version`
<!-- end generated -->

`correctives` 在这里只记每组的来源；驱动关系就在文件的驱动器里。换头工具会在这里加 `headSwap`（它的报告）。
