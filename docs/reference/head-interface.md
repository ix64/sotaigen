# 换头的接口

[English](../en/reference/head-interface.md)

冻结的约定 6：颈环和头部挂点。约定的是序号和名字，不是位置：位置随身体参数变。GLB 在 `extras.sotaigen.head`，`.blend` 在 `rig["sotaigen"]["head"]`。`sotaigen swap-head` 靠它换头，别的工具也可以靠它。

| 字段 | 内容 |
|---|---|
| `bones` | 头部挂点的骨：`head`、`jaw`、`leftEye`、`rightEye`，局部轴见[骨骼](skeleton.md) |
| `region` | 头的皮肤，`[起始顶点, 个数]`，连续一块，所有身体相同。`.blend` 里是顶点组 `headRegion`：头和身体是同一个网格，法线在环上连续 |
| `neckRing` | 颈环的顶点序号，从上往下看逆时针，所有身体相同。`.blend` 里是顶点组 `neckRing`。头的第一排在它上面 |
| `crownVertex` | 头顶的顶点序号 |
| `eyeLine`、`eyeX` | 左眼骨的骨头在头顶之下多少、离中线多远，单位都是“头顶到颈环中心的高度”：`eyeLine = (头顶 Z − 眼 Z) / 高`，`eyeX = abs(眼 X) / 高`，高 = 头顶 Z − 颈环顶点的平均 Z |

<!-- generated: topology -->
| 东西 | 值 |
|---|---|
| 顶点 | 12676 |
| 面 | 12764 |
| 面列表的 SHA-256 | `9b9e5718f450e1d5a20b5e9a5e056b9fa04290f7d28fd4b1d5461da0fbd4b48a` |
| 头的区块 | 从 8781 起 3313 个 |
| 颈环 | 36 个顶点 |
| 头顶的顶点 | 12093 |
<!-- end generated -->

- **环的权重：** 颈环上只有 `neck`、`head`，加一点 `upperChest` 和两侧 `Shoulder`；`jaw` 在环和颈上处处为 0。所以换头只需管颈环以上。
- **挂上去的东西：** 牙、舌和下半张脸绑 `jaw`，眼球绑眼骨，瞳孔的转心在眼骨的骨头上。眼骨的长度是占位眼球的半径，不是约定。下颌的前滑是 `jaw` 的一部分（驱动表的 `glide`），绑上就跟着走。
- **表情** 只约定名字（VRM 1.0 的预设名），不约定形状。
- `eyeLine`、`eyeX` 只用接口自己的字段量，所以换来的头也量得出，换头前后含义相同。

换头工具怎样认头文件（坐标、皮肤、颈口、挂点、权重）写在工具的说明里，不是约定。
