# GLB

[English](../en/reference/glb.md)

冻结的约定 4 的 GLB 部分：`extras.sotaigen` 的字段、两张驱动表的结构和诊断的代码。金样在 `tests/contracts/glb.json`，从两个性别各一个典型身体的文件读出。

## 文件里有什么

一个 glTF 2.0 的二进制文件：皮肤网格（带 UV、蒙皮权重、修正形状的变形目标）、骨骼节点、占位的五官、耳和头发（各是一个网格，表情是它们的变形目标），和 `extras.sotaigen`。坐标和单位见[坐标和拓扑](coordinates-topology.md)。

- **节点的静止旋转** 就是骨骼之间的相对旋转，根节点是单位阵。所以[骨骼](skeleton.md)的欧拉角直接写在节点的静止旋转之后，任何运行时含义相同。
- **`skin.joints`** 的顺序就是骨骼表的顺序。权重 16 位，每个顶点的和是 65535。
- `asset.generator` 是 `"sotaigen <版本>"`。

## `extras.sotaigen`

| 键 | 内容 | 进约定 |
|---|---|---|
| `format`、`version` | `"sotaigen/glb"`、`1`。字段的含义变了、去掉字段、驱动表的结构变了时加一；读者拒绝不认识的版本 | 是 |
| `generator` | `{name: "sotaigen", version: 包版本, fingerprint: 绑定的指纹}`；指纹不透明，只用来区分同一版本里生成代码不同的两个文件 | 是（指纹的值不是） |
| `name` | 参数和某个自带身体完全相同时写它的名字，否则没有这个键 | 是 |
| `parameters` | 生成它的完整[参数文件](parameters.md)，拿到 GLB 就能在同一版本里重新生成同一个身体 | 是 |
| `quality` | `{checked, found}`：`checked` 是跑过的检查，`found` 是[诊断](diagnostics.md)的列表 | 是 |
| `drivers` | 辅助骨的驱动表，见下 | 是 |
| `correctives` | 修正形状的驱动表，见下 | 结构是，目标名不是 |
| `expressions` | 文件里有的表情名，VRM 1.0 的预设名，在部件网格的 `targetNames` 里 | 是 |
| `head` | [换头的接口](head-interface.md) | 是 |
| `splitFrom` | UV 接缝复制出的每个顶点的原顶点 | 是 |
| `checkPoses` | 检查身体用的姿势，名字和角度随版本变 | 否 |

**变形目标名：** 皮肤上的修正目标名不透明。约定只保证 `correctives.groups` 里列出的每个名字都是皮肤网格上的一个变形目标，权重按表算；名字随身体和版本变，不要按名字解析含义。表情名是约定。

这一版的取值：

<!-- generated: glb-values -->
| 东西 | 值 |
|---|---|
| `correctives.groups[].name` | `axilla.left`, `axilla.right`, `groin`, `hang.left`, `hang.right` |
| `quality.checked` | `parameters`, `poses`, `uv`, `wrap` |
| `expressions` | `blink`, `happy`, `aa` |
<!-- end generated -->

读者遇到不认识的修正组、检查名和诊断代码要照样容下，它们可以在 0.1.x 里增加。

## 辅助骨的驱动表

`drivers`，每根辅助骨写全名和它跟随的骨，读者不拼名字。

| 键 | 内容 |
|---|---|
| `central` | 只指一根骨的名字（躯干、颈、头、下颌）；别的不带侧的名字指两侧，右侧取 `(x, -y, -z)` |
| `rhythm` | `share`：上臂摆动里肩胛分担的份额；`forwardLift`：前抬另外抬肩胛的份额；`arms`：每侧 `{arm, girdle, side}`，`side` 是侧举欧拉 Z 的符号 |
| `half` | `{bone, of, roll?}`：半转骨取 `of` 摆动的一半，有 `roll` 时另取 `of` 真实扭转的这一份 |
| `twist` | `{bone, of, share}`：扭转骨取 `of` 绕局部 Y 的真实扭转乘 `share`；上臂的读肩胛分担之后的上臂 |
| `glide` | `{骨: [x, y, z]}`：每张开一度滑动的米数，在骨的静止坐标系里，只有 `jaw` |

## 修正形状的驱动表

`correctives.groups`，一组是一侧的一个部位。`name` 是组名，`source` 是它怎么做的（`sculpted`、`field` 或 `ipc`）。

| 键 | 内容 |
|---|---|
| `axes` | `{bone, read, scale, keys}`：读 `bone` 的一个分量乘 `scale`，在 `keys` 上做平滑的区间插值 |
| `fades` | `{bone, read, limit}`：乘 `smoothstep(1 − abs(读数) / limit)` |
| `samples` | 每个目标在各轴上落在哪个键 |
| `targets` | 目标名，和 `samples` 一一对应 |

- **读数：** 姿势先按 `central` 展开成每根骨的意图（右侧镜像，后写的覆盖先写的），再分解成摆动和扭转：`swingX`、`twist`、`swingZ`，或 `largest`（三者绝对值的最大）。`scale` 只有 ±1，用来把右侧读成左侧。读的是姿势本身，不是肩胛分担之后的骨。
- **权重：** 一个目标的权重是各轴在它那一格上的插值权重之积，再乘淡出；淡出按骨逐根相乘，按骨第一次出现的顺序。
- 实验特性 IPC 接触表的文件（`sotaigen glb` 不写，开发工具的 `--contact` 写）里有带 `"kind"` 的组，不在约定之内。读者遇到不认识的 `kind` 要拒绝那一组。
