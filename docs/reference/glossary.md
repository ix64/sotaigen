# 术语表

[English](../en/reference/glossary.md)

文档和代码里的词，各取一种说法。左列是中文文档用的词，括号里是英文文档和代码用的词。

## 身体和参数

| 词 | 意思 |
|---|---|
| 身体（body） | 按一份参数做出的一个人：网格、骨骼、权重和修正形状 |
| 典型身体（typical body） | 自带的 12 个身体，`<性别>-<组>`，编辑器的起点，名字是约定 |
| 测试身体（test body） | 自带的、只给本工具测试用的身体，不是约定 |
| 参数（parameter） | 参数文件 `body` 里的一项，量法见[参数文件](parameters.md) |
| 参数文件（parameter file） | `format` 为 `sotaigen/params` 的 JSON，一个身体的全部参数 |
| 性别（sex） | `female` 或 `male`，决定有哪些参数、用哪套范围和规则 |
| 组（group） | 典型身体按比例和体格分的类：`chibi`、`compact`、`slim`、`long`、`balanced`、`rounded` |
| T | 躯干长 `torsoLength`。很多参数的范围按 T 的份额写 |
| 头身比（head count） | 身高除以头高 |
| 范围和规则（ranges and rules） | 参数各自的上下限，和跨参数的算术条件；不是约定 |

## 网格和骨骼

| 词 | 意思 |
|---|---|
| 模板（template） | 所有身体共用的网格，从 Blender Studio 的 CC0 素体切出，拓扑和顶点顺序固定 |
| 隐式场（implicit field） | 由参数搭出的体块，模板贴合到它上面 |
| 贴合（wrap） | 把模板贴到隐式场上 |
| 静止姿势（rest pose） | 绑定时的姿势：A 姿势，上臂离竖直向下 35° |
| 变形骨（deform bone） | 带蒙皮权重的骨，名字沿用 VRM humanoid |
| 控制骨（control bone） | `.blend` 里用来摆姿势的骨，变形骨名加 `_ctrl` |
| 半转骨、扭转骨（half bone, twist bone） | 由驱动表跟随别的骨的辅助骨，不直接摆 |
| 驱动表（driver table） | GLB 里说明辅助骨和修正形状怎样跟随姿势的两张表 |
| 修正形状（corrective） | 按姿势加权的形变目标（GLB 的 morph target、`.blend` 的形状键），补蒙皮在关节处的不足 |
| 表情（expression） | 五官部件上的形变目标，名字是 VRM 1.0 的预设名 |
| 检查姿势（check pose） | 用来检查身体的一组姿势，随 GLB 出厂，不是约定 |
| 干净范围（clean range） | 网格在多大的关节角度内还好看；比关节极限（真人的活动范围）小 |
| 颈环（neck ring） | 颈和头的接缝上的一圈顶点，换头的接口 |
| 头部挂点（head attachments） | `head`、`jaw`、两根眼骨，换来的头和五官绑在上面 |
| IPC 接触表（IPC contact table） | 按接触求解的修正形状，实验特性，`sotaigen[contact]` |

## 文件和流程

| 词 | 意思 |
|---|---|
| 约定（contract） | 下游可以依赖的名字、结构和单位，见[对外约定](README.md) |
| 金样（golden copy） | `tests/contracts/` 里约定的副本，测试拿代码和它对照 |
| 诊断（diagnostic） | 检查找到的一条 `{code, level, values}`，见[诊断](diagnostics.md) |
| 构建（build） | 把自带身体贴合、绑定、摆姿势、量指标，写到 `build/` |
| 下游（downstream） | 用本工具做角色的项目 |
| 用户（the maintainer） | 内部工作文档里的“用户”指项目的维护者，待决定的问题由维护者定 |
