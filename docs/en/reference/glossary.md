# Glossary

[中文](../../reference/glossary.md)

One word for each thing, in the documents and the code. The Chinese word is in brackets, for reading the
working documents.

## Bodies and parameters

| Word | Meaning |
|---|---|
| body (身体) | one person made from one set of parameters: mesh, bones, weights and correctives |
| typical body (典型身体) | the 12 bodies this tool comes with, `<sex>-<group>`, the editor's starting points; their names are contract |
| test body (测试身体) | bodies this tool comes with for its own tests only; not contract |
| parameter (参数) | one entry of a parameter file's `body`, measured as [Parameter file](parameters.md) says |
| parameter file (参数文件) | the JSON with `format` `sotaigen/params`: all of one body's parameters |
| sex (性别) | `female` or `male`: which parameters there are, and which ranges and rules hold |
| group (组) | the typical bodies' classes by proportion and build: `chibi`, `compact`, `slim`, `long`, `balanced`, `rounded` |
| T | the torso's length, `torsoLength`. Many parameters' ranges are shares of T |
| head count (头身比) | stature divided by head height |
| ranges and rules (范围和规则) | each parameter's bounds, and the arithmetic conditions across parameters; not contract |

## Mesh and bones

| Word | Meaning |
|---|---|
| template (模板) | the mesh every body shares, cut from Blender Studio's CC0 base mesh, its topology and vertex order fixed |
| implicit field (隐式场) | the masses built from the parameters, onto which the template is wrapped |
| wrap (贴合) | fitting the template onto the implicit field |
| rest pose (静止姿势) | the pose bound in: an A pose, the upper arms 35° off straight down |
| deform bone (变形骨) | a bone with skin weights, named as in VRM humanoid |
| control bone (控制骨) | a bone of the `.blend` to pose with: a deform bone's name plus `_ctrl` |
| half bone, twist bone (半转骨、扭转骨) | helper bones that follow others through the driver table, never posed directly |
| driver table (驱动表) | the GLB's two tables of how the helper bones and the correctives follow a pose |
| corrective (修正形状) | a morph target (the GLB) or shape key (the `.blend`) weighted by the pose, making up for skinning at a joint |
| expression (表情) | a morph target of the facial features, named by VRM 1.0's presets |
| check pose (检查姿势) | one of the poses a body is checked in, shipped with the GLB, not contract |
| clean range (干净范围) | how far a joint turns before the mesh stops looking right; narrower than the joint limits (people's range of motion) |
| neck ring (颈环) | the loop of vertices on the seam between neck and head: the head interface |
| head attachments (头部挂点) | `head`, `jaw` and the two eye bones, which a replacement head and its features are bound to |
| IPC contact table (IPC 接触表) | correctives solved by contact; experimental, `sotaigen[contact]` |

## Files and process

| Word | Meaning |
|---|---|
| contract (约定) | the names, structures and units a downstream may rely on: see [Contracts](README.md) |
| golden copy (金样) | a copy of a contract in `tests/contracts/`, which the tests hold the code to |
| diagnostic (诊断) | one `{code, level, values}` a check finds: see [Diagnostics](diagnostics.md) |
| build (构建) | wrapping, rigging, posing and measuring the bodies this tool comes with, into `build/` |
| downstream (下游) | a project making characters with this tool |
| the maintainer (用户) | in the working documents, 用户 ("the user") means the project's maintainer, who decides the open questions |
