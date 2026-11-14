# 导出与运行时

肩腋的姿势驱动和接触表见[肩腋与胸侧](../parts/shoulder-axilla.md#导出)。

## 现状

导出（含部件、覆盖法线、稀疏变形目标）、加载页（含部件开关、变形目标滑块、赛璐璐切换）、辅助骨驱动和一致性检查都已实现，`pnpm run runtime-check` 在真实 Chrome 里检查。默认导出带五官、耳、占位头发和整头平滑法线，页面打开就是带脸的模型。VRM 导出尚未实现，准备项和待验证问题见[VRM 1.0 差距](#vrm-10-差距)。

**还没解决：**

- 文件大小、导出和加载耗时、绘制开销需在当前身体上复测，见[大小和耗时](#大小和耗时)。一致性以 `pnpm run runtime-check` 的检查结果为准，门槛见下文。
- 颈部的明暗线有一个小台阶，来自几何，没有查是哪个领域（[已知限制](#已知限制)）。
- 调滑块得到的身体没有塌陷面一项（[导出分级](#导出分级)），导出的耗时在接入姿势修正和质量列表之后没有重新量（[大小和耗时](#大小和耗时)）。

```sh
# 导出到 build/bodies/<预设>/body.glb，带部件和平滑法线；不写预设名就是全部预设。--no-parts 只导出皮肤
uv run sotaigen-dev export female-long

# 浏览器检查的依赖，只有下面两条命令需要，uv run sotaigen-dev test 不需要
pnpm install

# Validator、真实 Chrome 里的一致性、加载耗时、截图。退出码非 0 表示不通过
# 报告在 build/reports/runtime-check/report.json，截图在同一目录的 shots/
pnpm run runtime-check

# 替身部件的 GLB，用 `shot.html?standin=1` 查看。只有手动看替身时才要，runtime-check 不用它
uv run scripts/standin_parts.py female-long

# 只看一个导出的文件（先导出），整个窗口只有身体，参数都可选。调表情和部件用编辑器的“显示”面板：
# http://127.0.0.1:8124/runtime/shot.html?body=female-long&pose=headTurn&look=toon&view=threequarter&framing=head&hide=hair&morph=blink:1,happy:0.5
pnpm run editor
```

| 文件 | 作用 |
|---|---|
| `sotaigen/output/glb.py` | 写 GLB，只读成品，不进指纹。`export_with_face` 是命令默认走的入口，`export_body` 不带部件，网格损坏时两者都拒绝（[导出分级](#导出分级)） |
| `web/runtime/driver.mjs` | 辅助骨驱动，不依赖 three.js，`uv run sotaigen-dev test` 用 Node 直接跑它 |
| `web/runtime/body.mjs` | 加载身体，编辑器和检查都用它 |
| `web/runtime/shot.ts`、`shot.html` | 按地址参数显示一个身体，没有界面也不带 React，用于截图和量加载耗时。服务是共用的 `web/serve.mjs` |
| `web/runtime/toon.mjs` | 页面的赛璐璐观感 |
| `scripts/standin_parts.py` | 替身部件，走遍真实部件没走的契约路径（透明、裁剪、抄来的权重、皮肤的变形目标），不是任何设计。`tests/test_export.py` 用它 |
| `web/runtime/check.mjs`、`check-page.mjs`、`check.html`、`scripts/runtime_check.py` | 浏览器检查。部件开关和表情滑块能否到达各网格，已移到 `web/tests/ui.spec.ts` |
| `tests/test_export.py` | 不要浏览器和 Blender，只要 `.node-version` 固定的 Node，npm 包不是前提 |

npm 依赖（均固定版本，发布超过 14 天）：`three` 0.186.1、`playwright-core` 1.64.1（只用系统 Chrome，不下载浏览器）、`gltf-validator` 2.0.0-dev.3.10（官方包只发布 dev 版本号）。

## 默认修正与可选接触

`sotaigen.corrective.recipe.Correctives.production` 统一组装修正：默认用雕塑的肩腋与腹股沟修正、隐式场垂臂修正，不加载 IPC，即使本机已有接触表也不读。`tests/test_default_path.py` 检查默认路径不加载可选接触依赖。

显式选择 `contact=True` 或 `uv run sotaigen-dev export --contact` 才走实验接触路径，后者另写 `body-ipc.glb`。编辑器在独立工作进程导出，显示预计耗时，不受调参滑块打断；它不受普通导出的速度目标限制。每组修正记录实际来源 `sculpted`、`field` 或 `ipc`；烘焙失败时退回可用修正并报告原因，不把回退标成 IPC 成功。各部位的失败与审计边界见[肩腋](../parts/shoulder-axilla.md#接触表)、[腹股沟](../parts/groin.md#接触表)，垂臂的默认算法见[绑定](rig.md#垂臂的隐式场修正)。

## 导出分级

GLB 和 [`.blend`](blend.md#导出分级) 用同一份判定（`sotaigen/output/quality.py` 的 `broken`、`BrokenBody`、`quality`）：

- **损坏，拒绝导出：** 贴合后有翻转的面、顶点越过中线、皮肤不闭合。不写文件；这一步在算修正之前，所以 `--contact` 不会为它开始烘焙。

| 入口 | 拒绝时 |
|---|---|
| `uv run sotaigen-dev export` | 这个身体的原因和找到的参数写到 stderr，接着做别的身体，最后以 1 退出 |
| `sotaigen glb` | 打印原因和找到的参数，以 1 退出。写出了文件就以 0 退出，警告和质量问题都在文件里 |
| `sotaigen.write_glb` | 抛 `BrokenBody`，不写文件。两个命令都调它 |
| 编辑器的“导出 GLB”和“带 IPC 接触导出” | 页面说明原因，不保存，再到后台找参数（[等待](editor.md#等待)）。调滑块生成的预览不拒绝，要能看到坏在哪里 |

- **质量，只提示：** 写进 `extras.sotaigen.quality`，和 `.blend` 的 `rig["sotaigen"]["quality"]` 同样的结构：`{"checked": [...], "found": [诊断...]}`。`checked` 写这次跑了哪些检查（`wrap`、`uv`、`parameters`，量了检查姿势时有 `poses`，烘焙了接触表时有 `contact`，`.blend` 另有 `touch`，换静止姿势时有 `rest`，换过头有 `headSwap`），所以“没量”和“量了没问题”分得开。`found` 的顺序是 `quality()`（脚接地、手脚接缝、检查姿势的塌陷面、UV 拉伸），`check` 也报的参数问题（接缝规则、目标没到、曲面），烘焙失败的接触表，`.blend` 再加触点和静止姿势。

**姿势检查的覆盖：** 预设和 IPC 导出由 `sotaigen.output.quality.glb_quality` 读取文件自身的修正，逐个摆检查姿势。普通调参生成和参数文件的 `sotaigen glb` 不执行这一项；检查是否执行以 `quality.checked` 是否含 `poses` 为准。预设参数的编辑器结果与 `uv run sotaigen-dev export` 逐字节一致，首次测量后可读缓存。其他质量检查随生成执行，耗时按当前环境记录。

编辑器留在缓存里的损坏身体的预览，`quality.found` 里有级别为 `broken` 的诊断，完好身体没有。

### 诊断

检查找到的每一条都是一个诊断 `{"code", "level", "values"}`，文件、工作进程的回答和页面都用它。代码、级别和每个代码的 `values` 是对外约定，表在[诊断](../reference/diagnostics.md)，金样是 `tests/contracts/diagnostics.json`。

- 实现在 `sotaigen/diagnostics.py`：Python 的 `Diagnostic` 是它的英文句子（`str` 的子类），另带 `code`、`level`、`values`，`json()` 是文件里存的样子；`sotaigen.check`、`InvalidSpec.problems`、`BrokenBody.reasons` 都是它。句子由 `SENTENCES` 按代码渲染，同一件事只有一种措辞，措辞不是约定。
- 规则名是 `sotaigen/params/rules.py` 里每条规则的第一项，页面的消息键是 `rule_<规则名>`；换头放行的颈规则另带 `forHeadSwap: true`。曲面不变量的名字见 `sotaigen.form.checks.field_failures`。
- 加代码时在 `SENTENCES` 写句子，`uv run sotaigen-dev contracts --write` 重写金样，再 `uv run sotaigen-dev reference` 重写约定说明的表。

### 哪项参数

拒绝分两步说，四条路径（`uv run sotaigen-dev blend`、`uv run sotaigen-dev export`、`sotaigen glb` 和 `blend`、编辑器）都一样：

- **部位，随贴合就知道：** 翻转的面和越中线的顶点按模板分区和高度归到页面的参数组（`sotaigen.form.wrap.body_part`），记在 `wrap.json` 的 `brokenAt`，只有损坏的身体才有这个键，完好身体的 `wrap.json` 不变。翻转、越中线和部位都量写出的 `wrapped.npy`，也就是抬完颈环之后；之后绑定阶段的静态造型（`rest.npy`）不在内，28 个自带身体上它最多移动 9 mm，没有把面翻过来（比较两份的面法线）。躯干分三段：躯干份额 0.6 以上是肩胸，0.3–0.6 是腰腹，以下前面是髋部、后面是臀部。诊断是 `wrap.flipped`，`at` 是 `{"neck": 2, "shoulder": 1}`，多的在前，英文句子写成 `flipped faces after the wrap: 3, at neck 2, shoulder 1`。皮肤不闭合说的是性别自己的面，和参数无关，句子里写明，也不给“去调整”。
- **参数，被拒绝时才试：** `sotaigen/mend.py` 把参数逐项退回 `<性别>-balanced` 在同一单位上的份额，重新求解和贴合，到第一项不再翻转、不再越中线为止。先试坏掉的部位那些组，再试其余，同组里离范围两端近的先试；规则或求解不接受的退回不试。说的是“退回这一项能修好，别的可能也行”，不是“这一项是原因”。试完都修不好就说修不好，只留部位。

逐项退回需要重复求解与贴合，因此只在拒绝导出后运行，不放进普通生成。诊断说明的是实际试出的可行调整，不从坏面位置、参数边界距离或规则余量推断原因。

**限制：** 退回并非最小改动；只试单项，无法解释需要多项同时调整的情况。耗时与成功率应在当前身体上复测。

## 已知限制

- **浏览器不再走真实部件没有的路径：** `runtime_check.py` 只跑真实文件，所以透明和裁剪部件、抄皮肤权重的部件、皮肤的变形目标不再进 Chrome，也不再有替身文件进 Validator。它们由 `tests/test_export.py` 按 glTF 规则回读，和 NumPy 比较；皮肤的变形目标和 UV 接缝还走 `future-fields.glb` 的 Validator。下颌还需覆盖张嘴与转头组合的真实浏览器回读，不能用单次抽样读数代表全部姿势。
- **页面的皮肤和 `review` 的色调不同：** 见“赛璐璐观感的评估”。
- **颈部的明暗线有一个小台阶：** 侧面和背面都能看到，不是 UV 接缝造成的（副本和原顶点的法线相同），应该来自几何，没有查是哪个领域。
- 采样动画可由 `sotaigen_dev/motion/clips.py glb` 烘焙成标准 glTF 的 rotation 和 translation 通道（`sotaigen.output.glb.baked_motion`），在普通播放器中还原；新输入的任意姿势仍需驱动代码。只转具名骨骼时辅助骨不动，`reach` 这类姿势差一百多毫米。
- **驱动读的是欧拉角：** `sided` 按 Rz·Ry·Rx 的角度算联动，所以四元数输入要先转回欧拉角（`eulerFromQuaternion`）。|y| 不超过 90° 时结果一致，在 y = 90° 的姿势上实测过（检查姿势 `wristRoll` 是 80°，在前臂的活动范围内，不再覆盖这条边界）；超过之后会转成另一组等价角度，旋转相同，肩胛联动却不同。修正形状的权重和扭转骨不受这一条影响：它们按摆动加扭转读（`swingTwist`，和 `sotaigen.rig.skinning.swing_twist` 相同），只看旋转本身；半转骨取摆动的一半，也只看旋转。
- 根节点接收 `rootTranslation`。文件的格式版本是 `extras.sotaigen.version`（现在是 1，[GLB](../reference/glb.md#extrassotaigen)），驱动表不另带版本，运行时拒绝不认识的版本。字段使用项目世界轴和毫米，运行时换为 glTF 轴和米。切换到省略该字段的姿势会回到静止根位置。
- **脚踝和手腕的明暗分界不是导出造成的：** 页面和 Blender 渲染都有，来自几何，属于贴合。
- **过期：** `build/bodies/<预设>/body.glb` 不受指纹管理，改了绑定要重新导出，文件里 `generator` 的指纹可以用来核对。`runtime_check.py` 每次都会重新导出。
- **只在一台机器上测过：** Chrome 156、ANGLE Metal、Apple M5。其他 GPU 和真实网络都没测。
- **薄壳的深度精度没有实测：** 页面相机的近裁剪是 0.01 m，24 位深度下 3 m 处的分辨率约 0.05 mm，30 m 处约 5 mm（估算）。五官薄壳的层间距 0.25–0.7 mm 在全身取景下够用，远距离可能闪烁，所以编辑器的相机把近裁剪拉到了 0.1 m。

**和 NumPy 不一致的一处：** `pose_verts` 拒绝 `root` 和未知骨骼名，`driver.mjs` 仍静默忽略。

## 部件契约

五官、耳、头发由头部代码生成，再由导出写入文件。头部代码照本节提供数据，不用读 `sotaigen/output/glb.py`。

本节是部件接入导出的接口。五官、耳、头发这些部件本身是占位，不算对外约定；表情只约定名字（[头部](../DECISIONS.md#头部)的“分工”）。

### 要提供什么

一个函数，给定预设的绑定结果 `rig = sotaigen.pipeline.rigged(name)`，返回部件列表和可选的皮肤法线。

```python
from sotaigen.output.glb import Material, Part, export_body

parts = [
    Part(
        name="eyeWhite",
        verts=verts,                 # V×3 float，mm，和身体同一坐标
        faces=faces,                 # 每个面一个顶点序号列表，从外面看逆时针
        material=Material(base_color=(0.96, 0.96, 0.94, 1.0)),
        morphs={"blink": blink_offsets, "happy": happy_offsets},   # 名字 → V×3，相对 verts 的偏移，mm
    ),
]
export_body("female-long", parts=parts, normals=head_normals)
```

头部的部件已经这样接好：`sotaigen.output.glb.export_with_face(name)` 和 `uv run sotaigen-dev export` 都用 `sotaigen.output.face` 的 `face_parts`、`head_normals`。

**`Part` 的字段：**

| 字段 | 必填 | 含义 |
|---|---|---|
| `name` | 是 | 字母开头，只含字母、数字、`_`、`-`。文件内唯一，不能是 `body` 或骨骼名。它就是节点名、网格名和页面上开关的名字 |
| `verts` | 是 | V×3，mm。Z 向上，前方 -Y，穿着者左侧 +X，和 `rig.verts` 相同。不需要自己转换成 glTF 坐标 |
| `faces` | 是 | 三角形、四边形或更多边，从外面看逆时针。四边形和多边形从第一个角切成扇形，切法不看预设的形状，所以所有预设共用一份三角形列表 |
| `material` | 是 | 一个部件只有一个材质。要多个材质就拆成多个部件 |
| `weights` | 否 | `None` 表示全部绑 `head` 骨，权重 1。否则是 V×B，列顺序同 `rig.bones`，和 `rig.weights` 一样：每行和为 1，最多 4 个非零。长到 t > 0.95 的东西从皮肤抄权重时用它 |
| `uvs` | 否 | V×2，glTF 约定（V 向下）。部件不做接缝拆分，需要接缝就自己复制顶点 |
| `normals` | 否 | V×3，长度任意但不能为 0，导出时归一化。不传就按面积加权算 |
| `morphs` | 否 | 名字 → V×3 偏移，mm，是相对 `verts` 的位移，不是绝对位置 |

**`Material` 的字段：**

| 字段 | 默认 | 含义 |
|---|---|---|
| `base_color` | 必填 | 线性空间 RGBA，0–1。不是 sRGB，sRGB 的颜色要先转换 |
| `alpha` | `"OPAQUE"` | `"OPAQUE"`、`"MASK"`（裁剪）、`"BLEND"`（透明，在所有不透明部件之后画） |
| `alpha_cutoff` | 0.5 | 只有 `MASK` 用 |
| `render_order` | 0 | 同一类（不透明或透明）里数字小的先画，相等时由运行时决定。比如眉毛要透过刘海，就把两者都设成 `BLEND`，眉毛的 `render_order` 比刘海大 |
| `double_sided` | `False` | 卡片、细条要两面都看得见时打开 |
| `outline` | `False` | 赛璐璐运行时是否给它描边。头发和耳打开，五官的薄壳保持关闭 |

构造 `Part` 和 `Material` 时就会检查形状、范围和名字，错了抛 `ValueError` 并说明是哪个部件的哪个字段。

### 变形目标的约定

- **同名即同一个表情：** 页面和运行时按名字一起驱动各部件的目标，所以 `blink` 在眼白、虹膜、睫毛上都叫 `blink`。VRM 以后要映射表情，建议直接用它的预设名（`blink`、`happy`、`aa`……）。
- **只给会动的部件：** 某个表情完全不影响某个部件，就不要把这个名字放进它的 `morphs`。显式传全 0 也合法，不占二进制，只占约 90 字节 JSON，用于想让目标序号固定的情况。
- **只有位置：** 表情不带法线变化，变形后法线保持静止时的值。薄壳贴在皮肤上时这没有影响。
- **有变化的判定：** 位移不超过 0.0001 mm 的顶点算没动，不存。所以浮点噪声不会让文件变大。
- **中性就是全部权重为 0：** 不要提供 `neutral` 目标。

### 皮肤的法线

`export_body` 和 `glb_bytes` 的 `normals` 是身体和头皮肤的覆盖法线：`len(rig.verts)`×3，rig 坐标，顺序同 `rig.verts`（未复制的顶点），长度不限但不能为 0。UV 接缝处的副本由导出自动跟随原顶点。

只想换头部时，先取默认值再混合：

```python
from sotaigen.output.glb import skin_normals

normals = skin_normals(rig.base, rig.verts)   # 没有接缝拆分的面积加权法线
normals[head] = 平滑结果
```

不传 `normals` 时就是 `skin_normals` 的结果，接缝副本和原顶点的法线相同。怎么算平滑法线由头部代码决定。

### 代码放哪里

`sotaigen.output.glb` 导入了 `sotaigen.pipeline`，而 `sotaigen.pipeline` 导入 `sotaigen.form.head`，所以 **`sotaigen/form/head.py` 不能导入 `sotaigen.output.glb`**，否则循环导入。部件的生成放在 `sotaigen.pipeline` 不导入的模块里（`sotaigen/output/face.py`），皮肤法线如果写在 `head.py` 里就只返回数组。`sotaigen/output/face.py` 导入 `sotaigen.output.glb`，所以 `export_with_face` 在函数里才导入它。`sotaigen/output/face.py` 不影响 `wrapped.npy`，没有构建阶段运行得到它，所以不进指纹。

`export_body` 本身默认不带部件，`--no-parts` 走的就是它。

### 现在不支持的

有需要就提，都能在不破坏现有字段的前提下加上：贴图（现在 `MASK`、`BLEND` 只看 `base_color` 的 alpha，`uvs` 写进文件但没有东西用它）、变形目标的法线、部件自己的节点变换。

### 文件里长什么样，要花多少字节

- 节点顺序是骨骼、`body`、各部件。每个部件有自己的网格、材质和变形目标，都用同一个 `skin`，节点没有变换。部件的数据都写在身体之后，所以身体的字节位置不受部件影响。
- 部件每个顶点 36 字节（位置、法线各 12，关节 4，权重 8），有 UV 再加 8。三角形每个 6 字节（顶点不到 65535 时）。
- 变形目标按实际移动的顶点存，每个 14 字节（序号 2、位移 12），约 140 字节 JSON。这比全存更大时自动改成全存。全 0 的目标不占二进制。
- 运行时 three.js 把目标展开成纹理，不论文件里怎样存，都是**每顶点每目标 16 字节**。所以皮肤上的目标在文件里很省，在显存里不省：13795 个顶点，每个目标 221 KB。

## GLB 约定

`extras.sotaigen` 的字段、两张驱动表的结构、变形目标名的约定和坐标都写在 [GLB](../reference/glb.md)，金样是 `tests/contracts/glb.json`。字段由 `sotaigen.output.metadata` 一处写，构建、`sotaigen glb`、API 和编辑器写的一样，所以同样的参数得到同样的字节（`tests/test_export.py` 和 `tests/test_editor.py` 核对）。下面是实现上的取舍。

每个文件嵌入完整 `parameters`，`name` 只在参数与自带身体完全相同时写；`generator` 不记录入口，避免同参数因入口不同产生不同字节。格式和字段含义以对外约定为准。

- **不加 Python 依赖：** GLB 是 12 字节文件头加 JSON 和二进制两个块，缓冲布局和蒙皮数据都要自己算，`pygltflib` 这类库只省下容器几十行。正确性由官方 Validator 和 `tests/test_export.py` 的回读把关。
- **坐标：** `(x, y, z)` mm 转成 `(x, z, -y)` m。这是纯旋转，绕向不变。glTF 的 +X 也是穿着者左侧，正面朝 +Z，和 VRM 相同。
- **局部轴不变：** 节点的静止旋转就是绑定层骨骼之间的相对旋转，根节点是单位阵。所以绑定层的欧拉角写在静止旋转之后，含义和 `sotaigen.rig.skeleton` 完全相同，运行时不需要换算坐标。
- **固定拓扑不破坏：** 前 `len(wrapped.npy)` 个顶点的顺序和 `wrapped.npy` 相同。四边形按模板上较短的对角线切分，所以所有预设共用一份三角形列表。`skin.joints` 的顺序就是 `weights.npz` 的列。
- **权重用 16 位：** 每顶点的和严格等于 65535，比浮点省一半（90 kB 对 180 kB），误差见下。
- **UV：** `export_body` 带上 `sotaigen.uv.export_fields` 的图集，接缝处复制顶点，皮肤的顶点数从 12676 变成 13985。接缝副本也在一致性比较之内。
- **皮肤的法线不随接缝拆开：** UV 接缝不是曲面的边，所以副本和原顶点共用 `skin_normals` 的同一条法线。存下的每条法线和曲面法线的差都是 0；拆开的话有约六分之一的法线偏 1° 以上，最大约 100°，赛璐璐的明暗线沿接缝有折痕。文件大小不变。
- **接缝和皮肤的变形目标：** `glb_bytes` 在有 `uvs` 和 `faceUvs` 时写 `TEXCOORD_0`，接缝处的顶点复制到末尾，来源记在 `extras.sotaigen.splitFrom`。`glb_bytes(..., morphs={名字: V×3 mm 偏移})` 给皮肤写变形目标和 `targetNames`，和部件一样是稀疏的，副本跟随原顶点。
- **Validator：** 2.0.0-dev.3.10。`pnpm run runtime-check` 要求默认文件和只有皮肤的文件都是 0 错误、0 警告；会有 1 条信息 `UNUSED_OBJECT`，因为还没有贴图。

## 辅助骨怎样驱动

**推荐运行时驱动，已实现。** GLB 带一张表（`extras.sotaigen.drivers`：肩胛比例、半转骨、扭转骨份额、躯干骨名，见[给绑定的接口](#给绑定的接口)），`driver.mjs` 每帧把姿势解析成全部骨骼的最终旋转。动画数据也要过这一层，不能直接写到骨骼上。需要给不受控的播放器时，再加烘焙作为补充。VRM 约束只作为 VRM 导出时的近似，不作为主方案。

| 方案 | 代价与边界 | 适用 |
|---|---|---|
| 运行时驱动 | 每个运行时都需实现驱动表，并通过一致性检查 | 页面里调整姿势、任意输入 |
| 烘进动画 | 只能播放已烘焙的动作，需验证关键帧之间的插值误差 | 不读驱动表的 glTF 播放器 |
| VRM 约束 | 不能完整表达现有联动，需在目标播放器中验证近似 | VRM 导出 |
| 只转具名骨骼 | 缺少辅助骨联动，不能重现本工具的姿势 | 静止显示 |

**VRM 约束做不到的三点：**

- **肩胛联动：** 约束不能改写来源骨，上臂要让出的份额没有地方减，还带 `max(…, 0)` 的单向截断。
- **上臂扭转：** 份额是负的（-0.8、-0.45），权重只能在 0–1。瞄准约束只能完全抵消滚转，不能重现这里的负份额。
- **半转骨：** `half_swing` 去掉了滚转，手的半转骨再加一份滚转，权重 0.5 的旋转约束做不到，每个节点也只能有一个约束。

## 运行时与 NumPy 一致

`scripts/runtime_check.py`（`pnpm run runtime-check [身体 ...]`）在真实 Chrome 里加载 GLB，走和页面相同的 `setPose`，用 three.js 自己的蒙皮着色器片段读回顶点，与 `pose_verts` 比较，全部检查姿势，读法有 GPU、three.js CPU 和经四元数往返三种。

- **门槛：** 任一读法差超过 1 mm 判失败（`runtime_check.py` 的 `LIMIT_MM`）。同一读法只转具名骨骼（没有驱动）时差不到 10 倍 `LIMIT_MM` 也判失败，证明这个检查能看出差异。
- **误差来源的检查：** 把三种读法与只使用 16 位权重的 NumPy 蒙皮对照，区分权重取整与浮点运算的误差；具体最大误差取当前运行报告。

`tests/test_export.py` 用 NumPy 按 glTF 规则回读同样的数据，包括替身走的那些路径，并在 Node 下让 `driver.mjs` 对比 `sided`（检查姿势加 200 个随机姿势）。所以驱动和 `sided` 漂移时，不用浏览器也会失败。

## 部件和变形目标与 NumPy 一致

同一个检查覆盖默认文件的皮肤和 10 个部件，全部姿势，权重全 0 和一组混合权重（`blink` 0.3、`happy` 0.4、`aa` 1，每个表情都用到）。NumPy 一侧是 `pose_verts(骨骼, 权重, 顶点 + Σ 权重 × 偏移, 姿势)`，也就是先变形后蒙皮，和 glTF 相同。

- **门槛：** 顶点和皮肤一样是 `LIMIT_MM`；读回的法线和给定值差不超过 0.01°（`NORMAL_LIMIT_DEGREES`）。
- **反向检查：** 同一份参考数据故意不给变形权重或不转骨骼，必须差很多，否则检查判失败。门槛是骨骼 5 mm（`BLIND_MM`）、表情 0.5 mm（`UNMORPHED_BLIND_MM`），表情里差得最少的是眉毛的 `happy`。

**自定义法线确实被 three.js 使用：**

- **读回：** three.js 里 `geometry.attributes.normal` 和给定值相同，接缝副本也在比较之内。给定法线必须和只有皮肤的文件差 5° 以上（`GIVEN_NORMALS_MUST_MOVE_DEGREES`），防止检查退化成比较相同输入。
- **渲染：** 同一个相机下，隐藏部件后，比较默认文件和只有皮肤的文件，再把后者的法线抄回去，检查渲染差异是否只来自法线。
- **默认法线不影响头：** `head_normals` 是显式传入的，所以皮肤默认法线的算法变了，默认文件的头部法线也不变。

**页面的操作：** 部件开关和表情滑块能否到达各网格由 `web/tests/ui.spec.ts` 考，对象是编辑器的检视面板：逐个关掉部件，逐个拉动表情，每一步读回状态，最后全部放回。赛璐璐的全身和头部截图由 `runtime_check.py` 写在 `build/reports/runtime-check/shots/`。

## 大小和耗时

运行 `uv run sotaigen-dev export` 和 `pnpm run runtime-check`，记录当前身体的 GLB 大小、导出耗时、浏览器加载及绘制耗时。默认导出与 `--no-parts` 分开比较；导出耗时要包含姿势修正和质量检查，缓存命中另记。浏览器、GPU、机器负载和依赖版本要随结果记录。回环下载不代表真实网络，尚未评估 Draco 或 meshopt 压缩。

## 稀疏变形目标

稀疏存储只写有变化的顶点及其序号，是否节省空间取决于变化顶点的比例。每种五官独立成节点还可省掉与它无关的目标，也允许分别设置材质。

稀疏 GLB 不保证降低运行时显存：three.js 会展开变形纹理。评估新增表情或皮肤目标时，应分别量文件大小和运行时纹理占用。重复使用同一目标来模拟多个表情会高估 gzip 压缩收益，不能当作真实表情的成本。

## 赛璐璐观感的评估

**编辑器采用赛璐璐显示。** 页面的 Toon 切换（`web/runtime/toon.mjs`，约 170 行）已经接近 `review` 的 toon 视图，同样的数值：`SHADE` 和 `LIT` 的比值、`TERMINATOR`、`KEY`（跟随相机）、`OUTLINE_PX`、`OUTLINE_BEHIND`。不引入 VRM 依赖，没有新的 npm 包。

**做法：**

- **材质：** `MeshToonMaterial`，把它的渐变查表换成带约 1 像素 `smoothstep` 的硬阶（`onBeforeCompile` 约 5 行）。用渐变图也能做到，但那是着色器里的边，多重采样抗不了锯齿；`review` 靠 16 次 TAA 取样。
- **颜色来自文件：** 皮肤和部件都用文件里的颜色，所以鼻和耳与皮肤的亮面、暗面都一样，`runtime_check.py` 检查赛璐璐下皮肤和同色部件是同一个颜色。
- **带色调的阴影：** `MeshToonMaterial` 的阶只缩放光强，所以环境光给 `SHADE/LIT`，主光给剩下的。材质亮面是文件的颜色，暗面是它乘以 `SHADE/LIT`。
- **描边：** 反面外扩的外壳，和 `render.py` 相同，但在顶点着色器里按深度算宽度，所以始终约 2.5 像素，跟随蒙皮和变形目标。没有用 three.js 的 `OutlineEffect`：看它的代码，外扩是在裁剪空间里做的，非正方形画面上横竖线宽会不同（只读了代码，没有运行对比）。

**和 `review` 的差距：**

- **色调：** 有意不同。`review` 是中性的灰模（亮 239,236,234，暗 170,170,184），不看部件，只用来判断形体，`blender/review/render.py` 没有变。页面是文件的皮肤色（亮 231,221,215，暗 164,158,169）。描边相同（39,39,48）。
- **明暗线：** 位置基本相同，页面上已经没有沿 UV 接缝的折痕。前臂背面的两块亮斑、臀部的亮块和大腿内侧的细亮线在 `review` 上也有，来自几何。
- **没有验证：** 没有逐像素量化；只比较了 `rest` 的四个视角；相机只在两边都能复现的取景下比较；真实 MToon 的 `shadingShift`、`toony` 等参数没有逐项对应，也没有在 VRM 运行时里看过。

**代价：** 描边会对皮肤和启用描边的部件多画一遍外壳，增加绘制与三角形开销。当前每帧和首帧成本需复测；其他 GPU 尚未验证。

**给编辑器的注意点：** `build/bodies/<预设>/body.glb` 默认带部件，`loadBody` 的 `mesh` 还是皮肤，`parts` 是部件。`makeToon` 打开时复制材质的颜色，皮肤不再是 `LIT`。主光跟随相机，和 `review` 一致，但这只是判断观感的约定，产品里要不要固定光源另说。`Material.outline` 决定部件是否描边，薄壳不描边。透明部件的外壳没有处理，所以不要给 `BLEND` 部件设 `outline`。

## VRM 1.0 差距

**写文件用 [VRM Add-on for Blender](https://github.com/saturday06/VRM-Addon-for-Blender)（[决定](../DECISIONS.md#绑定与动作)）。** 下面列出导出准备工作与现有驱动的表达限制。

本工具准备 T 姿势、人形骨映射、辅助骨约束近似和表情名，再交给插件写文件。插件由下游自行安装，不随本工具分发。

尚未用当前身体走通真实文件，也未核对换基准后的 `MCH-` 骨和 `rest.*` 键如何被处理；分发前须重新核对插件许可。既有最小扩展试验不代表完整交付验证。

| 差距 | 现状 | 要做什么 | 归属 |
|---|---|---|---|
| **T 姿势（最大）** | GLB 使用 A 姿势，手臂离竖直 35°；VRM 需 T 姿势并核对手掌、手指与腿足朝向 | 见下 | 已定（见下），涉及贴合、绑定、导出 |
| 肩胛联动和扭转 | VRM 约束表达不了，见上表 | 半转骨和前臂用约束，肩胛联动要写进动作数据，或在 VRM 里放弃 | 绑定、导出 |
| `meta` | 没有 | `name`、`authors`、`licenseUrl` 必填，还有一组默认从严的授权字段；已定默认填最宽松的选项（[开源和授权](../DECISIONS.md#产品与范围)） | 导出 |
| 表情 | 文件里有 `blink`、`happy`、`aa` | 其余预设表情都可选 | 头部 |
| 视线、眼骨、下颌 | 有 `leftEye`、`rightEye`、`jaw` 骨。VRM 的人形骨只传旋转，下颌的前滑在 VRM 里没有通道，要约束或放弃 | `lookAt` 用骨骼类型；都可选 | 头部 |
| 第一人称 | 身体和头是同一个网格，五官、头发是单独节点 | 可选。部件标 `thirdPersonOnly`，皮肤可能要拆网格或依赖播放器自动处理，没有验证 | 导出 |
| 材质 | PBR 材质，部件各一个。页面的赛璐璐不依赖 VRM（见上） | MToon 可选；贴图等 UV | 贴合、导出 |

**人形骨映射：** 按[骨骼约定](../reference/skeleton.md)建立映射；半转骨、扭转骨等辅助骨不当作人形骨。映射、朝向和静止姿势需在 VRM 导出及目标播放器中重新验证。

**T 姿势的影响：** VRM 人形动作以 T 姿势为基准，直接用 A 姿势会产生手臂偏移。导出时换基准可保持生成流水线的固定 A 姿势；若改模板本身，则贴合、权重、驱动和检查姿势都需要重新验证，因此未采用。

**当前交付：** GLB 和 `.blend`。采用导出时换基准：`.blend` 已能把 T 姿势定为静止姿势，约束换基准，手转到手掌朝下（[换静止姿势](blend.md#换静止姿势)）；GLB 仍是 A 姿势，VRM 导出本身还没做。

## 给绑定的接口

两张驱动表都由 Python 写、三边各读一份：`extras.sotaigen.drivers` 由 `driver_table` 从 `HALF_BONES`、`HALF_ROLLS`、`TWISTS`、`SHOULDER_RHYTHM`、`FORWARD_LIFT` 和骨架记录的 `glides` 写成，`extras.sotaigen.correctives` 由 `sotaigen.corrective.recipe.Correctives.groups` 写成，读它们的是 `web/runtime/driver.mjs`（`poseQuaternions`、`poseTranslations`、`correctiveWeights`）和 `blender/src/sotaigen_blender/rig.py`（`couple`、`correctives`），Python 自己用 `sotaigen.rig.skinning.sided` 和 `sotaigen.corrective.drive.weights`。

两张表的键和读法见 [GLB](../reference/glb.md#辅助骨的驱动表)。实现上要记住的：

- **权重的乘法顺序：** 淡出按骨逐根相乘，再按骨第一次出现的顺序相乘，三边按同样的顺序乘，所以 Python 和 JS 逐位相同。
- **实验的 IPC 肩部表：** 只在 `body-ipc.glb`，组带 `"kind": "simplices"` 和它自己的单纯形（`bone`、`scales`、`sampled`、`domain`、`points`、`cells`、`origins`、`inverse`、`twistLimit`），不进约定；`.blend` 不读它。
- **加一组修正**只是多一组表，三边的代码不用改。

下颌是唯一除了转动还会移动的骨：运行时用 `poseTranslations`，`baked_motion` 给它写平移通道，没有驱动的播放器（`setNamedOnly`）不滑。改上面这些常量的数值，导出和运行时会自动跟上。改 `sided` 的结构（比如加胯部联动）会让 `test_the_driver_in_javascript_is_body_rig_sided` 失败，这时要同步改 `driver.mjs` 和 `blender/src/sotaigen_blender/rig.py`；表的结构变了要升 `extras.sotaigen.version`。
