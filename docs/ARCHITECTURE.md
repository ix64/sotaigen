# 参数化人体工具：架构

本文只放稳定的全局契约：目标、流水线、数据约定和不变量。决定见 [DECISIONS.md](DECISIONS.md)，协作方式和命令见 [WORKFLOW.md](WORKFLOW.md)，各部分的现状和结论见 [README.md](README.md) 的索引。

## 1. 目标与边界

**目标：** 输入一组语义参数，输出可绑定、可驱动的动画风格人体基础模型，并能复用到不同角色。

**产品形态：** 在浏览器里调参，松手后由本地服务生成；导入 JSON 参数、导出模型。

**速度指标：**

- **调参：** 松手后 10 秒内看到这组参数的造型。没见过的参数也要做到，不能靠缓存达标。
- **导出：** 用户选定造型后，10 分钟内导出这组参数的模型。导出时可选的 IPC 精确接触不受这一条限制，页面要标出预计耗时。
- **怎么量：** 在基准测试机器（Apple M5）上，从松手或点导出算到结果可用。超出就是缺陷，不是可以接受的等待。

**形体设计依据。** 骨性地标、关节运动和肌群的位置关系取自人体解剖、影像测量及生物力学研究；胸部等软组织另参考组织解剖与表面形态研究。把这些关系简化为动画体块时，应说明保留了哪种结构、用哪个网格量近似它。参数的支持范围通过多体型、多姿势测试确定。具体角色只在下游项目里用来验证表达能力与适配，尺寸和衣物约束留在下游。

**判断标准：** 主要用途是动画风格虚拟角色，所以好坏看动画观感，最终由用户的眼睛判断。数字用来诊断问题、防止退化。

**下游用途：** 下游先在 Blender 里完善建模，再用于三类场合：

- **游戏里的角色交互：** 开放世界和视觉小说的角色扮演。动作以预先做好的动画为主，加上运行时的 IK 和动画混合；开放世界看全身的大范围动作，人多时要 LOD，视觉小说看上半身和特写。
- **2D 立绘的姿态参考：** 防止图像模型生成不符合规律的姿态。离线摆姿势，可以慢；看手、接触和各视角下的轮廓，关节极限要严格。
- **虚拟主播直播：** 实时动作捕捉，角度不受控；看上半身、肩和靠近脸的手；用 VRM 1.0，直播软件不执行自定义代码。

三者都要求绑定在 Blender 里能用。本工具在 Blender 中检查交付质量，游戏引擎里的还原由下游做；VRM 导出计划支持 1.0，尚未实现。产品取舍见[用途](DECISIONS.md#产品与范围)、[交付绑定与动作](DECISIONS.md#绑定与动作)和[局部加密](DECISIONS.md#形体)。

**动作的依据：** 优化某个部位的运动，先弄清肌肉和皮肤在这个动作里该怎么变，写成 [动态解剖卡](motion/README.md)，由用户定下保留、夸张、简化还是舍弃，再选技术。下面的不变量检查几何与数据一致性，动态解剖卡用于验收动作现象。

**不做：**

- 写实人体。
- 任何具体角色的数据。本工具不引入下游项目的代码和依赖，不携带任何角色的参数或网格；下游的经验可以参考，具体数值不能带过来。
- 服装设计。

## 2. 规模

当前拓扑规模见[坐标和拓扑](reference/coordinates-topology.md)，骨骼清单见[骨骼](reference/skeleton.md)。这里不重复维护数量。

## 3. 架构

### 3.1 流水线

`uv run sotaigen-dev build` 按下图顺序运行，并按输入内容的指纹跳过没变的阶段（见 [WORKFLOW.md](WORKFLOW.md)）。

```text
  sotaigen/data/bodies/<性别>/<typical|test>/*.json  规格：毫米，每项有量法
                    │ solved_controls    sotaigen/params/spec.py、sotaigen/form/solve.py、sotaigen/form/measures.py
                    │ 内部控制量 p（T 份额）
                    ▼
        landmarks / calibrated           sotaigen/form/anatomy.py
                    │ 地标 L + 隐式场 Body
                    ▼
   sotaigen/data/template/base-mesh.json ──► 贴合       sotaigen/form/wrap.py
   (固定拓扑)               │ 地标变形 → 投影到曲面 → 解折叠
                            │ 手脚整体缩放 → 颈部开口就位
                            ▼
                  build_head             sotaigen/form/head.py
                            │ wrapped.npy：模板顶点 + 头部顶点
                            ▼
          skeleton / skin_weights        sotaigen/rig/skeleton.py
                            │ skeleton.json + weights.npz（参考皮肤上固定）
                            ▼
                  静态腹股沟造型          sotaigen/form/groin.py
                            │ rest.npy：成品静止皮肤
                            ▼
                  姿势目标 + 蒙皮         sotaigen/corrective/axilla.py、sotaigen/corrective/groin.py
                            ▼
                  pose_verts + POSES     sotaigen/rig/skinning.py、sotaigen/rig/poses.py
                            │ poses/<姿势>.obj
                            ▼
                  指标                    sotaigen/metrics.py
                            │ poses.json、joints.json
                            ▼
                  渲染和拼图              sotaigen_dev/review.py、blender/review/render.py
                            │
                    导出 GLB              sotaigen/output/glb.py
```

### 3.2 分层

| 层 | 职责 | 模块 | 关键决定 |
|---|---|---|---|
| 规格 | 用户和文件看到的量，换算成内部控制量 | `sotaigen/params/spec.py`、`sotaigen/params/rules.py` | 长度相加得到身高，身高和头身比是读数 |
| 求解 | 由目标实测量反求内部控制量 | `sotaigen/form/solve.py`、`sotaigen/form/measures.py` | 到不了的目标照实报告，不改用户的数 |
| 比例 | 由参数算出骨骼和标志点 | `sotaigen/form/anatomy.py` 的 `landmarks` | 长度以躯干长 T 或头高为单位 |
| 体块 | 依附骨骼的椭球和圆锥，平滑融合成隐式曲面 | `sotaigen/form/anatomy.py`、`sotaigen/form/field.py` | 三次平滑融合，C2 连续，着色不露边 |
| 校准 | 求内部偏移，让实测腰宽、胯宽、头顶、下巴等于参数 | `calibrated`、`head_field` | “参数即结果” |
| 皮肤 | 固定拓扑的模板贴到曲面 | `sotaigen/form/wrap.py`、`sotaigen/form/foot.py` | 手脚保留模板网格，不走投影；脚先按解剖结构做局部位移，再就位 |
| 头部 | 函数生成，固定拓扑，缝到颈部开口 | `sotaigen/form/head.py` | 函数生成，不改造 CC0 头模 |
| 绑定 | 从地标放骨骼，从模板分段和比例推权重 | `sotaigen/rig/skeleton.py` | 半转骨、扭转骨、肩胛联动都在这一层；腰以下权重默认经过拟合 |
| 蒙皮 | 骨骼名、姿势到骨骼矩阵、线性混合蒙皮、delta mush | `sotaigen/rig/skinning.py` | 只摆姿势、不造绑定的模块（姿势目标、接触烘焙、指标）只依赖这一层，改权重代码不会让它们的缓存失效 |

**包的划分：** `sotaigen/` 分六个子包，导入只向下走，`tests/test_default_path.py` 守住其中接触那一条。

| 子包 | 内容 | 可以导入 |
|---|---|---|
| 根下 | `mesh`、`template`、`uv`、`sources`、`paths`、`errors`、`diagnostics`、`body`（各层共用）；`metrics`、`mend`、`pipeline`、`generation`、`api`、`cli`（在各层之上） | 共用的只用 numpy 和 scipy |
| `params` | 规格表、规则、参数文件 | 共用模块 |
| `form` | 体块、轮廓层、求解、贴合、头、腹股沟的静态造型 | `params` |
| `rig` | 骨骼、蒙皮、姿势、关节极限、静止姿势、触点 | `params`、`form` |
| `corrective` | 肩腋、腹股沟、垂臂、胸肌、三角肌的姿势修正和驱动，配方 `recipe` | 以上各层 |
| `contact` | IPC 接触表的求解、烘焙、审计和缓存，实验特性，要 `sotaigen[contact]` | 以上各层；只有 `recipe` 和 `output/quality` 在要接触表时从函数里导入它 |
| `output` | GLB、`.blend`、导出分级、五官部件、换头 | 以上各层 |

每个部位一个词，贯穿模块名：肩腋 `axilla`，腹股沟 `groin`（和参数 `groinDepth`、`groinRelief` 同词；`pelvisLean` 说的是骨盆本身），垂臂 `hang`，胸肌 `pectoral`，大腿 `thigh`。GLB 驱动表的组名也是这些词（`axilla.left`、`groin`、`hang.left`）；修正目标名不透明（[GLB 约定](systems/export.md#glb-约定)），腹股沟的目标名仍以 `pelvis` 开头，不为好看改名。`sotaigen_dev/` 是开发工具（`compare`、`displacement`、`review`、动作卡和动作库、压力测试、编辑器的规则生成和工作进程），不进发布的包；`dev/` 提供本地安装的 `sotaigen-dev` 命令入口，由开发依赖引用，转调这些模块；`scripts/` 只留彼此不导入的单文件诊断。

**公开入口：** `import sotaigen` 的名字都在 `sotaigen/api.py`：`read_spec`、`typical` 得到参数（`Spec`），`check` 列问题，`generate` 得到身体（`Body`，构建读出的身体也是它），`write_glb`、`write_blend` 写文件，错误都继承 `SotaigenError`。命令行 `sotaigen`（`sotaigen/cli.py`）、`uv run sotaigen-dev export`／`blend` 和编辑器的工作进程都调这两个写函数。包数据（模板、典型和测试身体、关节极限、滑块定义）在 `sotaigen/data/`；`.blend` 要 Blender 和 GPL 的包 `sotaigen-blender`（`blender/src/sotaigen_blender/`），`sotaigen` 只找到它的目录、在 Blender 里运行，不导入它；IPC 接触表要可选安装 `sotaigen[contact]`。

### 3.3 为什么是这个结构

- **隐式曲面解决衔接：** 肩、胯这些分支处不用手工建模，融合自动产生。
- **固定拓扑解决复用：** 所有体型顶点顺序相同，所以权重方案、UV、变形目标和服装转移都有条件共用一份。这是全系统最重要的约定。
- **规格与内部分开：** 文件保存实际尺寸和声明过的无量纲量，体块内部用 T 份额；仍属内部控制的例外见[参数](PARAMETERS.md#还不是成品尺寸的规格)。
- **截面驱动已被否定：** 它只证明不破面，没有形体结构，也无法复用。不要退回逐层填截面的做法。

### 3.4 数据约定

改动下列约定会影响所有模块：在报告里提出，改写本文，并在提交说明里写明影响。其中对下游冻结的部分（坐标、拓扑和顶点顺序、骨骼、参数文件、GLB 和 `.blend` 的字段）以[对外约定](reference/README.md)为准，这里只写模块之间怎么用它们；改那部分还要按[版本政策](reference/README.md#版本)升版本。

**坐标：** Z 是离地高度，前方是 -Y，穿着者左侧是 +X，单位 mm。身体左右对称（[坐标和拓扑](reference/coordinates-topology.md#坐标)）。

**参数：** 身体在 `sotaigen/data/bodies/<性别>/typical/` 和 `test/`，每个身体一份标准参数文件 `<组>.json`，名字是 `性别-组`（如 `female-long`），每个都是一份完整的规格。典型身体是网页的起点，测试身体只给测试用（见[身体](systems/presets.md)）。`sotaigen/form/bodies.py` 的 `load_specs(role)` 返回规格，`load_bodies()` 返回内部控制量（读到某个身体时才求解），并都按 `sotaigen/params/rules.py` 校验，坏参数报错。范围和规则直接读规格（`proportions`），是纯算术，页面每次拖动都要算；真正生成身体用 `solved_controls`。参数文件的结构和参数名见[参数文件](reference/parameters.md)，量法和读数的含义见 [PARAMETERS.md](PARAMETERS.md)；范围和规则不在对外约定之内，随形体改进，见[参数与预设](systems/params.md)。求解之后的代码只见内部控制量。

**地标 `L`：** `landmarks` 或 `calibrated` 返回的字典，X 是身体上的真实位置。曲面和测量用 `calibrated` 的 `L`。骨骼的地标统一从 `sotaigen.pipeline.rig_landmarks` 取，不要绕过它自己取；胯、膝、踝的 X 由 `sotaigen.rig.skeleton.skeleton` 从贴合后的网格量；腰腹与大腿的局部实测控制先还原参考皮肤，骨骼与权重都在该参考皮肤上求取，以保持同一骨架。

**隐式场：** `Body(points)` 接受 N×3 数组，返回值小于 0 表示在内部，但不是严格的距离场，要用距离时除以梯度模长。

**模板：** `sotaigen/data/template/base-mesh.json`，由 `sotaigen.template.load_template()` 读取，来源、字段和重新提取的方法见 `template/PROVENANCE.md`。字段有 `format`、`version`，`verts`、`faces`、`segments`（解剖分段），`rings`（关节环的顶点序号，只有左侧）、`openings`（颈部开口，即颈环），`landmarks`（拓扑里找不出的模板事实：上臂转心、腋部的顶点、脚的两个比例）。区域（`trunk`、`neck`、`leg`、`arm`、`hand`、`foot`）由分段推出，`load_template` 读入时补上 `regions`。加了环的模板另有 `refined`：`extract.py` 在哪些关节环旁加了几条环（`besideRings`）、新顶点从哪个序号起（`from`）、每个新顶点平分的那条边的两端（`parents`，后加的环可以平分先加的环的边）、加环前的面（`faces`）、每批加了多少个顶点（`batches`）。`sotaigen.template.coarse` 取回加环前的网格，`prolonged` 把上面的量插值到新顶点；只按边数计算、要保持毫米尺度的步骤（现在是权重）用这两个函数。

**成品皮肤：** `wrapped.npy` 保留贴合结果；`sotaigen.form.groin.rest_verts` 在同骨架上补入已选静态斜沟与相邻体块，另存 `rest.npy`。`rigged().verts`、编辑器和导出都使用 `rest.npy`。姿势目标在蒙皮前叠加；Python、网页驱动和 GLB 动画共用同一姿势网格。腹股沟尚未解决的技术限制见[腹部—大腿过渡](parts/groin.md#生产接入与技术检查)。

**网格顺序：** `wrapped.npy` 有 12676 个顶点，所有身体顺序相同（[拓扑](reference/coordinates-topology.md#拓扑)）：源网格的 8781 个，头部的 3313 个，再按批次接加的环，肘、膝 400 个，肩、腋 182 个。编号只追加：以后加的环排在最后，已有的序号都不变，头和颈环的序号也不随加环改变（[编号](../template/PROVENANCE.md#编号)）。`sotaigen.form.head.extended(base)` 返回带头部的模板，`sotaigen.form.head.template_rows` 从整张皮肤里取回模板的那些行。

**骨骼：** 骨架是 `sotaigen.rig.skinning.Skeleton`：按权重列顺序的骨骼，另记躯干长 `T`、`jaw` 的前滑 `glides` 和身体自己的腹股沟量 `groin`，存在 `skeleton.json` 的顶层，不放在骨骼的字典里。名字沿用 VRM humanoid，`left` 指穿着者左侧（[骨骼](reference/skeleton.md)）。局部 Y 沿骨骼，局部 Z 朝前，局部 X 是弯曲轴，肢体向前为正。每侧的辅助骨 `PalmCup` 位于手掌与无名指、小指指根之间，绕中指方向收拢掌缘。右侧姿势由左侧镜像，规则是 `(x, -y, -z)`。姿势可带 `rootTranslation`（项目世界轴、毫米），未提供时为零；每体型的接地下蹲由 `sotaigen.rig.poses.check_poses` 生成。

**权重：** 顶点数 × 骨骼数的数组，每顶点最多 4 个影响，和为 1。截断必须连续，不能直接丢弃（见 `limit`）。腹前弧、后腰凹入与大腿外侧弧度等局部轮廓先还原参考皮肤：先用 `sotaigen.form.anatomy.skin_reference(p, verts)` 还原参考皮肤，再计算骨骼与权重，应用到成品顶点。构建和编辑器共用这条路径，避免未移动的腹股沟权重被局部表面改动扰动。`upperThighGirth` 反求基础体块半径，不是这类固定骨架的局部轮廓；区别见[大腿控制](parts/hip-thigh.md#上胯到大腿过渡算法与两个实测控制)。

### 3.5 不变量

下列条件由测试把关，改动不得破坏。测试不允许跳过：`tests/conftest.py` 把任何 skip 和 xfail 都变成失败，已知缺陷写进领域文件。

| 不变量 | 范围 | 测试 |
|---|---|---|
| 实测腰宽、胯宽与参数相差不超过 2% | 全部身体 | `tests/test_body.py` |
| 隐式场左右对称，腿在裆部分开 | 全部身体 | `tests/test_body.py` |
| 贴合 0 翻转，0 个顶点越过中线 | 全部身体 | `tests/test_wrap.py` |
| 脚底落在地面，容差 0.5 mm | 全部身体 | `tests/test_wrap.py` |
| 足部塑形不改脚长、最低点和踝环，足弓内侧抬起、外缘贴地 | 模板 | `tests/test_foot.py` |
| 皮肤闭合，每条边恰好两个面 | 全部身体的共同模板 | `tests/test_head.py` |
| 头顶、下巴落在标志高度 | 典型身体 | `tests/test_head.py` |
| 转头、点头无塌陷面 | 典型身体 | `tests/test_head.py` |
| 权重归一，影响数不超过 4，静止姿势不移动顶点 | 全部身体 | `tests/test_rig.py` |
| 抬臂无塌陷面，手腕滚转后前臂粗细不低于 95% | 全部身体 | `tests/test_rig.py` |
| 下蹲时腿不穿过中线 | 全部身体 | `tests/test_rig.py` |
| 屈肘时肘尖离弯曲轴的距离不低于静止时的 98%，量源网格的顶点 | 全部身体 | `tests/test_motion_cards.py` |
| 膝截面不碰中线，成对的地标落在体块中线上 | 全部身体 | `tests/test_body.py` |
| 手腕折角不超过 35°、脚踝不超过 16°，接缝偏差分别不超过肢体末端半径的 0.65、0.52 | 全部身体 | `tests/test_wrap.py` |
| UV 无重叠、无翻转，拉伸在上限内 | 全部身体 | `tests/test_uv.py` |
| 导出的 GLB（含部件和变形目标）与 NumPy 蒙皮一致，驱动与 `sotaigen.rig.skinning.sided` 一致 | 典型身体，浏览器版本见 `pnpm run runtime-check` | `tests/test_export.py` |

五官、耳和头发是单独的部件，顶点数跨身体相同，贴在皮肤上，转头、点头时跟随（`tests/test_face.py`）。转头、点头和头部地标目前只测典型身体。下蹲以外姿势的自交只列出，不把关，因为接触是合法的。
