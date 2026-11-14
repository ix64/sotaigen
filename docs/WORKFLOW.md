# 协作和命令

## 1. 开工

每项工作在自己的 worktree 和分支里做，基线是最新 `main`。每个 worktree 要自己装环境，否则导入的会是别处的 `body`：

```sh
uv sync --all-extras
```

安装 [uv](https://docs.astral.sh/uv/)，版本取 `pyproject.toml` 的 `tool.uv.required-version`（别的版本会拒绝运行）。uv 按 `.python-version` 找或下载 Python；`--all-extras` 包含实验性 IPC 依赖，开发环境也会装上 GPL 的 `sotaigen-blender` 包（`blender/`）。

Node 推荐用 [fnm](https://github.com/Schniz/fnm) 管理，按它的安装说明启用 shell 集成，然后在仓库根目录运行：

```sh
fnm install && fnm use
npm install --global "$(node -p 'require("./package.json").packageManager')"
pnpm install
```

`fnm install` 和 `fnm use` 读取 `.node-version`；pnpm 版本只在 `package.json` 的 `packageManager` 里维护。页面、类型检查、浏览器检查和标记 `node` 的 Python 测试需要 Node；只有前端流程需要安装 JS 依赖。Python 的仓库开发命令用 `uv run sotaigen-dev <命令>`，`--help` 列出命令，`<命令> --help` 查看选项；下游命令仍是 `uv run sotaigen <命令>`。前端多步流程由根目录 `package.json` 的 `scripts` 定义，通过 `pnpm run <名称>` 执行，附加参数直接接在命令后。

渲染还需要 Blender，查找顺序是环境变量 `BLENDER`、`PATH` 里的 `blender`、macOS 的默认应用位置。没有 Blender 时不能出图和写 `.blend`，构建不受影响；测试里要 Blender 的用例（标记 `blender`）离开本次运行，结尾说明留下了几个。

参数文件的校验和生成通过 `uv run sotaigen`（`check`、`glb`、`blend`、`swap-head`），由 `pyproject.toml` 的 `project.scripts` 定义；也能不靠检出、装成包运行，退出码见 `sotaigen/cli.py`：0 完成，1 身体不行，2 命令行或文件不对，3 缺 Blender 或 `sotaigen-blender`。

## 2. 命令

`sotaigen-dev` 是 `dev/` 中的本地命令包，由 `uv sync` 随开发依赖安装；它不进入对外发布的包，也不依赖 Node。命令在该检出的根目录执行，路径参数按这个根目录解释。附加参数直接接在子命令后，不用额外的 `--`，如 `uv run sotaigen-dev build female-long --jobs 2`。

`test`、`test-merge`、`test-full` 决定测试层级，默认带 `-n 4 --dist loadfile`；后面的 pytest 参数可以覆盖它们，如 `uv run sotaigen-dev test tests/test_review.py -n 0`。检查收集结果也用 `-n 0`，避免并行测试的预构建。`lint --docs-only` 只检查文档。

| 命令 | 作用 |
|---|---|
| `uv run sotaigen-dev test` | 日常跑：不选中标记 `slow` 和 `bake` 的用例，也不选中需要启动 Blender 的 `blender` 用例，结尾说留下了几个。点名文件或用例时（`uv run sotaigen-dev test tests/test_blend.py`），其中的慢用例照跑。用 pytest-xdist 分 4 个进程跑，一个文件整个给一个进程；开跑前主进程先把全部身体建到最新，进程之间只读 `build/`。求解缓存有结果时约 5 分钟，改了求解器后第一次约 9 分钟（M5） |
| `uv run sotaigen-dev test-merge` | 合并前跑：只不选中 `bake`，其余全跑，用 `SOTAIGEN_TESTS=merge`。约 3700 个用例，约 18 分钟（M5）。分层只把用例挪到后面跑，不减少覆盖。新用例累计超过约 25 秒，或者启动 Blender，就标 `slow` 或 `blender` |
| `uv run sotaigen-dev test-full` | 连 `bake` 用例一起跑（`SOTAIGEN_TESTS=full`），没烘过的身体会逐个烘焙，只在要验证接触表时跑 |
| `uv run sotaigen-dev lint` | ruff 检查，加文档检查（`sotaigen_dev/check_docs.py`：过期标记、失效的相对链接和锚点）。不需要安装或构建前端；正文里的已知生成目录 `web/node_modules/`、`web/src/paraglide/` 不要求存在，源码路径和 Markdown 链接仍检查 |
| `uv run sotaigen-dev build [身体 ...] [--sex female] [--group long,balanced] [--stages wrap,rig,pose,metrics] [--force] [--jobs 4]` | 贴合、绑定、姿势、指标，写到 `build/bodies/`。默认全部身体，同时做 4 个；机器被别的分支占满时用 `--jobs` 调小。渲染不是构建阶段，`review` 按需渲染并按内容缓存 |
| `uv run sotaigen-dev export [身体 ...] [--no-parts] [--contact]` | 导出 `build/bodies/<身体>/body.glb`，默认带五官、耳、占位头发和平滑法线。姿势修正不烘焙：肩腋、腹股沟是雕塑修正，垂臂是隐式场修正（`sotaigen/corrective/hang_field.py`），几秒内算完；`--contact` 改为烘焙 IPC 接触表，每个没烘过的身体十几到几十分钟，写到 `body-ipc.glb`。网格损坏（翻转、越过中线、皮肤不闭合）时拒绝，不写文件，也不烘焙，以 1 退出；其余质量问题写进文件，见[导出分级](systems/export.md#导出分级) |
| `uv run sotaigen-dev blend [身体 ...] [--sex female] [--group long,balanced] [--t-pose] [--arm-abduction 度] [--ankle-spacing 毫米] [--palms-down] [--straight-arms] [--past-limits] [--out 文件]` | 写 `build/bodies/<身体>/body.blend`：绑定好的 `.blend`，控制骨、约束、修正形状、关节极限、五官部件和细分曲面，见 [blend](systems/blend.md)。要 Blender；网格损坏（翻转、越过中线、皮肤不闭合）时拒绝，并说明部位和退回哪项参数能修好，每试一项约 8 秒，见[哪项参数](systems/export.md#哪项参数)。给了静止姿势的参数时写 `body-rest.blend`，见[换静止姿势](systems/blend.md#换静止姿势)。参数文件用 `sotaigen blend FILE.json OUT.blend [同样的参数]`，编辑器的导出菜单也能导出（只有 A 姿势） |
| `uv run sotaigen swap-head BODY.blend HEAD.blend OUT.blend` | 把 `.blend` 的头换成外部的头，融合接缝、转移权重，换完按新头摆转头、点头、张嘴检查，约 8 秒，不烘焙，见[换头工具](parts/head.md#换头工具) |
| `uv run sotaigen-dev motion-blend run <身体 ...> --clip walk,wave` | 把真实动作打成 `.blend` 控制骨的关键帧，读回对照运行时，写 `build/reports/motion-blend/`；`sheet` 出图，`film` 渲染视频，不烘焙。见 [blend](systems/blend.md#真实动作) |
| `pnpm install` | 按 `pnpm-lock.yaml` 装页面的 JS 依赖到 `web/node_modules/`，依赖声明变化时会更新锁文件；CI 会检查锁文件是否与依赖声明一致。页面、类型检查和浏览器检查都要先装，`uv run sotaigen-dev test` 不需要 |
| `pnpm run runtime-check [身体 ...]` | Validator 和真实 Chrome 里的一致性检查，会先构建页面，要先 `pnpm install` |
| `pnpm run editor [端口]` | 构建页面并启动编辑器，所有页面都在这一个端口上。先运行 `pnpm install`。松开滑块后，本地 Python 服务生成用于显示和导出的 GLB。 |
| `pnpm run editor-dev [端口]` | 同上，但由同一个端口上的 Vite 提供页面和热更新，改页面时用 |
| `pnpm run web-build` | 只构建页面到 `build/web/app/` |
| `pnpm run web-check` | 页面的检查：类型、构建、Playwright（替身工作进程的界面用例和真实生成的冒烟用例），约 1 分钟，要系统 Chrome，不进 `uv run sotaigen-dev test` |
| `pnpm run editor-sweep [--name 身体] [--sliders a,b]` | 每个滑块在全部身体上的可用范围，和两端真实生成的身体与对比图，写 `build/reports/editor-sweep/`。形体改了要重跑 |
| `pnpm run typecheck` | `web/` 的类型检查，编辑器是 TypeScript，`web/runtime` 里的 `.mjs` 靠 JSDoc，不进 `uv run sotaigen-dev test`，先 `pnpm install`，约 1 秒 |
| `uv run sotaigen-dev review sheet ...` | 组合拼图，见第 4 节 |
| `uv run sotaigen-dev review render <身体 ...>` | 渲染单张 |
| `uv run sotaigen-dev review verify [--allow-stale]` | 从图上量像素，验证比例没有失真 |
| `uv run sotaigen-dev compare --against <另一个 build 目录> [身体 ...] [--all] [--allow-stale]` | 对比两份构建，见 [MERGING](MERGING.md) |
| `uv run sotaigen-dev displacement --against <改前 build> [身体 ...] [--build <改后 build>] [--pose a,b] [--images rest\|all\|none\|a,b]` | 两份构建逐顶点的位移：`build/reports/displacement/` 下的 `displacement.json`、`summary.txt` 和热图，见 [MERGING](MERGING.md) |
| `uv run sotaigen-dev stress oat\|random\|spec\|report ...` | 参数压力测试，定范围用，形体稳定后要重跑；各子命令用 `--help` 查看参数 |
| `uv run sotaigen-dev motion sweep <卡片> <身体 ...> [--against DIR]` | 按动态解剖卡看关节：角度序列、弯曲平面截面、皮肤相对肢段的抬升和滑动，写 `build/reports/motion/`，见 [docs/motion](motion/README.md) |
| `uv run sotaigen-dev motion folds <卡片> <身体 ...> [--against DIR]` | 把塌陷面分成卡片预期的折痕内外，只报告，不影响 `compare` 的退出码 |

诊断脚本用 `uv run <脚本路径>.py` 运行，例如 `uv run scripts/foot_study.py`；表中的路径都相对仓库根目录。已有 `sotaigen-dev` 子命令的工具使用上表入口。

| 脚本 | 用途 | 相关部分 |
|---|---|---|
| `scripts/check_targets.py` | 逐身体对比参数要求和实测的腰、胯宽，加贴合摘要 | 形体 |
| `scripts/silhouette.py` | 正、背、侧轮廓的极值，用来找鼓包 | 形体 |
| `scripts/abdomen_study.py --meshes` | 腹前弧、后腰凹入的实测目标与网格／姿势检查 | 形体 |
| `scripts/thigh_profile_study.py [身体 ...]` | 实测外侧弧与大腿围的组合端点、网格／姿势与 UV 检查 | 形体 |
| `scripts/sample_sheet.py` | 看随机样本 | 参数 |
| `scripts/foot_study.py [身体 ...] [--against DIR] [--views inner,under]` | 单独看左脚的内侧和脚底，打印足底、足背轮廓和足弓指数 | 形体 |
| `scripts/terminator.py` | toon 明暗交界线落在哪里 | 形体 |
| `scripts/locate.py` | 翻转和曲面误差出现在哪个分段、什么高度 | 贴合 |
| `scripts/raw_info.py` | 模板自身的包围盒和关节位置 | 贴合 |
| `scripts/joint_loops.py` | 各关节的边环数和弯曲时的折角 | 贴合 |
| `scripts/uv_check.py` | UV 的重叠、翻转、拉伸和密度 | 贴合 |
| `scripts/head_review.py` | 带五官和头发的头部拼图，`review` 还不支持部件；`--only jaw,gaze` 看张嘴的各档和视线 | 头部 |
| `scripts/head_swap_study.py [身体 ...] [--coarse 身体,身体]` | 用测试头走一遍换头，和原文件对照姿势，写 `build/reports/head-swap/` 的文件、报告和 `sheet.png`，默认 3 个身体，不烘焙 | 头部 |
| `scripts/clean_range.py measure\|summary\|sheets\|sheet` | 在典型身体上逐关节扫角度，量干净范围并出看图拼图，写 `build/reports/clean-range/` | 绑定 |
| `scripts/upper_body_study.py poses\|sheet\|weights\|section\|profile` | 上半身目标角度和复合姿势；`--variant` 换机制对照（关修正、改联动、改权重的原型，不改 `sotaigen/`），`profile` 把几种的正面顶轮廓叠在一张图上，写 `build/reports/upper-body/` | 绑定 |
| `scripts/touch_study.py [身体 ...] [--no-sheets]` | 手到头、手到脸的触点在典型身体上的解和穿插，写 `build/reports/touch/report.json`，拼图写 `build/reports/upper-body/sheets/` | 绑定 |
| `sotaigen_dev/motion/clips.py` | 取得和回放真实动作片段；`contacts` 测接地和脚滑，`glb` 输出带动画的 GLB | 绑定 |
| `scripts/rest_pose_study.py measure\|sheet` | 换静止姿势的文件在 12 个典型身体上和 A 姿势的差多少、出图，写 `build/reports/rest-pose/`，见[换静止姿势](systems/blend.md#换静止姿势) | 绑定 |
| `scripts/groin_strength_review.py <身体 ...>` | 静态腹股沟五档强度对比，站姿和屈髋，写 `build/reports/groin-strength-review/` | 腹股沟 |
| `scripts/groin_contact_study.py <身体> [--path double,single] [--angle 120] [--render 70,100] [--table 20,70,100,120]` | 沿一条路径解腹股沟接触，或用 `--table` 对比生成用的接触表和雕塑，写 `build/reports/groin-contact-study/` | 腹股沟 |

## 3. 构建产物和缓存

```text
build/bodies/<身体>/wrapped.npy, wrapped.obj, wrap.json        贴合
build/bodies/<身体>/rest.npy                                   静态腹股沟造型后的成品皮肤（绑定阶段）
build/bodies/<身体>/skeleton.json, weights.npz                 绑定（{"bones": [...], "T", "glides", "groin"}；weights、bones 列名）
build/bodies/<身体>/poses/<姿势>.obj                           姿势
build/bodies/<身体>/poses.json, joints.json                    指标
build/bodies/<身体>/renders/<姿势>/<视角>.png                  `review render` 写的单张渲染，不是构建阶段
build/bodies/<身体>/fingerprint.json                           各阶段的输入指纹
build/cache/                                                   工具自己的缓存（`review` 的渲染、编辑器的 GLB），可以随时删除
build/reports/                                                 各工具写给人看的报告和图
<用户缓存>/solve/                                     求解结果，所有检出和安装共用，键是 `solve._row` 能运行到的代码和参数的摘要，改了求解器的分支写到自己的摘要下，可以随时删除
<用户缓存>/axilla-contact/                            双侧肩腋接触表与审计，只有选了 IPC 才用到（导出 `--contact`、编辑器的 IPC 导出、完整层测试 `uv run sotaigen-dev test-full`）；共用，按输入和源码缓存，可删除。`<键>.failed.json` 记录烘焙失败，删除后才会重试
<用户缓存>/groin-contact/                            腹股沟接触表与审计，共用、缓存和失败记录同上；每个新身体约 5–10 分钟
<用户缓存>/hang-contact/                              垂臂接触表与审计，共用、缓存和失败记录同上；它接在肩腋表上，肩腋表的输入和代码都在键里；每个新身体约 9–12 分钟
build/bodies/<身体>/body.glb                                   导出，不属于构建，绑定重做时被删除
build/bodies/<身体>/body-ipc.glb                               `uv run sotaigen-dev export --contact` 写的带 IPC 接触表的导出，同样不属于构建
build/bodies/<身体>/body.blend                                 `uv run sotaigen-dev blend` 写的绑定好的 .blend，不属于构建，绑定重做时被删除
build/bodies/<身体>/body-rest.blend                            同上，换了静止姿势的（`--t-pose` 等），同样不属于构建
build/reports/runtime-check/                                    一致性报告和截图，28 个身体约 1 GB
```

- **在哪里：** `build/` 在检出目录下，装成包时在当前目录下，环境变量 `SOTAIGEN_BUILD` 可以改（`sotaigen/paths.py`）。用户缓存是 macOS 的 `~/Library/Caches/sotaigen`、Linux 的 `$XDG_CACHE_HOME/sotaigen`，`SOTAIGEN_CACHE` 可以改。`compare`、`displacement`、`review` 的 `--against` 也认没有 `bodies/` 的旧布局。
- **读结果要走入口：** `sotaigen.pipeline.ensure_built(name, stages)`、`rigged(name)`（一个 `sotaigen.Body`：`base`、`verts`、`L`、`bones`、`weights`、`wrap` 等）、`surface(name)`。直接读文件可能读到过期或缺失的结果。
- **代码按定义算，不按文件算：** 代码缓存的键都由 `sotaigen/sources.py` 的 `code_key` 算出。它从算结果的函数出发，顺着每个模块自己的绑定和 import（含函数内的）找到能运行到的顶层定义，哈希它们的语法树。所以注释、排版、docstring 和运行不到的定义都不算，新模块被运行到才算。拿不准时多算：模块导入时就运行的代码总算；`cache`、`dataclass` 这类之外的装饰器算导入时运行；用 `getattr` 动态取名、`globals()`、`exec` 的模块整个算，连同它导入的。对象的属性不追，类在哪里被点名就整个算。读了 `__doc__` 的模块，docstring 也算。键里含 Python 的主次版本号。`tests/test_sources.py` 在 `sys.monitoring` 下跑一遍求解和四个阶段，核对实际运行的 `sotaigen/` 函数都在键里。
- **指纹按内容算，逐级串联：** 每个阶段自己的输入是 `pipeline.STAGE_CODE` 里那个函数能运行到的代码，加全部数据：`sotaigen/` 下不是模块的文件，也就是包数据 `sotaigen/data/`，除了滑块定义和 UV 图集，`__pycache__` 和隐藏文件除外。改数据所有阶段都重做。只读成品的 `export`、`face`、`compare` 没有阶段运行得到，所以不进指纹，`tests/test_pipeline.py` 核对。`fingerprint.json` 里各阶段的值串联了上游，读者拿它和 `fingerprints()` 比。
- **结果不变就不往下传：** 一个阶段重做后结果逐字节相同，下游自己的输入又没变，下游就不重做。凭据是 `fingerprint.json` 的 `basis`：每个阶段记下建成时自己的输入和上游全部产物的哈希。重做时只删本阶段，和下游里自己输入已经变了的。别的下游产物先留着，指纹撤掉，等上游做完、核对还成立再恢复。所以 `--stages wrap` 之后下游文件可能还在，但没有指纹担保，要走 `ensure_built`。这依赖结果是确定的：同一身体分两个进程各建一遍，33 个产物逐字节相同。
- **改了什么就重做什么：** 改包数据，或改贴合运行得到的定义，全部身体失效，28 个身体 `--jobs 4` 约 3 分钟（M5，求解缓存有结果时；每个身体约 22 秒）。默认构建不烘焙接触表。选了 IPC 时：肩腋和骨盆接触缓存的键另含烘焙函数能运行到的代码（`contact.axilla_cache.BAKED_BY`、`contact.groin.BAKED_BY`、`contact.hang.BAKED_BY`，经 `sources.bake_key`）和 numpy、scipy、ipctk 的版本，求解缓存只看 `solve._row`。失效的身体要重新烘焙，每个身体十几到几十分钟；肩腋表只求解左侧，右侧由镜像得到。挪动烘焙真正运行的代码仍会让它失效，即使结果不变。
- **指纹不含 `uv.lock`、Python 补丁号和 Blender 的版本：** 升级后要加 `--force`。
- **不要修改共享对象：** `surface` 和 `rigged` 在进程内共享缓存，返回的场不能改，`verts`、`weights` 是只读的。
- **升级 uv：** 同时改 `pyproject.toml` 里的 `required-version`，以及根目录、`blender/pyproject.toml` 和 `dev/pyproject.toml` 的 `uv_build`。**升级 Python：** 改 `.python-version`，跨次版本时还要改 `pyproject.toml` 的 `requires-python` 并重锁 `uv.lock`。

## 4. 看图

涉及观感的改动必须渲染并亲自看图，再交给用户判断。

```sh
# 灰模查曲面起伏，赛璐璐描边查动画观感，透视看屏幕上的实际比例：每种观感一行
uv run sotaigen-dev review sheet female-long --pose armsDown --view front,threequarter,side,backquarter --look clay,toon,toon+persp
# 两个性别的匀称比较，同一比例、同一地面线（--group 是组，不写时取典型身体的组）
uv run sotaigen-dev review sheet --sex all --group balanced --pose rest --view all
# 局部特写，取景跟住摆姿势后的部位
uv run sotaigen-dev review sheet female-long --pose crouch --focus hip --view threequarter,side,back
# 改前／改后：主仓库的 build/ 就是改前
uv run sotaigen-dev review sheet --sex all --pose rest --against <主仓库>/build
# 剪影看比例和轮廓；配 --against 时每格多一个 CHANGE，红是轮廓长出来的，蓝是缩进去的
uv run sotaigen-dev review sheet female-long --look silhouette --view front,side --against <主仓库>/build
# 看拓扑
uv run sotaigen-dev review sheet female-long --wire --mm-per-px 1
```

- **比例可信的范围：** 正交相机下，同一张图统一 mm/像素，整像素粘贴不重采样，全身图按实际最低顶点落在地面线上。透视图的比例尺只对取景中心平面有效，地面线只表示该格身体的最低投影点，不是共同地板。
- **透视相机：** 在身体轴线上，距离 1.5 个身高，高度为身体最低点上方半个身高，相当于 50 mm 镜头拍 1.08 个身高。
- **描边：** 线宽按屏幕约 2.5 px 换算，所以换比例时线和身体的相对粗细会变。比较观感要用同一个 `--mm-per-px`。
- **剪影：** 平光单色，颜色和编辑器的剪影相同。配 `--against` 时，CHANGE 格把改前、改后两个网格渲染进同一个取景再叠起来：灰色是两边都有，红色是只有改后有（轮廓长出来），蓝色是只有改前有（缩进去）。同一个取景就是同一台相机，所以叠图逐像素对齐、不重采样。它只说明轮廓哪里变了，表面上的变化要看 `displacement` 的热图。剪影不能配 `--wire`。
- **过期输入：** `review.py` 只读，不重建。输入过期或缺失时拒绝并提示 `uv run sotaigen-dev build` 的命令；`--allow-stale` 放行，图上标 `OUT OF DATE`。`--against` 的目录不查过期。
- **局部取景** 来自摆姿势后的顶点和模板 `segments`，不用地标。默认看穿着者左侧，不对称姿势的右侧要看全身图。遮挡不处理。
- **缓存** 在 `build/cache/review/`，键是内容哈希，可以随时删除。
- **单张渲染** 是透明背景的 RGBA 图，写在 `renders/<姿势>/<视角>.png`。

## 5. 模块之间的交界

文件不按领域划分归属，同时改同一个文件靠 worktree 和合并解决（见 [MERGING](MERGING.md)）。这里只记改动时容易漏掉的耦合。

**新文件：** 新建模块、脚本、测试时，名字不要和已有的撞。不参与生成的数据放在自己的顶层目录，比如动作数据放 `motions/`，不要放进 `data/` 或 `sotaigen/`，那里的文件都算贴合的输入。

**许可：** 导入 `bpy` 或 `bmesh` 的文件是 GPL-3.0-or-later，第一行写 `# SPDX-License-Identifier: GPL-3.0-or-later`，都放进 `blender/`（`template/` 的两个提取脚本除外）。`blender/` 根下只放生产脚本，`.blend` 的缓存键只算它们；看图工具放 `blender/review/`，测试夹具放 `blender/fixtures/`。其余文件不写许可头，由根目录的 `REUSE.toml` 按路径给出许可：代码 Apache-2.0，`blender/` GPL，`data/` 和模板 CC0，`motions/clips/` 注明 CMU 的来源。复制别人的代码进来时，在 `REUSE.toml` 加一条，许可原文放进 `LICENSES/`。`tests/test_licenses.py` 检查前两条和文件的位置。

**测试的写法：** 考逻辑的测试用替身数据（假的身体名如 `petite-soft`、假的姿势和网格），不依赖自带身体的名字、姿势表和顶点数，所以增删身体和姿势不会让它们失败。点名真实身体是有意的两类：守不变量的测试（贴合、绑定、导出在典型或全部身体上成立，常用 `female-long` 作代表，增删身体时它们跟着 `body_specs()` 走）；守约定的 `tests/test_contracts.py`，拿 `tests/contracts/` 的金样守冻结的约定（[对外约定](reference/README.md)），它失败就是约定被破坏，不是测试过时。顶点数和顺序是冻结的约定，见下表。`rest` 姿势是工具的默认值，删掉它会让测试失败。`tests/test_metrics.py` 考的是 `metrics.py`，改 `pose_row` 的列名或 `collapsedFaces` 的含义时要在报告里写明。

**依赖：** Python 依赖要改 `pyproject.toml` 和 `uv.lock`。Node（`.node-version`）和 pnpm（`package.json` 的 `packageManager`）的版本已固定，JS 依赖只有一份，在 `web/package.json` 和 `pnpm-lock.yaml` 里，用 pnpm 改（`pnpm --dir web add -E 包@版本`），不用 npm 管理项目依赖（第 1 节用 npm 安装 pnpm 本身）。新依赖固定版本，遵守 14 天等待期，并在报告里写明；`pnpm-workspace.yaml` 的 `minimumReleaseAge` 对间接依赖也强制这一条，太新的版本解析不了。要跑安装脚本的包须先列进同一文件的 `allowBuilds`。升级 pnpm 时改 `package.json` 的 `packageManager`，并按第 1 节重新安装 pnpm；升级 Node 改 `.node-version`，跨主版本时还要改根目录 `package.json` 的 `engines.node`。

| 情况 | 办法 |
|---|---|
| 转头、点头的不变量要扩到全部身体（现在 `tests/test_head.py` 只测典型身体） | 在 `tests/test_rig.py` 里加全部身体的版本，不改 `tests/test_head.py` |
| 颈头过渡涉及 `head.py`、`seat_neck_ring` 和颈部体块 | 先只改 `head.py` 验证效果，不够时再动另外两处 |
| 改变顶点数或顶点顺序 | 顶点数和顺序是冻结的约定 ①（`tests/test_contracts.py`），要改就是破坏约定，先问用户；加环只能追加在末尾（`template/PROVENANCE.md` 的“编号”） |
| 改 `sotaigen.rig.skinning.sided` 的结构 | `web/runtime/driver.mjs` 和 `blender/src/sotaigen_blender/rig.py` 要和它一致，`tests/test_export.py` 会失败。驱动表的结构变了要升 `extras.sotaigen.version`。只改 `HALF_BONES`、`HALF_ROLLS`、`TWISTS`、`SHOULDER_RHYTHM`、`FORWARD_LIFT` 的数值时，导出会自动跟上 |
| `sotaigen/`、`data/` 或 `template/` 有改动 | 重启 `pnpm run editor`，让 Python 工作进程加载最新代码、参数和模板 |
| 改五官和头发的数据结构 | 契约见 `docs/systems/export.md`，改字段要同步导出 |
| 改变头部顶点或面 | `sotaigen/data/template/uv.npz` 要跟着重新生成（命令见 `template/PROVENANCE.md`），否则 `tests/test_uv.py` 失败 |
| 重新提取模板（比如要带 UV） | `verts`、`faces` 及其顺序不能变 |

## 6. 代理

代理（Claude、Codex）在本仓库并行工作的约定、报告格式和简报模板在 [给代理的约定](agents.md)。

## 7. 查文献

按问题选来源。引用前必须读到原文，读不到全文就如实写明核对范围；动态解剖卡里只读到摘要或检索摘录的依据标 `snippet`。

| 问题 | 先查 | 备用 |
|---|---|---|
| 人体解剖、体表、肌骨 | PubMed 连接器 | 没有连接器的工具和会话用 [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25497/) |
| 图形学和工程方法 | [OpenAlex](https://help.openalex.org/api/) 接口 | [Crossref](https://api.crossref.org/) 核对 DOI 和标题 |
| 较新的预印本 | arXiv | |
| 以上都查不到 | 请用户在 Google Scholar 手动查，不要自动抓取 | |

不要用 Elicit、Consensus 这类工具的总结当依据，要回到原文核对。

```sh
curl -s "https://api.openalex.org/works?search=pose%20space%20deformation&per-page=5&select=display_name,publication_year,doi,cited_by_count"
```

**OpenAlex 额度：** 不带密钥每天预算 $0.10，约 100 次搜索；免费密钥是它的 10 倍，翻页超过第一页也要用密钥。注册由用户本人完成。在所用 shell 的用户配置文件中设置环境变量 `OPENALEX_API_KEY`，不要放进仓库；`.env` 虽然已被 gitignore，密钥仍放在 shell 的配置里。调用时加 `-H "Authorization: Bearer $OPENALEX_API_KEY"`，不要写进 URL。价格会变，以[官方说明](https://help.openalex.org/access/pricing/)为准。

**核对结果：**

- 搜索的首条结果不能直接引用。OpenAlex 和 Crossref 都出现过首条是别的论文，也出现过首条是 ACM 2023 年再版条目的情况。要核对作者和标题，引用原始发表的年份和出处。
- PubMed 连接器按引用查找时，`pmid` 和 `key` 两个字段曾被填反，查到后用元数据核对。
- PubMed 连接器取回的全文不含图片。引用其内容时注明来自 PubMed，并附返回的 DOI 链接。

**找全文：**

- 连接器的全文为空不等于读不到，再试 PMC 页面。没有 PMC ID 的论文，连接器取不到全文，需要用户提供。
- 图形学论文先找作者主页和项目页，再试 ACM Digital Library。不能用 OpenAlex 的开放获取标记判断读不到全文，它把 IPC 2020 标成了 closed，而 ACM 已于 2026 年开放阅读。

**放在哪里：** 人体研究才能作为动态解剖卡里现象的 `evidence`。图形学论文回答的是用什么技术实现，写进卡片说明的[技术对照](motion/README.md#技术对照)和相关领域文档。

## 8. 发布检查

支持平台是 macOS 和 Linux。Blender 最低版本 5.2（LTS），只在 5.2.3 上测过；低于最低版本时拒绝，命令行退出码见[对外约定](reference/cli.md)。

CI 配置以 `.github/workflows/ci.yml` 为准：Linux、macOS 上 PR 跑日常层测试，`main` 和定时任务跑合并层测试（`uv run sotaigen-dev test-merge`），另有页面、wheel 安装与 Blender 用例；action 钉提交 SHA。发布前完成以下核对：

- 在干净检出中按锁文件安装，运行第 2 节的 ruff 和文档检查、合并层测试、`pnpm run web-check` 和包安装冒烟；逐个平台核对 CI 结果。
- 核对两个包的版本、对外约定、CHANGELOG、已知限制及中英文说明一致；形体改动后重出 README 配图，并由维护者确认。
- 核对发布目标和 GitHub Actions、私密漏洞报告的设置与 `SECURITY.md` 一致。推送、创建远端、打标签和公开内容都须维护者授权。


### 交付缺口

- **衣物适配尚未系统验证：** 在固定 A 姿势上，用标准贴身上衣、裙或裤、鞋检查典型身体动作中的穿插和观感；还缺衣下隐藏区域、衣物对位地标与量法。这是下游适配基础，不是服装设计。
- **IPC 核验未完成：** 腹股沟仍有烘焙失败身体，带 IPC 的 `compare`、`runtime-check`、`pnpm run web-check` 未完成；肩腋和垂臂须分别核对审计。再次批量烘焙前先问用户，具体限制见[腹股沟](parts/groin.md#接触表)、[肩腋](parts/shoulder-axilla.md#接触表)。
- **视觉验收：** [待用户决定](DECISIONS.md#待用户决定)中的发布阻塞项须得到确认。
- **对外限制：** 修好或发现新的缺陷时，同步更新中英文[已知限制](reference/limitations.md)，列明已验证和待验证的项目。
