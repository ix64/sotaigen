# 文档索引

## 从哪里读

| 你是 | 想知道什么 | 读 |
|---|---|---|
| 用户 | 什么在等我决定 | [DECISIONS](DECISIONS.md) 的“待用户决定” |
| 用户 | 某个部位做到哪了、还差什么 | [parts/](#身体部位) 里那份文件的开头 |
| 用户 | 参数是什么意思 | [PARAMETERS](PARAMETERS.md) |
| 下游 | 能依赖什么、怎样变 | [对外约定](reference/README.md)（[English](en/reference/README.md)） |
| 改某个部位的代理 | 当前实现、限制、技术依据 | [parts/](#身体部位)；动骨骼、权重或姿势变形前先读 [motion/](motion/README.md) |
| 改流水线的代理 | 契约、不变量、工具 | [ARCHITECTURE](ARCHITECTURE.md)、[systems/](#流水线) |
| 刚开工的代理 | 环境、命令、看图；并行、汇报和简报 | [WORKFLOW](WORKFLOW.md)、[给代理的约定](agents.md) |
| 合并的人 | 跑什么、退化怎么判断 | [MERGING](MERGING.md) |

## 内容放在哪里

同一项内容只在一个主位置维护。入口和其他文档可以用一句摘要帮助读者判断是否要点进去，不复制整段规则、数值表或实验结论。

| 内容 | 主位置 |
|---|---|
| 下游可依赖的名字、结构、单位和兼容政策 | [reference/](reference/README.md)；英文为规范文本，表由契约金样生成 |
| 生成流水线、模块依赖、共享数据与不变量 | [ARCHITECTURE](ARCHITECTURE.md) |
| 仍有效的产品取舍、理由和待用户决定事项 | [DECISIONS](DECISIONS.md)；证据链接到领域文档 |
| 参数量法、读数含义与现实测量参考 | [PARAMETERS](PARAMETERS.md)；规则与可达边界放 [systems/params](systems/params.md)，身体取值依据放 [systems/presets](systems/presets.md) |
| 操作命令、环境、构建产物、研究与发布流程 | [WORKFLOW](WORKFLOW.md)；合并门槛与比较工具放 [MERGING](MERGING.md) |
| 仓库最重要的约束、任务入口 | [AGENTS](../AGENTS.md)；并行隔离和交接细节放 [agents](agents.md) |
| 一个部位的形体、实现、证据与未解决问题 | 下列 `parts/` 文档 |
| 跨部位的机制、实现、验证边界 | 下列 `systems/` 文档 |
| 动作现象的观察与用户取舍 | [motion/](motion/README.md) 中的卡片；实现结果仍归领域文档 |

## 身体部位

| 文件 | 内容 |
|---|---|
| [head](parts/head.md) | 头、五官、头发 |
| [chest](parts/chest.md) | 胸型、胸肌支承、胸部量法 |
| [shoulder-axilla](parts/shoulder-axilla.md) | 肩腋形体、姿势修正、接触表 |
| [shoulder-neck-set](parts/shoulder-neck-set.md) | 肩颈相对胸廓的前后对位，现状和方案 |
| [torso](parts/torso.md) | 腰腹过渡、胸下缘、腹前弧和后腰 |
| [groin](parts/groin.md) | 小腹到大腿根的斜沟、屈髋相压、接触表 |
| [groin-evidence](parts/groin-evidence.md) | 腹股沟的外部证据摘录 |
| [hip-thigh](parts/hip-thigh.md) | 臀形、上胯到大腿、大腿轮廓 |
| [limbs](parts/limbs.md) | 腿段比例、膝、肘、手、脚，手脚接缝和关节边环 |
| [male-muscle](parts/male-muscle.md) | 男性肩胸背的四项肌肉参数 |

## 流水线

| 文件 | 内容 |
|---|---|
| [form](systems/form.md) | 隐式场的总体现状、缺陷和参考取舍 |
| [params](systems/params.md) | 求解、校验规则、范围依据和压力测试 |
| [presets](systems/presets.md) | 典型身体和测试身体的用途、比例来源与验证边界 |
| [wrap-uv](systems/wrap-uv.md) | 贴合、接缝上限、提速、UV |
| [rig](systems/rig.md) | 骨骼、权重拟合、关节极限、真实动作 |
| [export](systems/export.md) | GLB、运行时一致性、部件契约、VRM 差距 |
| [blend](systems/blend.md) | 交给下游的 `.blend`：怎么写、文件里有什么、命名（对外约定）、关节极限、和运行时的误差 |
| [touch](systems/touch.md) | 手到头、手到脸的触点：对外约定、按身体解关节角、编辑器和 `.blend` 的接入 |
| [editor](systems/editor.md) | 编辑器的界面、等待、参数文件、滑块 |
| [motion/](motion/README.md) | 动态解剖卡：先定现象取舍，再选技术 |

## 写文档的规则

- **只保留现行知识。** 被取代的决定、失效测量、一次性排期和探索流水账直接删除，过程需要追溯时看 Git 历史，不另建归档。仍在支撑当前设计的研究依据、验证边界与可复现方法保留在主位置。
- **临时实现不升级为永久要求。** “现在这样做”写在领域文档；只有仍有效、需要用户确认的取舍进入 `DECISIONS`。代码中还有的限制不能仅因它是临时决定就从已知限制里删掉。
- **删副本时修引用。** 直接链接内容的主位置；不让读者经过“决定 → 参数 → 系统”的多次跳转。涉及生成表时改金样或生成器，再生成，不手改表。
- **领域文档先写现状与未解决问题，** 再写当前做法、依据和验证方法。实验数字须说明对象与适用范围；已不适用于当前实现的数字不继续当作现状。
- **不写具体下游项目的信息。** 角色尺寸、资产清单、适配结果和试用经过留在下游；这里记录可复用的接口、问题和限制。
- **语言：** README、下游技能和对外约定有中英两版；CONTRIBUTING、SECURITY、CHANGELOG 只出英文；内部工作文档用中文，代码注释用英文。对外约定的规范文本与金样优先级见[规范文本](reference/README.md#规范文本)。
- **检查：** `uv run sotaigen-dev lint --docs-only` 查相对链接、锚点、代码路径和失效标记；对外约定还查中英结构一致。约定表由 `sotaigen_dev/reference.py` 从 `tests/contracts/` 生成，更新方法见[诊断](systems/export.md#诊断)。
