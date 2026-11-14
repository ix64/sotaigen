# 动态解剖卡

优化某个部位的运动时，容易一上来就去减少塌陷和穿插。这些数字只能说明网格没坏，说明不了动作对不对。卡片把问题换成：这个动作里，肌肉、骨点和皮肤该怎么变，我们要保留哪些。

## 顺序

1. **写卡。** 查解剖和体表资料，列出现象。先不看代码，不提修法。
2. **看现状。** `uv run sotaigen-dev motion sweep <卡片> <预设 ...>`，把每条现象的现状写进 `current`。
3. **用户定取舍。** 用户看图后，逐条定 `choice`，定完把 `status` 改成 `confirmed`。代理只能写 `proposal`，用户定下后改名 `decision`。
4. **选技术。** 按下面的对照表，给每条现象选能做到的最简单技术。
5. **实现并复看。** 用 `--against` 对比改前改后，报告里逐条写达成情况。
6. **指标兜底。** 跑 `compare`。退化仍要用户认可，折痕里的也一样。

没有卡的部位，先写卡。一张卡只写一种运动范围，够用就行，不追求全。

人的关节能转多大（活动范围）不写在卡里，见[关节的活动范围](range-of-motion.md)，数据在 `sotaigen/data/range-of-motion.json`。

## 字段

每张卡是 `docs/motion/<部位>.toml`，放在 `data/` 外面，所以改卡不会让构建重跑。`tests/test_motion_cards.py` 只查结构，不判断好不好看。

| 字段 | 含义 |
|---|---|
| `status` | `draft` 或 `confirmed`。`confirmed` 要求每条现象都有 `current`，且 `choice` 不是 `undecided` |
| `observed` | 现状看的是哪些体型、哪个提交，还缺什么参照 |
| `joint` | 关节的两根左侧骨骼和模板分段 |
| `sweeps` | 观察用的角度序列，`pose` 乘以每个角度 |
| `zones` | 体表区域。`axial` 是离关节的距离（以远端肢段长为单位，朝远端为正），`azimuth` 和 `spread` 是方位角（0 屈侧、180 背侧、+90 沿弯曲轴向外） |
| `phenomena` | 现象，字段见下 |
| `guards` | 哪个检查姿势的哪些区域预期有折痕，供 `motion folds` 使用 |

现象的字段：

- `kind`：`bone-motion` 骨运动、`bone-reveal` 骨点显露、`muscle-bulge` 肌腹隆起、`stretch-flatten` 拉长变平、`crease` 皱褶、`slide` 皮肤滑动、`contact` 软组织相压。
- `reading`：看图时怎么判断。
- `current`：现在做到什么程度，包括当前实现、数值范围、已知缺陷和验证缺口。
- `choice`：`keep` 保留、`exaggerate` 夸张、`simplify` 简化、`drop` 舍弃，未定时为 `undecided`。
- `decision`：用户选定的现象取舍与验收要求。暂缓项写清本卡的范围；当前算法和调参停止点写入 `current` 或领域文档。
- `evidence`：每条依据写 `strength`，分 `measured`（人体测量）、`textbook`（参考书）、`tertiary`（百科和教学页）、`snippet`（只读到检索摘要）、`observation`（有人看到但未核实）。研究的条件写进 `claim`，不要把收缩的结果当成随角度的结果。去哪里查见[查文献](../WORKFLOW.md#7-查文献)。图形学论文说明怎么实现，不作为现象的依据，写进下面的技术对照。

## 读图

`motion sweep` 把关节立直，弯曲轴朝 X，所以 `side` 总是沿弯曲轴看，不受静止姿势影响。躯干已隐藏，手保留。

- `sweep-<名字>.png`：灰模和 toon 的角度序列，加 `--against` 可比较另一次运行。
- `behaviour-<名字>.png`：第一行是弯曲平面的截面，黑线是皮肤，蓝线是两段肢体各自刚性转动的结果，两者的差就是蒙皮做的事。其余几行给皮肤着色，红色相对所在肢段抬起，蓝色内缩，紫色是滑动。
- `report.json`：按区域列出抬升、滑动、半径比例和 `reachMm`（离弯曲轴最远的距离）。这些数字只描述，没有及格线，也不要把它们做成门槛。
- 抬升和滑动是相对皮肤所在的模板分段算的。如果上臂段的皮肤被改成跟随前臂，它们的数字会变大，看起来像内缩，其实是改了跟随对象，这时看截面和 `reachMm`。

## 技术对照

只写本项目验证过的，其余标“待试”。验证后回来改这张表。证据在[绑定](../systems/rig.md)和各部位的文档。

| 现象类型 | 技术 | 运行时代价 | 本项目的结论 |
|---|---|---|---|
| 骨运动 | 加骨骼或改关节轴 | 标准播放器可用 | 肩带采用锁骨协同，不含独立肩胛骨（`scapular-rotation`） |
| 扭转 | 半转骨、扭转骨 | 标准播放器可用 | 前臂和上臂在用 |
| 大范围过渡、接缝硬边 | 权重拟合 | 无 | 腹股沟有效，腋下没有明显改善 |
| 网格排距不均处的权重台阶 | 按毫米平滑权重（`smooth_over_length`） | 无 | 颈有效：短颈转头的塌陷来自排挤在一起处的权重台阶；平滑半径太大时转动传得太远（仰头时后颈凹进去） |
| 肢段局部的形状 | 姿势变形目标 | 要写驱动或烘进动画，VRM 不直接支持 | 肩腋、腹股沟在用；腹股沟深屈髋的自交没有解决 |
| 骨点显露、背侧保体积 | 按方位角区分权重（`seat_elbow_tip`） | 无 | 肘尖有效。背侧范围要宽，网格按方位角的列数少，窄了会裂成几个尖；再平滑一轮去掉细线，多了转角会变圆 |
| 背侧成圆弧、屈侧留一道折 | 半转骨，加只平滑背侧的权重（`round_wrist_back`） | 一根骨 | 腕有效。半转骨去掉屈侧塌陷，平滑去掉背侧的台阶；半转骨要带手的滚转，否则多一圈箍痕；平滑半径太大时前臂背线外鼓 |
| 大角度软组织相压 | 待选 | 待验证 | 待试，权重解决不了两段肢体的厚度 |
| 肌腹随用力隆起 | 单独的表演控制 | 要写驱动或烘进动画 | 待试，它不只随角度变化 |
| 细小皱褶 | 法线或 toon 处理 | 待验证 | 待试 |

## 不要做

- 不要把卡片或这些描述量合成观感分数。
- 不要因为数字变好就算现象达成，要附图。
- 不要照搬写实细节。动画风格下舍弃是正常结果，但要由用户决定并写进卡片。


## 肩腋／胸侧卡

`shoulder-chest.toml` 的十二项取舍都已定（`confirmed`），包括上半身目标的四项（肩胛上回旋、肩顶轮廓、三角肌、手摸头）。目标角度和复合姿势的图用 `scripts/upper_body_study.py` 出，见[肩腋的诊断](../parts/shoulder-axilla.md#上半身目标角度的诊断)。

`motion sweep shoulder-chest` 默认用雕塑修正，不碰接触表。加 `--contact` 才读 `sotaigen.contact.axilla_cache.targets`（IPC 接触表），没有缓存时会烘焙，并读写用户缓存目录的 `axilla-contact/`。

相同的 `motion sweep` 入口会使用
`sotaigen_dev/motion/shoulder_sheet.py`：保留躯干及现有肩腋修正，在胸廓坐标系观察，
不套用肘卡的单铰链立直、隐藏躯干或肢段半径量法。输出角度序列、固定矢状截面、
皮肤相对胸廓的法向／切向位移着色与同一批表面点轨迹；没有观感分数或数值门槛。

```sh
uv run sotaigen-dev motion sweep shoulder-chest female-long --look clay
# 读一份冻结的网格和权重（目录里有 mesh.npz、bones.json）时必须显式给出；它保留冻结时的权重，不代表现行结果
uv run -m sotaigen_dev.motion.shoulder_sheet female-long --frozen SOURCE --render
```

Q版和小巧（`female-chibi`、`female-compact`）胸量为零，不能拿它们判断乳房随动。两组图分别写在
`build/reports/motion/shoulder-chest/` 和 `build/reports/motion/shoulder-chest-frozen/`，来源记在 `report.json`。
胸部前方与外上方取样只是几何观察点，不能当成临床解剖地标。肩腋分区尚未定义，
因此本卡暂不生成 `folds` 预期区域，也不冒用肘卡的圆柱分区。


## 腕

`wrist.toml` 是屈腕到 60° 的卡，取舍已定。手的弯曲轴是前后向的，所以标准坐标的弯曲轴不按“朝身体外侧”取，统一取骨骼局部 X 的反方向：肢体上它就朝外，手上朝拇指一侧（方位角 +90）。`side` 沿弯曲轴看，`back` 看手背，`front` 看手掌。伸腕是同一张卡的第二个 `sweeps`，方向由第一个定。

```sh
uv run sotaigen-dev motion sweep wrist male-rounded female-chibi
```

## 腹部—大腿根

`pelvic-flexion.toml` 是固定骨盆屈髋的现象卡，现象取舍已定（`confirmed`），造型的现状和没解决的问题见[腹部—大腿过渡](../parts/groin.md)。站姿斜沟、表面相对运动、屈髋折痕和体块相压分别记录；`keep` 只表示保留该现象，不表示形态已验收。整体视觉通过前不以穿插或塌面计数优化或排名，也不先划定“正确褶皱区”。

```sh
uv run sotaigen-dev motion sweep pelvic-flexion female-chibi female-long female-balanced
# 只更新截面、位移图和原始观察数据，不启动 Blender
uv run sotaigen-dev motion sweep pelvic-flexion female-long --no-render
# 对比另一份保存的运行；目录与肘部工具的 /poses 不同
uv run sotaigen-dev motion sweep pelvic-flexion female-long --against /path/to/saved/pelvic-flexion
```

该卡的 `review.renderer = "pelvis"` 由 `sotaigen_dev/motion/groin_sheet.py` 处理，不改肘部出图方式：

- `whole-single.png`、`whole-double.png`：0°、40°、100° 的全身关系。
- `sweep-<single|double>-<front|threequarter|high>.png`：每 20° 一格的腰腹至大腿整体，灰模与 toon 对照，模型完整保留。
- `behaviour-single.png`、`behaviour-double.png`：左髋轴与中央两条截面，再看表面相对刚性参照的法向、切向分量。只有这张诊断图隐去其他分段，大腿末端开口来自截断，不是原模型破洞。
- `observations.npz`：静止、摆姿、每点参照骨和位移分量；`report.json`、`card-snapshot.json`、`render.json` 记录代码、卡片、骨架、权重、姿势和机位。

骨盆向上作为观察坐标的竖轴，左髋屈曲轴作为横轴。参照按静止模板分段明确分配：`trunk` 相对 `hips`，左右 `upperLeg` 各自相对同侧股骨；不按顶点在关节上方或下方猜它属于哪段。模板分段处参照改变可能造成颜色分界。切向分量只是当前蒙皮相对该参照的位移，不是皮肤相对筋膜的真实滑移；蓝色刚性截面也不是人体目标。没有给这些量设及格线或综合评分。

同一体型、同组姿势的机位与比例固定；跨体型按身高统一显示比例，因此不能从不同体型的图上直接比较毫米尺寸。`--against` 会拒绝机位、比例、光照或采样姿势不一致的清单。

造型原则与观察方法见[腹部—大腿过渡](../parts/groin.md#造型与观察方法)。

生产接入由 `sotaigen/corrective/groin.py` 实现：静态造型在绑定参考皮肤之后应用；左右屈髋二维姿势目标与网页、GLB 共用。深屈髋仍有穿插，技术上没有完成。


## 颈

`neck.toml` 是转头、点头、仰头、侧屈的现象卡，取舍已定（`confirmed`）；当前用权重实现，见[颈部绑定](../systems/rig.md#颈部的权重按毫米平滑)。颈在中线上，肘卡的肢段坐标不适用，`zones` 暂不填，`motion sweep` 不能直接用；前后对照图用 `uv run sotaigen-dev review sheet <身体 ...> --pose ... --focus neck --against <另一个构建>` 出。

## 肩颈的前后对位

`shoulder-neck-set.toml` 看肩带、颈和头相对胸廓站在前后什么位置，以及位置变了以后垂臂和抬臂是否还成立，`confirmed`，八项取舍都已定。现状和方案在[肩颈的前后对位](../parts/shoulder-neck-set.md)。`zones` 暂不填，图用身体坐标系的侧面和斜前。

## 张嘴

`jaw.toml` 是张嘴的卡，取舍已定（`confirmed`）：前滑骨长的 0.0125/度，关节极限 27°，干净范围 20°，其余按简化做了。下颌在中线上，`motion sweep` 不能直接用，`zones` 不填；图用 `uv run scripts/head_review.py --only jaw,gaze --presets ...`（默认写到 `build/reports/review/head`），看各档张嘴、转头和点头加张嘴、视线四个方向，clay 和 toon。视线的眼骨不带皮肤，没有卡，活动范围在[同一张表](range-of-motion.md)里。
