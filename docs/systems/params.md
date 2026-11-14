# 参数、求解与校验

> 每个身体（`sotaigen/data/bodies/<性别>/`，见[预设](presets.md)）各是一份毫米规格。每个参数的控制量、单位、换算、范围和所属的局部轮廓都在 `sotaigen/params/spec.py` 的一张表 `ALL_SPEC` 里，每个参数一行；每个性别能用哪些参数、范围多少，在同一文件的 `SEXES`；规则在 `sotaigen/params/rules.py`，按性别分组（见下文“按性别”）。参数文件的对外约定见[参数文件](../reference/parameters.md)。“规格上的规则”说明范围与规则，“参数校验与编辑工具”说明调用方式。

**还没解决：**

- “手臂离开肋骨”的阈值 0.196 在矮身体上可能偏紧，需要按固定的 35° 静止角复核。
- 手脚接缝的三条拟合规则只证明挡住了已知的超限身体，没有按新规则重新抽样跑压力测试；手腕折角那条只靠 3 个超限点。
- 按真人身体放宽的限制每条只在少数身体上验证过，别的组合上可能仍会坏，要在当前体块上重跑压力测试。
- 头颈配比的下限 0.47 只验证了检查姿势的转头、点头，更大的角度没有试；上限 1.58 要放宽就得把头深写进规则，没有做。
- 厚度场乳房上的胸肌只构建、看图到 0.0225 T（`BREAST_RULES`）。
- 范围和规则是现行工程边界；形体、绑定或参数定义变化后，按下文“复测”重新验证。

## 规格上的规则

范围和规则都读规格，长度是躯干长的份额（`proportions`），不用求解，页面每次拖动都能算。各条限度的来历：

规则由曲面可达性、接缝、绑定和姿势压力测试确定，是模型的工程边界，不是人体统计范围。参数调整后重新生成，分别检查目标的到达值、接缝和姿势。

- **求解下限：** 胯厚、颈围和腕围受相邻体块与圆角限制，线性下限是可达边界的近似。胯厚规则按大腿上段围拟合；常数分别取女、男 `chibi` 的浅胯边界。
- **细颈：** 颈和斜方肌的圆角至多 1.5 倍颈半径。颈围下限采用颈长、`definition`、`softness` 扫描的线性下包络，避免挡住可达目标；拟合偏松时由界面报告目标与实际。0.26 T 的范围下限对应长颈上 0.03 T 的求解半径，头颈比下限由 `female-thinneckchibi` 守护。
- **手脚接缝：** 规则拟合接缝偏差与折角，优先覆盖已知超限组合，同时保留典型和测试身体。拟合会误挡接近边界的组合，手腕折角的样本尤其少；实际判定仍以贴合结果为准。
- **性别差异：** 男性的肩宽、肩高与上臂关系使用独立规则，`male-wideheavyarm` 守护粗臂边界。头颈比上限仍为 1.58，侧面轮廓先于颈环成为限制；待决定事项见 [DECISIONS](../DECISIONS.md#待用户决定)。
- **大腿与骨盆：** `upperThighGirth / hipBreadth ≥ 1.05` 限制宽胯细腿组合；大腿围滑块通过 `lifts` 同时抬高胯厚下限。两个性别的胯厚范围下限均为 0.30 T。

规则里的 1.088、5.689、5.742 是胸廓宽、上臂围、膝围在 `ALL_SPEC` 里的换算倍数（`CHEST_BREADTH_GAIN`、`UPPER_ARM_GIRTH_GAIN`、`KNEE_GIRTH_GAIN`），规则是在控制量上找到的，所以用同一个数换算。

| 规则 | 式子（长度单位 T） | 越过会怎样 |
|---|---|---|
| 肩要宽过胸 | `biacromialBreadth - chestBreadth/1.088 ≥ -0.02` | 腋下翻转 |
| 手臂离开肋骨 | `(biacromialBreadth - chestBreadth/1.088)/2 + upperArmLength·sin(35°) - upperArmGirth/5.689 ≥ 0.196`，35° 是手臂的静止角度 `ARM_REST_DEG` | 上臂内侧投影到肋骨上，翻转，再严重就越过中线。在矮身体上可能偏紧（见开头） |
| 手臂和窄肩 | 女性 `upperArmGirth/5.689 + 0.83·acromionRise − 0.15·(biacromialBreadth − 0.7) ≤ 0.094`；男性 `upperArmGirth/5.689 + 0.83·acromionRise ≤ 0.0805 + 0.259·(biacromialBreadth − 0.61)` | `armsUp` 腋下塌陷。依据是两个性别的压力测试，女性肩宽 0.72 T 以下比下一条紧（[测试身体](presets.md#测试身体)），男性 0.742 T 以下 |
| 手臂和肩高 | `upperArmGirth/5.689 + 0.83·acromionRise ≤ 0.097`，男性 `≤ 0.1148` | `armsUp` 腋下塌陷。0.097 是验证过的最远值（`male-rounded` 在 0.0966 时抬臂不塌，[年长男性](presets.md#男性的三个年长身体)）；男性采用独立压力测试边界 |
| 躯干不能太短 | `T / 身高 ≥ 0.2` | 模板落不到躯干上 |
| 头颈配比 | `0.47 ≤ neckGirth / headHeight ≤ 1.58` | 太粗：头深 0.90 的男性头上，侧面颈后缘越过后脑、颈前缘越过脸（`tests/test_head.py` 的侧面测试，1.62 起），下巴下鼓出小疙瘩；下巴到 1.80 仍在标志线上。太细：转头、点头可能塌陷，0.47 头时检查姿势还没塌，更细没有试 |
| 躯干不能太长 | `T / headHeight ≤ 3` | 下巴离开标志线 |
| 大腿填满半胯 | `upperThighGirth / hipBreadth ≥ 1.05` | 下蹲时两膝越过中线。1.05 是体块在宽胯上能做的最细，没有越过 |
| 脚和膝 | `0.186·footLength - 0.157·kneeGirth ≤ 0.01` | 小腿内倾，下蹲时两膝越过中线 |
| 踝高和脚长 | `0.24 ≤ ankleHeight / footLength ≤ 0.55` | 模板的脚型放不下 |
| 膝不粗过小腿 | `kneeGirth / calfGirth ≤ 1.06` | 腿的收窄翻转 |
| 胯厚下限 | `hipDepth ≥ 0.375·upperThighGirth + 0.0318`，男性 `+ 0.0262` | 大腿和骨盆顶住，目标到不了。常数是最浅的出厂胯；大腿上段围的滑块加粗时把胯厚抬到这里 |
| 颈围下限 | `neckGirth ≥ 0.51 - 5.36·neckLength + 0.61·definition + 0.215·softness` | 短颈上量到斜方肌和圆角，目标到不了 |
| 腕围下限 | `wristGirth ≥ 0.175 + 0.23·upperArmGirth - 0.22·lowerArmLength` | 手臂的收窄顶住，目标到不了 |

接缝规则只预警：

| 规则 | 式子 | 预防的 |
|---|---|---|
| 手和前臂 | `handLength²/(wristGirth·lowerArmLength) ≤ 0.887` | 手腕折角 |
| 手、前臂和上臂 | `18.5·handLength/lowerArmLength + 16.6·handLength²/(wristGirth·lowerArmLength) + 1.83·upperArmGirth/wristGirth + 0.22 ≤ 33.7` | 手腕折角，左边是折角的拟合（度，残差 0.7），上限 35° |
| 手腕和手 | `wristGirth/handLength ≤ 1.36` | 手的接缝偏差（Q版上 1.42 时超限） |
| 手腕、手和手臂 | `0.548·wristGirth/handLength + 0.221·lowerArmLength − 0.146·upperArmGirth − 0.112 ≤ 0.655` | 手的接缝偏差，左边是偏差的拟合（残差 0.023），上限 0.65 |
| 脚踝上限 | `(0.92·ankleGirth - 0.13·kneeGirth)/footLength ≤ 0.893` | 脚的接缝偏差。这条规则之内仍有不少身体超过 0.52，留着它是给下一条在踝很粗时封顶 |
| 脚踝、脚和膝 | `3.475·a − 1.321·a² − 0.355·kneeGirth − 1.572 ≤ 0.5`，`a = ankleGirth/footLength` | 脚的接缝偏差，左边是偏差的拟合（残差 0.026），上限 0.52。`a` 过 1.32 后左边反而变小，那里由上一条管住 |
| 脚踝下限 | `ankleGirth/footLength ≥ 0.75` | 脚踝折角 |
| 脚和身体 | `footLength / 身高 ≤ 0.159` | 脚踝折角 |

下限规则比求解真正停下的地方略松，因为拟合有离散。规则放过而目标到不了的，界面显示目标和实际。范围是工程边界，不是人群范围，每个性别的表是 `sotaigen/params/rules.py` 的 `RANGES[性别]`，以份额写，乘以这个身体自己的长度才是毫米。

### 按性别

每个身体属于一个性别（`sotaigen/data/bodies/<性别>/`，名字的前缀），只按它自己性别的参数、范围和规则校验。

- **参数和范围：** `sotaigen/params/spec.py` 的 `SEXES` 由 `sex_table(without=…, ranges=…)` 生成：`ALL_SPEC` 去掉这个性别没有的参数，`ranges` 里写的范围替换 `ALL_SPEC` 的。`ALL_SPEC` 的范围是女性压力测试得到的。性别自己的范围要来自这个性别身体的压力测试，并写明依据。
- **规则：** `sotaigen/params/rules.py` 的 `SEX_RULES`、`SEX_SEAM_RULES` 按性别列出要满足的规则表。两个性别限度不同的规则放在 `FEMALE_RULES`、`MALE_RULES`，句子相同、数字不同，每个数字是这个性别的身体验证到的最远值。
- **调用：** 每个校验函数都要给性别名，没有默认值，因为光看数字分不出是谁的身体：`problems`、`spec_problems`、`seam_problems`、`spec_bounds`、`random_spec`、`sample_spec`。`check_bodies` 从名字前缀取性别，参数文件和编辑器请求里写明性别。
- **页面：** `sotaigen_dev/editor/schema.py` 给每个性别生成一份表，页面按当前身体的性别取。性别没有的滑块不显示。规则里只能写算术，`sotaigen.params.rules` 里能读到的数值常数（如从 `sotaigen/params/spec.py` 导入的 `ARM_REST_DEG`）按它的值写进页面的表。

| | 女性 | 男性 |
|---|---|---|
| 乳房（`breast*` 六项，可选） | 有 | 没有，写了就拒绝 |
| 四项肌肉（`pectoralThickness`、`deltoidThickness`、`trapeziusThickness`、`latissimusThickness`） | 有 | 有 |
| 范围 | `ALL_SPEC` 的 | 同女性，只有肩宽上限 0.86 T、腰厚上限 0.44 T（`MALE_RANGES`） |
| 规则 | 全部（`RULES`），加 `FEMALE_RULES`、`BREAST_RULES` | `RULES`，加 `MALE_RULES`：“手臂和窄肩”的线、“手臂和肩高”0.1148、“胯厚下限”常数 0.0262（女性 0.097、0.0318），不含 `BREAST_RULES` |
| 接缝规则 | 全部 | 同女性 |

四项肌肉对两个性别都开放，依据是解剖：四块肌肉男女都有，起止点相同，差别只在量。全身磁共振测量中，男性骨骼肌平均 33.0 kg、女性 21.0 kg，上半身差 40%，下半身差 33%（Janssen 等 2000，[DOI](https://doi.org/10.1152/jappl.2000.89.1.81)）。所以按性别分开的是典型值，不是能不能用。

`BREAST_RULES` 管乳房上的胸肌，编辑器的规则副本只认纯算术，所以分开写。这是模型的限制，不是解剖结论。乳房大部分坐在胸大肌筋膜上，胸肌变厚把乳房一起托前，下缘藏在乳房下面，厚度场乳房上胸板这样做了。

- **厚度场乳房（`breastProjection` 大于 0）：** 胸肌最多 0.0225 T（匀称上约 12 mm，即男性匀称的值）。乳房坐在胸板上，胸板的下缘在乳房下面缓降到下皱襞，不再是一道横棱（[女性胸肌](../parts/chest.md#女性胸肌没有可见下缘)）。组合验证覆盖有乳房的典型身体和 `layerbig`；此上限不代表更厚胸肌已获得几何或视觉验证。
- **没有乳房的身体：** 照常可以用胸肌。

男性除表中差异外沿用女性的范围和规则；可达性与人群覆盖的区别见[验证边界](#范围的依据与验证边界)。

复测，每个性别各跑一次：

```sh
uv run sotaigen-dev stress random --sex female --count 120 --seed 17 --workers 4
```

一个性别约二十分钟，其中抽样在主进程里占一半。结论见[测试身体](presets.md#测试身体)。`oat` 模式仍按内部控制量移动，`spec` 模式读 `{"名字": {"body": {...}}}`。

## 范围的依据与验证边界

当前范围以 `sotaigen/params/spec.py` 的 `ALL_SPEC`、`MALE_RANGES` 为准，组合规则以上面的表为准。它们来自体块的可达性与贴合、绑定试验，属于可调整的工程边界；扫描结果需记录对应版本、抽样范围和检查项。

- **腰厚、大腿围和两个前倾：** 范围依据规则内随机女性身体的求解扫描，并覆盖自带身体。失败主要来自控制量到达求解边界，加迭代次数不能解决；单用腰厚／腰宽或大腿围／胯宽也解释不了可达性。`tests/test_solve.py` 的随机目标测试守住失败比例上限，当前比例须重新运行后报告。较粗大腿与浅胯搭配仍是表达限制，前倾量没有真人统计可比。
- **男性肩宽：** 上限已在 `male-long` 的加宽肩与窄胸组合上验证；其他男性组合和相应 IPC 烘焙仍待验证。女性上限没有随之放宽；真人分布见[男性成人数据](presets.md#男性成人的真人数据)。
- **男性手臂与胯厚：** 用越过女性规则的男性身体验证，分别由 `male-wideheavyarm`、`male-shallowhip` 守边界。上臂围的换算关系男女相同；屈臂围乘 0.85 的数据换算仍未在男性上核对。胯厚规则近似求解能到的地方，规则通过仍可能到不了目标。
- **接缝：** 规则只是预警，多参数靠边仍可能超限。现行身体应重新抽样贴合，不保留旧随机身体的失败比例作为结论。拟合误挡与样本不足见开头的未解决项。
- **人群覆盖：** 当前抽样在规则内进行，规则对合理真人比例的误挡还需另查。手脚、腰宽及粗大腿的范围仍受模型限制；典型身体的统计来源与风格调整见[身体设计](presets.md)。

### 男性头颈配比的待定边界

**颈对头上限 1.58 没有放宽**（[P18](../DECISIONS.md#待用户决定)）。下巴在 1.80 时仍在标志线上，所以越过时先坏的不是颈环，是侧面：`male-balanced`（头深 0.90）从 1.62 起 `tests/test_head.py` 的侧面测试不过，正面下巴下方鼓出小疙瘩，侧面下巴成钩；头深加到 0.95、0.99，或 `male-rounded`（头深 0.97），1.71 的侧面测试过。规则只按颈围比头高，没有头深，这是缺口。待决定的提案是男性上限随头深变：`neckGirth/headHeight ≤ 1.58 + 3·(headDepth − 0.90)`，封顶 1.66；斜率约 3 来自两组身体，代价是头深浅于 0.90 的男性身体比现在更紧，那里没有证据，规则只能写算术，做不到“只放宽”。

真人颈围与头高之比还受所选头身比影响；成人头高的推算和典型身体的调整见[成人头高](presets.md#成人头高)。上述少量身体上的观察是提案的依据，不是新规则已经得到验证。

## 求解起点

`estimated_controls` 用目标除以 `Param.gain` 作起点，骨盆倾角另用拟合。短颈上，下巴下的测量层可能落在斜方肌融合区，颈围起点因此按颈长修正，并限制在求解下限以上（`sotaigen/params/spec.py`）。它只改善迭代起点，不改变量法、输入目标或可达范围；典型倍数的散布须在当前体块上复测。

## 现状

构建指纹用于识别产物是否对应当前输入，`uv run sotaigen-dev build` 可重新构建。参数范围在 `sotaigen/params/spec.py`，校验在 `sotaigen/params/rules.py`；修改后按上文复测。构建成功不替代几何与视觉验收。

**加一个参数：** 在 `sotaigen/params/spec.py` 的 `ALL_SPEC` 里加一行，写明参照面 `basis`，可选项再写 `default`（不写时的值，按单位的份额；`None` 表示身体原样）；某个性别没有它，就写进这个性别 `SEXES` 的 `without`，它不能是必填项。改完重跑 `uv run sotaigen-dev schema --publish`。它若是局部轮廓，写上 `layer`，校准、实测和参考皮肤就会自动把它排除在外；新的轮廓种类要在 `CONTOURS` 或 `PROFILES` 里决定归属。要由实测反求的量，另在 `sotaigen/form/solve.py` 的 `TARGETS`、`LIMITS` 和 `SENSITIVITY` 里各加一项，三者按同一顺序。编辑器的滑块在 `sotaigen/data/sliders.json`，滑块的起点取参数表的缺省值。

**压力测试的入口：** `sotaigen_dev/stress.py`。`oat` 逐个参数从锚点往外二分，`random` 从一个性别的范围表随机组合（`--sex`），`spec` 跑文件里的具名参数，`report` 列失败。种子固定，流水线没有随机数，每行结果都带完整参数，可以单独重建。结果在 `build/reports/stress/`。

**改身体：** 增删身体不会让测试失败，但文档里写到身体名的地方要一起改。名字是 `性别-组`，性别和组的名字都不要带连字符：`sotaigen/pipeline.py` 的 `select` 和 `sotaigen_dev/review.py` 的 `catalog` 按连字符拆名字。新增参数要同时写进范围表，否则校验拒绝。身体分典型身体和测试身体，见[预设](presets.md)。

## 参数校验与编辑工具

### 参数校验

校验分四层，由便宜到贵，都在 `sotaigen/params/rules.py` 和 `sotaigen/form/checks.py`：

1. **范围：** 每个参数单独查。
2. **规则：** 参数之间的关系，管的是贴合和摆姿势才会暴露的问题。
3. **接缝规则：** 预测 `tests/test_wrap.py` 的接缝上限。那些上限要贴合之后才量得到，所以对没贴合过的参数，这是唯一的预警。它偏保守，也不精确。
4. **曲面：** `field_failures(field_measures(p))` 校准后实测，和 `tests/test_body.py` 是同一件事；结果受场与采样方法约束。腰宽、胯宽够不够得着取决于多个参数的非线性组合，线性规则分不开，所以直接量。

`check_bodies` 只查前两层：接缝由 `tests/test_wrap.py` 直接量，曲面太慢，而且 `tests/test_body.py` 已经覆盖。`sample_spec` 抽随机身体时可以加上第 4 层。

多个参数同时靠边时，范围内的组合仍可能略超接缝上限，线性规则分不出来，界面会报告实测问题。观感按 [WORKFLOW](../WORKFLOW.md#4-看图) 看图验收。

### 预设

典型身体与测试身体各有独立用途，来源和验证边界见[身体的设计](presets.md#身体的设计)。

### 成熟度与头身比

成熟感来自胸、很细的腰和强的腰背曲线，和头身比本身无关。所以成熟度和头身比分开：典型身体按真人年龄段各取比例，乳房和腰背曲线各自是参数。

### 风格控制

**风格通过现有参数表达。** 体型比例独立调节，分明程度和融合（`definition`、`softness`）、软组织、末端收窄和曲线用于风格调整。编辑器的 `tone`、`taper`、`curve` 宏滑块来自这个划分，它们的系数没有看图定过。

### 哪些参数随头身比共变

只借 Huxley 的形式 `y = b·x^a`，数据是手工设计的三个中性体型，不是生长数据，所以这是把设计意图写成式子，不是发现。长度都以 T 为单位，T 本身已经随头身变，所以 `a≈0` 的是跟着 T 走的。指数在 `sotaigen/data/sliders.json` 的 `heads.laws`，编辑器的头身比滑块和 `sotaigen/params/tools.py` 的 `with_head_count` 读它。

| 分组 | 参数（括号里是指数 a） | 处理 |
|---|---|---|
| 一个幂律就够（最大偏差 ≤5%） | `kneeGirth`（-0.62）、`ankleGirth`（-0.60）、`calfGirth`（-0.58）、`waistDepth`（-0.53）、`wristGirth`（-0.48）、`upperArmGirth`（-0.42）、`headWidthRatio`（-0.39）、`handLength`（0.34）、`legClearance`（-0.30） | 随头身共变 |
| 明显共变但一条幂律不够（偏差 7–17%） | `neckLength`（1.53）、`jawBreadthRatio`（-0.95）、`lowerArmLength`（0.86）、`upperArmLength`（0.53）、`crotchHeight`（0.44） | 随头身共变 |
| 和头身基本无关（\|a\| < 0.25） | `biacromialBreadth`、`chestBreadth`、`chestDepth`、`hipBreadth`、`hipDepth`、`waistBreadth`、`headDepth`、`footLength`、`upperThighGirth`、`neckGirth` | 不动，它们是体型的个性 |
| 不是头身的事 | `upperTrunkLean`、`pelvisLean`、乳房 | 归年龄感或成熟度 |
| 自由 | 身高、`acromionRise` | 不动 |

只改头身比而不共变时，大头身是气球头配细长脖子，小头身是方块身体；用这些幂律后中间的头身也连成一条顺的序列。压力测试也独立指向共变：手腕和手、脚踝和脚的比例要留在窄带里，否则接缝先坏。规格存的是实测值，头身比是读数，幂律只用在头身比滑块上。
