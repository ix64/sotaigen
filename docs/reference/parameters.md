# 参数文件

[English](../en/reference/parameters.md)

冻结的约定 3：文件的结构、参数名和单位。参数的范围和规则不在约定之内：它们随形体改进，同一份文件在新版本里可能变得无效，这时读者照样能打开，并说清哪一项不行。金样在 `tests/contracts/parameters.json`；JSON Schema 随包发布，在 `sotaigen/data/params.schema.json`（只管形状和范围，管不了跨参数的规则）。

## 结构

```json
{
  "format": "sotaigen/params",
  "version": 1,
  "sex": "female",
  "base": { "name": "female-long" },
  "generator": "sotaigen 0.1.0",
  "body": { "headHeight": 254.545, "torsoLength": 390.091, "...": "..." }
}
```

| 字段 | 内容 |
|---|---|
| `format` | 总是 `"sotaigen/params"` |
| `version` | 文件格式的版本，现在是 1 |
| `sex` | `"female"` 或 `"male"`，决定有哪些参数、用哪套范围和规则 |
| `body` | 参数，按名字 |
| `base` | 可选：参数从哪个身体起步，`{"name": …}` |
| `generator` | 可选：写文件的程序，`"sotaigen <版本>"` |

## 读文件的规则

| 情况 | 读者怎么做 |
|---|---|
| `format` 不对，或不是 JSON | 拒绝 |
| `version` 比读者新 | 拒绝，并说明 |
| 没有 `sex`，或它不是字符串 | 拒绝 |
| `sex` 是没有的性别 | 能打开，校验报告这个性别不存在 |
| 顶层有不认识的字段 | 保留，写回时原样带上 |
| `body` 缺必填参数，或有不认识的参数 | 能打开，校验不过，不生成；缺项不补默认值 |
| 超出范围或违反规则 | 能打开，校验不过，不夹紧 |

- **`version` 只在结构或含义变时加一。** 范围和规则变不加；加可选参数不加（缺了它身体不变）；去掉或改动必填参数要加。读者把旧版本迁移到新版本，不拒绝。
- **文件只存参数，** 不存滑块、头身比这类编辑手段。
- **数字一律按浮点写**（`124.0`），JSON 不区分 124 和 124.0。

## 单位

- `mm` 的参数是毫米；`比值` 的参数是纯数。
- 可选参数可以不写，不写时身体按它的默认值做，默认值见 JSON Schema。
- “参照面”说明这个值量在哪里。参照面是“内部控制量”的几项没有量法，同一个值在不同版本里做出的形状可以不同。

## 女性的参数

<!-- generated: parameters-female -->
| 参数 | 单位 | 必填 | 参照面 | 是什么 |
|---|---|---|---|---|
| `headHeight` | mm | 是 | 骨骼据以放置的地标之间 | 头高 |
| `neckLength` | mm | 是 | 骨骼据以放置的地标之间 | 颈长 |
| `torsoLength` | mm | 是 | 骨骼据以放置的地标之间 | 躯干长 |
| `crotchHeight` | mm | 是 | 骨骼据以放置的地标之间 | 裆高 |
| `kneeHeight` | mm | 是 | 骨骼据以放置的地标之间 | 膝高 |
| `ankleHeight` | mm | 是 | 骨骼据以放置的地标之间 | 脚踝高度 |
| `biacromialBreadth` | mm | 是 | 骨骼据以放置的地标之间 | 肩峰间距 |
| `acromionRise` | mm | 是 | 骨骼据以放置的地标之间 | 肩峰高差 |
| `waistRise` | mm | 是 | 骨骼据以放置的地标之间 | 腰地标在裆上的高度 |
| `chestBreadth` | mm | 是 | 成品曲面，由求解器反求 | 胸廓宽 |
| `chestDepth` | mm | 是 | 成品曲面，由求解器反求 | 胸廓厚 |
| `waistBreadth` | mm | 是 | 曲面在地标处的宽度，误差 2% 以内 | 腰宽 |
| `waistDepth` | mm | 是 | 成品曲面，由求解器反求 | 腰厚，腰地标高度上的前后距离 |
| `hipBreadth` | mm | 是 | 曲面在地标处的宽度，误差 2% 以内 | 髋宽 |
| `hipDepth` | mm | 是 | 成品曲面，由求解器反求 | 髋厚 |
| `upperArmLength` | mm | 是 | 骨骼据以放置的地标之间 | 上臂长 |
| `lowerArmLength` | mm | 是 | 骨骼据以放置的地标之间 | 前臂长 |
| `handLength` | mm | 是 | 骨骼据以放置的地标之间 | 手长 |
| `footLength` | mm | 是 | 骨骼据以放置的地标之间 | 脚长 |
| `upperArmGirth` | mm | 是 | 成品曲面，由求解器反求 | 上臂围 |
| `wristGirth` | mm | 是 | 成品曲面，由求解器反求 | 腕围 |
| `neckGirth` | mm | 是 | 成品曲面，由求解器反求 | 颈围 |
| `upperThighGirth` | mm | 是 | 成品曲面，由求解器反求 | 大腿上段围度 |
| `kneeGirth` | mm | 是 | 成品曲面，由求解器反求 | 膝围 |
| `calfGirth` | mm | 是 | 成品曲面，由求解器反求 | 小腿围 |
| `ankleGirth` | mm | 是 | 成品曲面，由求解器反求 | 踝围 |
| `legClearance` | mm | 是 | 内部控制量，同一个值做出什么随形体版本变 | 双腿间隙 |
| `upperTrunkLean` | mm | 是 | 成品曲面，由求解器反求 | 上身后靠 |
| `pelvisLean` | mm | 是 | 成品曲面，由求解器反求 | 骨盆前倾 |
| `headWidthRatio` | 比值 | 是 | 同一部位两个尺寸之比 | 头宽 |
| `headDepth` | 比值 | 是 | 内部控制量，同一个值做出什么随形体版本变 | 头深 |
| `jawBreadthRatio` | 比值 | 是 | 同一部位两个尺寸之比 | 下颌 |
| `definition` | 比值 | 是 | 内部控制量，同一个值做出什么随形体版本变 | 肌肉体块的分明程度 |
| `softness` | 比值 | 是 | 内部控制量，同一个值做出什么随形体版本变 | 体块之间融合的柔和程度 |
| `outerThighArc` | mm | 否 | 曲面离两处之间弦线的偏移 | 大腿：外侧弧度 |
| `abdomenFrontArc` | mm | 否 | 曲面离两处之间弦线的偏移 | 腰腹：腹前弧 |
| `lumbarHollow` | mm | 否 | 曲面离两处之间弦线的偏移 | 腰腹：后腰凹入 |
| `waistContourShift` | mm | 否 | 内部控制量，同一个值做出什么随形体版本变 | 腰腹：腰腹过渡高低 |
| `shoulderSetback` | mm | 否 | 曲面比不加这一项的同一身体移动多少 | 肩后移 |
| `thighFrontShift` | mm | 否 | 曲面比不加这一项的同一身体移动多少 | 大腿：大腿前侧 |
| `gluteProjection` | mm | 否 | 曲面比不加这一项的同一身体移动多少 | 臀部：翘度 |
| `glutePeakShift` | mm | 否 | 曲面比不加这一项的同一身体移动多少 | 臀部：最高点高低 |
| `gluteUpperFullness` | mm | 否 | 曲面比不加这一项的同一身体移动多少 | 臀部：上臀饱满度 |
| `gluteLowerFullness` | mm | 否 | 曲面比不加这一项的同一身体移动多少 | 臀部：下臀饱满度 |
| `outerHipShift` | mm | 否 | 曲面比不加这一项的同一身体移动多少 | 臀部：外髋高低 |
| `gluteFoldLength` | mm | 否 | 这一项所加形状的尺寸 | 臀部：臀下沟长度 |
| `gluteFoldDepth` | mm | 否 | 曲面比不加这一项的同一身体移动多少 | 臀部：臀下沟深浅 |
| `groinDepth` | mm | 否 | 曲面比不加这一项的同一身体移动多少 | 腹股沟：腹股沟深浅 |
| `groinRelief` | mm | 否 | 曲面比不加这一项的同一身体移动多少 | 腹股沟：周围起伏 |
| `pectoralThickness` | mm | 否 | 内部控制量，同一个值做出什么随形体版本变 | 胸肌 |
| `deltoidThickness` | mm | 否 | 内部控制量，同一个值做出什么随形体版本变 | 三角肌 |
| `trapeziusThickness` | mm | 否 | 内部控制量，同一个值做出什么随形体版本变 | 斜方肌 |
| `latissimusThickness` | mm | 否 | 内部控制量，同一个值做出什么随形体版本变 | 背阔肌 |
| `breastProjection` | mm | 否 | 组织离其下曲面最远多少 | 胸部：突出度 |
| `breastBaseBreadth` | mm | 否 | 这一项所加形状的尺寸 | 胸部：基底宽 |
| `breastFoldDrop` | mm | 否 | 这一项所加形状的尺寸 | 胸部：下缘高度 |
| `breastUpperPoleRatio` | 比值 | 否 | 同一部位两个尺寸之比 | 胸部：上半饱满度 |
| `breastGap` | mm | 否 | 这一项所加形状的尺寸 | 胸部：乳间距 |
| `breastApexRatio` | 比值 | 否 | 同一部位两个尺寸之比 | 胸部：最前点偏外 |
<!-- end generated -->

## 男性的参数

男性没有胸部的几项，其余和女性相同。

<!-- generated: parameters-male -->
| 参数 | 单位 | 必填 | 参照面 | 是什么 |
|---|---|---|---|---|
| `headHeight` | mm | 是 | 骨骼据以放置的地标之间 | 头高 |
| `neckLength` | mm | 是 | 骨骼据以放置的地标之间 | 颈长 |
| `torsoLength` | mm | 是 | 骨骼据以放置的地标之间 | 躯干长 |
| `crotchHeight` | mm | 是 | 骨骼据以放置的地标之间 | 裆高 |
| `kneeHeight` | mm | 是 | 骨骼据以放置的地标之间 | 膝高 |
| `ankleHeight` | mm | 是 | 骨骼据以放置的地标之间 | 脚踝高度 |
| `biacromialBreadth` | mm | 是 | 骨骼据以放置的地标之间 | 肩峰间距 |
| `acromionRise` | mm | 是 | 骨骼据以放置的地标之间 | 肩峰高差 |
| `waistRise` | mm | 是 | 骨骼据以放置的地标之间 | 腰地标在裆上的高度 |
| `chestBreadth` | mm | 是 | 成品曲面，由求解器反求 | 胸廓宽 |
| `chestDepth` | mm | 是 | 成品曲面，由求解器反求 | 胸廓厚 |
| `waistBreadth` | mm | 是 | 曲面在地标处的宽度，误差 2% 以内 | 腰宽 |
| `waistDepth` | mm | 是 | 成品曲面，由求解器反求 | 腰厚，腰地标高度上的前后距离 |
| `hipBreadth` | mm | 是 | 曲面在地标处的宽度，误差 2% 以内 | 髋宽 |
| `hipDepth` | mm | 是 | 成品曲面，由求解器反求 | 髋厚 |
| `upperArmLength` | mm | 是 | 骨骼据以放置的地标之间 | 上臂长 |
| `lowerArmLength` | mm | 是 | 骨骼据以放置的地标之间 | 前臂长 |
| `handLength` | mm | 是 | 骨骼据以放置的地标之间 | 手长 |
| `footLength` | mm | 是 | 骨骼据以放置的地标之间 | 脚长 |
| `upperArmGirth` | mm | 是 | 成品曲面，由求解器反求 | 上臂围 |
| `wristGirth` | mm | 是 | 成品曲面，由求解器反求 | 腕围 |
| `neckGirth` | mm | 是 | 成品曲面，由求解器反求 | 颈围 |
| `upperThighGirth` | mm | 是 | 成品曲面，由求解器反求 | 大腿上段围度 |
| `kneeGirth` | mm | 是 | 成品曲面，由求解器反求 | 膝围 |
| `calfGirth` | mm | 是 | 成品曲面，由求解器反求 | 小腿围 |
| `ankleGirth` | mm | 是 | 成品曲面，由求解器反求 | 踝围 |
| `legClearance` | mm | 是 | 内部控制量，同一个值做出什么随形体版本变 | 双腿间隙 |
| `upperTrunkLean` | mm | 是 | 成品曲面，由求解器反求 | 上身后靠 |
| `pelvisLean` | mm | 是 | 成品曲面，由求解器反求 | 骨盆前倾 |
| `headWidthRatio` | 比值 | 是 | 同一部位两个尺寸之比 | 头宽 |
| `headDepth` | 比值 | 是 | 内部控制量，同一个值做出什么随形体版本变 | 头深 |
| `jawBreadthRatio` | 比值 | 是 | 同一部位两个尺寸之比 | 下颌 |
| `definition` | 比值 | 是 | 内部控制量，同一个值做出什么随形体版本变 | 肌肉体块的分明程度 |
| `softness` | 比值 | 是 | 内部控制量，同一个值做出什么随形体版本变 | 体块之间融合的柔和程度 |
| `outerThighArc` | mm | 否 | 曲面离两处之间弦线的偏移 | 大腿：外侧弧度 |
| `abdomenFrontArc` | mm | 否 | 曲面离两处之间弦线的偏移 | 腰腹：腹前弧 |
| `lumbarHollow` | mm | 否 | 曲面离两处之间弦线的偏移 | 腰腹：后腰凹入 |
| `waistContourShift` | mm | 否 | 内部控制量，同一个值做出什么随形体版本变 | 腰腹：腰腹过渡高低 |
| `shoulderSetback` | mm | 否 | 曲面比不加这一项的同一身体移动多少 | 肩后移 |
| `thighFrontShift` | mm | 否 | 曲面比不加这一项的同一身体移动多少 | 大腿：大腿前侧 |
| `gluteProjection` | mm | 否 | 曲面比不加这一项的同一身体移动多少 | 臀部：翘度 |
| `glutePeakShift` | mm | 否 | 曲面比不加这一项的同一身体移动多少 | 臀部：最高点高低 |
| `gluteUpperFullness` | mm | 否 | 曲面比不加这一项的同一身体移动多少 | 臀部：上臀饱满度 |
| `gluteLowerFullness` | mm | 否 | 曲面比不加这一项的同一身体移动多少 | 臀部：下臀饱满度 |
| `outerHipShift` | mm | 否 | 曲面比不加这一项的同一身体移动多少 | 臀部：外髋高低 |
| `gluteFoldLength` | mm | 否 | 这一项所加形状的尺寸 | 臀部：臀下沟长度 |
| `gluteFoldDepth` | mm | 否 | 曲面比不加这一项的同一身体移动多少 | 臀部：臀下沟深浅 |
| `groinDepth` | mm | 否 | 曲面比不加这一项的同一身体移动多少 | 腹股沟：腹股沟深浅 |
| `groinRelief` | mm | 否 | 曲面比不加这一项的同一身体移动多少 | 腹股沟：周围起伏 |
| `pectoralThickness` | mm | 否 | 内部控制量，同一个值做出什么随形体版本变 | 胸肌 |
| `deltoidThickness` | mm | 否 | 内部控制量，同一个值做出什么随形体版本变 | 三角肌 |
| `trapeziusThickness` | mm | 否 | 内部控制量，同一个值做出什么随形体版本变 | 斜方肌 |
| `latissimusThickness` | mm | 否 | 内部控制量，同一个值做出什么随形体版本变 | 背阔肌 |
<!-- end generated -->
