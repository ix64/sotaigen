# 动作数据来源

`clips/*.json` 由 `sotaigen_dev/motion/clips.py fetch` 从下列原始文件缩减而来：从每秒 120 帧的 AMC 保留每秒 60 帧的世界空间骨骼旋转及根位移。评审仍按每秒 10 帧编号，例如第 24 帧对应 2.4 秒。原始文件不进仓库，下载到 `build/reports/motions/raw/`，下载后按下表的 SHA-256 校验。片段是本工具的测试数据，不是给别人用的格式：文件里的 `format`、`version`（`sotaigen/motion-clip`、1）只让读取时认出别的写法。

**来源：** [CMU Graphics Lab Motion Capture Database](http://mocap.cs.cmu.edu/)。

**授权：** 站点首页写明 “This dataset of motions is free for all uses.”。不是 CC0，但没有限制用途。站点建议的致谢：数据来自 mocap.cs.cmu.edu，数据库由 NSF EIA-0196217 资助。

| 片段 | 原始文件 | 内容 | 取用范围 |
|---|---|---|---|
| `walk` | `subjects/35/35_01.amc`，骨架 `35.asf` | 走 | 全部 |
| `run` | `subjects/35/35_17.amc`，骨架 `35.asf` | 跑 | 全部 |
| `sitStool` | `subjects/13/13_01.amc`，骨架 `13.asf` | 坐高凳再站起 | 前 6 秒 |
| `sitLow` | `subjects/13/13_04.amc`，骨架 `13.asf` | 坐矮凳，手托下巴 | 前 8 秒 |
| `wave` | `subjects/13/13_27.amc`，骨架 `13.asf` | 指挥交通、挥手、指点 | 前 12 秒 |

根位移取 AMC 的前三个根通道，减去首帧位置，换成项目坐标后，以源左股骨与胫骨长度之和归一化。目标模型按自己的腿长还原毫米，整段沿用首帧朝向，保留后续转身。ASF 骨长和 AMC 根位移使用同一单位。[CMU 文件格式说明](https://mocap.cs.cmu.edu/info.php)

接触区间由源骨架正向运动学算出的踝和前掌位置估计，未使用测力板。每个标记同时满足低速（每秒不超过 0.42 倍腿长）、低位（距本段高度第 10 百分位不超过 0.065 倍腿长）的相邻两帧才计为支撑。重定向时用目标网格的足底顶点校正地面，双点支撑时保留踝到前掌的水平朝向，在水平地面上校准足骨与趾骨的静止方向。摆腿沿用相对该方向的转动，脚跟或前掌单独支撑时允许绕接触处转动。远端旋转使用居中短窗抑制短促反转，髋膝两段 IK 保持骨长。方法参考 Kovar、Schreiner、Gleicher 的分段接触约束，当前实现只处理水平地面。[Footskate Cleanup for Motion Capture Editing](https://graphics.cs.wisc.edu/Papers/2002/KSG02/)

原始文件的 SHA-256（`sotaigen_dev/motion/clips.py` 的 `FILES` 是同一张表，`tests/test_motion_clips.py` 核对两边相同）：

| 文件 | SHA-256 |
|---|---|
| `35.asf` | `2a8e2eda3c0d7d828566b2a9a8ab36b2b8b3864110574e8b73c8f069fded416c` |
| `35_01.amc` | `0743f4ea48e7e199cd56b2810b5ce81f8ede08d32ff79aa4e363c44cc4fe33aa` |
| `35_17.amc` | `8096cd53878242dbc35e987288cef6ae81d4e1c0e207086639bbed5cc06c8fee` |
| `13.asf` | `eb33390bcbccb739d7622bd583fd9be55020868f68689aa6f2060ad688df10dc` |
| `13_01.amc` | `e65c743d86ea97a6c1b27d38a5ad95529c08042075eb490e54953030b2b230be` |
| `13_04.amc` | `bc5f5e5b6378413457f782e96c2beedf70c5abc6bdcabca513f8f5b0d966d7e9` |
| `13_27.amc` | `8679214b96f59a8c49210131373bca9303767e6780073ef35a3b36045955ccfe` |

**没有手指：** 这些数据的手只有一节，手部姿势是手写的，见 `sotaigen_dev/motion/clips.py` 的 `HAND_POSES`。

**手腕有固定偏差：** 手部标记不准，走路时同一受试者左腕 −49°、右腕 −24°，而绑定是对称的。打到 `.blend` 上时每段动作每只手减去中位数（[真实动作](../docs/systems/blend.md#真实动作)）。
