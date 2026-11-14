# 模板来源记录

`base-mesh.json` 是全部体型共用的固定拓扑模板：CC0 人体基础网格在颈部以下的部分，肘、膝两侧各加了 2 个环，肩、腋加了 5 个环，单位 mm，保持源文件自身的比例。

## 来源

| 项目 | 内容 |
|---|---|
| 素材 | Blender Studio，Human Base Meshes v1.4.1 |
| 对象 | `.blend` 里的 `GEO-body_female_stylized` |
| 官方页面 | <https://www.blender.org/download/demo-files/> |
| 官方下载 | <https://www.blender.org/download/demo/asset-bundles/human-base-meshes/human-base-meshes-bundle-v1.4.1.zip> |
| 源文件 | `human_base_meshes_bundle.blend`，49420489 字节 |
| 源文件 SHA-256 | `3c121505651140ceb4d69fd1d8923f7788ffadd81672f5be14845a5f2c75c137` |
| 授权 | CC0 |

**CC0 依据：** 上述官方页面在 Human Base Meshes v1.4.1 条目下标注 “49 MB – CC0”（2026-11-04 查看）。

`.blend` 不入库。本次核对的是 `.blend` 文件本身的哈希，没有重新下载压缩包比对。重新提取前请先核对上面的 SHA-256。

## 文件内容

由 `extract.py` 和 `loops.py` 生成，不手工编辑。

| 字段 | 含义 |
|---|---|
| `format`、`version` | `"sotaigen/template"` 和 1，`sotaigen.template.load_template` 只认这一组 |
| `note` | 一句话说明 |
| `source` | 素材名、对象名、授权、官方页面和 `.blend` 的 SHA-256 |
| `landmarks` | 从拓扑里找不出、代码要用的模板事实，用模板自己的编号和毫米，都在穿着者左侧，取值写在 `extract.py` 的 `LANDMARKS`：`shoulderPivot` 上臂的转心（贴合把它对到身体的肩关节）；`vertices` 腋部的顶点，`axillaFrontFold`、`axillaFloor`、`axillaBackFold` 是前褶顶、悬着的腋底和后褶顶，`axillaFrontChest`、`axillaFrontArm`、`axillaBackChest`、`axillaBackArm` 是两条褶的支撑在胸壁和上臂上的端点；`anklePerFoot`、`footInnerPerFoot` 是踝离脚底的高度和脚伸到踝内侧的距离，都是脚长的份额 |
| `openings` | 开口的边界顶点，目前只有颈部 `neck`，36 个顶点，就是颈环 |
| `rings` | 20 个关节环的顶点序号，只有左侧 |
| `segments` | 每个顶点的解剖分段，由源网格的 face set 决定。区域（`trunk`、`neck`、`leg`、`arm`、`hand`、`foot`）由分段推出，不存在文件里，见 `sotaigen.template.REGION_OF` |
| `verts` | 9363 个顶点，单位 mm，保留两位小数，顺序见下面“编号” |
| `faces` | 9344 个面，全是四边形 |
| `refined` | `besideRings` 在哪个关节环旁边加了几条环、新顶点的起始序号 `from`、每个新顶点平分的边的两端 `parents`、加环前的面 `faces`，`batches` 按先后列出每批加了多少个顶点，后加的批次还记着切了哪些带（`rungs`）和怎样切的（`splits`） |

## 编号

顶点的编号只追加，这是对外约定（[数据约定](../docs/ARCHITECTURE.md#34-数据约定)的“网格顺序”）：

1. 源网格的 8781 个顶点，序号是 `extract.py` 在加环前的面上编出来的。
2. 带头的皮肤（`sotaigen.form.head.extended`）里，头的 3313 个顶点接在源网格之后。模板文件本身没有头。
3. 加的环一批接一批排在最后：先是 `extract.py` 在肘、膝旁加的 400 个（`batches` 的第一批 `joints`），再是 `loops.py` 的 `axilla` 182 个。

所以以后再加环不移动任何已有的顶点，头和颈环的序号也不随加环改变。`landmarks` 的顶点都在源网格里。

## 重新提取

以下命令在仓库根目录运行，使用 `PATH` 中的 `blender`；如果 Blender 不在 `PATH` 中，先将环境变量 `BLENDER` 设为可执行文件的路径。把尖括号占位符替换为实际文件路径，路径有空格时加引号。

源文件是外部文件，必须带 `--disable-autoexec`，防止其中的脚本自动运行。

```sh
"${BLENDER:-blender}" -b --factory-startup --disable-autoexec \
    --python template/extract.py -- <human_base_meshes_bundle.blend>
```

输出写到 `sotaigen/data/template/base-mesh.json`，只有源网格和肘、膝的环。Blender 5.2.3 运行上面的命令得到的文件 SHA-256 是 `8cedf55c89e65d3e9f2d08011cdd8c8de7bba9d1bebe9bde1e0b3407fa40270a`。再按下面“UV”一节生成 `uv.npz`，最后运行 `uv run template/loops.py` 加上其余的环，得到现在这份，SHA-256 是 `ed39e02a4523ef9dfc2a559db3158669b2588321a46dba0e268952d584fba706`。

**加环：** 分两处。

- `extract.py` 的 `REFINE` 写在哪个关节环旁边加几条（现在是 `{"elbow": 2, "knee": 2}`）：每一侧最近的几条四边形带各从中间切一刀，所以 2 表示这个关节加 4 个环。源网格的顶点先按原来的方式编号，新顶点接在后面。`REFINE` 为空时只有源网格的 8781 个顶点（SHA-256 `f6af6bfe…`）。它的新顶点按面的顺序编号，所以以后不再往 `REFINE` 里加。
- 以后的环都加在 `loops.py` 的 `BATCHES` 里，一批一行，写每条要切的四边形带上的一条横边。它只读 JSON，不用 Blender：沿带走一圈，每条横边中点加一个顶点，带上每个四边形一分为二，UV 取横边两端的中点。新顶点一批接一批排在最后，所以加环从不移动已有的顶点。带必须闭合、不能自交、不能穿过颈部开口，否则报错。现在有一批 `axilla`：每侧肩缝的环和躯干一侧绕肩的环，加一条穿过两侧腋底的躯干带，共 182 个顶点（[肩腋](../docs/parts/shoulder-axilla.md#腋下加环)）。

## UV

`uv.npz` 是全身一张图集，由同一个 `.blend` 里的人工 UV 生成，由 `extract_uv.py` 写出，不手工编辑。头部和模板的拓扑变了就要重新生成，否则 `sotaigen.uv.load_uv` 报错：

```sh
"${BLENDER:-blender}" -b --factory-startup --disable-autoexec \
    --python template/extract_uv.py -- <human_base_meshes_bundle.blend> <shape.npy>
```

`<shape.npy>` 是量密度和展开头部用的形状。UV 由全部预设共用，所以现在这份用的是 12 个典型身体的 `wrapped.npy` 逐顶点平均后存的 `.npy`；只用一个身体时，别的身体可能超出 `tests/test_uv.py` 的上限。肘、膝新环的 UV 由 Blender 在源 UV 上插值，`extract_uv.py` 用 `extract.py` 同样切一遍，切出的模板必须和 `base-mesh.json` 相同，所以它要在 `loops.py` 之前运行。`loops.py` 再给后加的环补上 UV。

模板顶点的顺序是全系统的约定，权重方案和测试都依赖它。如果重新提取后文件内容变了，所有预设都要重新生成。
