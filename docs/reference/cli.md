# 命令行

[English](../en/reference/cli.md)

冻结的约定 7：`sotaigen` 的子命令、参数和退出码。输出的文字不在约定之内，要读结果就读文件，或者用 Python 接口。金样在 `tests/contracts/cli.json`。

<!-- generated: cli -->
| 子命令 | 参数 |
|---|---|
| `sotaigen blend` | `file`, `out`, `--t-pose`, `--arm-abduction`, `--ankle-spacing`, `--no-palms-down` / `--palms-down`, `--no-straight-arms` / `--straight-arms`, `--past-limits`, `--for-head-swap` |
| `sotaigen check` | `file` |
| `sotaigen glb` | `file`, `out` |
| `sotaigen scale` | `file`, `out`, `--stature`, `--heads`, `--torso` |
| `sotaigen swap-head` | `body`, `head`, `out`, `--scale`, `--parts-only`, `--keep-parts`, `--report` |

| 退出码 | 含义 |
|---|---|
| 0 | 完成 |
| 1 | 身体做不成或写不出：规则不过、网格损坏 |
| 2 | 命令行或文件不对 |
| 3 | 缺 Blender、Blender 低于最低版本，或缺 sotaigen-blender |
<!-- end generated -->

- `check FILE`：校验参数文件（范围、规则、曲面、贴合），没问题时打印 `valid`。
- `glb FILE OUT`：写绑定好的 GLB。
- `blend FILE OUT`：写绑定好的 `.blend`，要 Blender 5.2 或更新的版本和 `sotaigen-blender`。换静止姿势的参数（`--t-pose`、`--arm-abduction` 度、`--ankle-spacing` mm、`--palms-down`、`--straight-arms`、`--past-limits`）见 [.blend](blend.md)；`--for-head-swap` 放行对本工具的头太细、但要换头的颈。
- `swap-head BODY HEAD OUT`：把 `.blend` 的头换成别处做的头，见[换头的接口](head-interface.md)。
- `scale FILE OUT`：按身高（`--stature` mm）、头身比（`--heads`）或躯干长（`--torso` mm）改写参数文件。

缺 Blender 时退出码 3 发生在开始计算之前。
