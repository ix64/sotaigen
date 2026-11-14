# 诊断

[English](../en/reference/diagnostics.md)

冻结的约定 4 的一部分：检查找到的每一条都是一个诊断 `{"code", "level", "values"}`。代码和 `values` 的键是约定；英文句子只是渲染，措辞不是约定。金样在 `tests/contracts/diagnostics.json`。

- **`level`：** `error`（参数或文件不成立，不生成）、`broken`（网格坏了，不写文件）、`warning`（随文件走，写进 `quality.found`）。
- **单位：** 长度是 mm，角度是度，份额和比值是纯数。
- **加代码不算破坏约定，** 改名、删除或改 `values` 的键算。读者遇到不认识的代码要照样显示代码和值。
- `rule.<名字>` 是一条跨参数的规则不过，`short` 是缺多少（规则读的份额或比值）；规则本身不在约定之内，会随版本增删。`surface.<名字>` 是曲面的不变量不成立。

<!-- generated: diagnostics -->
| 代码 | `values` 的键 |
|---|---|
| `body.unknown` | `known`, `name` |
| `contact.failed` | `detail`, `group` |
| `file.format` | `format` |
| `file.newer` | `latest`, `version` |
| `file.no_sex` | — |
| `file.not_json` | `detail` |
| `file.not_object` | — |
| `file.not_parameters` | — |
| `file.unreadable` | `detail` |
| `file.version` | `version` |
| `foot.off_ground` | `limit`, `mm` |
| `head_swap.collapsed` | `count`, `pose` |
| `head_swap.kept_parts` | `group`, `objects` |
| `head_swap.openings` | `count` |
| `head_swap.touch_stale` | — |
| `head_swap.unplaced` | `bones` |
| `param.missing` | `key` |
| `param.not_number` | `key`, `value` |
| `param.outside` | `high`, `key`, `low`, `value` |
| `param.sex` | `sex` |
| `param.unknown` | `key` |
| `pose.collapsed` | `count`, `pose` |
| `rest.past_limit` | `axis`, `degrees`, `high`, `limb`, `low` |
| `rest.unclean` | `axis`, `degrees`, `high`, `limb`, `low` |
| `rule.<name>` | `short` |
| `seam.crease` | `degrees`, `limit`, `part` |
| `seam.miss` | `limit`, `part`, `share` |
| `skin.open` | `edges` |
| `surface.<name>` | — |
| `target.missed` | `asked`, `key`, `reached` |
| `target.unreachable` | `detail` |
| `touch.unreached` | `arrivalMm`, `contact`, `gapMm`, `palmDegrees`, `throughHead` |
| `uv.anisotropy` | `limit`, `value` |
| `uv.stretch` | `limit`, `value` |
| `wrap.flipped` | `at`, `count` |
| `wrap.midline` | `at`, `count` |
<!-- end generated -->
