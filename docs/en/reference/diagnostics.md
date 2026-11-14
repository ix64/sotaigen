# Diagnostics

[中文](../../reference/diagnostics.md)

Part of contract 4: everything a check finds is a diagnostic `{"code", "level", "values"}`. The codes
and the keys of `values` are contract; the English sentence is only a rendering, and its wording is not.
The golden copy is `tests/contracts/diagnostics.json`.

- **`level`:** `error` (the parameters or the file don't hold up; nothing is made), `broken` (the mesh is
  broken; no file is written), `warning` (travels with the file, in `quality.found`).
- **Units:** lengths in mm, angles in degrees, shares and ratios as plain numbers.
- **Adding a code is not breaking;** renaming or removing one, or changing the keys of its `values`, is.
  A reader shows a code it doesn't know as its code and values.
- `rule.<name>` is a rule across parameters that fails, `short` how far short it falls (in the share or
  ratio the rule reads); the rules themselves are not contract and come and go with versions.
  `surface.<name>` is an invariant of the surface that doesn't hold.

<!-- generated: diagnostics -->
| Code | Values |
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
