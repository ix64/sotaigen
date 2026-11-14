# Command line

[中文](../../reference/cli.md)

Contract 7: `sotaigen`'s subcommands, arguments and exit codes. The output's wording is not: to read a
result, read the file, or use the Python API. The golden copy is `tests/contracts/cli.json`.

<!-- generated: cli -->
| Command | Arguments |
|---|---|
| `sotaigen blend` | `file`, `out`, `--t-pose`, `--arm-abduction`, `--ankle-spacing`, `--no-palms-down` / `--palms-down`, `--no-straight-arms` / `--straight-arms`, `--past-limits`, `--for-head-swap` |
| `sotaigen check` | `file` |
| `sotaigen glb` | `file`, `out` |
| `sotaigen scale` | `file`, `out`, `--stature`, `--heads`, `--torso` |
| `sotaigen swap-head` | `body`, `head`, `out`, `--scale`, `--parts-only`, `--keep-parts`, `--report` |

| Exit code | Meaning |
|---|---|
| 0 | done |
| 1 | the body can't be made or written: a rule, a broken mesh |
| 2 | a wrong command line or file |
| 3 | Blender missing or older than the floor, or sotaigen-blender missing |
<!-- end generated -->

- `check FILE`: checks a parameter file (ranges, rules, the surface, the wrap) and prints `valid` when
  nothing is wrong.
- `glb FILE OUT`: writes the rigged GLB.
- `blend FILE OUT`: writes the rigged `.blend`; needs Blender 5.2 or later and `sotaigen-blender`. The
  arguments for another rest pose (`--t-pose`, `--arm-abduction` degrees, `--ankle-spacing` mm,
  `--palms-down`, `--straight-arms`, `--past-limits`) are in [.blend](blend.md); `--for-head-swap` lets
  a neck through that is too thin for this tool's head but will get another.
- `swap-head BODY HEAD OUT`: swaps a `.blend`'s head for one made elsewhere: see
  [Head interface](head-interface.md).
- `scale FILE OUT`: rewrites a parameter file to a stature (`--stature` mm), a head count (`--heads`) or
  a torso length (`--torso` mm).

Without Blender, exit code 3 comes before any computing.
