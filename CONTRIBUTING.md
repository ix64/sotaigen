# Contributing

Thank you for looking. sotaigen makes anime-style base bodies from measurements; changes are judged
first by how the bodies look and move, then by the numbers. This page is how to work here. The working
documents under `docs/` are in Chinese; the contracts in [docs/en/reference](docs/en/reference/README.md)
are in English too, and that version is canonical.

## Setting up

Install [uv](https://docs.astral.sh/uv/) at the `tool.uv.required-version` in `pyproject.toml`.
It finds or downloads the Python pinned in `.python-version`. For Node, use
[fnm](https://github.com/Schniz/fnm), with its shell integration enabled, then run these from the
repository root. `fnm install` and `fnm use` read `.node-version`; pnpm uses the root `package.json`.
Node is needed for the editor, browser checks, and the Python tests marked `node`.

```sh
uv sync --all-extras    # Python packages, sotaigen-blender included
fnm install && fnm use
npm install --global "$(node -p 'require("./package.json").packageManager')"
pnpm install              # the page's JavaScript packages, for the editor and its checks
```

Rendering and `.blend` files need [Blender](https://www.blender.org/) 5.2 or later, found through
`BLENDER`, then `blender` on `PATH`, then the macOS app. macOS and Linux are supported; Windows is not.
The page's checks use the system's Chrome; `SOTAIGEN_CHROME_CHANNEL=` (empty) uses Playwright's own
Chromium instead.

## Commands

`uv run sotaigen-dev --help` lists the repository's Python development commands. The local `dev/`
package is installed by `uv sync`, stays out of the published packages, and needs no Node. Arguments
are forwarded directly, with paths relative to the checkout root. For example,
`uv run sotaigen-dev test tests/test_review.py -n 0` runs one file without workers. The three test
commands select their tier and default to four workers. Use `lint --docs-only` for documentation.
The downstream CLI remains `uv run sotaigen`, and frontend commands use `pnpm run`.

| Command | What it does |
|---|---|
| `uv run sotaigen-dev build` | wraps, rigs, poses and measures every body this tool comes with, into `build/` |
| `uv run sotaigen-dev test` | the daily tests: leaves out the slow ones, those needing Blender, and the contact bakes |
| `uv run sotaigen-dev test-merge` | every test but the contact bakes; run it before a pull request |
| `uv run sotaigen-dev lint` | ruff, and the documentation checks |
| `pnpm run typecheck` | the page's types |
| `pnpm run web-check` | the page's build and its browser tests |
| `uv run sotaigen-dev review sheet female-long --pose crouch --view front,side` | renders a sheet to look at |
| `uv run sotaigen-dev compare --against <another build/>` | what a change did to every body's metrics |

The full list, with what each writes, is in [docs/WORKFLOW.md](docs/WORKFLOW.md) (in Chinese).

## What a change needs

- **Look at it.** A change to shapes, weights, correctives or poses comes with renders before and
  after (`uv run sotaigen-dev review sheet … --against <build before>`). Passing metrics are not the same as
  looking right.
- **Tests that pass as they are.** A skipped or xfailed test fails the run here. If a test is wrong,
  say why in the change rather than loosening it. A known defect goes into the documentation of its
  body part, not into a test mark.
- **Contracts.** The names, structures and units in [docs/en/reference](docs/en/reference/README.md)
  are frozen; `tests/test_contracts.py` holds the code to golden copies in `tests/contracts/`. A change
  that breaks one needs a reason beyond tidiness, a version bump and a migration note in
  [CHANGELOG.md](CHANGELOG.md). After changing a contract on purpose, rewrite the copies and the tables:
  `uv run sotaigen-dev contracts --write --blend`, then
  `uv run sotaigen-dev reference`.
- **No character data.** No parameters, meshes or likeness of any particular character, real or
  fictional. The tool stays independent of any downstream project.
- **Dependencies** are pinned, with the lock file committed, and at least 14 days old (uv and pnpm
  enforce it).

## Where things go

| Path | What | License |
|---|---|---|
| `sotaigen/` | the package: shapes, rig, correctives, output, command line | Apache-2.0 |
| `sotaigen/data/` | the template, the typical and test bodies, ranges of motion | CC0-1.0 |
| `blender/` | the package `sotaigen-blender`: everything that imports `bpy` | GPL-3.0-or-later |
| `sotaigen_dev/`, `scripts/` | developer tools and single-file diagnostics, not shipped | Apache-2.0 |
| `web/` | the editor and its runtime | Apache-2.0 (the shadcn/ui components MIT) |
| `tests/`, `docs/` | the tests and the documentation | Apache-2.0 |

A file that imports `bpy` or `bmesh` starts with `# SPDX-License-Identifier: GPL-3.0-or-later` and lives
in `blender/`. Other files carry no header; [REUSE.toml](REUSE.toml) gives each path's license, and
`tests/test_licenses.py` checks both.

## Commits and pull requests

- One change per commit, messages in English as [Conventional Commits](https://www.conventionalcommits.org/)
  (`fix(rig): …`, `docs(reference): …`).
- Code comments are in English. The working documents are in Chinese; a change to a contract updates both
  languages of the reference, which `uv run sotaigen-dev lint` checks for the same outline.
- Documents state what holds now; the history belongs in commit messages.
- AI agents working here follow [AGENTS.md](AGENTS.md) as well.

## Security

See [SECURITY.md](SECURITY.md).
