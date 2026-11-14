# Contracts

[中文](../../reference/README.md)

What sotaigen promises a downstream project: the names, structures and units it may rely on, and how
they change. Everything else may change with any version, including the wording of the output, the
geometry itself, and the parameters' ranges and rules.

## The frozen contracts

| # | Contract | Where |
|---|---|---|
| 1 | The template's topology and vertex order | [Coordinates and topology](coordinates-topology.md) |
| 2 | The bones' names, parents, order, local axes and rest pose, and the `.blend`'s control bone names | [Skeleton](skeleton.md), [.blend](blend.md) |
| 3 | The parameter file's structure, parameter names and units | [Parameter file](parameters.md) |
| 4 | The GLB's `extras.sotaigen` fields, the structure of its two driver tables and the diagnostics' codes; the same for the `.blend`'s `rig["sotaigen"]` | [GLB](glb.md), [.blend](blend.md), [Diagnostics](diagnostics.md) |
| 5 | The typical bodies' names | [Typical bodies](bodies.md) |
| 6 | The neck ring and the head's attachments | [Head interface](head-interface.md) |
| 7 | The command line's subcommands, arguments and exit codes | [Command line](cli.md) |

Not in the contracts: the corrective morph targets' names (only their agreement with the driver
table is), the check poses, the placeholder features, ears and hair (of the expressions only their
names are), the parameters' ranges and rules, the output's wording, and the geometry.

## Versions

- The packages follow [Semantic Versioning](https://semver.org/). `sotaigen` and `sotaigen-blender`
  are released together with the same version.
- **The contracts are frozen from 0.1.0.** Semantic Versioning lets 0.x change anything at any time;
  this promise is stricter: 0.x does not break the seven contracts above either.
- When a contract has to break (for a new feature, not for a better name), 0.x bumps the minor
  version (0.1 → 0.2) and 1.0 onwards the major one, with an entry and a migration note in the
  CHANGELOG. This is how Cargo and npm read 0.x: the 0.1.x releases are compatible with each other.
- Adding is not breaking: a new optional parameter, a new diagnostic code, a new corrective group, a
  new field. Readers must tolerate fields and codes they don't know.
- Each file has its own `format` and `version` (the parameter file, the GLB, the `.blend`), raised
  only when that format's meaning changes and apart from the package's version. A reader refuses a
  `version` newer than its own.

## How far compatibility goes

- Within a version: the same parameters give byte-identical GLBs, whether made from the command line,
  the Python API or the editor, given the same locked dependencies (`uv.lock`).
- Across versions: a parameter file always opens and means the same. **Geometry changes as shapes
  improve:** the same parameters may produce a different body. A file's
  `generator` (the package version and the rig's fingerprint) tells which made it.
- What a downstream made (a replacement head, clothes, edited weights) lines up by vertex index and
  bone name, so it still fits across versions; only the positions move with the shapes.

## The canonical text

This English text is canonical. Names, keys, codes and units are those of the golden copies in
`tests/contracts/`: the tables here are written from them by `sotaigen_dev/reference.py`, and
`tests/test_contracts.py` holds the code to them. Where the prose and a golden copy disagree, the golden
copy wins.

## See also

- [Glossary](glossary.md): what each word of the documents and the code means.
- [Known limitations](limitations.md): what 0.1.0 can't do yet or hasn't been checked for.

## License

The documentation is Apache-2.0, as the code is; generated bodies carry no conditions at all. See the
root [README](../../../README.md#license).
