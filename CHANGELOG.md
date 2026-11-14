# Changelog

All notable changes to sotaigen and sotaigen-blender, which are released together with one version.
The [contracts](docs/en/reference/README.md) are frozen from 0.1.0: a release that breaks one bumps the
minor version while in 0.x (the major one from 1.0) and says here how to migrate. Everything else, the
geometry included, may change in any release.

## 0.1.0 (unreleased)

The first public release.

- **Bodies:** an anime-style base body from a parameter file of measurements, female or male, with 12
  typical bodies to start from. Every body shares one topology of 12676 vertices and 76 bones.
- **Files:** a GLB with skin weights, correctives as morph targets, placeholder features, and the driver
  tables a runtime needs; a rigged `.blend` with control bones, constraints, simple-expression drivers,
  joint limits and no Python, whose rest pose can be changed (`--t-pose`, `--arm-abduction`,
  `--ankle-spacing`).
- **Tools:** the command line `sotaigen check`, `glb`, `blend`, `swap-head`, `scale`; the Python API
  `sotaigen.read_spec`, `generate`, `write_glb`, `write_blend`; an editor in the browser.
- **Frozen contracts:** the topology and vertex order, the skeleton, the parameter file, the GLB's and the
  `.blend`'s metadata and driver tables, the diagnostics' codes, the typical bodies' names, the head
  interface, and the command line. Every file format is at version 1.
- **Requirements:** Python 3.14; Blender 5.2 or later and `sotaigen-blender` for the `.blend`; macOS or
  Linux.
- **Experimental:** the IPC contact tables (`sotaigen[contact]`), outside the contracts.
- **Known limitations:** [docs/en/reference/limitations.md](docs/en/reference/limitations.md).
