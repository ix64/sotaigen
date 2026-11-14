# SPDX-License-Identifier: GPL-3.0-or-later
"""The scripts that run inside Blender to write and pose a body's .blend and to swap its head.

They use Blender's Python API, so they are GPL-3.0-or-later, apart from sotaigen, which is Apache-2.0
and only runs them in a Blender process of their own (sotaigen.output.blender): it finds this
package's folder without importing it.
"""

# The oldest Blender these scripts are made and checked in; sotaigen.output.blender holds the same floor
# and checks it before it runs them, this is for when they are run by hand.
MINIMUM_BLENDER = (5, 2)


def require_blender():
    """Stop, saying why, in a Blender older than MINIMUM_BLENDER."""
    import bpy

    if tuple(bpy.app.version[:2]) < MINIMUM_BLENDER:
        need = ".".join(map(str, MINIMUM_BLENDER))
        raise SystemExit(f"Blender {bpy.app.version_string} is older than {need}, the oldest sotaigen supports")
