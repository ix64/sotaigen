"""sotaigen: parametric, rigged anime-style base bodies. The library's names are sotaigen.api's.

They are loaded when first asked for, so that importing a part of the package doesn't load all of it.
"""

__all__ = [
    "BlenderMissing",
    "Body",
    "BrokenBody",
    "Diagnostic",
    "FileError",
    "InvalidSpec",
    "PastLimits",
    "SotaigenError",
    "Spec",
    "check",
    "generate",
    "read_spec",
    "typical",
    "typical_names",
    "write_blend",
    "write_glb",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module 'sotaigen' has no attribute {name!r}")
    from sotaigen import api

    return getattr(api, name)


def __dir__():
    return sorted([*globals(), *__all__])
