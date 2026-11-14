"""What the tool raises when it can't do what it was asked, as opposed to a bug: all are SotaigenError.

Each message says what is wrong and, where it can, what to do about it.
"""


class SotaigenError(Exception):
    """Something this tool can't do with what it was given."""


class BlenderMissing(SotaigenError, FileNotFoundError):
    """No Blender on this machine, or no sotaigen-blender to give it its scripts."""


class BlenderTooOld(BlenderMissing):
    """A Blender older than the oldest the .blend is made and checked in (sotaigen.output.blender)."""


class InvalidSpec(SotaigenError, ValueError):
    """Parameters this tool won't make a body of: past a range or a rule, or a target past what the body
    can reach. `problems` holds each sotaigen.diagnostics.Diagnostic."""

    def __init__(self, problems):
        super().__init__("; ".join(problems))
        self.problems = list(problems)
