"""The command line: check a parameter file, write its GLB or .blend, or swap a .blend's head.

    sotaigen check FILE.json
    sotaigen glb FILE.json OUT.glb
    sotaigen blend FILE.json OUT.blend [rest pose] [--for-head-swap]
    sotaigen swap-head BODY.blend HEAD.blend OUT.blend [--scale S] [--parts-only] [--keep-parts a,b]
    sotaigen scale FILE.json OUT.json [--stature MM] [--heads N] [--torso MM]

`sotaigen` is this module's main, as pyproject.toml's project.scripts names it; `sotaigen COMMAND --help`
says each command's options. `check` runs the ranges and rules, then the surface and the real wrap.
`glb` and `blend` write what sotaigen.write_glb and write_blend write (`blend` needs Blender and
sotaigen-blender): a broken mesh is refused, and every other problem only warns and goes into the file.
`--for-head-swap` lets a neck thinner than this tool's head allows through, for a head that replaces it;
the file's quality list says so, and the neck's own limits still hold. `scale` rewrites a parameter file
by sotaigen.params.tools: the whole body to a stature, the same stature in another head count, or another
torso length with everything measured in its shares carried along; then it checks the ranges and rules.

Exit codes, the same for every command:

    0  done: the file is valid, or written, warnings and all
    1  the body isn't: a range, a rule, a target past reach, a broken mesh, a joint past its limit;
       for `check`, anything found but a warning
    2  the command line or a file is wrong: no such file, not JSON, not a parameter file
    3  something to run it with is missing: Blender, or sotaigen-blender for a .blend
"""

import argparse
import sys
from pathlib import Path

from sotaigen.errors import BlenderMissing, InvalidSpec
from sotaigen.generation import examined
from sotaigen.mend import tell
from sotaigen.output import head_swap
from sotaigen.output.blend import write_blend
from sotaigen.output.blender import can_write_blend
from sotaigen.output.glb import write_glb
from sotaigen.output.quality import BrokenBody
from sotaigen.params.file import FileError, read_spec, write_parameter_file
from sotaigen.params.rules import NECK_THIN_FOR_HEAD, NUMERIC_SLACK, problems, rules, spec_bounds
from sotaigen.params.spec import MILLIMETRES, proportions
from sotaigen.params.tools import scaled, with_head_count, with_torso_length
from sotaigen.rig.rest_pose import PastLimits, add_arguments, request_of

DONE, REFUSED, USAGE, MISSING = 0, 1, 2, 3


def rule_mends(spec, sex):
    """For each rule a body breaks, the changes to one parameter at a time that would meet it.

    A rule's shortfall is a share of the torso or a ratio, which a person can't act on, so this says it in
    each parameter's own unit: millimetres for lengths, as the file holds them. Each change stays inside
    that parameter's range on this body, smallest share of its range first.
    """
    bounds = spec_bounds(spec, sex)
    found = {}
    for name, _, room in rules(sex):

        def met(key, value, room=room):
            try:
                return room(proportions({**spec, key: value})) >= -NUMERIC_SLACK
            except (KeyError, ZeroDivisionError):
                return False

        if room(proportions(spec)) >= -NUMERIC_SLACK:
            continue
        options = []
        for key, value in spec.items():
            if key not in bounds:
                continue
            low, high = bounds[key]
            for end in (low, high):
                if end == value or not met(key, end):
                    continue
                inside, outside = end, value
                for _ in range(40):
                    mid = (inside + outside) / 2
                    inside, outside = (mid, outside) if met(key, mid) else (inside, mid)
                options.append((abs(inside - value) / (high - low), key, inside - value))
                break
        unit = lambda key: " mm" if key in MILLIMETRES else ""
        found[name] = ", ".join(f"{key} {change:+.1f}{unit(key)}" for _, key, change in sorted(options)[:4])
    return found


def parser():
    top = argparse.ArgumentParser(prog="sotaigen", description=__doc__.splitlines()[0])
    commands = top.add_subparsers(dest="command", required=True, metavar="COMMAND")
    check = commands.add_parser("check", help="check a parameter file: rules, surface, wrap")
    check.add_argument("file", type=Path)
    glb = commands.add_parser("glb", help="write a parameter file's rigged GLB")
    glb.add_argument("file", type=Path)
    glb.add_argument("out", type=Path)
    blend = commands.add_parser("blend", help="write a parameter file's rigged .blend, with Blender")
    blend.add_argument("file", type=Path)
    blend.add_argument("out", type=Path)
    add_arguments(blend)
    blend.add_argument(
        "--for-head-swap",
        action="store_true",
        help="let a neck through that is too thin for this tool's head, for a head that replaces it",
    )
    swap = commands.add_parser("swap-head", help="swap a .blend's head for one made elsewhere, with Blender")
    head_swap.add_arguments(swap)
    scale = commands.add_parser("scale", help="rewrite a parameter file to a stature, head count or torso length")
    scale.add_argument("file", type=Path)
    scale.add_argument("out", type=Path)
    scale.add_argument("--stature", type=float, help="the whole body scaled to this many mm tall")
    scale.add_argument("--heads", type=float, help="the same stature in this many head heights")
    scale.add_argument("--torso", type=float, help="the torso this many mm long, its shares carried along")
    return top


def rescale(args):
    """`sotaigen scale`: the tools in order, stature last, so it is what the file ends up at."""
    try:
        spec = read_spec(args.file)
    except FileError as error:
        print(f"{args.file}: {error}", file=sys.stderr)
        return USAGE
    body = dict(spec.body)
    try:
        if args.torso is not None:
            body = with_torso_length(body, args.torso)
        if args.heads is not None:
            body = with_head_count(body, args.heads, spec.sex)
        if args.stature is not None:
            body = scaled(body, args.stature)
    except KeyError as error:
        print(f"{args.file}: body is missing {error.args[0]}", file=sys.stderr)
        return USAGE
    args.out.write_text(write_parameter_file(body, spec.sex, spec.base, spec.unknown))
    found = problems(body, spec.sex)
    print("\n".join(found) or "valid")
    print(args.out)
    return REFUSED if found else DONE


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "swap-head":
        return head_swap.run(args)
    if args.command == "scale":
        return rescale(args)
    for_head_swap = getattr(args, "for_head_swap", False)
    # Before any computing: a .blend needs Blender and the package of its scripts.
    if args.command == "blend" and (why := can_write_blend()):
        print(why, file=sys.stderr)
        return MISSING
    try:
        spec = read_spec(args.file)
    except FileError as error:
        print(f"{args.file}: {error}", file=sys.stderr)
        return USAGE
    if spec.unknown:
        print(f"note: fields this reader doesn't know: {', '.join(sorted(spec.unknown))}")
    invalid = [
        found for found in problems(spec.body, spec.sex)
        if not (for_head_swap and found.code == f"rule.{NECK_THIN_FOR_HEAD}")
    ]
    if invalid:
        mends = rule_mends(spec.body, spec.sex)
        for found in invalid:
            print(found)
            name = found.code.removeprefix("rule.")
            if found.code.startswith("rule.") and name in mends:
                print(f"  mend one of: {mends[name] or 'no single parameter in range'}")
        return REFUSED
    try:
        body, found = examined(spec, for_head_swap=for_head_swap)
    except InvalidSpec as error:
        print("\n".join(error.problems))
        return REFUSED
    print("\n".join(found) or "valid")
    if args.command == "check":
        # Only a seam rule's warning leaves the body as good as it can be told: a missed target, the
        # surface's misses and the wrap's warnings say it came out other than asked.
        seams_only = all(d.level == "warning" and d.code.startswith("rule.") for d in found)
        return DONE if seams_only else REFUSED
    try:
        if args.command == "glb":
            path = write_glb(body, args.out)
        else:
            path = write_blend(body, args.out, rest=request_of(args))
    except (BrokenBody, PastLimits) as error:
        print(error)
        tell(spec.body, spec.sex, body.wrap, sys.stdout)
        return REFUSED
    except BlenderMissing as error:
        print(error, file=sys.stderr)
        return MISSING
    except RuntimeError as error:
        # Blender's own output, which is long: its first line is what went wrong.
        print(f"{args.out}: not written: {str(error).splitlines()[0]}", file=sys.stderr)
        return REFUSED
    print(path, path.stat().st_size, "bytes")
    return DONE


if __name__ == "__main__":
    sys.exit(main())
