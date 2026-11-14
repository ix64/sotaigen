"""What checking a body or a file finds, as data: a code, a level and the values it is about.

The code and the values are the contract (docs/systems/export.md, 诊断); files store them as
{"code", "level", "values"}, and a reader that meets a code it doesn't know still shows the code and
the values. The English sentence is only a rendering, for the command line and for messages: a
Diagnostic is that sentence as a str, so it prints, joins and compares as one, and `json()` is what
goes into a file. Lengths are in mm, angles in degrees, shares and ratios as plain numbers.

    error    the parameters or the file can't be made into a body
    broken   the mesh is broken, so no file is written
    warning  travels with the file
"""

LEVELS = ("error", "broken", "warning")
# Each rule's sentence by its name, which sotaigen.params.rules fills in as it is imported.
RULE_SENTENCES = {}


def _where(values):
    at = values.get("at") or {}
    return ", at " + ", ".join(f"{part} {count}" for part, count in at.items()) if at else ""


def _short(value):
    return f"{value:.4g}"


CONTACT_WORDS = {"axilla": "shoulder", "groin": "pelvic", "hang": "hanging-arm"}


# Each code's English sentence, from its values. A code not listed reads as its code and values.
SENTENCES = {
    "file.unreadable": lambda v: f"can't be read: {v['detail']}",
    "file.not_json": lambda v: f"not JSON: {v['detail']}",
    "file.not_object": lambda v: "not a parameter file: the top level must be an object",
    "file.format": lambda v: f"not a parameter file: format is {v['format']}",
    "file.version": lambda v: f"bad version: {v['version']}",
    "file.newer": lambda v: f"version {v['version']} is newer than this reader, which reads up to {v['latest']}",
    "file.no_sex": lambda v: 'sex must name whose body this is, such as "female"',
    "file.not_parameters": lambda v: "body must be an object of parameters",
    "param.sex": lambda v: f"there is no sex {v['sex']!r}",
    "param.missing": lambda v: f"body is missing {v['key']}",
    "param.unknown": lambda v: f"body has unknown parameter {v['key']}",
    "param.not_number": lambda v: f"body {v['key']} is not a number: {v['value']}",
    "param.outside": lambda v: (
        f"body {v['key']} = {_short(v['value'])} is outside {_short(v['low'])} to {_short(v['high'])}"
    ),
    "body.unknown": lambda v: f"no body named {v['name']}; known: {', '.join(v['known'])}",
    "target.missed": lambda v: f"target: {v['key']} asked {v['asked']:.1f}, reached {v['reached']:.1f}",
    "target.unreachable": lambda v: f"target: {v['detail']}",
    "wrap.flipped": lambda v: f"flipped faces after the wrap: {v['count']}{_where(v)}",
    "wrap.midline": lambda v: f"vertices past the midline: {v['count']}{_where(v)}",
    "skin.open": lambda v: (
        f"the skin is not closed: {v['edges']} edges without exactly two faces, in the template, not from any parameter"
    ),
    "foot.off_ground": lambda v: f"foot off the ground: {v['mm']} mm, over {v['limit']}",
    "seam.crease": lambda v: f"{v['part']} seam crease: {v['degrees']} degrees, over {v['limit']}",
    "seam.miss": lambda v: f"{v['part']} seam miss: {v['share']:.2f} of the limb's radius, over {v['limit']}",
    "uv.stretch": lambda v: f"UV stretch: L2 {v['value']:.3f}, over {v['limit']}",
    "uv.anisotropy": lambda v: f"UV anisotropy: p95 {v['value']:.2f}, over {v['limit']}",
    "pose.collapsed": lambda v: f"collapsed faces in {v['pose']}: {v['count']}",
    "contact.failed": lambda v: f"{CONTACT_WORDS.get(v['group'], v['group'])} contact bake failed: {v['detail']}",
    "touch.unreached": lambda v: (
        f"touch {v['contact']}: {v['arrivalMm']} mm from the place, the palm {v['palmDegrees']} degrees off, "
        f"the hand {v['gapMm']} mm off the skin" + (", the arm through the head" if v["throughHead"] else "")
    ),
    "head_swap.touch_stale": lambda v: (
        "head swap: the hand-to-head pose assets were solved for the old head and were not solved again"
    ),
    "head_swap.unplaced": lambda v: (
        f"head swap: the head file places no {', '.join(v['bones'])}, so they stay where the old head had them"
    ),
    "head_swap.kept_parts": lambda v: (
        f"head swap: pieces of {', '.join(v['objects'])} have boundaries that touch themselves, so they stayed "
        f"parts: if the skin is among them, mark it with {v['group']}"
    ),
    "head_swap.openings": lambda v: f"head swap: the new skin keeps {v['count']} openings besides the neck",
    "head_swap.collapsed": lambda v: f"head swap: collapsed faces in {v['pose']}: {v['count']}",
    "rest.past_limit": lambda v: f"{v['limb']} {v['axis']} {v['degrees']} is outside {v['low']:g} to {v['high']:g}",
    "rest.unclean": lambda v: (
        f"rest pose: {v['limb']} {v['axis']} {v['degrees']} is outside the clean range {v['low']:g} to {v['high']:g}"
    ),
}


def sentence(code, values):
    """The English sentence of a code and its values."""
    if code.startswith("rule."):
        name = code.split(".", 1)[1]
        return f"{RULE_SENTENCES.get(name, name)}, short by {values['short']:.3f}"
    if code.startswith("surface."):
        return f"surface: {code.split('.', 1)[1]}"
    say = SENTENCES.get(code)
    if say is None:
        return f"{code}: {values}" if values else code
    return say(values)


class Diagnostic(str):
    """One finding: its English sentence as a str, with `code`, `level` and `values` beside it.

    A seam rule, a missed target or anything else that only warns has level "warning", and its sentence
    begins "warning: " where a person could take it for a refusal (the seam rules, whose sentences read
    like the rules').
    """

    __slots__ = ("code", "level", "values")

    def __new__(cls, code, level, values=None):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, not {level!r}")
        values = dict(values or {})
        text = sentence(code, values)
        if level == "warning" and code.startswith("rule."):
            text = f"warning: {text}"
        if values.get("forHeadSwap"):
            text += ": let through for a head swap, not checked on this tool's head"
        self = super().__new__(cls, text)
        self.code, self.level, self.values = code, level, values
        return self

    def json(self):
        """What a file stores: no sentence, which isn't part of the contract."""
        return {"code": self.code, "level": self.level, "values": self.values}

    @classmethod
    def from_json(cls, row):
        return cls(row["code"], row["level"], row.get("values"))

    def __reduce__(self):
        return (Diagnostic, (self.code, self.level, self.values))


def rule(name, short, level="error"):
    """A rule broken by `short`: a share of the torso or a ratio, as the rule reads its parameters."""
    return Diagnostic(f"rule.{name}", level, {"short": round(short, 3)})


def as_json(found):
    """A list of diagnostics as a file stores them."""
    return [d.json() for d in found]


def plain(value):
    """`value` with every Diagnostic in it as a file stores it, for json.dumps, which would write a
    Diagnostic as its sentence."""
    if isinstance(value, Diagnostic):
        return value.json()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [plain(item) for item in value]
    return value
