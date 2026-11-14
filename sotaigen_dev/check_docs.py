"""Keep the docs free of words that only meant something while they were being written, and of
dead relative links and anchors. Round words and task numbers lose their referent as soon as the
work moves on; the docs state current conclusions and the commit messages hold the process.

The contract reference (docs/reference/ and its English docs/en/reference/) is held closer: no dates,
commit hashes or names of bodies long gone, since a downstream reads it as what holds now, and the two
languages have the same files, the same headings' levels and the same generated tables.

    uv run sotaigen-dev lint --docs-only
    uv run -m sotaigen_dev.check_docs [root]
"""

import ast
import re
import sys
from pathlib import Path

STALE = re.compile(
    r"\bW[1-7]\b|\bWx\b|本轮|这轮|首轮|第一轮|上一轮|\bD\d{1,2}\b|\bP(?:9|10|15)\b"
)
# Open questions in DECISIONS keep their numbers: P18. P1, P5, P50 and P95 are percentiles.

# In the reference only: a date, a commit hash (7 to 40 hex digits with a letter and a digit, not part
# of a longer run such as a SHA-256), the presets that were deleted before the first release.
DATED = re.compile(r"\b20\d\d-\d\d-\d\d\b")
COMMIT = re.compile(r"(?<![0-9a-f])(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,40}(?![0-9a-f])")
OLD_BODIES = re.compile(r"\bh(?:45|55|60|70)n?\b")
REFERENCE = ("docs/reference", "docs/en/reference")
GENERATED = re.compile(r"<!-- generated: ([\w-]+) -->")

LINK = re.compile(r"\]\(([^)\s]+)\)")
HEADING = re.compile(r"#{1,6} (.+)")


# The documents at the root, besides docs/ and the downstream guides in skills/.
ROOT_DOCS = ("AGENTS.md", "README.md", "README.zh.md", "CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md")
# A path into the repository, in backticks: under one of its folders, with no placeholder in it.
TOP = ("sotaigen/", "sotaigen_dev/", "scripts/", "tests/", "web/", "blender/", "docs/", "template/", "motions/", "skills/")
CODE = re.compile(r"`([^`\s]+)`")
PLACEHOLDER = re.compile(r"[<>*{}…|$]|\.\.\.")
# These documented outputs do not exist in a checkout before web-install / web-build.
# Match whole path components, not arbitrary ignored files: source typos must still fail.
GENERATED_DIRS = ("web/node_modules", "web/src/paraglide")


def doc_files(root):
    for name in ROOT_DOCS:
        if (root / name).exists():
            yield root / name
    for pattern in ("*.md", "*.toml"):
        yield from sorted((root / "docs").rglob(pattern))
    yield from sorted((root / "skills").rglob("*.md"))


def missing_path(root, text):
    """A repository path that names nothing, as in `sotaigen/rig/nothing.py`; None for anything else.
    A trailing :line, #anchor or :name is left out, as are known generated directories.
    build/ is outside TOP because it is not committed either."""
    if not text.startswith(TOP) or PLACEHOLDER.search(text):
        return None
    path = re.split(r"[:#]", text, maxsplit=1)[0].rstrip(".,")
    # A format's name, sotaigen/glb, is not a path: a path ends in a file's extension or a slash.
    if not (path.endswith("/") or "." in path.rsplit("/", 1)[-1]):
        return None
    path = path.rstrip("/")
    if any(path == folder or path.startswith(folder + "/") for folder in GENERATED_DIRS):
        return None
    return None if (root / path).exists() else path


DOTTED = re.compile(r"(sotaigen(?:_dev|_blender)?(?:\.\w+)+)(?:\(\))?")


def defined(module):
    """The names a module defines at its top level, without importing it."""
    names = set()
    for node in ast.parse(module.read_text()).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for element in ast.walk(target):
                    if isinstance(element, ast.Name):
                        names.add(element.id)
            # A package's lazily loaded names (sotaigen.__all__) are its names too.
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets) and node.value is not None:
                names.update(e.value for e in ast.walk(node.value) if isinstance(e, ast.Constant) and isinstance(e.value, str))
        elif isinstance(node, ast.Import | ast.ImportFrom):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
    return names


def missing_name(root, text):
    """A dotted name into the packages that names nothing, as `sotaigen.rig.skeleton.sided`; None when its
    modules exist and the first name past them is defined there. Attributes past that aren't followed."""
    match = DOTTED.fullmatch(text)
    if not match:
        return None
    parts = match[1].split(".")
    base = {"sotaigen": root / "sotaigen", "sotaigen_dev": root / "sotaigen_dev",
            "sotaigen_blender": root / "blender" / "src" / "sotaigen_blender"}[parts[0]]
    if not base.is_dir():
        return None
    for index, part in enumerate(parts[1:], 1):
        if (base / part).is_dir():
            base = base / part
            continue
        if (base / f"{part}.py").is_file():
            base = base / f"{part}.py"
            if index + 1 < len(parts) and parts[index + 1] not in defined(base):
                return ".".join(parts[: index + 2])
            return None
        init = base / "__init__.py"
        return None if init.is_file() and part in defined(init) else ".".join(parts[: index + 1])
    return None


def anchors(path):
    """The anchors a renderer gives a Markdown file's headings, as GitHub spells them."""
    found, fence = set(), False
    for line in path.read_text().splitlines():
        if line.startswith("```"):
            fence = not fence
        match = HEADING.fullmatch(line)
        if match and not fence:
            text = re.sub(r"[^\w\- ]", "", match[1].replace("`", "").lower())
            found.add(text.replace(" ", "-"))
    return found


def link_problem(path, target):
    relative, _, anchor = target.partition("#")
    # build/ is not committed, so links into it only resolve where it has been built.
    if "://" in relative or "build/" in relative:
        return None
    file = path.parent / relative if relative else path
    if not file.exists():
        return "dead link"
    if anchor and file.suffix == ".md" and anchor.lower() not in anchors(file):
        return "dead anchor"
    return None


def in_reference(name):
    return any(name.startswith(folder + "/") for folder in REFERENCE)


def outline(path):
    """A Markdown file's headings' levels and its generated tables, in order: what both languages share."""
    found, fence = [], False
    for line in path.read_text().splitlines():
        if line.startswith("```"):
            fence = not fence
        if fence:
            continue
        heading = HEADING.fullmatch(line)
        if heading:
            found.append(f"h{len(line) - len(line.lstrip('#'))}")
        generated = GENERATED.match(line)
        if generated:
            found.append(generated[1])
    return found


def parity(root):
    """The two languages of the reference: the same files, each with the same outline."""
    chinese, english = (root / folder for folder in REFERENCE)
    if not chinese.is_dir() and not english.is_dir():
        return []
    names = lambda folder: {p.name for p in folder.glob("*.md")} if folder.is_dir() else set()
    found = [f"{REFERENCE[1]}/{n}: missing, {REFERENCE[0]}/{n} has no English" for n in sorted(names(chinese) - names(english))]
    found += [f"{REFERENCE[0]}/{n}: missing, {REFERENCE[1]}/{n} has no Chinese" for n in sorted(names(english) - names(chinese))]
    for name in sorted(names(chinese) & names(english)):
        if outline(chinese / name) != outline(english / name):
            found.append(f"{REFERENCE[1]}/{name}: headings or generated tables differ from {REFERENCE[0]}/{name}")
    return found


def problems(root):
    found = []
    for path in doc_files(root):
        name = path.relative_to(root).as_posix()
        reference = in_reference(name)
        for number, line in enumerate(path.read_text().splitlines(), 1):
            for match in STALE.finditer(line):
                found.append(f"{name}:{number}: stale marker {match[0]}")
            if reference:
                for pattern, what in ((DATED, "date"), (COMMIT, "commit hash"), (OLD_BODIES, "old body")):
                    found += [f"{name}:{number}: {what} {m[0]} in the reference" for m in pattern.finditer(line)]
            for target in LINK.findall(line):
                problem = link_problem(path, target)
                if problem:
                    found.append(f"{name}:{number}: {problem} {target}")
            for text in CODE.findall(line):
                gone = missing_path(root, text)
                if gone:
                    found.append(f"{name}:{number}: no such path {gone}")
                gone = missing_name(root, text)
                if gone:
                    found.append(f"{name}:{number}: no such name {gone}")
    return found + parity(root)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    found = problems(root)
    print("\n".join(found) if found else "docs ok")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
