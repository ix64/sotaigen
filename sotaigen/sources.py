"""What a computation can run, as a key for its caches.

A key covers the definitions a computation can reach from the functions that compute it, not whole
files. It follows names through each module's own bindings and imports, those inside functions
included, and hashes each definition's syntax tree, so comments, layout, docstrings and definitions
the computation never reaches don't count. Where it can't tell, it takes more: code a module runs
on import always counts, so does a decorator other than the known pure ones, and a module that looks
names up at run time (getattr with a computed name, globals(), exec, ...) counts whole, with all it
imports. Attributes of objects other than modules aren't followed: a class counts whole wherever it
is named. tests/test_sources.py runs real stages under sys.monitoring and checks that every function
of sotaigen/ they run is inside their key.
"""

import ast
import hashlib
import importlib.metadata
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# The packages whose imports a key follows: the tool, and the developer tools beside it.
PACKAGES = ("sotaigen", "sotaigen_dev")
# Decorators that only wrap what they decorate, so it counts only where it is used.
PURE_DECORATORS = {"cache", "lru_cache", "cached_property", "property", "staticmethod", "classmethod",
                   "dataclass", "contextmanager", "wraps"}
# Names that reach definitions without naming them in the source.
DYNAMIC = {"globals", "vars", "locals", "exec", "eval", "compile", "__import__", "import_module",
           "__dict__"}
NAME_LOOKUPS = {"getattr", "setattr", "hasattr", "delattr"}


def _is_docstring(stmt):
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)


def _bound(stmt):
    """The names a top-level statement binds."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {stmt.name}
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return {(a.asname or a.name).split(".")[0] for a in stmt.names}
    targets = {ast.Assign: lambda s: s.targets, ast.AnnAssign: lambda s: [s.target],
               ast.AugAssign: lambda s: [s.target], ast.TypeAlias: lambda s: [s.name]}.get(type(stmt))
    if targets is None:
        return set()
    return {n.id for t in targets(stmt) for n in ast.walk(t) if isinstance(n, ast.Name)}


def _is_main_block(stmt):
    """`if __name__ == "__main__":`, which runs only when the file is run as a script."""
    return isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Compare) \
        and getattr(stmt.test.left, "id", None) == "__name__"


def _runs_on_import(stmt):
    """Whether the statement does something on import beyond binding names others may use."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        for decorator in stmt.decorator_list:
            called = decorator.func if isinstance(decorator, ast.Call) else decorator
            name = called.attr if isinstance(called, ast.Attribute) else getattr(called, "id", None)
            if name not in PURE_DECORATORS:
                return True
        return False
    return not _bound(stmt) and not _is_main_block(stmt)


def _chain(node):
    """`a.b.c` as ["a", "b", "c"] when it starts from a name, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return [node.id, *reversed(parts)] if isinstance(node, ast.Name) else None


def _one_name_each(stmt):
    """A top-level import as one statement per name it binds, so a name reaches only its own."""
    if isinstance(stmt, ast.ImportFrom):
        return [ast.ImportFrom(stmt.module, [alias], stmt.level) for alias in stmt.names]
    if isinstance(stmt, ast.Import):
        return [ast.Import([alias]) for alias in stmt.names]
    return [stmt]


class Unit:
    """A top-level statement: the names it binds, its tree as text, and what it refers to."""

    def __init__(self, path, stmt, occurrence):
        self.names = _bound(stmt)
        tag = ",".join(sorted(self.names)) or type(stmt).__name__
        # Two bindings of one name keep their order; a definition moving in the file doesn't count.
        self.key = f"{path}:{tag}#{occurrence.get(tag, 0)}"
        occurrence[tag] = occurrence.get(tag, 0) + 1
        self.is_import = isinstance(stmt, (ast.Import, ast.ImportFrom))
        self.on_import = _runs_on_import(stmt)
        self.dump = ast.dump(stmt)
        self.used, self.chains, self.strings, self.imports, self.dynamic = set(), [], set(), [], False
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name):
                self.used.add(node.id)
                self.dynamic |= node.id in DYNAMIC
            elif isinstance(node, ast.Attribute):
                self.dynamic |= node.attr in DYNAMIC
                if chain := _chain(node):
                    self.chains.append(chain)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self.imports.append(node)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) in NAME_LOOKUPS:
                looked_up = node.args[1] if len(node.args) > 1 else None
                if isinstance(looked_up, ast.Constant) and isinstance(looked_up.value, str):
                    self.strings.add(looked_up.value)
                else:
                    self.dynamic = True


class Module:
    def __init__(self, path, source):
        tree = ast.parse(source, path)
        # A module that reads __doc__ shows its docstrings, so only there do they count.
        if "__doc__" not in source:
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                        and node.body and _is_docstring(node.body[0]):
                    node.body = node.body[1:] or ([] if isinstance(node, ast.Module) else [ast.Pass()])
        occurrence = {}
        self.units = [Unit(path, stmt, occurrence) for stmt in map(_one_name_each, tree.body) for stmt in stmt]
        self.binders = {}
        for index, unit in enumerate(self.units):
            for name in unit.names:
                self.binders.setdefault(name, []).append(index)
        self.top_imports = [node for unit in self.units if unit.is_import for node in unit.imports]


@lru_cache(maxsize=1024)
def _parsed(path, source):
    return Module(path, source)


class Code:
    """A checkout's modules, read as they are asked for, by their path below the checkout."""

    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.modules = {}

    def module(self, path):
        if path not in self.modules:
            file = self.root / path
            self.modules[path] = _parsed(path, file.read_text()) if file.is_file() else None
        return self.modules[path]

    def find(self, parts):
        """The module or package at `parts` below the checkout, as a path, or None."""
        base = "/".join(parts)
        return next((p for p in (f"{base}.py", f"{base}/__init__.py") if self.module(p)), None)

    def packages(self, path):
        """The packages that importing `path` runs first."""
        parts = path.split("/")[:-1]
        return [p for i in range(1, len(parts) + 1) if (p := self.find(parts[:i])) and p != path]

    def imported(self, node, path):
        """What an import statement reaches in the packages: [(module, name), ...], name None for a module,
        and {bound name: module} for the names it binds to whole modules."""
        reached, modules = [], {}
        if isinstance(node, ast.ImportFrom):
            if node.level:
                parts = path.split("/")[:-1]
                parts = parts[: len(parts) - (node.level - 1)]
            elif node.module and node.module.split(".")[0] in PACKAGES:
                parts = []
            else:
                return reached, modules
            parts = [*parts, *node.module.split(".")] if node.module else parts
            module = self.find(parts)
            if module:
                reached.append((module, None))
            for alias in node.names:
                if submodule := self.find([*parts, alias.name]):
                    reached.append((submodule, None))
                    modules[alias.asname or alias.name] = submodule
                elif module:
                    reached.append((module, alias.name))
        else:
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] not in PACKAGES:
                    continue
                reached += [(m, None) for i in range(1, len(parts) + 1) if (m := self.find(parts[:i]))]
                if alias.asname and (m := self.find(parts)):
                    modules[alias.asname] = m
                elif not alias.asname:
                    modules[parts[0]] = None  # `sotaigen.x.y` is read from the chain itself
        return reached, modules

    def reach(self, roots):
        """The units `roots` can run: "path" for a whole module, "path:name" for one binding."""
        picked, entered, whole, work = {}, set(), set(), []

        def enter(path):
            """Importing a module runs its packages, its code on import and its top-level imports."""
            if path in entered:
                return
            entered.add(path)
            for package in self.packages(path):
                enter(package)
            for index, unit in enumerate(self.module(path).units):
                if unit.on_import:
                    work.append((path, index))
                if unit.is_import or unit.on_import:
                    for node in unit.imports:
                        for module, _ in self.imported(node, path)[0]:
                            enter(module)

        def everything(path):
            if path in whole:
                return
            whole.add(path)
            enter(path)
            units = self.module(path).units
            work.extend((path, index) for index in range(len(units)))
            for unit in units:
                for node in unit.imports:
                    for module, _ in self.imported(node, path)[0]:
                        everything(module)

        def binding(path, name):
            enter(path)
            work.extend((path, index) for index in self.module(path).binders.get(name, ()))

        for root in roots:
            path, _, name = root.partition(":")
            if self.module(path) is None:
                raise FileNotFoundError(f"no module {path}")
            if not name:
                everything(path)
            elif name in self.module(path).binders:
                binding(path, name)
            else:
                raise NameError(f"{path} binds no {name}")

        top = {}
        while work:
            path, index = work.pop()
            if (path, index) in picked:
                continue
            unit = picked[(path, index)] = self.module(path).units[index]
            if unit.dynamic:
                everything(path)
            if path not in top:
                top[path] = {}
                for node in self.module(path).top_imports:
                    top[path] |= self.imported(node, path)[1]
            modules = dict(top[path])
            # A reached definition's own imports run with it.
            for node in unit.imports:
                reached, bound = self.imported(node, path)
                modules |= bound
                for module, name in reached:
                    binding(module, name) if name else enter(module)
            for name in unit.used | unit.strings:
                binding(path, name)
            for chain in unit.chains:
                self.follow(chain, modules, enter, binding)
            for module in {m for m in modules.values() if m}:
                for name in unit.strings:
                    binding(module, name)
        return picked

    def follow(self, chain, modules, enter, binding):
        """Reach what `module.name`, `package.module.name`, ... names, for a chain from a module."""
        if chain[0] not in modules:
            return
        if modules[chain[0]] is None:  # the package itself, by `import sotaigen.x`
            for i in range(len(chain), 0, -1):
                if module := self.find(chain[:i]):
                    enter(module)
                    if i < len(chain):
                        binding(module, chain[i])
                    return
            return
        module = modules[chain[0]]
        for part in chain[1:]:
            binding(module, part)
            submodule = self.find([*module.removesuffix("/__init__.py").removesuffix(".py").split("/"), part])
            if submodule is None:
                return
            enter(submodule)
            module = submodule

    def key(self, roots):
        """A digest of what `roots` can run, and of the Python that parses and runs it."""
        digest = hashlib.sha256(f"python {sys.version_info.major}.{sys.version_info.minor}\n".encode())
        for unit in sorted(self.reach(roots).values(), key=lambda u: u.key):
            digest.update(f"{unit.key}\n{unit.dump}\n".encode())
        return digest.hexdigest()


def code_key(roots, root=ROOT):
    """A digest of the code `roots` can run (see the module's docstring), as paths below `root`."""
    return Code(root).key(roots)


def reached(roots, root=ROOT):
    """What code_key hashes: the reached definitions as "path:name#n", sorted."""
    return sorted(unit.key for unit in Code(root).reach(roots).values())


def bake_key(roots, base, verts, bones, weights):
    """A contact bake's cache key: the rig it is given, the code `roots` can run and the solver's."""
    digest = hashlib.sha256()
    for array in (verts, weights, base["verts"]):
        array = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())

    def plain(value):
        return value.tolist() if isinstance(value, np.ndarray) else value.item()

    rig = {
        "faces": base["faces"],
        "segments": base["segments"],
        "landmarks": base.get("landmarks"),
        "bones": bones,
        "record": {key: getattr(bones, key, None) for key in ("T", "glides", "groin")},
    }
    digest.update(json.dumps(rig, default=plain, sort_keys=True, separators=(",", ":")).encode())
    digest.update(code_key(roots).encode())
    for name in ("numpy", "scipy", "ipctk"):
        digest.update(f"{name}={importlib.metadata.version(name)}".encode())
    return digest.hexdigest()
