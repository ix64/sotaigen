"""sotaigen.sources: what a computation's key covers, on small trees, and on the real stages as they run."""

import json
import subprocess
import sys
import textwrap

import pytest

from sotaigen import pipeline, sources


@pytest.fixture
def tree(tmp_path):
    """Write modules below tmp_path/sotaigen/, and read back keys and reached definitions."""

    class Tree:
        root = tmp_path

        def write(self, **modules):
            for name, text in modules.items():
                path = tmp_path / "sotaigen" / f"{name.replace('.', '/')}.py"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(textwrap.dedent(text))

        def key(self, *roots):
            return sources.code_key(roots, tmp_path)

        def reached(self, *roots):
            return {k.rsplit("#", 1)[0] for k in sources.reached(roots, tmp_path)}

    (tmp_path / "sotaigen").mkdir()
    (tmp_path / "sotaigen" / "__init__.py").write_text("")
    return Tree()


def test_comments_layout_and_docstrings_are_not_code(tree):
    tree.write(a="def f():\n    return 1\n")
    before = tree.key("sotaigen/a.py:f")
    tree.write(a='"""A."""\n\n# Note.\n\n\ndef f():\n    """F."""\n    return (\n        1\n    )  # one\n')
    assert tree.key("sotaigen/a.py:f") == before
    tree.write(a="def f():\n    return 2\n")
    assert tree.key("sotaigen/a.py:f") != before


def test_a_module_that_reads_its_docstrings_counts_them(tree):
    tree.write(a='"""Usage."""\n\n\ndef f():\n    return __doc__\n')
    before = tree.key("sotaigen/a.py:f")
    tree.write(a='"""Other usage."""\n\n\ndef f():\n    return __doc__\n')
    assert tree.key("sotaigen/a.py:f") != before


def test_only_what_the_root_can_reach_counts(tree):
    tree.write(a="""
        LIMIT = 3
        OTHER = 4


        def helper():
            return LIMIT


        def f():
            return helper()


        def g():
            return OTHER
    """)
    assert tree.reached("sotaigen/a.py:f") == {"sotaigen/a.py:f", "sotaigen/a.py:helper", "sotaigen/a.py:LIMIT"}
    before = tree.key("sotaigen/a.py:f")
    tree.write(a="LIMIT = 3\nOTHER = 5\n\ndef helper():\n    return LIMIT\n\ndef f():\n    return helper()\n"
                 "\ndef g():\n    return OTHER\n\ndef h():\n    return 0\n")
    assert tree.key("sotaigen/a.py:f") == before


def test_names_are_followed_through_imports_of_every_kind(tree):
    tree.write(
        a="""
            from . import b
            from .c import thing as renamed
            import sotaigen.d
            import sotaigen.e as e


            def f():
                from .g import late
                return b.one() + renamed() + sotaigen.d.two() + e.three() + late()
        """,
        b="def one():\n    return 1\n\ndef unused():\n    return 0\n",
        c="def thing():\n    return 1\n",
        d="def two():\n    return 2\n",
        e="def three():\n    return 3\n",
        g="def late():\n    return 4\n",
    )
    reached = tree.reached("sotaigen/a.py:f")
    for name in ("sotaigen/b.py:one", "sotaigen/c.py:thing", "sotaigen/d.py:two", "sotaigen/e.py:three", "sotaigen/g.py:late"):
        assert name in reached
    assert "sotaigen/b.py:unused" not in reached


def test_one_name_of_an_import_reaches_only_its_own(tree):
    tree.write(a="from .b import used, unused\n\n\ndef f():\n    return used()\n",
               b="def used():\n    return 1\n\ndef unused():\n    return 0\n")
    reached = tree.reached("sotaigen/a.py:f")
    assert "sotaigen/b.py:used" in reached
    assert "sotaigen/b.py:unused" not in reached


def test_what_a_module_runs_on_import_always_counts(tree):
    tree.write(a="from . import b\n\n\ndef f():\n    return b.one()\n",
               b="REGISTRY = []\nREGISTRY.append(1)\n\n\ndef one():\n    return 1\n")
    assert "sotaigen/b.py:Expr" in tree.reached("sotaigen/a.py:f")
    assert "sotaigen/__init__.py:Expr" not in tree.reached("sotaigen/a.py:f")


def test_a_decorator_with_effects_counts_on_import(tree):
    tree.write(a="from . import b\n\n\ndef f():\n    return b.one()\n",
               b="from .reg import register\n\n\n@register\ndef hidden():\n    return 2\n\n\ndef one():\n    return 1\n",
               reg="def register(f):\n    return f\n")
    assert "sotaigen/b.py:hidden" in tree.reached("sotaigen/a.py:f")
    tree.write(b="from functools import cache\n\n\n@cache\ndef hidden():\n    return 2\n\n\ndef one():\n    return 1\n")
    assert "sotaigen/b.py:hidden" not in tree.reached("sotaigen/a.py:f")


def test_a_script_block_does_not_count_for_a_function_but_does_for_the_whole_module(tree):
    tree.write(a="def f():\n    return 1\n\n\nif __name__ == '__main__':\n    print(f())\n")
    assert tree.reached("sotaigen/a.py:f") == {"sotaigen/a.py:f"}
    assert "sotaigen/a.py:If" in tree.reached("sotaigen/a.py")


def test_a_class_counts_whole(tree):
    tree.write(a="""
        class Mesh:
            def __init__(self):
                self.n = 1

            def area(self):
                return 2


        def f():
            return Mesh().area()
    """)
    before = tree.key("sotaigen/a.py:f")
    tree.write(a="class Mesh:\n    def __init__(self):\n        self.n = 1\n\n    def area(self):\n        return 3\n"
                 "\n\ndef f():\n    return Mesh().area()\n")
    assert tree.key("sotaigen/a.py:f") != before


def test_a_module_that_looks_names_up_counts_whole_with_what_it_imports(tree):
    tree.write(a="from . import b\n\n\ndef f(name):\n    return getattr(b, name)()\n\n\ndef g():\n    return 0\n",
               b="def one():\n    return 1\n")
    reached = tree.reached("sotaigen/a.py:f")
    assert {"sotaigen/a.py:g", "sotaigen/b.py:one"} <= reached
    tree.write(a="from . import b\n\n\ndef f():\n    return getattr(b, 'one')()\n\n\ndef g():\n    return 0\n")
    reached = tree.reached("sotaigen/a.py:f")
    assert "sotaigen/b.py:one" in reached
    assert "sotaigen/a.py:g" not in reached


def test_a_second_binding_of_a_name_counts_and_keeps_its_order(tree):
    tree.write(a="X = 1\nX = 2\n\n\ndef f():\n    return X\n")
    before = tree.key("sotaigen/a.py:f")
    tree.write(a="X = 2\nX = 1\n\n\ndef f():\n    return X\n")
    assert tree.key("sotaigen/a.py:f") != before


def test_a_missing_root_is_an_error(tree):
    with pytest.raises(FileNotFoundError):
        tree.key("sotaigen/nope.py:f")
    tree.write(a="def f():\n    return 1\n")
    with pytest.raises(NameError):
        tree.key("sotaigen/a.py:g")


# Runs in a fresh interpreter, so that nothing an earlier test cached in this process hides a call.
TRACE = """
import json, sys, tempfile
from pathlib import Path
from sotaigen import pipeline
from sotaigen.form import solve

ran = set()
monitoring = sys.monitoring
tool = monitoring.PROFILER_ID
monitoring.use_tool_id(tool, "trace")
package = str(pipeline.ROOT / "sotaigen")


def started(code, offset):
    if code.co_filename.startswith(package) and code.co_qualname != "<module>":
        ran.add((str(Path(code.co_filename).relative_to(pipeline.ROOT)), code.co_firstlineno, code.co_qualname))
    return monitoring.DISABLE


monitoring.register_callback(tool, monitoring.events.PY_START, started)


def traced(run):
    ran.clear()
    monitoring.restart_events()
    monitoring.set_events(tool, monitoring.events.PY_START)
    try:
        run()
    finally:
        monitoring.set_events(tool, 0)
    return sorted(ran)


name = sys.argv[1]
out = Path(tempfile.mkdtemp())
# What a solve of this body is asked, caught on its way in; then the solve itself, traced.
asked, kept = [], (solve._read_disk, solve._write_disk, solve._row)
solve._read_disk, solve._write_disk = lambda key: None, lambda key, row: None
solve._row = lambda *args: asked.append(args) or {"controls": {}, "measured": {}}
solve._solved(pipeline.bodies().specs[name])
solve._read_disk, solve._write_disk, solve._row = kept
solve._memory.clear()
runs = {"sotaigen/form/solve.py:_row": traced(lambda: solve._row(*asked[0]))}
for stage in ("wrap", "rig", "pose", "metrics"):
    runs[pipeline.STAGE_CODE[stage][0]] = traced(lambda: pipeline.RUN[stage](name, out))
print(json.dumps(runs))
"""


def lines_of(path):
    """Each top-level unit of a module and the lines it spans, from its own syntax tree."""
    import ast

    tree = ast.parse((pipeline.ROOT / path).read_text())
    spans = []
    for stmt in tree.body:
        first = min([stmt.lineno, *(d.lineno for d in getattr(stmt, "decorator_list", []))])
        spans.append((first, stmt.end_lineno, sources._bound(stmt)))
    return spans


@pytest.mark.slow
def test_every_function_the_real_stages_run_is_in_their_key():
    name = next(iter(pipeline.bodies()))
    result = subprocess.run([sys.executable, "-c", TRACE, name], capture_output=True, text=True, cwd=pipeline.ROOT,
                            check=False)
    assert result.returncode == 0, result.stderr[-3000:]
    runs = json.loads(result.stdout.splitlines()[-1])
    for root, ran in runs.items():
        reached = {k.rsplit("#", 1)[0] for k in sources.reached([root])}
        outside = []
        for path, line, qualname in ran:
            # The top-level definition the function belongs to, by the lines it spans.
            names = next((names for first, last, names in lines_of(path) if first <= line <= last), set())
            if not any(f"{path}:{n}" in reached for n in names):
                outside.append(f"{path}:{line} {qualname}")
        assert len(ran) > 20, root
        assert not outside, f"{root} ran code outside its key: {outside}"
