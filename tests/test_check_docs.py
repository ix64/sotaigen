from sotaigen_dev.check_docs import problems


def tree(tmp_path, text):
    (tmp_path / "docs").mkdir()
    (tmp_path / "AGENTS.md").write_text("entry\n")
    (tmp_path / "docs" / "note.md").write_text(text)
    return tmp_path


def test_round_words_and_task_numbers_are_reported(tmp_path):
    found = problems(tree(tmp_path, "本轮没有做\n见 W5 和 D8\n"))
    assert [item.split(": ")[1] for item in found] == [
        "stale marker 本轮",
        "stale marker W5",
        "stale marker D8",
    ]


def test_percentiles_open_questions_and_file_names_pass(tmp_path):
    root = tree(tmp_path, "P95 和 P50，待决定的 P14，文件 w5-rig.md，W10 不是任务号\n")
    assert problems(root) == []


def test_dead_relative_links_are_reported_and_live_ones_pass(tmp_path):
    root = tree(
        tmp_path,
        "## anchor\n\n[a](missing.md) [b](note.md#anchor) [c](https://example.com/x.md) "
        "[d](../build/x.png)\n",
    )
    assert [item.split(": ")[1] for item in problems(root)] == ["dead link missing.md"]


def test_anchors_follow_the_headings_of_the_target_file(tmp_path):
    root = tree(tmp_path, "## 接触表（用 `IPC`）\n\n[a](#接触表用-ipc) [b](#没有这节)\n")
    assert [item.split(": ")[1] for item in problems(root)] == ["dead anchor #没有这节"]


def reference(tmp_path, chinese, english):
    root = tree(tmp_path, "note\n")
    for folder, text in (("docs/reference", chinese), ("docs/en/reference", english)):
        (tmp_path / folder).mkdir(parents=True)
        (tmp_path / folder / "glb.md").write_text(text)
    return root


def test_the_reference_has_no_dates_commits_or_old_bodies(tmp_path):
    text = "# GLB\n\nas of 2026-11-08, since a1b2c3d, on h55-soft; the faces' 9b9e5718f450e1d5a20b5e9a5e056b9fa04290f7d28fd4b1d5461da0fbd4b48a\n"
    found = problems(reference(tmp_path, text, text))
    whats = sorted({item.split(": ")[1].rsplit(" in the", 1)[0] for item in found})
    assert whats == ["commit hash a1b2c3d", "date 2026-11-08", "old body h55"]


def test_the_working_docs_may_keep_their_dates(tmp_path):
    assert problems(tree(tmp_path, "定于 2026-11-08，见 a1b2c3d\n")) == []


def test_the_two_languages_of_the_reference_have_one_outline(tmp_path):
    chinese = "# GLB\n\n## 字段\n\n<!-- generated: glb-values -->\n<!-- end generated -->\n"
    assert problems(reference(tmp_path, chinese, "# GLB\n\n## Fields\n\n<!-- generated: glb-values -->\n<!-- end generated -->\n")) == []
    (tmp_path / "b").mkdir()
    found = problems(reference(tmp_path / "b", chinese, "# GLB\n\n## Fields\n\n### More\n"))
    assert found == ["docs/en/reference/glb.md: headings or generated tables differ from docs/reference/glb.md"]
    (tmp_path / "b" / "docs" / "reference" / "cli.md").write_text("# cli\n")
    assert "docs/en/reference/cli.md: missing, docs/reference/cli.md has no English" in problems(tmp_path / "b")


def test_a_path_or_a_name_into_the_repository_must_name_something(tmp_path):
    root = tree(tmp_path, "`sotaigen/real.py` `sotaigen/gone.py:12` `sotaigen/glb` `sotaigen/<name>.py` "
                          "`sotaigen.real.there` `sotaigen.real.gone` `sotaigen.missing.x` `sotaigen.listed`\n")
    (tmp_path / "sotaigen").mkdir()
    (tmp_path / "sotaigen" / "real.py").write_text("there = 1\n")
    (tmp_path / "sotaigen" / "__init__.py").write_text('__all__ = ["listed"]\n')
    found = sorted(item.split(": ", 1)[1] for item in problems(root))
    assert found == ["no such name sotaigen.missing", "no such name sotaigen.real.gone", "no such path sotaigen/gone.py"]


def test_the_guides_in_skills_are_checked_too(tmp_path):
    root = tree(tmp_path, "note\n")
    (tmp_path / "skills" / "guide").mkdir(parents=True)
    (tmp_path / "skills" / "guide" / "SKILL.md").write_text("[gone](nowhere.md)\n")
    assert [item.split(": ")[1] for item in problems(root)] == ["dead link nowhere.md"]


def test_generated_paths_need_no_frontend_install_in_a_clean_checkout(tmp_path):
    root = tree(tmp_path, "`web/node_modules/` `web/node_modules/vite/package.json` "
                          "`web/src/paraglide/` `web/src/paraglide/messages.js:12`\n")
    assert not (root / "web").exists()
    assert problems(root) == []


def test_generated_path_exceptions_do_not_hide_typos_or_dead_links(tmp_path):
    root = tree(tmp_path, "`web/node_modules-old/package.json` `web/src/paraglide-old/messages.js` "
                          "`web/src/paraglides/` `web/src/missing.ts` "
                          "[generated](../web/src/paraglide/messages.js)\n")
    assert sorted(item.split(": ", 1)[1] for item in problems(root)) == [
        "dead link ../web/src/paraglide/messages.js",
        "no such path web/node_modules-old/package.json",
        "no such path web/src/missing.ts",
        "no such path web/src/paraglide-old/messages.js",
        "no such path web/src/paraglides",
    ]
