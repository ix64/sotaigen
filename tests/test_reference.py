"""The contract reference's tables are what the golden copies say (sotaigen_dev/reference.py writes
them), and each page links to its other language."""

from sotaigen_dev import reference


def test_the_generated_tables_are_up_to_date():
    stale = reference.stale()
    assert not stale, f"run: uv run sotaigen-dev reference ({[p.name for p in stale]})"


def test_each_page_links_to_its_other_language():
    for language, path in reference.files():
        other = "../../reference/" if language == "en" else "../en/reference/"
        assert f"]({other}{path.name})" in path.read_text().split("\n\n", 2)[1], path
