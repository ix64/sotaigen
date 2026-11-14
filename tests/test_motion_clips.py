"""The motion clips are test data with their sources' digests written down twice, in the code that
checks a download and in motions/SOURCES.md, which says where they came from."""

import json
import re
from pathlib import Path

import pytest

from sotaigen_dev.motion import clips

ROOT = Path(__file__).resolve().parents[1]


def test_the_sources_list_names_the_digests_fetch_checks():
    table = dict(re.findall(r"\| `([0-9_]+\.(?:asf|amc))` \| `([0-9a-f]{64})` \|", (ROOT / "motions" / "SOURCES.md").read_text()))
    assert table == clips.FILES


@pytest.mark.parametrize("name", clips.clip_names())
def test_each_clip_says_what_it_is(name):
    recorded = json.loads((clips.CLIPS_DIR / f"{name}.json").read_text())
    assert (recorded["format"], recorded["version"]) == (clips.CLIP_FORMAT, clips.CLIP_VERSION)
    assert {"rootPath", "footContacts"} <= recorded.keys()


def test_a_clip_written_another_way_is_refused(tmp_path, monkeypatch):
    (tmp_path / "old.json").write_text(json.dumps({"name": "old", "fps": 60.0, "rotations": {}}))
    monkeypatch.setattr(clips, "CLIPS_DIR", tmp_path)
    with pytest.raises(ValueError, match="run fetch again"):
        clips.load_clip("old")
