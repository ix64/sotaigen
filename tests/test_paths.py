"""Where the tool keeps its caches and its build, and that SOTAIGEN_CACHE and SOTAIGEN_BUILD move them."""

from pathlib import Path

from sotaigen import paths


def test_the_cache_is_the_users_unless_named(tmp_path, monkeypatch):
    monkeypatch.setenv("SOTAIGEN_CACHE", str(tmp_path / "named"))
    assert paths.cache_dir() == tmp_path / "named"
    monkeypatch.delenv("SOTAIGEN_CACHE")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(paths.sys, "platform", "darwin")
    assert paths.cache_dir() == tmp_path / "Library" / "Caches" / "sotaigen"
    monkeypatch.setattr(paths.sys, "platform", "linux")
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert paths.cache_dir() == tmp_path / ".cache" / "sotaigen"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert paths.cache_dir() == tmp_path / "xdg" / "sotaigen"


def test_the_build_is_the_checkouts_or_the_working_folders_unless_named(tmp_path, monkeypatch):
    monkeypatch.delenv("SOTAIGEN_BUILD", raising=False)
    assert paths.build_dir() == paths.ROOT / "build"
    installed = tmp_path / "site-packages"
    (installed / "sotaigen").mkdir(parents=True)
    monkeypatch.setattr(paths, "ROOT", installed)
    monkeypatch.chdir(tmp_path)
    assert paths.build_dir() == tmp_path / "build"
    monkeypatch.setenv("SOTAIGEN_BUILD", "elsewhere")
    assert paths.build_dir() == (tmp_path / "elsewhere").resolve()


def test_the_data_ships_inside_the_package():
    assert paths.DATA.parent == paths.PACKAGE
    assert (paths.DATA / "template" / "base-mesh.json").is_file()
    assert Path(paths.__file__).resolve().parent == paths.PACKAGE
