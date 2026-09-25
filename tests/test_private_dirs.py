"""Tests for kokoro_gui/engine/paths.py: the cache and user stores are
0o700 on POSIX, and a cache dir owned by another user is swapped for a
per-user one. Windows relies on profile ACLs, so these are POSIX-only."""
import os
import stat
import sys

import pytest

from kokoro_gui.engine import paths
from kokoro_gui.engine.paths import ensure_private_dir

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="mode bits don't decide access on Windows")


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def test_new_dir_is_private(tmp_path):
    target = tmp_path / "cache"
    assert ensure_private_dir(str(target)) == str(target)
    assert _mode(target) == 0o700


def test_existing_open_dir_owned_by_us_is_tightened(tmp_path, capsys):
    target = tmp_path / "cache"
    target.mkdir()
    os.chmod(target, 0o755)
    assert ensure_private_dir(str(target)) == str(target)
    assert _mode(target) == 0o700
    assert "private" in capsys.readouterr().out


def test_dir_owned_by_another_user_falls_back_and_is_left_alone(tmp_path, monkeypatch, capsys):
    target = tmp_path / "cache"
    target.mkdir()
    os.chmod(target, 0o755)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(paths.os, "getuid", lambda: os.stat(target).st_uid + 1)

    used = ensure_private_dir(str(target))

    assert used == str(tmp_path / "xdg" / "kokorogui" / "cache")
    assert _mode(used) == 0o700
    assert _mode(target) == 0o755
    out = capsys.readouterr().out
    assert str(target) in out and used in out


def test_store_owned_by_another_user_stays_put_with_a_warning(tmp_path, monkeypatch, capsys):
    target = tmp_path / "custom_voices"
    target.mkdir()
    monkeypatch.setattr(paths.os, "getuid", lambda: os.stat(target).st_uid + 1)
    assert ensure_private_dir(str(target), fallback=False) == str(target)
    assert "another user" in capsys.readouterr().out
