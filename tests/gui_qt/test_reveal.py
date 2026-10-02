"""`reveal` picks the right file-manager call per platform and refuses a
path that isn't there."""
import os

import pytest

pytest.importorskip("PySide6")

from kokoro_gui.qt import reveal as reveal_module  # noqa: E402


@pytest.fixture
def calls(monkeypatch):
    popen, opened = [], []
    monkeypatch.setattr(reveal_module.subprocess, "Popen", lambda args, *a, **k: popen.append(args))
    monkeypatch.setattr(reveal_module.QDesktopServices, "openUrl",
                        staticmethod(lambda url: opened.append(url.toLocalFile()) or True))
    return popen, opened


def test_a_missing_or_empty_path_returns_false(calls, tmp_path):
    popen, opened = calls
    assert reveal_module.reveal(str(tmp_path / "nope.wav")) is False
    assert reveal_module.reveal("") is False
    assert reveal_module.reveal(None) is False
    assert popen == [] and opened == []


def test_windows_selects_the_file_in_explorer(calls, tmp_path, monkeypatch):
    popen, opened = calls
    monkeypatch.setattr(reveal_module.sys, "platform", "win32")
    target = tmp_path / "a.wav"
    target.write_bytes(b"x")
    assert reveal_module.reveal(str(target)) is True
    assert popen == [["explorer", "/select,", os.path.normpath(os.path.realpath(target))]]
    assert opened == []


def test_macos_reveals_the_file_with_open_r(calls, tmp_path, monkeypatch):
    popen, _opened = calls
    monkeypatch.setattr(reveal_module.sys, "platform", "darwin")
    target = tmp_path / "a.wav"
    target.write_bytes(b"x")
    assert reveal_module.reveal(str(target)) is True
    assert popen == [["open", "-R", os.path.realpath(target)]]


def test_other_platforms_open_the_containing_folder(calls, tmp_path, monkeypatch):
    popen, opened = calls
    monkeypatch.setattr(reveal_module.sys, "platform", "linux")
    target = tmp_path / "a.wav"
    target.write_bytes(b"x")
    assert reveal_module.reveal(str(target)) is True
    assert popen == []
    assert [os.path.normpath(o) for o in opened] == [os.path.dirname(os.path.realpath(target))]


@pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
def test_a_folder_opens_directly_on_every_platform(calls, tmp_path, monkeypatch, platform):
    popen, opened = calls
    monkeypatch.setattr(reveal_module.sys, "platform", platform)
    assert reveal_module.reveal(str(tmp_path)) is True
    assert popen == []
    assert [os.path.normpath(o) for o in opened] == [os.path.realpath(tmp_path)]
