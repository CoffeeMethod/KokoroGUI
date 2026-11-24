"""Fixtures for `tests/headless`: isolated storage dirs, no real audio device,
and a working directory inside tmp_path (so `audio_output/headless` is too)."""
from unittest.mock import MagicMock

import pytest

from kokoro_gui.engine import runtime


@pytest.fixture(autouse=True)
def headless_env(isolated_dirs, monkeypatch, tmp_path):
    playback = MagicMock()
    playback.AVAILABLE = True
    monkeypatch.setattr(runtime, "playback", playback)
    monkeypatch.chdir(tmp_path)
    return playback
