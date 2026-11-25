"""Fixtures for `tests/headless`: isolated storage dirs, no real audio device,
and a working directory inside tmp_path (so `audio_output/headless` is too)."""
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from kokoro_gui.engine import runtime
from kokoro_gui.engine.runner import EngineRunner, ModelBase
from kokoro_gui.engines import registry
from kokoro_gui.engines.base import (
    BackendHooksMixin, ConfigField, ConfigFieldType, EngineCapabilities, Synthesis, VoiceInfo, common_fields,
)
from kokoro_gui.headless.host import ALLOW_DOWNLOAD_ENV

VOICES = {"a": ["v1", "v2"], "b": ["w1"]}


@pytest.fixture(autouse=True)
def headless_env(isolated_dirs, monkeypatch, tmp_path):
    playback = MagicMock()
    playback.AVAILABLE = True
    monkeypatch.setattr(runtime, "playback", playback)
    monkeypatch.chdir(tmp_path)
    return playback


@pytest.fixture
def offline(monkeypatch):
    """No network. Loopback passes: asyncio's self-pipe uses it on Windows."""
    real = socket.socket.connect
    attempts = []

    def connect(self, address):
        if isinstance(address, tuple) and address and address[0] in ("127.0.0.1", "::1", "localhost"):
            return real(self, address)
        attempts.append(address)
        raise OSError(f"network: {address}")

    monkeypatch.setattr(socket.socket, "connect", connect, raising=False)
    monkeypatch.delenv(ALLOW_DOWNLOAD_ENV, raising=False)
    yield
    assert attempts == []


@pytest.fixture
def fake(isolated_dirs):
    """An engine "fake" whose load and speak run the test's functions. Speech
    is 2400 frames a word, and `words` decides whether it has word times."""
    control = SimpleNamespace(load=None, speak=None, words=True, spoken=[], voices=VOICES, closed=0)

    class Model(ModelBase):
        engine_id = "fake"
        sample_rate = 24000

        def load(self, lang_code, device):
            if control.load:
                control.load()
            return True

        def synthesize(self, text, voice, speed, lang_code, params):
            control.spoken.append((text, voice, speed, lang_code))
            if control.speak:
                control.speak(text)
            spoken = text.split()
            t = np.arange(2400 * len(spoken)) / 24000.0
            words = [[w, i * 0.1, (i + 1) * 0.1] for i, w in enumerate(spoken)] if control.words else []
            return Synthesis((0.3 * np.sin(2 * np.pi * 330 * t)).astype(np.float32), words)

        def engine_version(self):
            return "1"

    class Adapter(BackendHooksMixin):
        id = "fake"
        display_name = "Fake"
        capabilities = EngineCapabilities()

        def __init__(self):
            self._engine = EngineRunner(Model())

        @property
        def engine(self):
            return self._engine

        @classmethod
        def get_config_schema(cls):
            return [ConfigField("lang_code", "Language", ConfigFieldType.CHOICE, default="a", group="Generation"),
                    ConfigField("voice", "Voice", ConfigFieldType.CHOICE, default="v1", group="Generation"),
                    *common_fields(caching_default=False)]

        def builtin_voices(self, lang_code=None):
            return [VoiceInfo(id=name, display_name=name, lang_code=code) for code, names in control.voices.items()
                    for name in names if lang_code in (None, code)]

        def cancel(self):
            self._engine.cancel()

        def stop(self):
            control.closed += 1
            super().stop()

    registry.register_engine("fake", Adapter, display_name="Fake")
    yield control
    registry.unregister_engine("fake")
