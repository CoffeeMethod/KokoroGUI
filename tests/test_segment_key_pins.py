"""Pins the exact segment key for each backend (ENGINE_AGNOSTIC plan, A1).

A segment file is named by its key and the dirty check compares stored keys,
so a refactor that shifts a key by one byte stales every clip in every
project without breaking anything visibly. These hex strings were taken
from the code before the engine-agnostic refactor and must never change:
if one fails, the refactor changed a key input, not the test.

The installed `kokoro` package version enters Kokoro's key, so it's patched
to the version `requirements.txt` pins; Audio8's key carries its model id;
Dummy's carries "unknown".
"""
import importlib.metadata

import pytest

from kokoro_gui.engine import caching
from kokoro_gui.engines import audio8_tts
from kokoro_gui.engines.audio8_tts import Audio8BackendAdapter, Audio8Engine
from kokoro_gui.engines.dummy import DummyBackendAdapter, DummyEngine
from kokoro_gui.engines.kokoro import KokoroBackendAdapter
from tests.conftest import StubEngine

TEXT = "The quick brown fox jumps over the lazy dog."
PT_BYTES = b"not really a tensor, only hashed"
WAV_BYTES = b"RIFF fixed bytes standing in for a reference wav"
TRANSCRIPT = "This is what the reference says."


@pytest.fixture(autouse=True)
def _pinned_kokoro_version(monkeypatch):
    real = importlib.metadata.version

    def version(name):
        return "0.9.4" if name == "kokoro" else real(name)

    monkeypatch.setattr(importlib.metadata, "version", version)


@pytest.fixture
def isolated_audio8_refs(tmp_path, monkeypatch):
    refs_dir = tmp_path / "audio8_refs"
    refs_dir.mkdir()
    monkeypatch.setattr(audio8_tts, "AUDIO8_REFS_DIR", str(refs_dir))
    return refs_dir


def _config(**overrides):
    config = {"voice": "af_heart", "speed": 1.0, "pitch": 0.0, "lang_code": "a", "take": 0}
    config.update(overrides)
    return config


def test_kokoro_builtin_voice_key(isolated_dirs):
    backend = KokoroBackendAdapter(StubEngine())
    key = caching.segment_key(TEXT, _config(), backend)
    assert key == "9255535fc432a148a0f9392a3c96ac3e87c093511766a9765aa2a466f9a77f98"


def test_kokoro_custom_mix_key(isolated_dirs):
    (isolated_dirs.custom_voices / "my_mix.pt").write_bytes(PT_BYTES)
    backend = KokoroBackendAdapter(StubEngine())
    key = caching.segment_key(TEXT, _config(voice="my_mix", speed=1.2, take=2), backend)
    assert key == "0db5f1dba14d89c58b5b13ee95de46485b7b6f35f363b7f8384baaca95e07b7e"


def test_kokoro_key_with_pitch_and_british_english(isolated_dirs):
    backend = KokoroBackendAdapter(StubEngine())
    key = caching.segment_key(TEXT, _config(voice="bm_daniel", pitch=2.0, lang_code="b"), backend)
    assert key == "6a57abb67b908015f1548e49933f2b5084fe204388d65a20ef82c803a9f2c29f"


def test_dummy_key(isolated_dirs):
    engine = DummyEngine()
    try:
        backend = DummyBackendAdapter(engine)
        key = caching.segment_key(TEXT, _config(voice="dummy"), backend)
    finally:
        engine.worker.stop()
    assert key == "20ffc6da3faa450511a619de7f9c02af9d1d9cfabbcead41fa3a0877278b7385"


def test_audio8_reference_key(isolated_dirs, isolated_audio8_refs):
    (isolated_audio8_refs / "narrator.wav").write_bytes(WAV_BYTES)
    (isolated_audio8_refs / "narrator.txt").write_text(TRANSCRIPT, encoding="utf-8")
    engine = Audio8Engine()
    try:
        backend = Audio8BackendAdapter(engine)
        config = _config(voice="narrator", lang_code="English", temperature=0.7, top_k=40)
        key = caching.segment_key(TEXT, config, backend)
    finally:
        engine.worker.stop()
    assert key == "1f9a892343b0be41d365f7f9cb0d2702c6c13617272bbad0a6ce2c112b732b27"
