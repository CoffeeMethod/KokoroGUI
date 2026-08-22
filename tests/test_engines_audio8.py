"""Tests for the Audio8 TTS engine backend (kokoro_gui/engines/audio8_tts.py) -
a real, non-Kokoro second backend built on the same TTSEngineBackend contract
`tests/test_engine_backend.py` covers for Kokoro/Dummy.

Never touches the real `transformers`/Audio8 model - `Audio8Engine.generate_segment`
(the one method that would load/call it) is monkeypatched in every test that
exercises generation, mirroring how `fake_pipeline` keeps Kokoro's tests off
the real `kokoro.KPipeline`/eSpeak NG.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import kokoro_engine
from kokoro_gui.engines import audio8_tts, registry
from kokoro_gui.engines.audio8_tts import (
    Audio8BackendAdapter, Audio8Engine, Audio8ReferenceStore,
)
from kokoro_gui.engines.base import ConfigField, EngineCapabilities, VoiceInfo


@pytest.fixture
def isolated_audio8_refs(tmp_path, monkeypatch):
    refs_dir = tmp_path / "audio8_refs"
    monkeypatch.setattr(audio8_tts, "AUDIO8_REFS_DIR", str(refs_dir))
    return refs_dir


@pytest.fixture
def audio8_engine(isolated_audio8_refs, isolated_dirs):
    e = Audio8Engine()
    yield e
    e.worker.stop()


@pytest.fixture
def a_wav(tmp_path):
    path = tmp_path / "sample.wav"
    audio = (0.1 * np.sin(2 * np.pi * 220 * np.arange(1600) / 16000)).astype(np.float32)
    sf.write(str(path), audio, 16000)
    return str(path)


# --- registration / capabilities / schema -----------------------------------

def test_audio8_registered_and_shaped_like_a_real_backend():
    assert "audio8" in registry.list_engines()
    caps = Audio8BackendAdapter.capabilities
    assert isinstance(caps, EngineCapabilities)
    assert caps.supports_voice_mixing is False
    assert caps.supports_voice_cloning is True
    assert caps.supports_multi_speaker_script is True
    assert caps.supports_jit_streaming is False


def test_get_engine_wraps_the_given_engine_instance(audio8_engine):
    backend = registry.get_engine("audio8", engine=audio8_engine)
    assert isinstance(backend, Audio8BackendAdapter)
    assert backend.engine is audio8_engine


def test_config_schema_shape(audio8_engine):
    backend = registry.get_engine("audio8", engine=audio8_engine)
    schema = backend.get_config_schema()
    assert all(isinstance(f, ConfigField) for f in schema)

    keys = {f.key for f in schema}
    assert keys == {"lang_code", "voice", "speed", "split_pattern", "format", "num_threads", "caching"}

    by_key = {f.key: f for f in schema}
    # Fixed, engine-declared language list - NOT GUI-resolved like Kokoro's.
    assert by_key["lang_code"].choices is not None
    assert ("English", "English") in by_key["lang_code"].choices
    # Voice IS GUI-resolved (from saved references), same convention as Kokoro.
    assert by_key["voice"].choices is None
    # Parallelism is capped low - see module docstring on the shared model lock.
    assert by_key["num_threads"].max == 4


def test_importing_module_does_not_load_the_model():
    """Importing this module (which happens at every app startup, to
    register the backend) must not trigger `AutoModel.from_pretrained`/a
    download - only `init_pipeline_async`/first generation does.
    (`transformers` itself is already an indirect hard dependency via the
    `kokoro` package, so the meaningful guarantee is "no model load", not
    "no transformers import" - see this module's docstring.)"""
    code = (
        "import kokoro_engine, kokoro_gui.engines.audio8_tts as a8; "
        "print(a8._model is None and a8._processor is None)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             cwd=str(Path(__file__).resolve().parent.parent))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


# --- Audio8ReferenceStore ----------------------------------------------------

def test_reference_store_round_trip(isolated_audio8_refs, a_wav):
    assert Audio8ReferenceStore.list_references() == []

    saved_path = Audio8ReferenceStore.save_reference("Alice", a_wav, "Hello, this is Alice speaking.")
    assert os.path.exists(saved_path)
    assert Audio8ReferenceStore.list_references() == ["Alice"]
    assert Audio8ReferenceStore.get_transcript("Alice") == "Hello, this is Alice speaking."

    Audio8ReferenceStore.delete_reference("Alice")
    assert Audio8ReferenceStore.list_references() == []


def test_reference_store_sanitizes_path_traversal_name(isolated_audio8_refs, a_wav):
    saved_path = Audio8ReferenceStore.save_reference("../../evil", a_wav, "transcript")

    # Must land inside AUDIO8_REFS_DIR under the basename, not escape it -
    # same pattern as tests/test_resolve_voice_path.py's traversal guard.
    assert os.path.exists(saved_path)
    assert os.path.dirname(saved_path) == str(isolated_audio8_refs)
    assert os.path.basename(saved_path) == "evil.wav"


def test_reference_store_ignores_incomplete_pairs(isolated_audio8_refs):
    os.makedirs(str(isolated_audio8_refs), exist_ok=True)
    (isolated_audio8_refs / "orphan.wav").write_bytes(b"not a real wav")
    assert Audio8ReferenceStore.list_references() == []


def test_get_voices_reflects_saved_references(isolated_audio8_refs, a_wav):
    backend = Audio8BackendAdapter()
    try:
        assert backend.get_voices() == []
        Audio8ReferenceStore.save_reference("Bob", a_wav, "This is Bob.")
        assert backend.get_voices() == [VoiceInfo(id="Bob", display_name="Bob", lang_code=None, is_custom=True)]
    finally:
        backend.cancel()
        backend.engine.worker.stop()


# --- Audio8Engine: resolve_voice_path / resolve_voice_transcript ------------

def test_resolve_voice_path_and_transcript_for_saved_reference(audio8_engine, isolated_audio8_refs, a_wav):
    Audio8ReferenceStore.save_reference("Carol", a_wav, "Carol's voice sample.")

    resolved = audio8_engine.resolve_voice_path("Carol")
    assert os.path.isabs(resolved)
    assert resolved.endswith("Carol.wav")

    assert audio8_engine.resolve_voice_transcript(resolved) == "Carol's voice sample."


def test_resolve_voice_path_falls_back_to_literal_file(audio8_engine, a_wav):
    # Not a saved reference name, but an existing absolute wav path - usable
    # ad-hoc even before being saved under a name.
    assert audio8_engine.resolve_voice_path(a_wav) == a_wav


def test_resolve_voice_transcript_empty_when_no_sidecar(audio8_engine, a_wav):
    assert audio8_engine.resolve_voice_transcript(a_wav) == ""


# --- Audio8Engine.process_chunk_task -----------------------------------------

def _fake_segment(monkeypatch, engine, freq=220.0, sr=44100, n=2200):
    def _gen(text, ref_wav_path, ref_transcript, speed, lang_code):
        t = np.arange(n) / sr
        return (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    monkeypatch.setattr(engine, "generate_segment", _gen)


def test_process_chunk_task_writes_44100hz_audio(audio8_engine, isolated_audio8_refs, isolated_dirs, a_wav, monkeypatch):
    Audio8ReferenceStore.save_reference("Dana", a_wav, "Dana's reference line.")
    _fake_segment(monkeypatch, audio8_engine)

    config = {
        "lang_code": "English", "voice": audio8_engine.resolve_voice_path("Dana"),
        "speed": 1.0, "split_pattern": r"\n+", "filename": "out", "time_id": "1",
        "out_dir": str(isolated_dirs.out_dir), "format": "wav", "caching": False,
        "apply_fx": False,
    }
    files = audio8_engine.process_chunk_task((0, "Hello there.", config), None)

    assert len(files) == 1
    data, sr = sf.read(files[0]["path"])
    assert sr == 44100
    assert np.max(np.abs(data)) > 0.01


# process_chunk_task's own segment-cache-enabled integration test lives in
# tests/test_caching.py (the only module allowed to enable that setting -
# see tests/test_meta_caching_policy.py) as
# test_audio8_process_chunk_task_caching_keys_on_transcript.


def test_cancel_sets_cancel_event(audio8_engine):
    assert not audio8_engine.cancel_event.is_set()
    audio8_engine.cancel()
    assert audio8_engine.cancel_event.is_set()
