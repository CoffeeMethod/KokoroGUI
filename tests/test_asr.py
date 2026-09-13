"""Tests for the auto-transcription helpers (kokoro_gui/engine/asr.py): the
default Audio8/Audio8-ASR-0.1B engine and the offline Vosk engine. Neither
ever touches its real model - `_get_model`/`_get_vosk_model` are monkeypatched
to fakes in every test that exercises `transcribe_wav`.
"""
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import kokoro_gui.engine.asr as asr
from tests.conftest import strip_ansi


def _write_pcm16_mono_wav(path, n_frames=1600, framerate=16000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(b"\x00\x00" * n_frames)


class _FakeInputs(dict):
    def __init__(self):
        super().__init__(input_ids=_FakeTensor())


class _FakeTensor:
    shape = (1, 7)


class _FakeProcessor:
    def apply_chat_template(self, conversation, **kwargs):
        assert conversation[0]["content"][0]["type"] == "audio"
        return _FakeInputs()

    def decode(self, token_ids, skip_special_tokens=True):
        return "  hello from the fake model  "


class _FakeModel:
    def generate(self, **kwargs):
        return [[0] * 7 + [1, 2, 3]]  # prompt tokens + 3 "generated" tokens


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch):
    # _get_model/_get_vosk_model each cache process-wide singletons - make
    # sure one test's fake model doesn't leak into another's assertions
    # about load behavior.
    monkeypatch.setattr(asr, "_model", None)
    monkeypatch.setattr(asr, "_processor", None)
    monkeypatch.setattr(asr, "_vosk_models", {})


def test_transcribe_wav_returns_stripped_decoded_text(monkeypatch, tmp_path):
    monkeypatch.setattr(asr, "_get_model", lambda: (_FakeModel(), _FakeProcessor()))

    wav_path = str(tmp_path / "ref.wav")
    result = asr.transcribe_wav(wav_path)

    assert result == "hello from the fake model"


def test_transcribe_wav_wraps_generation_errors(monkeypatch, tmp_path):
    class _BoomModel:
        def generate(self, **kwargs):
            raise RuntimeError("out of memory")

    monkeypatch.setattr(asr, "_get_model", lambda: (_BoomModel(), _FakeProcessor()))

    with pytest.raises(RuntimeError, match="Transcription failed"):
        asr.transcribe_wav(str(tmp_path / "ref.wav"))


def test_get_model_raises_clear_error_without_transformers(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def _no_transformers(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("no module named transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_transformers)

    with pytest.raises(RuntimeError, match="transformers"):
        asr._get_model()


def test_importing_module_does_not_load_the_model():
    """Importing this module (which happens whenever the Voice Reference
    dock is built) must not trigger `AutoModel.from_pretrained`/download -
    only calling `transcribe_wav` (or `_get_model`) does. (`transformers`
    itself is already an indirect hard dependency via the `kokoro` package,
    so the meaningful guarantee here is "no model load", not "no
    transformers import" - see this module's docstring.) Checked in a fresh
    interpreter, importing `kokoro_engine` first to match real app startup
    order (kokoro_gui/engine/__init__.py's own transitive import chain back
    to kokoro_engine.py means importing any of its submodules cold, without
    kokoro_engine already in sys.modules, hits an unrelated pre-existing
    circular-import ordering requirement)."""
    code = (
        "import kokoro_engine, kokoro_gui.engine.asr as asr; "
        "print(asr._model is None and asr._processor is None)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             cwd=str(Path(__file__).resolve().parent.parent))
    assert result.returncode == 0, result.stderr
    assert strip_ansi(result.stdout).strip() == "True"


# --- Vosk engine ----------------------------------------------------------

class _FakeRecognizer:
    """First `AcceptWaveform` call reports a finalized chunk ("hello"); every
    call after that reports "still listening" (mirrors Vosk's real streaming
    behavior enough to exercise the accumulate-then-join loop)."""

    def __init__(self, model, rate):
        self.model = model
        self.rate = rate
        self._calls = 0

    def SetWords(self, words):
        pass

    def AcceptWaveform(self, data):
        self._calls += 1
        return self._calls == 1

    def Result(self):
        return json.dumps({"text": "hello"})

    def FinalResult(self):
        return json.dumps({"text": "world"})


class _FakeVoskModule:
    KaldiRecognizer = _FakeRecognizer

    class Model:
        def __init__(self, path):
            self.path = path

    @staticmethod
    def SetLogLevel(level):
        pass


def test_transcribe_wav_vosk_joins_recognizer_output(monkeypatch, tmp_path):
    wav_path = tmp_path / "ref.wav"
    _write_pcm16_mono_wav(wav_path, n_frames=8000)  # >4000 frames -> AcceptWaveform called twice

    monkeypatch.setitem(sys.modules, "vosk", _FakeVoskModule())
    monkeypatch.setattr(asr, "_get_vosk_model", lambda path: object())

    result = asr.transcribe_wav(str(wav_path), engine="vosk", model_path=str(tmp_path))

    assert result == "hello world"


def test_transcribe_wav_vosk_converts_stereo_wav_before_transcribing(monkeypatch, tmp_path):
    wav_path = tmp_path / "stereo.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00\x00\x00" * 100)

    monkeypatch.setitem(sys.modules, "vosk", _FakeVoskModule())
    monkeypatch.setattr(asr, "_get_vosk_model", lambda path: object())

    result = asr.transcribe_wav(str(wav_path), engine="vosk", model_path=str(tmp_path))

    assert result == "hello world"


def test_transcribe_wav_vosk_cleans_up_the_converted_temp_file(monkeypatch, tmp_path):
    wav_path = tmp_path / "ref.wav"
    _write_pcm16_mono_wav(wav_path, n_frames=8000)
    converted_path = tmp_path / "converted.wav"
    _write_pcm16_mono_wav(converted_path, n_frames=8000)

    monkeypatch.setattr(asr, "_ensure_pcm16_mono", lambda path: (str(converted_path), str(converted_path)))
    monkeypatch.setitem(sys.modules, "vosk", _FakeVoskModule())
    monkeypatch.setattr(asr, "_get_vosk_model", lambda path: object())

    asr.transcribe_wav(str(wav_path), engine="vosk", model_path=str(tmp_path))

    assert not converted_path.exists()


def test_get_vosk_model_raises_clear_error_without_vosk_package(monkeypatch, tmp_path):
    import builtins
    real_import = builtins.__import__

    def _no_vosk(name, *args, **kwargs):
        if name == "vosk":
            raise ImportError("no module named vosk")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_vosk)

    with pytest.raises(RuntimeError, match="vosk"):
        asr._get_vosk_model(str(tmp_path))


def test_get_vosk_model_requires_a_model_path(monkeypatch):
    monkeypatch.setitem(sys.modules, "vosk", _FakeVoskModule())

    with pytest.raises(RuntimeError, match="model folder"):
        asr._get_vosk_model("")


def test_get_vosk_model_requires_an_existing_directory(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "vosk", _FakeVoskModule())
    missing = str(tmp_path / "does-not-exist")

    with pytest.raises(RuntimeError, match="not found"):
        asr._get_vosk_model(missing)


def test_ensure_pcm16_mono_passes_already_correct_wav_through_unchanged(tmp_path):
    wav_path = tmp_path / "ref.wav"
    _write_pcm16_mono_wav(wav_path)

    result_path, temp_path = asr._ensure_pcm16_mono(str(wav_path))

    assert result_path == str(wav_path)
    assert temp_path is None


def test_ensure_pcm16_mono_downmixes_stereo(tmp_path):
    wav_path = tmp_path / "stereo.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00\x00\x00" * 100)

    result_path, temp_path = asr._ensure_pcm16_mono(str(wav_path))
    try:
        assert temp_path == result_path
        assert result_path != str(wav_path)
        with wave.open(result_path, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
    finally:
        os.remove(result_path)


def test_ensure_pcm16_mono_converts_float_samples(tmp_path):
    wav_path = tmp_path / "float.wav"
    sf.write(str(wav_path), np.zeros(100, dtype=np.float32), 16000, subtype="FLOAT")

    result_path, temp_path = asr._ensure_pcm16_mono(str(wav_path))
    try:
        assert temp_path == result_path
        with wave.open(result_path, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
    finally:
        os.remove(result_path)


# --- VOSK_MODEL_PATH (env-configured, not a GUI setting) -------------------

def test_get_vosk_model_path_reads_and_strips_env_var(monkeypatch):
    monkeypatch.setenv("VOSK_MODEL_PATH", "  /some/model/dir  ")
    assert asr.get_vosk_model_path() == "/some/model/dir"


def test_get_vosk_model_path_defaults_to_empty_when_unset(monkeypatch):
    monkeypatch.delenv("VOSK_MODEL_PATH", raising=False)
    assert asr.get_vosk_model_path() == ""


def test_reload_vosk_model_path_rereads_dotenv_from_cwd(monkeypatch, tmp_path):
    monkeypatch.delenv("VOSK_MODEL_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("VOSK_MODEL_PATH=/from/dotenv\n")

    result = asr.reload_vosk_model_path()

    assert result == "/from/dotenv"
    assert os.environ["VOSK_MODEL_PATH"] == "/from/dotenv"


def test_reload_vosk_model_path_overrides_a_stale_value(monkeypatch, tmp_path):
    monkeypatch.setenv("VOSK_MODEL_PATH", "/stale")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("VOSK_MODEL_PATH=/fresh\n")

    assert asr.reload_vosk_model_path() == "/fresh"


def test_set_vosk_model_path_creates_dotenv_when_none_exists(monkeypatch, tmp_path):
    monkeypatch.delenv("VOSK_MODEL_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / ".env").exists()

    asr.set_vosk_model_path(str(tmp_path / "my-model"))

    assert os.environ["VOSK_MODEL_PATH"] == str(tmp_path / "my-model")
    assert "my-model" in (tmp_path / ".env").read_text()


def test_set_vosk_model_path_updates_an_existing_dotenv_in_place(monkeypatch, tmp_path):
    monkeypatch.delenv("VOSK_MODEL_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("SOME_OTHER_KEY=keep-me\nVOSK_MODEL_PATH=/old\n")

    asr.set_vosk_model_path("/new")

    contents = (tmp_path / ".env").read_text()
    assert "SOME_OTHER_KEY=keep-me" in contents
    assert "/new" in contents
    assert "/old" not in contents
    assert os.environ["VOSK_MODEL_PATH"] == "/new"


def test_set_vosk_model_path_strips_whitespace(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    asr.set_vosk_model_path("  /padded/path  ")

    assert os.environ["VOSK_MODEL_PATH"] == "/padded/path"
    assert asr.get_vosk_model_path() == "/padded/path"


# --- engine registry / dispatch --------------------------------------------

def test_transcribe_wav_rejects_unknown_engine(tmp_path):
    with pytest.raises(ValueError, match="Unknown ASR engine"):
        asr.transcribe_wav(str(tmp_path / "ref.wav"), engine="nope")


def test_get_asr_engine_roundtrip():
    info = asr.get_asr_engine("vosk")
    assert info.id == "vosk"

    with pytest.raises(ValueError, match="Unknown ASR engine"):
        asr.get_asr_engine("nope")
