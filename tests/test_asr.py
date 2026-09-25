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
    monkeypatch.setattr(asr, "_whisper_models", {})
    monkeypatch.setattr(asr, "_whisper_cuda_failed", False)


def test_transcribe_wav_returns_stripped_decoded_text(monkeypatch, tmp_path):
    monkeypatch.setattr(asr, "_get_model", lambda: (_FakeModel(), _FakeProcessor()))

    wav_path = str(tmp_path / "ref.wav")
    result = asr.transcribe_wav(wav_path, engine="audio8")

    assert result == "hello from the fake model"


def test_transcribe_wav_wraps_generation_errors(monkeypatch, tmp_path):
    class _BoomModel:
        def generate(self, **kwargs):
            raise RuntimeError("out of memory")

    monkeypatch.setattr(asr, "_get_model", lambda: (_BoomModel(), _FakeProcessor()))

    with pytest.raises(RuntimeError, match="Transcription failed"):
        asr.transcribe_wav(str(tmp_path / "ref.wav"), engine="audio8")


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
    order (the cold order is covered by
    `test_the_module_imports_cold_for_the_cli`)."""
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
            # The real `vosk.Model` raises this for a folder it can't load,
            # a missing one included.
            if not os.path.isdir(path):
                raise Exception("Failed to create a model")
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


def test_get_vosk_model_names_the_folder_when_vosk_rejects_it(monkeypatch, tmp_path):
    """The real `vosk.Model` raises for a missing folder and for one without
    model files alike; the error names the path and what belongs there."""
    monkeypatch.setitem(sys.modules, "vosk", _FakeVoskModule())
    missing = str(tmp_path / "does-not-exist")

    with pytest.raises(RuntimeError, match="does-not-exist.*alphacephei"):
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


# --- Whisper engine ---------------------------------------------------------

class _FakeWord:
    def __init__(self, word, start, end):
        self.word, self.start, self.end = word, start, end


class _FakeSegment:
    def __init__(self, text, words):
        self.text = text
        self.words = [_FakeWord(*w) for w in words]


class _FakeWhisperModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        segments = [
            _FakeSegment(" Hello there.", [(" Hello", 0.0, 0.4), (" there.", 0.45, 0.9)]),
            _FakeSegment(" General Kenobi.", [(" General", 1.2, 1.6), (" Kenobi.", 1.65, 2.3)]),
        ]
        return iter(segments), object()


def test_whisper_is_the_default_engine():
    assert asr.DEFAULT_ASR_ENGINE == "whisper"
    assert asr.ASR_ENGINES[0].id == "whisper"


def test_transcribe_wav_whisper_joins_segment_text(monkeypatch, tmp_path):
    fake = _FakeWhisperModel()
    monkeypatch.setattr(asr, "_get_whisper_model", lambda name=None: fake)
    assert asr.transcribe_wav(str(tmp_path / "ref.wav")) == "Hello there. General Kenobi."
    assert fake.calls[0][1] == {"word_timestamps": True, "vad_filter": True}


def test_transcribe_wav_words_whisper_returns_ordered_timed_words(monkeypatch, tmp_path):
    monkeypatch.setattr(asr, "_get_whisper_model", lambda name=None: _FakeWhisperModel())
    words = asr.transcribe_wav_words(str(tmp_path / "ref.wav"), engine="whisper")
    assert [w for w, _s, _e in words] == ["Hello", "there.", "General", "Kenobi."]
    times = [t for _w, s, e in words for t in (s, e)]
    assert times == sorted(times)


def test_transcribe_wav_words_audio8_has_no_timestamps(tmp_path):
    with pytest.raises(NotImplementedError, match="no word timestamps"):
        asr.transcribe_wav_words(str(tmp_path / "ref.wav"), engine="audio8")


class _WordsRecognizer(_FakeRecognizer):
    def Result(self):
        return json.dumps({"text": "hello", "result": [{"word": "hello", "start": 0.1, "end": 0.5}]})

    def FinalResult(self):
        return json.dumps({"text": "world", "result": [{"word": "world", "start": 0.6, "end": 1.0}]})


def test_transcribe_wav_words_vosk_returns_its_word_results(monkeypatch, tmp_path):
    wav_path = tmp_path / "ref.wav"
    _write_pcm16_mono_wav(wav_path, n_frames=8000)
    fake_vosk = _FakeVoskModule()
    fake_vosk.KaldiRecognizer = _WordsRecognizer
    monkeypatch.setitem(sys.modules, "vosk", fake_vosk)
    monkeypatch.setattr(asr, "_get_vosk_model", lambda path: object())

    words = asr.transcribe_wav_words(str(wav_path), engine="vosk", model_path=str(tmp_path))

    assert words == [("hello", 0.1, 0.5), ("world", 0.6, 1.0)]


def test_whisper_model_name_and_size_follow_the_env(monkeypatch):
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    assert asr.get_whisper_model_name() == "large-v3-turbo"
    assert asr.whisper_model_size() == "1.6 GB"
    monkeypatch.setenv("WHISPER_MODEL", "small")
    assert asr.get_whisper_model_name() == "small"
    assert asr.whisper_model_size() == "484 MB"
    monkeypatch.setenv("WHISPER_MODEL", "some/custom-model")
    assert asr.whisper_model_size() is None


def test_whisper_model_cached_probes_without_downloading(monkeypatch):
    import faster_whisper.utils as fw_utils

    calls = []

    def fake_download(name, local_files_only=False, **kwargs):
        calls.append((name, local_files_only))
        raise FileNotFoundError("not cached")

    monkeypatch.setattr(fw_utils, "download_model", fake_download)
    assert asr.whisper_model_cached("large-v3-turbo") is False
    assert calls == [("large-v3-turbo", True)]


def test_whisper_model_cached_accepts_a_local_model_folder(tmp_path):
    assert asr.whisper_model_cached(str(tmp_path)) is True


def test_get_whisper_model_raises_clear_error_without_faster_whisper(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _no_fw(name, *args, **kwargs):
        if name == "faster_whisper" or name.startswith("faster_whisper."):
            raise ImportError("no faster_whisper")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_fw)
    with pytest.raises(RuntimeError, match="faster-whisper"):
        asr._get_whisper_model("tiny")


def test_cli_words_flag_prints_timed_words(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(asr, "_get_whisper_model", lambda name=None: _FakeWhisperModel())
    assert asr._main([str(tmp_path / "ref.wav"), "whisper", "--words"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    assert lines[0].split() == ["0.00", "0.40", "Hello"]


def test_importing_module_does_not_load_whisper():
    code = (
        "import sys, kokoro_engine, kokoro_gui.engine.asr as asr; "
        "print(asr._whisper_models == {} and 'faster_whisper' not in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            cwd=str(Path(__file__).resolve().parent.parent))
    assert result.returncode == 0, result.stderr
    assert strip_ansi(result.stdout).strip() == "True"


class _BrokenCudaModel(_FakeWhisperModel):
    kokoro_device = "cuda"

    def transcribe(self, path, **kwargs):
        raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")


def test_whisper_falls_back_to_the_cpu_when_cuda_fails(monkeypatch, tmp_path):
    """CTranslate2 can see a GPU through the driver without the CUDA 12
    runtime libraries installed; the first run then fails. It retries on
    the CPU and stays there."""
    devices = []

    def fake_get(name=None):
        device, _compute = asr._whisper_device_and_compute_type()
        devices.append("cpu" if asr._whisper_cuda_failed else "cuda")
        return _FakeWhisperModel() if asr._whisper_cuda_failed else _BrokenCudaModel()

    monkeypatch.setattr(asr, "_get_whisper_model", fake_get)
    assert asr.transcribe_wav(str(tmp_path / "ref.wav")) == "Hello there. General Kenobi."
    assert devices == ["cuda", "cpu"]
    assert asr._whisper_cuda_failed
    assert asr._whisper_device_and_compute_type() == ("cpu", "int8")


def test_a_cpu_failure_is_not_retried(monkeypatch, tmp_path):
    class _BrokenCpuModel(_FakeWhisperModel):
        def transcribe(self, path, **kwargs):
            raise RuntimeError("bad audio")

    monkeypatch.setattr(asr, "_get_whisper_model", lambda name=None: _BrokenCpuModel())
    with pytest.raises(RuntimeError, match="Transcription failed: bad audio"):
        asr.transcribe_wav(str(tmp_path / "ref.wav"))
    assert not asr._whisper_cuda_failed


def test_the_module_imports_cold_for_the_cli():
    """`python -m kokoro_gui.engine.asr` imports the engine package before
    kokoro_engine; that order used to hit a circular import."""
    for code in ("import kokoro_gui.engine.asr", "import kokoro_gui.engine.caching"):
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                cwd=str(Path(__file__).resolve().parent.parent))
        assert result.returncode == 0, result.stderr
