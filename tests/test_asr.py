"""Tests for the auto-transcription helper (kokoro_gui/engine/asr.py) built on
Audio8/Audio8-ASR-0.1B. Never touches the real model - `_get_model` is
monkeypatched to a fake model/processor pair in every test that exercises
`transcribe_wav`.
"""
import subprocess
import sys
from pathlib import Path

import pytest

import kokoro_gui.engine.asr as asr


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
    # _get_model caches a process-wide singleton - make sure one test's fake
    # model doesn't leak into another's assertions about load behavior.
    monkeypatch.setattr(asr, "_model", None)
    monkeypatch.setattr(asr, "_processor", None)


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
    assert result.stdout.strip() == "True"
