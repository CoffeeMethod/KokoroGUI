"""Tests for KokoroEngine.mix_voices (kokoro_engine.py:280-337)."""
import asyncio
import os

import pytest
import torch

import kokoro_engine


@pytest.mark.parametrize("op", ["mix", "add", "subtract", "multiply", "divide"])
def test_mix_voices_op_produces_saved_tensor(engine, fake_pipeline, isolated_dirs, op):
    engine.pipeline = fake_pipeline
    success, path, tensor = asyncio.run(engine.mix_voices("af_heart", "af_bella", 0.5, "myvoice", op=op))

    assert success is True
    assert os.path.exists(path)
    loaded = torch.load(path)
    assert torch.equal(loaded, tensor)


def test_mix_voices_missing_voice_returns_error_tuple(engine, fake_pipeline, monkeypatch):
    engine.pipeline = fake_pipeline
    monkeypatch.setattr(fake_pipeline, "load_voice", lambda name: None)

    success, msg, tensor = asyncio.run(engine.mix_voices("af_heart", "af_bella", 0.5, "myvoice"))

    assert success is False
    assert tensor is None
    assert isinstance(msg, str) and msg


def test_mix_voices_name_basename_sanitized(engine, fake_pipeline, isolated_dirs):
    engine.pipeline = fake_pipeline
    success, path, _ = asyncio.run(engine.mix_voices("af_heart", "af_bella", 0.5, "../../evil"))

    assert success is True
    assert os.path.dirname(path) == str(isolated_dirs.custom_voices)
    assert os.path.basename(path) == "evil.pt"


def test_mix_voices_uses_existing_pipeline_before_thread_local_fallback(engine, fake_pipeline, monkeypatch):
    engine.pipeline = fake_pipeline

    def _boom(lang_code="a"):
        raise AssertionError("get_thread_pipeline should not be used when engine.pipeline is already set")

    monkeypatch.setattr(kokoro_engine, "get_thread_pipeline", _boom)

    success, path, _ = asyncio.run(engine.mix_voices("af_heart", "af_bella", 0.5, "myvoice"))

    assert success is True
