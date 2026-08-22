"""Tests for KokoroEngine.generate_preview (kokoro_engine.py:339-430)."""
import asyncio
import os

import torch


def test_generate_preview_writes_wav_file(engine, fake_pipeline, tmp_path):
    out_path = str(tmp_path / "preview.wav")
    ok = asyncio.run(engine.generate_preview("Hello there.", "af_heart", 1.0, out_path))

    assert ok is True
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_generate_preview_truncates_multispeaker_to_two_segments(engine, fake_pipeline, tmp_path, monkeypatch):
    calls = []
    orig_call = fake_pipeline.__call__

    def spy(self, text, voice=None, speed=1.0, split_pattern=r"\n+"):
        calls.append(text)
        return orig_call(text, voice=voice, speed=speed, split_pattern=split_pattern)

    monkeypatch.setattr(type(fake_pipeline), "__call__", spy)

    text = "[SpkA]: one\n\n[SpkB]: two\n\n[SpkC]: three"
    out_path = str(tmp_path / "preview.wav")
    asyncio.run(engine.generate_preview(text, "af_heart", 1.0, out_path))

    assert len(calls) == 2


def test_generate_preview_applies_lexicon_from_extra_config(engine, fake_pipeline, tmp_path, monkeypatch):
    seen = {}
    orig_call = fake_pipeline.__call__

    def spy(self, text, voice=None, speed=1.0, split_pattern=r"\n+"):
        seen["text"] = text
        return orig_call(text, voice=voice, speed=speed, split_pattern=split_pattern)

    monkeypatch.setattr(type(fake_pipeline), "__call__", spy)

    out_path = str(tmp_path / "preview.wav")
    asyncio.run(engine.generate_preview(
        "Hello world.", "af_heart", 1.0, out_path,
        extra_config={"lexicon": {"world": "planet"}},
    ))

    assert "planet" in seen["text"]


def test_generate_preview_voice_tensor_sets_pipeline_voices_dict(engine, fake_pipeline, tmp_path):
    tensor = torch.zeros(510, 1, 256)
    out_path = str(tmp_path / "preview.wav")
    ok = asyncio.run(engine.generate_preview("Hello.", "ignored_voice", 1.0, out_path, voice_tensor=tensor))

    assert ok is True
    assert "_preview_temp" in fake_pipeline.voices


def test_generate_preview_no_pipeline_returns_false(engine, monkeypatch, tmp_path):
    import kokoro_engine
    monkeypatch.setattr(kokoro_engine, "get_thread_pipeline", lambda lang_code="a": None)

    out_path = str(tmp_path / "preview.wav")
    ok = asyncio.run(engine.generate_preview("Hello.", "af_heart", 1.0, out_path))

    assert ok is False
