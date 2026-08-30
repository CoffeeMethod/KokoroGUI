"""Tests for kokoro_engine.KokoroEngine.init_pipeline_async's fallback when
a foreign-format lang_code reaches KPipeline directly.

Motivating bug report: a user who'd previously used the Audio8 backend (whose
lang_code values are full language names, e.g. "English" - see
kokoro_gui/engines/audio8_tts.py) saw "Pipeline Init Failed" on every launch
with the Kokoro engine active, even though the Settings dock's own combo
(kokoro_gui/qt/docks/settings_dock.py's SchemaFormWidget._set_combo)
reconciles an unrecognized stored value to a valid default when building the
Language field - confirmed independently, against the user's real
config_qt.json, to correctly resolve "English" to "a" before ever calling
init_pipeline_async. Since the exact mechanism by which a foreign value
still reached the real kokoro.KPipeline (whose own internal validation
raised the reported error) couldn't be pinned down further without a live
repro, this fallback hardens init_pipeline_async itself as a second,
independent line of defense - whatever reaches it, a value KPipeline
rejects gets one retry against Kokoro's own safe default instead of
surfacing a raw third-party AssertionError.
"""
import asyncio

import kokoro_engine


def test_falls_back_to_safe_default_when_lang_code_is_invalid(engine, monkeypatch):
    calls = []

    def fake_kpipeline(lang_code="a"):
        calls.append(lang_code)
        if lang_code != "a":
            raise AssertionError((lang_code, {"a": "American English"}))
        return object()

    monkeypatch.setattr(kokoro_engine, "KPipeline", fake_kpipeline)

    result = asyncio.run(engine.init_pipeline_async("English"))

    assert result is True
    assert calls == ["English", "a"]  # tried the given value, then fell back once
    assert engine.pipeline is not None


def test_does_not_retry_when_the_safe_default_itself_fails(engine, monkeypatch):
    """"a" failing is a real problem (missing model, no network, etc.) - not
    the foreign-lang_code case this fallback exists for - so it must not
    mask that failure behind a pointless second attempt with the same
    value."""
    calls = []

    def fake_kpipeline(lang_code="a"):
        calls.append(lang_code)
        raise RuntimeError("no network")

    monkeypatch.setattr(kokoro_engine, "KPipeline", fake_kpipeline)

    result = asyncio.run(engine.init_pipeline_async("a"))

    assert result is False
    assert calls == ["a"]


def test_valid_lang_code_succeeds_without_a_retry(engine, monkeypatch):
    calls = []

    def fake_kpipeline(lang_code="a"):
        calls.append(lang_code)
        return object()

    monkeypatch.setattr(kokoro_engine, "KPipeline", fake_kpipeline)

    result = asyncio.run(engine.init_pipeline_async("b"))

    assert result is True
    assert calls == ["b"]
