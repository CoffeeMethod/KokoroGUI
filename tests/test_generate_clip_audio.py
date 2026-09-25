"""Tests for KokoroEngine.generate_clip_audio (kokoro_gui/engine/conversion.py) -
the per-clip Generate entry point for the DAW redesign's timeline dock.
Mirrors tests/test_generate_preview.py's conventions."""
import asyncio
import os


def test_generate_clip_audio_writes_output_and_returns_segment_dicts(engine, fake_pipeline, make_config):
    config = make_config()
    results = asyncio.run(engine.generate_clip_audio((0, "Hello world.", config)))

    assert len(results) == 1
    assert os.path.exists(results[0]["path"])
    assert results[0]["text"] == "Hello world."
    assert results[0]["duration"] > 0


def test_generate_clip_audio_resolves_voice_path(engine, fake_pipeline, make_config, monkeypatch):
    calls = []
    original_resolve = engine.resolve_voice_path

    def spy(voice):
        calls.append(voice)
        return original_resolve(voice)

    monkeypatch.setattr(engine, "resolve_voice_path", spy)

    config = make_config(voice="af_heart")
    asyncio.run(engine.generate_clip_audio((0, "Hello.", config)))

    assert calls == ["af_heart"]


def test_generate_clip_audio_creates_missing_out_dir(engine, fake_pipeline, make_config, tmp_path):
    missing_dir = tmp_path / "brand_new_out_dir"
    assert not missing_dir.exists()

    config = make_config(out_dir=str(missing_dir))
    results = asyncio.run(engine.generate_clip_audio((0, "Hello.", config)))

    assert missing_dir.exists()
    assert os.path.exists(results[0]["path"])


def test_generate_clip_audio_clears_stale_cancel_event(engine, fake_pipeline, make_config):
    engine.cancel_event.set()  # simulate a previously cancelled run left set

    config = make_config()
    results = asyncio.run(engine.generate_clip_audio((0, "Hello.", config)))

    assert len(results) == 1  # not silently [] because cancel_event was still set


def test_generate_clip_audio_does_not_mutate_caller_config(engine, fake_pipeline, make_config):
    config = make_config(voice="af_heart")
    original = dict(config)

    asyncio.run(engine.generate_clip_audio((0, "Hello.", config)))

    assert config == original  # generate_clip_audio copies the dict before resolving voice


# Note: a caching-enabled cache-hit test for generate_clip_audio lives in
# tests/test_caching.py instead of here - tests/test_meta_caching_policy.py
# enforces that the caching config flag is only ever turned on in that one
# module.


def test_generate_clip_audio_applies_the_lexicon(engine, fake_pipeline, make_config, monkeypatch):
    seen = []
    real_call = type(fake_pipeline).__call__

    def spy(self, text, *a, **k):
        seen.append(text)
        return real_call(self, text, *a, **k)

    monkeypatch.setattr(type(fake_pipeline), "__call__", spy)
    config = make_config(lexicon={"Nguyen": "Win"})
    results = asyncio.run(engine.generate_clip_audio((0, "Mr Nguyen arrived.", config)))

    assert seen == ["Mr Win arrived."]
    assert results[0]["text"] == "Mr Win arrived."


def test_generate_clip_audio_reports_words_onset_and_tail(engine, fake_pipeline, make_config):
    config = make_config()
    results = asyncio.run(engine.generate_clip_audio((0, "Hello brave world.", config)))

    words = results[0]["words"]
    assert [w[0] for w in words] == ["Hello", "brave", "world."]
    assert words[0][1] == 0.0
    assert all(a[2] <= b[1] + 1e-6 for a, b in zip(words, words[1:]))
    # FakePipeline's sine starts at 0, so the first sample under the
    # threshold is the onset and the audio runs to the end.
    assert results[0]["onset_s"] >= 0.0
    assert results[0]["tail_s"] >= 0.0
