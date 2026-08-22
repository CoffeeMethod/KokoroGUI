"""Tests for process_chunk_task's caching logic (kokoro_gui/engine/caching.py)
and the compute_cache_key helper it's built on
(PLAN_qt_and_engine_abstraction.md workstream 2).

This is the ONLY test module allowed to pass caching=True - see
tests/test_meta_caching_policy.py for the enforced guard.
"""
import os

import numpy as np
import pytest
import soundfile as sf

import kokoro_engine
from kokoro_gui.engine.caching import CACHE_SCHEMA_VERSION, compute_cache_key


def _hash(text, config, eff_speed=None, lang_code=None, engine_id="kokoro"):
    return compute_cache_key(
        text, config["voice"],
        eff_speed if eff_speed is not None else config["speed"],
        lang_code if lang_code is not None else config["lang_code"],
        engine_id,
    )


def test_cache_miss_writes_raw_pre_fx_audio(engine, fake_pipeline, isolated_dirs, make_config):
    config = make_config(caching=True, volume=0.5)
    results = engine.process_chunk_task((0, "Hello world.", config), None)

    h = _hash("Hello world.", config)
    cache_file = isolated_dirs.cache_dir / f"{h}_0.wav"
    assert cache_file.exists()

    cached_audio, _ = sf.read(str(cache_file))
    output_audio, _ = sf.read(results[0]["path"])

    # Cache stores the RAW pipeline output; the on-disk output segment has
    # volume=0.5 applied on top - so the cached peak should be noticeably louder.
    assert np.max(np.abs(cached_audio)) > np.max(np.abs(output_audio)) * 1.5


def test_cache_hit_skips_pipeline_call(engine, isolated_dirs, make_config, monkeypatch):
    config = make_config(caching=True)
    text = "Hello world."
    h = _hash(text, config)

    audio = (0.1 * np.sin(2 * np.pi * 220 * np.arange(1200) / 24000)).astype(np.float32)
    sf.write(str(isolated_dirs.cache_dir / f"{h}_0.wav"), audio, 24000)

    def _boom(lang_code="a"):
        raise AssertionError("pipeline should not be called on a cache hit")

    monkeypatch.setattr(kokoro_engine, "get_thread_pipeline", _boom)

    results = engine.process_chunk_task((0, text, config), None)

    assert len(results) == 1
    assert os.path.exists(results[0]["path"])


def test_cache_key_ignores_split_pattern(engine, fake_pipeline, make_config, monkeypatch):
    text = "Hello world."
    config1 = make_config(caching=True, split_pattern=r"\n+")
    engine.process_chunk_task((0, text, config1), None)

    def _boom(lang_code="a"):
        raise AssertionError("pipeline should not be called - same hash should hit cache")

    monkeypatch.setattr(kokoro_engine, "get_thread_pipeline", _boom)

    # Known limitation (kokoro_engine.py:618-621): split_pattern is not part
    # of the cache key, so a different split_pattern that yields the same
    # predicted segment count still counts as a cache hit.
    config2 = make_config(caching=True, split_pattern=r"\n\n+")
    results = engine.process_chunk_task((0, text, config2), None)

    assert len(results) == 1


def test_cache_partial_files_missing_forces_regeneration(engine, fake_pipeline, isolated_dirs, make_config):
    text = "Seg one.\n\nSeg two."
    config = make_config(caching=True)
    h = _hash(text, config)

    # Only the first of the two expected segments is cached.
    audio = (0.1 * np.sin(2 * np.pi * 220 * np.arange(1200) / 24000)).astype(np.float32)
    sf.write(str(isolated_dirs.cache_dir / f"{h}_0.wav"), audio, 24000)

    results = engine.process_chunk_task((0, text, config), None)

    assert len(results) == 2
    assert (isolated_dirs.cache_dir / f"{h}_0.wav").exists()
    assert (isolated_dirs.cache_dir / f"{h}_1.wav").exists()


def test_pitch_affects_cache_key(engine, fake_pipeline, isolated_dirs, make_config):
    text = "Hello world."
    config_a = make_config(caching=True, pitch=0.0)
    config_b = make_config(caching=True, pitch=5.0)

    engine.process_chunk_task((0, text, config_a), None)
    engine.process_chunk_task((0, text, config_b), None)

    cache_files = list(isolated_dirs.cache_dir.glob("*_0.wav"))
    assert len(cache_files) == 2


def test_speed_affects_cache_key(engine, fake_pipeline, isolated_dirs, make_config):
    text = "Hello world."
    config_a = make_config(caching=True, speed=1.0)
    config_b = make_config(caching=True, speed=1.5)

    engine.process_chunk_task((0, text, config_a), None)
    engine.process_chunk_task((0, text, config_b), None)

    cache_files = list(isolated_dirs.cache_dir.glob("*_0.wav"))
    assert len(cache_files) == 2


# --- Workstream 2 hardening: compute_cache_key in isolation -----------------

def test_compute_cache_key_is_deterministic_and_sha256():
    h1 = compute_cache_key("Hello.", "af_heart", 1.0, "a")
    h2 = compute_cache_key("Hello.", "af_heart", 1.0, "a")

    assert h1 == h2
    assert len(h1) == 64  # sha256 hex digest, not md5's 32
    assert all(c in "0123456789abcdef" for c in h1)


def test_compute_cache_key_takes_only_what_it_needs():
    # Not a whole config dict - just the five inputs that actually determine
    # a segment's content. out_dir/filename/format/normalize/trim/the FX
    # chain/num_threads/etc. never even get a chance to leak into the hash,
    # because the function has nowhere to read them from.
    import inspect

    params = list(inspect.signature(compute_cache_key).parameters)
    assert params == ["text", "voice", "eff_speed", "lang_code", "engine_id", "engine_version"]


def test_compute_cache_key_differs_by_each_input():
    base = compute_cache_key("Hello.", "af_heart", 1.0, "a")

    assert compute_cache_key("Goodbye.", "af_heart", 1.0, "a") != base
    assert compute_cache_key("Hello.", "af_bella", 1.0, "a") != base
    assert compute_cache_key("Hello.", "af_heart", 1.5, "a") != base
    assert compute_cache_key("Hello.", "af_heart", 1.0, "b") != base
    assert compute_cache_key("Hello.", "af_heart", 1.0, "a", engine_id="dummy") != base
    assert compute_cache_key("Hello.", "af_heart", 1.0, "a", engine_version="1.2.3") != \
        compute_cache_key("Hello.", "af_heart", 1.0, "a", engine_version="1.2.4")


# --- Workstream 2 hardening: cache invalidation through process_chunk_task --

def test_custom_voice_content_change_invalidates_cache(engine, fake_pipeline, isolated_dirs, make_config):
    """Remixing and re-saving a custom voice under the same name (a real
    workflow - VoiceMixingMixin.mix_voices) must invalidate old cache
    entries for that name, since the name alone no longer identifies what
    was actually generated."""
    voice_path = isolated_dirs.custom_voices / "MyMix.pt"
    voice_path.write_bytes(b"tensor-content-v1")
    resolved = engine.resolve_voice_path("MyMix")

    text = "Hello world."
    config = make_config(caching=True, voice=resolved)

    engine.process_chunk_task((0, text, config), None)
    files_v1 = set(isolated_dirs.cache_dir.glob("*_0.wav"))
    assert len(files_v1) == 1

    # Re-save under the same path/name with different content - and force a
    # different mtime so the in-memory fingerprint cache can't coast on a
    # coarse filesystem timestamp resolution masking the change.
    voice_path.write_bytes(b"tensor-content-v2-longer-and-different")
    future = os.path.getmtime(voice_path) + 5
    os.utime(voice_path, (future, future))

    engine.process_chunk_task((0, text, config), None)
    files_v2 = set(isolated_dirs.cache_dir.glob("*_0.wav"))

    assert len(files_v2) == 2
    assert files_v1 < files_v2  # old entry untouched, a new one was added


def test_engine_id_change_invalidates_cache(engine, fake_pipeline, isolated_dirs, make_config):
    text = "Hello world."
    config_a = make_config(caching=True, engine_id="kokoro")
    config_b = make_config(caching=True, engine_id="some-other-engine")

    engine.process_chunk_task((0, text, config_a), None)
    engine.process_chunk_task((0, text, config_b), None)

    cache_files = list(isolated_dirs.cache_dir.glob("*_0.wav"))
    assert len(cache_files) == 2


def test_engine_version_change_invalidates_cache(engine, fake_pipeline, isolated_dirs, make_config, monkeypatch):
    import kokoro_gui.engine.caching as caching_mod

    text = "Hello world."
    config = make_config(caching=True)

    monkeypatch.setattr(caching_mod, "get_engine_version", lambda engine_id="kokoro": "1.0.0")
    engine.process_chunk_task((0, text, config), None)

    monkeypatch.setattr(caching_mod, "get_engine_version", lambda engine_id="kokoro": "2.0.0")
    engine.process_chunk_task((0, text, config), None)

    cache_files = list(isolated_dirs.cache_dir.glob("*_0.wav"))
    assert len(cache_files) == 2


def test_schema_version_bump_invalidates_cache(engine, fake_pipeline, isolated_dirs, make_config, monkeypatch):
    import kokoro_gui.engine.caching as caching_mod

    text = "Hello world."
    config = make_config(caching=True)

    engine.process_chunk_task((0, text, config), None)

    monkeypatch.setattr(caching_mod, "CACHE_SCHEMA_VERSION", CACHE_SCHEMA_VERSION + 1)
    engine.process_chunk_task((0, text, config), None)

    cache_files = list(isolated_dirs.cache_dir.glob("*_0.wav"))
    assert len(cache_files) == 2
