"""Tests for process_chunk_task's caching logic (kokoro_engine.py:568-705).

This is the ONLY test module allowed to pass caching=True - see
tests/test_meta_caching_policy.py for the enforced guard.
"""
import hashlib
import os

import numpy as np
import pytest
import soundfile as sf

import kokoro_engine


def _hash(text, voice, speed, lang_code):
    return hashlib.md5(f"{text}|{voice}|{speed}|{lang_code}".encode("utf-8")).hexdigest()


def test_cache_miss_writes_raw_pre_fx_audio(engine, fake_pipeline, isolated_dirs, make_config):
    config = make_config(caching=True, volume=0.5)
    results = engine.process_chunk_task((0, "Hello world.", config), None)

    h = _hash("Hello world.", config["voice"], config["speed"], config["lang_code"])
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
    h = _hash(text, config["voice"], config["speed"], config["lang_code"])

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
    h = _hash(text, config["voice"], config["speed"], config["lang_code"])

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
