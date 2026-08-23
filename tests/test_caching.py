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
from kokoro_gui.engines import audio8_tts
from kokoro_gui.engines.audio8_tts import Audio8Engine, Audio8ReferenceStore


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
    # Not a whole config dict - just the inputs that actually determine a
    # segment's content. out_dir/filename/format/normalize/trim/the FX
    # chain/num_threads/etc. never even get a chance to leak into the hash,
    # because the function has nowhere to read them from. `extra` is the one
    # deliberate escape hatch - see test_compute_cache_key_extra_* below.
    import inspect

    params = list(inspect.signature(compute_cache_key).parameters)
    assert params == ["text", "voice", "eff_speed", "lang_code", "engine_id", "engine_version", "extra"]


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


def test_compute_cache_key_extra_none_matches_no_extra_arg():
    """`extra` is additive - omitting it entirely and passing `extra=None`
    must hash identically, and both must match what the function produced
    before `extra` existed (Kokoro's/Dummy's call sites never pass it)."""
    without_arg = compute_cache_key("Hello.", "af_heart", 1.0, "a")
    with_none = compute_cache_key("Hello.", "af_heart", 1.0, "a", extra=None)
    with_empty = compute_cache_key("Hello.", "af_heart", 1.0, "a", extra={})
    assert without_arg == with_none == with_empty


def test_compute_cache_key_extra_dict_changes_hash():
    """Audio8Engine folds a reference transcript into `extra` so editing the
    transcript for the same reference wav (same name, same content hash)
    still invalidates the cache - see kokoro_gui/engines/audio8_tts.py's
    process_chunk_task."""
    base = compute_cache_key("Hello.", "/refs/alice.wav", 1.0, "English", engine_id="audio8",
                              extra={"ref_transcript": "Hi there."})
    changed = compute_cache_key("Hello.", "/refs/alice.wav", 1.0, "English", engine_id="audio8",
                                 extra={"ref_transcript": "Hi there!"})
    same = compute_cache_key("Hello.", "/refs/alice.wav", 1.0, "English", engine_id="audio8",
                              extra={"ref_transcript": "Hi there."})
    assert base != changed
    assert base == same


def test_schema_version_bump_invalidates_cache(engine, fake_pipeline, isolated_dirs, make_config, monkeypatch):
    import kokoro_gui.engine.caching as caching_mod

    text = "Hello world."
    config = make_config(caching=True)

    engine.process_chunk_task((0, text, config), None)

    monkeypatch.setattr(caching_mod, "CACHE_SCHEMA_VERSION", CACHE_SCHEMA_VERSION + 1)
    engine.process_chunk_task((0, text, config), None)

    cache_files = list(isolated_dirs.cache_dir.glob("*_0.wav"))
    assert len(cache_files) == 2


# --- Audio8Engine: its own hand-rolled caching (kokoro_gui/engines/audio8_tts.py) --
#
# Audio8Engine doesn't use CachingMixin (see that module's docstring - 24000Hz
# hardcoding vs. its own 44100Hz), but still participates in the same
# CACHE_DIR via compute_cache_key directly, with a reference transcript
# folded in through the new `extra` parameter above.

def test_audio8_process_chunk_task_caching_keys_on_transcript(isolated_dirs, tmp_path, monkeypatch):
    """Same reference wav, different transcript (the 'auto-transcript was
    wrong, I fixed it' workflow) must be treated as a cache miss."""
    import numpy as np

    monkeypatch.setattr(audio8_tts, "AUDIO8_REFS_DIR", str(tmp_path / "audio8_refs"))

    wav_path = tmp_path / "ref.wav"
    ref_audio = (0.1 * np.sin(2 * np.pi * 220 * np.arange(1600) / 16000)).astype(np.float32)
    sf.write(str(wav_path), ref_audio, 16000)
    Audio8ReferenceStore.save_reference("Eve", str(wav_path), "Original transcript.")

    engine = Audio8Engine()
    try:
        def _fake_segment(text, ref_wav_path, ref_transcript, speed, lang_code):
            t = np.arange(2200) / 44100
            return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        monkeypatch.setattr(engine, "generate_segment", _fake_segment)

        voice_path = engine.resolve_voice_path("Eve")
        config = {
            "lang_code": "English", "voice": voice_path, "speed": 1.0, "split_pattern": r"\n+",
            "filename": "out", "time_id": "1", "out_dir": str(isolated_dirs.out_dir),
            "format": "wav", "caching": True, "apply_fx": False,
        }
        engine.process_chunk_task((0, "Hello there.", config), None)
        first_cache_files = set(isolated_dirs.cache_dir.glob("*_0.wav"))
        assert len(first_cache_files) == 1

        # Re-save the same name with a different transcript (same wav content).
        Audio8ReferenceStore.save_reference("Eve", str(wav_path), "Corrected transcript!")
        engine.process_chunk_task((0, "Hello there.", config), None)
        second_cache_files = set(isolated_dirs.cache_dir.glob("*_0.wav"))

        assert len(second_cache_files) == 2
        assert first_cache_files < second_cache_files
    finally:
        engine.worker.stop()


def test_audio8_process_chunk_task_caching_keys_on_sampling_knobs(isolated_dirs, tmp_path, monkeypatch):
    """Changing a sampling knob (temperature/top_p/top_k/max_new_tokens)
    changes what the model would generate, so it must be a cache miss too -
    same reasoning as the transcript test above, folded into `extra` the
    same way."""
    import numpy as np

    monkeypatch.setattr(audio8_tts, "AUDIO8_REFS_DIR", str(tmp_path / "audio8_refs"))

    wav_path = tmp_path / "ref.wav"
    ref_audio = (0.1 * np.sin(2 * np.pi * 220 * np.arange(1600) / 16000)).astype(np.float32)
    sf.write(str(wav_path), ref_audio, 16000)
    Audio8ReferenceStore.save_reference("Faye", str(wav_path), "Faye's reference line.")

    engine = Audio8Engine()
    try:
        def _fake_segment(text, ref_wav_path, ref_transcript, speed, lang_code):
            t = np.arange(2200) / 44100
            return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        monkeypatch.setattr(engine, "generate_segment", _fake_segment)

        voice_path = engine.resolve_voice_path("Faye")
        config = {
            "lang_code": "English", "voice": voice_path, "speed": 1.0, "split_pattern": r"\n+",
            "filename": "out", "time_id": "1", "out_dir": str(isolated_dirs.out_dir),
            "format": "wav", "caching": True, "apply_fx": False, "temperature": 0.8,
        }
        engine.process_chunk_task((0, "Hello there.", config), None)
        first_cache_files = set(isolated_dirs.cache_dir.glob("*_0.wav"))
        assert len(first_cache_files) == 1

        config["temperature"] = 1.2  # only the sampling knob changes
        engine.process_chunk_task((0, "Hello there.", config), None)
        second_cache_files = set(isolated_dirs.cache_dir.glob("*_0.wav"))

        assert len(second_cache_files) == 2
        assert first_cache_files < second_cache_files
    finally:
        engine.worker.stop()


def test_audio8_process_chunk_task_caches_every_segment_in_a_multi_segment_chunk(isolated_dirs, tmp_path, monkeypatch):
    """A chunk that splits into more than one segment (split_pattern
    matching within one chunk's text, e.g. two newline-separated lines) must
    cache/read *every* segment - not just the first - on both the write and
    the cache-hit path."""
    import numpy as np

    monkeypatch.setattr(audio8_tts, "AUDIO8_REFS_DIR", str(tmp_path / "audio8_refs"))

    wav_path = tmp_path / "ref.wav"
    ref_audio = (0.1 * np.sin(2 * np.pi * 220 * np.arange(1600) / 16000)).astype(np.float32)
    sf.write(str(wav_path), ref_audio, 16000)
    Audio8ReferenceStore.save_reference("Zoe", str(wav_path), "Zoe's reference line.")

    engine = Audio8Engine()
    try:
        def _fake_segment(text, ref_wav_path, ref_transcript, speed, lang_code):
            t = np.arange(2200) / 44100
            return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        monkeypatch.setattr(engine, "generate_segment", _fake_segment)

        voice_path = engine.resolve_voice_path("Zoe")
        config = {
            "lang_code": "English", "voice": voice_path, "speed": 1.0, "split_pattern": r"\n+",
            "filename": "out", "time_id": "1", "out_dir": str(isolated_dirs.out_dir),
            "format": "wav", "caching": True, "apply_fx": False,
        }
        text = "Segment one.\nSegment two."

        files = engine.process_chunk_task((0, text, config), None)
        assert len(files) == 2
        cache_files = set(isolated_dirs.cache_dir.glob("*.wav"))
        assert len(cache_files) == 2  # _0.wav and _1.wav

        # Cache hit path must also produce both segments, not just the first.
        def _boom(*a, **k):
            raise AssertionError("generate_segment should not be called on a cache hit")
        monkeypatch.setattr(engine, "generate_segment", _boom)

        files_hit = engine.process_chunk_task((0, text, config), None)
        assert len(files_hit) == 2
    finally:
        engine.worker.stop()
