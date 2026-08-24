"""Tests for kokoro_gui/daw/dirty.py - verifies it stays a thin, faithful
read of kokoro_gui/engine/caching.py's own cache-hash/segment-prediction
logic rather than a second, divergent implementation (same spirit as
tests/test_meta_caching_policy.py's policing of tests/test_caching.py)."""
from kokoro_gui.daw.dirty import (
    compute_expected_cache_hash,
    is_clip_dirty,
    predict_segment_texts,
)
from kokoro_gui.daw.models import Clip, Segment
from kokoro_gui.engine.caching import compute_cache_key


def _config(**overrides):
    base = {"voice": "af_bella", "speed": 1.0, "lang_code": "a", "engine_id": "kokoro"}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# predict_segment_texts
# ---------------------------------------------------------------------------

def test_predict_segment_texts_splits_on_default_pattern_and_strips_blank_lines():
    text = "First line.\n\nSecond line.\n   \nThird."
    assert predict_segment_texts(text, _config()) == ["First line.", "Second line.", "Third."]


def test_predict_segment_texts_honors_custom_split_pattern():
    text = "a|b|c"
    assert predict_segment_texts(text, _config(split_pattern=r"\|")) == ["a", "b", "c"]


def test_predict_segment_texts_empty_text_is_empty_list():
    assert predict_segment_texts("   \n\n  ", _config()) == []


# ---------------------------------------------------------------------------
# compute_expected_cache_hash
# ---------------------------------------------------------------------------

def test_compute_expected_cache_hash_matches_compute_cache_key_directly():
    config = _config()
    expected = compute_cache_key("hello", "af_bella", 1.0, "a", "kokoro")
    assert compute_expected_cache_hash("hello", config) == expected


def test_compute_expected_cache_hash_applies_pitch_speed_compensation():
    config = _config(pitch=12.0)  # +12 semitones halves the effective speed
    expected = compute_cache_key("hello", "af_bella", 0.5, "a", "kokoro")
    assert compute_expected_cache_hash("hello", config) == expected


def test_compute_expected_cache_hash_changes_with_voice():
    config_a = _config(voice="af_bella")
    config_b = _config(voice="af_sarah")
    assert compute_expected_cache_hash("hello", config_a) != compute_expected_cache_hash("hello", config_b)


def test_compute_expected_cache_hash_ignores_out_dir_and_format():
    # Mirrors compute_cache_key's own contract: only text/voice/speed/lang
    # affect the hash - out_dir/format/etc. apply after cache read/generation.
    config_a = _config(out_dir="/a", format="wav")
    config_b = _config(out_dir="/b", format="flac")
    assert compute_expected_cache_hash("hello", config_a) == compute_expected_cache_hash("hello", config_b)


# ---------------------------------------------------------------------------
# is_clip_dirty
# ---------------------------------------------------------------------------

def _generated_clip(text, config):
    """Builds a Clip whose segments look like they were generated from
    `text`/`config` right now - i.e. a clean, non-dirty clip."""
    cache_hash = compute_expected_cache_hash(text, config)
    segments = [
        Segment(order_index=i, text=seg_text, cache_key=cache_hash)
        for i, seg_text in enumerate(predict_segment_texts(text, config))
    ]
    return Clip(segments=segments)


def test_never_generated_clip_is_dirty():
    clip = Clip(segments=[])
    assert is_clip_dirty(clip, "hello", _config()) is True


def test_freshly_generated_clip_is_not_dirty():
    config = _config()
    clip = _generated_clip("hello world", config)
    assert is_clip_dirty(clip, "hello world", config) is False


def test_editing_text_marks_clip_dirty():
    config = _config()
    clip = _generated_clip("hello world", config)
    assert is_clip_dirty(clip, "hello galaxy", config) is True


def test_changing_voice_marks_clip_dirty():
    config = _config()
    clip = _generated_clip("hello world", config)
    assert is_clip_dirty(clip, "hello world", _config(voice="af_sarah")) is True


def test_changing_segment_count_marks_clip_dirty():
    config = _config()
    clip = _generated_clip("only one line", config)
    assert is_clip_dirty(clip, "line one\n\nline two", config) is True
