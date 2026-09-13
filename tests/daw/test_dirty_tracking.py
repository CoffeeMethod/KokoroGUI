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


# ---------------------------------------------------------------------------
# segment_key through a key function (Claude/PLAN_tbaw_bundle.md section 2.3)
# ---------------------------------------------------------------------------

import os

import pytest

from kokoro_gui.engine.caching import segment_key


class _FakeBackend:
    """The duck type `segment_key` needs: a voice dir to resolve names in,
    a version string, and optional extra inputs."""

    id = "kokoro"

    def __init__(self, voices_dir, version="1.0", extra=None):
        self.voices_dir = voices_dir
        self.version = version
        self.extra = extra or {}

    def engine_version(self):
        return self.version

    def cache_key_extra(self, config):
        return dict(self.extra)

    def resolve_voice_file(self, name, project_dir=None):
        path = os.path.join(self.voices_dir, f"{os.path.basename(name)}.pt")
        return os.path.abspath(path) if os.path.isfile(path) else None


def _key_fn_for(backend, config):
    return lambda text, clip, engine_version=None: segment_key(text, config, backend, engine_version)


def _clip_with_file(tmp_path, text, key, version=None, name="seg.wav"):
    path = tmp_path / name
    path.write_bytes(b"RIFF")
    return Clip(segments=[Segment(order_index=0, text=text, cache_key=key, audio_path=str(path),
                                  engine_version=version)])


def test_segment_key_for_a_custom_voice_does_not_change_when_the_voice_dir_moves(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "Mix.pt").write_bytes(b"tensor")
    (dir_b / "Mix.pt").write_bytes(b"tensor")
    config = _config(voice="Mix")

    key_a = segment_key("hello", config, _FakeBackend(str(dir_a)))
    key_b = segment_key("hello", config, _FakeBackend(str(dir_b)))
    key_path = segment_key("hello", _config(voice=str(dir_b / "Mix.pt")), _FakeBackend(str(dir_a)))

    assert key_a == key_b == key_path
    (dir_b / "Mix.pt").write_bytes(b"different tensor")
    assert segment_key("hello", config, _FakeBackend(str(dir_b))) != key_a


def test_segment_key_requires_lang_code(tmp_path):
    with pytest.raises(KeyError):
        segment_key("hello", {"voice": "af_bella", "speed": 1.0}, _FakeBackend(str(tmp_path)))


def test_segment_key_extra_inputs_dirty_the_clip(tmp_path):
    """An Audio8 transcript edit changes `cache_key_extra`, which the key
    function folds in - the clip is dirty even though config is unchanged."""
    config = _config(voice="af_bella")
    before = _FakeBackend(str(tmp_path), extra={"ref_transcript": "Hi there."})
    after = _FakeBackend(str(tmp_path), extra={"ref_transcript": "Hi there!"})
    key_before = segment_key("hello", config, before)
    clip = _clip_with_file(tmp_path, "hello", key_before)

    assert is_clip_dirty(clip, "hello", config, key_fn=_key_fn_for(before, config)) is False
    assert is_clip_dirty(clip, "hello", config, key_fn=_key_fn_for(after, config)) is True


def test_take_enters_the_key_only_when_non_zero(tmp_path):
    backend = _FakeBackend(str(tmp_path))
    base = segment_key("hello", _config(), backend)
    assert segment_key("hello", _config(take=0), backend) == base
    assert segment_key("hello", _config(take=1), backend) != base


def test_stored_engine_version_wins_while_the_file_is_present(tmp_path):
    """TB9: a bundle generated with another model version opens clean."""
    config = _config()
    old = _FakeBackend(str(tmp_path), version="0.9.4")
    installed = _FakeBackend(str(tmp_path), version="0.7.11")
    key_old = segment_key("hello", config, old)
    key_fn = _key_fn_for(installed, config)

    clip = _clip_with_file(tmp_path, "hello", key_old, version="0.9.4")
    assert is_clip_dirty(clip, "hello", config, key_fn=key_fn) is False

    os.remove(clip.segments[0].audio_path)
    assert is_clip_dirty(clip, "hello", config, key_fn=key_fn) is True


def test_segment_whose_file_was_deleted_is_dirty(tmp_path):
    """TB11: close-time GC or an undo across it can't leave a clip silent."""
    config = _config()
    backend = _FakeBackend(str(tmp_path))
    key_fn = _key_fn_for(backend, config)
    clip = _clip_with_file(tmp_path, "hello", segment_key("hello", config, backend))
    assert is_clip_dirty(clip, "hello", config, key_fn=key_fn) is False
    os.remove(clip.segments[0].audio_path)
    assert is_clip_dirty(clip, "hello", config, key_fn=key_fn) is True


def test_segment_without_audio_path_is_not_treated_as_missing():
    """A hand-built segment (no file at all) still compares by key only."""
    config = _config()
    clip = _generated_clip("hello world", config)
    assert is_clip_dirty(clip, "hello world", config) is False


def test_build_segments_from_results_stamps_key_take_and_version_from_the_engine():
    from kokoro_gui.daw.dirty import build_segments_from_results, take_from_results

    results = [
        {"path": "a.wav", "text": "hello", "duration": 1.0, "cache_key": "k1", "take": 2, "engine_version": "v"},
        {"path": "b.wav", "text": "world", "duration": 1.0, "cache_key": "k1", "take": 2, "engine_version": "v"},
    ]
    segments = build_segments_from_results("fallback", results)
    assert [s.cache_key for s in segments] == ["k1", "k1"]
    assert [s.engine_version for s in segments] == ["v", "v"]
    assert take_from_results(results) == 2
    assert build_segments_from_results("fallback", [{"path": "a.wav", "text": "x", "duration": 1.0}])[0].cache_key == "fallback"
