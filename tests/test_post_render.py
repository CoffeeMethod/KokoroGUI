"""Tests for kokoro_gui/audio/post.py: read-time post-processing of raw clip
segments, and process_chunk_task's `raw_output` contract that feeds it.
No Qt, no audio device."""
import numpy as np
import pytest
import soundfile as sf

from kokoro_gui.audio import post
from kokoro_gui.audio.mixer import load_clip_samples


def _tone(path, seconds=0.5, rate=8000, amplitude=0.25):
    t = np.arange(int(rate * seconds)) / rate
    sf.write(str(path), (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32), rate)
    return str(path)


# -- post_key ---------------------------------------------------------------

def test_post_key_ignores_generation_keys_and_key_order():
    a = {"voice": "af_bella", "speed": 1.0, "volume": 1.5, "reverb_enabled": True}
    b = {"reverb_enabled": True, "volume": 1.5, "voice": "af_sarah", "speed": 0.7, "out_dir": "x"}
    assert post.post_key(a) == post.post_key(b)


def test_post_key_changes_when_any_post_key_changes():
    base = {"volume": 1.0, "reverb_enabled": False, "eq_bass": 0.0}
    for key, value in (("volume", 0.5), ("reverb_enabled", True), ("eq_bass", 3.0),
                       ("normalize", True), ("trim_silence", True), ("pitch", 2.0), ("apply_fx", False)):
        changed = dict(base, **{key: value})
        assert post.post_key(changed) != post.post_key(base), key


def test_extract_post_config_keeps_only_post_keys():
    cfg = {"voice": "v", "volume": 2.0, "apply_fx": True, "reverb_wet_level": 0.4, "raw_output": True}
    assert post.extract_post_config(cfg) == {"volume": 2.0, "apply_fx": True, "reverb_wet_level": 0.4}


# -- render -----------------------------------------------------------------

def test_render_with_no_config_returns_the_file_as_is(tmp_path):
    path = _tone(tmp_path / "a.wav")
    raw, _rate = sf.read(path, dtype="float32")
    assert np.array_equal(post.render(path, None, 8000), raw)
    assert np.array_equal(post.render(path, {"volume": 1.0, "apply_fx": False}, 8000), raw)


def test_render_applies_volume_and_memoizes_per_post_key(tmp_path):
    path = _tone(tmp_path / "a.wav")
    raw, _rate = sf.read(path, dtype="float32")

    loud = post.render(path, {"volume": 2.0, "apply_fx": False}, 8000)
    assert np.allclose(loud, raw * 2.0, atol=1e-6)
    # Same key -> the cached array object, not a re-render.
    assert post.render(path, {"volume": 2.0, "apply_fx": False, "voice": "ignored"}, 8000) is loud
    # Different post key -> a different render.
    quiet = post.render(path, {"volume": 0.5, "apply_fx": False}, 8000)
    assert np.allclose(quiet, raw * 0.5, atol=1e-6)


def test_render_applies_the_fx_chain_when_enabled(tmp_path):
    path = _tone(tmp_path / "a.wav")
    raw, _rate = sf.read(path, dtype="float32")
    wet = post.render(path, {"apply_fx": True, "gain_enabled": True, "gain_db": 6.0}, 8000)
    assert len(wet) == len(raw)
    assert np.max(np.abs(wet)) > np.max(np.abs(raw)) * 1.5


def test_trim_changes_rendered_duration(tmp_path):
    rate = 8000
    silence = np.zeros(rate // 2, dtype=np.float32)
    tone = np.full(rate // 2, 0.3, dtype=np.float32)
    path = tmp_path / "padded.wav"
    sf.write(str(path), np.concatenate([silence, tone, silence]), rate)

    assert post.rendered_duration_s(str(path), None, rate) == 1.5
    assert post.rendered_duration_s(str(path), {"trim_silence": True}, rate) == 0.5


def test_mixer_load_clip_samples_goes_through_post(tmp_path):
    path = _tone(tmp_path / "a.wav")
    raw, _rate = sf.read(path, dtype="float32")
    assert np.allclose(load_clip_samples(path, 8000, {"volume": 0.5, "apply_fx": False}), raw * 0.5, atol=1e-6)
    assert np.array_equal(load_clip_samples(path, 8000), raw)


# -- raw_output in process_chunk_task --------------------------------------

def test_process_chunk_task_raw_output_skips_post_processing(engine, fake_pipeline, make_config, tmp_path):
    config = make_config(out_dir=str(tmp_path), volume=0.1, normalize=True)
    baked = engine.process_chunk_task((0, "hello world", config), None)
    raw = engine.process_chunk_task((1, "hello world", dict(config, raw_output=True)), None)

    assert baked and raw
    assert baked[0]["raw"] is False and raw[0]["raw"] is True
    baked_audio, _ = sf.read(baked[0]["path"], dtype="float32")
    raw_audio, _ = sf.read(raw[0]["path"], dtype="float32")
    # Baked: volume 0.1 then normalize to 0.98 peak. Raw: the pipeline's output untouched.
    assert np.isclose(np.max(np.abs(baked_audio)), 0.98, atol=1e-3)
    assert not np.isclose(np.max(np.abs(raw_audio)), 0.98, atol=1e-3)


def test_generate_clip_audio_marks_segments_raw(engine, fake_pipeline, make_config, tmp_path):
    import asyncio

    config = make_config(out_dir=str(tmp_path), volume=0.1)
    results = asyncio.run(engine.generate_clip_audio((0, "hello world", config)))
    assert results and all(r["raw"] is True for r in results)
    # The caller's dict is untouched (generate_clip_audio copies it).
    assert "raw_output" not in config


# -- duration hint (phase 2, C1) -------------------------------------------------


def test_duration_hint_accounts_for_trim_and_pitch():
    from kokoro_gui.audio.post import duration_hint
    from kokoro_gui.daw.models import Segment

    segment = Segment(duration=2.0, onset_s=0.1, tail_s=0.3)
    assert duration_hint(segment, {}) == 2.0
    assert duration_hint(segment, {"trim_silence": True}) == pytest.approx(1.6)
    assert duration_hint(segment, {"trim_silence": True, "pitch": 12}) == pytest.approx(0.8)
    assert duration_hint(Segment(duration=2.0), {}) is None


def test_rendered_duration_uses_the_hint_without_reading_the_file(tmp_path, monkeypatch):
    from kokoro_gui.audio import post

    post.clear_render_cache()

    def no_reads(_path, _range_s=None):
        raise AssertionError("read the file")

    monkeypatch.setattr(post, "_read_mono", no_reads)
    assert post.rendered_duration_s(str(tmp_path / "never.wav"), {}, 24000, hint=1.25) == 1.25


def test_rendered_duration_prefers_a_memoized_render_over_the_hint(tmp_path):
    import soundfile as sf

    from kokoro_gui.audio import post

    path = str(tmp_path / "a.wav")
    sf.write(path, np.full(8000, 0.5, dtype=np.float32), 8000)
    post.clear_render_cache()
    post.render(path, {}, 8000)
    assert post.rendered_duration_s(path, {}, 8000, hint=9.0) == 1.0


# -- slices (Segment.range, render_slice) -------------------------------------------


def _ramp(path, frames=8000, rate=8000):
    """Sample i holds i / frames, so a slice's first value names its frame."""
    sf.write(str(path), (np.arange(frames) / frames).astype(np.float32), rate, subtype="FLOAT")
    return str(path)


def test_render_with_a_range_reads_only_those_frames(tmp_path):
    path = _ramp(tmp_path / "ramp.wav")
    raw, _rate = sf.read(path, dtype="float32")

    part = post.render(path, None, 8000, range_s=(0.25, 0.5))
    assert np.array_equal(part, raw[2000:4000])
    assert np.array_equal(post.render_slice(path, 0.25, 0.5, None, 8000), part)
    assert np.array_equal(load_clip_samples(path, 8000, None, range_s=[0.25, 0.5]), part)


def test_a_slice_never_decodes_the_whole_file(tmp_path, monkeypatch):
    path = _ramp(tmp_path / "ramp.wav")

    def whole_file_read(*_args, **_kwargs):
        raise AssertionError("read the whole file")

    monkeypatch.setattr(sf, "read", whole_file_read)
    assert len(post.render(path, None, 8000, range_s=(0.0, 0.1))) == 800


def test_render_memo_is_keyed_by_range(tmp_path):
    path = _ramp(tmp_path / "ramp.wav")
    whole = post.render(path, None, 8000)
    first = post.render(path, None, 8000, range_s=(0.0, 0.25))
    second = post.render(path, None, 8000, range_s=(0.25, 0.5))

    assert len(whole) == 8000 and len(first) == len(second) == 2000
    assert not np.array_equal(first, second)
    assert post.render(path, None, 8000, range_s=[0.0, 0.25]) is first
    assert post.render(path, None, 8000) is whole


def test_a_range_is_clamped_to_the_file_and_an_empty_one_is_empty(tmp_path):
    path = _ramp(tmp_path / "ramp.wav")  # 1 s
    raw, _rate = sf.read(path, dtype="float32")

    assert np.array_equal(post.render(path, None, 8000, range_s=(0.75, 3.0)), raw[6000:])
    assert np.array_equal(post.render(path, None, 8000, range_s=(-1.0, 0.1)), raw[:800])
    for empty in ((0.5, 0.5), (0.6, 0.4), (2.0, 3.0)):
        out = post.render(path, {"volume": 2.0, "apply_fx": False}, 8000, range_s=empty)
        assert out.dtype == np.float32 and len(out) == 0, empty


def test_a_slice_is_post_processed_and_resampled(tmp_path):
    path = _ramp(tmp_path / "ramp.wav")
    raw, _rate = sf.read(path, dtype="float32")

    loud = post.render(path, {"volume": 2.0, "apply_fx": False}, 8000, range_s=(0.25, 0.5))
    assert np.allclose(loud, raw[2000:4000] * 2.0, atol=1e-6)
    assert len(post.render(path, None, 16000, range_s=(0.25, 0.5))) == 4000


def test_trim_silence_does_not_apply_inside_a_slice(tmp_path):
    rate = 8000
    silence = np.zeros(rate // 2, dtype=np.float32)
    tone = np.full(rate // 2, 0.3, dtype=np.float32)
    path = tmp_path / "padded.wav"
    sf.write(str(path), np.concatenate([silence, tone, silence]), rate)

    # The range says where the audio starts; trim would move it.
    sliced = post.render(str(path), {"trim_silence": True}, rate, range_s=(0.25, 1.25))
    assert len(sliced) == rate
    assert post.rendered_duration_s(str(path), {"trim_silence": True}, rate, range_s=(0.25, 1.25)) == 1.0


def test_duration_hint_with_a_range_is_its_length_over_pitch():
    from kokoro_gui.daw.models import Segment

    segment = Segment(duration=9.0, range=[1.0, 3.0])
    assert post.duration_hint(segment, {}) == 2.0
    # Trim leaves a slice alone, so the hint does too.
    assert post.duration_hint(segment, {"trim_silence": True}) == 2.0
    assert post.duration_hint(segment, {"pitch": 12}) == pytest.approx(1.0)
    assert post.duration_hint(Segment(range=[3.0, 1.0]), {}) == 0.0


def test_segment_range_rejects_malformed_values():
    from kokoro_gui.daw.models import Segment

    assert post.segment_range(Segment(range=[1, 2.5])) == (1.0, 2.5)
    assert post.segment_range(Segment()) is None
    for bad in ([1.0], ["a", "b"], "12", [float("nan"), 1.0]):
        assert post.segment_range(Segment(range=bad)) is None, bad


def test_rendered_duration_of_a_range_reads_only_the_slice(tmp_path):
    path = _ramp(tmp_path / "ramp.wav")
    post.clear_render_cache()
    assert post.rendered_duration_s(path, None, 8000, range_s=(0.5, 0.75)) == 0.25
    assert post.rendered_duration_s(path, None, 8000) == 1.0


# -- time stretch (phase 5, D4) -----------------------------------------------------


def test_time_stretch_is_a_post_key():
    assert "time_stretch" in post.POST_KEYS
    assert post.post_key({"time_stretch": 1.1}) != post.post_key({})
    assert post.extract_post_config({"time_stretch": 1.1, "speed": 1.2}) == {"time_stretch": 1.1}


def test_time_stretch_divides_the_rendered_length_and_keeps_1_as_identity(tmp_path):
    path = _tone(tmp_path / "a.wav", seconds=1.0)
    raw, _rate = sf.read(path, dtype="float32")
    post.clear_render_cache()

    assert post.is_identity({"time_stretch": 1.0, "apply_fx": False})
    assert not post.is_identity({"time_stretch": 1.1, "apply_fx": False})
    assert np.array_equal(post.render(path, {"time_stretch": 1.0, "apply_fx": False}, 8000), raw)

    faster = post.render(path, {"time_stretch": 1.25, "apply_fx": False}, 8000)
    slower = post.render(path, {"time_stretch": 0.8, "apply_fx": False}, 8000)
    assert faster.ndim == 1 and faster.dtype == np.float32
    assert len(faster) == pytest.approx(len(raw) / 1.25, abs=2)
    assert len(slower) == pytest.approx(len(raw) / 0.8, abs=2)


def test_time_stretch_is_clamped_and_tolerates_junk(tmp_path):
    from kokoro_gui.engine.audio_fx import TIME_STRETCH_MAX, clamp_time_stretch

    assert clamp_time_stretch(None) == 1.0
    assert clamp_time_stretch("fast") == 1.0
    assert clamp_time_stretch(float("nan")) == 1.0
    assert clamp_time_stretch(1000) == TIME_STRETCH_MAX
    path = _tone(tmp_path / "a.wav", seconds=1.0)
    post.clear_render_cache()
    assert len(post.render(path, {"time_stretch": "fast", "apply_fx": False}, 8000)) == 8000


def test_duration_hint_divides_by_the_stretch_after_trim_and_pitch(tmp_path):
    from kokoro_gui.daw.models import Segment

    segment = Segment(duration=2.0, onset_s=0.1, tail_s=0.3)
    assert post.duration_hint(segment, {"time_stretch": 1.25}) == pytest.approx(1.6)
    assert post.duration_hint(segment, {"trim_silence": True, "pitch": 12, "time_stretch": 0.8}) \
        == pytest.approx(1.0)
    ranged = Segment(duration=5.0, range=[1.0, 3.0])
    assert post.duration_hint(ranged, {"time_stretch": 2.0}) == pytest.approx(1.0)
    # The hint agrees with what a render measures.
    path = _tone(tmp_path / "b.wav", seconds=1.0)
    post.clear_render_cache()
    measured = post.rendered_duration_s(path, {"time_stretch": 1.1, "apply_fx": False}, 8000)
    assert measured == pytest.approx(post.duration_hint(Segment(duration=1.0, onset_s=0.0, tail_s=0.0),
                                                        {"time_stretch": 1.1}), abs=1e-3)
