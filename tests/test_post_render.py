"""Tests for kokoro_gui/audio/post.py: read-time post-processing of raw clip
segments, and process_chunk_task's `raw_output` contract that feeds it.
No Qt, no audio device."""
import numpy as np
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
