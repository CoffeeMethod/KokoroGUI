"""Tests for KokoroEngine.process_audio - the DSP chain
(kokoro_engine.py:123-262): trim -> volume -> pitch -> Pedalboard FX -> normalize.

Uses synthetic sine waves; assertions are on shape/length/non-silence/
toggle-changes-output, not perceptual audio correctness (out of scope).
"""
import numpy as np
import pytest


def _sine(duration_s=1.0, sr=24000, freq=220.0, amp=0.3):
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def test_process_audio_noop_config_returns_similar_length(engine):
    audio = _sine()
    out = engine.process_audio(audio.copy(), 24000, {})
    assert abs(len(out) - len(audio)) <= 2


def test_trim_silence_removes_leading_trailing_silence(engine):
    sr = 24000
    silence = np.zeros(int(sr * 0.2))
    tone = _sine(duration_s=0.5, amp=0.5)
    audio = np.concatenate([silence, tone, silence])

    trimmed = engine.process_audio(audio.copy(), sr, {"trim_silence": True})
    untrimmed = engine.process_audio(audio.copy(), sr, {"trim_silence": False})

    assert len(trimmed) < len(untrimmed)


def test_volume_scales_amplitude(engine):
    audio = _sine(amp=0.4)
    out = engine.process_audio(audio.copy(), 24000, {"volume": 0.5})
    assert np.allclose(out, audio * 0.5)


def test_pitch_shift_changes_length_via_resample(engine):
    audio = _sine(duration_s=1.0)
    out = engine.process_audio(audio.copy(), 24000, {"pitch": 12.0})
    expected_len = int(len(audio) / 2.0)
    assert abs(len(out) - expected_len) <= 2
    assert len(out) != len(audio)


def test_normalize_peaks_near_unity(engine):
    audio = _sine(amp=0.05)
    out = engine.process_audio(audio.copy(), 24000, {"normalize": True})
    assert np.max(np.abs(out)) > 0.9


def test_eq_bass_treble_shelf_filters_change_output(engine):
    audio = _sine()
    flat = engine.process_audio(audio.copy(), 24000, {})
    shaped = engine.process_audio(audio.copy(), 24000, {"eq_bass": 6.0, "eq_treble": -6.0})
    assert shaped.shape != flat.shape or not np.allclose(shaped, flat)


FX_TOGGLE_CASES = [
    ("reverb_enabled", {}),
    ("comp_enabled", {}),
    ("distortion_enabled", {}),
    ("chorus_enabled", {}),
    ("phaser_enabled", {}),
    ("clipping_enabled", {}),
    ("bitcrush_enabled", {}),
    ("gsm_enabled", {}),
    ("highpass_enabled", {}),
    ("lowpass_enabled", {}),
    ("pitch_shift_enabled", {"pitch_shift_semitones": 5.0}),
    ("delay_enabled", {}),
    ("limiter_enabled", {}),
    ("gain_enabled", {"gain_db": 6.0}),
]


@pytest.mark.parametrize("fx_key,overrides", FX_TOGGLE_CASES, ids=[c[0] for c in FX_TOGGLE_CASES])
def test_fx_toggle_changes_output(engine, fx_key, overrides):
    # Loud enough that level-dependent FX (compressor/limiter/clipping/gain)
    # actually have something to act on.
    audio = _sine(amp=0.9)
    disabled_config = dict(overrides)
    enabled_config = dict(overrides)
    enabled_config[fx_key] = True

    out_disabled = engine.process_audio(audio.copy(), 24000, disabled_config)
    out_enabled = engine.process_audio(audio.copy(), 24000, enabled_config)

    assert out_enabled.shape != out_disabled.shape or not np.allclose(out_enabled, out_disabled)
