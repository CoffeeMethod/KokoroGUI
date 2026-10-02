"""Tests for kokoro_gui/audio/loudness.py: measurement, gain to a target
under a peak ceiling. numpy and scipy only; pyloudnorm is the reference
for the integrated value."""
import math

import numpy as np
import pytest

from kokoro_gui.audio import loudness

pyln = pytest.importorskip("pyloudnorm")

RATE = 24000


def _sine(seconds=3.0, amp=0.5, freq=1000.0, rate=RATE):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_a_sine_measures_what_pyloudnorm_says():
    x = _sine()
    report = loudness.measure(x, RATE)
    assert report.integrated_lufs == pytest.approx(pyln.Meter(RATE).integrated_loudness(x.astype(np.float64)), abs=0.01)
    # A 1 kHz sine is about -0.7 LU off its RMS level (K-weighting is near
    # flat there): amplitude 0.5 is -9.0 dBFS RMS, so about -9.0 LUFS.
    assert report.integrated_lufs == pytest.approx(-9.0, abs=0.5)
    assert report.rms_dbfs == pytest.approx(-9.03, abs=0.05)
    assert report.sample_peak_dbfs == pytest.approx(-6.02, abs=0.01)
    assert report.duration_s == pytest.approx(3.0)


def test_stereo_dual_mono_reads_three_lu_above_mono():
    x = _sine()
    mono = loudness.measure(x, RATE).integrated_lufs
    stereo = loudness.measure(np.stack([x, x], axis=1), RATE).integrated_lufs
    assert stereo - mono == pytest.approx(3.01, abs=0.05)


def test_true_peak_is_at_least_the_sample_peak():
    # A sine whose crests fall between samples: the oversampled peak is higher.
    t = np.arange(RATE) / RATE
    x = np.sin(2 * np.pi * (RATE / 4.0) * t + math.pi / 4)
    x = (0.9 * x * np.hanning(len(x))).astype(np.float32)  # no hard edges to ring
    report = loudness.measure(x, RATE)
    assert report.true_peak_dbtp >= report.sample_peak_dbfs
    # The samples hit 0.9 * sin(45 deg) = 0.64 at best; the waveform between them reaches 0.9.
    assert report.true_peak_dbtp == pytest.approx(20 * math.log10(0.9), abs=0.15)
    assert report.sample_peak_dbfs < report.true_peak_dbtp - 2.0


def test_true_peak_chunking_matches_one_pass(monkeypatch):
    x = _sine(seconds=1.0, amp=0.7, freq=3000.0)
    whole = loudness.measure(x, RATE).true_peak_dbtp
    monkeypatch.setattr(loudness, "_PEAK_CHUNK", 1000)
    assert loudness.measure(x, RATE).true_peak_dbtp == pytest.approx(whole, abs=0.01)


def test_silence_gives_minus_infinity_without_raising():
    report = loudness.measure(np.zeros(RATE * 2, dtype=np.float32), RATE)
    assert report.integrated_lufs == -math.inf
    assert report.true_peak_dbtp == -math.inf
    assert report.noise_floor_dbfs == -math.inf
    assert loudness.measure(np.zeros(0, dtype=np.float32), RATE).duration_s == 0.0


def test_audio_under_one_gating_block_has_no_integrated_loudness():
    report = loudness.measure(_sine(seconds=0.3), RATE)
    assert report.integrated_lufs == -math.inf
    assert math.isfinite(report.true_peak_dbtp)
    assert math.isfinite(report.rms_dbfs)


def test_noise_floor_finds_the_quiet_stretch():
    rng = np.random.default_rng(1)
    noise_rms = 10 ** (-60 / 20)
    quiet = (rng.standard_normal(RATE) * noise_rms).astype(np.float32)  # 1 s of -60 dBFS noise
    x = np.concatenate([_sine(seconds=3.0, amp=0.5), quiet, _sine(seconds=1.0, amp=0.5)])
    report = loudness.measure(x, RATE)
    assert report.noise_floor_dbfs == pytest.approx(-60.0, abs=1.0)


def test_noise_floor_of_a_clip_shorter_than_a_window_is_its_rms():
    x = _sine(seconds=0.02)
    report = loudness.measure(x, RATE)
    assert report.noise_floor_dbfs == pytest.approx(report.rms_dbfs, abs=0.01)


def test_to_dict_has_every_field():
    d = loudness.measure(_sine(), RATE).to_dict()
    assert set(d) == {"integrated_lufs", "true_peak_dbtp", "sample_peak_dbfs", "rms_dbfs",
                      "noise_floor_dbfs", "duration_s"}


def _report(lufs, tp):
    return loudness.LoudnessReport(lufs, tp, tp, -20.0, -60.0, 1.0)


def test_gain_to_target_reaches_the_target_when_the_peak_allows():
    gain, limited = loudness.gain_to_target(_report(-20.0, -10.0), -16.0, -1.0)
    assert gain == pytest.approx(4.0)
    assert not limited


def test_gain_to_target_backs_off_when_the_ceiling_would_be_crossed():
    gain, limited = loudness.gain_to_target(_report(-20.0, -4.0), -16.0, -1.0)
    assert gain == pytest.approx(3.0)  # the peak has 3 dB of room, the target wants 4
    assert limited


def test_gain_to_target_attenuates_a_loud_file_and_ignores_silence():
    assert loudness.gain_to_target(_report(-10.0, -0.5), -16.0, -1.0) == (pytest.approx(-6.0), False)
    assert loudness.gain_to_target(_report(-math.inf, -math.inf), -16.0, -1.0) == (0.0, False)


def test_applied_gain_lands_on_the_target():
    x = _sine(amp=0.1)
    report = loudness.measure(x, RATE)
    gain, limited = loudness.gain_to_target(report, -16.0, -1.0)
    assert not limited
    after = loudness.measure(loudness.apply_gain(x, gain), RATE)
    assert after.integrated_lufs == pytest.approx(-16.0, abs=0.05)
    assert loudness.apply_gain(x, gain).dtype == np.float32
