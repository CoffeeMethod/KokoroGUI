"""Tests for kokoro_gui/qt/waveform_data.py's min/max peak decimation - pure
NumPy, no Qt import at all, mirroring tests/test_caching.py's convention of
testing this logic with zero GUI dependency."""
import numpy as np
import pytest
import soundfile as sf

from kokoro_gui.qt.waveform_data import compute_peaks, load_peaks_from_file


def test_mono_basic_peaks():
    # 8 samples split into 4 buckets of 2 each.
    samples = np.array([0.0, 1.0, -1.0, 0.5, 0.2, 0.2, -0.9, 0.9], dtype=np.float32)
    peaks = compute_peaks(samples, sample_rate=8, bucket_count=4)
    assert peaks.shape == (4, 2)
    np.testing.assert_allclose(peaks[0], [0.0, 1.0])
    np.testing.assert_allclose(peaks[1], [-1.0, 0.5])
    np.testing.assert_allclose(peaks[2], [0.2, 0.2])
    np.testing.assert_allclose(peaks[3], [-0.9, 0.9])


def test_stereo_downmix_averages_channels():
    # Left channel all 1.0, right channel all -1.0 -> average is 0.0 everywhere.
    left = np.ones(4, dtype=np.float32)
    right = -np.ones(4, dtype=np.float32)
    stereo = np.stack([left, right], axis=1)
    peaks = compute_peaks(stereo, sample_rate=4, bucket_count=1)
    np.testing.assert_allclose(peaks[0], [0.0, 0.0])


def test_bucket_count_exceeds_sample_count():
    samples = np.array([0.3, -0.5, 0.8], dtype=np.float32)
    peaks = compute_peaks(samples, sample_rate=3, bucket_count=10)
    assert peaks.shape == (10, 2)
    for i, value in enumerate(samples):
        np.testing.assert_allclose(peaks[i], [value, value])
    for i in range(len(samples), 10):
        np.testing.assert_allclose(peaks[i], [0.0, 0.0])


def test_all_silence_produces_all_zero_peaks():
    samples = np.zeros(1000, dtype=np.float32)
    peaks = compute_peaks(samples, sample_rate=1000, bucket_count=50)
    assert peaks.shape == (50, 2)
    assert np.all(peaks == 0.0)


def test_empty_samples_returns_zero_array():
    samples = np.zeros(0, dtype=np.float32)
    peaks = compute_peaks(samples, sample_rate=1000, bucket_count=20)
    assert peaks.shape == (20, 2)
    assert np.all(peaks == 0.0)


def test_single_bucket_covers_whole_buffer():
    samples = np.array([0.1, -0.7, 0.9, -0.2, 0.05], dtype=np.float32)
    peaks = compute_peaks(samples, sample_rate=5, bucket_count=1)
    np.testing.assert_allclose(peaks[0], [samples.min(), samples.max()])


def test_load_peaks_from_file_reads_wav_and_duration(tmp_path):
    sample_rate = 8000
    duration_seconds = 0.5
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "tone.wav"
    sf.write(str(path), data, sample_rate)

    peaks, duration = load_peaks_from_file(str(path), bucket_count=64)

    assert peaks.shape == (64, 2)
    assert duration == pytest.approx(len(data) / sample_rate, rel=1e-6)
