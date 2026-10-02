"""kokoro_gui/audio/limiter.py: the export peak limiter."""
import numpy as np
import pytest

from kokoro_gui.audio.limiter import limit_peaks, limit_peaks_together


def _db(x):
    return 20 * np.log10(np.max(np.abs(x)))


def _speechlike(rate=8000, seconds=3.0):
    rng = np.random.default_rng(3)
    x = rng.standard_normal((int(rate * seconds), 2)).astype(np.float32) * 0.1
    x[4000] = 0.9  # a single spike well over the ceiling
    x[12000, 1] = -0.8
    return x


def test_no_sample_passes_the_ceiling():
    out = limit_peaks(_speechlike(), 8000, -3.5)
    assert _db(out) <= -3.5 + 0.01


def test_audio_under_the_ceiling_is_returned_unchanged():
    x = _speechlike() * 0.1
    out = limit_peaks(x, 8000, -3.5)
    assert np.array_equal(out, x) and out is not x


def test_only_the_neighbourhood_of_a_peak_changes():
    x = _speechlike()
    out = limit_peaks(x, 8000, -3.5, window_s=0.01)
    changed = np.flatnonzero(np.any(out != x, axis=1))
    assert changed.min() >= 4000 - 160 and changed.max() <= 12000 + 160
    assert np.array_equal(out[1000:3500], x[1000:3500])
    assert np.array_equal(out[6000:11000], x[6000:11000])


def test_the_gain_is_shared_by_the_channels():
    x = _speechlike()
    out = limit_peaks(x, 8000, -3.5)
    ratio_l = out[3990:4010, 0] / x[3990:4010, 0]
    ratio_r = out[3990:4010, 1] / x[3990:4010, 1]
    assert np.allclose(ratio_l, ratio_r, rtol=1e-4)


def test_it_keeps_length_dtype_and_mono_shape():
    mono = _speechlike()[:, 0]
    out = limit_peaks(mono, 8000, -6.0)
    assert out.shape == mono.shape and out.dtype == np.float32
    assert _db(out) <= -6.0 + 0.01


def test_chunk_edges_are_seamless(monkeypatch):
    from kokoro_gui.audio import limiter

    x = _speechlike()
    whole = limit_peaks(x, 8000, -3.5)
    monkeypatch.setattr(limiter, "_CHUNK", 1000)
    assert np.allclose(limit_peaks(x, 8000, -3.5), whole, atol=1e-6)


def test_empty_input_is_fine():
    assert limit_peaks(np.zeros((0, 2), dtype=np.float32), 8000, -3.0).shape == (0, 2)


def test_companions_get_the_gain_of_the_main_signal():
    """Two halves that add up to the main signal still add up to it after the limiter."""
    x = _speechlike()
    part = x * np.float32(0.4)
    limited, limited_part = limit_peaks_together(x, [part], 8000, -3.5)
    assert np.array_equal(limited, limit_peaks(x, 8000, -3.5))
    assert np.allclose(limited_part, limited * 0.4, atol=1e-6)
    assert np.array_equal(part, x * np.float32(0.4))  # the input isn't touched


def test_companions_of_silence_come_back_as_they_were():
    silent = np.zeros((100, 2), dtype=np.float32)
    other = np.ones((100, 2), dtype=np.float32) * 0.5
    out = limit_peaks_together(silent, [other], 8000, -3.5)
    assert len(out) == 2 and np.array_equal(out[1], other)
