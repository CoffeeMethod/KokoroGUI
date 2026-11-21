import numpy as np
from scipy import signal

from kokoro_gui.audio import roommatch

RATE = 24000


def _gain_db(fir, lo, hi):
    w, h = signal.freqz(fir, worN=4096, fs=RATE)
    sel = (w >= lo) & (w <= hi)
    return 20 * np.log10(np.abs(h[sel]).mean())


def _noise(seed, n=RATE * 2):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32) * 0.1


def _flat_within_1db(fir):
    w, h = signal.freqz(fir, worN=2048, fs=RATE)
    sel = (w > 100) & (w < 10000)
    return np.abs(20 * np.log10(np.abs(h[sel]))).max() < 1.0


def test_lowpassed_reference_cuts_highs():
    target = _noise(1)
    b, a = signal.butter(4, 2000, fs=RATE)
    reference = signal.lfilter(b, a, _noise(2)).astype(np.float32)
    fir = roommatch.eq_match(reference, target, RATE)
    assert _gain_db(fir, 6000, 10000) < _gain_db(fir, 100, 800) - 10


def test_identical_inputs_are_flat():
    x = _noise(3)
    assert _flat_within_1db(roommatch.eq_match(x, x, RATE))


def test_gain_is_clamped():
    target = _noise(4)
    b, a = signal.butter(6, 500, fs=RATE)
    reference = signal.lfilter(b, a, _noise(5)).astype(np.float32)
    fir = roommatch.eq_match(reference, target, RATE)
    w, h = signal.freqz(fir, worN=2048, fs=RATE)
    sel = (w > 100) & (w < 11000)
    db = 20 * np.log10(np.abs(h[sel]) + 1e-12)
    assert db.min() > -13.5 and db.max() < 13.5
    assert db.min() < -9


def test_taps_clamped_and_odd():
    x = _noise(6)
    assert len(roommatch.eq_match(x, x, RATE, taps=10)) == 257
    assert len(roommatch.eq_match(x, x, RATE, taps=5000)) == 1025
    assert len(roommatch.eq_match(x, x, RATE, taps=512)) == 513


def test_silent_or_short_input_is_unit_impulse():
    x = _noise(7)
    for ref, tgt in ((np.zeros(RATE), x), (x, np.zeros(RATE)), (x[:50], x)):
        fir = roommatch.eq_match(ref, tgt, RATE)
        assert fir.sum() == 1.0 and np.count_nonzero(fir) == 1


def test_stereo_input_works():
    x = _noise(8)
    st = np.stack([x, x], axis=1)
    assert _flat_within_1db(roommatch.eq_match(st, st, RATE))


def test_room_tone_finds_quiet_stretch():
    rng = np.random.default_rng(9)
    loud = rng.standard_normal(RATE * 2) * 0.3
    quiet = rng.standard_normal(RATE * 2) * 0.001
    sig = np.concatenate([loud, quiet, loud]).astype(np.float32)
    tone = roommatch.room_tone(sig, RATE)
    assert len(tone) == RATE
    assert np.abs(tone).max() < 0.01
    assert -70 < roommatch.noise_floor_db(sig, RATE) < -50


def test_room_tone_short_and_stereo():
    short = _noise(10, n=RATE // 2)
    assert len(roommatch.room_tone(short, RATE)) == len(short)
    st = np.stack([_noise(11, RATE * 3)] * 2, axis=1)
    assert roommatch.room_tone(st, RATE).shape[1] == 2
    assert roommatch.noise_floor_db(np.zeros(RATE * 2, np.float32), RATE) == -120.0
