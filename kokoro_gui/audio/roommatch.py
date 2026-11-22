"""Room matching for a generated correction clip.

A line typed into a recording sounds out of place because the synthesis has no
room around it. This module measures that gap. `eq_match` builds a linear-phase
FIR that pulls the spectrum of a target (the generated clip) toward a reference
(the recording). `room_tone` returns the quietest stretch of the recording, and
`noise_floor_db` gives its level, so the tone can be mixed under the clip.

Pure numpy and scipy, no Qt, no file IO. Nothing calls it yet; the per-clip FX
wiring is plan 33 part B and waits on a design decision.
"""

import numpy as np
from scipy import signal

MAX_GAIN_DB = 12.0
MIN_TAPS = 256
MAX_TAPS = 1024
_SILENCE_RMS = 1e-6
_BAND_OCT = 1.0 / 3.0
_FLOOR_DB = -120.0


def _mono(x):
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 2:
        # (frames, channels) is the app's layout; (channels, frames) works too.
        axis = 1 if a.shape[1] <= a.shape[0] else 0
        a = a.mean(axis=axis)
    return a.reshape(-1)


def _impulse(taps):
    ir = np.zeros(taps, dtype=np.float32)
    ir[taps // 2] = 1.0
    return ir


def _rms(a):
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


def _smooth_third_octave(freqs, mag):
    """Average `mag` over a 1/3-octave window around each frequency, in power."""
    power = mag * mag
    out = np.empty_like(power)
    half = 2.0 ** (_BAND_OCT / 2.0)
    for i, f in enumerate(freqs):
        if f <= 0.0:
            out[i] = power[i]
            continue
        sel = (freqs >= f / half) & (freqs <= f * half)
        out[i] = power[sel].mean()
    return np.sqrt(out)


def eq_match(reference, target, rate, taps=512):
    """Linear-phase FIR that makes `target` sound spectrally like `reference`.

    `taps` is clamped to 256..1024. An even count becomes odd (+1), so the
    filter is type I with a whole-sample delay of (len - 1) // 2 and a valid
    gain at Nyquist. Both signals may be mono or stereo; stereo is averaged to
    mono for analysis. Welch magnitude spectra are divided (reference over
    target), smoothed in 1/3-octave bands and limited to +-12 dB. The gain is
    held flat below 50 Hz and above 0.95 * Nyquist, where estimates are poor.

    Silent or very short input in either signal returns a unit impulse (no
    change), as does an unusable `rate`. Returns float32.
    """
    taps = int(min(max(int(taps), MIN_TAPS), MAX_TAPS))
    if taps % 2 == 0:
        taps += 1
    ref, tgt = _mono(reference), _mono(target)
    rate = float(rate)
    if (
        rate <= 0
        or min(ref.size, tgt.size) < 256
        or _rms(ref) < _SILENCE_RMS
        or _rms(tgt) < _SILENCE_RMS
    ):
        return _impulse(taps)
    f1, p_ref = signal.welch(ref, fs=rate, nperseg=min(2048, ref.size))
    f2, p_tgt = signal.welch(tgt, fs=rate, nperseg=min(2048, tgt.size))
    # Put both on the coarser grid so the ratio is bin for bin.
    grid = f1 if f1.size <= f2.size else f2
    p_ref = np.interp(grid, f1, p_ref)
    p_tgt = np.interp(grid, f2, p_tgt)
    eps = max(float(p_tgt.max()), float(p_ref.max())) * 1e-10 + 1e-30
    ratio = np.sqrt((p_ref + eps) / (p_tgt + eps))
    ratio = _smooth_third_octave(grid, ratio)
    lim = 10.0 ** (MAX_GAIN_DB / 20.0)
    ratio = np.clip(ratio, 1.0 / lim, lim)
    nyq = rate / 2.0
    lo, hi = 50.0, 0.95 * nyq
    inside = (grid >= lo) & (grid <= hi)
    if hi <= lo or not inside.any():
        return _impulse(taps)
    g = ratio.copy()
    g[grid < lo] = ratio[inside][0]
    g[grid > hi] = ratio[inside][-1]
    # firwin2 needs a grid that runs from 0 to Nyquist with no repeats.
    xs = grid.astype(np.float64).copy()
    xs[0], xs[-1] = 0.0, nyq
    keep = np.concatenate(([True], np.diff(xs) > 0))
    fir = signal.firwin2(taps, xs[keep] / nyq, g[keep], window="hamming")
    return fir.astype(np.float32)


def room_tone(recording, rate, length_s=1.0):
    """The quietest `length_s` stretch of `recording`, by RMS.

    RMS is measured over 50 ms hops and a stretch starts on a hop boundary. A
    recording no longer than `length_s` comes back whole. Mono or stereo in,
    the same layout out.
    """
    a = np.asarray(recording, dtype=np.float32)
    n = a.shape[0]
    want = int(round(float(length_s) * float(rate)))
    if want <= 0 or n <= want:
        return a.copy()
    mono = _mono(a)
    hop = max(1, int(0.05 * rate))
    nwin = max(1, want // hop)
    nhops = n // hop
    if nhops <= nwin:
        return a[:want].copy()
    e = np.add.reduceat(mono[: nhops * hop] ** 2, np.arange(0, nhops * hop, hop))
    csum = np.concatenate(([0.0], np.cumsum(e)))
    sums = csum[nwin:] - csum[:-nwin]
    start = int(np.argmin(sums)) * hop
    return a[start : start + want].copy()


def noise_floor_db(recording, rate):
    """Level in dBFS (RMS) of the recording's room tone; -120 for silence."""
    tone = room_tone(recording, rate)
    r = _rms(_mono(tone)) if tone.size else 0.0
    return _FLOOR_DB if r <= 0 else max(_FLOOR_DB, float(20.0 * np.log10(r)))
