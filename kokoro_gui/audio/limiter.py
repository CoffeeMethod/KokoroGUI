"""An offline peak limiter for export (plan 14). numpy and scipy, no Qt.

`limit_peaks(samples, rate, ceiling_dbfs)` keeps every sample at or under the
ceiling. Where a peak would pass it, the gain dips to bring that sample down,
starting `window_s` before it and ending `window_s` after, so the change is a
short smooth dip and not a hard clip. The gain is shared by all channels, so
the stereo image holds. The mix is already in memory, so the limiter looks
ahead freely and adds no delay.

pedalboard's `Limiter` isn't used: it adds make-up gain and clips at 0 dBFS
instead of holding a ceiling (pedalboard 0.9.23), which an ACX peak limit of
-3 dBFS can't use.

How: the gain each sample needs is `ceiling / |peak|` (1 when it's under).
A running minimum over `2 * window + 1` samples spreads every dip to its
neighbours, then a moving average of the same width smooths the edges. Every
sample the average covers has a minimum at or under the needed gain of the
peak at its centre, so the average can't be higher than that gain.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d

DEFAULT_WINDOW_S = 0.01
_CHUNK = 1 << 21  # frames per pass, so a book-length mix doesn't need a gain array as long as itself


def limit_peaks(samples, rate: int, ceiling_dbfs: float, window_s: float = DEFAULT_WINDOW_S) -> np.ndarray:
    """`samples` (`(frames,)` or `(frames, channels)`) as float32 with no
    sample above `ceiling_dbfs`. Audio already under it comes back unchanged."""
    return limit_peaks_together(samples, (), rate, ceiling_dbfs, window_s)[0]


def limit_peaks_together(samples, companions, rate: int, ceiling_dbfs: float,
                         window_s: float = DEFAULT_WINDOW_S) -> list:
    """`[limited samples, *limited companions]`. The gain comes from
    `samples` alone, and every companion (same length, any channel count) gets
    that same gain. A stem export uses it so the stems still add up to the
    limited mix."""
    out = np.array(samples, dtype=np.float32)
    others = [np.array(c, dtype=np.float32) for c in companions]
    if out.size == 0:
        return [out] + others
    frames = out.reshape(len(out), -1)
    follow = [c.reshape(len(c), -1) for c in others]
    ceiling = 10.0 ** (float(ceiling_dbfs) / 20.0)
    width = max(1, int(round(float(window_s) * int(rate))))
    size = 2 * width + 1
    n = len(frames)
    for start in range(0, n, _CHUNK):
        end = min(n, start + _CHUNK)
        # 2 * width of context each side: the minimum and the average each reach `width`.
        lo, hi = max(0, start - 2 * width), min(n, end + 2 * width)
        peak = np.abs(frames[lo:hi]).max(axis=1)
        if not (peak > ceiling).any():
            continue
        need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-12)).astype(np.float32)
        gain = uniform_filter1d(minimum_filter1d(need, size, mode="nearest"), size, mode="nearest")
        for target in [frames] + follow:
            target[start:end] *= gain[start - lo:end - lo, None]
    return [out] + others
