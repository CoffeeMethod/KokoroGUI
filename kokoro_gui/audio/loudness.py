"""Loudness measurement and gain (plan 12). numpy and scipy, no Qt.

`measure(samples, rate)` reads a mono `(frames,)` or `(frames, channels)`
float array and returns a `LoudnessReport`:

- `integrated_lufs`: ITU-R BS.1770 gated loudness from `pyloudnorm.Meter`.
  Channels count equally, so a dual-mono stereo file reads 3 LU above the
  same signal exported as mono. `-inf` for silence, or for audio shorter
  than one 0.4 s gating block.
- `true_peak_dbtp`: the largest absolute sample of a 4x polyphase upsample
  (`scipy.signal.resample_poly`), the BS.1770 true-peak estimate.
- `sample_peak_dbfs`: the largest absolute sample as stored.
- `rms_dbfs`: RMS over the whole file, every channel.
- `noise_floor_dbfs`: the 10th percentile of the RMS of consecutive 50 ms
  windows. ACX measures the quietest stretches of a chapter, so the
  quietest tenth of the windows stands in for the room tone between
  phrases. A file shorter than one window uses its own RMS.

Levels are `-inf` for digital silence. `pyloudnorm` is imported inside
`measure`, so the app starts without it (`available()` says whether it's
there).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

# pyloudnorm's gating block; shorter audio has no integrated loudness.
MIN_LUFS_SECONDS = 0.4
TRUE_PEAK_OVERSAMPLE = 4
# Frames resampled at a time, with PAD frames of lead-in and tail so a
# block's edges don't ring into the peak (a book is hours long).
_PEAK_CHUNK = 1 << 20
_PEAK_PAD = 64
NOISE_WINDOW_S = 0.05
NOISE_PERCENTILE = 10.0


@dataclass(frozen=True)
class LoudnessReport:
    integrated_lufs: float
    true_peak_dbtp: float
    sample_peak_dbfs: float
    rms_dbfs: float
    noise_floor_dbfs: float
    duration_s: float

    def to_dict(self) -> dict:
        return asdict(self)


def available() -> bool:
    """Whether `pyloudnorm` imports."""
    try:
        import pyloudnorm  # noqa: F401
    except Exception:  # noqa: BLE001 - a broken install counts as missing
        return False
    return True


def _db(linear: float) -> float:
    return 20.0 * math.log10(linear) if linear > 0.0 else -math.inf


def _as_frames(samples) -> np.ndarray:
    arr = np.asarray(samples)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr.astype(np.float32, copy=False)


def _true_peak(frames: np.ndarray) -> float:
    """Linear true peak of a `(frames, channels)` array."""
    n = len(frames)
    peak = 0.0
    from scipy.signal import resample_poly

    for ch in range(frames.shape[1]):
        column = frames[:, ch]
        for start in range(0, n, _PEAK_CHUNK):
            end = min(n, start + _PEAK_CHUNK)
            lo, hi = max(0, start - _PEAK_PAD), min(n, end + _PEAK_PAD)
            up = resample_poly(column[lo:hi], TRUE_PEAK_OVERSAMPLE, 1)
            body = up[(start - lo) * TRUE_PEAK_OVERSAMPLE:(end - lo) * TRUE_PEAK_OVERSAMPLE]
            if len(body):
                peak = max(peak, float(np.max(np.abs(body))))
    return peak


def _noise_floor(frames: np.ndarray, rate: int) -> float:
    """dBFS of the 10th percentile of 50 ms window RMS values."""
    window = max(1, int(round(NOISE_WINDOW_S * rate)))
    count = len(frames) // window
    if count == 0:
        return _db(float(np.sqrt(np.mean(np.square(frames, dtype=np.float64)))))
    body = frames[:count * window].astype(np.float64)
    power = np.square(body).mean(axis=1).reshape(count, window).mean(axis=1)
    return _db(float(np.sqrt(np.percentile(power, NOISE_PERCENTILE))))


def measure(samples, rate: int) -> LoudnessReport:
    rate = int(rate)
    frames = _as_frames(samples)
    n = len(frames)
    duration = n / float(rate) if rate > 0 else 0.0
    if n == 0:
        return LoudnessReport(-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, 0.0)
    sample_peak = float(np.max(np.abs(frames)))
    rms = float(np.sqrt(np.mean(np.square(frames, dtype=np.float64))))
    if sample_peak == 0.0:
        return LoudnessReport(-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, duration)

    lufs = -math.inf
    if duration >= MIN_LUFS_SECONDS:
        import pyloudnorm

        meter = pyloudnorm.Meter(rate)
        mono = frames.shape[1] == 1
        value = float(meter.integrated_loudness(frames[:, 0] if mono else frames))
        lufs = value if math.isfinite(value) else -math.inf
    return LoudnessReport(
        integrated_lufs=lufs,
        true_peak_dbtp=_db(_true_peak(frames)),
        sample_peak_dbfs=_db(sample_peak),
        rms_dbfs=_db(rms),
        noise_floor_dbfs=_noise_floor(frames, rate),
        duration_s=duration,
    )


def gain_to_target(report: LoudnessReport, target_lufs: float, ceiling_dbtp: float) -> tuple:
    """`(gain_db, limited)`: the gain that brings the integrated loudness to
    `target_lufs`, cut back so the true peak stays at or under
    `ceiling_dbtp`. `limited` is True when the cut kept the file short of
    the target (there is no limiter). Silence, or audio too short to have a
    loudness, gets `(0.0, False)`."""
    if not math.isfinite(report.integrated_lufs):
        return 0.0, False
    gain = float(target_lufs) - report.integrated_lufs
    if math.isfinite(report.true_peak_dbtp):
        room = float(ceiling_dbtp) - report.true_peak_dbtp
        if gain > room:
            return room, True
    return gain, False


def apply_gain(samples, gain_db: float) -> np.ndarray:
    """`samples` scaled by `gain_db`, as float32."""
    return (np.asarray(samples, dtype=np.float32) * np.float32(10.0 ** (float(gain_db) / 20.0))).astype(np.float32)
