"""Block mixing shared by the live `Transport` and the offline exporter.

`LoadedClip` is one clip's mono float32 samples already at the mix sample
rate, positioned at `start_frame`. `mix_block` sums every clip overlapping
`[frame, frame + frames)` with a plain gain sum (grill Q21) and clips to
+-1. Pure numpy, no audio device, so the arithmetic is testable on its own.

`load_clip_samples` reads a wav (or anything soundfile can open), downmixes
to mono and resamples to the target rate once; the result is memoized by
`(path, mtime, target_rate)` so re-loading an unchanged arrangement is
free.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

_SAMPLE_CACHE: dict = {}


@dataclass
class LoadedClip:
    clip_id: str
    start_frame: int
    samples: np.ndarray  # mono float32
    gain: float = 1.0

    @property
    def end_frame(self) -> int:
        return self.start_frame + len(self.samples)


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or len(samples) == 0:
        return samples.astype(np.float32, copy=False)
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(source_rate), int(target_rate))
        return resample_poly(samples, target_rate // g, source_rate // g).astype(np.float32)
    except Exception:
        # Linear interpolation fallback - good enough for a playhead
        # preview when scipy isn't importable.
        duration = len(samples) / float(source_rate)
        target_len = max(1, int(round(duration * target_rate)))
        src_x = np.linspace(0.0, duration, num=len(samples), endpoint=False)
        dst_x = np.linspace(0.0, duration, num=target_len, endpoint=False)
        return np.interp(dst_x, src_x, samples).astype(np.float32)


def load_clip_samples(path: str, target_rate: int) -> np.ndarray:
    """Mono float32 at `target_rate`. Raises whatever soundfile raises for
    an unreadable path - callers decide whether to skip the clip."""
    import soundfile as sf

    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None
    key = (os.path.abspath(path), mtime, int(target_rate))
    cached = _SAMPLE_CACHE.get(key)
    if cached is not None:
        return cached
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1).astype(np.float32) if data.shape[1] > 1 else data[:, 0]
    out = resample(mono, int(rate), int(target_rate))
    _SAMPLE_CACHE[key] = out
    return out


def clear_sample_cache() -> None:
    _SAMPLE_CACHE.clear()


def mix_block(clips: list, frame: int, frames: int, out: np.ndarray | None = None) -> np.ndarray:
    """Sum of every `LoadedClip` overlapping `[frame, frame + frames)`,
    clipped to +-1. `out`, if given, is a float32 array of length `frames`
    that gets zeroed and filled in place."""
    if out is None:
        out = np.zeros(frames, dtype=np.float32)
    else:
        out[:] = 0.0
    block_end = frame + frames
    for clip in clips:
        if clip.end_frame <= frame or clip.start_frame >= block_end:
            continue
        lo = max(frame, clip.start_frame)
        hi = min(block_end, clip.end_frame)
        src = clip.samples[lo - clip.start_frame:hi - clip.start_frame]
        out[lo - frame:hi - frame] += src * clip.gain
    np.clip(out, -1.0, 1.0, out=out)
    return out


def total_frames(clips: list) -> int:
    return max((c.end_frame for c in clips), default=0)
