"""Block mixing shared by the live `Transport` and the offline exporter.

`LoadedClip` is one clip's (or one segment's) mono float32 samples already
at the mix sample rate, positioned at `start_frame`, with what the mixer
applies on top: a scalar gain (track fader), left/right pan gains, linear
fade-in/out ramps in frames, and an optional volume automation envelope in
absolute frames. `mix_block` sums every clip overlapping
`[frame, frame + frames)` into a `(frames, 2)` stereo block (grill Q21's
plain gain sum) and clips to +-1. Pure numpy, no audio device, so the
arithmetic is testable on its own.

`src` is a view into the memoized render (`kokoro_gui.audio.post`'s cache):
every gain is applied to a copy, never in place.

`load_clip_samples` reads a wav (or anything soundfile can open), applies
the clip's post-processing config (`kokoro_gui.audio.post`), downmixes to
mono and resamples to the target rate once (or only a slice of it, with
`range_s`); `post` memoizes the result by
`(path, mtime, post_key, target_rate, range_s)` so re-loading an unchanged
arrangement is free and an FX change re-renders only the clips it touched.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

CHANNELS = 2


def pan_gains(pan: float) -> tuple:
    """`(left, right)` for `pan` in [-1, 1]: a constant-power law scaled so
    the centre is unity in each channel (`left**2 + right**2 == 2`). Centre
    therefore plays exactly as the mono mixer did, and a mono export (the
    average of the two channels) of a centred clip is unchanged."""
    pan = max(-1.0, min(1.0, float(pan or 0.0)))
    angle = (pan + 1.0) * math.pi / 4.0
    return math.cos(angle) * math.sqrt(2.0), math.sin(angle) * math.sqrt(2.0)


@dataclass
class LoadedClip:
    clip_id: str
    start_frame: int
    samples: np.ndarray  # mono float32
    gain: float = 1.0
    gain_l: float = 1.0
    gain_r: float = 1.0
    fade_in_frames: int = 0
    fade_out_frames: int = 0
    # `(times, gains)` float arrays in absolute frames, or None.
    automation: Optional[tuple] = None

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


def load_clip_samples(path: str, target_rate: int, post_config: dict | None = None,
                      range_s: tuple | None = None) -> np.ndarray:
    """Mono float32 at `target_rate`, post-processed per `post_config`
    (`kokoro_gui.audio.post.render`, which owns the memo). `range_s`
    (`(start_s, end_s)` into the file) loads only that slice. Raises
    whatever soundfile raises for an unreadable path - callers decide
    whether to skip the clip."""
    from kokoro_gui.audio import post

    return post.render(path, post_config, int(target_rate), range_s)


def clear_sample_cache() -> None:
    from kokoro_gui.audio import post

    post.clear_render_cache()


def automation_arrays(points, sample_rate: int) -> Optional[tuple]:
    """`Track.automation`'s `[seconds, gain]` pairs as `(frames, gains)`
    float arrays sorted by time, or None when there are no points."""
    cleaned = []
    for point in points or []:
        try:
            seconds, gain = float(point[0]), float(point[1])
        except (TypeError, ValueError, IndexError):
            continue
        cleaned.append((max(0.0, seconds), max(0.0, min(2.0, gain))))
    if not cleaned:
        return None
    cleaned.sort()
    times = np.array([s * sample_rate for s, _g in cleaned], dtype=np.float64)
    gains = np.array([g for _s, g in cleaned], dtype=np.float64)
    return times, gains


def _envelope(clip: LoadedClip, lo: int, hi: int) -> Optional[np.ndarray]:
    """Per-frame multiplier for absolute frames `[lo, hi)` from the clip's
    fades and automation, or None when neither touches that range."""
    env = None
    n = len(clip.samples)
    rel_lo, rel_hi = lo - clip.start_frame, hi - clip.start_frame
    fi = clip.fade_in_frames
    if fi > 0 and rel_lo < fi:
        env = np.ones(hi - lo, dtype=np.float32)
        a, b = rel_lo, min(rel_hi, fi)
        env[:b - a] = np.arange(a, b, dtype=np.float32) / float(fi)
    fo = clip.fade_out_frames
    if fo > 0 and rel_hi > n - fo:
        if env is None:
            env = np.ones(hi - lo, dtype=np.float32)
        a = max(rel_lo, n - fo)
        ramp = (n - np.arange(a, rel_hi, dtype=np.float32)) / float(fo)
        env[a - rel_lo:] *= ramp
    if clip.automation is not None:
        times, gains = clip.automation
        auto = np.interp(np.arange(lo, hi, dtype=np.float64), times, gains).astype(np.float32)
        env = auto if env is None else env * auto
    return env


def mix_block(clips: list, frame: int, frames: int, out: np.ndarray | None = None) -> np.ndarray:
    """Sum of every `LoadedClip` overlapping `[frame, frame + frames)` as a
    `(frames, 2)` block, clipped to +-1. `out`, if given, is a float32 array
    of that shape that gets zeroed and filled in place."""
    if out is None:
        out = np.zeros((frames, CHANNELS), dtype=np.float32)
    else:
        out[:] = 0.0
    block_end = frame + frames
    for clip in clips:
        if clip.end_frame <= frame or clip.start_frame >= block_end:
            continue
        lo = max(frame, clip.start_frame)
        hi = min(block_end, clip.end_frame)
        src = clip.samples[lo - clip.start_frame:hi - clip.start_frame]
        seg = src * clip.gain
        env = _envelope(clip, lo, hi)
        if env is not None:
            seg = seg * env
        out[lo - frame:hi - frame, 0] += seg * clip.gain_l
        out[lo - frame:hi - frame, 1] += seg * clip.gain_r
    np.clip(out, -1.0, 1.0, out=out)
    return out


def total_frames(clips: list) -> int:
    return max((c.end_frame for c in clips), default=0)
