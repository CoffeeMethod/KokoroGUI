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

Ducking (phase 5 P2, grill Q30) is a sidechain inside `mix_block`, used
when the caller passes a `DuckState`. Clips with `sidechain` set (speech:
everything but music beds and ducked clips) are summed first; their level
(`max(|left|, |right|)` per frame) drives an envelope follower; clips with
`duck` set are then multiplied by `1 - depth * clamp(env / threshold, 0,
1)`, `depth = 1 - 10 ** (duck_db / 20)`, threshold -30 dBFS. The follower
works on fixed hops of `DUCK_HOP_S`: each hop's peak moves `env` one step
of a one-pole filter (attack 10 ms when the peak is above `env`, release
300 ms below it), and the gain ramps linearly across the next hop to the
value the new `env` gives. The state (`env`, the ramp's ends, the partial
hop) lives on the `DuckState` the caller keeps between blocks, and hops
don't depend on where blocks start or end, so the transport's device-sized
blocks and the exporter's larger ones give the same samples.

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

DEFAULT_DUCK_DB = -12.0
DUCK_THRESHOLD_DB = -30.0
DUCK_ATTACK_S = 0.010
DUCK_RELEASE_S = 0.300
DUCK_HOP_S = 0.005


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
    # Turned down by the sidechain (a clip on a `Track.duck` track).
    duck: bool = False
    # Feeds the sidechain: speech. False for a music bed and a ducked clip.
    sidechain: bool = True

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


class DuckState:
    """The sidechain's envelope between `mix_block` calls: the smoothed
    speech level, the gain ramp across the current hop, and how much of
    that hop is already mixed. The transport keeps one while playing and
    resets it on a seek; the exporter makes a fresh one per mixdown."""

    def __init__(self, sample_rate: int, duck_db: float = DEFAULT_DUCK_DB,
                 threshold_db: float = DUCK_THRESHOLD_DB):
        self.sample_rate = int(sample_rate)
        self.hop = max(1, int(round(self.sample_rate * DUCK_HOP_S)))
        hop_s = self.hop / float(self.sample_rate)
        self.attack = 1.0 - math.exp(-hop_s / DUCK_ATTACK_S)
        self.release = 1.0 - math.exp(-hop_s / DUCK_RELEASE_S)
        self.threshold = 10.0 ** (float(threshold_db) / 20.0)
        self.set_duck_db(duck_db)
        self.reset()

    def set_duck_db(self, duck_db: float) -> None:
        try:
            duck_db = min(0.0, float(duck_db))
        except (TypeError, ValueError):
            duck_db = DEFAULT_DUCK_DB
        self.duck_db = duck_db
        self.depth = 1.0 - 10.0 ** (duck_db / 20.0)

    def reset(self) -> None:
        self.env = 0.0
        self.gain_from = 1.0
        self.gain_to = 1.0
        self.peak = 0.0
        self.fill = 0

    def gain_for(self, env: float) -> float:
        return 1.0 - self.depth * min(max(env / self.threshold, 0.0), 1.0)

    def _end_hop(self, peak: float) -> None:
        coef = self.attack if peak > self.env else self.release
        self.env += coef * (peak - self.env)
        self.gain_from = self.gain_to
        self.gain_to = self.gain_for(self.env)
        self.peak = 0.0
        self.fill = 0

    def _ramp(self, positions: np.ndarray) -> np.ndarray:
        """The current hop's gain at `positions` (fractions of the hop), in
        float32 the same way the whole-hop path computes it, so a hop split
        across two blocks gives the same values as one that isn't."""
        start = np.float32(self.gain_from)
        return start + (np.float32(self.gain_to) - start) * positions

    def gains(self, level: np.ndarray) -> np.ndarray:
        """Per-frame duck gains for a block whose sidechain level is
        `level` (one value per frame), advancing the state past it."""
        n = len(level)
        hop = self.hop
        out = np.empty(n, dtype=np.float32)
        ramp = np.arange(hop, dtype=np.float32) / float(hop)
        # Finish the hop the previous block left open.
        head = min(hop - self.fill, n)
        if head:
            out[:head] = self._ramp(ramp[self.fill:self.fill + head])
            self.peak = max(self.peak, float(level[:head].max()))
            self.fill += head
            if self.fill >= hop:
                self._end_hop(self.peak)
        # Whole hops: the recursion runs per hop on floats, the ramps are
        # built as one array.
        full = (n - head) // hop
        if full:
            peaks = level[head:head + full * hop].reshape(full, hop).max(axis=1)
            starts = np.empty(full, dtype=np.float32)
            ends = np.empty(full, dtype=np.float32)
            for k in range(full):
                starts[k], ends[k] = self.gain_from, self.gain_to
                self._end_hop(float(peaks[k]))
            out[head:head + full * hop] = (starts[:, None] + (ends - starts)[:, None] * ramp[None, :]).reshape(-1)
        # The start of a hop the next block finishes.
        rest = head + full * hop
        if rest < n:
            tail = n - rest
            out[rest:] = self._ramp(ramp[:tail])
            self.peak = float(level[rest:].max())
            self.fill = tail
        return out


def _add_clip(out: np.ndarray, clip: LoadedClip, frame: int, block_end: int,
              scale: Optional[np.ndarray] = None) -> None:
    """Adds `clip`'s part of `[frame, block_end)` into `out`; `scale` is an
    optional per-frame multiplier for the whole block (the duck gain)."""
    lo = max(frame, clip.start_frame)
    hi = min(block_end, clip.end_frame)
    src = clip.samples[lo - clip.start_frame:hi - clip.start_frame]
    seg = src * clip.gain
    env = _envelope(clip, lo, hi)
    if env is not None:
        seg = seg * env
    if scale is not None:
        seg = seg * scale[lo - frame:hi - frame]
    out[lo - frame:hi - frame, 0] += seg * clip.gain_l
    out[lo - frame:hi - frame, 1] += seg * clip.gain_r


def mix_block(clips: list, frame: int, frames: int, out: np.ndarray | None = None,
              duck: Optional[DuckState] = None) -> np.ndarray:
    """Sum of every `LoadedClip` overlapping `[frame, frame + frames)` as a
    `(frames, 2)` block, clipped to +-1. `out`, if given, is a float32 array
    of that shape that gets zeroed and filled in place. With `duck`, the
    sidechain runs (see the module docstring) and `duck` is advanced past
    this block; without it, `LoadedClip.duck` is ignored."""
    if out is None:
        out = np.zeros((frames, CHANNELS), dtype=np.float32)
    else:
        out[:] = 0.0
    block_end = frame + frames
    active = [c for c in clips if c.end_frame > frame and c.start_frame < block_end]
    if duck is None:
        for clip in active:
            _add_clip(out, clip, frame, block_end)
    else:
        for clip in active:
            if clip.sidechain and not clip.duck:
                _add_clip(out, clip, frame, block_end)
        gains = duck.gains(np.abs(out).max(axis=1) if frames else np.zeros(0, dtype=np.float32))
        for clip in active:
            if not clip.sidechain and not clip.duck:
                _add_clip(out, clip, frame, block_end)
        for clip in active:
            if clip.duck:
                _add_clip(out, clip, frame, block_end, gains)
    np.clip(out, -1.0, 1.0, out=out)
    return out


def total_frames(clips: list) -> int:
    return max((c.end_frame for c in clips), default=0)
