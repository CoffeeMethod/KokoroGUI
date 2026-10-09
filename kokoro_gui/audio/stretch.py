"""Pitch-preserving time stretch for the live `Transport` (plan 27, grill PG7).

`Stretcher` is a streaming WSOLA: waveform-similarity overlap-add. It reads
the source through a `pull(n)` callback that returns the next `n` source
frames, cuts overlapping grains out of them, and adds the grains into the
output at a fixed hop. The grains are spaced `rate` times further apart in
the source than in the output, so the audio runs `rate` times as fast and
every voice keeps its pitch. Each grain starts where it best continues the
last one: within +-`TOLERANCE_S` of its nominal place, the offset whose first
half has the highest normalized cross-correlation with what would have
followed the previous grain (the search runs on the channel sum, the grain
is cut from both channels).

Pure numpy, no Qt, no audio device, so it is testable offline
(`stretch_array`). It is a plain Python object with state between calls, and
the transport drives it from the PortAudio callback: `process` allocates a
few small arrays per hop (the same order as `mixer.mix_block`) and reads
the source only through `pull`, which keeps the mixer's duck state in step
with the source frames actually pulled.

The 30 ms Hann grain at 50% overlap sums to exactly 1 (periodic window), so
a constant signal keeps its level. The first grain starts at the first
source frame with no fade, so a stretch begun by a seek is audible at once.
Nothing here guesses the rate: `reset(rate)` sets it, and it may be 0.5 to
2.0 (the transport clamps; `RATE_MIN` and `RATE_MAX` say so).

`played * rate` is the source distance, in frames, that the output handed
out so far stands for. The transport adds it to the frame the stretch began
at, which is what `positionChanged` reports.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

RATE_MIN = 0.5
RATE_MAX = 2.0
GRAIN_S = 0.030
TOLERANCE_S = 0.010
# The least the stretcher asks `pull` for at once. A bigger pull costs one
# mixer call instead of several; the source runs a little ahead of the output.
PULL_FRAMES = 2048


class Stretcher:
    def __init__(self, sample_rate: int, channels: int = 2):
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        grain = int(round(self.sample_rate * GRAIN_S))
        grain += grain % 2
        self.grain = max(grain, 64)
        self.hop = self.grain // 2
        self.tolerance = max(1, int(round(self.sample_rate * TOLERANCE_S)))
        # Periodic Hann: the rising half plus the falling half is exactly 1.
        window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(self.grain) / self.grain)
        self._window = window.astype(np.float32)[:, None]
        self._fall = self._window[self.hop:]
        self.rate = 1.0
        self.reset(1.0)

    def reset(self, rate: float) -> None:
        """Forgets everything and starts a new stream, at `rate`."""
        self.rate = min(RATE_MAX, max(RATE_MIN, float(rate)))
        self._src = np.zeros((0, self.channels), dtype=np.float32)
        self._origin = 0  # stream frame of `_src[0]`
        self._fifo = np.zeros((0, self.channels), dtype=np.float32)
        self._ola = np.zeros((self.grain, self.channels), dtype=np.float32)
        self._template = None
        self._k = 0
        self.played = 0

    # -- the stream -------------------------------------------------------------

    def _ensure(self, upto: int, pull: Callable) -> None:
        """Buffers source up to stream frame `upto` (exclusive)."""
        missing = upto - (self._origin + len(self._src))
        if missing > 0:
            chunk = np.asarray(pull(max(missing, PULL_FRAMES)), dtype=np.float32)
            self._src = np.concatenate((self._src, chunk)) if len(self._src) else chunk

    def _best_start(self, lo: int, hi: int, nominal: int) -> int:
        """The grain start in `[lo, hi]` whose first `hop` frames best
        continue `_template`; `nominal` when the template is silent."""
        hop = self.hop
        template = self._template.sum(axis=1)
        template_energy = float(np.dot(template, template))
        if template_energy < 1e-12:
            return nominal
        seg = self._src[lo - self._origin:hi - self._origin + hop].sum(axis=1)
        correlation = np.correlate(seg, template, "valid")
        squares = np.concatenate(([0.0], np.cumsum(seg.astype(np.float64) ** 2)))
        energy = squares[hop:] - squares[:-hop]
        score = correlation / np.sqrt(energy + 1e-9 * hop)
        return lo + int(np.argmax(score))

    def _next_hop(self, pull: Callable) -> None:
        grain, hop, tol = self.grain, self.hop, self.tolerance
        if self._template is None:
            # Nothing came before the first grain: its "previous" tail is the
            # first source frames themselves, so no fade-in is heard.
            self._ensure(grain, pull)
            self._template = self._src[:hop].copy()
            self._ola[:hop] = self._template * self._fall
            self._ola[hop:] = 0.0
        nominal = int(round(self._k * hop * self.rate))
        lo = max(0, nominal - tol)
        hi = nominal + tol
        if lo > self._origin:
            self._src = self._src[lo - self._origin:]
            self._origin = lo
        self._ensure(hi + grain, pull)
        start = self._best_start(lo, hi, nominal)
        at = start - self._origin
        self._ola += self._src[at:at + grain] * self._window
        self._fifo = np.concatenate((self._fifo, self._ola[:hop]))
        self._ola[:hop] = self._ola[hop:]
        self._ola[hop:] = 0.0
        self._template = self._src[at + hop:at + 2 * hop].copy()
        self._k += 1

    def process(self, frames: int, pull: Callable) -> np.ndarray:
        """The next `frames` output frames as a `(frames, channels)` float32
        array. `pull(n)` returns the next `n` source frames, shape
        `(n, channels)`, contiguous with the last pull."""
        while len(self._fifo) < frames:
            self._next_hop(pull)
        out = self._fifo[:frames].copy()
        self._fifo = self._fifo[frames:]
        self.played += frames
        return out

    @property
    def source_frames(self) -> float:
        """Source distance the output so far stands for, in frames."""
        return self.played * self.rate


def stretch_array(samples: np.ndarray, sample_rate: int, rate: float) -> np.ndarray:
    """`samples` (`(n,)` or `(n, channels)`) played `rate` times as fast, the
    pitch unchanged: `round(n / rate)` frames out. The offline path the tests
    and a measurement use; the transport streams through `Stretcher`."""
    data = np.asarray(samples, dtype=np.float32)
    mono = data.ndim == 1
    if mono:
        data = data[:, None]
    stretcher = Stretcher(sample_rate, data.shape[1])
    stretcher.reset(rate)
    position = [0]

    def pull(count: int) -> np.ndarray:
        chunk = data[position[0]:position[0] + count]
        position[0] += count
        if len(chunk) < count:
            chunk = np.concatenate((chunk, np.zeros((count - len(chunk), data.shape[1]), dtype=np.float32)))
        return chunk

    out = stretcher.process(int(round(len(data) / stretcher.rate)), pull)
    return out[:, 0] if mono else out
