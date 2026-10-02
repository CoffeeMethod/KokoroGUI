"""Word timings and silence bounds for `CachingMixin.process_chunk_task`.

Kokoro's `KPipeline` yields `Result` objects that unpack as `(graphemes,
phonemes, audio)` and carry `tokens`, each with `text`, `start_ts` and
`end_ts` in seconds (English only: the other-language branch yields no
tokens). `words_from_tokens` turns those into `Segment.words` rows.
`TimedResult` is the same shape for pipelines without a model (the Dummy
backend, the test suite's `FakePipeline`), with evenly spaced words.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from kokoro_gui.engine.audio_fx import TRIM_THRESHOLD


@dataclass
class TimedToken:
    text: str
    start_ts: Optional[float]
    end_ts: Optional[float]


class TimedResult(tuple):
    """A `(graphemes, phonemes, audio)` triple with a `tokens` attribute."""

    def __new__(cls, graphemes, phonemes, audio, tokens=None):
        self = super().__new__(cls, (graphemes, phonemes, audio))
        self.tokens = tokens
        return self


def even_tokens(text: str, duration_s: float) -> list:
    """One `TimedToken` per whitespace word, spread evenly over `duration_s`."""
    words = text.split()
    if not words or duration_s <= 0:
        return []
    step = duration_s / len(words)
    return [TimedToken(w, i * step, (i + 1) * step) for i, w in enumerate(words)]


def words_from_tokens(tokens, offset_s: float = 0.0) -> list:
    """`[[text, start_s, end_s], ...]` from tokens that have times, shifted
    by `offset_s`. Punctuation-only tokens are skipped."""
    words = []
    for token in tokens or []:
        start, end = getattr(token, "start_ts", None), getattr(token, "end_ts", None)
        text = str(getattr(token, "text", "") or "").strip()
        if start is None or end is None or not any(ch.isalnum() for ch in text):
            continue
        words.append([text, round(float(start) + offset_s, 4), round(float(end) + offset_s, 4)])
    return words


def silence_bounds(audio, sample_rate: int) -> tuple:
    """`(onset_s, tail_s)`: seconds before the first and after the last
    sample above `TRIM_THRESHOLD`, the same test `process_audio`'s trim
    applies. `(0.0, 0.0)` when nothing crosses it (trim leaves such audio
    alone)."""
    audio = np.asarray(audio).reshape(-1)
    mask = np.abs(audio) > TRIM_THRESHOLD
    if not np.any(mask):
        return 0.0, 0.0
    start = int(np.argmax(mask))
    tail = int(np.argmax(mask[::-1]))
    return round(start / float(sample_rate), 6), round(tail / float(sample_rate), 6)
