"""Where to cut a clip when the user splits it at a time on the timeline.

`offset_at` maps a playhead time to a text offset inside the clip under it.
It is pure: the clip's placement, its text and a callable that finds a
word's document offsets go in, so it runs without Qt or an engine. The app
wraps it (`QtTTSApp.split_offset_at`), `Document.split_clip` does the cut
and `undo.SplitClipCommand` makes it one undo step.

A cut always falls on the first letter of a word and leaves text on both
sides. The word the playhead is on starts the second half, the next word
when the playhead is in a gap, and the second word of the clip when the
playhead is on the first.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from kokoro_gui.daw.arrangement import segment_timeline

_WORD = re.compile(r"\S+")


def first_cut_from(text: str, rel: int) -> Optional[int]:
    """The first word start at or after `rel` in `text` that has text before
    it, or None. Every word start has text after it, so the result leaves
    text on both sides."""
    for match in _WORD.finditer(text):
        start = match.start()
        if start >= rel and text[:start].strip():
            return start
    return None


def timed_words(placed, start: int, word_offsets: Callable) -> list:
    """`(start_s, end_s, rel_offset)` for each word of the clip's segments
    that has a time and a place in the text, in text order. `rel_offset` is
    relative to the clip's start `start`; times are timeline seconds."""
    words = []
    for segment, seg_start, scale in segment_timeline(placed):
        for index, word in enumerate(segment.words or []):
            span = word_offsets(segment, index)
            if span is None:
                continue
            words.append((seg_start + float(word[1]) * scale, seg_start + float(word[2]) * scale,
                          span[0] - start))
    words.sort(key=lambda w: w[2])
    return words


def offset_at(placed, seconds: float, text: str, start: int, word_offsets: Callable) -> Optional[int]:
    """The document offset to cut `placed`'s clip at for timeline time
    `seconds`, or None when no cut leaves text on both sides.

    `text` is the clip's text and `start` its document offset.
    `word_offsets(segment, index)` returns the `(start, end)` document
    offsets of a segment's word, or None. With word times the cut is the
    start of the word the playhead is on, or of the next word in a gap.
    Without them it is proportional by characters, then moved forward to
    the next word start. The first word's start isn't a cut, so a playhead
    on the first word cuts before the second."""
    words = timed_words(placed, start, word_offsets)
    if words:
        under = next((w for w in words if w[0] <= seconds < w[1]), None)
        after = next((w for w in words if w[0] > seconds), None)
        hit = under or after
        if hit is None:
            return None
        rel = hit[2]
    else:
        if placed.duration_s <= 0 or not text:
            return None
        fraction = (seconds - placed.start_s) / placed.duration_s
        rel = round(min(max(fraction, 0.0), 1.0) * len(text))
    cut = first_cut_from(text, rel)
    return None if cut is None else start + cut
