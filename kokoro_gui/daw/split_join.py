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


def _word_at_time(placed, seconds: float):
    """`(segment, index)` of the word the playhead is on in `placed`'s
    segments, else the next word after it, else None. Word times are
    `Segment.words` entries `[text, start_s, end_s]` mapped onto the timeline
    by `segment_timeline`."""
    following = None
    for segment, seg_start, scale in segment_timeline(placed):
        for index, word in enumerate(segment.words or []):
            start_s = seg_start + float(word[1]) * scale
            end_s = seg_start + float(word[2]) * scale
            if start_s <= seconds < end_s:
                return segment, index
            if start_s > seconds and following is None:
                following = (segment, index)
    return following


def offset_at(placed, seconds: float, text: str, start: int, word_offsets: Callable) -> Optional[int]:
    """The document offset to cut `placed`'s clip at for timeline time
    `seconds`, or None when no cut leaves text on both sides.

    `text` is the clip's text and `start` its document offset.
    `word_offsets(segment, index)` returns the `(start, end)` document
    offsets of a segment's word, or None. With word times the cut is the
    start of the word the playhead is on, or of the next word in a gap.
    Without them, or when the word has no place in the text, it is
    proportional by characters. Either way the cut moves forward to a word
    start with text before it, so a playhead on the first word cuts before
    the second."""
    rel = None
    hit = _word_at_time(placed, seconds)
    if hit is not None:
        span = word_offsets(*hit)
        if span is not None:
            rel = span[0] - start
    if rel is None:
        if placed.duration_s <= 0 or not text:
            return None
        fraction = (seconds - placed.start_s) / placed.duration_s
        rel = round(min(max(fraction, 0.0), 1.0) * len(text))
    cut = first_cut_from(text, rel)
    return None if cut is None else start + cut
