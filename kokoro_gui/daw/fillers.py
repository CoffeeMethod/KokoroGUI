"""Filler words in imported recordings (plan 32).

`find_fillers(document)` walks the recording clips (`imported.is_recording_clip`)
and returns every "um", "uh" and the like it finds, as ranges of the
transcript. Removing a range is an ordinary text delete, so the words under it
go with the audio (`imported.py`) and one undo gives both back.

Two kinds. A SAFE filler is a sound no sentence needs ("um", "uh", "er", "erm",
"ah", "hmm" and their stretched spellings). A CONTEXT filler is a phrase that is
also real English ("like", "you know", "I mean", "sort of", "kind of"), so it
only matches when the speaker set it off: a comma after it, and a comma before
it or the start of a sentence. "So, like, we went" matches, "I like it" does
not.

Each range takes one neighbouring space, and the comma after the word, so the
text reads right once the cut is made. A range never leaves its clip: the match
runs over one clip's text at a time.

Qt-free, like the rest of `kokoro_gui/daw/`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from kokoro_gui.daw import imported

SAFE = "safe"
CONTEXT = "context"

# Whole words only. A hyphen or an apostrophe next to the word means it is
# part of another ("uh-huh", "ah-ha"), so it is left alone.
_EDGE = r"[\w'’-]"
_SAFE_WORDS = r"u+m+|u+h+m*|e{1,2}r|e{1,2}rm+|a{1,2}h+|h{1,2}m+"
_SAFE = re.compile(rf"(?<!{_EDGE})(?:{_SAFE_WORDS})(?!{_EDGE})", re.IGNORECASE)
_CONTEXT_WORDS = r"like|you[ \t]+know|i[ \t]+mean|sort[ \t]+of|kind[ \t]+of"
# The comma straight after the phrase is part of the match test, not the
# phrase; `_context_hit` checks what stands before it.
_CONTEXT = re.compile(rf"(?<!{_EDGE})(?:{_CONTEXT_WORDS})(?=,)", re.IGNORECASE)
_SENTENCE_END = ".!?…"
_CLOSERS = "\"'”’)]"
_BLANKS = " \t"


@dataclass(frozen=True)
class Filler:
    """One filler: `[start, end)` is what removing it deletes (the word, the
    comma after it and one space), `word_start`/`word_end` the word itself,
    `text` the word as spoken."""

    start: int
    end: int
    text: str
    kind: str
    clip_id: str
    word_start: int
    word_end: int


def find_fillers(document) -> list:
    """Every filler in the document's recording clips, in text order, with
    document character offsets. Ranges never overlap. A hit needs at least one
    timed word under it, so text typed into a recording (which has no audio)
    is never offered. Generated clips, music beds and subprojects are
    skipped."""
    text = document.text
    out: list = []
    for clip in document.clips:
        if not imported.is_recording_clip(clip):
            continue
        extent = document.clip_extent(clip.id)
        if extent is None:
            continue
        lo, hi = extent
        timed = imported.clip_words(document, clip)
        out.extend(_clip_fillers(clip.id, text, lo, hi, timed))
    out.sort(key=lambda hit: hit.start)
    return out


def _clip_fillers(clip_id: str, text: str, lo: int, hi: int, timed: list) -> list:
    clip_text = text[lo:hi]
    matches = [(m.start(), m.end(), SAFE) for m in _SAFE.finditer(clip_text)]
    matches += [(m.start(), m.end(), CONTEXT) for m in _CONTEXT.finditer(clip_text) if _context_hit(clip_text, m)]
    matches.sort()
    hits: list = []
    floor = 0
    for start, end, kind in matches:
        if start < floor or not any(w_end > lo + start and w_start < lo + end for w_start, w_end, *_ in timed):
            continue
        range_start, range_end = _removal(clip_text, start, end, floor)
        hits.append(Filler(lo + range_start, lo + range_end, clip_text[start:end], kind, clip_id, lo + start, lo + end))
        floor = range_end
    return hits


def _context_hit(clip_text: str, match: re.Match) -> bool:
    """A CONTEXT phrase counts when a comma follows it (the lookahead) and a
    comma or the start of a sentence stands before it."""
    before = clip_text[:match.start()].rstrip(_BLANKS)
    if not before or before.endswith(("\n", ",")):
        return True
    return before.rstrip(_CLOSERS).endswith(tuple(_SENTENCE_END)) and len(before) < match.start()


def _removal(clip_text: str, start: int, end: int, floor: int) -> tuple:
    """The range removing the word at `[start, end)` deletes: the word, a comma
    after it and the spaces after that, or, with no space after it, the
    spaces before it (and the comma before those when a sentence ends right
    after, so "store, um." reads "store."). Never below `floor`."""
    if clip_text[end:end + 1] == ",":
        end += 1
    if _is_blank(clip_text, end):
        while _is_blank(clip_text, end):
            end += 1
        return start, end
    while start > floor and _is_blank(clip_text, start - 1):
        start -= 1
    if start > floor and clip_text[start - 1] == "," and end < len(clip_text) and clip_text[end] in _SENTENCE_END:
        start -= 1
    return start, end


def _is_blank(text: str, index: int) -> bool:
    return 0 <= index < len(text) and text[index] in _BLANKS


def timed_span(document, filler: Filler) -> Optional[tuple]:
    """`(start_s, end_s)` in the recording file of the words under the filler's
    word, or None when none is timed. What the dialog's Play button plays."""
    clip = document.get_clip(filler.clip_id)
    if clip is None:
        return None
    times = [(start_s, end_s) for w_start, w_end, _source, start_s, end_s in imported.clip_words(document, clip)
             if w_end > filler.word_start and w_start < filler.word_end]
    if not times:
        return None
    return min(t[0] for t in times), max(t[1] for t in times)
