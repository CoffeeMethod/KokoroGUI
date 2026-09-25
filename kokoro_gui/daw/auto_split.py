"""Auto-split planning (item 7, "Auto-split on generation + combined-vs-
separate clip generation", of the DAW-for-text remaining-work roadmap):
turns a `[Speaker:FX]:`-tagged document (and, optionally, its untagged
narration) into a list of `(start, end, character_id)` triples ready to feed
into `Document.assign_character_to_range`-shaped calls - the automated
equivalent of invoking the transcript panel's Characters menu once per tag
(or, in "auto-split" mode, once per paragraph within each tag's span).

Pure planning, no Qt: `plan_auto_split_clips` never mutates `document` - the
caller (`QtTTSApp.auto_split_and_generate`) applies the plan by pushing one
`AssignCharacterCommand` (kokoro_gui/daw/undo.py) per triple, in ascending
`start` order, so the result is undoable like every other clip-creating
action in the app. Applying them in that order against the same, unchanging
`document.text` is safe: none of these commands are text edits, so no
triple's offsets are invalidated by an earlier one being applied first (see
`Document.assign_character_to_range`'s docstring - it only adds/removes/
splits clips, never touches `document.text`).

Two decisions this module encodes (settled in the roadmap's item 7 write-up,
not re-derived here):

- "Combined" mode = one triple per matched `[Speaker:FX]:` span (the whole
  block of text between one tag and the next, tag markup included - exactly
  `find_character_fx_spans`'s own `(start, end)`). "Auto-split" mode = the
  same spans, each additionally split on blank-line/paragraph boundaries -
  the same splitting convention `TextExtractionMixin.smart_split` uses for
  `\n\n`, replicated here (`_paragraph_ranges` below) with real offsets into
  the original document text rather than `smart_split`'s own detached text
  pieces, and skipping empty/whitespace-only pieces exactly like
  `smart_split`'s own `[c for c in chunks if c.strip()]` does.
- Untagged narration only gets auto-clipped when `document.characters` has
  EXACTLY ONE character - use that one, unambiguously. Zero or two-or-more
  characters leaves untagged stretches without a clip, exactly matching
  today's existing "some text just has no clip" state for text nobody's
  manually assigned a character to.
"""
from __future__ import annotations

import re

from kokoro_gui.engine.text_extraction import PAUSE_MARKER_PATTERN, find_character_fx_spans

_PAUSE_MARKER = re.compile(PAUSE_MARKER_PATTERN)


def find_pause_markers(text: str) -> list:
    """`[(start, end, seconds)]` for every `[pause:x]` in `text`."""
    return [(m.start(), m.end(), float(m.group(1))) for m in _PAUSE_MARKER.finditer(text)]


def _carve_pauses(text: str, triples: list) -> tuple:
    """Cuts every `[pause:x]` marker out of the planned ranges, so a marker
    stays in the transcript as untagged text and is never spoken. A marker
    inside a range splits it in two. Returns `(triples, gaps)`: `gaps` maps
    a resulting range's start to the pause that precedes it, which becomes
    that clip's `gap_before_s`."""
    markers = find_pause_markers(text)
    if not markers:
        return triples, {}
    out = []
    gaps = {}
    pending = None
    index = 0

    def emit(start, end, character_id):
        nonlocal pending
        if not text[start:end].strip():
            return
        out.append((start, end, character_id))
        if pending is not None:
            gaps[start] = pending
            pending = None

    for start, end, character_id in sorted(triples, key=lambda t: t[0]):
        while index < len(markers) and markers[index][0] < start:
            pending = markers[index][2]
            index += 1
        cursor = start
        while index < len(markers) and markers[index][1] <= end:
            m_start, m_end, seconds = markers[index]
            emit(cursor, m_start, character_id)
            pending = seconds
            cursor = m_end
            index += 1
        emit(cursor, end, character_id)
    return out, gaps


def plan_pause_gaps(document, triples: list) -> dict:
    """`{start: seconds}` for the planned ranges a `[pause:x]` precedes,
    from `plan_auto_split_clips`'s triples (which have the markers carved
    out already). The caller hands each as `gap_before_s` to the
    `AssignCharacterCommand` that creates that clip."""
    return _carve_pauses(document.text, triples)[1]


def _paragraph_ranges(text: str, base_offset: int) -> list:
    """Splits `text` (a substring of the document starting at `base_offset`
    in the document's own coordinates) into `(start, end)` ranges on
    `smart_split`'s `text.split('\n\n')` boundary rule, translated back into
    absolute document offsets, skipping empty/whitespace-only pieces."""
    ranges = []
    cursor = base_offset
    pieces = text.split("\n\n")
    for i, piece in enumerate(pieces):
        piece_start = cursor
        piece_end = cursor + len(piece)
        if piece.strip():
            ranges.append((piece_start, piece_end))
        cursor = piece_end
        if i < len(pieces) - 1:
            cursor += 2  # the "\n\n" separator consumed by str.split, restored
    return ranges


def _untagged_gaps(text: str, covered: list) -> list:
    """The complement of `covered` (a list of `(start, end)` ranges, already
    in ascending offset order since `find_character_fx_spans` returns
    offset-ordered spans) within `[0, len(text))`."""
    gaps = []
    cursor = 0
    for start, end in covered:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < len(text):
        gaps.append((cursor, len(text)))
    return gaps


def plan_auto_split_clips(document, split_by_paragraph: bool):
    """Returns `(triples, unmatched_span_names)`:

    - `triples`: `list[(start, end, character_id)]`, ascending `start` order.
    - `unmatched_span_names`: the `speaker_name` of every tagged span that
      matched no existing `Character` (via `Document.get_character_by_name`)
      - those spans contribute zero triples and never raise; the caller
      surfaces this list as a warning rather than silently dropping it.
    """
    text = document.text
    spans = find_character_fx_spans(text)

    triples = []
    unmatched = []
    covered = []

    for span in spans:
        covered.append((span.start, span.end))
        character = document.get_character_by_name(span.speaker_name)
        if character is None:
            unmatched.append(span.speaker_name)
            continue

        if split_by_paragraph:
            for p_start, p_end in _paragraph_ranges(text[span.start:span.end], span.start):
                triples.append((p_start, p_end, character.id))
        else:
            triples.append((span.start, span.end, character.id))

    if len(document.characters) == 1:
        only_character = document.characters[0]
        for gap_start, gap_end in _untagged_gaps(text, covered):
            if split_by_paragraph:
                for p_start, p_end in _paragraph_ranges(text[gap_start:gap_end], gap_start):
                    triples.append((p_start, p_end, only_character.id))
            elif text[gap_start:gap_end].strip():
                triples.append((gap_start, gap_end, only_character.id))

    # A subproject's placeholder line is never retagged.
    triples = [t for t in triples if not document.overlaps_nested(t[0], t[1])]
    triples.sort(key=lambda t: t[0])
    triples, _gaps = _carve_pauses(text, triples)
    return triples, unmatched
