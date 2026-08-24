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

from kokoro_gui.engine.text_extraction import find_character_fx_spans


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

    triples.sort(key=lambda t: t[0])
    return triples, unmatched
