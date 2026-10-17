"""Find and replace over a transcript (plan 34), Qt-free.

`find_all` finds a literal needle in a string, with Match case and Whole
word. `find_in_document` runs it over a `Document`'s text and drops every
hit that touches a placeholder line (a subproject's title or a music bed's
file name), because typing there is refused too. `plan_replace` says what a
replacement would do before it is applied: how many matches, in how many
clips, how many of those clips are up to date now and will turn stale, and
how many imported recordings will split. `replace_command` turns the edits
into one `CompositeCommand`, which the app pushes on the undo stack of a
project whose transcript is not on screen.

A replacement is two text edits, not one: the new text goes in just after
the match and the match is deleted afterwards. One edit over the match
would consume a clip whose whole text is the match, and the new text would
come out untagged; this way it inherits the clip of the text on its left,
the way typing inside the match does. Inside an imported recording the new
text is untagged and the recording splits around it (grill PG10: typing's
rule), whichever order the two edits run in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from kokoro_gui.daw.imported import is_recording_clip
from kokoro_gui.daw.models import PLACEHOLDER
from kokoro_gui.daw.undo import CompositeCommand, TextEditCommand

_WORD = re.compile(r"\w")


def _pattern(needle: str, match_case: bool, whole_word: bool):
    """The regex for a literal `needle`. Whole word puts a word boundary at
    each end of the needle that is a word character, so "-x-" still matches
    between spaces and "cat" does not match inside "concatenate"."""
    body = re.escape(needle)
    if whole_word:
        if _WORD.match(needle[0]):
            body = r"(?<!\w)" + body
        if _WORD.match(needle[-1]):
            body = body + r"(?!\w)"
    return re.compile(body, 0 if match_case else re.IGNORECASE)


def find_all(text: str, needle: str, match_case: bool = False, whole_word: bool = False) -> list:
    """Every `(start, end)` of `needle` in `text`, left to right and not
    overlapping: "aa" in "aaaa" is two hits, in "aaa" one. An empty needle
    has none."""
    if not needle or not text:
        return []
    return [match.span() for match in _pattern(needle, match_case, whole_word).finditer(text)]


def placeholder_spans(document) -> list:
    """`(start, end)` of every placeholder run, in order."""
    spans = []
    pos = 0
    for run in document.runs:
        end = pos + len(run.text)
        if run.kind == PLACEHOLDER and end > pos:
            spans.append((pos, end))
        pos = end
    return spans


def find_in_document(document, needle: str, match_case: bool = False, whole_word: bool = False) -> list:
    """`find_all` over the transcript, without the hits that touch a
    placeholder line."""
    hits = find_all(document.text, needle, match_case, whole_word)
    if not hits:
        return hits
    blocked = placeholder_spans(document)
    if not blocked:
        return hits
    kept = []
    i = 0
    for start, end in hits:
        while i < len(blocked) and blocked[i][1] <= start:
            i += 1
        if i < len(blocked) and blocked[i][0] < end:
            continue
        kept.append((start, end))
    return kept


def edits_for(document, hits, replacement: str) -> list:
    """`(start, end, replacement)` for every hit whose text changes. A hit
    that already reads as the replacement is left out."""
    text = document.text
    return [(start, end, replacement) for start, end in hits if text[start:end] != replacement]


@dataclass
class ReplacePlan:
    """What replacing `edits` would do to one transcript."""

    edits: list = field(default_factory=list)
    # Distinct clips (not subprojects) the edits touch.
    clips: int = 0
    # Of those, the generated clips that are up to date now: they turn stale.
    stale: int = 0
    # Imported recordings an edit lands in; one that has text on both sides
    # of a replaced word splits around the new text.
    recordings: int = 0

    @property
    def matches(self) -> int:
        return len(self.edits)


def plan_replace(document, hits, replacement: str) -> ReplacePlan:
    """The `ReplacePlan` of replacing `hits` with `replacement`."""
    edits = edits_for(document, hits, replacement)
    plan = ReplacePlan(edits=edits)
    if not edits:
        return plan
    index = document.index()
    touched: set = set()
    for start, end, _new in edits:
        for run, _r_start, _r_end in index.runs_in(start, end):
            if run.clip_id is not None and run.kind != PLACEHOLDER:
                touched.add(run.clip_id)
    clips = [clip for clip in (index.clips.get(clip_id) for clip_id in touched) if clip is not None]
    recordings = [clip for clip in clips if is_recording_clip(clip)]
    generated = [clip for clip in clips if not is_recording_clip(clip) and not clip.is_nested]
    try:
        stale_now = document.dirty_ids()
    except Exception:  # noqa: BLE001 - a plan must not fail on a half-built document
        stale_now = frozenset()
    plan.clips = len(recordings) + len(generated)
    plan.recordings = len(recordings)
    plan.stale = sum(1 for clip in generated if clip.id not in stale_now)
    return plan


def replace_command(document, edits):
    """One `CompositeCommand` that applies `edits` to `document`: for each
    edit, last to first so no offset moves, the new text is inserted at the
    match's end and the match is then deleted (see the module docstring).
    Pushing it on the document's undo stack makes the whole replacement one
    undo step."""
    text = document.text
    commands = []
    for start, end, new in sorted(edits, reverse=True):
        if new:
            text = text[:end] + new + text[end:]
            commands.append(TextEditCommand(end, 0, len(new), text))
        text = text[:start] + text[end:]
        commands.append(TextEditCommand(start, end - start, 0, text))
    return CompositeCommand(commands)
