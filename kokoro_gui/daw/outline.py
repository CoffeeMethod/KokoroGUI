"""The Outline dock's rows (plan 20): a book's chapters with a status and a
length, or a project's markers when it has no subprojects. Qt-free and
derived: nothing here is stored. "Proofed" is the nested clip's own
`status == "approved"` (`daw.models.CLIP_STATUSES`), which already saves.

A chapter is a nested clip (a subproject) in text order. Its length and
start come from the arrangement, so a subproject without a mixdown shows
the same estimate the timeline block does, flagged `estimated`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from kokoro_gui.daw import markers as marker_ops

MISSING = "missing"
NOT_STARTED = "not started"
IN_PROGRESS = "in progress"
DONE = "done"
PROOFED = "proofed"

CHAPTERS = "chapters"
MARKERS = "markers"
EMPTY = "empty"

APPROVED = "approved"


@dataclass(frozen=True)
class OutlineRow:
    title: str
    status: str  # derived label (MISSING ... PROOFED); "" on a marker row
    duration_s: float
    estimated: bool
    start_s: float
    state: str = ""  # `app.nested_state`: "ok", "stale" or "missing"; "" on a marker row
    clip_id: Optional[str] = None
    marker_id: Optional[str] = None
    clip_status: str = ""  # the clip's own `Clip.status`


@dataclass(frozen=True)
class Outline:
    kind: str
    rows: list
    total_s: float
    estimated: bool


def derived_status(state: str, clip_status: str, started: Optional[bool]) -> str:
    """The chapter's label. `state` is "ok" / "stale" / "missing", `started`
    is whether the child has any generated clip: None when that isn't known."""
    if state == "missing":
        return MISSING
    if state == "ok":
        return PROOFED if clip_status == APPROVED else DONE
    # A child that isn't open can't say whether any of its clips are
    # generated, and the dock doesn't open children to find out, so a stale
    # one reads "in progress".
    return NOT_STARTED if started is False else IN_PROGRESS


def chapter_rows(document, arrangement, state_of: Callable,
                 started_of: Optional[Callable] = None) -> list:
    """One row per nested clip, in text order. `state_of(clip)` answers
    "ok" / "stale" / "missing"; `started_of(clip)` is True, False or None
    (unknown) and may be left out."""
    out = []
    for placed in arrangement.placed:
        clip = placed.clip
        if not clip.is_nested:
            continue
        state = state_of(clip)
        started = started_of(clip) if started_of is not None else None
        clip_status = getattr(clip, "status", "") or ""
        out.append(OutlineRow(
            title=document.clip_text(clip), status=derived_status(state, clip_status, started),
            duration_s=placed.duration_s, estimated=placed.estimated, start_s=placed.start_s,
            state=state, clip_id=clip.id, clip_status=clip_status))
    return out


def marker_rows(document, arrangement) -> list:
    """One row per marker; its length runs to the next marker, the last one
    to the end of the arrangement. The status stays empty."""
    found = marker_ops.list_markers(document.settings)
    out = []
    for index, marker in enumerate(found):
        start = marker["seconds"]
        end = found[index + 1]["seconds"] if index + 1 < len(found) else arrangement.total_duration_s
        end = max(start, end)
        estimated = any(p.estimated and p.start_s < end and p.end_s > start for p in arrangement.placed)
        out.append(OutlineRow(title=marker["name"], status="", duration_s=end - start, estimated=estimated,
                              start_s=start, marker_id=marker["id"]))
    return out


def build(document, arrangement, state_of: Callable, started_of: Optional[Callable] = None) -> Outline:
    """Chapters when the document has subprojects, else its markers. A
    chapter total sums the chapter lengths; a marker total is the whole
    arrangement, the running length of the episode."""
    chapters = chapter_rows(document, arrangement, state_of, started_of)
    if chapters:
        return Outline(CHAPTERS, chapters, sum(r.duration_s for r in chapters),
                       any(r.estimated for r in chapters))
    markers = marker_rows(document, arrangement)
    estimated = any(p.estimated for p in arrangement.placed)
    if markers:
        return Outline(MARKERS, markers, arrangement.total_duration_s, estimated)
    return Outline(EMPTY, [], arrangement.total_duration_s, estimated)


def format_hms(seconds, estimated: bool = False) -> str:
    """`h:mm:ss`, with a "~" in front for an estimate."""
    whole = int(round(max(0.0, float(seconds or 0.0))))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{'~' if estimated else ''}{hours}:{minutes:02d}:{secs:02d}"
