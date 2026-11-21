"""SelectionModel: item 1 ("Sync layer") of the DAW-for-text redesign's
remaining-work roadmap. One canonical, app-wide notion of "what's currently
selected" - a clip, a character (via a timeline lane label), a plain text
range, or nothing - shared between `TranscriptEditor` and `TimelineView` so
clicking either side selects the same thing on both.

`changed` is bare (no-arg), matching `EngineSignalBridge.finished`'s existing
style (kokoro_gui/qt/signals.py): receivers read the model's current state
off its attributes rather than the signal carrying a payload, which keeps
every consumer's connect() the same regardless of which field actually
changed.

State is intentionally flat and mutually exclusive rather than a tagged
union type - `kind` is a computed discriminator over the three optional
fields so consumers never have to reconstruct "which one is set" themselves.

Several clips (plan 35): `selected_clip_ids` is a tuple with the primary clip
last, and `selected_clip_id` is that primary, so a reader that knows only one
clip keeps working. `select_clips`, `toggle_clip` and `extend_to` change it;
each emits `changed` at most once. Selection is runtime state: nothing here
reaches a project file.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, Signal


class SelectionModel(QObject):
    """Mutators are idempotent (setting the same value twice does not
    re-emit `changed`) and mutually exclusive (selecting one of clip/
    character/range clears the other two fields)."""

    changed = Signal()
    # UI4: the clip the transport is currently inside. Separate from, and
    # non-exclusive with, the user's selection - playback must never clobber
    # what they have selected. Its own signal so `changed` consumers (the
    # Settings/FX tabs) don't re-render 30 times a second.
    playingChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_clip_id: Optional[str] = None
        # The whole clip selection, primary last; read it through
        # `selected_clip_ids`, which falls back to the primary alone when a
        # caller set `selected_clip_id` directly.
        self._clip_ids: tuple = ()
        self.selected_character_id: Optional[str] = None
        self.selected_range: Optional[tuple[int, int]] = None
        self.playing_clip_id: Optional[str] = None
        # The open project the selected clip belongs to (phase 4). Set by the
        # app when the selection changes; clip ids are unique across
        # projects, so this is bookkeeping for the docks, not a key.
        self.project_id: Optional[str] = None

    @property
    def kind(self) -> str:
        if self.selected_clip_id is not None:
            return "clip"
        if self.selected_character_id is not None:
            return "character"
        if self.selected_range is not None:
            return "range"
        return "none"

    @property
    def selected_clip_ids(self) -> tuple:
        """Every selected clip, the primary (`selected_clip_id`) last."""
        primary = self.selected_clip_id
        if primary is None:
            return ()
        if self._clip_ids and self._clip_ids[-1] == primary:
            return self._clip_ids
        return (primary,)

    def _set(self, clip_ids: tuple, character_id: Optional[str],
             selected_range: Optional[tuple[int, int]]) -> None:
        if (clip_ids, character_id, selected_range) == (
            self.selected_clip_ids, self.selected_character_id, self.selected_range
        ):
            return
        self._clip_ids = clip_ids
        self.selected_clip_id = clip_ids[-1] if clip_ids else None
        self.selected_character_id = character_id
        self.selected_range = selected_range
        self.changed.emit()

    def select_clip(self, clip_id: str) -> None:
        self._set((clip_id,), None, None)

    def select_clips(self, clip_ids, primary: Optional[str] = None) -> None:
        """Selects `clip_ids` (duplicates dropped, order kept) with `primary`
        moved last; `primary` defaults to the last id. No ids clears."""
        ids = list(dict.fromkeys(cid for cid in clip_ids if cid is not None))
        if primary is not None:
            if primary in ids:
                ids.remove(primary)
            ids.append(primary)
        if not ids:
            self.clear()
            return
        self._set(tuple(ids), None, None)

    def toggle_clip(self, clip_id: str) -> None:
        """Ctrl+click: adds `clip_id` as the primary, or drops it when it is
        already selected (the last one before it becomes the primary)."""
        ids = list(self.selected_clip_ids)
        if clip_id in ids:
            ids.remove(clip_id)
        else:
            ids.append(clip_id)
        self.select_clips(ids)

    def extend_to(self, clip_id: str, order) -> None:
        """Shift+click: every clip in `order` (clip ids in text order) from the
        primary to `clip_id`, both included. The primary stays the anchor, so
        a second Shift+click extends from the same clip. With no primary, or
        one `order` doesn't hold, it selects `clip_id` alone."""
        order = list(order)
        anchor = self.selected_clip_id
        if anchor is None or anchor not in order or clip_id not in order:
            self.select_clip(clip_id)
            return
        lo, hi = sorted((order.index(anchor), order.index(clip_id)))
        self.select_clips(order[lo:hi + 1], primary=anchor)

    def select_character(self, character_id: str) -> None:
        self._set((), character_id, None)

    def select_range(self, start: int, end: int) -> None:
        if end <= start:
            raise ValueError(f"select_range requires end > start, got start={start}, end={end}")
        self._set((), None, (start, end))

    def clear(self) -> None:
        self._set((), None, None)

    def set_playing_clip(self, clip_id: Optional[str]) -> None:
        if clip_id == self.playing_clip_id:
            return
        self.playing_clip_id = clip_id
        self.playingChanged.emit()
