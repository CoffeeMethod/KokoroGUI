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

    def _set(self, clip_id: Optional[str], character_id: Optional[str],
             selected_range: Optional[tuple[int, int]]) -> None:
        if (clip_id, character_id, selected_range) == (
            self.selected_clip_id, self.selected_character_id, self.selected_range
        ):
            return
        self.selected_clip_id = clip_id
        self.selected_character_id = character_id
        self.selected_range = selected_range
        self.changed.emit()

    def select_clip(self, clip_id: str) -> None:
        self._set(clip_id, None, None)

    def select_character(self, character_id: str) -> None:
        self._set(None, character_id, None)

    def select_range(self, start: int, end: int) -> None:
        if end <= start:
            raise ValueError(f"select_range requires end > start, got start={start}, end={end}")
        self._set(None, None, (start, end))

    def clear(self) -> None:
        self._set(None, None, None)

    def set_playing_clip(self, clip_id: Optional[str]) -> None:
        if clip_id == self.playing_clip_id:
            return
        self.playing_clip_id = clip_id
        self.playingChanged.emit()
