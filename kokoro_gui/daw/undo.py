"""Undo/redo (item 4 of the DAW-for-text redesign's remaining-work roadmap):
a plain-Python `Command`/`UndoStack` pair, deliberately NOT named
`QUndoCommand`/`QUndoStack` and NOT importing anything from `PySide6`.

`kokoro_gui/daw/` is a verified zero-Qt-imports package today (see this
package's `__init__.py`) - `Document` stays constructible/testable without a
`QApplication`, and this module preserves that. This is a deliberate,
documented deviation from the more obvious "just use QUndoStack" precedent:
the one-stack-per-`Document` intent is kept, the Qt dependency is not.

No dirty/clean-flag tracking here - the app already autosaves on every
change (see `kokoro_gui.qt.settings`), so adding one would be new, unrequested
complexity.
"""
from __future__ import annotations

import copy


class Command:
    """Base class for one undoable action against a `Document`. Concrete
    commands implement `do`/`undo`; the base versions raise so a stub
    command (see `MoveClipCommand` etc. below) fails loudly if ever pushed
    before it's actually implemented, rather than silently no-op'ing."""

    def do(self, document) -> None:
        raise NotImplementedError

    def undo(self, document) -> None:
        raise NotImplementedError


class UndoStack:
    """One stack per `Document`, constructed with a reference to the
    `Document` it operates on so call sites never need to thread `document`
    through every `push`/`undo`/`redo` call themselves."""

    def __init__(self, document):
        self._document = document
        self._undo: list = []
        self._redo: list = []

    def push(self, command: Command) -> None:
        """Runs `command.do(document)`, records it as the most recent undoable
        action, and invalidates any redo history - a new action after an
        undo makes the undone-and-now-abandoned branch unreachable, the same
        behavior every standard undo stack has."""
        command.do(self._document)
        self._undo.append(command)
        self._redo.clear()

    def undo(self) -> None:
        """No-op if there's nothing to undo."""
        if not self._undo:
            return
        command = self._undo.pop()
        command.undo(self._document)
        self._redo.append(command)

    def redo(self) -> None:
        """No-op if there's nothing to redo."""
        if not self._redo:
            return
        command = self._redo.pop()
        command.do(self._document)
        self._undo.append(command)

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)


class AssignCharacterCommand(Command):
    """Wraps `Document.assign_character_to_range` (the Characters-menu /
    paste-splitting primitive). `do()` snapshots (deep-copies) every clip
    that call is about to remove or split BEFORE calling it, so `undo()` can
    restore them verbatim - same ids, same `segments` - which is what lets
    previously-generated audio survive an undo/redo round trip via the cache
    (see `kokoro_gui/daw/dirty.py`).

    `assign_character_to_range` doesn't just remove the overlapping clips -
    it also creates a fresh "new" clip for `[start, end)` plus zero or more
    fresh "leftover" clips for whatever those overlapping clips left outside
    that range (see its docstring in models.py). `undo()` needs to remove
    *all* of those newly-created clips, not just the returned one - tracked
    here via an id-set diff (`document.clips` before vs. after the call)
    rather than trusting a single stored id, since the leftovers' ids are
    never handed back to the caller at all.
    """

    def __init__(self, start: int, end: int, character_id):
        self.start = start
        self.end = end
        self.character_id = character_id
        self._removed_clips: list = []
        self._added_clip_ids: set = set()
        self.new_clip_id: "str | None" = None

    def do(self, document) -> None:
        self._removed_clips = [
            copy.deepcopy(clip)
            for clip in document.clips
            if clip.start_offset < self.end and clip.end_offset > self.start
        ]
        before_ids = {clip.id for clip in document.clips}

        new_clip = document.assign_character_to_range(self.start, self.end, self.character_id)

        self.new_clip_id = new_clip.id
        self._added_clip_ids = {clip.id for clip in document.clips} - before_ids

    def undo(self, document) -> None:
        document.clips[:] = [clip for clip in document.clips if clip.id not in self._added_clip_ids]
        document.clips.extend(copy.deepcopy(clip) for clip in self._removed_clips)


class TextEditCommand(Command):
    """Wraps `Document.apply_text_change` for one text edit.

    The hard part, per `apply_text_change`'s own docstring: an offset that
    fell strictly inside the replaced range "has no single well-defined
    mapping" and collapses to the edit's start. Naively replaying the edit
    in reverse (swapped remove/add counts) does NOT correctly restore a
    clip whose boundary sat inside the original edited range - a clip that
    survives the forward edit (not fully consumed) but has one endpoint
    collapsed to `position` can end up at the wrong offset even after a
    "correct" reverse call, because the reverse call's own collapse rule
    doesn't know what the pre-edit value used to be.

    Mitigation: `do()` snapshots `(clip.id, start_offset, end_offset)` for
    every clip overlapping `[position, position + chars_removed)` BEFORE the
    forward call runs, in addition to deep-copying whatever
    `apply_text_change` itself reports as fully removed. `undo()` replays
    the edit in reverse, re-appends the deep-copied removed clips, then
    force-restores every snapshotted clip's exact pre-edit
    `start_offset`/`end_offset` by id - overwriting whatever the reverse
    call computed rather than trusting it, since that's the only way to
    correctly undo the lossy case. This is a safe blanket approach: for a
    clip whose offsets the reverse call would have gotten right anyway, the
    overwrite just reassigns the same values.
    """

    def __init__(self, position: int, chars_removed: int, chars_added: int, old_text: str, new_text: str):
        self.position = position
        self.chars_removed = chars_removed
        self.chars_added = chars_added
        self.old_text = old_text
        self.new_text = new_text
        self._removed_clips: list = []
        self._overlap_snapshot: list = []

    def do(self, document) -> None:
        removed_end = self.position + self.chars_removed
        self._overlap_snapshot = [
            (clip.id, clip.start_offset, clip.end_offset)
            for clip in document.clips
            if clip.start_offset < removed_end and clip.end_offset > self.position
        ]

        removed = document.apply_text_change(self.position, self.chars_removed, self.chars_added, self.new_text)
        self._removed_clips = [copy.deepcopy(clip) for clip in removed]

    def undo(self, document) -> None:
        document.apply_text_change(self.position, self.chars_added, self.chars_removed, self.old_text)

        document.clips.extend(copy.deepcopy(clip) for clip in self._removed_clips)

        for clip_id, start_offset, end_offset in self._overlap_snapshot:
            clip = document.get_clip(clip_id)
            if clip is not None:
                clip.start_offset = start_offset
                clip.end_offset = end_offset


class MoveClipCommand(Command):
    """Item 8 ("Drag-to-reassign a clip to a different track"): the "just
    move it visually" half of Q9's reassign-vs-move prompt - relocates a
    clip to a different track's lane while leaving every other field
    (`character_id` included) untouched. `do()` snapshots the clip's
    current `track_id` before overwriting it, so `undo()` can restore it
    verbatim."""

    def __init__(self, clip_id: str, new_track_id: str):
        self.clip_id = clip_id
        self.new_track_id = new_track_id
        self._previous_track_id: "str | None" = None

    def do(self, document) -> None:
        clip = document.get_clip(self.clip_id)
        if clip is None:
            return
        self._previous_track_id = clip.track_id
        clip.track_id = self.new_track_id

    def undo(self, document) -> None:
        clip = document.get_clip(self.clip_id)
        if clip is None:
            return
        clip.track_id = self._previous_track_id


class ReassignTrackCommand(Command):
    """Item 8's "reassign" half of Q9's prompt: moves a clip to a different
    track's lane AND reassigns its `character_id` to match that track's
    character - unlike `MoveClipCommand`, which only ever touches
    `track_id`. `do()` snapshots both `track_id` and `character_id` before
    overwriting them, so `undo()` restores both verbatim."""

    def __init__(self, clip_id: str, new_track_id: str, new_character_id):
        self.clip_id = clip_id
        self.new_track_id = new_track_id
        self.new_character_id = new_character_id
        self._previous_track_id: "str | None" = None
        self._previous_character_id = None

    def do(self, document) -> None:
        clip = document.get_clip(self.clip_id)
        if clip is None:
            return
        self._previous_track_id = clip.track_id
        self._previous_character_id = clip.character_id
        clip.track_id = self.new_track_id
        clip.character_id = self.new_character_id

    def undo(self, document) -> None:
        clip = document.get_clip(self.clip_id)
        if clip is None:
            return
        clip.track_id = self._previous_track_id
        clip.character_id = self._previous_character_id


class SetClipFxCommand(Command):
    """Item 5 ("Per-clip FX button"): sets/clears one `Clip`'s
    `fx_override` - a resolved FX-values dict (the `ALLOWED_FX_PRESET_KEYS`
    shape), not a preset name, so an override survives the source preset
    later being renamed or deleted. `fx_values=None` clears the override
    back to "no clip-level FX, defer to the character's fx_preset."

    Deep-copies on the way in and out (both `do()`'s stored `fx_values` and
    `undo()`'s snapshot) so a caller mutating its own dict after construction
    - or a later edit mutating `clip.fx_override` in place - can never alias
    back into this command's undo history.
    """

    def __init__(self, clip_id: str, fx_values):
        self.clip_id = clip_id
        self.fx_values = copy.deepcopy(fx_values) if fx_values else fx_values
        self._previous = None

    def do(self, document) -> None:
        clip = document.get_clip(self.clip_id)
        if clip is None:
            return
        self._previous = copy.deepcopy(clip.fx_override) if clip.fx_override else clip.fx_override
        clip.fx_override = copy.deepcopy(self.fx_values) if self.fx_values else self.fx_values

    def undo(self, document) -> None:
        clip = document.get_clip(self.clip_id)
        if clip is None:
            return
        clip.fx_override = copy.deepcopy(self._previous) if self._previous else self._previous


# The split-or-create primitive item 7 ("Auto-split on generation") and
# item 9 ("Sub-range TTS replacement") will reuse is exactly
# `assign_character_to_range` - a plain alias, not a new class.
SplitClipCommand = AssignCharacterCommand
