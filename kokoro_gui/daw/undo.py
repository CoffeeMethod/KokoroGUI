"""Undo/redo (item 4 of the DAW-for-text redesign's remaining-work roadmap):
a plain-Python `Command`/`UndoStack` pair, deliberately NOT named
`QUndoCommand`/`QUndoStack` and NOT importing anything from `PySide6`.

`kokoro_gui/daw/` is a verified zero-Qt-imports package today (see this
package's `__init__.py`) - `Document` stays constructible/testable without a
`QApplication`, and this module preserves that. This is a deliberate,
documented deviation from the more obvious "just use QUndoStack" precedent:
the one-stack-per-`Document` intent is kept, the Qt dependency is not.

Per Claude/PLAN_text_editor_redesign.md's undo-granularity grill: ordinary
interactive typing in the real GUI does NOT go through this stack at all -
it's recorded on the live `QTextDocument`'s own native undo (Qt already
coalesces keystrokes like a word processor, for free), coordinated with this
stack by `kokoro_gui.qt.transcript_editor` so Ctrl+Z pops whichever of the
two histories has the more recent action. This stack still handles every
*non-typing* document operation (character/FX assignment, clip moves, FX
overrides) - see docs/timeline_dock.py's use of `TextEditCommand` for the one
still-custom-stack text mutation, the sub-range TTS replace (a
programmatic, non-interactive text replacement triggered by a button, not
typing).

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
        # Optional zero-arg hook, set from the Qt layer (see
        # kokoro_gui.qt.undo_coordinator.UndoCoordinator) so it can log this
        # stack's pushes into the same interleaved order log as the live
        # QTextDocument's native undo, without this module importing
        # anything Qt-related itself. `None` (the default) means nobody's
        # listening - `push`/`clear_redo` stay plain no-ops toward it.
        self.on_push = None

    def push(self, command: Command) -> None:
        """Runs `command.do(document)`, records it as the most recent undoable
        action, and invalidates any redo history - a new action after an
        undo makes the undone-and-now-abandoned branch unreachable, the same
        behavior every standard undo stack has."""
        command.do(self._document)
        self._undo.append(command)
        self._redo.clear()
        if self.on_push is not None:
            self.on_push()

    def clear_redo(self) -> None:
        """Discards redo history without touching undo - called by
        `UndoCoordinator` when the live QTextDocument's native stack records
        a fresh (unrelated) edit, so a stale "redo the character
        assignment I just undid" doesn't survive an intervening edit on the
        other stack (ordinary "any new action clears redo" semantics,
        just applied across two independent stacks instead of one)."""
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
    gutter-dropdown / paste-splitting primitive). Rather than trying to
    reconstruct exactly which runs/clips a split touched, `do()` snapshots
    (deep-copies) the WHOLE `document.runs`/`document.clips` lists BEFORE
    calling it, and `undo()` restores both verbatim - `Run`/`Clip` are small
    plain-data dataclasses, so a full deep copy is cheap, and it sidesteps
    the lossy-reconstruction trap the retired offset-based version had to
    work around with a partial-snapshot-plus-reverse-replay (see
    `TextEditCommand` below for the same reasoning applied to text edits).

    `redo()` re-runs `do()` from the (now-restored) pre-split state, so it
    re-derives a fresh snapshot and re-applies the exact same split each
    time - correct across any number of undo/redo cycles.
    """

    def __init__(self, start: int, end: int, character_id):
        self.start = start
        self.end = end
        self.character_id = character_id
        self._pre_runs: "list | None" = None
        self._pre_clips: "list | None" = None
        self.new_clip_id: "str | None" = None

    def do(self, document) -> None:
        self._pre_runs = copy.deepcopy(document.runs)
        self._pre_clips = copy.deepcopy(document.clips)
        new_clip = document.assign_character_to_range(self.start, self.end, self.character_id)
        self.new_clip_id = new_clip.id

    def undo(self, document) -> None:
        document.runs = copy.deepcopy(self._pre_runs)
        document.clips = copy.deepcopy(self._pre_clips)


class TextEditCommand(Command):
    """Wraps `Document.replace_text` for one text edit - used only for
    non-interactive, custom-stack text mutations (the sub-range TTS replace
    button; see docs/timeline_dock.py), NOT for ordinary typing in the
    transcript editor, which now rides Qt's own native `QTextDocument` undo
    instead (see this module's docstring).

    Same snapshot-the-whole-run-list strategy as `AssignCharacterCommand`,
    for the same reason: once an edit fully consumes a clip, there's no
    longer enough information left in `position`/`chars_removed`/
    `chars_added` alone to know which of the surviving text's *other* runs
    that clip's characters used to belong to, so a naive "replay the edit in
    reverse" can mis-tag the restored text. Snapshotting avoids the problem
    entirely instead of solving it.
    """

    def __init__(self, position: int, chars_removed: int, chars_added: int, new_text: str):
        self.position = position
        self.chars_removed = chars_removed
        self.chars_added = chars_added
        self.new_text = new_text
        self._pre_runs: "list | None" = None
        self._pre_clips: "list | None" = None

    def do(self, document) -> None:
        self._pre_runs = copy.deepcopy(document.runs)
        self._pre_clips = copy.deepcopy(document.clips)
        document.replace_text(self.position, self.chars_removed, self.chars_added, self.new_text)

    def undo(self, document) -> None:
        document.runs = copy.deepcopy(self._pre_runs)
        document.clips = copy.deepcopy(self._pre_clips)


class MoveClipCommand(Command):
    """Item 8 ("Drag-to-reassign a clip to a different track"): the "just
    move it visually" half of Q9's reassign-vs-move prompt - relocates a
    clip to a different track's lane while leaving every other field
    (`character_id` included) untouched. `do()` snapshots the clip's
    current `track_id` before overwriting it, so `undo()` can restore it
    verbatim. Untouched by the run-list rework - it only ever mutates a
    `Clip`'s own fields, never `document.runs`."""

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
    overwriting them, so `undo()` restores both verbatim. Untouched by the
    run-list rework - it only ever mutates a `Clip`'s own fields, never
    `document.runs`."""

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

    `preset_name` (UI shell pass) is recorded alongside, in
    `clip.overrides["fx_preset"]`, so the gutter and the transcript header
    can name the preset the values came from; the Settings tab's clip-mode
    FX combo writes the same key. Clearing (`fx_values=None`) drops the
    name too.

    Deep-copies on the way in and out so a caller mutating its own dict
    after construction - or a later edit mutating `clip.fx_override` in
    place - can never alias back into this command's undo history.
    """

    def __init__(self, clip_id: str, fx_values, preset_name=None):
        self.clip_id = clip_id
        self.fx_values = copy.deepcopy(fx_values) if fx_values else fx_values
        self.preset_name = preset_name
        self._previous = None
        self._previous_name = None
        self._had_name = False

    def do(self, document) -> None:
        clip = document.get_clip(self.clip_id)
        if clip is None:
            return
        self._previous = copy.deepcopy(clip.fx_override) if clip.fx_override else clip.fx_override
        self._had_name = "fx_preset" in clip.overrides
        self._previous_name = clip.overrides.get("fx_preset")
        clip.fx_override = copy.deepcopy(self.fx_values) if self.fx_values else self.fx_values
        if self.fx_values is None:
            clip.overrides.pop("fx_preset", None)
        elif self.preset_name:
            clip.overrides["fx_preset"] = self.preset_name

    def undo(self, document) -> None:
        clip = document.get_clip(self.clip_id)
        if clip is None:
            return
        clip.fx_override = copy.deepcopy(self._previous) if self._previous else self._previous
        if self._had_name:
            clip.overrides["fx_preset"] = self._previous_name
        else:
            clip.overrides.pop("fx_preset", None)


class SetClipTimestampCommand(Command):
    """UI9: a horizontal drag on the timeline pins a clip to an explicit
    start time (`Clip.timeline_timestamp`, seconds). `None` unpins it so
    `compute_arrangement` places it after its text-order predecessor
    again."""

    def __init__(self, clip_id: str, timestamp):
        self.clip_id = clip_id
        self.timestamp = timestamp
        self._previous = None

    def do(self, document) -> None:
        clip = document.get_clip(self.clip_id)
        if clip is None:
            return
        self._previous = clip.timeline_timestamp
        clip.timeline_timestamp = self.timestamp

    def undo(self, document) -> None:
        clip = document.get_clip(self.clip_id)
        if clip is None:
            return
        clip.timeline_timestamp = self._previous


class MoveClipBeforeCommand(Command):
    """UI9 / grill Q13: dragging a clip to before another clip on the
    timeline also moves its text to just before that clip's text. Moves
    every run tagged `clip_id` (in order) to immediately before the first
    run tagged `before_clip_id`. Same whole-run-list snapshot strategy as
    `AssignCharacterCommand`. Untagged text between the moved clip's runs
    stays where it was; only the tagged runs travel.

    Also pins the moved clip's `timeline_timestamp` to `timestamp` when one
    is given (the drop position), so the drag's visual result and the text
    reorder land in one undoable step."""

    def __init__(self, clip_id: str, before_clip_id: str, timestamp=None):
        self.clip_id = clip_id
        self.before_clip_id = before_clip_id
        self.timestamp = timestamp
        self._pre_runs = None
        self._pre_clips = None
        self._previous_timestamp = None

    def do(self, document) -> None:
        self._pre_runs = copy.deepcopy(document.runs)
        self._pre_clips = copy.deepcopy(document.clips)
        clip = document.get_clip(self.clip_id)
        if clip is None or self.clip_id == self.before_clip_id:
            return
        moving = [r for r in document.runs if r.clip_id == self.clip_id]
        if not moving:
            return
        remaining = [r for r in document.runs if r.clip_id != self.clip_id]
        insert_at = next((i for i, r in enumerate(remaining) if r.clip_id == self.before_clip_id), None)
        if insert_at is None:
            return
        document.runs = remaining[:insert_at] + moving + remaining[insert_at:]
        document._normalize_runs()
        self._previous_timestamp = clip.timeline_timestamp
        if self.timestamp is not None:
            clip.timeline_timestamp = self.timestamp

    def undo(self, document) -> None:
        document.runs = copy.deepcopy(self._pre_runs)
        document.clips = copy.deepcopy(self._pre_clips)


# The split-or-create primitive item 7 ("Auto-split on generation") and
# item 9 ("Sub-range TTS replacement") will reuse is exactly
# `assign_character_to_range` - a plain alias, not a new class.
SplitClipCommand = AssignCharacterCommand
