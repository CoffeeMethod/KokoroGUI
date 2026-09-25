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
import uuid


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
        # Optional `(document, command) -> Command | None`, asked after each
        # push for a command that has to ride along in the same undo step
        # (kokoro_gui/daw/lanes.py's relane). A module-level function, so a
        # deep copy of the stack never captures a bound object.
        self.follow_up = None

    def push(self, command: Command) -> None:
        """Runs `command.do(document)`, records it as the most recent undoable
        action, and invalidates any redo history - a new action after an
        undo makes the undone-and-now-abandoned branch unreachable, the same
        behavior every standard undo stack has. A `follow_up` command runs
        right after and is recorded with it as one `CompositeCommand`."""
        command.do(self._document)
        if self.follow_up is not None:
            extra = self.follow_up(self._document, command)
            if extra is not None:
                extra.do(self._document)
                command = CompositeCommand([command, extra])
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


class CompositeCommand(Command):
    """Several commands as one undo step: `do` runs them in order, `undo`
    in reverse."""

    def __init__(self, commands):
        self.commands = list(commands)

    def do(self, document) -> None:
        for command in self.commands:
            command.do(document)

    def undo(self, document) -> None:
        for command in reversed(self.commands):
            command.undo(document)


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

    def __init__(self, start: int, end: int, character_id, clip_fields: "dict | None" = None):
        self.start = start
        self.end = end
        self.character_id = character_id
        # Set on the new clip after the split, e.g. `{"gap_before_s": 1.5}`
        # from a `[pause:x]` marker, so a redo recreates it too.
        self.clip_fields = dict(clip_fields or {})
        self._pre_runs: "list | None" = None
        self._pre_clips: "list | None" = None
        self._created_track_ids: list = []
        self.new_clip_id: "str | None" = None

    def do(self, document) -> None:
        self._pre_runs = copy.deepcopy(document.runs)
        self._pre_clips = copy.deepcopy(document.clips)
        track_ids = {t.id for t in document.tracks}
        new_clip = document.assign_character_to_range(self.start, self.end, self.character_id)
        for name, value in self.clip_fields.items():
            setattr(new_clip, name, copy.deepcopy(value))
        self.new_clip_id = new_clip.id
        # A character's first use makes its track (grill PR4); undo takes it
        # away again.
        self._created_track_ids = [t.id for t in document.tracks if t.id not in track_ids]

    def undo(self, document) -> None:
        document.runs = copy.deepcopy(self._pre_runs)
        document.clips = copy.deepcopy(self._pre_clips)
        created = set(self._created_track_ids)
        if created:
            document.tracks = [t for t in document.tracks if t.id not in created]


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


class RippleCommand(Command):
    """Ripple on regenerate: moves each clip in `shifts` (`{clip_id:
    seconds}`, from `arrangement.plan_ripple`) along the timeline by adding
    to its `timeline_timestamp`, never below 0. Undo puts back the exact
    previous values."""

    def __init__(self, shifts: dict):
        self.shifts = dict(shifts)
        self._previous: dict = {}

    def do(self, document) -> None:
        self._previous = {}
        for clip_id, shift in self.shifts.items():
            clip = document.get_clip(clip_id)
            if clip is None or clip.timeline_timestamp is None:
                continue
            self._previous[clip_id] = clip.timeline_timestamp
            clip.timeline_timestamp = max(0.0, float(clip.timeline_timestamp) + float(shift))

    def undo(self, document) -> None:
        for clip_id, timestamp in self._previous.items():
            clip = document.get_clip(clip_id)
            if clip is not None:
                clip.timeline_timestamp = timestamp


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


_MISSING = object()


def _field_target(document, target_kind: str, target_id):
    """The object a `SetFieldCommand` edits: a clip, track or character by
    id, or the document itself for `"document"` (whose `settings` dict is
    where project-level values live)."""
    if target_kind == "clip":
        return document.get_clip(target_id)
    if target_kind == "track":
        return document.get_track(target_id)
    if target_kind == "character":
        return document.get_character(target_id)
    if target_kind == "document":
        return document
    raise ValueError(f"unknown SetFieldCommand target kind {target_kind!r}")


class SetFieldCommand(Command):
    """Sets one field on a clip, track, character or the document, undoably.
    With `key`, the field is a dict (`clip.overrides`, `document.settings`)
    and the command sets `field[key]`; `value=None` with a key removes the
    entry. Values are deep-copied both ways, so a caller mutating its list
    afterwards (a marker list, an automation lane) can't reach the history.

    `SetFieldCommand("clip", id, "fade_in_s", 0.2)`,
    `SetFieldCommand("document", None, "settings", 0.5, key="gap_s")`."""

    def __init__(self, target_kind: str, target_id, field: str, value, key=None):
        self.target_kind = target_kind
        self.target_id = target_id
        self.field = field
        self.key = key
        self.value = copy.deepcopy(value)
        self._previous = _MISSING

    def do(self, document) -> None:
        target = _field_target(document, self.target_kind, self.target_id)
        if target is None:
            return
        if self.key is None:
            self._previous = copy.deepcopy(getattr(target, self.field))
            setattr(target, self.field, copy.deepcopy(self.value))
            return
        container = getattr(target, self.field)
        previous = container.get(self.key, _MISSING)
        self._previous = previous if previous is _MISSING else copy.deepcopy(previous)
        if self.value is None:
            container.pop(self.key, None)
        else:
            container[self.key] = copy.deepcopy(self.value)

    def undo(self, document) -> None:
        target = _field_target(document, self.target_kind, self.target_id)
        if target is None:
            return
        if self.key is None:
            if self._previous is not _MISSING:
                setattr(target, self.field, copy.deepcopy(self._previous))
            return
        container = getattr(target, self.field)
        if self._previous is _MISSING:
            container.pop(self.key, None)
        else:
            container[self.key] = copy.deepcopy(self._previous)


class SetActiveTakeCommand(Command):
    """Makes parked take `index` a clip's active take: its segment list
    swaps with `clip.segments`, the outgoing list is parked under the
    outgoing take index, and `clip.overrides["take"]` follows."""

    def __init__(self, clip_id: str, index: int):
        self.clip_id = clip_id
        self.index = int(index)
        self._previous_index = None

    def _swap(self, clip, to_index: int) -> int:
        from_index = int(clip.overrides.get("take", 0) or 0)
        incoming = clip.takes.pop(to_index, None)
        if incoming is None:
            return from_index
        if clip.segments:
            clip.takes[from_index] = clip.segments
        clip.segments = incoming
        if to_index:
            clip.overrides["take"] = to_index
        else:
            clip.overrides.pop("take", None)
        return from_index

    def do(self, document) -> None:
        clip = document.get_clip(self.clip_id)
        if clip is None or self.index not in clip.takes:
            self._previous_index = None
            return
        self._previous_index = self._swap(clip, self.index)

    def undo(self, document) -> None:
        clip = document.get_clip(self.clip_id)
        if clip is None or self._previous_index is None:
            return
        self._swap(clip, self._previous_index)


class DeleteTakeCommand(Command):
    """Drops parked take `index` from a clip. The files stay until
    close-time GC finds nothing referencing them, so undo can bring the
    take back."""

    def __init__(self, clip_id: str, index: int):
        self.clip_id = clip_id
        self.index = int(index)
        self._segments = None

    def do(self, document) -> None:
        clip = document.get_clip(self.clip_id)
        self._segments = clip.takes.pop(self.index, None) if clip is not None else None

    def undo(self, document) -> None:
        clip = document.get_clip(self.clip_id)
        if clip is not None and self._segments is not None:
            clip.takes[self.index] = self._segments


class ReplaceWithNestedCommand(Command):
    """New Subproject (phase 4): `[start, end)` of the text, with the clips
    inside it, leaves this document (the app has already copied them into
    the child) and one placeholder run for the child takes its place, on
    the "Subprojects" track. The nested clip keeps `clip_id` across redo, so
    the open child stays attached. Same whole-list snapshot as
    `AssignCharacterCommand`; undo brings the text back and leaves the
    child's project dir for close-time eviction."""

    def __init__(self, start: int, end: int, child: dict, title: str, clip_id: str):
        self.start = start
        self.end = end
        self.child = dict(child)
        self.title = title
        self.clip_id = clip_id
        self._pre = None

    def do(self, document) -> None:
        self._pre = (copy.deepcopy(document.runs), copy.deepcopy(document.clips), copy.deepcopy(document.tracks))
        if self.end > self.start:
            text = document.text
            document.replace_text(self.start, self.end - self.start, 0, text[:self.start] + text[self.end:])
        clip = document.insert_nested_clip(self.start, self.child, self.title)
        # The id is the command's, so redo re-creates the same clip.
        for run in document.runs:
            if run.clip_id == clip.id:
                run.clip_id = self.clip_id
        clip.id = self.clip_id
        clip.track_id = document.subprojects_track(create=True)

    def undo(self, document) -> None:
        runs, clips, tracks = self._pre
        document.runs = copy.deepcopy(runs)
        document.clips = copy.deepcopy(clips)
        document.tracks = copy.deepcopy(tracks)


class ImportCuesCommand(Command):
    """Subtitle import (phase 5 D2): each cue becomes a paragraph appended
    to the end of the text (a blank line before it) and a clip over it,
    locked in time at the cue's start (`timeline_timestamp`, `pinned`),
    with the cue's text as `source_text` and its length as
    `overrides["target_duration_s"]`. The transcript line is the cue's text
    on one line (a subtitle's line breaks are layout); `source_text` keeps
    them.

    `cues` are `kokoro_gui.daw.subtitles.Cue`s, `character_ids` the
    character for each, and `new_characters` the `Character`s the speaker
    mapping made, added to the document in the same step. Clip ids are
    fixed here, so a redo recreates the same clips. Undo restores the runs
    and clips and removes the characters and tracks `do` added."""

    def __init__(self, cues, character_ids, new_characters=()):
        import uuid

        cues = list(cues)
        character_ids = list(character_ids)
        if len(cues) != len(character_ids):
            raise ValueError("ImportCuesCommand needs one character id per cue")
        self.rows = [
            (" ".join(cue.text.split()), cue.text, float(cue.start_s), float(cue.end_s), character_id,
             uuid.uuid4().hex)
            for cue, character_id in zip(cues, character_ids)
        ]
        self.new_characters = [copy.deepcopy(c) for c in new_characters]
        self._pre = None
        self._created_track_ids: list = []
        self._added_character_ids: list = []

    @property
    def clip_ids(self) -> list:
        return [row[-1] for row in self.rows]

    def do(self, document) -> None:
        from kokoro_gui.daw.models import Clip, Run

        self._pre = (copy.deepcopy(document.runs), copy.deepcopy(document.clips))
        track_ids = {t.id for t in document.tracks}
        known = {c.id for c in document.characters}
        self._added_character_ids = []
        for character in self.new_characters:
            if character.id not in known:
                document.characters.append(copy.deepcopy(character))
                self._added_character_ids.append(character.id)

        tail = document.text[-2:]
        for line, source_text, start_s, end_s, character_id, clip_id in self.rows:
            if not tail or tail == "\n\n":
                separator = ""
            elif tail.endswith("\n"):
                separator = "\n"
            else:
                separator = "\n\n"
            if separator:
                document.runs.append(Run(text=separator))
            clip = Clip(
                character_id=character_id, track_id=document.track_for_character(character_id, create=True),
                timeline_timestamp=start_s, pinned=True, source_text=source_text,
                overrides={"target_duration_s": max(0.0, end_s - start_s)}, id=clip_id,
            )
            document.clips.append(clip)
            document.runs.append(Run(text=line, clip_id=clip.id, kind=clip.run_kind))
            tail = line[-2:]
        document._normalize_runs()
        self._created_track_ids = [t.id for t in document.tracks if t.id not in track_ids]

    def undo(self, document) -> None:
        runs, clips = self._pre
        document.runs = copy.deepcopy(runs)
        document.clips = copy.deepcopy(clips)
        created = set(self._created_track_ids)
        if created:
            document.tracks = [t for t in document.tracks if t.id not in created]
        added = set(self._added_character_ids)
        if added:
            document.characters = [c for c in document.characters if c.id not in added]


class ImportBedCommand(Command):
    """File > Import Audio's music bed (phase 5 P2, grill Q30): a paragraph
    at the end of the transcript holding the file's name as a placeholder
    run, and an imported clip playing `path` on the "Music" track (made
    when there is none), pinned at `at_s`. `undo` restores the runs and
    clips and removes the track if this command made it. The clip's and
    the track's ids are fixed here, so a redo recreates the same ones."""

    def __init__(self, path: str, title: str, at_s: float = 0.0):
        self.path = path
        self.title = title or "Audio"
        self.at_s = max(0.0, float(at_s))
        self.clip_id = uuid.uuid4().hex
        self._new_track_id = uuid.uuid4().hex
        self._pre = None
        self._created_track_id = None

    def do(self, document) -> None:
        from kokoro_gui.daw.models import Clip, Run, Track

        self._pre = (copy.deepcopy(document.runs), copy.deepcopy(document.clips))
        track_id = document.music_track()
        self._created_track_id = None
        if track_id is None:
            order = max((t.order_index for t in document.tracks), default=-1) + 1
            document.tracks.append(Track(name="Music", order_index=order, role="music", id=self._new_track_id))
            track_id = self._created_track_id = self._new_track_id
        tail = document.text[-2:]
        if tail and tail != "\n\n":
            document.runs.append(Run(text="\n" if tail.endswith("\n") else "\n\n"))
        clip = Clip(source="imported", original_audio_path=self.path, track_id=track_id,
                    timeline_timestamp=self.at_s, pinned=True, id=self.clip_id)
        document.clips.append(clip)
        document.runs.append(Run(text=self.title, clip_id=clip.id, kind=clip.run_kind))
        document._normalize_runs()

    def undo(self, document) -> None:
        runs, clips = self._pre
        document.runs = copy.deepcopy(runs)
        document.clips = copy.deepcopy(clips)
        if self._created_track_id is not None:
            document.tracks = [t for t in document.tracks if t.id != self._created_track_id]


class RelaneCommand(Command):
    """Puts every clip on the track the document's track layout says
    (kokoro_gui/daw/lanes.py): the unified layout's lane rule, or each
    clip's own character track. Creates the tracks that needs. The plan is
    made in `do`, so a redo after an undone split (whose new clip gets a
    fresh id) lanes what is there; `undo` restores each clip's previous
    `track_id` and removes the tracks `do` created (grill PR4)."""

    def __init__(self):
        self._previous: dict = {}
        self._created_track_ids: list = []

    def do(self, document) -> None:
        from kokoro_gui.daw.lanes import plan_relane

        plan = plan_relane(document)
        document.tracks.extend(plan.new_tracks)
        self._created_track_ids = [t.id for t in plan.new_tracks]
        self._previous = {}
        for clip_id, track_id in plan.assignments.items():
            clip = document.get_clip(clip_id)
            if clip is None:
                continue
            self._previous[clip_id] = clip.track_id
            clip.track_id = track_id

    def undo(self, document) -> None:
        for clip_id, track_id in self._previous.items():
            clip = document.get_clip(clip_id)
            if clip is not None:
                clip.track_id = track_id
        created = set(self._created_track_ids)
        if created:
            document.tracks = [t for t in document.tracks if t.id not in created]


# The split-or-create primitive item 7 ("Auto-split on generation") and
# item 9 ("Sub-range TTS replacement") will reuse is exactly
# `assign_character_to_range` - a plain alias, not a new class.
SplitClipCommand = AssignCharacterCommand
