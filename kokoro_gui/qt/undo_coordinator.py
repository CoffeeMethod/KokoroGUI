"""Coordinates undo/redo between two independent histories, per
Claude/PLAN_text_editor_redesign.md's undo-granularity grill answer: the
live `QTextDocument`'s own native undo (ordinary typing - Qt already
coalesces keystrokes like a word processor, for free) and the DAW
`Document`'s custom `UndoStack` (kokoro_gui/daw/undo.py - character/FX
assignment, clip moves, FX overrides). The two stacks are deliberately kept
separate rather than merged into one - this class is the "coordinated" half
of that choice: Ctrl+Z/Ctrl+Shift+Z, wherever triggered from (the transcript
editor's own keyPressEvent, or the app-wide Undo/Redo menu actions), pop
whichever history has the more recent action, using an interleaved
chronological order log rather than merging the histories themselves.

Why a log instead of just comparing `isUndoAvailable()`/`can_undo()`: those
only say "is there *anything* to undo" on each side, not "which side's
top-of-stack action happened most recently" - the log is what answers that.
A fresh push on either stack clears the OTHER stack's redo too
(`_on_native_command_added`/`_on_custom_command_pushed` below), matching
ordinary "a new action makes the undone-and-abandoned branch unreachable"
undo-stack semantics extended across two stacks instead of one; undone
entries go on a redo log so redo replays them in the order they were undone.

Joined steps (phase 5 P3): an edit of imported recording text changes
words Qt's native undo can't give back, so the editor records it twice, as
the native text edit and as a `TextEditCommand` on the custom stack, and
`push_joined` makes the two one log entry (`_Joined`). Undoing it undoes
the native edit with the editor's sync switched off (`run_joined`) and then
the custom command(s), whose snapshot restores the runs and words. A paste
of timed text adds its `ApplyWordsCommand` to the same entry. When the
native text history is wiped (`native_history_cleared`, after the editor
reloads its text), native entries go and a joined entry keeps only its
custom half.

While an undo or redo runs (`replaying`), the text changes it makes are
replays, not edits: the editor applies them to the document and pushes
nothing, so the stacks and the order log stay as they were.
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtGui import QTextDocument

NATIVE = "native"
CUSTOM = "custom"


class _Joined:
    """One undo step made of the native edit on top of Qt's stack and the
    last `count` commands on the custom stack."""

    def __init__(self, count: int = 1):
        self.count = count


class UndoCoordinator:
    def __init__(self, qt_text_document: QTextDocument, daw_undo_stack, on_custom_stack_changed: Callable[[], None],
                 run_joined: Optional[Callable[[Callable[[], None]], None]] = None):
        self._qt_text_document = qt_text_document
        self._daw_undo_stack = daw_undo_stack
        # Called after every custom-stack undo/redo (never after a native
        # one - a native text undo/redo already re-syncs itself through the
        # editor's ordinary contentsChange handling) so the caller can
        # repaint the live editor's QTextCharFormat runs to match whatever
        # app.document.runs now says, and refresh save/timeline state.
        self._on_custom_stack_changed = on_custom_stack_changed
        # `run_joined(step)` runs a joined entry's undo or redo: the
        # editor calls `step` with its document sync off, then brings the
        # document's text in line and repaints. None runs `step` and
        # `on_custom_stack_changed`.
        self._run_joined = run_joined

        self._order: list = []  # NATIVE | CUSTOM | _Joined, most-recent last
        self._redo_order: list = []  # undone entries, most-recently-undone last
        # True between Qt adding an undo command and the editor handling
        # the text change it belongs to (`edit_seen`): the change got a
        # command of its own instead of merging into the previous one.
        self._fresh_native = False
        # True while `undo`/`redo` move either stack.
        self.replaying = False

        qt_text_document.undoCommandAdded.connect(self._on_native_command_added)
        daw_undo_stack.on_push = self._on_custom_command_pushed

    def _on_native_command_added(self) -> None:
        self._order.append(NATIVE)
        self._redo_order.clear()
        self._fresh_native = True
        self._daw_undo_stack.clear_redo()

    def _on_custom_command_pushed(self) -> None:
        self._order.append(CUSTOM)
        self._redo_order.clear()
        self._qt_text_document.clearUndoRedoStacks(QTextDocument.Stacks.RedoStack)

    def edit_seen(self) -> None:
        """The editor handled one text change; the next change is fresh only
        if Qt adds a command for it."""
        self._fresh_native = False

    def push_joined(self, command) -> None:
        """Pushes `command` onto the custom stack as part of the native edit
        that just happened, so one undo takes back both. It joins the native
        entry Qt added for the edit, or the joined entry the edit merged
        into (Qt coalesces adjacent edits), or, when neither is on top of
        the log, stands as its own custom entry."""
        fresh = self._fresh_native
        on_push, self._daw_undo_stack.on_push = self._daw_undo_stack.on_push, None
        try:
            self._daw_undo_stack.push(command)
        finally:
            self._daw_undo_stack.on_push = on_push
        self._redo_order.clear()
        top = self._order[-1] if self._order else None
        if fresh and top == NATIVE:
            self._order[-1] = _Joined(1)
        elif not fresh and isinstance(top, _Joined):
            top.count += 1
        else:
            self._order.append(CUSTOM)

    def native_history_cleared(self) -> None:
        """Qt's text undo history was wiped (the editor reloaded its text):
        drops the native entries and keeps each joined entry's custom
        commands as custom entries, so the order log matches the stacks
        again."""
        order = []
        for kind in self._order:
            if isinstance(kind, _Joined):
                order.extend([CUSTOM] * kind.count)
            elif kind != NATIVE:
                order.append(kind)
        self._order = order
        self._redo_order.clear()

    def can_undo(self) -> bool:
        return bool(self._order)

    def can_redo(self) -> bool:
        return self._daw_undo_stack.can_redo() or self._qt_text_document.isRedoAvailable()

    def _joined(self, step: Callable[[], None]) -> None:
        if self._run_joined is not None:
            self._run_joined(step)
        else:
            step()
            self._on_custom_stack_changed()

    def undo(self) -> None:
        """No-op if there's nothing to undo on either side."""
        if not self._order:
            return
        self.replaying = True
        try:
            self._undo()
        finally:
            self.replaying = False

    def _undo(self) -> None:
        kind = self._order.pop()
        if kind == NATIVE:
            self._qt_text_document.undo()
        elif kind == CUSTOM:
            self._daw_undo_stack.undo()
            self._on_custom_stack_changed()
        else:
            def _step():
                self._qt_text_document.undo()
                for _ in range(kind.count):
                    self._daw_undo_stack.undo()

            self._joined(_step)
        self._redo_order.append(kind)

    def redo(self) -> None:
        """Replays the most recently undone entry. Redoing pushes a fresh
        entry back onto the order log, same as an original action - it's
        "the most recent thing that happened" again, for the next undo.
        With nothing on the redo log (an undo made outside this class),
        resumes whichever side has redo available - at most one ever does,
        since a fresh push on either stack clears the other's redo."""
        self.replaying = True
        try:
            self._redo()
        finally:
            self.replaying = False

    def _redo(self) -> None:
        kind = self._redo_order.pop() if self._redo_order else None
        if isinstance(kind, _Joined):
            def _step():
                self._qt_text_document.redo()
                for _ in range(kind.count):
                    self._daw_undo_stack.redo()

            self._joined(_step)
            self._order.append(kind)
        elif kind == CUSTOM or (kind is None and self._daw_undo_stack.can_redo()):
            if not self._daw_undo_stack.can_redo():
                return
            self._daw_undo_stack.redo()
            self._order.append(CUSTOM)
            self._on_custom_stack_changed()
        elif self._qt_text_document.isRedoAvailable():
            self._qt_text_document.redo()
            self._order.append(NATIVE)
