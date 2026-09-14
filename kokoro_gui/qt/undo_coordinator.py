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
Redo doesn't need the log at all (see `redo()`'s docstring) - only one side
can ever have redo available at a time, since a fresh push on either stack
clears the OTHER stack's redo too (`_on_native_command_added`/
`_on_custom_command_pushed` below), matching ordinary "a new action makes
the undone-and-abandoned branch unreachable" undo-stack semantics extended
across two stacks instead of one.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtGui import QTextDocument


class UndoCoordinator:
    def __init__(self, qt_text_document: QTextDocument, daw_undo_stack, on_custom_stack_changed: Callable[[], None]):
        self._qt_text_document = qt_text_document
        self._daw_undo_stack = daw_undo_stack
        # Called after every custom-stack undo/redo (never after a native
        # one - a native text undo/redo already re-syncs itself through the
        # editor's ordinary contentsChange handling) so the caller can
        # repaint the live editor's QTextCharFormat runs to match whatever
        # app.document.runs now says, and refresh save/timeline state.
        self._on_custom_stack_changed = on_custom_stack_changed

        self._order: list = []  # "native" | "custom", most-recent last

        qt_text_document.undoCommandAdded.connect(self._on_native_command_added)
        daw_undo_stack.on_push = self._on_custom_command_pushed

    def _on_native_command_added(self) -> None:
        self._order.append("native")
        self._daw_undo_stack.clear_redo()

    def _on_custom_command_pushed(self) -> None:
        self._order.append("custom")
        self._qt_text_document.clearUndoRedoStacks(QTextDocument.Stacks.RedoStack)

    def can_undo(self) -> bool:
        return bool(self._order)

    def can_redo(self) -> bool:
        return self._daw_undo_stack.can_redo() or self._qt_text_document.isRedoAvailable()

    def undo(self) -> None:
        """No-op if there's nothing to undo on either side."""
        if not self._order:
            return
        kind = self._order.pop()
        if kind == "native":
            self._qt_text_document.undo()
        else:
            self._daw_undo_stack.undo()
            self._on_custom_stack_changed()

    def redo(self) -> None:
        """Resumes whichever side currently has redo available - at most one
        ever does (see the module docstring), so there's no ambiguity to
        resolve via the order log the way `undo()` needs it. Redoing pushes
        a fresh entry back onto the order log, same as an original action -
        it's "the most recent thing that happened" again, for the next
        undo."""
        if self._daw_undo_stack.can_redo():
            self._daw_undo_stack.redo()
            self._order.append("custom")
            self._on_custom_stack_changed()
        elif self._qt_text_document.isRedoAvailable():
            self._qt_text_document.redo()
            self._order.append("native")
