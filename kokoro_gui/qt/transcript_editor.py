"""The transcript panel's text editor: a `QTextEdit` that stays in sync with
a `kokoro_gui.daw.models.Document`, paints per-character highlighting, and
offers a right-click Characters menu for UI-driven voice assignment (Q20 -
the authoring path that coexists with the `[Speaker:FX]:` inline syntax,
lowering into the same clip metadata).

Rebuilt per Claude/PLAN_text_editor_redesign.md's "core inversion": the
tagged run list (`Document.runs`) is primary, plain text is a derived view.
Concretely:

- **`CharacterFxHighlighter`'s two-pass reconciliation is gone** (the clip
  overlay + separately regex-scanned un-tagged inline-syntax overlay it used
  to need). `ClipHighlighter` below is a single pass over `app.document.runs`
  - there's no more "un-tagged but recognized" text to separately overlay,
  since the `[Speaker:FX]:` shorthand now converts into a REAL tagged run the
  moment it's recognized (see "shorthand" below), rather than staying
  perpetually un-clipped and cosmetically painted.
- **Deliberate deviation from the plan's literal "paint via `QTextCharFormat`
  directly on the document" phrasing: highlighting stays a `QSyntaxHighlighter`
  overlay, not a real edit to the `QTextDocument`'s own character formatting.**
  Verified directly against this PySide6 version: `QTextDocument.
  setUndoRedoEnabled()` **clears the undo/redo history outright** on every
  transition, in either direction, even with no edits made while disabled -
  so a "disable undo, paint the color, re-enable undo" dance around a real
  `cursor.setCharFormat()` call (the literal reading of the plan) would wipe
  a user's typing history every single time a clip gets tagged. A
  `QSyntaxHighlighter`'s `setFormat()` paints a presentation-only overlay
  that never touches the document's real formatting or its undo stack at
  all, which is what actually makes "coordinated, not merged" undo (below)
  possible without this landmine. `ClipHighlighter.rehighlight()` is called
  after loading text, after any custom-stack undo/redo, and after any
  tagging operation - Qt's own incremental per-block mechanism already
  re-highlights on ordinary typing, same as the retired highlighter.
- **Clip identity is never read back off the live `QTextDocument`.** "What
  clip covers this position" always resolves through
  `app.document.clip_covering`/`clip_extent` (the Qt-free run list, kept in
  sync via `_on_contents_change` below), never by inspecting Qt formatting.
  One consequence: unlike the plan's "copy/paste... travel with a normal
  rich-text copy" framing, a highlighter overlay carries nothing into the
  clipboard at all (it was never part of the document's real content), so
  `CHARACTER_ID_MIME_TYPE`'s explicit side-channel MIME type is kept for
  paste-splitting, not retired.
- **Undo is coordinated, not merged** (the grill's chosen answer): ordinary
  typing now rides the live `QTextDocument`'s own native undo
  (`setUndoRedoEnabled(True)`, restored here for the first time since the
  offset-based model needed it off) instead of pushing a `TextEditCommand`
  per keystroke. Character assignment (context menu, paste-splitting, the
  `[Speaker:FX]:` shorthand) still goes through `app.document.undo_stack`
  (kokoro_gui/daw/undo.py) - and, since it's a highlighter repaint rather
  than a real document edit, has nothing to hide from the native stack in
  the first place. `undo_coordinator` (kokoro_gui.qt.undo_coordinator.
  UndoCoordinator) is what makes Ctrl+Z pop whichever of the two histories
  acted most recently; `QtTTSApp.undo`/`redo` delegate to the same
  coordinator so the app-wide Undo/Redo menu actions behave identically to
  pressing Ctrl+Z with the editor focused.
- **`[Speaker:FX]:` shorthand (TE6)** converts into a real tagged run "on
  completing the line" (the grill's chosen answer): pressing Enter (or the
  editor losing focus, to catch a last line with no trailing Enter) checks
  the just-finished line against the tag pattern and, on a match against a
  known character, tags that whole line via the same
  `assign_character_to_range` primitive the Characters menu uses. A line
  already covered by a clip is left alone (no repeated reassignment on
  every revisit).
- **Left gutter (TE3)**: `TranscriptGutter` below is a code-editor-style
  line-number gutter that shows "Character: X" (plus "FX" when that clip
  carries an `fx_override`) instead of numbers, painted only where either
  changes from the previous line, and clickable - the "persistent,
  always-visible assignment control" TE3 asked for, applying to the same
  `assign_character_to_range` primitive the right-click Characters menu
  uses. That menu stays as a secondary path (the plan's own stated default
  for this pass; whether it's worth dropping once the gutter has proven
  itself is an open question for later, not this rebuild).

Sibling to schema_form.py (a standalone custom-widget module, not nested
under docks/) since `TranscriptEditor`/`ClipHighlighter`/`TranscriptGutter`
don't need their own subpackage.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from PySide6.QtCore import QMimeData, QRect, QSize, Qt
from PySide6.QtGui import QColor, QKeySequence, QPainter, QSyntaxHighlighter, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QMenu, QTextEdit, QWidget

from kokoro_gui.daw.undo import AssignCharacterCommand
from kokoro_gui.qt.undo_coordinator import UndoCoordinator

# Same tag syntax kokoro_gui.engine.text_extraction._SPEAKER_FX_TAG_PATTERN
# matches, anchored to the start of a line rather than searched anywhere in
# it - the "on completing the line" recognition only ever considers whether
# the line, as a whole, OPENS with a tag (see _try_recognize_shorthand_line).
_SHORTHAND_LINE_PATTERN = re.compile(r"^\[([^\]\n]{1,100})\]:\s*")

GUTTER_WIDTH_PX = 140
GUTTER_BACKGROUND_COLOR = "#1e1e1e"
GUTTER_TEXT_COLOR = "#dddddd"


class ClipHighlighter(QSyntaxHighlighter):
    """Paints each text block by whichever `Clip` a run covers, using the
    matching `Character`'s `highlight_color`. A single pass over
    `app.document.runs` - see this module's docstring for why this stays a
    presentation-only overlay rather than a real `QTextCharFormat` edit on
    the document itself.

    `daw_document_provider` is a zero-arg callable returning the current
    `kokoro_gui.daw.models.Document` (not a captured reference), so a future
    "switch project" action that reassigns `app.document` doesn't require
    rebuilding this highlighter.
    """

    def __init__(self, qt_text_document, daw_document_provider: Callable[[], object]):
        super().__init__(qt_text_document)
        self._daw_document_provider = daw_document_provider

    def highlightBlock(self, block_text: str) -> None:  # noqa: N802 (Qt override)
        daw_doc = self._daw_document_provider()
        if daw_doc is None:
            return

        block_start = self.currentBlock().position()
        block_end = block_start + len(block_text)

        pos = 0
        for run in daw_doc.runs:
            run_start, run_end = pos, pos + len(run.text)
            pos = run_end
            if run.clip_id is None or run_start >= block_end or run_end <= block_start:
                continue
            clip = daw_doc.get_clip(run.clip_id)
            character = daw_doc.get_character(clip.character_id) if clip is not None else None
            if character is None:
                continue
            lo = max(run_start, block_start) - block_start
            hi = min(run_end, block_end) - block_start
            fmt = QTextCharFormat()
            fmt.setBackground(QColor(character.highlight_color))
            self.setFormat(lo, hi - lo, fmt)


class TranscriptGutter(QWidget):
    """Left gutter beside the transcript editor (TE3) - a code-editor-style
    line-number gutter, but showing "Character: X" (plus "FX" when that
    clip carries an `fx_override`) instead of numbers, drawn only where
    either changes from the previous line. Labels are interactive: clicking
    one opens a character picker that reassigns the clicked line's covering
    clip (or, for a still-untagged line, just that one line) via the same
    `assign_character_to_range` primitive the Characters menu uses.

    Built as a child of `TranscriptEditor` itself (not a separate sibling in
    some outer layout) - `QTextEdit` has no public `firstVisibleBlock()`/
    `contentOffset()` the way `QPlainTextEdit` does, so this positions each
    line via `document().documentLayout().blockBoundingRect(block)`,
    translated by the editor's own vertical scrollbar value (verified
    directly: `cursorForPosition(QPoint(0, 0)).block()` lands on exactly the
    block this same arithmetic places at the viewport's top edge).
    """

    def __init__(self, editor: "TranscriptEditor"):
        super().__init__(editor)
        self.editor = editor
        self._label_rects: list = []  # [(QRect, line_start, line_end)]
        editor.verticalScrollBar().valueChanged.connect(lambda _value: self.update())
        editor.textChanged.connect(self.update)

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        return QSize(GUTTER_WIDTH_PX, 0)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.fillRect(event.rect(), QColor(GUTTER_BACKGROUND_COLOR))
        painter.setPen(QColor(GUTTER_TEXT_COLOR))
        self._label_rects = []

        daw_doc = self.editor.app.document
        qt_doc = self.editor.document()
        layout = qt_doc.documentLayout()
        scroll = self.editor.verticalScrollBar().value()

        previous_key = None
        block = qt_doc.begin()
        while block.isValid():
            rect = layout.blockBoundingRect(block).translated(0, -scroll)
            if rect.bottom() < 0:
                block = block.next()
                continue
            if rect.top() > self.height():
                break

            line_start = block.position()
            line_end = line_start + len(block.text())
            clip = daw_doc.clip_covering(line_start)
            character = daw_doc.get_character(clip.character_id) if clip is not None else None
            key = (character.id if character is not None else None, bool(clip.fx_override) if clip else False)

            if key != previous_key and character is not None:
                label = f"Character: {character.name}"
                if clip.fx_override:
                    label += "  FX"
                label_rect = QRect(4, int(rect.top()), self.width() - 8, max(int(rect.height()), 1))
                painter.drawText(label_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), label)
                self._label_rects.append((label_rect, line_start, line_end))

            previous_key = key
            block = block.next()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        pos = event.position().toPoint()
        for rect, line_start, line_end in self._label_rects:
            if rect.contains(pos):
                menu = self._build_picker_menu(line_start, line_end)
                if menu is not None:
                    menu.exec(event.globalPosition().toPoint())
                return

    def _build_picker_menu(self, line_start: int, line_end: int) -> Optional[QMenu]:
        """Split out from `mousePressEvent` so tests can inspect/trigger the
        picker without ever calling the blocking `.exec()` - same precedent
        as `TranscriptEditor._build_context_menu`. Widens `[line_start,
        line_end)` out to the clicked clip's full extent first (a label
        represents the whole clip, not just the one line it happens to be
        painted next to)."""
        daw_doc = self.editor.app.document
        clip = daw_doc.clip_covering(line_start)
        if clip is not None:
            extent = daw_doc.clip_extent(clip.id)
            if extent is not None:
                line_start, line_end = extent
        if line_end <= line_start:
            return None

        menu = QMenu(self)
        for character in daw_doc.characters:
            action = menu.addAction(character.name)
            action.triggered.connect(
                lambda checked=False, cid=character.id: self.editor._push_assign_character(line_start, line_end, cid)
            )
        return menu


class TranscriptEditor(QTextEdit):
    """The Direct Text tab's editor (see docks/generation_dock.py). Keeps
    `app.document` in sync with every keystroke via `Document.replace_text`,
    and adds a Characters-menu/copy-paste/shorthand authoring path on top of
    plain text entry.

    Note: `self.document()` (Qt's `QTextDocument`) and `self.app.document`
    (the DAW `Document`) are two different objects with the same short name.
    Always spell `self.app.document` out in full in this class - never alias
    it to a local `document` variable.
    """

    CHARACTER_ID_MIME_TYPE = "application/x-kokorogui-character-id"

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        # Ordinary typing now rides Qt's own native undo/redo (coalesced
        # like a word processor, for free) - see this module's docstring.
        self.setUndoRedoEnabled(True)
        self._suppress_contents_change = False
        # None outside of an active insertFromMimeData call; an int
        # accumulator while one is in progress, since a single paste can
        # fire more than one contentsChange signal (e.g. removing a prior
        # selection, then inserting) and the *net* chars added is what
        # assign_character_to_range needs.
        self._paste_chars_accumulator: Optional[int] = None
        # Item 1 ("Sync layer"): guards against _on_selection_model_changed's
        # own setTextCursor() call bouncing straight back into
        # _on_cursor_position_changed and re-selecting - same pattern as
        # _suppress_contents_change above.
        self._updating_from_model = False

        self._highlighter = ClipHighlighter(self.document(), lambda: self.app.document)
        self.undo_coordinator = UndoCoordinator(
            self.document(), self.app.document.undo_stack, self._on_custom_stack_changed
        )

        # Left gutter (TE3) - reserves its own width via setViewportMargins
        # rather than sitting in an outer layout, so it scrolls/resizes in
        # lockstep with the text automatically (see resizeEvent below).
        self._gutter = TranscriptGutter(self)
        self.setViewportMargins(GUTTER_WIDTH_PX, 0, 0, 0)

        self.document().contentsChange.connect(self._on_contents_change)
        self.cursorPositionChanged.connect(self._on_cursor_position_changed)
        self.app.selection.changed.connect(self._on_selection_model_changed)

        self.load_text(self.app.document.text)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._gutter.setGeometry(0, 0, GUTTER_WIDTH_PX, self.height())

    # -- Document sync -----------------------------------------------------

    def _on_contents_change(self, position: int, chars_removed: int, chars_added: int) -> None:
        if self._paste_chars_accumulator is not None:
            self._paste_chars_accumulator += chars_added
        if self._suppress_contents_change:
            return
        new_text = self.toPlainText()
        self.app.document.replace_text(position, chars_removed, chars_added, new_text)
        self.app.schedule_save()
        self.app.refresh_timeline()

    def load_text(self, text: str) -> None:
        """Sets the editor's text without treating it as a user edit -
        `app.document.replace_text` is not called. Used at construction to
        seed from `app.document.text`, and reusable for a future "load a
        different project" action."""
        self._suppress_contents_change = True
        try:
            self.setPlainText(text)
        finally:
            self._suppress_contents_change = False
        self.rehighlight()

    def rehighlight(self) -> None:
        """Repaints every block's highlight (and the gutter's labels) from
        `app.document.runs` - the one entry point every tagging operation
        (Characters menu, paste-splitting, shorthand recognition,
        auto-split, drag-reassign) calls after mutating the document, and
        what a custom-stack undo/redo calls too (a native text undo/redo
        re-highlights itself via Qt's own incremental per-block mechanism,
        same as ordinary typing - but the gutter still needs an explicit
        nudge even then, since plain typing can change which line a label
        belongs on without necessarily changing document.runs)."""
        self._highlighter.rehighlight()
        self._gutter.update()

    def _on_custom_stack_changed(self) -> None:
        """Called by `undo_coordinator` after every custom-stack undo/redo -
        unlike a native text undo/redo (which re-syncs itself through the
        ordinary `_on_contents_change`/highlighter path), a custom-stack
        action changed `app.document.runs`/`clips` directly, with no signal
        Qt can observe on its own."""
        self.rehighlight()
        self.app.schedule_save()
        self.app.refresh_timeline()

    def _push_assign_character(self, start: int, end: int, character_id) -> None:
        """Shared tail end of every character-assignment authoring path
        (Characters menu, paste-splitting, the `[Speaker:FX]:` shorthand) -
        pushes the undoable command, repaints, and notifies the rest of the
        app the same way every other document mutation does."""
        self.app.document.undo_stack.push(AssignCharacterCommand(start, end, character_id))
        self.rehighlight()
        self.app.schedule_save()
        self.app.refresh_timeline()

    # -- Selection sync (item 1, "Sync layer") ------------------------------

    def _on_cursor_position_changed(self) -> None:
        """Reacts to the built-in `QTextEdit.cursorPositionChanged` signal by
        pushing the caret's current position/selection into `app.selection`,
        skipped while we're the ones moving the cursor in response to a
        selection made elsewhere (`_on_selection_model_changed` below)."""
        if self._updating_from_model:
            return
        cursor = self.textCursor()
        clip = self.app.document.clip_covering(cursor.selectionStart())
        if clip is not None:
            self.app.selection.select_clip(clip.id)
        elif cursor.hasSelection():
            self.app.selection.select_range(cursor.selectionStart(), cursor.selectionEnd())
        else:
            self.app.selection.clear()

    def _on_selection_model_changed(self) -> None:
        """Reacts to `app.selection.changed` by moving the caret to match a
        clip selected elsewhere (e.g. a timeline click). Only clip selections
        round-trip here - a range/character/none change didn't originate from
        the timeline selecting a clip, so there's nothing for the transcript
        to move to."""
        clip_id = self.app.selection.selected_clip_id
        if clip_id is None:
            return
        clip = self.app.document.get_clip(clip_id)
        if clip is None:
            return
        extent = self.app.document.clip_extent(clip_id)
        if extent is None:
            return
        start, end = extent

        cursor = self.textCursor()
        if cursor.selectionStart() == start and cursor.selectionEnd() == end:
            return  # Already there - avoid a redundant round-trip.

        self._updating_from_model = True
        try:
            text_len = len(self.toPlainText())
            start = max(0, min(start, text_len))
            end = max(0, min(end, text_len))
            new_cursor = QTextCursor(self.document())
            new_cursor.setPosition(start)
            new_cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            self.setTextCursor(new_cursor)
            self.ensureCursorVisible()
        finally:
            self._updating_from_model = False

    # -- Undo/redo coordination + [Speaker:FX]: shorthand recognition -------

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.matches(QKeySequence.StandardKey.Undo):
            self.undo_coordinator.undo()
            event.accept()
            return
        if event.matches(QKeySequence.StandardKey.Redo):
            self.undo_coordinator.redo()
            event.accept()
            return

        # TE6/the "on completing the line" grill answer: capture the
        # about-to-be-finished line's plain (position, text) BEFORE letting
        # Enter actually split it - only in the common no-selection case, so
        # a "replace this selection with a newline" edit doesn't have to
        # decide which of possibly several lines just "completed". Captured
        # as plain data (not a QTextBlock handle) since the handle's
        # identity across the upcoming split isn't something to rely on.
        pending_line = None
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not self.textCursor().hasSelection():
            block = self.textCursor().block()
            pending_line = (block.position(), block.text())

        super().keyPressEvent(event)

        if pending_line is not None:
            self._try_recognize_shorthand_line(*pending_line)

    def focusOutEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().focusOutEvent(event)
        cursor = self.textCursor()
        if not cursor.hasSelection():
            block = cursor.block()
            self._try_recognize_shorthand_line(block.position(), block.text())

    def _try_recognize_shorthand_line(self, line_start: int, line_text: str) -> None:
        match = _SHORTHAND_LINE_PATTERN.match(line_text)
        if match is None:
            return
        speaker_name = match.group(1).split(":", 1)[0].strip()
        character = self.app.document.get_character_by_name(speaker_name)
        if character is None:
            return
        line_end = line_start + len(line_text)
        if line_end <= line_start:
            return
        if self.app.document.clip_covering(line_start) is not None:
            return  # already tagged - don't reassign on every revisit
        self._push_assign_character(line_start, line_end, character.id)

    # -- Characters menu -----------------------------------------------------

    def contextMenuEvent(self, event) -> None:  # noqa: N802 (Qt override)
        menu = self._build_context_menu()
        menu.exec(event.globalPos())

    def _build_context_menu(self) -> QMenu:
        """Split out from `contextMenuEvent` so tests can inspect the menu's
        contents without ever calling the blocking `.exec()`."""
        menu = self.createStandardContextMenu()
        menu.addSeparator()

        characters_menu = menu.addMenu("Characters")
        characters = self.app.document.characters
        characters_menu.setEnabled(self.textCursor().hasSelection() and bool(characters))
        for character in characters:
            action = characters_menu.addAction(character.name)
            action.triggered.connect(lambda checked=False, cid=character.id: self._assign_character(cid))

        return menu

    def _assign_character(self, character_id: str) -> None:
        cursor = self.textCursor()
        if not cursor.hasSelection():
            return
        self._push_assign_character(cursor.selectionStart(), cursor.selectionEnd(), character_id)

    # -- Copy/paste split-vs-inherit semantics ------------------------------
    # See this module's docstring for why CHARACTER_ID_MIME_TYPE is kept
    # (a highlighter overlay carries nothing into the clipboard on its own).

    def createMimeDataFromSelection(self) -> QMimeData:  # noqa: N802 (Qt override)
        mime = super().createMimeDataFromSelection()
        cursor = self.textCursor()
        if cursor.hasSelection():
            source_clip = self.app.document.clip_covering(cursor.selectionStart())
            if source_clip is not None and source_clip.character_id:
                mime.setData(self.CHARACTER_ID_MIME_TYPE, source_clip.character_id.encode("utf-8"))
        return mime

    def _extract_source_character_id(self, source: QMimeData) -> Optional[str]:
        if not source.hasFormat(self.CHARACTER_ID_MIME_TYPE):
            return None
        raw = bytes(source.data(self.CHARACTER_ID_MIME_TYPE)).decode("utf-8")
        return raw or None

    def insertFromMimeData(self, source: QMimeData) -> None:  # noqa: N802 (Qt override)
        source_character_id = self._extract_source_character_id(source)
        cursor = self.textCursor()
        insert_position = cursor.selectionStart() if cursor.hasSelection() else cursor.position()

        self._paste_chars_accumulator = 0
        try:
            super().insertFromMimeData(source)
        finally:
            chars_added = self._paste_chars_accumulator
            self._paste_chars_accumulator = None

        splits_enabled = self.app.settings.get("character_fx_paste_splits", True)
        if source_character_id and splits_enabled and chars_added:
            self._push_assign_character(insert_position, insert_position + chars_added, source_character_id)
        # Else: replace_text's existing "insertion inside an existing clip
        # extends it" behavior already is "silently inherit the
        # destination's formatting" - nothing more to do.
