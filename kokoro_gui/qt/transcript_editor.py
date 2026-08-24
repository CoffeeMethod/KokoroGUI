"""The transcript panel's text editor: a `QTextEdit` that stays in sync with
a `kokoro_gui.daw.models.Document`, paints per-character/FX highlighting,
and offers a right-click Characters menu for UI-driven voice/FX assignment
(Q20 - the authoring path that coexists with the `[Speaker:FX]:` inline
syntax, lowering into the same clip metadata).

Sibling to schema_form.py (a standalone custom-widget module, not nested
under docks/) since `TranscriptEditor` and `CharacterFxHighlighter` are
tightly coupled enough to share a file.
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QMimeData
from PySide6.QtGui import QColor, QSyntaxHighlighter, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QMenu, QTextEdit

from kokoro_gui.daw.undo import AssignCharacterCommand, TextEditCommand
from kokoro_gui.engine.text_extraction import find_character_fx_spans


class CharacterFxHighlighter(QSyntaxHighlighter):
    """Paints each text block by whichever `Clip` (or, failing that, live
    `[Speaker:FX]:` tag) covers it, using the matching `Character`'s
    `highlight_color`.

    `daw_document_provider` is a zero-arg callable returning the current
    `kokoro_gui.daw.models.Document` (not a captured reference), so a future
    "switch project" action that reassigns `app.document` doesn't require
    rebuilding this highlighter.

    Text edits are rehighlighted automatically by Qt's own incremental,
    per-block mechanism combined with `QTextDocument.revision()`-keyed span
    caching below. Any change to `daw_document_provider()`'s clips/characters
    that *isn't* a text edit (a Characters-menu assignment, a splitting
    paste, an initial `TranscriptEditor.load_text` call) has no signal Qt
    can observe - callers of those mutations must explicitly call
    `rehighlight()` afterward.
    """

    def __init__(self, qt_text_document, daw_document_provider: Callable[[], object]):
        super().__init__(qt_text_document)
        self._daw_document_provider = daw_document_provider
        self._cached_revision: Optional[int] = None
        self._cached_spans: list = []

    def _spans_for_current_text(self) -> list:
        doc = self.document()
        revision = doc.revision()
        if revision != self._cached_revision:
            self._cached_spans = find_character_fx_spans(doc.toPlainText())
            self._cached_revision = revision
        return self._cached_spans

    @staticmethod
    def _overlaps_any_clip(daw_doc, start: int, end: int) -> bool:
        return any(c.start_offset < end and c.end_offset > start for c in daw_doc.clips)

    @staticmethod
    def _format_for_color(hex_color: str) -> QTextCharFormat:
        fmt = QTextCharFormat()
        fmt.setBackground(QColor(hex_color))
        return fmt

    def highlightBlock(self, block_text: str) -> None:  # noqa: N802 (Qt override)
        daw_doc = self._daw_document_provider()
        if daw_doc is None:
            return

        block_start = self.currentBlock().position()
        block_end = block_start + len(block_text)

        # Pass 1: clip overlay - clip metadata always wins over inline syntax.
        for clip in daw_doc.clips:
            if clip.start_offset >= block_end or clip.end_offset <= block_start:
                continue
            character = daw_doc.get_character(clip.character_id)
            if character is None:
                continue
            lo = max(clip.start_offset, block_start) - block_start
            hi = min(clip.end_offset, block_end) - block_start
            self.setFormat(lo, hi - lo, self._format_for_color(character.highlight_color))

        # Pass 2: live inline `[Speaker:FX]:` syntax, for text no clip covers yet.
        for span in self._spans_for_current_text():
            if span.start >= block_end or span.end <= block_start:
                continue
            if self._overlaps_any_clip(daw_doc, span.start, span.end):
                continue
            character = next((c for c in daw_doc.characters if c.name == span.speaker_name), None)
            if character is None:
                continue
            lo = max(span.start, block_start) - block_start
            hi = min(span.end, block_end) - block_start
            self.setFormat(lo, hi - lo, self._format_for_color(character.highlight_color))


class TranscriptEditor(QTextEdit):
    """The Direct Text tab's editor (see docks/generation_dock.py). Keeps
    `app.document` in sync with every keystroke via
    `Document.apply_text_change`, and adds a Characters-menu/copy-paste
    authoring path on top of plain text entry.

    Note: `self.document()` (Qt's `QTextDocument`) and `self.app.document`
    (the DAW `Document`) are two different objects with the same short name.
    Always spell `self.app.document` out in full in this class - never alias
    it to a local `document` variable.
    """

    CHARACTER_ID_MIME_TYPE = "application/x-kokorogui-character-id"

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        # Item 4 ("Undo/redo"): text edits now route through
        # `app.document.undo_stack` (see _on_contents_change below) instead
        # of `Document.apply_text_change` directly - Qt's own built-in text
        # undo/redo must never be listening at the same time, or Ctrl+Z would
        # fight the new stack over the same keystroke.
        self.setUndoRedoEnabled(False)
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

        self._highlighter = CharacterFxHighlighter(self.document(), lambda: self.app.document)
        self.document().contentsChange.connect(self._on_contents_change)
        self.cursorPositionChanged.connect(self._on_cursor_position_changed)
        self.app.selection.changed.connect(self._on_selection_model_changed)

        self.load_text(self.app.document.text)

    # -- Document sync -----------------------------------------------------

    def _on_contents_change(self, position: int, chars_removed: int, chars_added: int) -> None:
        if self._paste_chars_accumulator is not None:
            self._paste_chars_accumulator += chars_added
        if self._suppress_contents_change:
            return
        # app.document.text is still the pre-change text at this point in
        # the signal handler - Qt's contentsChange doesn't hand over the
        # inserted characters themselves, only position/counts, so this is
        # the only place old_text is still available to snapshot.
        old_text = self.app.document.text
        new_text = self.toPlainText()
        command = TextEditCommand(position, chars_removed, chars_added, old_text, new_text)
        self.app.document.undo_stack.push(command)
        self.app.schedule_save()
        self.app.refresh_timeline()

    def load_text(self, text: str) -> None:
        """Sets the editor's text without treating it as a user edit -
        `app.document.apply_text_change` is not called. Used at construction
        to seed from `app.document.text`, and reusable for a future "load a
        different project" action."""
        self._suppress_contents_change = True
        try:
            self.setPlainText(text)
        finally:
            self._suppress_contents_change = False
        self._highlighter.rehighlight()

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

        cursor = self.textCursor()
        if cursor.selectionStart() == clip.start_offset and cursor.selectionEnd() == clip.end_offset:
            return  # Already there - avoid a redundant round-trip.

        self._updating_from_model = True
        try:
            text_len = len(self.toPlainText())
            start = max(0, min(clip.start_offset, text_len))
            end = max(0, min(clip.end_offset, text_len))
            new_cursor = QTextCursor(self.document())
            new_cursor.setPosition(start)
            new_cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            self.setTextCursor(new_cursor)
            self.ensureCursorVisible()
        finally:
            self._updating_from_model = False

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
        command = AssignCharacterCommand(cursor.selectionStart(), cursor.selectionEnd(), character_id)
        self.app.document.undo_stack.push(command)
        self._highlighter.rehighlight()
        self.app.schedule_save()
        self.app.refresh_timeline()

    # -- Copy/paste split-vs-inherit semantics ------------------------------

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
            command = AssignCharacterCommand(
                insert_position, insert_position + chars_added, source_character_id
            )
            self.app.document.undo_stack.push(command)
            self._highlighter.rehighlight()
            self.app.refresh_timeline()
        # Else: apply_text_change's existing "insertion inside an existing
        # clip extends it" behavior already is "silently inherit the
        # destination's formatting" - nothing more to do.
