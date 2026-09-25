"""The transcript panel's text editor: a `QTextEdit` that stays in sync with
a `kokoro_gui.daw.models.Document`, paints per-character highlighting, and
offers a right-click Characters menu for UI-driven voice assignment (Q20 -
the authoring path that coexists with the `[Speaker:FX]:` inline syntax,
lowering into the same clip metadata).

Rebuilt per Claude/PLAN_text_editor_redesign.md's "core inversion": the
tagged run list (`Document.runs`) is primary, plain text is a derived view.
Concretely:

- **`ClipHighlighter` is a single pass over `app.document.runs`** - the
  `[Speaker:FX]:` shorthand converts into a real tagged run the moment it's
  recognized, so there's no separately-overlaid "recognized but un-tagged"
  text any more.
- **Highlighting stays a `QSyntaxHighlighter` overlay, not a real edit to
  the `QTextDocument`'s formatting.** `QTextDocument.setUndoRedoEnabled()`
  clears the undo history on every transition in this PySide6 version, so
  painting through `cursor.setCharFormat()` would wipe typing history each
  time a clip gets tagged. A highlighter's `setFormat()` never touches the
  document's real formatting or its undo stack.
- **Clip identity is never read back off the live `QTextDocument`.** "What
  clip covers this position" always resolves through
  `app.document.clip_covering`/`clip_extent`. A highlighter overlay carries
  nothing into the clipboard, so `CHARACTER_ID_MIME_TYPE` stays as an
  explicit side channel for paste-splitting.
- **Undo is coordinated, not merged**: typing rides the `QTextDocument`'s
  native undo; character assignment goes through `app.document.undo_stack`.
  `undo_coordinator` pops whichever acted most recently.
- **`[Speaker:FX]:` shorthand (TE6)** converts on completing the line
  (Enter, or focus-out for a last line without one).

UI-shell pass (Claude/PLAN_ui_shell_redesign.md section 2):

- The gutter labels once per `(character, fx)` change, as two lines
  (`Narrator` / `FX: Echo`), and draws a play button beside each dirty
  clip's first line (UI3). Clicking it runs `app.generate_clip(clip_id)`.
- Runs belonging to a dirty clip get a dashed underline
  (`ClipHighlighter`'s second pass).
- Split rules (UI2): a thin line at every boundary `plan_auto_split_clips`
  would produce with the current "split by paragraph" setting, plus every
  existing clip boundary, painted over the viewport after `super()`.
  Recomputed 150ms after the last edit.
- The clip being played back (`SelectionModel.playing_clip_id`) is shown as
  a translucent `ExtraSelection` and scrolled into view (UI4). The word
  under the playhead gets a second, stronger one (`set_playing_word`, fed by
  `QtTTSApp.word_at` from the segments' stored word times).
- Ctrl+click seeks the transport to the word under the pointer
  (`QtTTSApp.seek_to_offset`).
- A clip with `source_text` shows it as the gutter label's tooltip.
- Colors come from `kokoro_gui.qt.theme.current()` (UI10).

Note: `self.document()` (Qt's `QTextDocument`) and `self.app.document` (the
DAW `Document`) are two different objects with the same short name. Always
spell `self.app.document` out in full in this class.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from PySide6.QtCore import QEvent, QMimeData, QRect, QSize, Qt, QTimer
from PySide6.QtGui import (
    QColor, QFont, QKeySequence, QPainter, QPen, QPolygon, QSyntaxHighlighter, QTextCharFormat,
    QTextCursor,
)
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QMenu, QTextEdit, QToolTip, QWidget

from kokoro_gui.daw.auto_split import plan_auto_split_clips
from kokoro_gui.daw.undo import AssignCharacterCommand
from kokoro_gui.qt import theme
from kokoro_gui.qt.undo_coordinator import UndoCoordinator

# Same tag syntax kokoro_gui.engine.text_extraction._SPEAKER_FX_TAG_PATTERN
# matches, anchored to the start of a line rather than searched anywhere in
# it - the "on completing the line" recognition only ever considers whether
# the line, as a whole, OPENS with a tag (see _try_recognize_shorthand_line).
_SHORTHAND_LINE_PATTERN = re.compile(r"^\[([^\]\n]{1,100})\]:\s*")

GUTTER_WIDTH_PX = 140
# Character highlights tint the text rather than paint over it, so the
# same hex reads on the light and the dark panel.
HIGHLIGHT_ALPHA = 90
GUTTER_BUTTON_PX = 16
SPLIT_RULE_DEBOUNCE_MS = 150
_FX_PLACEHOLDER = "Select FX Preset..."


def clip_fx_name(daw_doc, clip) -> Optional[str]:
    """The FX preset name a clip resolves to for display: its own named
    override (`overrides["fx_preset"]`, set by the FX combo / Settings tab
    / timeline FX menu), else "custom" when it carries resolved
    `fx_override` values with no recorded name, else the character's
    attached `fx_preset`, else None."""
    if clip is None:
        return None
    own = clip.overrides.get("fx_preset")
    if own and own != _FX_PLACEHOLDER:
        return own
    if clip.fx_override:
        return "custom"
    character = daw_doc.get_character(clip.character_id)
    if character is not None:
        name = character.preset_data.get("fx_preset")
        if name and name != _FX_PLACEHOLDER:
            return name
    return None


class ClipHighlighter(QSyntaxHighlighter):
    """Paints each text block by whichever `Clip` a run covers, using the
    matching `Character`'s `highlight_color`, then dash-underlines the runs
    of every dirty clip. `dirty_ids()` is computed once per rehighlight
    cycle and invalidated by the editor on every content change and after
    generation finishes.

    `daw_document_provider` is a zero-arg callable returning the current
    `kokoro_gui.daw.models.Document` (not a captured reference), so a
    "switch project" action that reassigns `app.document` doesn't require
    rebuilding this highlighter.
    """

    def __init__(self, qt_text_document, daw_document_provider: Callable[[], object]):
        super().__init__(qt_text_document)
        self._daw_document_provider = daw_document_provider
        self._dirty_ids: Optional[set] = None

    def invalidate_dirty(self) -> None:
        self._dirty_ids = None

    def dirty_ids(self) -> set:
        if self._dirty_ids is None:
            daw_doc = self._daw_document_provider()
            try:
                self._dirty_ids = {c.id for c in daw_doc.dirty_clips()} if daw_doc is not None else set()
            except Exception:
                self._dirty_ids = set()
        return self._dirty_ids

    def rehighlight(self) -> None:  # noqa: N802 (Qt override)
        self.invalidate_dirty()
        super().rehighlight()

    def highlightBlock(self, block_text: str) -> None:  # noqa: N802 (Qt override)
        daw_doc = self._daw_document_provider()
        if daw_doc is None:
            return

        block_start = self.currentBlock().position()
        block_end = block_start + len(block_text)
        dirty = self.dirty_ids()
        underline_color = QColor(theme.current().dirty_underline)

        pos = 0
        for run in daw_doc.runs:
            run_start, run_end = pos, pos + len(run.text)
            pos = run_end
            if run.clip_id is None or run_start >= block_end or run_end <= block_start:
                continue
            clip = daw_doc.get_clip(run.clip_id)
            character = daw_doc.get_character(clip.character_id) if clip is not None else None
            lo = max(run_start, block_start) - block_start
            hi = min(run_end, block_end) - block_start
            if hi <= lo:
                continue
            fmt = QTextCharFormat()
            if run.kind == "placeholder":
                # A subproject's line (phase 4): its title, read-only.
                fmt.setBackground(QColor(theme.current().panel_alt))
                fmt.setFontItalic(True)
            elif character is not None:
                tint = QColor(character.highlight_color)
                tint.setAlpha(HIGHLIGHT_ALPHA)
                fmt.setBackground(tint)
            if clip is not None and clip.id in dirty:
                fmt.setUnderlineStyle(QTextCharFormat.UnderlineStyle.DashUnderline)
                fmt.setUnderlineColor(underline_color)
            self.setFormat(lo, hi - lo, fmt)


class TranscriptGutter(QWidget):
    """Left gutter beside the transcript editor (TE3, reshaped by UI3).
    Shows the character name and, on a second line, the resolved FX preset
    wherever the `(character, fx)` pair changes from the previous line,
    and a small play button beside each dirty clip's first visible line.
    Labels open a character picker for the clicked clip; the button runs a
    scoped Generate for that one clip.

    Built as a child of `TranscriptEditor` itself - `QTextEdit` has no
    public `firstVisibleBlock()`/`contentOffset()`, so lines are positioned
    via `document().documentLayout().blockBoundingRect(block)` translated
    by the editor's vertical scrollbar value.
    """

    def __init__(self, editor: "TranscriptEditor"):
        super().__init__(editor)
        self.editor = editor
        self._label_rects: list = []  # [(QRect, line_start, line_end)]
        self._button_rects: list = []  # [(QRect, clip_id)]
        self.setMouseTracking(True)
        editor.verticalScrollBar().valueChanged.connect(lambda _value: self.update())
        editor.textChanged.connect(self.update)

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        return QSize(GUTTER_WIDTH_PX, 0)

    def _label_key(self, daw_doc, clip):
        if clip is None:
            return (None, None)
        return (clip.character_id, clip_fx_name(daw_doc, clip))

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        pal = theme.current()
        painter = QPainter(self)
        painter.fillRect(event.rect(), QColor(pal.gutter_bg))
        self._label_rects = []
        self._button_rects = []

        daw_doc = self.editor.app.document
        qt_doc = self.editor.document()
        layout = qt_doc.documentLayout()
        scroll = self.editor.verticalScrollBar().value()
        dirty_ids = {c.id for c in daw_doc.dirty_clips()}
        labelled_dirty: set = set()

        base_font = QFont(self.font())
        small_font = QFont(base_font)
        small_font.setPointSizeF(max(6.0, base_font.pointSizeF() - 1.5))
        metrics_h = painter.fontMetrics().height()

        previous_key = None
        block = qt_doc.begin()
        while block.isValid():
            rect = layout.blockBoundingRect(block).translated(0, -scroll)
            if rect.bottom() < 0:
                # Off the top - still track the key so the first visible
                # line labels only if it differs from the hidden line above.
                clip = daw_doc.clip_covering(block.position())
                previous_key = self._label_key(daw_doc, clip)
                if clip is not None and clip.id in dirty_ids:
                    labelled_dirty.add(clip.id)
                block = block.next()
                continue
            if rect.top() > self.height():
                break

            line_start = block.position()
            line_end = line_start + len(block.text())
            clip = daw_doc.clip_covering(line_start)
            character = daw_doc.get_character(clip.character_id) if clip is not None else None
            key = self._label_key(daw_doc, clip)

            top = int(rect.top())
            line_h = max(int(rect.height()), 1)
            text_right = self.width() - GUTTER_BUTTON_PX - 10

            if key != previous_key and character is not None:
                name_rect = QRect(4, top, text_right - 4, metrics_h)
                painter.setFont(base_font)
                painter.setPen(QColor(character.highlight_color))
                painter.drawText(name_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                                 painter.fontMetrics().elidedText(character.name, Qt.TextElideMode.ElideRight,
                                                                  name_rect.width()))
                fx_name = key[1]
                label_height = metrics_h
                if fx_name:
                    fx_text = f"FX: {fx_name}"
                    if line_h >= 2 * metrics_h - 2:
                        fx_rect = QRect(4, top + metrics_h, text_right - 4, metrics_h)
                        painter.setFont(small_font)
                        painter.setPen(QColor(pal.gutter_text))
                        painter.drawText(fx_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                                         painter.fontMetrics().elidedText(fx_text, Qt.TextElideMode.ElideRight,
                                                                          fx_rect.width()))
                        label_height = 2 * metrics_h
                    else:
                        # One-line run: the FX line goes to the tooltip.
                        self.setToolTip(fx_text)
                label_rect = QRect(4, top, text_right - 4, max(label_height, line_h))
                self._label_rects.append((label_rect, line_start, line_end))

            if clip is not None and clip.id in dirty_ids and clip.id not in labelled_dirty:
                labelled_dirty.add(clip.id)
                btn = QRect(self.width() - GUTTER_BUTTON_PX - 4, top + max(0, (min(line_h, metrics_h) - GUTTER_BUTTON_PX) // 2),
                            GUTTER_BUTTON_PX, GUTTER_BUTTON_PX)
                self._draw_play_button(painter, btn, pal)
                self._button_rects.append((btn, clip.id))

            previous_key = key
            block = block.next()

    @staticmethod
    def _draw_play_button(painter: QPainter, rect: QRect, pal) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor(pal.dirty_underline), 1))
        painter.setBrush(QColor(pal.dirty_underline))
        tri = QPolygon([
            QPoint(rect.left() + 4, rect.top() + 3),
            QPoint(rect.right() - 3, rect.center().y()),
            QPoint(rect.left() + 4, rect.bottom() - 2),
        ])
        painter.drawPolygon(tri)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

    def button_rects(self) -> list:
        return list(self._button_rects)

    def tooltip_at(self, pos) -> Optional[str]:
        """The source text of the clip whose label is at `pos`, if any."""
        for rect, line_start, _line_end in self._label_rects:
            if rect.contains(pos):
                clip = self.editor.app.document.clip_covering(line_start)
                if clip is not None and clip.source_text:
                    return f"Source: {clip.source_text}"
        return None

    def event(self, event) -> bool:  # noqa: N802 (Qt override)
        if event.type() == QEvent.Type.ToolTip:
            text = self.tooltip_at(event.pos())
            if text:
                QToolTip.showText(event.globalPos(), text, self)
                return True
        return super().event(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        pos = event.position().toPoint()
        for rect, clip_id in self._button_rects:
            if rect.contains(pos):
                self.editor.app.generate_clip(clip_id)
                return
        for rect, line_start, line_end in self._label_rects:
            if rect.contains(pos):
                menu = self._build_picker_menu(line_start, line_end)
                if menu is not None:
                    menu.exec(event.globalPosition().toPoint())
                return

    def _build_picker_menu(self, line_start: int, line_end: int) -> Optional[QMenu]:
        """Split out from `mousePressEvent` so tests can inspect/trigger the
        picker without ever calling the blocking `.exec()`. Widens
        `[line_start, line_end)` out to the clicked clip's full extent first
        (a label represents the whole clip, not just the one line it happens
        to be painted next to)."""
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
    """The transcript panel's editor (see docks/transcript_dock.py). Keeps
    `app.document` in sync with every keystroke via `Document.replace_text`,
    and adds a Characters-menu/copy-paste/shorthand authoring path on top of
    plain text entry.
    """

    CHARACTER_ID_MIME_TYPE = "application/x-kokorogui-character-id"

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        # Ordinary typing rides Qt's own native undo/redo - see the module
        # docstring.
        self.setUndoRedoEnabled(True)
        self._suppress_contents_change = False
        # None outside of an active insertFromMimeData call; an int
        # accumulator while one is in progress, since a single paste can
        # fire more than one contentsChange signal (e.g. removing a prior
        # selection, then inserting) and the *net* chars added is what
        # assign_character_to_range needs.
        self._paste_chars_accumulator: Optional[int] = None
        # Guards against _on_selection_model_changed's own setTextCursor()
        # call bouncing straight back into _on_cursor_position_changed.
        self._updating_from_model = False

        self._highlighter = ClipHighlighter(self.document(), lambda: self.app.document)
        self.undo_coordinator = UndoCoordinator(
            self.document(), self.app.document.undo_stack, self._on_custom_stack_changed
        )

        # Left gutter - reserves its own width via setViewportMargins so it
        # scrolls/resizes in lockstep with the text (see resizeEvent).
        self._gutter = TranscriptGutter(self)
        self.setViewportMargins(GUTTER_WIDTH_PX, 0, 0, 0)

        # Split rules (UI2): boundaries as document offsets, recomputed on a
        # debounce after edits.
        self._split_boundaries: list = []
        self._split_timer = QTimer(self)
        self._split_timer.setSingleShot(True)
        self._split_timer.setInterval(SPLIT_RULE_DEBOUNCE_MS)
        self._split_timer.timeout.connect(self.refresh_split_rules)

        self._playing_clip_id: Optional[str] = None
        self._playing_clip_selections: list = []
        self._playing_word: Optional[tuple] = None

        self.document().contentsChange.connect(self._on_contents_change)
        self.cursorPositionChanged.connect(self._on_cursor_position_changed)
        self.app.selection.changed.connect(self._on_selection_model_changed)
        if hasattr(self.app.selection, "playingChanged"):
            self.app.selection.playingChanged.connect(self._on_playing_changed)
        if hasattr(self.app, "themeChanged"):
            self.app.themeChanged.connect(self._on_theme_changed)

        self.load_text(self.app.document.text)
        self._apply_theme_colors()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._gutter.setGeometry(0, 0, GUTTER_WIDTH_PX, self.height())

    # -- theme ---------------------------------------------------------------

    def _apply_theme_colors(self) -> None:
        pal = theme.current()
        self.setStyleSheet(f"QTextEdit {{ background: {pal.panel}; color: {pal.text}; }}")
        font = QFont(self.font())
        font.setPointSize(theme.EDITOR_FONT_POINT_SIZE)
        self.setFont(font)

    def _on_theme_changed(self) -> None:
        self._apply_theme_colors()
        self.rehighlight()
        self.viewport().update()

    # -- Document sync -----------------------------------------------------

    def _on_contents_change(self, position: int, chars_removed: int, chars_added: int) -> None:
        if self._paste_chars_accumulator is not None:
            self._paste_chars_accumulator += chars_added
        if self._suppress_contents_change:
            return
        new_text = self.toPlainText()
        self.app.document.replace_text(position, chars_removed, chars_added, new_text)
        # The highlighter's own contentsChange slot ran before this one (it
        # connected first, at construction) against the pre-edit run list -
        # re-paint the touched blocks now that the run list caught up.
        self._highlighter.invalidate_dirty()
        first = self.document().findBlock(position)
        last = self.document().findBlock(position + max(chars_added, 0))
        block = first
        while block.isValid():
            self._highlighter.rehighlightBlock(block)
            if block == last:
                break
            block = block.next()
        self._split_timer.start()
        self.app.schedule_save()
        self.app.refresh_timeline()

    def load_text(self, text: str) -> None:
        """Sets the editor's text without treating it as a user edit -
        `app.document.replace_text` is not called. Used at construction to
        seed from `app.document.text`, and by "load a different project"."""
        # A reload isn't the user moving the caret: it mustn't select the
        # clip that happens to sit at offset 0 (a subproject's line would
        # move the docks into it).
        self._suppress_contents_change = True
        was_updating, self._updating_from_model = self._updating_from_model, True
        try:
            self.setPlainText(text)
        finally:
            self._suppress_contents_change = False
            self._updating_from_model = was_updating
        self.rehighlight()

    def rebind_document(self) -> None:
        """After `app.document` was swapped for another `Document` (File >
        New/Open): reload the text, point the undo coordinator at the new
        stack, repaint."""
        self.undo_coordinator = UndoCoordinator(
            self.document(), self.app.document.undo_stack, self._on_custom_stack_changed
        )
        self.document().clearUndoRedoStacks()
        self.load_text(self.app.document.text)

    def rehighlight(self) -> None:
        """Repaints every block's highlight (and the gutter's labels) from
        `app.document.runs` - the one entry point every tagging operation
        calls after mutating the document, and what a custom-stack
        undo/redo calls too."""
        self._highlighter.rehighlight()
        self._gutter.update()
        self.refresh_split_rules()

    def _on_custom_stack_changed(self) -> None:
        """Called by `undo_coordinator` after every custom-stack undo/redo -
        a custom-stack action changed `app.document.runs`/`clips` directly,
        with no signal Qt can observe on its own. A command that changed the
        text (a sub-range replace, a reorder, New Subproject) reloads it."""
        if self.toPlainText() != self.app.document.text:
            self.load_text(self.app.document.text)
        self.rehighlight()
        self.app.schedule_save()
        self.app.refresh_timeline()

    def _push_assign_character(self, start: int, end: int, character_id) -> None:
        """Shared tail end of every character-assignment authoring path
        (Characters menu, gutter picker, header combo, paste-splitting, the
        `[Speaker:FX]:` shorthand). A range over a subproject's placeholder
        line is refused."""
        if self.app.document.overlaps_nested(start, end):
            self.app.set_status("A subproject's line can't be assigned a character.", "warning")
            return
        self.app.document.undo_stack.push(AssignCharacterCommand(start, end, character_id))
        self.rehighlight()
        self.app.schedule_save()
        self.app.refresh_timeline()

    # -- split rules (UI2) ---------------------------------------------------

    def split_boundaries(self) -> list:
        return list(self._split_boundaries)

    def refresh_split_rules(self) -> None:
        daw_doc = self.app.document
        text_len = len(daw_doc.text)
        boundaries = set()
        for clip in daw_doc.clips:
            extent = daw_doc.clip_extent(clip.id)
            if extent is not None:
                boundaries.add(extent[0])
                boundaries.add(extent[1])
        try:
            triples, _unmatched = plan_auto_split_clips(
                daw_doc, split_by_paragraph=bool(self.app.settings.get("auto_split_by_paragraph", False))
            )
        except Exception:
            triples = []
        for start, end, _cid in triples:
            boundaries.add(start)
            boundaries.add(end)
        boundaries.discard(0)
        boundaries.discard(text_len)
        self._split_boundaries = sorted(b for b in boundaries if 0 < b < text_len)
        self.viewport().update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().paintEvent(event)
        if not self._split_boundaries:
            return
        pal = theme.current()
        painter = QPainter(self.viewport())
        pen = QPen(QColor(pal.split_rule), 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        width = self.viewport().width()
        text = self.toPlainText()
        for offset in self._split_boundaries:
            cursor = QTextCursor(self.document())
            cursor.setPosition(min(offset, len(text)))
            rect = self.cursorRect(cursor)
            if rect.bottom() < 0 or rect.top() > self.viewport().height():
                continue
            at_line_start = offset == 0 or text[offset - 1] == "\n"
            if at_line_start:
                y = rect.top()
                painter.drawLine(0, y, width, y)
            else:
                # Mid-line boundary: a short vertical tick at the caret x.
                painter.drawLine(rect.left(), rect.top(), rect.left(), rect.bottom())

    # -- playing clip (UI4) --------------------------------------------------

    def _on_playing_changed(self) -> None:
        clip_id = self.app.selection.playing_clip_id
        if clip_id == self._playing_clip_id:
            return
        self._playing_clip_id = clip_id
        selections = []
        if clip_id is not None:
            extent = self.app.document.clip_extent(clip_id)
            clip = self.app.document.get_clip(clip_id)
            if extent is not None and clip is not None:
                character = self.app.document.get_character(clip.character_id)
                color = QColor(character.highlight_color) if character else QColor(theme.current().playing_highlight)
                color.setAlpha(110)
                sel = QTextEdit.ExtraSelection()
                sel.cursor = QTextCursor(self.document())
                sel.cursor.setPosition(extent[0])
                sel.cursor.setPosition(extent[1], QTextCursor.MoveMode.KeepAnchor)
                sel.format.setBackground(color)
                selections.append(sel)
                self._scroll_offset_into_view(extent[0])
        self._playing_clip_selections = selections
        self._playing_word = None
        self._apply_extra_selections()

    def set_playing_word(self, span: Optional[tuple]) -> None:
        """Highlights document offsets `[start, end)` as the word being
        played, or clears it with None."""
        span = tuple(span) if span else None
        if span == self._playing_word:
            return
        self._playing_word = span
        self._apply_extra_selections()

    def playing_word(self) -> Optional[tuple]:
        return self._playing_word

    def _apply_extra_selections(self) -> None:
        selections = list(self._playing_clip_selections)
        if self._playing_word is not None:
            text_len = len(self.toPlainText())
            start, end = (max(0, min(v, text_len)) for v in self._playing_word)
            if end > start:
                color = QColor(theme.current().playing_highlight)
                color.setAlpha(220)
                sel = QTextEdit.ExtraSelection()
                sel.cursor = QTextCursor(self.document())
                sel.cursor.setPosition(start)
                sel.cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
                sel.format.setBackground(color)
                sel.format.setFontUnderline(True)
                selections.append(sel)
        self.setExtraSelections(selections)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # A placeholder line's first click put its subproject in the docks;
        # the double-click enters it (NP6).
        if hasattr(self.app, "enter_recently_selected_subproject") and self.app.enter_recently_selected_subproject():
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().mousePressEvent(event)
        if (event.button() == Qt.MouseButton.LeftButton
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            offset = self.cursorForPosition(event.position().toPoint()).position()
            self.app.seek_to_offset(offset)

    def _scroll_offset_into_view(self, offset: int) -> None:
        cursor = QTextCursor(self.document())
        cursor.setPosition(max(0, min(offset, len(self.toPlainText()))))
        rect = self.cursorRect(cursor)
        bar = self.verticalScrollBar()
        viewport_h = self.viewport().height()
        if rect.top() < 0:
            bar.setValue(bar.value() + rect.top() - 8)
        elif rect.bottom() > viewport_h:
            bar.setValue(bar.value() + rect.bottom() - viewport_h + 8)

    # -- Selection sync -------------------------------------------------------

    def _on_cursor_position_changed(self) -> None:
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
            return
        if start <= cursor.position() < end and not cursor.hasSelection():
            return  # the caret already sits inside this clip - leave it alone

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

    # -- current target for the header combos ---------------------------------

    def current_target_range(self) -> Optional[tuple]:
        """The `[start, end)` a header-combo change applies to: the text
        selection if there is one, else the caret's whole clip, else the
        caret's line. None for an empty document."""
        cursor = self.textCursor()
        if cursor.hasSelection():
            return (cursor.selectionStart(), cursor.selectionEnd())
        clip = self.app.document.clip_covering(cursor.position())
        if clip is not None:
            return self.app.document.clip_extent(clip.id)
        block = cursor.block()
        start = block.position()
        end = start + len(block.text())
        return (start, end) if end > start else None

    def current_clip(self):
        cursor = self.textCursor()
        return self.app.document.clip_covering(cursor.selectionStart())

    # -- Undo/redo coordination + [Speaker:FX]: shorthand recognition -------

    def edit_touches_placeholder(self, start: int, end: int, inserting: bool = False) -> bool:
        """True when an edit over `[start, end)` would change part of a
        subproject's placeholder line (phase 4): cutting into it, or typing
        strictly inside it. Removing a whole placeholder is allowed (the
        subproject leaves the parent)."""
        for run, r_start, r_end in self.app.document._iter_runs_with_offsets():
            if run.kind != "placeholder":
                continue
            if inserting and end == start and r_start < start < r_end:
                return True
            if end > start and r_start < end and r_end > start and not (start <= r_start and r_end <= end):
                return True
        return False

    def _key_edit_range(self, event):
        """`(start, end, inserting)` the key would edit, or None for a key
        that edits nothing."""
        cursor = self.textCursor()
        if event.matches(QKeySequence.StandardKey.Copy) or event.matches(QKeySequence.StandardKey.SelectAll):
            return None
        start, end = cursor.selectionStart(), cursor.selectionEnd()
        if event.key() == Qt.Key.Key_Backspace:
            return (start, end, False) if end > start else (max(0, start - 1), start, False)
        if event.key() == Qt.Key.Key_Delete:
            return (start, end, False) if end > start else (start, start + 1, False)
        if event.matches(QKeySequence.StandardKey.Cut):
            return (start, end, False)
        if event.text() or event.matches(QKeySequence.StandardKey.Paste):
            return (start, end, end == start)
        return None

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.matches(QKeySequence.StandardKey.Undo):
            self.undo_coordinator.undo()
            event.accept()
            return
        if event.matches(QKeySequence.StandardKey.Redo):
            self.undo_coordinator.redo()
            event.accept()
            return
        edit_range = self._key_edit_range(event)
        if edit_range is not None and self.edit_touches_placeholder(*edit_range):
            self.app.set_status("A subproject's line is read-only; select it to edit the subproject.", "warning")
            event.accept()
            return

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

    def createMimeDataFromSelection(self) -> QMimeData:  # noqa: N802 (Qt override)
        # A plain QMimeData, not the QTextEditMimeData super() returns: that
        # private subclass reports a fixed formats() list while it still
        # holds its fragment, so a custom setData() is invisible to
        # hasFormat() for an in-process paste or drag. Copying the text and
        # HTML over keeps ordinary paste targets working.
        source = super().createMimeDataFromSelection()
        mime = QMimeData()
        mime.setText(source.text())
        if source.hasHtml():
            mime.setHtml(source.html())
        if not self.app.settings.get("character_fx_copy", True):
            return mime
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
        if self.edit_touches_placeholder(cursor.selectionStart(), cursor.selectionEnd(),
                                         inserting=not cursor.hasSelection()):
            self.app.set_status("A subproject's line is read-only; select it to edit the subproject.", "warning")
            return

        self._paste_chars_accumulator = 0
        try:
            super().insertFromMimeData(source)
        finally:
            chars_added = self._paste_chars_accumulator
            self._paste_chars_accumulator = None

        splits_enabled = self.app.settings.get("character_fx_paste_splits", True)
        if source_character_id and splits_enabled and chars_added:
            self._push_assign_character(insert_position, insert_position + chars_added, source_character_id)
