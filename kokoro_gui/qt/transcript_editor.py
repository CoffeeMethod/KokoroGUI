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
  (`ClipHighlighter`'s second pass). A dirty clip with a duration target
  whose text reads longer than the target (`fit.reading_rate_ratio`) gets a
  wavy amber or red one instead, so a dub line is seen to be too long while
  it is typed.
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

import bisect
import json
import os
import re
from typing import Callable, Optional

from PySide6.QtCore import QEvent, QMimeData, QRect, QSize, Qt, QTimer
from PySide6.QtGui import (
    QColor, QDragLeaveEvent, QFont, QKeySequence, QPainter, QPen, QPolygon, QSyntaxHighlighter, QTextCharFormat,
    QTextCursor,
)
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QMenu, QTextEdit, QToolTip, QWidget

from kokoro_gui.daw import fit as fit_ops
from kokoro_gui.daw import imported
from kokoro_gui.daw.auto_split import plan_auto_split_clips
from kokoro_gui.daw.undo import ApplyWordsCommand, AssignCharacterCommand, TextEditCommand
from kokoro_gui.qt import project as project_io
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
# Alpha of the underline under imported recording words (phase 5 P3): faint,
# so it reads as "this carries audio" without competing with the dirty one.
IMPORTED_UNDERLINE_ALPHA = 110
# A clipboard's timed-words payload bigger than this is ignored.
MAX_WORDS_PAYLOAD_BYTES = 16 * 1024 * 1024
UNTIMED_TOOLTIP = "No character: this text has no audio. Assign a character to generate it."


def _is_word_entry(word) -> bool:
    """Whether a pasted `words` entry has the `Run.words` shape:
    `[char_start, char_end, source, start_s, end_s]` with a source name
    string and numbers elsewhere. `apply_words` checks the values."""
    if not isinstance(word, (list, tuple)) or len(word) != 5:
        return False
    if not isinstance(word[2], str) or not word[2]:
        return False
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (word[0], word[1], word[3], word[4]))


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
    of every dirty clip (wave-underlines one that reads past its duration
    target, `rate_levels()`). `dirty_ids()` is computed once per rehighlight
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
        self._rate_levels: Optional[dict] = None
        self._word_spans: Optional[list] = None
        self._untimed_gaps: Optional[list] = None

    def invalidate_dirty(self) -> None:
        self._dirty_ids = None
        self._rate_levels = None
        self._word_spans = None
        self._untimed_gaps = None

    def word_spans(self) -> list:
        """`imported.word_spans` of the document, once per cycle: the words
        that carry recorded audio, underlined faintly."""
        if self._word_spans is None:
            daw_doc = self._daw_document_provider()
            self._word_spans = imported.word_spans(daw_doc) if daw_doc is not None else []
        return self._word_spans

    def untimed_gaps(self) -> list:
        """`imported.untimed_gaps` of the document, once per cycle: text
        typed into a recording, drawn greyed."""
        if self._untimed_gaps is None:
            daw_doc = self._daw_document_provider()
            self._untimed_gaps = imported.untimed_gaps(daw_doc) if daw_doc is not None else []
        return self._untimed_gaps

    def rate_levels(self) -> dict:
        """`{clip_id: "over" | "far_over"}` for each dirty clip whose text,
        read at its speed and learned pace, runs past its duration target.
        A clean clip's real length is on the timeline instead."""
        if self._rate_levels is None:
            self._rate_levels = {}
            daw_doc = self._daw_document_provider()
            clips = [c for c in getattr(daw_doc, "clips", None) or []
                     if fit_ops.TARGET_KEY in (c.overrides or {}) and c.id in self.dirty_ids()]
            if clips:
                try:
                    rates = fit_ops.speaking_rates(daw_doc)
                    for clip in clips:
                        level = fit_ops.fit_level(fit_ops.reading_rate_ratio(daw_doc, clip, rates))
                        if level in ("over", "far_over"):
                            self._rate_levels[clip.id] = level
                except Exception:
                    self._rate_levels = {}
        return self._rate_levels

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
        rate_levels = self.rate_levels()
        pal = theme.current()
        underline_color = QColor(pal.dirty_underline)

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
                # A subproject's line (phase 4) or a music bed's (phase 5
                # P2): its title or file name, read-only.
                fmt.setBackground(QColor(theme.current().panel_alt))
                fmt.setFontItalic(True)
            elif character is not None:
                tint = QColor(character.highlight_color)
                tint.setAlpha(HIGHLIGHT_ALPHA)
                fmt.setBackground(tint)
            if clip is not None and clip.id in rate_levels:
                fmt.setUnderlineStyle(QTextCharFormat.UnderlineStyle.WaveUnderline)
                fmt.setUnderlineColor(QColor(pal.fit_far_over if rate_levels[clip.id] == "far_over"
                                             else pal.fit_over))
            elif clip is not None and clip.id in dirty:
                fmt.setUnderlineStyle(QTextCharFormat.UnderlineStyle.DashUnderline)
                fmt.setUnderlineColor(underline_color)
            self.setFormat(lo, hi - lo, fmt)

        # Imported recording text (phase 5 P3): a faint underline under
        # every word that carries audio, and text typed into a recording
        # (no character, no audio) greyed. Both go over the run's format.
        faint = QColor(pal.text_muted)
        faint.setAlpha(IMPORTED_UNDERLINE_ALPHA)
        for lo, hi in self._spans_in(self.word_spans(), block_start, block_end):
            fmt = QTextCharFormat(self.format(lo))
            if fmt.underlineStyle() == QTextCharFormat.UnderlineStyle.NoUnderline:
                fmt.setUnderlineStyle(QTextCharFormat.UnderlineStyle.SingleUnderline)
                fmt.setUnderlineColor(faint)
                self.setFormat(lo, hi - lo, fmt)
        for lo, hi in self._spans_in(self.untimed_gaps(), block_start, block_end):
            fmt = QTextCharFormat(self.format(lo))
            fmt.setForeground(QColor(pal.text_muted))
            self.setFormat(lo, hi - lo, fmt)

    @staticmethod
    def _spans_in(spans: list, block_start: int, block_end: int):
        """`(lo, hi)` offsets inside the block of each sorted
        `(doc_start, doc_end)` span that overlaps it."""
        index = bisect.bisect_left(spans, (block_start, block_start))
        # A span starting before the block may still reach into it.
        index = max(0, index - 1)
        for start, end in spans[index:]:
            if start >= block_end:
                break
            lo, hi = max(start, block_start) - block_start, min(end, block_end) - block_start
            if hi > lo:
                yield lo, hi


def placeholder_label(clip) -> str:
    """The gutter label on a placeholder line: "Subproject" for a nested
    clip, "Audio" for a music bed."""
    return "Audio" if getattr(clip, "is_bed", False) else "Subproject"


class TranscriptGutter(QWidget):
    """Left gutter beside the transcript editor (TE3, reshaped by UI3).
    Shows the character name and, on a second line, the resolved FX preset
    wherever the `(character, fx)` pair changes from the previous line,
    and a small play button beside each dirty clip's first visible line.
    Labels open a character picker for the clicked clip; the button runs a
    scoped Generate for that one clip.

    Imported recordings (phase 5 P3): a recording clip is never stale, so it
    gets a play-only button instead (outlined, where Generate's is filled)
    that plays the timeline from the clip's start (`app.play_clip`). A line
    holding text typed into a recording (`imported.untimed_gaps`) gets a
    small hollow circle: that text has no character and no audio.

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
        self._play_rects: list = []  # [(QRect, clip_id)]
        self._mark_rects: list = []  # [(QRect, line_start, line_end)]
        self.setMouseTracking(True)
        editor.verticalScrollBar().valueChanged.connect(lambda _value: self.update())
        editor.textChanged.connect(self.update)

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        return QSize(GUTTER_WIDTH_PX, 0)

    def _label_key(self, daw_doc, clip):
        if clip is None:
            return (None, None)
        if clip.is_nested:
            return ("nested", clip.id)
        if clip.is_bed:
            return ("bed", clip.id)
        return (clip.character_id, clip_fx_name(daw_doc, clip))

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        pal = theme.current()
        painter = QPainter(self)
        painter.fillRect(event.rect(), QColor(pal.gutter_bg))
        self._label_rects = []
        self._button_rects = []
        self._play_rects = []
        self._mark_rects = []

        daw_doc = self.editor.app.document
        qt_doc = self.editor.document()
        layout = qt_doc.documentLayout()
        scroll = self.editor.verticalScrollBar().value()
        dirty_ids = {c.id for c in daw_doc.dirty_clips()}
        labelled_dirty: set = set()
        recording_ids = {c.id for c in daw_doc.clips
                         if imported.is_recording_clip(c) and any(s.audio_path for s in c.segments)}
        gaps = imported.untimed_gaps(daw_doc)

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
                if clip is not None and (clip.id in dirty_ids or clip.id in recording_ids):
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

            if key != previous_key and clip is not None and clip.has_placeholder:
                # A subproject's line (phase 4) or a music bed's (phase 5
                # P2): labelled, no picker, and never a play button (a bed
                # is never stale).
                name_rect = QRect(4, top, text_right - 4, metrics_h)
                painter.setFont(base_font)
                painter.setPen(QColor(pal.gutter_text))
                painter.drawText(name_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                                 placeholder_label(clip))
            elif key != previous_key and character is not None:
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
            elif clip is not None and clip.id in recording_ids and clip.id not in labelled_dirty:
                labelled_dirty.add(clip.id)
                btn = QRect(self.width() - GUTTER_BUTTON_PX - 4, top + max(0, (min(line_h, metrics_h) - GUTTER_BUTTON_PX) // 2),
                            GUTTER_BUTTON_PX, GUTTER_BUTTON_PX)
                self._draw_play_button(painter, btn, pal, outline=True)
                self._play_rects.append((btn, clip.id))

            if any(g_start < line_end and g_end > line_start for g_start, g_end in gaps):
                mark = QRect(self.width() - 2 * GUTTER_BUTTON_PX - 8,
                             top + max(0, (min(line_h, metrics_h) - GUTTER_BUTTON_PX) // 2),
                             GUTTER_BUTTON_PX, GUTTER_BUTTON_PX)
                self._draw_untimed_mark(painter, mark, pal)
                self._mark_rects.append((mark, line_start, line_end))

            previous_key = key
            block = block.next()

    @staticmethod
    def _draw_play_button(painter: QPainter, rect: QRect, pal, outline: bool = False) -> None:
        """Generate's filled triangle, or with `outline` a recording's
        play-only one in the gutter's text color."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = QColor(pal.gutter_text if outline else pal.dirty_underline)
        painter.setPen(QPen(color, 1))
        if outline:
            painter.setBrush(Qt.BrushStyle.NoBrush)
        else:
            painter.setBrush(color)
        tri = QPolygon([
            QPoint(rect.left() + 4, rect.top() + 3),
            QPoint(rect.right() - 3, rect.center().y()),
            QPoint(rect.left() + 4, rect.bottom() - 2),
        ])
        painter.drawPolygon(tri)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

    @staticmethod
    def _draw_untimed_mark(painter: QPainter, rect: QRect, pal) -> None:
        """The "no character" mark: a small hollow circle."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor(pal.text_muted), 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(rect.center(), 4, 4)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

    def button_rects(self) -> list:
        return list(self._button_rects)

    def play_rects(self) -> list:
        """`(QRect, clip_id)` of each recording clip's play-only button."""
        return list(self._play_rects)

    def mark_rects(self) -> list:
        """`(QRect, line_start, line_end)` of each "no character" mark."""
        return list(self._mark_rects)

    def tooltip_at(self, pos) -> Optional[str]:
        """The source text of the clip whose label is at `pos`, if any, or
        what a "no character" mark means."""
        for rect, _line_start, _line_end in self._mark_rects:
            if rect.contains(pos):
                return UNTIMED_TOOLTIP
        for rect, _clip_id in self._play_rects:
            if rect.contains(pos):
                return "Play this recording"
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
        for rect, clip_id in self._play_rects:
            if rect.contains(pos):
                self.editor.app.play_clip(clip_id)
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
        # True while a paste of timed text is inserted: every text change
        # then goes through `TextEditCommand`, so the insert and its
        # `ApplyWordsCommand` undo as one step (`push_joined`).
        self._joined_edit = False
        # Guards against _on_selection_model_changed's own setTextCursor()
        # call bouncing straight back into _on_cursor_position_changed.
        self._updating_from_model = False

        self._highlighter = ClipHighlighter(self.document(), lambda: self.app.document)
        self.undo_coordinator = UndoCoordinator(
            self.document(), self.app.document.undo_stack, self._on_custom_stack_changed, self._run_joined
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
            self.undo_coordinator.edit_seen()
            return
        new_text = self.toPlainText()
        document = self.app.document
        if self._joined_edit or document.edit_touches_imported(position, chars_removed):
            # Imported recording text (phase 5 P3): Qt's native undo would
            # give back the characters but not their word timing, so the
            # edit is also a `TextEditCommand`, joined to the native step.
            self.undo_coordinator.push_joined(TextEditCommand(position, chars_removed, chars_added, new_text))
        else:
            document.replace_text(position, chars_removed, chars_added, new_text)
        self.undo_coordinator.edit_seen()
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
        # setPlainText cleared Qt's undo history.
        if getattr(self, "undo_coordinator", None) is not None:
            self.undo_coordinator.native_history_cleared()
        self.rehighlight()

    def rebind_document(self) -> None:
        """After `app.document` was swapped for another `Document` (File >
        New/Open): reload the text, point the undo coordinator at the new
        stack, repaint."""
        self.undo_coordinator = UndoCoordinator(
            self.document(), self.app.document.undo_stack, self._on_custom_stack_changed, self._run_joined
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
        # A subtitle import adds characters in the same step; its undo
        # takes them away again.
        dock = getattr(self.app, "transcript_dock", None)
        if dock is not None:
            dock.refresh_character_choices()
        self.app.schedule_save()
        self.app.refresh_timeline()

    def _run_joined(self, step) -> None:
        """Undo or redo of a joined step (`UndoCoordinator.push_joined`):
        `step` moves Qt's text and the custom stack together, with the
        document sync off since the custom commands restore the runs. Text
        Qt coalesced into the same native step after the joined edit (a
        few more keystrokes) reaches the document afterwards
        (`_align_document_text`)."""
        self._suppress_contents_change = True
        try:
            step()
        finally:
            self._suppress_contents_change = False
        self._align_document_text()
        self._on_custom_stack_changed()

    def _align_document_text(self) -> None:
        """Brings `app.document`'s text in line with the editor's by one
        `replace_text` over the span where they differ, as if typed."""
        old, new = self.app.document.text, self.toPlainText()
        if old == new:
            return
        prefix = 0
        limit = min(len(old), len(new))
        while prefix < limit and old[prefix] == new[prefix]:
            prefix += 1
        suffix = 0
        while suffix < limit - prefix and old[-1 - suffix] == new[-1 - suffix]:
            suffix += 1
        self.app.document.replace_text(prefix, len(old) - prefix - suffix, len(new) - prefix - suffix, new)

    def _push_assign_character(self, start: int, end: int, character_id) -> None:
        """Shared tail end of every character-assignment authoring path
        (Characters menu, gutter picker, header combo, paste-splitting, the
        `[Speaker:FX]:` shorthand). A range over a placeholder line (a
        subproject's or a music bed's) is refused."""
        if self.app.document.overlaps_nested(start, end):
            self.app.set_status("A subproject's or an audio file's line can't be assigned a character.",
                                "warning")
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
        placeholder line (a subproject's, phase 4, or a music bed's, phase
        5 P2): cutting into it, or typing strictly inside it. Removing a
        whole placeholder is allowed (the subproject or the bed leaves the
        project)."""
        return self._placeholder_touched(start, end, inserting) is not None

    def _placeholder_touched(self, start: int, end: int, inserting: bool = False):
        """The clip whose placeholder line `edit_touches_placeholder` found,
        or None."""
        document = self.app.document
        for run, r_start, r_end in document._iter_runs_with_offsets():
            if run.kind != "placeholder":
                continue
            if inserting and end == start and r_start < start < r_end:
                return document.get_clip(run.clip_id) or run
            if end > start and r_start < end and r_end > start and not (start <= r_start and r_end <= end):
                return document.get_clip(run.clip_id) or run
        return None

    def _placeholder_status(self, start: int, end: int, inserting: bool = False) -> None:
        clip = self._placeholder_touched(start, end, inserting)
        if getattr(clip, "is_bed", False):
            self.app.set_status("An imported audio file's line is read-only; it shows the file's name.", "warning")
        else:
            self.app.set_status("A subproject's line is read-only; select it to edit the subproject.", "warning")

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
            self._placeholder_status(*edit_range)
            event.accept()
            return

        pending_line = None
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not self.textCursor().hasSelection():
            block = self.textCursor().block()
            pending_line = (block.position(), block.text())

        # A key that may edit imported recording text runs in an edit block
        # of its own, so Qt gives it a fresh native undo command rather than
        # merging it into the previous keystrokes', and the command it is
        # joined to (`_on_contents_change`) undoes exactly this edit. A
        # paste wraps itself (`insertFromMimeData`).
        wrap = (edit_range is not None and not event.matches(QKeySequence.StandardKey.Paste)
                and self._key_may_edit_imported(event, *edit_range))
        edit_block = QTextCursor(self.document()) if wrap else None
        if edit_block is not None:
            edit_block.beginEditBlock()
        try:
            super().keyPressEvent(event)
        finally:
            if edit_block is not None:
                edit_block.endEditBlock()

        if pending_line is not None:
            self._try_recognize_shorthand_line(*pending_line)

    def _key_may_edit_imported(self, event, start: int, end: int, inserting: bool) -> bool:
        """True when the key's edit (`_key_edit_range`) may change imported
        recording text. A Backspace or Delete with no selection can take a
        whole word (with Ctrl), so it counts when its line or a neighbour
        holds recording text."""
        document = self.app.document
        if inserting:
            return document.edit_touches_imported(start, 0)
        if event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete) and not self.textCursor().hasSelection():
            qt_doc = self.document()
            first, last = qt_doc.findBlock(start), qt_doc.findBlock(end)
            start = max(0, first.position() - 1)
            end = min(len(document.text), last.position() + last.length())
        return end > start and document.edit_touches_imported(start, end - start)

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
        cursor = self.textCursor()
        if cursor.hasSelection():
            # Imported recording words carry their timing (phase 5 P3).
            payload = imported.words_payload(self.app.document, cursor.selectionStart(), cursor.selectionEnd())
            if payload["words"]:
                mime.setData(imported.WORDS_MIME_TYPE, json.dumps(payload).encode("utf-8"))
        if not self.app.settings.get("character_fx_copy", True):
            return mime
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

    @staticmethod
    def words_from_mime(source: QMimeData) -> Optional[dict]:
        """The `imported.WORDS_MIME_TYPE` payload of a paste or drop as
        `{"words": list, "sources": dict}`, or None when there is none, it
        doesn't parse or no word has the right shape. Clipboard data is
        untrusted: malformed words and source entries that aren't dicts
        are left out here, and `apply_words` checks each word's values."""
        if not source.hasFormat(imported.WORDS_MIME_TYPE):
            return None
        raw = bytes(source.data(imported.WORDS_MIME_TYPE))
        if not raw or len(raw) > MAX_WORDS_PAYLOAD_BYTES:
            return None
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return None
        if not isinstance(data, dict) or not isinstance(data.get("words"), list):
            return None
        words = [list(w) for w in data["words"] if _is_word_entry(w)]
        if not words:
            return None
        sources = data.get("sources")
        sources = {str(k): v for k, v in sources.items() if isinstance(v, dict)} if isinstance(sources, dict) else {}
        return {"words": words, "sources": sources}

    def insertFromMimeData(self, source: QMimeData) -> None:  # noqa: N802 (Qt override)
        source_character_id = self._extract_source_character_id(source)
        cursor = self.textCursor()
        insert_position = cursor.selectionStart() if cursor.hasSelection() else cursor.position()
        if self.edit_touches_placeholder(cursor.selectionStart(), cursor.selectionEnd(),
                                         inserting=not cursor.hasSelection()):
            self._placeholder_status(cursor.selectionStart(), cursor.selectionEnd(),
                                     inserting=not cursor.hasSelection())
            return
        payload = self.words_from_mime(source)

        # One edit block: one native undo command and one contents change
        # for the whole paste, never merged into what was typed before.
        self._paste_chars_accumulator = 0
        joined_before, self._joined_edit = self._joined_edit, self._joined_edit or payload is not None
        edit_block = QTextCursor(self.document())
        edit_block.beginEditBlock()
        try:
            super().insertFromMimeData(source)
        finally:
            edit_block.endEditBlock()
            chars_added = self._paste_chars_accumulator
            self._paste_chars_accumulator = None
            self._joined_edit = joined_before

        if payload is not None and chars_added:
            if chars_added == len(source.text()):
                self._apply_pasted_words(insert_position, chars_added, payload, source_character_id)
            else:
                self.app.set_status("Pasted without timing: the text changed on the way in.", "warning")
            return

        splits_enabled = self.app.settings.get("character_fx_paste_splits", True)
        if source_character_id and splits_enabled and chars_added:
            self._push_assign_character(insert_position, insert_position + chars_added, source_character_id)

    def _apply_pasted_words(self, position: int, length: int, payload: dict, character_id) -> None:
        """Tags a paste of timed text as imported recording text
        (`ApplyWordsCommand`, joined to the paste's undo step). A source this
        project lacks is imported from the path the payload names when that
        file is still there (`_import_pasted_source`); words whose source
        can't be found lose their timing, which the status bar says. The
        new clip takes the copied text's character when this project has
        it."""
        document = self.app.document
        known = {name for name in document.sources if document.source_path(name)}
        added = {}
        for name, entry in payload["sources"].items():
            if name not in known:
                local = self._import_pasted_source(str(name), entry)
                if local is not None:
                    added[str(name)] = local
        words = payload["words"]
        usable = [w for w in words if _is_word_entry(w) and (w[2] in known or w[2] in added)]
        if not usable:
            self.app.set_status("Pasted without timing: the recording it came from isn't in this project.",
                                "warning")
            return
        if document.get_character(character_id) is None:
            character_id = None
        self.undo_coordinator.push_joined(ApplyWordsCommand(position, length, usable, added, character_id))
        if len(usable) < len(words):
            self.app.set_status("Some pasted words lost their timing: their recording isn't in this project.",
                                "warning")
        self.rehighlight()
        self.app.schedule_save()
        self.app.refresh_timeline()

    def _import_pasted_source(self, name: str, entry) -> Optional[dict]:
        """A pasted word's source (another project's recording) copied into
        this project (`project.import_audio_file`): its `Document.sources`
        entry, or None when the path is gone, isn't a file under the
        projects root, or its content isn't `name`."""
        path = entry.get("path") if isinstance(entry, dict) else None
        project_dir = self.app.project_dir
        if not isinstance(path, str) or not path or not project_dir:
            return None
        real = os.path.realpath(os.path.abspath(path))
        root = os.path.realpath(project_io.projects_root())
        if not real.startswith(root + os.sep) or not os.path.isfile(real):
            return None
        if os.path.splitext(os.path.basename(real))[0] != name:
            return None
        try:
            stored = project_io.import_audio_file(real, project_dir)
        except (OSError, project_io.ProjectError):
            return None
        source, local = imported.source_entry(stored)
        return local if source == name else None

    # -- Drag and drop -------------------------------------------------------

    def _is_own_drag(self, event) -> bool:
        return event.source() in (self, self.viewport())

    def dropEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """A drag inside the editor, or a drop of timed text, goes through
        the same path as cut and paste: the dragged text is removed first
        (its own undo step), then inserted with `insertFromMimeData`, so
        imported words keep their timing. Qt's own drop does both in one
        edit, which reads to the document as the whole stretch between
        the two places being retyped. Other drops are Qt's."""
        mime = event.mimeData()
        own = self._is_own_drag(event)
        if mime is None or not mime.hasText() or not (own or mime.hasFormat(imported.WORDS_MIME_TYPE)):
            super().dropEvent(event)
            return
        position = self.cursorForPosition(event.position().toPoint()).position()
        cursor = self.textCursor()
        start, end = cursor.selectionStart(), cursor.selectionEnd()
        moving = own and event.dropAction() == Qt.DropAction.MoveAction and end > start
        # Clears Qt's drop caret.
        super().dragLeaveEvent(QDragLeaveEvent())
        if moving and start <= position <= end:
            event.setDropAction(Qt.DropAction.IgnoreAction)
            event.accept()
            return
        if moving:
            if self.edit_touches_placeholder(start, end):
                self._placeholder_status(start, end)
                event.ignore()
                return
            removal = QTextCursor(self.document())
            removal.setPosition(start)
            removal.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            removal.removeSelectedText()
            if position >= end:
                position -= end - start
        target = QTextCursor(self.document())
        target.setPosition(max(0, min(position, len(self.toPlainText()))))
        self.setTextCursor(target)
        self.insertFromMimeData(mime)
        # The source must not delete the dragged text again.
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()
