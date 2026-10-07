"""Edit > Remove Filler Words (plan 32).

Lists every filler `daw/fillers.find_fillers` found in the recordings, each
with its surrounding words, a checkbox and a Play button. The sounds no
sentence needs ("um", "uh") start checked; the phrases that are also real
English ("like", "you know") start unchecked, since only the listener can tell.
The dialog only chooses. `QtTTSApp.remove_filler_words` deletes the checked
ranges through `TranscriptEditor.delete_ranges`, so the cut is one undo step.
"""
from __future__ import annotations

import html
from typing import Optional

import playback
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
)

from kokoro_gui.daw import fillers
from kokoro_gui.daw.fillers import SAFE, Filler

# Characters of the transcript shown each side of a filler.
CONTEXT_CHARS = 36
# Seconds of the recording played before and after the filler, so the ear
# hears the word in its sentence.
PLAY_MARGIN_S = 0.3


class FillerDialog(QDialog):
    def __init__(self, document, hits: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Remove Filler Words")
        self.resize(640, 440)
        self._document = document
        self._hits: list = list(hits)
        self.play_buttons: list = []

        layout = QVBoxLayout(self)
        intro = QLabel("Check the fillers to cut. Each cut takes its audio with it, and one Undo "
                       "gives them all back. Phrases that are also real words, like \"like\" "
                       "and \"you know\", start unchecked.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.table = QTableWidget(len(self._hits), 3, self)
        self.table.setHorizontalHeaderLabels(["Filler", "In the transcript", ""])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(1, self.table.horizontalHeader().ResizeMode.Stretch)
        for row, hit in enumerate(self._hits):
            item = QTableWidgetItem(hit.text)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if hit.kind == SAFE else Qt.CheckState.Unchecked)
            self.table.setItem(row, 0, item)
            label = QLabel(self._context_html(hit))
            label.setTextFormat(Qt.TextFormat.RichText)
            self.table.setCellWidget(row, 1, label)
            button = QToolButton()
            button.setText("Play")
            button.setToolTip("Play the filler with a moment of the recording around it")
            button.setEnabled(self._source_of(hit) is not None)
            button.clicked.connect(lambda _checked=False, i=row: self.play(i))
            self.table.setCellWidget(row, 2, button)
            self.play_buttons.append(button)
        self.table.itemChanged.connect(lambda _item: self._sync())
        layout.addWidget(self.table, 1)

        row = QHBoxLayout()
        self.check_all_button = QPushButton("Check all")
        self.check_all_button.clicked.connect(lambda: self.set_all(True))
        self.check_none_button = QPushButton("Uncheck all")
        self.check_none_button.clicked.connect(lambda: self.set_all(False))
        row.addWidget(self.check_all_button)
        row.addWidget(self.check_none_button)
        row.addStretch(1)
        layout.addLayout(row)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
                                        self)
        self.remove_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._sync()

    # -- rows ------------------------------------------------------------------

    def _context_html(self, hit: Filler) -> str:
        text = self._document.text
        extent = self._document.clip_extent(hit.clip_id) or (0, len(text))
        lo, hi = max(extent[0], hit.word_start - CONTEXT_CHARS), min(extent[1], hit.word_end + CONTEXT_CHARS)
        before = html.escape(text[lo:hit.word_start])
        word = html.escape(text[hit.word_start:hit.word_end])
        after = html.escape(text[hit.word_end:hi])
        return f"{'...' if lo > extent[0] else ''}{before}<b>{word}</b>{after}{'...' if hi < extent[1] else ''}"

    def _source_of(self, hit: Filler) -> Optional[tuple]:
        """`(path, start_s, end_s)` Play reads, or None without a timed word
        or a recording file on disk."""
        span = fillers.timed_span(self._document, hit)
        if span is None:
            return None
        source, start_s, end_s = span
        path = (self._document.sources.get(source) or {}).get("path")
        return (path, start_s, end_s) if path else None

    def row_count(self) -> int:
        return len(self._hits)

    def hits(self) -> list:
        return list(self._hits)

    def is_checked(self, row: int) -> bool:
        return self.table.item(row, 0).checkState() == Qt.CheckState.Checked

    def set_checked(self, row: int, checked: bool) -> None:
        self.table.item(row, 0).setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)

    def set_all(self, checked: bool) -> None:
        for row in range(len(self._hits)):
            self.set_checked(row, checked)

    def checked_fillers(self) -> list:
        """The fillers whose box is ticked, in text order."""
        return [hit for row, hit in enumerate(self._hits) if self.is_checked(row)]

    def _sync(self) -> None:
        count = len(self.checked_fillers())
        self.remove_button.setText(f"Remove {count}" if count else "Remove")
        self.remove_button.setEnabled(count > 0)

    def play(self, row: int) -> bool:
        """Plays the filler's slice of the recording, with `PLAY_MARGIN_S`
        either side. False when there is nothing to play."""
        source = self._source_of(self._hits[row])
        if source is None:
            return False
        path, start_s, end_s = source
        playback.play_range(path, max(0.0, start_s - PLAY_MARGIN_S), end_s + PLAY_MARGIN_S)
        return True
