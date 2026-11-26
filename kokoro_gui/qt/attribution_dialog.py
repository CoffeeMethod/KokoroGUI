"""Edit > Attribute Dialogue (plan 26).

Shows `daw/attribution.plan` for the project on screen: one row per quote
with the proposed character (a combo the reviewer can change), the
confidence and the reason. The dialog is modeless, so the transcript stays
usable while it is open: clicking a row selects the quote there, and every
planned quote carries a temporary highlight in its character's color, or in
the theme's warning color when the plan isn't sure. Apply pushes one
`CompositeCommand` of `AssignCharacterCommand`s, so one Undo takes it all
back. Closing the dialog any way clears the highlight.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from kokoro_gui.daw import attribution
from kokoro_gui.daw.undo import AssignCharacterCommand, CompositeCommand
from kokoro_gui.i18n import N_, ntr, tr
from kokoro_gui.qt import tips

# Characters of a quote shown in the Excerpt column.
EXCERPT_CHARS = 70
SCOPE_SELECTION = "selection"
SCOPE_ALL = "all"
CONFIDENCE_LABELS = {attribution.HIGH: N_("High"), attribution.MEDIUM: N_("Medium"), attribution.LOW: N_("Low")}


class AttributionDialog(QDialog):
    def __init__(self, app, document, editor, selection: Optional[tuple] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Attribute Dialogue"))
        self.setModal(False)
        self.resize(760, 520)
        self.app = app
        self._document = document
        self._editor = editor
        self._selection = selection if selection and selection[1] > selection[0] else None
        self._text = ""
        self._assignments: list = []
        self._rows: list = []
        self._chosen: list = []

        layout = QVBoxLayout(self)
        intro = QLabel(tr("Quotes with a speech tag go to the character it names. Untagged quotes in a two-person "
                          "exchange take turns. Check the rows marked Low, then Apply. One Undo takes it all back."))
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        self.narrator_combo = QComboBox()
        self.narrator_combo.addItem(tr("Leave narration alone"), None)
        for character in document.characters:
            self.narrator_combo.addItem(character.name, character.id)
        self.narrator_combo.setToolTip(tr("Who reads the text between the quotes, speech tags included."))
        form.addRow(tr("Narrator:"), self.narrator_combo)
        self.scope_combo = QComboBox()
        if self._selection is not None:
            self.scope_combo.addItem(tr("Selection"), SCOPE_SELECTION)
        self.scope_combo.addItem(tr("Whole transcript"), SCOPE_ALL)
        form.addRow(tr("Scope:"), self.scope_combo)
        layout.addLayout(form)

        self.only_uncertain = QCheckBox(tr("Only uncertain"))
        self.only_uncertain.setToolTip(tr("Show only the quotes the plan is least sure of."))
        layout.addWidget(self.only_uncertain)

        self.table = QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels([tr("Excerpt"), tr("Character"), tr("Confidence"), tr("Reason")])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, header.ResizeMode.Stretch)
        header.setSectionResizeMode(3, header.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, 1)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply
                                        | QDialogButtonBox.StandardButton.Cancel, self)
        self.apply_button = self.buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.apply_button.clicked.connect(self.apply)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.narrator_combo.currentIndexChanged.connect(lambda _i: self.replan())
        self.scope_combo.currentIndexChanged.connect(lambda _i: self.replan())
        self.only_uncertain.toggled.connect(lambda _on: self._apply_filter())
        self.table.currentCellChanged.connect(lambda row, _col, _prow, _pcol: self.reveal_row(row))
        self.replan()
        tips.label_rows(self)

    # -- planning ----------------------------------------------------------------

    def narrator_id(self) -> Optional[str]:
        return self.narrator_combo.currentData()

    def scope(self) -> tuple:
        if self.scope_combo.currentData() == SCOPE_SELECTION and self._selection is not None:
            return self._selection
        return 0, len(self._document.text)

    def replan(self) -> None:
        """Plans again from the transcript as it is now. Edits made to the
        rows are lost."""
        self._text = self._document.text
        start, end = self.scope()
        self._assignments = attribution.plan(self._text, self._document.characters, self.narrator_id(), start, end)
        self._rows = [a for a in self._assignments if a.is_quote]
        self._chosen = [a.character_id for a in self._rows]
        self._fill_table()
        self._apply_filter()
        self._update_marks()
        self._update_summary()

    def _fill_table(self) -> None:
        self.table.setRowCount(len(self._rows))
        for row, assignment in enumerate(self._rows):
            excerpt = " ".join(self._text[assignment.start:assignment.end].split())
            if len(excerpt) > EXCERPT_CHARS:
                excerpt = excerpt[:EXCERPT_CHARS - 3] + "..."
            self.table.setItem(row, 0, QTableWidgetItem(excerpt))
            combo = QComboBox()
            combo.setToolTip(tr("Who speaks this quote."))
            combo.addItem(tr("Nobody"), None)
            for character in self._document.characters:
                combo.addItem(character.name, character.id)
            combo.setCurrentIndex(max(0, combo.findData(assignment.character_id)))
            combo.currentIndexChanged.connect(lambda _i, r=row, c=combo: self.set_character(r, c.currentData()))
            self.table.setCellWidget(row, 1, combo)
            self.table.setItem(row, 2, QTableWidgetItem(tr(CONFIDENCE_LABELS.get(assignment.confidence, ""))))
            self.table.setItem(row, 3, QTableWidgetItem(assignment.reason))

    def _apply_filter(self) -> None:
        only = self.only_uncertain.isChecked()
        for row, assignment in enumerate(self._rows):
            self.table.setRowHidden(row, only and assignment.confidence != attribution.LOW)

    def _update_summary(self) -> None:
        count = len(self._rows)
        unsure = sum(1 for a in self._rows if a.confidence == attribution.LOW)
        narration = len(self._assignments) - count
        parts = [ntr("{n} quote, {unsure} uncertain.", "{n} quotes, {unsure} uncertain.", count, unsure=unsure)]
        if narration:
            name = self.narrator_combo.currentText()
            parts.append(ntr("{n} narration stretch goes to {name}.", "{n} narration stretches go to {name}.",
                             narration, name=name))
        self.summary.setText(" ".join(parts))
        self.apply_button.setEnabled(bool(self.planned_commands()))

    # -- rows --------------------------------------------------------------------

    def row_count(self) -> int:
        return len(self._rows)

    def rows(self) -> list:
        return list(self._rows)

    def chosen(self, row: int) -> Optional[str]:
        return self._chosen[row]

    def set_character(self, row: int, character_id: Optional[str]) -> None:
        """The reviewer's pick for `row`; Apply uses it."""
        self._chosen[row] = character_id
        combo = self.table.cellWidget(row, 1)
        if combo is not None and combo.currentData() != character_id:
            combo.setCurrentIndex(max(0, combo.findData(character_id)))
        self._update_marks()
        self.apply_button.setEnabled(bool(self.planned_commands()))

    def reveal_row(self, row: int) -> None:
        """Selects the row's quote in the transcript and scrolls to it."""
        if 0 <= row < len(self._rows) and self._editor is not None and self._document.text == self._text:
            assignment = self._rows[row]
            self._editor.select_span(assignment.start, assignment.end)

    def _update_marks(self) -> None:
        if self._editor is None:
            return
        marks = []
        for assignment, character_id in zip(self._rows, self._chosen):
            character = self._document.get_character(character_id)
            unsure = character is None or (character_id == assignment.character_id
                                           and assignment.confidence == attribution.LOW)
            marks.append((assignment.start, assignment.end, None if unsure else character.highlight_color))
        self._editor.set_attribution_marks(marks)

    # -- apply -------------------------------------------------------------------

    def planned_commands(self) -> list:
        """The `AssignCharacterCommand`s Apply pushes, in ascending start
        order: every row with a character, plus the narration."""
        ranges = [(a.start, a.end, cid) for a, cid in zip(self._rows, self._chosen) if cid is not None]
        ranges += [(a.start, a.end, a.character_id) for a in self._assignments
                   if not a.is_quote and a.character_id is not None]
        return [AssignCharacterCommand(s, e, cid) for s, e, cid in sorted(ranges)]

    def apply(self) -> bool:
        """Pushes the plan as one undo step and closes. When the transcript
        changed since the plan, or another project is on screen, plans
        again instead and returns False."""
        if self.app.document is not self._document:
            self.summary.setText(tr("Another project is on screen. Close this and open Attribute Dialogue there."))
            return False
        if self.app.is_busy():
            self.summary.setText(tr("Finish or cancel the current job first."))
            return False
        if self._document.text != self._text:
            self.replan()
            self.summary.setText(tr("The transcript changed, so the list was planned again. Check it and Apply."))
            return False
        commands = self.planned_commands()
        if not commands:
            return False
        self._document.undo_stack.push(CompositeCommand(commands))
        if self._editor is not None:
            self._editor.rehighlight()
        self.app.schedule_save()
        self.app.refresh_timeline()
        quotes = sum(1 for cid in self._chosen if cid is not None)
        self.app.set_status(ntr("Attributed {n} quote.", "Attributed {n} quotes.", quotes), "success")
        self.accept()
        return True

    def done(self, result) -> None:  # noqa: D401 (Qt override)
        if self._editor is not None:
            self._editor.clear_attribution_marks()
        super().done(result)
