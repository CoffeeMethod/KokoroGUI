"""Lexicon > Find words to check (plan 22).

Lists the words of the open projects' transcripts that an engine is likely to
misread (`daw/harvest.candidates`): names, acronyms, numbers and, with the
spelling dictionary, unknown words. Each row has a "Say it as" cell. Hear
speaks the word's context with that cell applied, through
`app.preview_text`; Add writes a whole-word rule to the lexicon; Add all
filled writes every row that has an answer in one change.

The dialog only chooses. Rules go in through `LexiconDock.add_rules`, which
saves, repaints and re-runs the stale check once for the whole batch.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from kokoro_gui.daw import harvest
from kokoro_gui.daw.harvest import ACRONYM, DIGITS, NAME, UNKNOWN, Candidate
from kokoro_gui.engine.lexicon import apply_lexicon, normalize_rules

# The rows shown, most frequent first. A book has thousands of numbers and
# unknown words; a row costs two buttons.
MAX_ROWS = 500
KIND_LABELS = {NAME: "Name", ACRONYM: "Acronym", DIGITS: "Number", UNKNOWN: "Unknown word"}
FILTER_LABELS = {NAME: "Names", ACRONYM: "Acronyms", DIGITS: "Numbers", UNKNOWN: "Unknown words"}
WORD_COL, COUNT_COL, KIND_COL, CONTEXT_COL, SAY_COL, HEAR_COL, ADD_COL = range(7)


class HarvestDialog(QDialog):
    def __init__(self, app, parent=None):
        super().__init__(parent or app)
        self.app = app
        self.setWindowTitle("Find words to check")
        self.resize(900, 520)
        self._say: dict = {}      # word -> what the user typed in "Say it as"
        self._added: set = set()  # words whose rule this dialog wrote
        self._shown: list = []    # the Candidate on each table row
        self._populating = False

        texts, names = self._transcripts()
        self.dictionary = self._load_dictionary()
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.candidates: list = harvest.candidates(
                "\n\n".join(texts), self.dictionary, lexicon=app.settings.get("lexicon"), ignore=names)
        finally:
            QGuiApplication.restoreOverrideCursor()

        layout = QVBoxLayout(self)
        intro = QLabel("Words an engine may say wrong, most frequent first. Type how each should be said, "
                       "press Hear to check it in its sentence, then Add. Each answer becomes a whole word "
                       "rule in the Lexicon; names and acronyms match case.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        filters = QHBoxLayout()
        self.kind_checks: dict = {}
        for kind in (NAME, ACRONYM, DIGITS, UNKNOWN):
            box = QCheckBox(FILTER_LABELS[kind])
            box.setChecked(True)
            if kind == UNKNOWN and self.dictionary is None:
                box.setChecked(False)
                box.setEnabled(False)
                box.setToolTip("Needs the spelling dictionary for this character's language "
                               "(the pyspellchecker package, English, Spanish, French, Italian or Portuguese).")
            box.toggled.connect(lambda _on: self._populate())
            filters.addWidget(box)
            self.kind_checks[kind] = box
        filters.addStretch(1)
        layout.addLayout(filters)

        self.table = QTableWidget(0, 7, self)
        self.table.setHorizontalHeaderLabels(["Word", "Count", "Kind", "In the transcript", "Say it as", "", ""])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                                   | QAbstractItemView.EditTrigger.EditKeyPressed
                                   | QAbstractItemView.EditTrigger.AnyKeyPressed)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(CONTEXT_COL, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(SAY_COL, QHeaderView.ResizeMode.Interactive)
        for col in (WORD_COL, COUNT_COL, KIND_COL, HEAR_COL, ADD_COL):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setColumnWidth(SAY_COL, 180)
        self.table.itemChanged.connect(self._item_changed)
        layout.addWidget(self.table, 1)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        bottom = QHBoxLayout()
        self.add_all_button = QPushButton("Add all filled")
        self.add_all_button.setToolTip("Add a rule for every row that has a Say it as answer.")
        self.add_all_button.clicked.connect(lambda _c=False: self.add_all_filled())
        bottom.addWidget(self.add_all_button)
        bottom.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        bottom.addWidget(buttons)
        layout.addLayout(bottom)
        self._populate()

    # -- what to scan --------------------------------------------------------

    def _transcripts(self) -> tuple:
        """`(transcripts, character names)` of every open project: the root
        and any open subprojects."""
        texts, names = [], []
        for project in self.app.open_projects():
            document = project.document
            texts.append(document.text)
            names.extend(character.name for character in document.characters)
        return texts, names

    def _load_dictionary(self):
        """The spelling dictionary for the active character's language, with
        the character names and the lexicon's words as known; None without
        one."""
        try:
            return self.app.spell_dictionary_for(self.app.active_character())
        except Exception:  # noqa: BLE001 - no spelling is a smaller dialog, not a failed one
            return None

    # -- the table -----------------------------------------------------------

    def _wanted(self, candidate: Candidate) -> bool:
        return any(self.kind_checks[kind].isChecked() for kind in candidate.kinds)

    def _populate(self) -> None:
        """Rebuilds the rows from the kind filters, keeping what was typed."""
        wanted = [c for c in self.candidates if self._wanted(c)]
        self._shown = wanted[:MAX_ROWS]
        self._populating = True
        try:
            self.table.clearContents()
            self.table.setRowCount(len(self._shown))
            for row, candidate in enumerate(self._shown):
                added = candidate.word in self._added
                cells = (
                    (WORD_COL, candidate.word),
                    (COUNT_COL, str(candidate.count)),
                    (KIND_COL, ", ".join(KIND_LABELS[k] for k in candidate.kinds)),
                    (CONTEXT_COL, candidate.context),
                )
                for col, text in cells:
                    item = QTableWidgetItem(text)
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    self.table.setItem(row, col, item)
                self.table.item(row, COUNT_COL).setTextAlignment(Qt.AlignmentFlag.AlignRight
                                                                 | Qt.AlignmentFlag.AlignVCenter)
                say = QTableWidgetItem(self._say.get(candidate.word, ""))
                flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                say.setFlags(flags if added else flags | Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, SAY_COL, say)
                hear = QPushButton("Hear")
                hear.setToolTip("Speak the sentence with this answer applied.")
                hear.clicked.connect(lambda _c=False, r=row: self.hear(r))
                self.table.setCellWidget(row, HEAR_COL, hear)
                add = QPushButton("Added" if added else "Add")
                add.setEnabled(not added)
                add.setToolTip("Add a whole word rule to the Lexicon.")
                add.clicked.connect(lambda _c=False, r=row: self.add(r))
                self.table.setCellWidget(row, ADD_COL, add)
        finally:
            self._populating = False
        self._sync()

    def _item_changed(self, item: QTableWidgetItem) -> None:
        if self._populating or item.column() != SAY_COL:
            return
        if 0 <= item.row() < len(self._shown):
            self._say[self._shown[item.row()].word] = item.text()
        self._sync()

    def _sync(self) -> None:
        total = sum(1 for c in self.candidates if self._wanted(c))
        if not self.candidates:
            self.summary.setText("Nothing to check: no names, acronyms or numbers the Lexicon doesn't cover.")
        elif total > len(self._shown):
            self.summary.setText(f"Showing the {len(self._shown)} most frequent of {total}.")
        else:
            self.summary.setText(f"{total} to check.")
        self.add_all_button.setEnabled(bool(self.filled_rows()))

    # -- rows ----------------------------------------------------------------

    def row_count(self) -> int:
        return self.table.rowCount()

    def words(self) -> list:
        """The word on each row, top to bottom."""
        return [c.word for c in self._shown]

    def say_as(self, row: int) -> str:
        return self._say.get(self._shown[row].word, "").strip()

    def set_say_as(self, row: int, text: str) -> None:
        """Types `text` into the row's Say it as cell."""
        self.table.item(row, SAY_COL).setText(text)

    def is_added(self, row: int) -> bool:
        return self._shown[row].word in self._added

    def filled_rows(self) -> list:
        """The rows with an answer that no rule was written for."""
        return [row for row in range(len(self._shown)) if self.say_as(row) and not self.is_added(row)]

    def _rule(self, row: int) -> dict:
        candidate = self._shown[row]
        return harvest.whole_word_rule(candidate.word, self.say_as(row), candidate.kinds)

    # -- actions -------------------------------------------------------------

    def hear(self, row: int) -> None:
        """Speaks the row's context. With an answer typed, the answer is
        applied after the rules already in the Lexicon, as generation would,
        and the saved rules are not applied a second time."""
        candidate = self._shown[row]
        saved = normalize_rules(self.app.settings.get("lexicon"))
        if not self.say_as(row):
            self.app.preview_text(candidate.context)
            return
        text = apply_lexicon(candidate.context, saved + [self._rule(row)])
        self.app.preview_text(text, lexicon=[])

    def add(self, row: int) -> bool:
        """Adds the row's rule. False when the cell is empty or the rule was
        already added."""
        return self._add_rows([row]) == 1

    def add_all_filled(self) -> int:
        """Adds a rule for every filled row, in one lexicon change. Returns
        how many."""
        return self._add_rows(self.filled_rows())

    def _add_rows(self, rows: list) -> int:
        rows = [row for row in rows if self.say_as(row) and not self.is_added(row)]
        if not rows:
            return 0
        self.app.lexicon_dock.add_rules([self._rule(row) for row in rows])
        self._added.update(self._shown[row].word for row in rows)
        self._populate()
        return len(rows)
