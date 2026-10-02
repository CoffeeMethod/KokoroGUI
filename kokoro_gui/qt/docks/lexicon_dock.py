"""Lexicon dock: the rules in `self.app.settings["lexicon"]`, a list of
`{"find", "replace", "mode", "case"}` applied in order (`engine/lexicon.py`).
Saves eagerly (bypasses the debounced autosave every other field uses). A
Test field under the table shows a sentence after the rules."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDockWidget, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMessageBox, QPushButton, QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from kokoro_gui.engine.lexicon import check_rule, normalize_rules, spoken

# (label, rule mode): what the Mode column offers.
MODE_CHOICES = (("Exact text", "literal"), ("Whole word", "word"), ("Pattern", "regex"))
FIND_COL, REPLACE_COL, MODE_COL, CASE_COL, DELETE_COL = range(5)


def replacement_for_storage(text: str, mode: str) -> str:
    """What goes into a rule's `replace` for the `text` the user typed. The
    engine hands `replace` to `re.sub` as a template, so an exact or whole
    word rule doubles its backslashes and speaks what was typed; a pattern
    rule keeps them, so `\\1` is a group."""
    return text if mode == "regex" else text.replace("\\", "\\\\")


def replacement_for_display(stored: str, mode: str) -> str:
    """The inverse of `replacement_for_storage`."""
    return stored if mode == "regex" else stored.replace("\\\\", "\\")


def format_reading(pairs) -> str:
    """`explain_text`'s `[(token, spoken form), ...]` as one line: each token
    that has a spoken form, then the form between slashes. A token with none
    (a bare `$`) or whose form is itself (punctuation) is left out."""
    shown = [f"{token.strip()} /{spoken.strip()}/" for token, spoken in pairs
             if token.strip() and spoken.strip() and token.strip() != spoken.strip()]
    return "  ".join(shown) if shown else "nothing to say"


class LexiconDock(QDockWidget):
    # (request number, `explain_text`'s pairs or None): emitted on the
    # engine's thread, delivered on the GUI thread.
    explained = Signal(int, object)

    def __init__(self, app, parent=None):
        super().__init__("Lexicon", parent)
        self.setObjectName("dock_lexicon")
        self.app = app
        self._populating = False

        content = QWidget()
        layout = QVBoxLayout(content)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Find", "Replace", "Mode", "Match case", ""])
        columns = self.table.horizontalHeader()
        columns.setSectionResizeMode(FIND_COL, QHeaderView.ResizeMode.Stretch)
        columns.setSectionResizeMode(REPLACE_COL, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        # Rules apply in the order of the row numbers: drag a number to move a rule.
        header = self.table.verticalHeader()
        header.setSectionsMovable(True)
        header.setToolTip("Rules apply from the top. Drag a number to reorder.")
        header.sectionMoved.connect(self._section_moved)
        self.table.itemChanged.connect(self._item_changed)
        layout.addWidget(self.table, 1)

        self.empty_label = QLabel("No rules defined.")
        layout.addWidget(self.empty_label)

        order_row = QHBoxLayout()
        self.up_button = QPushButton("Move up")
        self.up_button.clicked.connect(lambda _c=False: self.move_rule(self.table.currentRow(), -1))
        self.down_button = QPushButton("Move down")
        self.down_button.clicked.connect(lambda _c=False: self.move_rule(self.table.currentRow(), 1))
        order_row.addWidget(self.up_button)
        order_row.addWidget(self.down_button)
        order_row.addStretch(1)
        layout.addLayout(order_row)

        add_row = QHBoxLayout()
        add_row.addWidget(QLabel("Find:"))
        self.orig_edit = QLineEdit()
        add_row.addWidget(self.orig_edit)
        add_row.addWidget(QLabel("Replace:"))
        self.replace_edit = QLineEdit()
        add_row.addWidget(self.replace_edit)
        self.mode_combo = QComboBox()
        for label, mode in MODE_CHOICES:
            self.mode_combo.addItem(label, mode)
        add_row.addWidget(self.mode_combo)
        self.case_check = QCheckBox("Match case")
        add_row.addWidget(self.case_check)
        add_btn = QPushButton("Add Rule")
        add_btn.clicked.connect(self.add_rule)
        add_row.addWidget(add_btn)
        layout.addLayout(add_row)

        # Why the last edit was refused (an invalid pattern, an empty Find).
        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setTextFormat(Qt.TextFormat.PlainText)
        self.error_label.setStyleSheet("color: #c0392b;")
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

        note = QLabel("Rules apply in order, before generation; a change marks the clips it affects stale. "
                      "Whole word skips a match inside a longer word. Pattern is a regular expression, and "
                      "\\1 in Replace is its first group. Rules ignore case unless Match case is on.")
        note.setWordWrap(True)
        layout.addWidget(note)

        layout.addWidget(QLabel("Test:"))
        self.test_edit = QLineEdit()
        self.test_edit.setPlaceholderText("Type a sentence to see it after the rules")
        self.test_edit.textChanged.connect(self.update_test)
        layout.addWidget(self.test_edit)
        self.after_label = self._result_label()
        layout.addWidget(self.after_label)
        # "<engine> reads: ...", only for an engine that can explain how it reads text.
        self.reads_label = self._result_label()
        self.reads_label.setVisible(False)
        layout.addWidget(self.reads_label)
        self._reads_token = 0
        self._reads_name = ""
        self._reads_timer = QTimer(self)
        self._reads_timer.setSingleShot(True)
        self._reads_timer.setInterval(350)
        self._reads_timer.timeout.connect(self._request_reading)
        self.explained.connect(self._show_reading)

        # Scrolls when the dock is short, so its tab never sets how short the
        # column it shares with the other tabs can get.
        self.table.setMinimumHeight(120)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(content)
        self.setWidget(scroll)
        self.refresh_list()

    @staticmethod
    def _result_label() -> QLabel:
        # User text: a sentence like `<b>x</b>` must show as text, not render.
        label = QLabel("")
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    # -- the rules -----------------------------------------------------------

    def rules(self) -> list:
        """The stored rule list. A settings value in the old dict shape (set
        by a test or a script) is converted and written back first."""
        lexicon = self.app.settings.get("lexicon")
        fixed = normalize_rules(lexicon)
        if fixed != lexicon:
            self.app.settings["lexicon"] = lexicon = fixed
        return lexicon

    def _show_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.setVisible(bool(message))

    def add_rule(self) -> None:
        orig = self.orig_edit.text().strip()
        rep = self.replace_edit.text().strip()
        if not orig:
            QMessageBox.warning(self, "Error", "Original text cannot be empty.")
            return
        mode = self.mode_combo.currentData()
        case = self.case_check.isChecked()
        stored = replacement_for_storage(rep, mode)
        problem = check_rule(orig, stored, mode, case)
        if problem:
            self._show_error(f"Rule not added: {problem}")
            return
        self._show_error("")
        self.rules().append({"find": orig, "replace": stored, "mode": mode, "case": case})
        self.orig_edit.clear()
        self.replace_edit.clear()
        self._rules_changed()

    def delete_rule(self, index: int) -> None:
        rules = self.rules()
        if 0 <= index < len(rules):
            del rules[index]
            self._rules_changed()

    def move_rule(self, index: int, delta: int) -> None:
        """Moves rule `index` up (`delta` -1) or down (+1) in the order."""
        rules = self.rules()
        target = index + delta
        if not (0 <= index < len(rules) and 0 <= target < len(rules)):
            return
        rules[index], rules[target] = rules[target], rules[index]
        self._rules_changed()
        self.table.selectRow(target)

    def _section_moved(self, _logical: int, _old: int, _new: int) -> None:
        # Let the drag finish before the table is rebuilt under it.
        QTimer.singleShot(0, self._apply_header_order)

    def _apply_header_order(self) -> None:
        """Reorders the rules to the row numbers' dragged order, then puts
        the header back to plain 1..n over the rebuilt rows."""
        header = self.table.verticalHeader()
        rules = self.rules()
        count = header.count()
        if count != len(rules):
            return
        order = [header.logicalIndex(visual) for visual in range(count)]
        if order == list(range(count)):
            return
        moved = self.table.currentRow()
        header.blockSignals(True)
        for visual in range(count):
            header.moveSection(header.visualIndex(visual), visual)
        header.blockSignals(False)
        rules[:] = [rules[logical] for logical in order]
        self._rules_changed()
        if 0 <= moved < count:
            self.table.selectRow(order.index(moved))

    def _item_changed(self, item: QTableWidgetItem) -> None:
        """An edit in the Find, Replace or Match case cell."""
        if self._populating:
            return
        row, col = item.row(), item.column()
        rules = self.rules()
        if not 0 <= row < len(rules):
            return
        rule = dict(rules[row])
        if col == FIND_COL:
            rule["find"] = item.text().strip()
        elif col == REPLACE_COL:
            rule["replace"] = replacement_for_storage(item.text(), rule["mode"])
        elif col == CASE_COL:
            rule["case"] = item.checkState() == Qt.CheckState.Checked
        else:
            return
        self._update_rule(row, rule)

    def _mode_changed(self, row: int, combo: QComboBox) -> None:
        if self._populating or not 0 <= row < len(self.rules()):
            return
        rule = dict(self.rules()[row])
        rule["mode"] = combo.currentData()
        self._update_rule(row, rule)

    def _update_rule(self, row: int, rule: dict) -> None:
        """Stores `rule` at `row` when it is usable. Otherwise shows why not
        and puts the old rule back in the row. The stored replacement keeps
        its meaning across a mode change, so the Replace cell is redrawn."""
        problem = ("Find cannot be empty." if not rule["find"]
                   else check_rule(rule["find"], rule["replace"], rule["mode"], rule["case"]))
        if problem:
            self._show_error(f"Not saved: {problem}")
            self._redraw_row(row)
            return
        self._show_error("")
        self.rules()[row] = rule
        self._redraw_row(row)
        self._rules_changed(rebuild=False)

    def _rules_changed(self, rebuild: bool = True) -> None:
        """Saves, redraws the table, and re-runs the dirty check: the
        lexicon is a generation input, so a rule stales the clips whose
        text it rewrites."""
        self.app.save_settings()
        if rebuild:
            self.refresh_list()
        else:
            self.update_test()
        if self.app.editor is not None:
            self.app.editor.rehighlight()
        self.app.refresh_timeline()

    # -- the table -----------------------------------------------------------

    def _redraw_row(self, row: int) -> None:
        """Row `row`'s cells from its stored rule, in place (an edit handler
        can't rebuild the table it is running in)."""
        rule = self.rules()[row]
        self._populating = True
        try:
            self.table.item(row, FIND_COL).setText(rule["find"])
            self.table.item(row, REPLACE_COL).setText(replacement_for_display(rule["replace"], rule["mode"]))
            self.table.item(row, CASE_COL).setCheckState(
                Qt.CheckState.Checked if rule["case"] else Qt.CheckState.Unchecked)
            combo = self.table.cellWidget(row, MODE_COL)
            combo.setCurrentIndex(max(0, combo.findData(rule["mode"])))
        finally:
            self._populating = False

    def refresh_list(self) -> None:
        rules = self.rules()
        self._populating = True
        try:
            self.table.clearContents()
            self.table.setRowCount(len(rules))
            for row, rule in enumerate(rules):
                self.table.setItem(row, FIND_COL, QTableWidgetItem(rule["find"]))
                self.table.setItem(row, REPLACE_COL, QTableWidgetItem(
                    replacement_for_display(rule["replace"], rule["mode"])))
                combo = QComboBox()
                for label, mode in MODE_CHOICES:
                    combo.addItem(label, mode)
                combo.setCurrentIndex(max(0, combo.findData(rule["mode"])))
                combo.currentIndexChanged.connect(lambda _i, r=row, c=combo: self._mode_changed(r, c))
                self.table.setCellWidget(row, MODE_COL, combo)
                case_item = QTableWidgetItem()
                case_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
                                   | Qt.ItemFlag.ItemIsSelectable)
                case_item.setCheckState(Qt.CheckState.Checked if rule["case"] else Qt.CheckState.Unchecked)
                self.table.setItem(row, CASE_COL, case_item)
                del_btn = QPushButton("X")
                del_btn.setToolTip("Delete this rule")
                del_btn.clicked.connect(lambda _c=False, r=row: self.delete_rule(r))
                self.table.setCellWidget(row, DELETE_COL, del_btn)
        finally:
            self._populating = False
        self.empty_label.setVisible(not rules)
        self.table.setVisible(bool(rules))
        self.update_test()

    # -- the Test field ------------------------------------------------------

    def show_text(self, text: str) -> None:
        """Puts `text` in the Test field and brings this tab forward."""
        self.test_edit.setText(" ".join(text.split()))
        self.show()
        self.raise_()

    def after_rules(self, text: str) -> str:
        """`text` as generation hands it to the engine: markup stripped,
        then the rules."""
        return spoken(text, self.rules())

    def update_test(self, *_args) -> None:
        text = self.test_edit.text()
        self.after_label.setText(f"After your rules: {self.after_rules(text)}" if text.strip() else "")
        self._reads_token += 1  # an answer in flight is for older text
        if text.strip():
            self._reads_timer.start()
        else:
            self._reads_timer.stop()
            self.reads_label.setVisible(False)

    def _explaining_backend(self):
        """The active character's backend when it can explain how it reads
        text, else None."""
        try:
            backend = self.app.backend
        except Exception:  # noqa: BLE001 - no document or engine yet
            return None
        return backend if getattr(backend, "explains_text", False) else None

    def _request_reading(self) -> None:
        """Asks the active engine how it reads the sentence after the rules.
        The answer comes back on the engine's worker and reaches
        `_show_reading` on the GUI thread."""
        backend = self._explaining_backend()
        text = self.after_rules(self.test_edit.text())
        if backend is None or not text.strip():
            self.reads_label.setVisible(False)
            return
        lang_code = self.app.engine_settings(backend.id).get("lang_code")
        future = backend.explain(text, lang_code)
        if future is None:
            self.reads_label.setVisible(False)
            return
        self._reads_token += 1
        token = self._reads_token
        # "Kokoro (local)" reads as "Kokoro".
        self._reads_name = (getattr(backend, "display_name", "") or backend.id).split(" (")[0]
        self.reads_label.setText(f"{self._reads_name} reads: ...")
        self.reads_label.setVisible(True)
        future.add_done_callback(lambda f, t=token: self._reading_done(t, f))

    def _reading_done(self, token: int, future) -> None:
        """Runs on the engine's thread: hands the result to the GUI thread."""
        try:
            pairs = future.result()
        except Exception:  # noqa: BLE001 - a missing language package, a cancelled job
            pairs = None
        try:
            self.explained.emit(token, pairs)
        except RuntimeError:  # the dock closed while the engine was answering
            pass

    def _show_reading(self, token: int, pairs) -> None:
        if token != self._reads_token:
            return
        reading = format_reading(pairs) if pairs else "not available"
        self.reads_label.setText(f"{self._reads_name} reads: {reading}")
        self.reads_label.setVisible(True)
