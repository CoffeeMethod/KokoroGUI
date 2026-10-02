"""The import wizard: File > Import Text and the welcome dialog's New from
text show it between reading a book and putting the text in a project.

Left, the book's sections with a checkbox each (title, word count); a
section that looks like front or back matter starts unticked
(`text_cleanup.guess_skip`). Right, the cleanup rules as checkboxes and the
selected section's text before and after them. Bottom, where the text goes:
a new project with one subproject per section, a new project with one
transcript, or the current project at the caret.

`choices()` gives `(target, [(title, cleaned_text), ...])` for the ticked
sections. It isn't called `result()`: `QDialog` already has one, the
accept/reject code. The ticked rules are remembered in
`settings["import_rules"]` (`{rule_id: bool}`) when the dialog is accepted.
The dialog changes nothing else: `QtTTSApp` runs the existing import
paths on the choices.
"""
from __future__ import annotations

import os

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPlainTextEdit, QPushButton, QRadioButton, QSplitter, QVBoxLayout, QWidget,
)

from kokoro_gui.engine import text_cleanup

TARGET_SECTIONS = "sections"
TARGET_TRANSCRIPT = "transcript"
TARGET_ADD = "add"
TARGETS = (TARGET_SECTIONS, TARGET_TRANSCRIPT, TARGET_ADD)
TARGET_LABELS = {
    TARGET_SECTIONS: "New project, one subproject per section",
    TARGET_TRANSCRIPT: "New project, one transcript",
    TARGET_ADD: "Add to current project",
}

PREVIEW_DELAY_MS = 150
# A chapter can run to a few hundred thousand characters; the preview shows
# the head of it. `result()` always cleans the whole text.
PREVIEW_CHARS = 20_000


def word_count(text: str) -> int:
    return len(text.split())


class ImportDialog(QDialog):
    """`ImportDialog(app, path, sections)`. `targets` is which radio buttons
    exist; `default_target` the one that starts selected (the first enabled
    one when it isn't available). A one-section file has no per-section
    radio, since there is nothing to split."""

    def __init__(self, app, path: str, sections: list, targets=TARGETS, default_target=None, parent=None):
        super().__init__(parent or app)
        self.app = app
        self.path = path
        self.sections = [(str(title), str(text)) for title, text in sections]
        self.setWindowTitle(f"Import {os.path.basename(path)}")
        self.resize(1080, 680)

        root = QVBoxLayout(self)
        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split, 1)

        # -- left: the sections ------------------------------------------------
        left = QWidget()
        left_box = QVBoxLayout(left)
        left_box.setContentsMargins(0, 0, 0, 0)
        left_box.addWidget(QLabel("Sections to import"))
        self.section_list = QListWidget()
        for title, text in self.sections:
            item = QListWidgetItem(f"{title}  ({word_count(text):,} words)")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked if text_cleanup.guess_skip(title, text)
                               else Qt.CheckState.Checked)
            self.section_list.addItem(item)
        self.section_list.currentRowChanged.connect(lambda _row: self.refresh_preview())
        self.section_list.itemChanged.connect(lambda _item: self._update_ok())
        left_box.addWidget(self.section_list, 1)
        buttons = QHBoxLayout()
        self.all_btn = QPushButton("All")
        self.none_btn = QPushButton("None")
        self.all_btn.clicked.connect(lambda: self._set_all_checked(True))
        self.none_btn.clicked.connect(lambda: self._set_all_checked(False))
        buttons.addWidget(self.all_btn)
        buttons.addWidget(self.none_btn)
        buttons.addStretch(1)
        left_box.addLayout(buttons)
        split.addWidget(left)

        # -- right: the rules and the preview -----------------------------------
        right = QWidget()
        right_box = QVBoxLayout(right)
        right_box.setContentsMargins(0, 0, 0, 0)
        rules_group = QGroupBox("Cleanup")
        rules_box = QVBoxLayout(rules_group)
        saved = app.settings.get("import_rules")
        ticked = set(text_cleanup.enabled_rule_ids(saved))
        self.rule_checks: dict = {}
        for rule in text_cleanup.RULES:
            check = QCheckBox(rule.label)
            check.setChecked(rule.id in ticked)
            check.toggled.connect(lambda _on: self._schedule_preview())
            rules_box.addWidget(check)
            self.rule_checks[rule.id] = check
        right_box.addWidget(rules_group)

        previews = QHBoxLayout()
        before_box, self.before_edit = self._preview_pane("Before")
        after_box, self.after_edit = self._preview_pane("After")
        previews.addLayout(before_box)
        previews.addLayout(after_box)
        right_box.addLayout(previews, 1)
        self.preview_note = QLabel("")
        right_box.addWidget(self.preview_note)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)

        # -- bottom: the target -------------------------------------------------
        target_group = QGroupBox("Put the text in")
        target_box = QVBoxLayout(target_group)
        self.target_radios: dict = {}
        for target in TARGETS:
            if target not in targets:
                continue
            radio = QRadioButton(TARGET_LABELS[target])
            target_box.addWidget(radio)
            self.target_radios[target] = radio
        sections_radio = self.target_radios.get(TARGET_SECTIONS)
        if sections_radio is not None and len(self.sections) < 2:
            sections_radio.setEnabled(False)
        root.addWidget(target_group)
        self._select_default_target(default_target)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        root.addWidget(self.button_box)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(PREVIEW_DELAY_MS)
        self._preview_timer.timeout.connect(self.refresh_preview)

        self._update_ok()
        if self.sections:
            self.section_list.setCurrentRow(0)

    # -- building --------------------------------------------------------------

    @staticmethod
    def _preview_pane(title: str):
        box = QVBoxLayout()
        box.addWidget(QLabel(title))
        edit = QPlainTextEdit()
        edit.setReadOnly(True)
        box.addWidget(edit, 1)
        return box, edit

    def _select_default_target(self, default_target) -> None:
        usable = [t for t, radio in self.target_radios.items() if radio.isEnabled()]
        chosen = default_target if default_target in usable else (usable[0] if usable else None)
        if chosen is not None:
            self.target_radios[chosen].setChecked(True)

    # -- state -----------------------------------------------------------------

    def target(self) -> str | None:
        for target, radio in self.target_radios.items():
            if radio.isChecked():
                return target
        return None

    def enabled_rules(self) -> list:
        return [rule_id for rule_id, check in self.rule_checks.items() if check.isChecked()]

    def rule_state(self) -> dict:
        return {rule_id: check.isChecked() for rule_id, check in self.rule_checks.items()}

    def is_section_checked(self, row: int) -> bool:
        return self.section_list.item(row).checkState() == Qt.CheckState.Checked

    def checked_rows(self) -> list:
        return [row for row in range(self.section_list.count()) if self.is_section_checked(row)]

    def _set_all_checked(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for row in range(self.section_list.count()):
            self.section_list.item(row).setCheckState(state)

    def _update_ok(self) -> None:
        ok = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        ok.setEnabled(bool(self.checked_rows()) and self.target() is not None)

    # -- preview ---------------------------------------------------------------

    def _schedule_preview(self) -> None:
        timer = getattr(self, "_preview_timer", None)
        if timer is not None:  # a rule check toggles while the dialog is still being built
            timer.start()

    def refresh_preview(self) -> None:
        """Shows the selected section before and after the ticked rules."""
        self._preview_timer.stop()
        row = self.section_list.currentRow()
        if not 0 <= row < len(self.sections):
            self.before_edit.setPlainText("")
            self.after_edit.setPlainText("")
            self.preview_note.setText("")
            return
        text = self.sections[row][1]
        shown = text[:PREVIEW_CHARS]
        self.before_edit.setPlainText(shown)
        self.after_edit.setPlainText(text_cleanup.apply_rules(shown, self.enabled_rules()))
        self.preview_note.setText(f"The preview shows the first {PREVIEW_CHARS:,} characters of this section."
                                  if len(text) > PREVIEW_CHARS else "")

    # -- result ----------------------------------------------------------------

    def choices(self):
        """`(target, [(title, cleaned_text), ...])` for the ticked sections,
        in book order. A section with nothing left after cleanup is
        dropped."""
        rules = self.enabled_rules()
        parts = []
        for row in self.checked_rows():
            title, text = self.sections[row]
            cleaned = text_cleanup.apply_rules(text, rules)
            if cleaned.strip():
                parts.append((title, cleaned))
        return self.target(), parts

    def accept(self) -> None:
        self.app.settings["import_rules"] = self.rule_state()
        self.app.schedule_save()
        super().accept()
