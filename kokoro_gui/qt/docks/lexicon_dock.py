"""Lexicon dock: find/replace rules stored in `self.app.settings["lexicon"]`.
Saves eagerly (bypasses the debounced autosave every other field uses)."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDockWidget, QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)


class LexiconDock(QDockWidget):
    def __init__(self, app, parent=None):
        super().__init__("Lexicon", parent)
        self.setObjectName("dock_lexicon")
        self.app = app

        content = QWidget()
        layout = QVBoxLayout(content)

        add_row = QHBoxLayout()
        add_row.addWidget(QLabel("Original Text:"))
        self.orig_edit = QLineEdit()
        add_row.addWidget(self.orig_edit)
        add_row.addWidget(QLabel("Replacement:"))
        self.replace_edit = QLineEdit()
        add_row.addWidget(self.replace_edit)
        add_btn = QPushButton("Add Rule")
        add_btn.clicked.connect(self.add_rule)
        add_row.addWidget(add_btn)
        layout.addLayout(add_row)

        self.list_scroll = QScrollArea()
        self.list_scroll.setWidgetResizable(True)
        self._list_container = QWidget()
        self._list_layout = QVBoxLayout(self._list_container)
        self.list_scroll.setWidget(self._list_container)
        layout.addWidget(self.list_scroll, 1)

        layout.addWidget(QLabel("Note: Replacements are case-insensitive. Applied before generation; a change marks the clips it affects stale."))

        self.setWidget(content)
        self.refresh_list()

    def add_rule(self) -> None:
        orig = self.orig_edit.text().strip()
        rep = self.replace_edit.text().strip()
        if not orig:
            QMessageBox.warning(self, "Error", "Original text cannot be empty.")
            return

        if "lexicon" not in self.app.settings:
            self.app.settings["lexicon"] = {}
        self.app.settings["lexicon"][orig] = rep
        self.orig_edit.clear()
        self.replace_edit.clear()
        self._rules_changed()

    def delete_rule(self, key: str) -> None:
        if key in self.app.settings.get("lexicon", {}):
            del self.app.settings["lexicon"][key]
            self._rules_changed()

    def _rules_changed(self) -> None:
        """Saves, redraws the list, and re-runs the dirty check: the
        lexicon is a generation input, so a rule stales the clips whose
        text it rewrites."""
        self.app.save_settings()
        self.refresh_list()
        if self.app.editor is not None:
            self.app.editor.rehighlight()
        self.app.refresh_timeline()

    def refresh_list(self) -> None:
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        lexicon = self.app.settings.get("lexicon", {})
        if not lexicon:
            self._list_layout.addWidget(QLabel("No rules defined."))
            return

        for orig, rep in lexicon.items():
            row = QFrame()
            row_layout = QHBoxLayout(row)
            row_layout.addWidget(QLabel(orig))
            row_layout.addWidget(QLabel("->"))
            row_layout.addWidget(QLabel(rep))
            row_layout.addStretch(1)
            del_btn = QPushButton("X")
            del_btn.clicked.connect(lambda _c=False, k=orig: self.delete_rule(k))
            row_layout.addWidget(del_btn)
            self._list_layout.addWidget(row)
