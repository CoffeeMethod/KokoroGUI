"""The three small forms behind plan 25: File > Save as Template..., File >
New from Template... (and the welcome dialog's button) and File > Add
Credits....

They collect choices and change nothing. `QtTTSApp` (through
`SubprojectsMixin` in subprojects.py) reads `values()` or `selected()` and
does the work. The template store is `kokoro_gui/daw/templates.py`.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QVBoxLayout,
)

from kokoro_gui.daw import templates


class SaveTemplateDialog(QDialog):
    """`SaveTemplateDialog(parent, name, sections)`. `sections` is one
    `(title, text)` per subproject; `text` is None when it can't be read
    (the subproject holds subprojects of its own, or its file is gone). Each
    row has a "keep its text" box, ticked for a short text and disabled when
    there is none to keep. `values()` is `(name, [(title, text_or_empty)])`."""

    def __init__(self, parent, name: str, sections: list):
        super().__init__(parent)
        self.setWindowTitle("Save as Template")
        self.sections = [(str(title), text) for title, text in sections]
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit(name)
        self.name_edit.setMaxLength(templates.MAX_NAME_CHARS)
        form.addRow("Template name:", self.name_edit)
        layout.addLayout(form)
        layout.addWidget(QLabel(
            "A template keeps the project's settings, its library characters and its subprojects.\n"
            "Tick Keep text for a subproject whose text should come back, such as an intro."))
        self.keep_boxes = []
        for title, text in self.sections:
            box = QCheckBox(self._label(title, text))
            has_text = bool(text and text.strip())
            box.setEnabled(has_text)
            box.setChecked(has_text and len(text) < templates.KEEP_TEXT_LIMIT)
            if text is None:
                box.setToolTip("This subproject's text can't be read.")
            layout.addWidget(box)
            self.keep_boxes.append(box)
        if not self.sections:
            layout.addWidget(QLabel("This project has no subprojects. The template keeps its settings."))
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.name_edit.textChanged.connect(self._sync_buttons)
        self._sync_buttons()

    @staticmethod
    def _label(title: str, text) -> str:
        if text is None:
            return f"{title or 'Subproject'} (text not available)"
        return f"Keep text of {title or 'Subproject'} ({len(text):,} characters)"

    def _sync_buttons(self) -> None:
        usable = templates.safe_name(self.name_edit.text()) is not None
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(usable)

    def values(self) -> tuple:
        rows = []
        for (title, text), box in zip(self.sections, self.keep_boxes):
            rows.append((title, text if box.isChecked() and text else ""))
        return self.name_edit.text().strip(), rows


class NewFromTemplateDialog(QDialog):
    """`NewFromTemplateDialog(parent)`: the saved templates in a list.
    `selected()` is the chosen `Template` or None. Delete removes the
    template's file after a question."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("New from Template")
        self.resize(420, 320)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Start a new project from a saved template."))
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(lambda _item: self.accept())
        self.list.currentItemChanged.connect(lambda *_: self._sync())
        layout.addWidget(self.list, 1)
        row = QHBoxLayout()
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self.delete_selected)
        row.addWidget(self.delete_btn)
        row.addStretch(1)
        layout.addLayout(row)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.reload()

    def reload(self) -> None:
        self.list.clear()
        for template in templates.list_templates():
            count = len(template.sections)
            item = QListWidgetItem(f"{template.name}  ({count} subproject{'s' if count != 1 else ''})")
            item.setData(Qt.ItemDataRole.UserRole, template.stem)
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)
        else:
            self.list.addItem("No templates yet. File > Save as Template... makes one.")
            self.list.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
        self._sync()

    def _sync(self) -> None:
        has = self._current_stem() is not None
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(has)
        self.delete_btn.setEnabled(has)

    def _current_stem(self):
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def selected(self):
        stem = self._current_stem()
        return templates.load_template(stem) if stem else None

    def delete_selected(self) -> bool:
        stem = self._current_stem()
        if not stem:
            return False
        answer = QMessageBox.question(self, "Delete template", f"Delete the template \"{stem}\"?")
        if answer != QMessageBox.StandardButton.Yes:
            return False
        templates.delete_template(stem)
        self.reload()
        return True


FIELD_LABELS = {
    "title": "Title", "subtitle": "Subtitle", "author": "Author", "narrator": "Narrator",
    "publisher": "Publisher", "year": "Year",
}


class CreditsDialog(QDialog):
    """`CreditsDialog(parent, fields)`: one line edit per credit field and a
    preview of the two texts they make. `values()` is the field dict."""

    def __init__(self, parent, fields: dict):
        super().__init__(parent)
        self.setWindowTitle("Add Credits")
        self.resize(520, 360)
        fields = templates.clean_credit_fields(fields)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.edits = {}
        for name in templates.CREDIT_FIELDS:
            edit = QLineEdit(fields[name])
            edit.textChanged.connect(self._refresh)
            self.edits[name] = edit
            form.addRow(FIELD_LABELS[name] + ":", edit)
        layout.addLayout(form)
        self.opening_label = QLabel()
        self.closing_label = QLabel()
        for label in (self.opening_label, self.closing_label):
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(label)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._refresh()

    def values(self) -> dict:
        return templates.clean_credit_fields({name: edit.text() for name, edit in self.edits.items()})

    def _refresh(self) -> None:
        opening, closing = templates.credit_texts(self.values())
        self.opening_label.setText(f"Opening: {opening or '(none, there is no title, author or narrator)'}")
        self.closing_label.setText(f"Closing: {closing}")
