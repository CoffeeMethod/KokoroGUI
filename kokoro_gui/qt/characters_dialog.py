"""Edit > Characters... - the character library dialog (UI13: name, color,
voice, FX preset; the Audio8 reference-pair picker stays in the Voice
Reference dock until the WF4 global library exists).

Edits `app.document.characters` directly. Adding a character also adds a
`Track` for it (Q8's auto-placement default, same as migration.py does).
Removing one is refused while any clip still uses it.

`apply_changes()` is separated from the widgets so tests can drive the
dialog without `exec()`.
"""
from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from kokoro_gui.daw.models import DEFAULT_HIGHLIGHT_PALETTE, Character, Track
from kokoro_gui.qt.fx_presets import list_fx_preset_names

_FX_NONE = "(none)"


class CharactersDialog(QDialog):
    def __init__(self, app, parent=None):
        super().__init__(parent or app)
        self.app = app
        self.setWindowTitle("Characters")
        self.resize(560, 360)
        self._current: Character | None = None
        self._loading = False

        root = QHBoxLayout(self)

        left = QVBoxLayout()
        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._on_current_changed)
        left.addWidget(self.list, 1)
        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self.add_character)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self.remove_current)
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.remove_btn)
        left.addLayout(btn_row)
        root.addLayout(left, 1)

        right = QWidget()
        form = QFormLayout(right)
        self.name_edit = QLineEdit()
        self.name_edit.textEdited.connect(self._on_name_edited)
        form.addRow("Name:", self.name_edit)

        color_row = QHBoxLayout()
        self.color_btn = QPushButton()
        self.color_btn.setFixedWidth(60)
        self.color_btn.clicked.connect(self._pick_color)
        self.color_edit = QLineEdit()
        self.color_edit.setPlaceholderText("#rrggbb")
        self.color_edit.editingFinished.connect(self._on_color_edited)
        color_row.addWidget(self.color_btn)
        color_row.addWidget(self.color_edit, 1)
        form.addRow("Color:", color_row)

        self.voice_combo = QComboBox()
        self.voice_combo.setEditable(True)
        for voice in self.app.get_all_voices():
            self.voice_combo.addItem(voice)
        self.voice_combo.currentTextChanged.connect(self._on_voice_changed)
        form.addRow("Voice:", self.voice_combo)

        self.fx_combo = QComboBox()
        self.fx_combo.addItem(_FX_NONE)
        for name in list_fx_preset_names(self.app.project_dir):
            self.fx_combo.addItem(name)
        self.fx_combo.currentTextChanged.connect(self._on_fx_changed)
        form.addRow("FX preset:", self.fx_combo)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.buttons.rejected.connect(self.accept)
        self.buttons.accepted.connect(self.accept)
        form.addRow(self.buttons)
        root.addWidget(right, 2)

        self.reload()

    # -- list ------------------------------------------------------------------

    def reload(self) -> None:
        self._loading = True
        try:
            self.list.clear()
            for character in self.app.document.characters:
                item = QListWidgetItem(character.name)
                item.setData(0x0100, character.id)
                item.setForeground(QColor(character.highlight_color))
                self.list.addItem(item)
        finally:
            self._loading = False
        if self.list.count():
            self.list.setCurrentRow(0)
        else:
            self._show(None)

    def _on_current_changed(self, current, _previous) -> None:
        if self._loading:
            return
        cid = current.data(0x0100) if current is not None else None
        self._show(self.app.document.get_character(cid) if cid else None)

    def _show(self, character: Character | None) -> None:
        self._current = character
        self._loading = True
        try:
            enabled = character is not None
            for w in (self.name_edit, self.color_btn, self.color_edit, self.voice_combo, self.fx_combo, self.remove_btn):
                w.setEnabled(enabled)
            if character is None:
                self.name_edit.clear()
                self.color_edit.clear()
                self.color_btn.setStyleSheet("")
                return
            self.name_edit.setText(character.name)
            self._set_color_widgets(character.highlight_color)
            voice = character.preset_data.get("voice", "")
            if voice and self.voice_combo.findText(voice) < 0:
                self.voice_combo.addItem(voice)
            self.voice_combo.setCurrentText(voice or "")
            fx = character.preset_data.get("fx_preset") or _FX_NONE
            if fx == "Select FX Preset...":
                fx = _FX_NONE
            if self.fx_combo.findText(fx) < 0:
                self.fx_combo.addItem(fx)
            self.fx_combo.setCurrentText(fx)
        finally:
            self._loading = False

    def _set_color_widgets(self, color: str) -> None:
        self.color_edit.setText(color)
        self.color_btn.setStyleSheet(f"background-color: {color};")

    # -- edits -------------------------------------------------------------------

    def _on_name_edited(self, text: str) -> None:
        if self._loading or self._current is None:
            return
        self._current.name = text
        item = self.list.currentItem()
        if item is not None:
            item.setText(text)
        for track in self.app.document.tracks:
            if track.character_id == self._current.id:
                track.name = text
        self._changed()

    def _pick_color(self) -> None:
        if self._current is None:
            return
        color = QColorDialog.getColor(QColor(self._current.highlight_color), self, "Highlight color")
        if color.isValid():
            self.set_color(color.name())

    def _on_color_edited(self) -> None:
        if self._loading or self._current is None:
            return
        text = self.color_edit.text().strip()
        if QColor(text).isValid():
            self.set_color(QColor(text).name())

    def set_color(self, color: str) -> None:
        if self._current is None:
            return
        self._current.highlight_color = color
        self._set_color_widgets(color)
        item = self.list.currentItem()
        if item is not None:
            item.setForeground(QColor(color))
        self._changed()

    def _on_voice_changed(self, text: str) -> None:
        if self._loading or self._current is None:
            return
        if text:
            self._current.preset_data["voice"] = text
        else:
            self._current.preset_data.pop("voice", None)
        self._changed()

    def _on_fx_changed(self, text: str) -> None:
        if self._loading or self._current is None:
            return
        if text and text != _FX_NONE:
            self._current.preset_data["fx_preset"] = text
        else:
            self._current.preset_data.pop("fx_preset", None)
        self._changed()

    def add_character(self) -> Character:
        doc = self.app.document
        index = len(doc.characters)
        color = DEFAULT_HIGHLIGHT_PALETTE[index % len(DEFAULT_HIGHLIGHT_PALETTE)]
        base = "Character"
        names = {c.name for c in doc.characters}
        name = base
        n = 2
        while name in names:
            name = f"{base} {n}"
            n += 1
        character = Character.from_preset_dict(name, {"voice": self.app.settings.get("voice", "af_heart")},
                                               highlight_color=color, backend_id=self.app.backend.id)
        doc.characters.append(character)
        doc.tracks.append(Track(name=name, character_id=character.id, order_index=len(doc.tracks)))
        self.reload()
        self.list.setCurrentRow(self.list.count() - 1)
        self._changed()
        return character

    def remove_current(self) -> None:
        character = self._current
        if character is None:
            return
        doc = self.app.document
        in_use = [c for c in doc.clips if c.character_id == character.id]
        if in_use:
            QMessageBox.warning(self, "In use",
                                f"{character.name} is used by {len(in_use)} clip(s). Reassign them first.")
            return
        doc.characters = [c for c in doc.characters if c.id != character.id]
        doc.tracks = [t for t in doc.tracks if t.character_id != character.id]
        for i, track in enumerate(sorted(doc.tracks, key=lambda t: t.order_index)):
            track.order_index = i
        self.reload()
        self._changed()

    def _changed(self) -> None:
        self.app.on_characters_changed()
