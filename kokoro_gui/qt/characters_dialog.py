"""Edit > Characters... - the project's characters and their link to the
global character library (UI13: name, color, voice, FX preset; phase 3:
the library, grill WF4-WF7 and WF12).

Each row carries a scope badge: "Library" for a character linked to a
library entry (`Character.library_id`), "Local" for one that lives only in
this project, and "Library (not found here)" for a linked character whose
entry isn't in this machine's library (the inlined snapshot plays).

Editing a linked character's voice, FX, color or variants writes the
library entry too, then re-resolves the open document (the live link,
WF5). Renaming edits this project's name only; "Rename in library" copies
it to the entry. "Promote to Library" turns a local character into a new
entry (WF7). "Add" is a split button: a new local character, or a linked
copy of library entries not yet in the project.

A character on a cloning backend (Audio8) also gets a variants table:
variant name -> reference (`Character.variants`), which a clip picks with
the transcript's Variant combo.

Edits `app.document.characters` directly. Adding a character makes no
`Track`; `Document.assign_character_to_range` makes one on first use
(grill PR4). Removing one is refused while any clip still uses it; the
library entry stays.

Every button calls a public method (`add_character`, `add_from_library`,
`promote_current`, `rename_in_library`, ...), so tests drive the dialog
without `exec()`.
"""
from __future__ import annotations

import copy

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QColorDialog, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QMessageBox, QPushButton,
    QStyledItemDelegate, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget,
)

from kokoro_gui.daw import library as library_ops
from kokoro_gui.daw.models import DEFAULT_HIGHLIGHT_PALETTE, Character
from kokoro_gui.engines.registry import DEFAULT_ENGINE_ID
from kokoro_gui.qt import theme
from kokoro_gui.qt.fx_presets import list_fx_preset_names

# NP3's three scopes (phase 4): the global library, the project (a root
# character linked at project scope, which its subprojects follow), and
# this file only. A linked character neither store has on this machine plays
# its inlined snapshot.
SCOPE_LOCAL = "Local"
SCOPE_GLOBAL = "Global"
SCOPE_PROJECT = "Project"
SCOPE_MISSING = "Linked (not found here)"
SCOPE_LIBRARY = SCOPE_GLOBAL  # the phase 3 name
_SCOPE_LABELS = {"local": SCOPE_LOCAL, "global": SCOPE_GLOBAL, "project": SCOPE_PROJECT, "missing": SCOPE_MISSING}
_ID_ROLE = 0x0100
_SCOPE_ROLE = 0x0101

_FX_NONE = "(none)"


class _ScopeBadgeDelegate(QStyledItemDelegate):
    """Paints the row's scope right-aligned in the muted text color."""

    def paint(self, painter, option, index) -> None:  # noqa: N802 (Qt override)
        super().paint(painter, option, index)
        scope = index.data(_SCOPE_ROLE)
        if not scope:
            return
        painter.save()
        painter.setPen(QColor(theme.current().text_muted))
        rect = QRect(option.rect).adjusted(0, 0, -6, 0)
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), scope)
        painter.restore()


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
        self.list.setItemDelegate(_ScopeBadgeDelegate(self.list))
        self.list.currentItemChanged.connect(self._on_current_changed)
        left.addWidget(self.list, 1)
        btn_row = QHBoxLayout()
        # Split button: the face adds a local character, the arrow offers
        # "From library...".
        self.add_btn = QToolButton()
        self.add_btn.setText("Add")
        self.add_btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.add_btn.clicked.connect(self.add_character)
        add_menu = QMenu(self.add_btn)
        self.add_local_action = add_menu.addAction("New local character")
        self.add_local_action.triggered.connect(self.add_character)
        self.add_library_action = add_menu.addAction("From library...")
        self.add_library_action.triggered.connect(self.pick_from_library)
        self.add_btn.setMenu(add_menu)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self.remove_current)
        btn_row.addWidget(self.add_btn)
        btn_row.addWidget(self.remove_btn)
        left.addLayout(btn_row)
        self.promote_btn = QPushButton("Promote to Library")
        self.promote_btn.setToolTip("Make this character a library entry, shared with every project.")
        self.promote_btn.clicked.connect(self.promote_current)
        left.addWidget(self.promote_btn)
        # In a subproject (NP3): share a character with the whole book.
        self.promote_project_btn = QPushButton("Promote to Project")
        self.promote_project_btn.setToolTip("Make this character the top-level project's, shared by every "
                                            "subproject in it.")
        self.promote_project_btn.clicked.connect(self.promote_to_project)
        self.promote_project_btn.hide()
        left.addWidget(self.promote_project_btn)
        root.addLayout(left, 1)

        right = QWidget()
        form = QFormLayout(right)
        self.name_edit = QLineEdit()
        self.name_edit.setToolTip("The name in this project. The library entry keeps its own.")
        self.name_edit.textEdited.connect(self._on_name_edited)
        form.addRow("Name:", self.name_edit)

        scope_row = QHBoxLayout()
        self.scope_label = QLabel()
        self.rename_in_library_btn = QPushButton("Rename in library")
        self.rename_in_library_btn.setToolTip("Give the library entry this project's name for the character.")
        self.rename_in_library_btn.clicked.connect(self.rename_in_library)
        scope_row.addWidget(self.scope_label, 1)
        scope_row.addWidget(self.rename_in_library_btn)
        form.addRow("Scope:", scope_row)

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

        # The engine this character generates with (grill V3): each
        # character picks its own; the app keeps every engine in use
        # resident.
        self.engine_combo = QComboBox()
        for label, engine_id in self.app.engine_choices():
            self.engine_combo.addItem(label, engine_id)
        self.engine_combo.activated.connect(lambda _i: self._on_engine_picked())
        form.addRow("Engine:", self.engine_combo)

        self.voice_combo = QComboBox()
        self.voice_combo.setEditable(True)
        self.voice_combo.currentTextChanged.connect(self._on_voice_changed)
        form.addRow("Voice:", self.voice_combo)

        self.fx_combo = QComboBox()
        self.fx_combo.addItem(_FX_NONE)
        for name in list_fx_preset_names(self.app.project_dir):
            self.fx_combo.addItem(name)
        self.fx_combo.currentTextChanged.connect(self._on_fx_changed)
        form.addRow("FX preset:", self.fx_combo)

        self.variants_label = QLabel("Variants:")
        variants_box = QWidget()
        variants_layout = QVBoxLayout(variants_box)
        variants_layout.setContentsMargins(0, 0, 0, 0)
        self.variants_table = QTableWidget(0, 2)
        self.variants_table.setHorizontalHeaderLabels(["Variant", "Reference"])
        self.variants_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.variants_table.verticalHeader().setVisible(False)
        self.variants_table.itemChanged.connect(lambda _item: self._commit_variants())
        variants_layout.addWidget(self.variants_table)
        variant_buttons = QHBoxLayout()
        self.add_variant_btn = QPushButton("Add variant")
        self.add_variant_btn.clicked.connect(self.add_variant)
        self.remove_variant_btn = QPushButton("Remove variant")
        self.remove_variant_btn.clicked.connect(self.remove_variant)
        variant_buttons.addWidget(self.add_variant_btn)
        variant_buttons.addWidget(self.remove_variant_btn)
        variants_layout.addLayout(variant_buttons)
        self.variants_box = variants_box
        form.addRow(self.variants_label, variants_box)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.buttons.rejected.connect(self.accept)
        self.buttons.accepted.connect(self.accept)
        form.addRow(self.buttons)
        root.addWidget(right, 2)

        self.reload()

    # -- list ------------------------------------------------------------------

    @property
    def library(self):
        return self.app.character_library

    def scope_of(self, character: Character) -> str:
        return _SCOPE_LABELS[self.app.character_scope(character)]

    def _in_subproject(self) -> bool:
        focus = getattr(self.app, "focus", None)
        return focus is not None and focus.parent_id is not None

    def reload(self) -> None:
        self._loading = True
        try:
            self.list.clear()
            for character in self.app.document.characters:
                item = QListWidgetItem(character.name)
                item.setData(_ID_ROLE, character.id)
                item.setData(_SCOPE_ROLE, self.scope_of(character))
                item.setForeground(QColor(character.highlight_color))
                self.list.addItem(item)
        finally:
            self._loading = False
        if self.list.count():
            self.list.setCurrentRow(0)
        else:
            self._show(None)

    def _select(self, character: Character) -> None:
        for row in range(self.list.count()):
            if self.list.item(row).data(_ID_ROLE) == character.id:
                self.list.setCurrentRow(row)
                return

    def _on_current_changed(self, current, _previous) -> None:
        if self._loading:
            return
        cid = current.data(_ID_ROLE) if current is not None else None
        self._show(self.app.document.get_character(cid) if cid else None)

    def _show_scope(self, character: Character | None) -> None:
        scope = self.scope_of(character) if character is not None else ""
        self.scope_label.setText(scope)
        self.promote_btn.setEnabled(scope in (SCOPE_LOCAL, SCOPE_PROJECT))
        self.promote_project_btn.setVisible(self._in_subproject())
        self.promote_project_btn.setEnabled(self._in_subproject() and scope == SCOPE_LOCAL)
        self.rename_in_library_btn.setEnabled(scope == SCOPE_GLOBAL)
        item = self.list.currentItem()
        if item is not None and character is not None:
            item.setData(_SCOPE_ROLE, scope)

    def _show(self, character: Character | None) -> None:
        self._current = character
        self._loading = True
        try:
            enabled = character is not None
            for w in (self.name_edit, self.color_btn, self.color_edit, self.engine_combo, self.voice_combo,
                      self.fx_combo, self.remove_btn):
                w.setEnabled(enabled)
            self._show_scope(character)
            if character is None:
                self.name_edit.clear()
                self.color_edit.clear()
                self.color_btn.setStyleSheet("")
                return
            self.name_edit.setText(character.name)
            self._set_color_widgets(character.highlight_color)
            index = self.engine_combo.findData(character.backend_id or DEFAULT_ENGINE_ID)
            self.engine_combo.setCurrentIndex(max(0, index))
            self._fill_voices(character)
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
            self._fill_variants(character)
        finally:
            self._loading = False

    # -- variants (cloning backends) ---------------------------------------------

    def _supports_variants(self, character) -> bool:
        backend = self.app._backend_for(character.backend_id or DEFAULT_ENGINE_ID) if character is not None else None
        return bool(backend is not None and getattr(backend.capabilities, "supports_voice_cloning", False))

    def _reference_combo(self, value: str) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)
        for voice in self.app.get_all_voices(backend=self.app.backend_for_character(self._current)):
            combo.addItem(voice)
        if value and combo.findText(value) < 0:
            combo.addItem(value)
        combo.setCurrentText(value or "")
        combo.currentTextChanged.connect(lambda _t: self._commit_variants())
        return combo

    def _fill_variants(self, character) -> None:
        shown = self._supports_variants(character)
        self.variants_label.setVisible(shown)
        self.variants_box.setVisible(shown)
        self.variants_table.blockSignals(True)
        try:
            self.variants_table.setRowCount(0)
            for name, reference in sorted((character.variants or {}).items()):
                self._append_variant_row(name, reference)
        finally:
            self.variants_table.blockSignals(False)

    def _append_variant_row(self, name: str, reference: str) -> None:
        row = self.variants_table.rowCount()
        self.variants_table.insertRow(row)
        self.variants_table.setItem(row, 0, QTableWidgetItem(name))
        self.variants_table.setCellWidget(row, 1, self._reference_combo(reference))

    def variant_rows(self) -> dict:
        """The table as `{variant: reference}`, blank names or references
        left out."""
        out = {}
        for row in range(self.variants_table.rowCount()):
            item = self.variants_table.item(row, 0)
            combo = self.variants_table.cellWidget(row, 1)
            name = item.text().strip() if item is not None else ""
            reference = combo.currentText().strip() if combo is not None else ""
            if name and reference:
                out[name] = reference
        return out

    def _commit_variants(self) -> None:
        if self._loading or self._current is None:
            return
        variants = self.variant_rows()
        if variants != (self._current.variants or {}):
            self._current.variants = variants
            self._changed()

    def add_variant(self) -> None:
        if self._current is None:
            return
        names = set(self.variant_rows())
        name, n = "variant", 2
        while name in names:
            name, n = f"variant {n}", n + 1
        self.variants_table.blockSignals(True)
        try:
            self._append_variant_row(name, self._current.preset_data.get("voice", ""))
        finally:
            self.variants_table.blockSignals(False)
        self._commit_variants()

    def remove_variant(self) -> None:
        row = self.variants_table.currentRow()
        if row < 0:
            row = self.variants_table.rowCount() - 1
        if row >= 0:
            self.variants_table.removeRow(row)
            self._commit_variants()

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
        # The project's name only (WF12); "Rename in library" is separate.
        self._changed(write_through=False)

    def _pick_color(self) -> None:
        if self._current is None:
            return
        for i, preset in enumerate(DEFAULT_HIGHLIGHT_PALETTE):
            QColorDialog.setCustomColor(i, QColor(preset))
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

    def refresh_voices(self) -> None:
        """A voice editor saved or deleted a voice: relist the current
        character's voices, keeping its choice."""
        if self._current is not None:
            self._show(self._current)

    def _fill_voices(self, character) -> None:
        """The voice list of the character's own engine."""
        self.voice_combo.clear()
        for voice in self.app.get_all_voices(backend=self.app.backend_for_character(character)):
            self.voice_combo.addItem(voice)

    def _on_engine_picked(self) -> None:
        engine_id = self.engine_combo.currentData()
        self.set_engine(engine_id)

    def set_engine(self, engine_id: str) -> bool:
        """Switches the current character to `engine_id` (and, when linked,
        its library entry): `app.set_character_engine`, which also picks the
        voice (grill EN2). Refused while a job runs."""
        character = self._current
        if character is None or not engine_id or engine_id == (character.backend_id or DEFAULT_ENGINE_ID):
            return False
        ok = self.app.set_character_engine(character, engine_id)
        self._show(character)
        return ok

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
        """A new local character (WF6). No track until the transcript uses
        it (grill PR4)."""
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
        character = self.app.make_character(name, color)
        doc.characters.append(character)
        self.reload()
        self.list.setCurrentRow(self.list.count() - 1)
        self._changed(write_through=False)
        return character

    # -- the library (phase 3) ----------------------------------------------------

    def library_entries_to_add(self) -> list:
        """Library entries no character in this project links to yet."""
        linked = {c.library_id for c in self.app.document.characters if c.library_id}
        return [entry for entry in self.library.list() if entry.library_id not in linked]

    def add_from_library(self, library_ids) -> list:
        """Inlines a linked copy of each entry (fresh document id,
        `library_id` set) and returns the new records."""
        added = []
        for library_id in library_ids:
            entry = self.library.get(library_id)
            if entry is None:
                continue
            record = library_ops.linked_copy(entry)
            self.app.document.characters.append(record)
            added.append(record)
        if added:
            self.reload()
            self._select(added[-1])
            self._changed(write_through=False)
        return added

    def pick_from_library(self) -> list:
        entries = self.library_entries_to_add()
        if not entries:
            QMessageBox.information(self, "Character library",
                                    "Every library character is already in this project.")
            return []
        picker = QDialog(self)
        picker.setWindowTitle("Add from library")
        layout = QVBoxLayout(picker)
        listing = QListWidget()
        listing.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        for entry in entries:
            item = QListWidgetItem(entry.name)
            item.setData(_ID_ROLE, entry.library_id)
            item.setForeground(QColor(entry.highlight_color))
            listing.addItem(item)
        listing.itemDoubleClicked.connect(lambda _item: picker.accept())
        layout.addWidget(listing)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(picker.accept)
        buttons.rejected.connect(picker.reject)
        layout.addWidget(buttons)
        if picker.exec() != QDialog.DialogCode.Accepted:
            return []
        return self.add_from_library([item.data(_ID_ROLE) for item in listing.selectedItems()])

    def promote_current(self) -> str | None:
        """WF7: saves a local character as a new library entry and links this
        record to it. A project-scope character (NP3) moves on to the
        library under the id it already has, so every subproject linked to
        it follows through the root's record."""
        character = self._current
        if character is None:
            return None
        scope = self.app.character_scope(character)
        if scope == "local":
            library_id = self.library.save(character)
            character.library_id = library_id
        elif scope == "project":
            root_record = next(c for c in self.app.root.document.characters
                               if c.library_id == character.library_id)
            library_id = self.library.save(root_record)
        else:
            return None
        self._show_scope(character)
        self._changed(write_through=False)
        return library_id

    def promote_to_project(self) -> str | None:
        """NP3, from a subproject: the local character becomes one of the
        top-level project's (a root record under a fresh project-scope
        `library_id`) and this record links to it."""
        from kokoro_gui.daw.models import _new_id

        character = self._current
        if character is None or character.library_id or not self._in_subproject():
            return None
        library_id = library_ops.new_project_scope_id()
        record = copy.deepcopy(character)
        record.id = _new_id()
        record.library_id = library_id
        self.app.root.document.characters.append(record)
        character.library_id = library_id
        self._show_scope(character)
        self._changed(write_through=False)
        return library_id

    def rename_in_library(self) -> bool:
        character = self._current
        if character is None or not character.library_id:
            return False
        entry = self.library.get(character.library_id)
        if entry is None:
            return False
        entry.name = character.name
        self.library.save(entry)
        return True

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
        self._changed(write_through=False)

    def _changed(self, write_through: bool = True) -> None:
        """After an edit: `app.commit_character_edit` writes a linked
        character through to its library entry (WF5) and refreshes."""
        self.app.commit_character_edit(self._current, write_through=write_through)
