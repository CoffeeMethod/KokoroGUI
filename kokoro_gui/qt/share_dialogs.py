"""Export and import of character and FX share files (`.kgshare`, plan 40).

Two small dialogs collect choices and change nothing; `export_to_file` and
`import_from_file` run the file dialogs and the work, so the Characters dialog
and the Audio FX dock share one path. The format, its checks and the writes are
`kokoro_gui/daw/share.py`.

- `ExportShareDialog`: tick the characters (and, from the FX dock, the FX) to
  put in the file, and whether the FX and voice files the characters use go
  with them.
- `ImportShareDialog`: a row per thing the file holds, with a tick, and for
  one whose name is taken a choice: Keep both, Replace, Skip. Something already
  here and identical is shown ticked off.

Imported characters land in the character library, not in the open project;
Add > From library... in Edit > Characters brings them in.
"""
from __future__ import annotations

import os
import re

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QPlainTextEdit, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from kokoro_gui.daw import revision, share
from kokoro_gui.engines import registry

FILE_FILTER = f"KokoroGUI share files (*{share.EXTENSION});;All files (*)"
_CHOICES = (("Keep both", share.KEEP), ("Replace", share.REPLACE), ("Skip", share.SKIP))


class ExportShareDialog(QDialog):
    """`ExportShareDialog(parent, characters=(), fx_names=(), preselect=(), title="Export")`.
    `characters` are `Character`s, `fx_names` FX names; either list is hidden
    when empty. `preselect` holds the ids of the characters ticked at the
    start (all of them when empty). `selected_characters()`,
    `selected_fx()` and `options()` are the answers."""

    def __init__(self, parent, characters=(), fx_names=(), preselect=(), title: str = "Export"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(380, 420)
        self._characters = list(characters)
        layout = QVBoxLayout(self)
        self.character_list = QListWidget()
        self.fx_list = QListWidget()
        if self._characters:
            layout.addWidget(QLabel("Characters to export:"))
            for character in self._characters:
                item = QListWidgetItem(character.name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                ticked = not preselect or character.id in preselect
                item.setCheckState(Qt.CheckState.Checked if ticked else Qt.CheckState.Unchecked)
                self.character_list.addItem(item)
            layout.addWidget(self.character_list, 1)
        else:
            self.character_list.hide()
        if fx_names:
            layout.addWidget(QLabel("FX to export:"))
            for name in fx_names:
                item = QListWidgetItem(name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if len(fx_names) == 1 or name in preselect
                                   else Qt.CheckState.Unchecked)
                self.fx_list.addItem(item)
            layout.addWidget(self.fx_list, 1)
        else:
            self.fx_list.hide()
        self.include_fx = QCheckBox("Include the FX these characters use")
        self.include_fx.setChecked(True)
        self.include_voices = QCheckBox("Include their voice files (Mixes and References)")
        self.include_voices.setChecked(True)
        if self._characters:
            layout.addWidget(self.include_fx)
            layout.addWidget(self.include_voices)
        else:
            self.include_fx.hide()
            self.include_voices.hide()
        layout.addWidget(QLabel("A voice file is the voice itself. Leave it out if the other person has it."))
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.character_list.itemChanged.connect(lambda _item: self._sync())
        self.fx_list.itemChanged.connect(lambda _item: self._sync())
        self._sync()

    @staticmethod
    def _checked(listing: QListWidget) -> list:
        return [row for row in range(listing.count())
                if listing.item(row).checkState() == Qt.CheckState.Checked]

    def selected_characters(self) -> list:
        return [self._characters[row] for row in self._checked(self.character_list)]

    def selected_fx(self) -> list:
        return [self.fx_list.item(row).text() for row in self._checked(self.fx_list)]

    def options(self) -> tuple:
        return self.include_fx.isChecked(), self.include_voices.isChecked()

    def _sync(self) -> None:
        anything = bool(self.selected_characters() or self.selected_fx())
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(anything)


class ImportShareDialog(QDialog):
    """`ImportShareDialog(parent, info, items, kinds=None)`. `info` is a
    `ShareInfo`, `items` the `share.plan_import` list; `kinds` limits the rows
    to those `ImportItem.kind`s (the FX dock shows FX and impulse responses
    only). `choices()` maps each ticked row's key to Keep both, Replace or
    Skip."""

    def __init__(self, parent, info, items, kinds=None, title: str = "Import"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(560, 380)
        self.items = [item for item in items if kinds is None or item.kind in kinds]
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"{os.path.basename(info.path)} holds:"))
        self.table = QTableWidget(len(self.items), 4)
        self.table.setHorizontalHeaderLabels(["Import", "Kind", "Name", "If the name is taken"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._combos: dict = {}
        for row, item in enumerate(self.items):
            self._fill_row(row, item)
        self.table.resizeColumnsToContents()
        layout.addWidget(self.table, 1)
        notes = list(info.warnings)
        if info.ignored:
            notes.append(f"{info.ignored} other file{'s' if info.ignored != 1 else ''} in it will not be used.")
        self.notes = QPlainTextEdit("\n".join(notes))
        self.notes.setReadOnly(True)
        self.notes.setMaximumHeight(70)
        self.notes.setVisible(bool(notes))
        layout.addWidget(self.notes)
        if "character" in {item.kind for item in self.items}:
            layout.addWidget(QLabel("Imported characters go into the character library. "
                                    "Add > From library... puts them in a project."))
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.table.itemChanged.connect(lambda _item: self._sync())
        self._sync()

    def _fill_row(self, row: int, item) -> None:
        tick = QTableWidgetItem()
        usable = not item.unavailable and not item.same
        tick.setFlags(Qt.ItemFlag.ItemIsUserCheckable | (Qt.ItemFlag.ItemIsEnabled if usable else Qt.ItemFlag.NoItemFlags))
        tick.setCheckState(Qt.CheckState.Checked if usable else Qt.CheckState.Unchecked)
        self.table.setItem(row, 0, tick)
        kind = item.label
        if item.detail and item.kind in (share.KIND_VOICE, share.KIND_CHARACTER):
            kind = f"{kind} ({registry.get_display_name(item.detail)})"
        for column, text in ((1, kind), (2, item.name)):
            cell = QTableWidgetItem(text)
            cell.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, column, cell)
        if item.unavailable:
            self.table.setItem(row, 3, self._note(f"Can't import: {item.unavailable}"))
        elif item.same:
            self.table.setItem(row, 3, self._note("Already here"))
        elif item.clash:
            combo = QComboBox()
            for label, value in _CHOICES:
                combo.addItem(label, value)
            self.table.setCellWidget(row, 3, combo)
            self._combos[item.key] = combo
        else:
            self.table.setItem(row, 3, self._note(""))

    @staticmethod
    def _note(text: str) -> QTableWidgetItem:
        cell = QTableWidgetItem(text)
        cell.setFlags(Qt.ItemFlag.ItemIsEnabled)
        return cell

    def choices(self) -> dict:
        out = {}
        for row, item in enumerate(self.items):
            if self.table.item(row, 0).checkState() != Qt.CheckState.Checked:
                continue
            combo = self._combos.get(item.key)
            out[item.key] = combo.currentData() if combo is not None else share.KEEP
        return out

    def _sync(self) -> None:
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(self.choices()))


def _fx_dir() -> str:
    """`app.FX_PRESETS_DIR`, read at call time (a test patches it). A local
    import: `app.py` imports the Characters dialog, which imports this module."""
    import kokoro_gui.qt.app as qt_app_module

    return qt_app_module.FX_PRESETS_DIR


def _default_file_name(characters, fx_names) -> str:
    if len(characters) == 1:
        base = characters[0].name
    elif not characters and len(fx_names) == 1:
        base = fx_names[0]
    else:
        base = ""
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", base).strip(" .")
    return (base or ("fx" if fx_names and not characters else "characters")) + share.EXTENSION


def _refresh_after_import(app, result) -> None:
    """Every list that could show what came in: voices, the FX combos, the
    linked characters (a replaced library entry reaches the open project)."""
    revision.bump_files()
    if any(line.startswith("Voice ") for line in result.imported):
        app.refresh_voice_choices()
    if app.fx_dock is not None:
        app.fx_dock.refresh_presets()
    if any(line.startswith("Character ") for line in result.imported):
        app.resolve_library()


def export_to_file(app, parent, *, characters=(), fx_names=(), preselect=(), title: str = "Export Characters",
                   path: str | None = None):
    """Asks what to put in, where to save it, and writes it. Returns the
    `ExportReport`, or None when the user cancelled or the export failed (a
    message says why). `path` skips the save dialog."""
    dialog = ExportShareDialog(parent, characters, fx_names, preselect, title)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    chosen, chosen_fx = dialog.selected_characters(), dialog.selected_fx()
    include_fx, include_voices = dialog.options()
    if path is None:
        path, _filter = QFileDialog.getSaveFileName(parent, title, _default_file_name(chosen, chosen_fx), FILE_FILTER)
        if not path:
            return None
    if not path.lower().endswith(share.EXTENSION):
        path += share.EXTENSION
    try:
        report = share.export_share(path, chosen, fx_names=chosen_fx, include_fx=include_fx,
                                    include_voices=include_voices, project_dir=getattr(app, "project_dir", None),
                                    fx_dir=_fx_dir())
    except (share.ShareError, OSError) as e:
        QMessageBox.critical(parent, title, f"Couldn't write the share file: {e}")
        return None
    parts = []
    if report.characters:
        parts.append(f"{report.characters} character{'s' if report.characters != 1 else ''}")
    if report.fx:
        parts.append(f"{len(report.fx)} FX")
    if report.irs:
        parts.append(f"{len(report.irs)} impulse response{'s' if len(report.irs) != 1 else ''}")
    if report.voices:
        parts.append(f"{len(report.voices)} voice{'s' if len(report.voices) != 1 else ''}")
    message = f"Saved {', '.join(parts) or 'an empty share file'} to {os.path.basename(path)}."
    if report.warnings:
        message += "\n\n" + "\n".join(report.warnings)
    QMessageBox.information(parent, title, message)
    return report


def import_from_file(app, parent, *, kinds=None, title: str = "Import Characters", path: str | None = None):
    """Asks for a share file, shows what it holds and imports the ticked
    rows. Returns the `ImportResult`, or None when the user cancelled or the
    file was refused (a message says why). `kinds` limits the rows
    (`share.KIND_*`); `path` skips the open dialog."""
    if path is None:
        path, _filter = QFileDialog.getOpenFileName(parent, title, "", FILE_FILTER)
        if not path:
            return None
    try:
        info = share.inspect_share(path)
    except (share.ShareError, OSError) as e:
        QMessageBox.critical(parent, title, str(e))
        return None
    items = share.plan_import(info, library=app.character_library, fx_dir=_fx_dir())
    dialog = ImportShareDialog(parent, info, items, kinds, title)
    if not dialog.items:
        QMessageBox.information(parent, title, "There is nothing of that kind in this share file.")
        return None
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    choices = dialog.choices()
    if kinds is not None:
        choices = {key: value for key, value in choices.items() if key.split(":", 1)[0] in kinds}
    try:
        result = share.import_share(info, choices, library=app.character_library, fx_dir=_fx_dir())
    except (share.ShareError, OSError) as e:
        QMessageBox.critical(parent, title, f"Nothing was imported: {e}")
        return None
    _refresh_after_import(app, result)
    lines = [f"Imported {len(result.imported)} item{'s' if len(result.imported) != 1 else ''}."]
    lines.extend(result.imported)
    lines.extend(result.skipped)
    lines.extend(result.warnings)
    QMessageBox.information(parent, title, "\n".join(lines))
    return result
