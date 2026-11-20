"""Exports dock (plan 38): every export, newest first, from the exports log
(`daw/exports_log.py`). Play, Show in Folder and Delete act on the selected row.

Delete asks first, lists the files, and removes only what `exports_log.delete_targets`
returns: the audio and the extras the export wrote beside it. The dock keeps no state of
its own; it rebuilds from the log when shown and after every export.
"""
from __future__ import annotations

import os
import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDockWidget, QHBoxLayout, QHeaderView, QMessageBox, QPushButton, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)

import playback
from kokoro_gui.daw import exports_log
from kokoro_gui.qt import reveal

COLUMNS = ("Project", "When", "Length", "Format")
_ENTRY = Qt.ItemDataRole.UserRole


def format_length(seconds: float) -> str:
    total = int(round(max(0.0, seconds)))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def format_when(stamp: float) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(stamp))
    except (OverflowError, OSError, ValueError):
        return ""


class OutputDock(QDockWidget):
    def __init__(self, app, parent=None):
        super().__init__("Exports", parent)
        self.setObjectName("dock_output")
        self.app = app
        self._stale = True
        self._playing: str | None = None

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)

        self.project_only = QCheckBox("This project only")
        self.project_only.toggled.connect(lambda _checked: self.refresh())
        layout.addWidget(self.project_only)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(list(COLUMNS))
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.currentItemChanged.connect(lambda *_a: self._sync_buttons())
        self.tree.itemDoubleClicked.connect(lambda *_a: self.toggle_play())
        self.tree.setMinimumHeight(48)
        layout.addWidget(self.tree, 1)

        buttons = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        self.reveal_button = QPushButton("Show in Folder")
        self.reveal_button.clicked.connect(self.reveal_selected)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_selected)
        for button in (self.play_button, self.reveal_button, self.delete_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.setWidget(content)
        self.visibilityChanged.connect(self._on_visibility_changed)
        self._sync_buttons()

    # -- rebuilding ---------------------------------------------------------

    def refresh(self) -> None:
        """Called after an export. A dock behind another tab only notes that
        it is out of date; it rebuilds when shown."""
        self._stale = True
        if self.isVisible():
            self.rebuild()

    def _on_visibility_changed(self, visible: bool) -> None:
        if visible and self._stale:
            self.rebuild()

    def rebuild(self) -> None:
        self._stale = False
        keep = self.selected_entry()
        rows = exports_log.entries(self.app.exports_log_file)
        if self.project_only.isChecked():
            here = self.app.project_path or ""
            rows = [r for r in rows if r["project_path"] == here]
        self.tree.clear()
        for entry in rows:
            title = entry["project_title"] or "Untitled"
            name = os.path.basename(entry["path"])
            item = QTreeWidgetItem([f"{title}: {name}" + ("  (missing)" if entry["missing"] else ""),
                                    format_when(entry["when"]), format_length(entry["duration_s"]),
                                    entry["format"].upper()])
            item.setData(0, _ENTRY, entry)
            item.setToolTip(0, entry["path"])
            self.tree.addTopLevelItem(item)
            if keep is not None and keep["path"] == entry["path"] and keep["when"] == entry["when"]:
                self.tree.setCurrentItem(item)
        self._sync_buttons()

    def selected_entry(self) -> dict | None:
        item = self.tree.currentItem()
        return item.data(0, _ENTRY) if item is not None else None

    def _sync_buttons(self) -> None:
        entry = self.selected_entry()
        present = entry is not None and not entry["missing"]
        self.play_button.setEnabled(present)
        self.reveal_button.setEnabled(present)
        self.delete_button.setEnabled(present)
        self.play_button.setText("Stop" if present and self._playing == entry["path"] else "Play")

    # -- actions ------------------------------------------------------------

    def toggle_play(self) -> None:
        entry = self.selected_entry()
        if entry is None or entry["missing"]:
            return
        if self._playing == entry["path"]:
            playback.stop()
            self._playing = None
        else:
            try:
                playback.play(entry["path"])
            except Exception as e:  # noqa: BLE001 - a format the audio library can't read
                self._playing = None
                self.app.set_status(f"Can't play {os.path.basename(entry['path'])}: {e}", "error")
            else:
                self._playing = entry["path"]
        self._sync_buttons()

    def reveal_selected(self) -> None:
        entry = self.selected_entry()
        if entry is not None:
            reveal.reveal(entry["path"])

    def delete_selected(self) -> bool:
        """Asks, then deletes the audio and its listed extras. True when files went."""
        entry = self.selected_entry()
        if entry is None:
            return False
        targets = exports_log.delete_targets(entry)
        if not targets:
            return False
        listing = "\n".join(os.path.basename(t) for t in targets)
        answer = QMessageBox.question(
            self, "Delete export",
            f"Delete these {len(targets)} file(s) from\n{os.path.dirname(targets[0])}?\n\n{listing}\n\n"
            "This can't be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return False
        if self._playing == entry["path"]:
            playback.stop()
            self._playing = None
        failed = 0
        for target in targets:
            try:
                os.remove(target)
            except OSError:
                failed += 1
        if not failed:
            exports_log.forget(self.app.exports_log_file, entry["path"])
        self.app.set_status(f"Deleted {len(targets) - failed} file(s)." if not failed
                            else f"Couldn't delete {failed} of {len(targets)} file(s).",
                            "success" if not failed else "error")
        self.rebuild()
        return True
