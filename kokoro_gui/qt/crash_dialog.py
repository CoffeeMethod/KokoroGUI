"""The dialog an uncaught exception opens, and the bridge that gets it there
from any thread. `main.py` creates one `CrashBridge`, installs the excepthooks
with `bridge.report`, and then builds the window. The dialog never quits the
app, and it isn't modal, so the user can still save the project."""
from __future__ import annotations

import os

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout,
)

from kokoro_gui import logging_setup
from kokoro_gui.qt.reveal import reveal

MESSAGE = "Something went wrong. The details are saved in the log."


class CrashDialog(QDialog):
    def __init__(self, traceback_text: str, log_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("KokoroGUI error")
        self.resize(640, 420)
        self.log_path = log_path
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(MESSAGE))
        path_label = QLabel(log_path)
        path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(path_label)
        self.text_edit = QPlainTextEdit(traceback_text)
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

        buttons = QDialogButtonBox()
        self.copy_button = QPushButton("Copy")
        self.copy_button.clicked.connect(self.copy_text)
        self.open_log_button = QPushButton("Open Log Folder")
        self.open_log_button.clicked.connect(self.open_log_folder)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.reject)
        buttons.addButton(self.copy_button, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(self.open_log_button, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(self.close_button, QDialogButtonBox.ButtonRole.RejectRole)
        layout.addWidget(buttons)

    def text(self) -> str:
        return self.text_edit.toPlainText()

    def copy_text(self) -> None:
        QGuiApplication.clipboard().setText(self.text())

    def open_log_folder(self) -> None:
        if not reveal(self.log_path):
            reveal(os.path.dirname(self.log_path))


class CrashBridge(QObject):
    """Lives in the GUI thread. `report` is safe to call from any thread:
    Qt queues the signal across threads, so the dialog is always built in the
    GUI thread. At most one dialog is open at a time; an exception that
    arrives while it's open is only logged (the excepthook already did)."""

    crashed = Signal(str)

    def __init__(self, log_path: str, parent: QObject | None = None):
        super().__init__(parent)
        self.log_path = log_path
        self.window = None  # the dialog's parent, set once the main window exists
        self.dialog: CrashDialog | None = None
        self.crashed.connect(self.present)

    def report(self, exc_type, exc, tb) -> None:
        """The callback for `logging_setup.install_excepthooks`."""
        self.crashed.emit(logging_setup.format_exception_text(exc_type, exc, tb))

    @Slot(str)
    def present(self, traceback_text: str) -> CrashDialog | None:
        if self.dialog is not None and self.dialog.isVisible():
            return None
        dialog = CrashDialog(traceback_text, self.log_path, self.window)
        dialog.finished.connect(self._forget)
        self.dialog = dialog
        dialog.show()
        return dialog

    def _forget(self, *_) -> None:
        if self.dialog is not None:
            self.dialog.deleteLater()
        self.dialog = None
