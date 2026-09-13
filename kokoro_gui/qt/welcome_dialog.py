"""Welcome dialog (grill WF2, revised): recent projects, Resume, New, New
from text, Open, shown over the main window on launch.

The window has already loaded the last project by the time this opens, so
the engine warms up underneath, Escape is a free Resume, and New inherits
the loaded project's characters (WF3) the same way File > New does. Every
pick is a one-line call into `QtTTSApp` (`open_project`, `new_project`,
`import_text(..., target="new")`, `open_project_dialog`); nothing here
touches the document directly.

Opened with `open()` rather than `exec()`: window-modal but asynchronous,
so the engine's init status still reaches the transport dock and pytest-qt
can drive the dialog without a blocked event loop. `main.py` is the only
launch-time trigger (`QtTTSApp.show_welcome_if_enabled`); the test fixture
and `scripts/render_screenshot.py` never see it.
"""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMenu, QPushButton, QVBoxLayout,
)

from kokoro_gui.qt import project as project_io

MISSING_SUFFIX = " (missing)"
TEXT_FILTER = "Documents (*.txt *.pdf *.epub)"


class WelcomeDialog(QDialog):
    def __init__(self, app, parent=None):
        super().__init__(parent or app)
        self.app = app
        self.setWindowTitle("Welcome")
        self.resize(720, 400)

        root = QVBoxLayout(self)
        body = QHBoxLayout()
        root.addLayout(body, 1)

        left = QVBoxLayout()
        left.addWidget(QLabel("Recent projects"))
        self.list = QListWidget()
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._show_row_menu)
        self.list.currentItemChanged.connect(self._on_current_changed)
        self.list.itemActivated.connect(lambda item: self.choose(item.data(Qt.ItemDataRole.UserRole)))
        left.addWidget(self.list, 1)
        self.clear_btn = QPushButton("Clear list")
        self.clear_btn.clicked.connect(self.clear_recent)
        left.addWidget(self.clear_btn, 0, Qt.AlignmentFlag.AlignLeft)
        body.addLayout(left, 3)

        right = QVBoxLayout()
        details = QGroupBox("Details")
        form = QFormLayout(details)
        self.path_label = QLabel("-")
        self.path_label.setWordWrap(True)
        self.modified_label = QLabel("-")
        self.characters_label = QLabel("-")
        self.clips_label = QLabel("-")
        form.addRow("Path:", self.path_label)
        form.addRow("Modified:", self.modified_label)
        form.addRow("Characters:", self.characters_label)
        form.addRow("Clips:", self.clips_label)
        right.addWidget(details)

        self.open_btn = QPushButton("Open")
        self.open_btn.setDefault(True)
        self.open_btn.clicked.connect(self._open_selected)
        self.new_btn = QPushButton("New project")
        self.new_btn.clicked.connect(self.new_project)
        self.new_from_text_btn = QPushButton("New from text file...")
        self.new_from_text_btn.clicked.connect(self._new_from_text_dialog)
        self.open_other_btn = QPushButton("Open other...")
        self.open_other_btn.clicked.connect(self.open_other)
        for btn in (self.open_btn, self.new_btn, self.new_from_text_btn, self.open_other_btn):
            right.addWidget(btn)
        right.addStretch(1)
        body.addLayout(right, 2)

        self.show_at_startup = QCheckBox("Show at startup")
        self.show_at_startup.setChecked(bool(app.settings.get("show_welcome", True)))
        self.show_at_startup.toggled.connect(self._on_show_toggled)
        root.addWidget(self.show_at_startup, 0, Qt.AlignmentFlag.AlignLeft)

        self.reload()

    # --- list ------------------------------------------------------------------

    def reload(self) -> None:
        """Rebuilds the rows from `settings["recent_projects"]`; the open
        project is listed first and starts selected."""
        self.show_at_startup.setChecked(bool(self.app.settings.get("show_welcome", True)))
        self.list.clear()
        current = self.app.project_path
        recent = [p for p in self.app.settings.get("recent_projects", []) if isinstance(p, str)]
        if current:
            recent = [current] + [p for p in recent if os.path.abspath(p) != os.path.abspath(current)]
        for path in recent:
            item = QListWidgetItem(project_io.project_title(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            if not os.path.isfile(path):
                item.setText(item.text() + MISSING_SUFFIX)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.list.addItem(item)
        first_enabled = next((i for i in range(self.list.count())
                              if self.list.item(i).flags() & Qt.ItemFlag.ItemIsEnabled), None)
        if first_enabled is not None:
            self.list.setCurrentRow(first_enabled)
        else:
            self.list.setCurrentRow(-1)
            self._on_current_changed(None, None)

    def paths(self) -> list:
        return [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())]

    def selected_path(self) -> str | None:
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def _on_current_changed(self, current, _previous) -> None:
        path = current.data(Qt.ItemDataRole.UserRole) if current is not None else None
        summary = project_io.project_summary(path) if path else None
        if summary is None:
            self.path_label.setText(path or "-")
            for label in (self.modified_label, self.characters_label, self.clips_label):
                label.setText("-")
        else:
            self.path_label.setText(summary["path"])
            self.modified_label.setText(summary["modified"].strftime("%Y-%m-%d %H:%M"))
            self.characters_label.setText(str(summary["characters"]))
            self.clips_label.setText(str(summary["clips"]))
        is_current = bool(path) and self.app.project_path is not None and \
            os.path.abspath(path) == os.path.abspath(self.app.project_path)
        self.open_btn.setText("Resume" if is_current else "Open")
        self.open_btn.setEnabled(bool(path) and (is_current or summary is not None))

    def _show_row_menu(self, pos) -> None:
        item = self.list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        remove = menu.addAction("Remove from list")
        remove.triggered.connect(lambda: self.remove_from_recent(item.data(Qt.ItemDataRole.UserRole)))
        menu.exec(self.list.mapToGlobal(pos))

    def remove_from_recent(self, path: str) -> None:
        project_io.forget_recent(self.app.settings, path)
        self.app.schedule_save()
        self.app._rebuild_recent_menu()
        self.reload()

    def clear_recent(self) -> None:
        project_io.clear_recent(self.app.settings)
        self.app.schedule_save()
        self.app._rebuild_recent_menu()
        self.reload()

    # --- picks -------------------------------------------------------------------

    def _open_selected(self) -> None:
        path = self.selected_path()
        if path:
            self.choose(path)

    def choose(self, path: str) -> None:
        """Open `path`, or just close when it's the project already loaded."""
        self.accept()
        current = self.app.project_path
        if current and os.path.abspath(path) == os.path.abspath(current):
            return
        self.app.open_project(path)

    def new_project(self) -> None:
        self.accept()
        self.app.new_project()

    def _new_from_text_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "New project from text", filter=TEXT_FILTER)
        if path:
            self.new_from_text(path)

    def new_from_text(self, path: str) -> None:
        self.accept()
        self.app.import_text(path, target="new")

    def open_other(self) -> None:
        self.accept()
        self.app.open_project_dialog()

    def _on_show_toggled(self, checked: bool) -> None:
        self.app.settings["show_welcome"] = bool(checked)
        self.app.schedule_save()
