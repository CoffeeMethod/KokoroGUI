"""Welcome dialog (grill WF2, revised): recent projects, Resume, New, New
from template, New from text, Open, shown over the main window on launch.

The window has already loaded the last project by the time this opens, so
the engine warms up underneath, Escape is a free Resume, and New starts
with the character library's characters the same way File > New does. New
from text opens the import wizard on the file (grill NP8: it starts on
one subproject per chapter for an EPUB or a PDF with an outline). Every
pick is a one-line call into `QtTTSApp` (`open_project`, `new_project`,
`new_from_ebook`, `open_project_dialog`); nothing here touches the
document directly.

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

from kokoro_gui.i18n import N_, tr
from kokoro_gui.qt import project as project_io

MISSING_SUFFIX = " (missing)"
TEXT_FILTER = N_("Documents (*.txt *.pdf *.epub)")


def _format_duration(seconds: float) -> str:
    seconds = int(round(seconds or 0))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class WelcomeDialog(QDialog):
    def __init__(self, app, parent=None):
        super().__init__(parent or app)
        self.app = app
        self.setWindowTitle(tr("Welcome"))
        self.resize(720, 400)

        root = QVBoxLayout(self)
        body = QHBoxLayout()
        root.addLayout(body, 1)

        left = QVBoxLayout()
        left.addWidget(QLabel(tr("Recent projects")))
        self.list = QListWidget()
        self.list.setToolTip(tr("Your recent projects. Double-click to open one, right-click to take it off the list."))
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._show_row_menu)
        self.list.currentItemChanged.connect(self._on_current_changed)
        self.list.itemActivated.connect(lambda item: self.choose(item.data(Qt.ItemDataRole.UserRole)))
        left.addWidget(self.list, 1)
        self.clear_btn = QPushButton(tr("Clear list"))
        self.clear_btn.setToolTip(tr("Empty the list. The project files stay."))
        self.clear_btn.clicked.connect(self.clear_recent)
        left.addWidget(self.clear_btn, 0, Qt.AlignmentFlag.AlignLeft)
        body.addLayout(left, 3)

        right = QVBoxLayout()
        details = QGroupBox(tr("Details"))
        form = QFormLayout(details)
        self.path_label = QLabel("-")
        self.path_label.setWordWrap(True)
        self.modified_label = QLabel("-")
        self.characters_label = QLabel("-")
        self.clips_label = QLabel("-")
        self.subprojects_label = QLabel("-")
        self.duration_label = QLabel("-")
        self.engines_label = QLabel("-")
        form.addRow(tr("Path:"), self.path_label)
        form.addRow(tr("Modified:"), self.modified_label)
        form.addRow(tr("Characters:"), self.characters_label)
        form.addRow(tr("Clips:"), self.clips_label)
        form.addRow(tr("Subprojects:"), self.subprojects_label)
        form.addRow(tr("Audio:"), self.duration_label)
        form.addRow(tr("Engines:"), self.engines_label)
        right.addWidget(details)

        self.open_btn = QPushButton(tr("Open"))
        self.open_btn.setToolTip(tr("Open the selected project."))
        self.open_btn.setDefault(True)
        self.open_btn.clicked.connect(self._open_selected)
        self.new_btn = QPushButton(tr("New project"))
        self.new_btn.setToolTip(tr("Start an empty project."))
        self.new_btn.clicked.connect(self.new_project)
        self.new_from_template_btn = QPushButton(tr("New from template..."))
        self.new_from_template_btn.setToolTip(tr("Start a project from a template you saved."))
        self.new_from_template_btn.clicked.connect(self.new_from_template)
        self.new_from_text_btn = QPushButton(tr("New from text file..."))
        self.new_from_text_btn.setToolTip(tr("Start a project from a .txt, .pdf or .epub."))
        self.new_from_text_btn.clicked.connect(self._new_from_text_dialog)
        self.open_other_btn = QPushButton(tr("Open other..."))
        self.open_other_btn.setToolTip(tr("Open a project that isn't on the list."))
        self.open_other_btn.clicked.connect(self.open_other)
        for widget in (self.open_btn, self.new_btn, self.new_from_template_btn, self.new_from_text_btn,
                       self.open_other_btn):
            right.addWidget(widget)
        right.addStretch(1)
        body.addLayout(right, 2)

        self.show_at_startup = QCheckBox(tr("Show at startup"))
        self.show_at_startup.setToolTip(tr("Show this window each time KokoroGUI starts."))
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
            for label in (self.modified_label, self.characters_label, self.clips_label,
                          self.subprojects_label, self.duration_label, self.engines_label):
                label.setText("-")
        else:
            self.path_label.setText(summary["path"])
            self.modified_label.setText(summary["modified"].strftime("%Y-%m-%d %H:%M"))
            self.characters_label.setText(str(summary["characters"]))
            self.clips_label.setText(str(summary["clips"]))
            self.subprojects_label.setText(str(summary.get("subprojects", 0)))
            duration = summary.get("duration_s")
            self.duration_label.setText(_format_duration(duration) if duration is not None else "-")
            engines = summary.get("engines") or []
            self.engines_label.setText(", ".join(engines) if engines else "-")
        is_current = bool(path) and self.app.project_path is not None and \
            os.path.abspath(path) == os.path.abspath(self.app.project_path)
        self.open_btn.setText(tr("Resume") if is_current else tr("Open"))
        self.open_btn.setEnabled(bool(path) and (is_current or summary is not None))

    def _show_row_menu(self, pos) -> None:
        item = self.list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        remove = menu.addAction(tr("Remove from list"))
        remove.setToolTip(tr("Take it off this list. The project file stays."))
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

    def new_from_template(self) -> None:
        self.accept()
        self.app.new_from_template_dialog()

    def _new_from_text_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("New project from text"), filter=tr(TEXT_FILTER))
        if path:
            self.new_from_text(path)

    def new_from_text(self, path: str) -> None:
        self.accept()
        self.app.new_from_ebook(path)

    def open_other(self) -> None:
        self.accept()
        self.app.open_project_dialog()

    def _on_show_toggled(self, checked: bool) -> None:
        self.app.settings["show_welcome"] = bool(checked)
        self.app.schedule_save()
