"""Options > Storage...: how much disk the two derived caches take, a Clear
button for each, and the size limit the generated audio cache is trimmed to.

The project working copies (`cache/projects/`) are listed so the size is no
surprise, but they have no button: the app manages them, and an open
project's unsaved work lives there. All the file work is in
`engine/cache_admin.py`, which only ever deletes top-level files named like
a cache entry.

Sizes are measured on a worker thread (a directory scan can be long on a
big cache), and so is Clear. The folders are resolved on the GUI thread
before the worker starts.
"""
from __future__ import annotations

import threading
from typing import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSpinBox, QVBoxLayout,
)

from kokoro_gui.engine import cache_admin
from kokoro_gui.qt.reveal import reveal

LIMIT_SETTING = "segment_cache_max_mb"
MAX_LIMIT_MB = 1_000_000

SEGMENTS = "segments"
REFS = "refs"
PROJECTS = "projects"

ROWS = (
    (SEGMENTS, "Generated audio cache (whole-document and JIT)"),
    (REFS, "Audio8 reference codes"),
    (PROJECTS, "Project working copies (managed automatically)"),
)

MEASURING = "Measuring..."


def trim_in_background(max_mb, folder: str, report: Callable[[str], None] = print) -> threading.Thread | None:
    """Trims the generated audio cache in `folder` to `max_mb` on a daemon
    thread and returns it. None when the limit is off (0 or not a number).
    `report` gets one line when something was removed or the trim failed."""
    if isinstance(max_mb, bool) or not isinstance(max_mb, (int, float)) or not max_mb > 0:
        return None

    def _work():
        try:
            removed = cache_admin.trim_to_setting(max_mb, folder)
        except Exception as e:  # noqa: BLE001 - a trim must never take the app down
            report(f"Couldn't trim the generated audio cache: {e}")
            return
        if removed.count:
            report(f"Trimmed the generated audio cache to {max_mb} MB: removed {cache_admin.describe(removed)}.")

    thread = threading.Thread(target=_work, name="cache-trim", daemon=True)
    thread.start()
    return thread


class StorageDialog(QDialog):
    # (what was asked, its result or None, the error or None), from the worker.
    _workDone = Signal(object)

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.setWindowTitle("Storage")
        self.setObjectName("storage_dialog")
        self.resize(640, 260)
        self._thread: threading.Thread | None = None
        self.size_labels: dict[str, QLabel] = {}
        self.clear_buttons: dict[str, QPushButton] = {}
        self._usage: dict[str, cache_admin.Usage] = {}

        layout = QVBoxLayout(self)
        grid = QGridLayout()
        grid.setColumnStretch(0, 1)
        for row, (key, title) in enumerate(ROWS):
            grid.addWidget(QLabel(title), row, 0)
            size = QLabel(MEASURING)
            size.setObjectName(f"storage_size_{key}")
            grid.addWidget(size, row, 1)
            self.size_labels[key] = size
            if key != PROJECTS:
                button = QPushButton("Clear...")
                button.setObjectName(f"storage_clear_{key}")
                button.clicked.connect(lambda _checked=False, k=key: self.clear(k))
                grid.addWidget(button, row, 2)
                self.clear_buttons[key] = button
        layout.addLayout(grid)

        limit_row = QHBoxLayout()
        limit_row.addWidget(QLabel("Keep the generated audio cache under"))
        self.limit_spin = QSpinBox()
        self.limit_spin.setObjectName("storage_limit")
        self.limit_spin.setRange(0, MAX_LIMIT_MB)
        self.limit_spin.setSingleStep(256)
        self.limit_spin.setSuffix(" MB")
        self.limit_spin.setSpecialValueText("No limit")
        self.limit_spin.setToolTip("When the cache is over this, the least recently used audio is deleted: "
                                   "at launch and after a whole-document or JIT generate. 0 keeps everything.")
        limit = app.settings.get(LIMIT_SETTING, cache_admin.DEFAULT_MAX_MB)
        self.limit_spin.setValue(limit if isinstance(limit, int) and not isinstance(limit, bool) else 0)
        self.limit_spin.valueChanged.connect(self._on_limit_changed)
        self.limit_spin.editingFinished.connect(self.apply_limit)
        limit_row.addWidget(self.limit_spin)
        limit_row.addStretch(1)
        layout.addLayout(limit_row)

        self.note = QLabel("")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        layout.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        self.show_folder_button = QPushButton("Show cache folder")
        self.show_folder_button.clicked.connect(self.show_cache_folder)
        buttons.addButton(self.show_folder_button, QDialogButtonBox.ButtonRole.ActionRole)
        layout.addWidget(buttons)

        self._workDone.connect(self._on_work_done)
        self.refresh()

    # -- the worker ------------------------------------------------------------

    def is_working(self) -> bool:
        return self._thread is not None

    def wait_idle(self, timeout_s: float = 30.0) -> None:
        """Blocks until the worker has finished and its result has been
        applied. For tests and scripts."""
        from PySide6.QtWidgets import QApplication

        thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
        QApplication.processEvents()

    def _start(self, what: str, work: Callable[[], object]) -> bool:
        if self._thread is not None:
            return False

        def _target():
            result = error = None
            try:
                result = work()
            except Exception as e:  # noqa: BLE001 - shown in the dialog
                error = e
            try:
                self._workDone.emit((what, result, error))
            except RuntimeError:  # the dialog was closed while it worked
                pass

        self._thread = threading.Thread(target=_target, name=f"storage-{what}", daemon=True)
        self._sync_buttons()
        self._thread.start()
        return True

    def _on_work_done(self, payload) -> None:
        what, result, error = payload
        self._thread = None
        if error is not None:
            self.note.setText(f"Couldn't finish: {error}")
        elif what == "measure":
            self._usage = result
            for key, usage in result.items():
                self.size_labels[key].setText(cache_admin.describe(usage))
        elif what == "trim":
            self.note.setText(f"Removed {cache_admin.describe(result)} of the least recently used audio."
                              if result.count else "The generated audio cache is already under the limit.")
        else:  # a clear
            key, removed = result
            noun = dict(ROWS)[key].split(" (")[0].lower()
            message = f"Cleared {cache_admin.describe(removed)} from the {noun}."
            self.note.setText(message)
            self.app.set_status(message, "success")
        self._sync_buttons()
        if error is None and what != "measure":
            self.refresh()

    def _sync_buttons(self) -> None:
        busy_app = self.app.is_busy()
        working = self._thread is not None
        for button in self.clear_buttons.values():
            button.setEnabled(not working and not busy_app)
            button.setToolTip("Wait for the current job to finish." if busy_app else "")

    # -- actions ---------------------------------------------------------------

    def refresh(self) -> None:
        """Measures all three folders again."""
        for label in self.size_labels.values():
            label.setText(MEASURING)
        segments, refs, projects = cache_admin.segment_cache_dir(), None, cache_admin.projects_dir()
        try:
            refs = cache_admin.ref_codes_dir()
        except Exception:  # noqa: BLE001 - Audio8 may not import; its row then reads 0
            pass

        def _measure():
            return {
                SEGMENTS: cache_admin.segment_cache_usage(segments),
                REFS: cache_admin.ref_codes_usage(refs) if refs else cache_admin.NOTHING,
                PROJECTS: cache_admin.projects_usage(projects),
            }

        self._start("measure", _measure)

    def clear(self, key: str) -> bool:
        """The Clear button of the `key` row: asks with the size, then
        deletes. Returns whether it started."""
        if key not in self.clear_buttons or self._thread is not None:
            return False
        if self.app.is_busy():
            self.note.setText("Wait for the current job to finish before clearing a cache.")
            return False
        usage = self._usage.get(key, cache_admin.NOTHING)
        if not usage.count:
            self.note.setText("Nothing to clear.")
            return False
        if key == SEGMENTS:
            folder, delete, title = cache_admin.segment_cache_dir(), cache_admin.clear_segment_cache, \
                "Clear the generated audio cache"
            text = (f"Delete {cache_admin.describe(usage)} of generated audio?\n\n"
                    "Clips in your projects keep their audio. A whole-document or JIT generate "
                    "synthesizes again what it used to find here.")
        else:
            folder, delete, title = cache_admin.ref_codes_dir(), cache_admin.clear_ref_codes, \
                "Clear the Audio8 reference codes"
            text = (f"Delete {cache_admin.describe(usage)} of reference codes?\n\n"
                    "Nothing is lost. The next generate encodes each reference again.")
        answer = QMessageBox.question(self, title, text,
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                                      QMessageBox.StandardButton.Cancel)
        if answer != QMessageBox.StandardButton.Yes:
            return False
        return self._start("clear", lambda: (key, delete(folder)))

    def _on_limit_changed(self, value: int) -> None:
        self.app._set_setting(LIMIT_SETTING, int(value))

    def apply_limit(self) -> None:
        """Trims to the limit now, then measures again."""
        if self._thread is not None:
            return
        folder = cache_admin.segment_cache_dir()
        max_mb = self.limit_spin.value()

        if max_mb > 0:
            self._start("trim", lambda: cache_admin.trim_to_setting(max_mb, folder))

    def show_cache_folder(self) -> None:
        reveal(cache_admin.segment_cache_dir())
