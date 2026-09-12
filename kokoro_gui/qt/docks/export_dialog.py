"""File > Export... (section 6 of Claude/PLAN_ui_shell_redesign.md).

Holds what left the Settings tab: output folder, base filename, format,
"also write .srt", "keep per-clip files". Values persist per project in
`app.project_settings["export"]`, falling back to the old `config_qt.json`
keys (`out_dir`/`filename`/`format`/`export_subtitles`/`separate`) so an
existing user's choices carry over.

`run_export()` refuses (with the count) while any clip is dirty, offering
"Generate first" / "Export anyway", then schedules `mixdown()` on the
engine worker via `run_coro` and reports through the Transport dock's
progress bar. A document with no clips at all still gets the whole-text
`start_conversion()` path - that's unchanged, this dialog is for clip
documents.
"""
from __future__ import annotations

import asyncio
import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLineEdit,
    QMessageBox, QPushButton, QWidget,
)

from kokoro_gui.daw.arrangement import compute_arrangement
from kokoro_gui.daw.mixdown import mixdown

FORMATS = ("wav", "mp3", "flac", "ogg")


def export_defaults(app) -> dict:
    project = dict(app.project_settings.get("export", {})) if isinstance(app.project_settings, dict) else {}
    return {
        "out_dir": project.get("out_dir", app.settings.get("out_dir", "audio_output")),
        "filename": project.get("filename", app.settings.get("filename", "output")),
        "format": project.get("format", app.settings.get("format", "wav")),
        "srt": bool(project.get("srt", app.settings.get("export_subtitles", False))),
        "keep_clip_files": bool(project.get("keep_clip_files", app.settings.get("separate", False))),
    }


class ExportDialog(QDialog):
    exportFinished = Signal(bool, str)

    def __init__(self, app, parent=None):
        super().__init__(parent or app)
        self.app = app
        self.setWindowTitle("Export")
        values = export_defaults(app)

        form = QFormLayout(self)
        dir_row = QWidget()
        dir_layout = QHBoxLayout(dir_row)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        self.out_dir_edit = QLineEdit(values["out_dir"])
        browse = QPushButton("...")
        browse.clicked.connect(self._browse_dir)
        dir_layout.addWidget(self.out_dir_edit, 1)
        dir_layout.addWidget(browse)
        form.addRow("Output folder:", dir_row)

        self.filename_edit = QLineEdit(values["filename"])
        form.addRow("Base filename:", self.filename_edit)

        self.format_combo = QComboBox()
        self.format_combo.addItems(FORMATS)
        self.format_combo.setCurrentText(values["format"] if values["format"] in FORMATS else "wav")
        form.addRow("Format:", self.format_combo)

        self.srt_check = QCheckBox("Also write .srt subtitles")
        self.srt_check.setChecked(values["srt"])
        form.addRow("", self.srt_check)

        self.keep_clips_check = QCheckBox("Keep per-clip files next to the mixdown")
        self.keep_clips_check.setChecked(values["keep_clip_files"])
        form.addRow("", self.keep_clips_check)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Export")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        form.addRow(self.buttons)

    def _browse_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select output folder", self.out_dir_edit.text())
        if d:
            self.out_dir_edit.setText(d)

    def values(self) -> dict:
        return {
            "out_dir": self.out_dir_edit.text().strip() or "audio_output",
            # basename(): the free-text filename is a path sink, same
            # sanitization _assemble_config applies.
            "filename": os.path.basename(self.filename_edit.text().strip()) or "output",
            "format": self.format_combo.currentText(),
            "srt": self.srt_check.isChecked(),
            "keep_clip_files": self.keep_clips_check.isChecked(),
        }


def run_export(app, values: dict, parent=None) -> bool:
    """Validates, remembers `values` in the project, and schedules the
    mixdown. Returns False when nothing was scheduled."""
    parent = parent or app
    document = app.document
    if not document.clips:
        QMessageBox.information(parent, "Nothing to export",
                                "This project has no clips yet. Assign characters to text and generate first.")
        return False
    if app.transport_dock.is_busy():
        QMessageBox.warning(parent, "Busy", "Finish or cancel the current job before exporting.")
        return False

    dirty = document.dirty_clips()
    if dirty:
        box = QMessageBox(parent)
        box.setWindowTitle("Clips out of date")
        box.setText(f"{len(dirty)} clip(s) are out of date and will be silent in the export.")
        generate_btn = box.addButton("Generate first", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Export anyway", QMessageBox.ButtonRole.ActionRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is generate_btn:
            app.on_generate_clicked()
            return False
        if clicked is None or box.buttonRole(clicked) == QMessageBox.ButtonRole.RejectRole:
            return False

    app.project_settings["export"] = dict(values)
    app.schedule_save()

    out_path = os.path.join(values["out_dir"], f"{values['filename']}.{values['format']}")
    arrangement = compute_arrangement(document, engine_id=app.backend.id)
    sample_rate = app.project_sample_rate()

    app.transport_dock.set_busy(True)
    app.transport_dock.set_status("Exporting...", "busy")
    app.transport_dock.set_progress(0, "")

    def _progress(fraction: float, detail: str) -> None:
        app.exportProgress.emit(fraction * 100.0, detail)

    async def _run():
        return await asyncio.to_thread(
            mixdown, document, out_path, values["format"], sample_rate,
            values["srt"], values["keep_clip_files"], arrangement, app.backend.id, _progress,
        )

    def _done(future):
        try:
            result = future.result()
            extras = []
            if result.srt_path:
                extras.append("srt")
            if result.clip_files:
                extras.append(f"{len(result.clip_files)} clip files")
            suffix = f" (+ {', '.join(extras)})" if extras else ""
            app.exportFinished.emit(True, f"Exported {result.audio_path}{suffix}")
        except Exception as e:  # noqa: BLE001 - surfaced to the status line
            app.exportFinished.emit(False, f"Export failed: {e}")

    future = app.engine.worker.run_coro(_run())
    future.add_done_callback(_done)
    return True
