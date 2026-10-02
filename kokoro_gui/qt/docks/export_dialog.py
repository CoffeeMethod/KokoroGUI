"""File > Export... (section 6 of Claude/PLAN_ui_shell_redesign.md).

Three tabs. "Audio" holds the output folder, base filename (a template:
`{project}`, `{date}`, `{time}`, `{range}`, see `mixdown.expand_name`, with
a live preview underneath), format, the mp3 bitrate (shown for mp3 only),
sample rate, channels (stereo, or mono as the average of the two), "Normalize
loudness" (a LUFS target under a true-peak ceiling, `kokoro_gui.audio.loudness`)
and a range (the whole project or between two markers). "Extras" holds "also
write .srt" (per clip or per word), "also write a cue sheet (.csv)"
(kokoro_gui/daw/mixdown.py's `write_cue_sheet`) and "keep per-clip files".
"Project file" holds the bundle options below. A new option goes into the tab
it belongs to, in the same `values()` and `export_defaults()` pair.
Values persist per project in
`app.project_settings["export"]`, falling back to the old `config_qt.json`
keys (`out_dir`/`filename`/`format`/`export_subtitles`/`separate`) so an
existing user's choices carry over.

Also the project bundle's options (grill TB14, TB16): whether generated
audio goes into the `.tbaw` and in which format (wav or flac for new
segments), and whether the reference video goes in (off by default). They live in `app.project_settings["bundle"]`
(`kokoro_gui.qt.project.bundle_options`) and are applied on OK, before the
dirty-clips prompt.

`run_export()` refuses (with the count) while any clip is dirty, offering
"Generate first" / "Export anyway", asks "Replace / Add number / Cancel"
when the file it would write already exists, then schedules `mixdown()` on the
engine worker via `run_coro` and reports through the Transport dock's
progress bar. A document with no clips at all still gets the whole-text
`start_conversion()` path - that's unchanged, this dialog is for clip
documents.

`run_measure_loudness()` (File > Measure Loudness...) renders the same mix
on the worker thread and shows a `LoudnessDialog`; no file is written.
"""
from __future__ import annotations

import asyncio
import math
import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QTabWidget, QVBoxLayout, QWidget,
)

from kokoro_gui.audio import loudness as loudness_mod
from kokoro_gui.daw import markers as marker_ops
from kokoro_gui.daw.mixdown import expand_name, mixdown, name_context, render_mix, unused_path
from kokoro_gui.qt import project as project_io

FORMATS = ("wav", "mp3", "flac", "ogg")
BUNDLE_AUDIO_FORMATS = ("wav", "flac")
BITRATES_KBPS = (128, 192, 256, 320)
DEFAULT_BITRATE_KBPS = 192
OUTPUT_RATES = (22050, 24000, 44100, 48000)
DEFAULT_TARGET_LUFS = -16.0
DEFAULT_CEILING_DBTP = -1.0
NAME_TOKEN_HELP = ("Tokens: {project} (the project title), {date} (YYYY-MM-DD), {time} (HHMMSS), "
                   "{range} (the range below, or \"full\").")


def _choice(value, allowed: tuple, default):
    """`value` when it is one of `allowed`, else `default` (a hand-edited
    project.json)."""
    return value if value in allowed and not isinstance(value, bool) else default


def _export_target(app):
    """`(document, project_settings)` of what Export writes: the project the
    timeline shows (`app.level`, phase 4), its subprojects as their
    mixdowns."""
    level = getattr(app, "level", None)
    if level is not None:
        return level.document, level.project_settings
    return app.document, app.project_settings


def export_defaults(app) -> dict:
    _document, settings = _export_target(app)
    project = dict(settings.get("export", {})) if isinstance(settings, dict) else {}
    return {
        "out_dir": project.get("out_dir", app.settings.get("out_dir", "audio_output")),
        "filename": project.get("filename", app.settings.get("filename", "output")),
        "format": project.get("format", app.settings.get("format", "wav")),
        "srt": bool(project.get("srt", app.settings.get("export_subtitles", False))),
        "keep_clip_files": bool(project.get("keep_clip_files", app.settings.get("separate", False))),
        "channels": 1 if project.get("channels") == 1 else 2,
        "srt_words": bool(project.get("srt_words", False)),
        "cue_sheet": bool(project.get("cue_sheet", False)),
        "normalize_loudness": bool(project.get("normalize_loudness", False)),
        "target_lufs": _number(project.get("target_lufs"), DEFAULT_TARGET_LUFS, -30.0, -5.0),
        "ceiling_dbtp": _number(project.get("ceiling_dbtp"), DEFAULT_CEILING_DBTP, -6.0, 0.0),
        "bitrate_kbps": _choice(project.get("bitrate_kbps"), BITRATES_KBPS, DEFAULT_BITRATE_KBPS),
        "sample_rate": _choice(project.get("sample_rate"), OUTPUT_RATES, None),  # None: the project's rate
    }


def output_name(app, template: str, range_label: str | None = None) -> str:
    """The base filename `template` expands to for this project: its title
    (`project_io.display_title`), today's date and time, and the range."""
    _document, settings = _export_target(app)
    context = name_context(project_io.display_title(settings, getattr(app, "project_path", None)), range_label)
    return expand_name(template, context)


def _number(value, default: float, lo: float, hi: float) -> float:
    """`value` as a float inside `[lo, hi]`, or `default` when it isn't a
    number (a hand-edited project.json)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(hi, max(lo, number)) if math.isfinite(number) else default


def format_levels(report) -> str:
    """"-16.0 LUFS, -1.2 dBTP", or plain words for silence."""
    if report is None or not math.isfinite(report.integrated_lufs):
        return "no measurable loudness"
    return f"{report.integrated_lufs:.1f} LUFS, {report.true_peak_dbtp:.1f} dBTP"


def _mix_inputs(app) -> dict:
    """The keyword arguments `render_mix` and `mixdown` share, resolved on
    the GUI thread: the arrangement, the mix rate, each clip's
    post-processing (it reads dock state) and each subproject's mixdown file
    (phase 4)."""
    level = getattr(app, "level", None)
    arrangement = app.build_arrangement()
    post_configs = {p.clip.id: app.post_config_for_clip(p.clip, level) for p in arrangement.placed}
    nested = {p.clip.id: app.nested_audio_path(p.clip, level)
              for p in arrangement.placed if p.clip.is_nested} if hasattr(app, "nested_audio_path") else {}
    return {
        "sample_rate": app.project_sample_rate(),
        "arrangement": arrangement,
        "engine_id": app.backend.id,
        "post_config_for_clip": lambda clip: post_configs.get(clip.id),
        "nested_audio_path": lambda clip: nested.get(clip.id),
    }


class ExportDialog(QDialog):
    exportFinished = Signal(bool, str)

    def __init__(self, app, parent=None):
        super().__init__(parent or app)
        self.app = app
        self.setWindowTitle("Export")
        values = export_defaults(app)

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.audio_form = self._add_tab("Audio")
        self.extras_form = self._add_tab("Extras")
        self.project_form = self._add_tab("Project file")
        self._build_audio_tab(self.audio_form, values)
        self._build_extras_tab(self.extras_form, values)
        self._build_project_tab(self.project_form)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Export")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._sync_bitrate_row()
        self._update_name_preview()

    def _add_tab(self, title: str) -> QFormLayout:
        page = QWidget()
        form = QFormLayout(page)
        self.tabs.addTab(page, title)
        return form

    def _build_audio_tab(self, form: QFormLayout, values: dict) -> None:
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
        self.filename_edit.setToolTip(NAME_TOKEN_HELP)
        form.addRow("Base filename:", self.filename_edit)
        self.name_preview_label = QLabel()
        self.name_preview_label.setToolTip(NAME_TOKEN_HELP)
        form.addRow("", self.name_preview_label)

        self.format_combo = QComboBox()
        self.format_combo.addItems(FORMATS)
        self.format_combo.setCurrentText(values["format"] if values["format"] in FORMATS else "wav")
        form.addRow("Format:", self.format_combo)

        self.bitrate_combo = QComboBox()
        for kbps in BITRATES_KBPS:
            self.bitrate_combo.addItem(f"{kbps} kbps", kbps)
        self.bitrate_combo.setCurrentIndex(self.bitrate_combo.findData(values["bitrate_kbps"]))
        self.bitrate_combo.setToolTip("Constant bitrate. At sample rates below 32 kHz, MP3 tops out at 160 kbps, "
                                      "so 192 and up come out at 160 there.")
        form.addRow("MP3 bitrate:", self.bitrate_combo)

        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItem(f"Project rate ({self.app.project_sample_rate()} Hz)", None)
        for rate in OUTPUT_RATES:
            self.sample_rate_combo.addItem(f"{rate} Hz", rate)
        self.sample_rate_combo.setCurrentIndex(max(0, self.sample_rate_combo.findData(values["sample_rate"])))
        self.sample_rate_combo.setToolTip("A different rate resamples the finished mix once before it is "
                                          "written. Per-clip files keep the project rate.")
        form.addRow("Sample rate:", self.sample_rate_combo)

        self.channels_combo = QComboBox()
        self.channels_combo.addItem("Stereo", 2)
        self.channels_combo.addItem("Mono", 1)
        self.channels_combo.setCurrentIndex(self.channels_combo.findData(values["channels"]))
        form.addRow("Channels:", self.channels_combo)

        # Normalize the mix to a loudness target (kokoro_gui.audio.loudness).
        self.normalize_check = QCheckBox("Normalize loudness")
        self.normalize_check.setChecked(values["normalize_loudness"])
        form.addRow("", self.normalize_check)
        self.target_spin = QDoubleSpinBox()
        self.target_spin.setRange(-30.0, -5.0)
        self.target_spin.setDecimals(1)
        self.target_spin.setSingleStep(0.5)
        self.target_spin.setValue(values["target_lufs"])
        self.target_spin.setToolTip("Integrated loudness the mix is brought to (ITU-R BS.1770, in LUFS).")
        form.addRow("Target (LUFS):", self.target_spin)
        self.ceiling_spin = QDoubleSpinBox()
        self.ceiling_spin.setRange(-6.0, 0.0)
        self.ceiling_spin.setDecimals(1)
        self.ceiling_spin.setSingleStep(0.5)
        self.ceiling_spin.setValue(values["ceiling_dbtp"])
        self.ceiling_spin.setToolTip("The gain stops short of the target rather than push the true peak past "
                                     "this. There is no limiter.")
        form.addRow("True peak ceiling (dBTP):", self.ceiling_spin)
        self.normalize_check.toggled.connect(self._sync_loudness_rows)
        if not loudness_mod.available():
            self.normalize_check.setChecked(False)
            self.normalize_check.setEnabled(False)
            self.normalize_check.setToolTip("Needs the pyloudnorm package (pip install pyloudnorm).")
        self._sync_loudness_rows()

        # Whole project, or between two markers (kokoro_gui/daw/markers.py).
        self.range_combo = QComboBox()
        self.range_combo.addItem("Whole project", None)
        found = marker_ops.list_markers(_export_target(self.app)[0].settings)
        for a, b in zip(found, found[1:]):
            self.range_combo.addItem(f"{a['name']} to {b['name']}", (a["seconds"], b["seconds"]))
        loop = self.app.loop_range() if hasattr(self.app, "loop_range") else None
        if loop is not None:
            self.range_combo.addItem("Loop region", loop)
        self.range_combo.setEnabled(self.range_combo.count() > 1)
        form.addRow("Range:", self.range_combo)

        self.filename_edit.textChanged.connect(self._update_name_preview)
        self.format_combo.currentTextChanged.connect(self._update_name_preview)
        self.format_combo.currentTextChanged.connect(self._sync_bitrate_row)
        self.range_combo.currentIndexChanged.connect(self._update_name_preview)

    def _build_extras_tab(self, form: QFormLayout, values: dict) -> None:
        self.srt_check = QCheckBox("Also write .srt subtitles")
        self.srt_check.setChecked(values["srt"])
        form.addRow("", self.srt_check)
        self.srt_words_check = QCheckBox("One subtitle per word")
        self.srt_words_check.setChecked(values["srt_words"])
        self.srt_words_check.setToolTip("Uses the word times stored with each segment; clips without them are left out.")
        form.addRow("", self.srt_words_check)

        self.cue_sheet_check = QCheckBox("Also write a cue sheet (.csv)")
        self.cue_sheet_check.setChecked(values["cue_sheet"])
        form.addRow("", self.cue_sheet_check)

        self.keep_clips_check = QCheckBox("Keep per-clip files next to the mixdown")
        self.keep_clips_check.setChecked(values["keep_clip_files"])
        form.addRow("", self.keep_clips_check)

    def _build_project_tab(self, form: QFormLayout) -> None:
        bundle = project_io.bundle_options(_export_target(self.app)[1])
        self.bundle_audio_check = QCheckBox("Bundle generated audio in the project file")
        self.bundle_audio_check.setChecked(bool(bundle["include_generated_audio"]))
        self.bundle_audio_check.setToolTip("Off gives a small .tbaw whose every clip regenerates on open.")
        form.addRow("Project:", self.bundle_audio_check)
        self.bundle_imported_check = QCheckBox("Bundle imported audio in the project file")
        self.bundle_imported_check.setChecked(bool(bundle["include_imported_audio"]))
        self.bundle_imported_check.setToolTip("Music beds, the source track and imported recordings, under "
                                              "audio/imported/. Off keeps them out of the .tbaw; a copy "
                                              "opened elsewhere plays without them.")
        form.addRow("", self.bundle_imported_check)
        self.bundle_format_combo = QComboBox()
        self.bundle_format_combo.addItems(BUNDLE_AUDIO_FORMATS)
        self.bundle_format_combo.setCurrentText(bundle["audio_format"])
        self.bundle_format_combo.setToolTip("Format of newly generated segments; existing ones keep theirs.")
        form.addRow("Bundle audio format:", self.bundle_format_combo)
        self.bundle_video_check = QCheckBox("Bundle the reference video in the project file")
        self.bundle_video_check.setChecked(bool(bundle["include_video"]))
        self.bundle_video_check.setToolTip("Stored uncompressed under video/. Off keeps only the video's path, "
                                           "so the .tbaw stays small.")
        form.addRow("", self.bundle_video_check)

    def _sync_bitrate_row(self, *_args) -> None:
        self.audio_form.setRowVisible(self.bitrate_combo, self.format_combo.currentText() == "mp3")

    def _update_name_preview(self, *_args) -> None:
        name = output_name(self.app, self.filename_edit.text(), self.range_label())
        self.name_preview_label.setText(f"Writes {name}.{self.format_combo.currentText()}")

    def range_label(self) -> str | None:
        """The range combo's text for `{range}`, None (so "full") for the whole project."""
        return None if self.range_combo.currentData() is None else self.range_combo.currentText()

    def _sync_loudness_rows(self) -> None:
        on = self.normalize_check.isChecked()
        self.target_spin.setEnabled(on)
        self.ceiling_spin.setEnabled(on)

    def _browse_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select output folder", self.out_dir_edit.text())
        if d:
            self.out_dir_edit.setText(d)

    def bundle_values(self) -> dict:
        return {
            "include_generated_audio": self.bundle_audio_check.isChecked(),
            "include_imported_audio": self.bundle_imported_check.isChecked(),
            "include_video": self.bundle_video_check.isChecked(),
            "audio_format": self.bundle_format_combo.currentText(),
        }

    def values(self) -> dict:
        return {
            "out_dir": self.out_dir_edit.text().strip() or "audio_output",
            # The template as typed (`{project}` and friends stay in it);
            # basename(): the free-text filename is a path sink, same
            # sanitization _assemble_config applies.
            "filename": os.path.basename(self.filename_edit.text().strip()) or "output",
            "format": self.format_combo.currentText(),
            "srt": self.srt_check.isChecked(),
            "keep_clip_files": self.keep_clips_check.isChecked(),
            "channels": self.channels_combo.currentData(),
            "srt_words": self.srt_words_check.isChecked(),
            "cue_sheet": self.cue_sheet_check.isChecked(),
            "normalize_loudness": self.normalize_check.isChecked(),
            "target_lufs": self.target_spin.value(),
            "ceiling_dbtp": self.ceiling_spin.value(),
            "bitrate_kbps": self.bitrate_combo.currentData(),
            "sample_rate": self.sample_rate_combo.currentData(),
        }

    def range_s(self):
        """`(start_s, end_s)` or None for the whole project. Not saved with
        the export settings: markers move."""
        data = self.range_combo.currentData()
        return tuple(data) if data else None


def _ask_existing(parent, path: str) -> str | None:
    """"replace", "number" or None (cancel) for an output file that already exists."""
    box = QMessageBox(parent)
    box.setWindowTitle("File exists")
    box.setText(f"{os.path.basename(path)} already exists in {os.path.dirname(path) or '.'}.")
    replace_btn = box.addButton("Replace", QMessageBox.ButtonRole.DestructiveRole)
    number_btn = box.addButton("Add number", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
    box.exec()
    clicked = box.clickedButton()
    if clicked is replace_btn:
        return "replace"
    return "number" if clicked is number_btn else None


def run_export(app, values: dict, parent=None, bundle: dict | None = None, range_s=None,
               range_label: str | None = None) -> bool:
    """Validates, remembers `values` (and the `bundle` options, if given) in
    the project, and schedules the mixdown. `range_label` fills `{range}` in
    the filename template. Returns False when nothing was scheduled."""
    parent = parent or app
    document, project_settings = _export_target(app)
    if bundle is not None:
        project_settings["bundle"] = project_io.bundle_options({"bundle": bundle})
        app.schedule_save()
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

    # The stored filename is the template; the file gets the expanded name.
    name = output_name(app, values["filename"], range_label)
    out_path = os.path.join(values["out_dir"], f"{name}.{values['format']}")
    if os.path.exists(out_path):
        choice = _ask_existing(parent, out_path)
        if choice is None:
            return False
        if choice == "number":
            out_path = unused_path(out_path)

    project_settings["export"] = dict(values)
    app.schedule_save()

    # Resolved on the GUI thread (they read dock state); the export thread
    # only applies them.
    inputs = _mix_inputs(app)
    loudness = {"target_lufs": values.get("target_lufs", DEFAULT_TARGET_LUFS),
                "ceiling_dbtp": values.get("ceiling_dbtp", DEFAULT_CEILING_DBTP)} \
        if values.get("normalize_loudness") and loudness_mod.available() else None

    app.transport_dock.set_busy(True)
    app.transport_dock.set_status("Exporting...", "busy")
    app.transport_dock.set_progress(0, "")

    def _progress(fraction: float, detail: str) -> None:
        app.exportProgress.emit(fraction * 100.0, detail)

    async def _run():
        return await asyncio.to_thread(
            mixdown, document, out_path, fmt=values["format"], include_srt=values["srt"],
            keep_clip_files=values["keep_clip_files"], progress=_progress, channels=values.get("channels", 2),
            range_s=range_s, srt_granularity="word" if values.get("srt_words") else "clip",
            include_cue_sheet=bool(values.get("cue_sheet")), loudness=loudness,
            bitrate_kbps=values.get("bitrate_kbps"), out_rate=values.get("sample_rate"), **inputs,
        )

    def _done(future):
        try:
            result = future.result()
            extras = []
            if result.srt_path:
                extras.append("srt")
            if result.clip_files:
                extras.append(f"{len(result.clip_files)} clip files")
            if result.cue_sheet_path:
                extras.append("cue sheet")
            suffix = f" (+ {', '.join(extras)})" if extras else ""
            if result.loudness_after is not None:
                suffix += f". Measured {format_levels(result.loudness_after)}"
                if result.loudness_limited:
                    suffix += "; target not reached: peak-limited"
            app.exportWrote.emit(result.audio_path)
            app.exportFinished.emit(True, f"Exported {result.audio_path}{suffix}")
        except Exception as e:  # noqa: BLE001 - surfaced to the status line
            app.exportFinished.emit(False, f"Export failed: {e}")

    future = app.backend.run(_run())
    future.add_done_callback(_done)
    return True


class LoudnessDialog(QDialog):
    """The numbers `loudness.measure` found for the project's mix."""

    def __init__(self, report, note: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Loudness")
        self.report = report
        form = QFormLayout(self)
        self.value_labels: dict = {}
        for key, label, text in (
            ("integrated", "Integrated loudness:", _level(report.integrated_lufs, "LUFS")),
            ("true_peak", "True peak:", _level(report.true_peak_dbtp, "dBTP")),
            ("sample_peak", "Sample peak:", _level(report.sample_peak_dbfs, "dBFS")),
            ("rms", "RMS:", _level(report.rms_dbfs, "dBFS")),
            ("noise_floor", "Noise floor:", _level(report.noise_floor_dbfs, "dBFS")),
            ("duration", "Length:", f"{report.duration_s:.1f} s"),
        ):
            value = QLabel(text)
            self.value_labels[key] = value
            form.addRow(label, value)
        hint = QLabel("The noise floor is the quietest tenth of the 50 ms windows, the way ACX measures it.")
        hint.setWordWrap(True)
        form.addRow(hint)
        if note:
            self.note_label = QLabel(note)
            self.note_label.setWordWrap(True)
            form.addRow(self.note_label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        form.addRow(buttons)


def _level(value: float, unit: str) -> str:
    return f"{value:.1f} {unit}" if math.isfinite(value) else "silent"


def run_measure_loudness(app, parent=None) -> bool:
    """File > Measure Loudness...: renders the mix `run_export` would write
    (the export dialog's channel choice, the whole project) on the engine
    worker, measures it and hands the report to `app.loudnessMeasured`. No
    file is written. Returns False when nothing was scheduled."""
    parent = parent or app
    document, _project_settings = _export_target(app)
    if not document.clips:
        QMessageBox.information(parent, "Nothing to measure",
                                "This project has no clips yet. Assign characters to text and generate first.")
        return False
    if app.transport_dock.is_busy():
        QMessageBox.warning(parent, "Busy", "Finish or cancel the current job before measuring.")
        return False
    if not loudness_mod.available():
        QMessageBox.warning(parent, "Loudness", "Measuring loudness needs the pyloudnorm package "
                                                "(pip install pyloudnorm).")
        return False

    channels = export_defaults(app)["channels"]
    inputs = _mix_inputs(app)
    sample_rate = inputs["sample_rate"]
    stale = len(document.dirty_clips())

    app.transport_dock.set_busy(True)
    app.transport_dock.set_status("Measuring loudness...", "busy")
    app.transport_dock.set_progress(0, "")

    def _progress(fraction: float, detail: str) -> None:
        app.exportProgress.emit(fraction * 100.0, detail)

    def _job():
        mix = render_mix(document, channels=channels, progress=_progress, **inputs)
        _progress(0.85, "Measuring loudness")
        return loudness_mod.measure(mix.samples, sample_rate)

    async def _run():
        return await asyncio.to_thread(_job)

    note = "Measured as " + ("mono." if channels == 1 else "stereo, the export's default.")
    if stale:
        note += f" {stale} clip(s) are out of date and count as silence."

    def _done(future):
        try:
            app.loudnessMeasured.emit(future.result(), note)
        except Exception as e:  # noqa: BLE001 - surfaced to the status line
            app.loudnessMeasured.emit(None, f"Measuring loudness failed: {e}")

    future = app.backend.run(_run())
    future.add_done_callback(_done)
    return True
