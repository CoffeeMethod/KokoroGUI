"""File > Export... (section 6 of Claude/PLAN_ui_shell_redesign.md).

Four tabs. "Audio" holds a preset (`kokoro_gui/daw/export_presets.py`: ACX,
Apple Podcasts, Spotify, YouTube; it fills the fields below, and editing one
puts the combo back on "Custom"), the output folder, base filename (a template:
`{project}`, `{date}`, `{time}`, `{range}`, see `mixdown.expand_name`, with
a live preview underneath), format, the mp3 bitrate (shown for mp3 only),
sample rate, channels (stereo, or mono as the average of the two), "Normalize
loudness" (by LUFS under a true-peak ceiling, or by RMS under a peak limiter,
`kokoro_gui.audio.loudness`), silence at the start and end, a range (the whole
project or between two markers) and "Split into" (one file, or one per
subproject or marker range, named `NN - <title>`; the range is ignored then).
After an export that split, failed a preset check or warned, the report dialog
lists every file. "Extras" holds "also
write .srt" (per clip or per word), "also write a cue sheet (.csv)"
(kokoro_gui/daw/mixdown.py's `write_cue_sheet`), "keep per-clip files" and the
stems: one file per track or per character, and a dialogue stem without the
music (`render_mix(stems=...)`, named `<base>_<stem>.<ext>`), and the text
files of `kokoro_gui/daw/transcripts.py` (WebVTT, a speaker SRT, Podcasting 2.0
transcript and chapters JSON, plain text, show notes; stored as the list
`export["extras"]`, with `export["transcript_speakers"]` for the names).
"Tags" holds the file tags (`kokoro_gui/daw/tagging.py`): title (the project's
when left empty), artist, show, episode or track number, year, description, a
cover image (a path with a 64 px preview; the picture is never copied into the
project) and "Write chapter markers into MP3". They're stored as the dict
`export["tags"]` and written into mp3, flac and ogg files after the export; a
wav has no tags, and a split export titles each file with its chapter. The tab
is disabled without `mutagen`.
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

from PySide6.QtCore import QRegularExpression, Qt, Signal
from PySide6.QtGui import QPixmap, QRegularExpressionValidator
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
    QWidget,
)

from kokoro_gui.audio import loudness as loudness_mod
from kokoro_gui.daw import markers as marker_ops
from kokoro_gui.daw.export_presets import CUSTOM_ID, PRESETS, SPLIT_MODES, get_preset, preset_ids
from kokoro_gui.daw.mixdown import (
    STEM_MODES, expand_name, mixdown, mixdown_chapters, name_context, plan_chapters, render_mix, unused_path,
)
from kokoro_gui.daw import tagging
from kokoro_gui.daw.transcripts import clean_extras
from kokoro_gui.qt import project as project_io

FORMATS = ("wav", "mp3", "flac", "ogg")
BUNDLE_AUDIO_FORMATS = ("wav", "flac")
BITRATES_KBPS = (128, 192, 256, 320)
DEFAULT_BITRATE_KBPS = 192
OUTPUT_RATES = (22050, 24000, 44100, 48000)
DEFAULT_TARGET_LUFS = -16.0
DEFAULT_CEILING_DBTP = -1.0
DEFAULT_RMS_DBFS = -20.0
DEFAULT_LIMITER_DBFS = -3.5
NORMALIZE_MODES = ("lufs", "rms")
SPLIT_LABELS = {None: "One file", "subprojects": "One file per subproject", "markers": "One file per marker range"}
STEM_LABELS = {None: "None", "track": "One per track", "character": "One per character"}
STEM_TIP = ("Each stem is the mix with only that track's (or character's) clips, the same length as the mix, so "
            "they line up at 0 in an editor. They get the mix's loudness gain (and, in RMS mode, its peak "
            "limiter), so they add up to it. A music bed under ducking is ducked by the whole mix's speech, "
            "not just the stem's. A project with timecode turned on adds its start timecode to each name "
            "(<name>_<stem>_01000000.wav); the audio still starts at 0.")
EXTRA_LABELS = {
    "vtt": "WebVTT transcript (.vtt)",
    "srt_speakers": "SRT transcript (.speakers.srt)",
    "transcript_json": "Podcasting 2.0 transcript (.transcript.json)",
    "txt": "Plain text transcript (.txt)",
    "chapters_json": "Podcasting 2.0 chapters (.chapters.json)",
    "show_notes": "Show notes with timestamps (.show-notes.md)",
}
EXTRA_TIPS = {
    "srt_speakers": "Next to the plain .srt, which has no names. Clips with no character get no name.",
    "chapters_json": "One chapter per marker, or per subproject when there are no markers.",
    "show_notes": "A Markdown list, one chapter per line with its time and the marker's note under it. "
                  "Chapters come from the markers, or the subprojects when there are none.",
}
# The genre tag a preset implies; a Custom export and YouTube get none.
PRESET_GENRES = {"acx": "Audiobook", "apple": "Podcast", "apple_mono": "Podcast", "spotify": "Podcast"}
COVER_PREVIEW_PX = 64
MAX_PAD_S = 10.0
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
        "normalize_mode": _choice(project.get("normalize_mode"), NORMALIZE_MODES, "lufs"),
        "target_rms_dbfs": _number(project.get("target_rms_dbfs"), DEFAULT_RMS_DBFS, -40.0, -6.0),
        "limiter_dbfs": _number(project.get("limiter_dbfs"), DEFAULT_LIMITER_DBFS, -12.0, 0.0),
        "head_s": _number(project.get("head_s"), 0.0, 0.0, MAX_PAD_S),
        "tail_s": _number(project.get("tail_s"), 0.0, 0.0, MAX_PAD_S),
        "split": _choice(project.get("split"), SPLIT_MODES, None),
        "preset": _choice(project.get("preset"), preset_ids(), CUSTOM_ID),
        "stems": _choice(project.get("stems"), STEM_MODES, None),
        "dialogue_stem": bool(project.get("dialogue_stem", False)),
        "extras": clean_extras(project.get("extras")),
        "transcript_speakers": bool(project.get("transcript_speakers", True)),
        "tags": tagging.clean_settings(project.get("tags")),
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


def chapter_plan(app, split: str, preset=None, arrangement=None):
    """The files a split export writes for the project the timeline shows
    (`mixdown.plan_chapters`); `preset` supplies the length limit a chapter
    is cut at."""
    document, _settings = _export_target(app)
    max_file_s = preset.max_minutes * 60.0 if preset is not None and preset.max_minutes else None
    return plan_chapters(document, arrangement or app.build_arrangement(), split, max_file_s)


def tag_options(app, values: dict) -> dict | None:
    """`mixdown(tags=...)` for the stored values: the tag fields (the title
    falls back to the project's, the genre comes from the preset), the cover
    path and the chapter flag. None when tags are switched off or `mutagen`
    is missing."""
    tags = tagging.clean_settings(values.get("tags"))
    if not tags["enabled"] or not tagging.available():
        return None
    _document, settings = _export_target(app)
    preset = get_preset(values.get("preset"))
    return {
        "title": tags["title"] or project_io.display_title(settings, getattr(app, "project_path", None)),
        "artist": tags["artist"], "album": tags["album"], "track": tags["track"], "year": tags["year"],
        "description": tags["description"], "genre": PRESET_GENRES.get(preset.id, "") if preset else "",
        "cover": tags["cover"], "chapters": tags["chapters"],
    }


def _loudness_options(values: dict):
    """`mixdown(loudness=...)` for the stored values, or None when
    "Normalize loudness" is off or pyloudnorm is missing."""
    if not values.get("normalize_loudness") or not loudness_mod.available():
        return None
    if values.get("normalize_mode") == "rms":
        return {"mode": "rms", "target_rms_dbfs": values.get("target_rms_dbfs", DEFAULT_RMS_DBFS),
                "limiter_dbfs": values.get("limiter_dbfs", DEFAULT_LIMITER_DBFS)}
    return {"mode": "lufs", "target_lufs": values.get("target_lufs", DEFAULT_TARGET_LUFS),
            "ceiling_dbtp": values.get("ceiling_dbtp", DEFAULT_CEILING_DBTP)}


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
        self.tags_form = self._add_tab("Tags")
        self.project_form = self._add_tab("Project file")
        self._build_audio_tab(self.audio_form, values)
        self._build_extras_tab(self.extras_form, values)
        self._build_tags_tab(self.tags_form, values["tags"])
        self._build_project_tab(self.project_form)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Export")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._sync_bitrate_row()
        self._sync_split()
        self._sync_tags()
        self._update_name_preview()

    def _add_tab(self, title: str) -> QFormLayout:
        page = QWidget()
        form = QFormLayout(page)
        self.tabs.addTab(page, title)
        return form

    def _build_audio_tab(self, form: QFormLayout, values: dict) -> None:
        self._applying = False  # True while a preset fills the fields, so that doesn't read as an edit
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Custom", CUSTOM_ID)
        for preset in PRESETS:
            self.preset_combo.addItem(preset.label, preset.id)
        self.preset_combo.setToolTip("Fills the format, loudness, silence and split fields below with what the "
                                     "platform asks for, and checks each file after the export. Changing a field "
                                     "afterwards goes back to Custom.")
        form.addRow("Preset:", self.preset_combo)

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
        self.normalize_mode_combo = QComboBox()
        self.normalize_mode_combo.addItem("Loudness (LUFS)", "lufs")
        self.normalize_mode_combo.addItem("RMS level, with a peak limiter", "rms")
        self.normalize_mode_combo.setCurrentIndex(self.normalize_mode_combo.findData(values["normalize_mode"]))
        self.normalize_mode_combo.setToolTip("ACX measures RMS and peaks; the podcast platforms measure LUFS.")
        form.addRow("Normalize by:", self.normalize_mode_combo)
        self.rms_spin = QDoubleSpinBox()
        self.rms_spin.setRange(-40.0, -6.0)
        self.rms_spin.setDecimals(1)
        self.rms_spin.setSingleStep(0.5)
        self.rms_spin.setValue(values["target_rms_dbfs"])
        self.rms_spin.setToolTip("RMS level of the whole file the mix is brought to.")
        form.addRow("Target RMS (dBFS):", self.rms_spin)
        self.limiter_spin = QDoubleSpinBox()
        self.limiter_spin.setRange(-12.0, 0.0)
        self.limiter_spin.setDecimals(1)
        self.limiter_spin.setSingleStep(0.5)
        self.limiter_spin.setValue(values["limiter_dbfs"])
        self.limiter_spin.setToolTip("No sample goes above this. The limiter dips the gain around a peak "
                                     "instead of clipping it.")
        form.addRow("Peak limiter (dBFS):", self.limiter_spin)
        self.normalize_check.toggled.connect(self._sync_loudness_rows)
        self.normalize_mode_combo.currentIndexChanged.connect(self._sync_loudness_rows)
        if not loudness_mod.available():
            self.normalize_check.setChecked(False)
            self.normalize_check.setEnabled(False)
            self.normalize_check.setToolTip("Needs the pyloudnorm package (pip install pyloudnorm).")
            for row in range(1, self.preset_combo.count()):
                self.preset_combo.model().item(row).setEnabled(False)
            self.preset_combo.setToolTip("Presets check the loudness, which needs the pyloudnorm package "
                                         "(pip install pyloudnorm).")
        self._sync_loudness_rows()

        self.head_spin = self._seconds_spin(values["head_s"], "Silence added before the first sample.")
        form.addRow("Silence at start (s):", self.head_spin)
        self.tail_spin = self._seconds_spin(values["tail_s"], "Silence added after the last sample.")
        form.addRow("Silence at end (s):", self.tail_spin)

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

        # One file, or one per subproject or marker range (`mixdown.plan_chapters`).
        self.split_combo = QComboBox()
        for mode in SPLIT_MODES:
            self.split_combo.addItem(SPLIT_LABELS[mode], mode)
        document = _export_target(self.app)[0]
        available = {None: True, "subprojects": bool(document.nested_clips()),
                     "markers": len(marker_ops.list_markers(document.settings)) >= 2}
        for row, mode in enumerate(SPLIT_MODES):
            self.split_combo.model().item(row).setEnabled(available[mode])
        stored = max(0, self.split_combo.findData(values["split"]))
        self.split_combo.setCurrentIndex(stored if self.split_combo.model().item(stored).isEnabled() else 0)
        self.split_combo.setToolTip("Subprojects and marker ranges each become a file named NN - <title>, in "
                                    "the output folder. The base filename and the range are not used.")
        form.addRow("Split into:", self.split_combo)

        self.preset_combo.setCurrentIndex(max(0, self.preset_combo.findData(values["preset"])))
        if not self.preset_combo.model().item(self.preset_combo.currentIndex()).isEnabled():
            self.preset_combo.setCurrentIndex(0)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        for signal in (
            self.format_combo.currentIndexChanged, self.bitrate_combo.currentIndexChanged,
            self.sample_rate_combo.currentIndexChanged, self.channels_combo.currentIndexChanged,
            self.normalize_check.toggled, self.normalize_mode_combo.currentIndexChanged,
            self.target_spin.valueChanged, self.ceiling_spin.valueChanged, self.rms_spin.valueChanged,
            self.limiter_spin.valueChanged, self.head_spin.valueChanged, self.tail_spin.valueChanged,
            self.split_combo.currentIndexChanged,
        ):
            signal.connect(self._mark_custom)
        self.split_combo.currentIndexChanged.connect(self._sync_split)
        self.split_combo.currentIndexChanged.connect(self._update_name_preview)
        self.preset_combo.currentIndexChanged.connect(self._update_name_preview)

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

        self.stems_combo = QComboBox()
        for mode in STEM_MODES:
            self.stems_combo.addItem(STEM_LABELS[mode], mode)
        self.stems_combo.setCurrentIndex(max(0, self.stems_combo.findData(values["stems"])))
        self.stems_combo.setToolTip(STEM_TIP)
        form.addRow("Stems:", self.stems_combo)
        self.dialogue_stem_check = QCheckBox("Also a dialogue stem (everything but the music)")
        self.dialogue_stem_check.setChecked(values["dialogue_stem"])
        self.dialogue_stem_check.setToolTip(STEM_TIP)
        form.addRow("", self.dialogue_stem_check)

        self.extra_checks = {}
        for number, (key, label) in enumerate(EXTRA_LABELS.items()):
            check = QCheckBox(label)
            check.setChecked(key in values["extras"])
            if key in EXTRA_TIPS:
                check.setToolTip(EXTRA_TIPS[key])
            self.extra_checks[key] = check
            form.addRow("Text files:" if number == 0 else "", check)
        self.speakers_check = QCheckBox("Speaker names in the transcripts")
        self.speakers_check.setChecked(values["transcript_speakers"])
        self.speakers_check.setToolTip("Off leaves the character names out of the WebVTT, SRT, JSON and text "
                                       "transcripts. A [Name:FX]: tag never appears in them either way.")
        form.addRow("", self.speakers_check)

    def _build_tags_tab(self, form: QFormLayout, tags: dict) -> None:
        self._stored_tags_enabled = tags["enabled"]  # kept as it was while mutagen is missing
        self.tags_check = QCheckBox("Write tags into the file")
        self.tags_check.setChecked(tags["enabled"])
        form.addRow("", self.tags_check)

        _document, settings = _export_target(self.app)
        self.tag_title_edit = QLineEdit(tags["title"])
        self.tag_title_edit.setPlaceholderText(project_io.display_title(settings, getattr(self.app, "project_path", None)))
        form.addRow("Title:", self.tag_title_edit)
        self.tag_artist_edit = QLineEdit(tags["artist"])
        form.addRow("Artist:", self.tag_artist_edit)
        self.tag_album_edit = QLineEdit(tags["album"])
        form.addRow("Album / show:", self.tag_album_edit)
        self.tag_track_edit = QLineEdit(tags["track"])
        self.tag_track_edit.setValidator(QRegularExpressionValidator(QRegularExpression(r"\d{0,4}")))
        form.addRow("Episode / track number:", self.tag_track_edit)
        self.tag_year_edit = QLineEdit(tags["year"])
        self.tag_year_edit.setValidator(QRegularExpressionValidator(QRegularExpression(r"\d{0,4}")))
        form.addRow("Year:", self.tag_year_edit)
        self.tag_description_edit = QPlainTextEdit(tags["description"])
        self.tag_description_edit.setFixedHeight(72)
        form.addRow("Description:", self.tag_description_edit)

        cover_row = QWidget()
        cover_layout = QHBoxLayout(cover_row)
        cover_layout.setContentsMargins(0, 0, 0, 0)
        self.cover_edit = QLineEdit(tags["cover"])
        self.cover_edit.setPlaceholderText("A JPEG or PNG, up to 5 MB")
        self.cover_edit.setToolTip("Embedded in mp3, flac and ogg files. The project keeps the path, not the picture.")
        self.cover_browse = QPushButton("...")
        self.cover_browse.clicked.connect(self._browse_cover)
        self.cover_preview = QLabel()
        self.cover_preview.setFixedSize(COVER_PREVIEW_PX, COVER_PREVIEW_PX)
        self.cover_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cover_layout.addWidget(self.cover_edit, 1)
        cover_layout.addWidget(self.cover_browse)
        cover_layout.addWidget(self.cover_preview)
        form.addRow("Cover image:", cover_row)
        self.cover_note_label = QLabel()
        self.cover_note_label.setWordWrap(True)
        form.addRow("", self.cover_note_label)

        self.tag_chapters_check = QCheckBox("Write chapter markers into MP3")
        self.tag_chapters_check.setChecked(tags["chapters"])
        self.tag_chapters_check.setToolTip("ID3 chapter frames: one per marker, or per subproject when there are "
                                           "no markers, so podcast apps show a chapter list. mp3 only.")
        form.addRow("", self.tag_chapters_check)
        self.tags_note_label = QLabel()
        self.tags_note_label.setWordWrap(True)
        form.addRow("", self.tags_note_label)

        if not tagging.available():
            self.tags_check.setChecked(False)
            self.tags_check.setEnabled(False)
            index = self.tabs.indexOf(self.tags_form.parentWidget())
            self.tabs.setTabEnabled(index, False)
            self.tabs.setTabToolTip(index, "Tags need the mutagen package (pip install mutagen).")
        for signal in (self.tags_check.toggled, self.format_combo.currentTextChanged,
                       self.split_combo.currentIndexChanged, self.cover_edit.textChanged):
            signal.connect(self._sync_tags)

    def _browse_cover(self) -> None:
        start = os.path.dirname(self.cover_edit.text().strip())
        path, _filter = QFileDialog.getOpenFileName(self, "Select cover image", start,
                                                    "Images (*.png *.jpg *.jpeg);;All files (*)")
        if path:
            self.cover_edit.setText(path)

    def _sync_tags(self, *_args) -> None:
        """Greys out what the tags can't use: everything when they're off, the
        title and number in a split export (each file gets its own) and the
        chapter box outside mp3. Shows the cover's preview, or why it won't
        be used."""
        on = self.tags_check.isChecked()
        fmt = self.format_combo.currentText()
        split = self.split_combo.currentData() is not None
        for widget in (self.tag_artist_edit, self.tag_album_edit, self.tag_year_edit, self.tag_description_edit,
                       self.cover_edit, self.cover_browse):
            widget.setEnabled(on)
        split_tip = "A split export titles each file with its chapter and numbers it, so this isn't used."
        self.tag_title_edit.setEnabled(on and not split)
        self.tag_title_edit.setToolTip(split_tip if split else "Left empty, the project's title is used.")
        self.tag_track_edit.setEnabled(on and not split)
        self.tag_track_edit.setToolTip(split_tip if split else "")
        self.tag_chapters_check.setEnabled(on and fmt == "mp3")
        cover = self.cover_edit.text().strip()
        self.cover_preview.clear()
        self.cover_note_label.setText("")
        if cover:
            try:
                _real, data, *_rest = tagging.check_cover(cover)
            except tagging.CoverError as e:
                reason = str(e)
                self.cover_note_label.setText(f"{reason[0].upper()}{reason[1:]}; it will be left out.")
            else:
                pixmap = QPixmap()
                if pixmap.loadFromData(data):
                    self.cover_preview.setPixmap(pixmap.scaled(
                        COVER_PREVIEW_PX, COVER_PREVIEW_PX, Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation))
                else:
                    self.cover_note_label.setText("Qt can't show this image, but it will still be embedded.")
        if not tagging.supports(fmt):
            note = "WAV files can't carry tags. Pick mp3, flac or ogg to write them."
        elif split:
            note = "Each file is titled with its chapter and numbered n/total; the rest is shared."
        else:
            note = ""
        self.tags_note_label.setText(note)

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
        fmt = self.format_combo.currentText()
        split = self.split_combo.currentData()
        if split:
            chapters = chapter_plan(self.app, split, self.preset()).chapters
            if chapters:
                more = f" and {len(chapters) - 1} more" if len(chapters) > 1 else ""
                self.name_preview_label.setText(f"Writes {chapters[0].name}.{fmt}{more}")
            else:
                self.name_preview_label.setText("Nothing to split: no chapters found")
            return
        name = output_name(self.app, self.filename_edit.text(), self.range_label())
        self.name_preview_label.setText(f"Writes {name}.{fmt}")

    def preset(self):
        """The chosen `ExportPreset`, or None for Custom."""
        return get_preset(self.preset_combo.currentData())

    def _seconds_spin(self, value: float, tip: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, MAX_PAD_S)
        spin.setDecimals(2)
        spin.setSingleStep(0.25)
        spin.setValue(value)
        spin.setToolTip(tip)
        return spin

    def _on_preset_changed(self, *_args) -> None:
        """Fills the fields from the chosen preset (`ExportPreset.values`)."""
        preset = self.preset()
        if preset is None:
            return
        v = preset.values
        self._applying = True
        try:
            self.format_combo.setCurrentText(v["format"])
            self.bitrate_combo.setCurrentIndex(max(0, self.bitrate_combo.findData(v["bitrate_kbps"])))
            self.sample_rate_combo.setCurrentIndex(max(0, self.sample_rate_combo.findData(v["sample_rate"])))
            self.channels_combo.setCurrentIndex(max(0, self.channels_combo.findData(v["channels"])))
            self.normalize_check.setChecked(v["normalize_loudness"])
            self.normalize_mode_combo.setCurrentIndex(max(0, self.normalize_mode_combo.findData(v["normalize_mode"])))
            self.target_spin.setValue(v["target_lufs"])
            self.ceiling_spin.setValue(v["ceiling_dbtp"])
            self.rms_spin.setValue(v["target_rms_dbfs"])
            self.limiter_spin.setValue(v["limiter_dbfs"])
            self.head_spin.setValue(v["head_s"])
            self.tail_spin.setValue(v["tail_s"])
            row = max(0, self.split_combo.findData(v["split"]))
            self.split_combo.setCurrentIndex(row if self.split_combo.model().item(row).isEnabled() else 0)
        finally:
            self._applying = False
        self._sync_loudness_rows()

    def _mark_custom(self, *_args) -> None:
        if not self._applying and self.preset_combo.currentData() != CUSTOM_ID:
            self.preset_combo.setCurrentIndex(0)

    def _sync_split(self, *_args) -> None:
        """The range doesn't apply to a split export."""
        self.range_combo.setEnabled(self.split_combo.currentData() is None and self.range_combo.count() > 1)

    def range_label(self) -> str | None:
        """The range combo's text for `{range}`, None (so "full") for the whole project."""
        return None if self.range_combo.currentData() is None else self.range_combo.currentText()

    def _sync_loudness_rows(self, *_args) -> None:
        on = self.normalize_check.isChecked()
        rms = self.normalize_mode_combo.currentData() == "rms"
        self.normalize_mode_combo.setEnabled(on)
        for lufs_row in (self.target_spin, self.ceiling_spin):
            lufs_row.setEnabled(on)
            self.audio_form.setRowVisible(lufs_row, not rms)
        for rms_row in (self.rms_spin, self.limiter_spin):
            rms_row.setEnabled(on)
            self.audio_form.setRowVisible(rms_row, rms)

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
            "normalize_mode": self.normalize_mode_combo.currentData(),
            "target_rms_dbfs": self.rms_spin.value(),
            "limiter_dbfs": self.limiter_spin.value(),
            "head_s": self.head_spin.value(),
            "tail_s": self.tail_spin.value(),
            "split": self.split_combo.currentData(),
            "preset": self.preset_combo.currentData(),
            "stems": self.stems_combo.currentData(),
            "dialogue_stem": self.dialogue_stem_check.isChecked(),
            "extras": [key for key, check in self.extra_checks.items() if check.isChecked()],
            "transcript_speakers": self.speakers_check.isChecked(),
            "tags": self.tag_values(),
        }

    def tag_values(self) -> dict:
        """The Tags tab as the dict stored in `export["tags"]`; the cover as
        an absolute path."""
        cover = self.cover_edit.text().strip()
        return tagging.clean_settings({
            "enabled": self.tags_check.isChecked() if tagging.available() else self._stored_tags_enabled,
            "title": self.tag_title_edit.text(), "artist": self.tag_artist_edit.text(),
            "album": self.tag_album_edit.text(), "track": self.tag_track_edit.text(),
            "year": self.tag_year_edit.text(), "description": self.tag_description_edit.toPlainText(),
            "cover": os.path.abspath(cover) if cover else "",
            "chapters": self.tag_chapters_check.isChecked(),
        })

    def range_s(self):
        """`(start_s, end_s)` or None for the whole project. Not saved with
        the export settings: markers move."""
        data = self.range_combo.currentData()
        return tuple(data) if data else None


def _ask_existing(parent, path: str, more: int = 0) -> str | None:
    """"replace", "number" or None (cancel) for an output file that already
    exists. `more` is how many other files of a split export exist too."""
    box = QMessageBox(parent)
    box.setWindowTitle("File exists")
    also = f" ({more} more of the files already exist)" if more else ""
    box.setText(f"{os.path.basename(path)} already exists in {os.path.dirname(path) or '.'}{also}.")
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

    # Resolved on the GUI thread (they read dock state); the export thread
    # only applies them.
    inputs = _mix_inputs(app)
    preset = get_preset(values.get("preset"))
    split = values.get("split")
    plan = None
    if split:
        plan = chapter_plan(app, split, preset, inputs["arrangement"])
        if not plan.chapters:
            QMessageBox.information(parent, "Nothing to split",
                                    "There are no subprojects or marker pairs to split the export by. "
                                    "Pick \"One file\" instead.")
            return False
        targets = [os.path.join(values["out_dir"], f"{c.name}.{values['format']}") for c in plan.chapters]
    else:
        # The stored filename is the template; the file gets the expanded name.
        name = output_name(app, values["filename"], range_label)
        targets = [os.path.join(values["out_dir"], f"{name}.{values['format']}")]
    existing = [t for t in targets if os.path.exists(t)]
    numbered = False
    if existing:
        choice = _ask_existing(parent, existing[0], len(existing) - 1) if len(existing) > 1 \
            else _ask_existing(parent, existing[0])
        if choice is None:
            return False
        numbered = choice == "number"
    out_path = unused_path(targets[0]) if numbered else targets[0]

    project_settings["export"] = dict(values)
    app.schedule_save()

    loudness = _loudness_options(values)
    # The checks read the loudness measurement, which needs pyloudnorm.
    checks = preset.checks if preset is not None and loudness_mod.available() else ()

    app.transport_dock.set_busy(True)
    app.transport_dock.set_status("Exporting...", "busy")
    app.transport_dock.set_progress(0, "")

    def _progress(fraction: float, detail: str) -> None:
        app.exportProgress.emit(fraction * 100.0, detail)

    options = dict(
        fmt=values["format"], include_srt=values["srt"], keep_clip_files=values["keep_clip_files"],
        progress=_progress, channels=values.get("channels", 2),
        srt_granularity="word" if values.get("srt_words") else "clip",
        include_cue_sheet=bool(values.get("cue_sheet")), loudness=loudness,
        bitrate_kbps=values.get("bitrate_kbps"), out_rate=values.get("sample_rate"),
        head_s=values.get("head_s", 0.0), tail_s=values.get("tail_s", 0.0), checks=checks,
        stems=values.get("stems"), dialogue_stem=bool(values.get("dialogue_stem")),
        extras=tuple(clean_extras(values.get("extras"))),
        transcript_speakers=bool(values.get("transcript_speakers", True)), tags=tag_options(app, values), **inputs,
    )

    async def _run():
        if plan is not None:
            return await asyncio.to_thread(mixdown_chapters, document, values["out_dir"], plan, numbered=numbered,
                                           **options)
        return await asyncio.to_thread(mixdown, document, out_path, range_s=range_s, **options)

    def _done(future):
        try:
            result = future.result()
            extras = []
            if values["srt"]:
                extras.append("srt")
            if result.clip_files:
                extras.append(f"{len(result.clip_files)} clip files")
            if values.get("cue_sheet"):
                extras.append("cue sheet")
            if result.stem_files:
                extras.append(f"{len(result.stem_files)} stem{'s' if len(result.stem_files) != 1 else ''}")
            if result.text_files:
                extras.append(f"{len(result.text_files)} text file{'s' if len(result.text_files) != 1 else ''}")
            if any(f.tagged for f in result.files):
                extras.append("tags")
            suffix = f" (+ {', '.join(extras)})" if extras else ""
            if len(result.files) > 1:
                message = f"Exported {len(result.files)} files to {os.path.dirname(result.audio_path)}{suffix}"
            else:
                message = f"Exported {result.audio_path}{suffix}"
                if result.loudness_after is not None:
                    message += f". Measured {format_levels(result.loudness_after)}"
                    if result.loudness_limited:
                        message += "; target not reached: peak-limited"
            failed = [f for f in result.files if f.failed]
            if checks:
                if not failed:
                    message += f". Passed the {preset.label} checks"
                elif len(result.files) > 1:
                    message += f"; {len(failed)} failed the {preset.label} checks (see report)"
                else:
                    message += f". Failed {len(failed[0].failed)} of the {preset.label} checks (see report)"
            for warning in result.warnings:
                message += f". {warning[0].upper()}{warning[1:]}"
            app.exportWrote.emit(result.audio_path)
            if len(result.files) > 1 or failed or result.warnings:
                app.exportReport.emit(result, preset.label if checks else "")
            app.exportFinished.emit(True, message)
        except Exception as e:  # noqa: BLE001 - surfaced to the status line
            app.exportFinished.emit(False, f"Export failed: {e}")

    future = app.backend.run(_run())
    future.add_done_callback(_done)
    return True


class ExportReportDialog(QDialog):
    """What a finished export wrote: one row per file with its measurements
    and, when a preset ran its checks, which of them it failed."""
    COLUMNS = ("File", "Length", "Loudness", "True peak", "RMS", "Noise floor", "Checks")

    def __init__(self, result, preset_label: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export report")
        self.resize(900, 360)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(len(result.files), len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        for row, entry in enumerate(result.files):
            report = entry.report
            if not preset_label:
                verdict = ""
            else:
                verdict = "Failed: " + "; ".join(entry.failed) if entry.failed else "Passed"
            cells = [
                os.path.basename(entry.path), _duration(entry.duration_s),
                _level(report.integrated_lufs, "LUFS") if report else "",
                _level(report.true_peak_dbtp, "dBTP") if report else "",
                _level(report.rms_dbfs, "dBFS") if report else "",
                _level(report.noise_floor_dbfs, "dBFS") if report else "",
                verdict,
            ]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(entry.path if column == 0 else verdict)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)
        self.summary_label = QLabel(self._summary(result, preset_label))
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _summary(result, preset_label: str) -> str:
        lines = []
        if preset_label:
            failed = sum(1 for f in result.files if f.failed)
            lines.append(f"{len(result.files) - failed} of {len(result.files)} files passed the "
                         f"{preset_label} checks.")
        lines.extend(f"{w[0].upper()}{w[1:]}." for w in result.warnings)
        return " ".join(lines)


def _duration(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02}:{secs:02}"


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
