"""Options > Settings...: the program and project settings, in one window
laid out like OBS's (a page list on the left, the page on the right, OK /
Cancel / Apply). The Settings tab keeps only what shapes a voice.

Pages:

- General: theme, device. (The engine a new character gets is the Settings
  tab's Engine row with nothing selected, beside the voice it defaults to.)
- Generation: where text is cut (`spec.SEGMENTATION_KEYS`) and the default
  output format. App settings, read by `SettingsDock.get_state()`.
- Performance: the segment cache (shared), then each installed engine's
  "Advanced" schema fields (threads, Audio8's reference-encoding cache),
  stored in that engine's `settings["engines"][<id>]` bucket.
- Project: the subproject's title (when one is focused), gaps (clip,
  paragraph, chapter, after a heading, the speaker-change jitter), the
  heading's speed, auto-crossfade, ripple, onset alignment.
  `Document.settings`.
- Timeline: track layout, ducking, timecode. `Document.settings`.
- Source track: the imported original dialogue's offset, or its removal.

Nothing is written until Apply or OK. Apply writes only the values that
differ from what the window opened with: app settings directly, each
project value as one `SetFieldCommand` (so a track layout change still
gets its relane as the same undo step), then rebuilds the pages from the
new state.
"""
from __future__ import annotations

import dataclasses
import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QPushButton, QScrollArea, QSpinBox, QStackedWidget, QVBoxLayout, QWidget,
)

from kokoro_gui.audio.mixer import DEFAULT_DUCK_DB
from kokoro_gui.daw.arrangement import (
    DEFAULT_GAP_S, DEFAULT_HEADING_GAP_S, DEFAULT_PARAGRAPH_GAP_S, align_onset_enabled, gap_jitter_range,
)
from kokoro_gui.daw.reference import SOURCE_TRACK_KEY, source_track_settings
from kokoro_gui.daw.timecode import FRAME_RATES, tc_to_frames, timecode_settings
from kokoro_gui.daw.undo import SetFieldCommand
from kokoro_gui.engines import registry as engine_registry
from kokoro_gui.engines.base import ConfigFieldType, per_engine_fields
from kokoro_gui.qt import project as project_io
from kokoro_gui.qt import spec, theme
from kokoro_gui.qt.schema_form import SchemaFormWidget

PAGES = ("General", "Generation", "Performance", "Project", "Timeline", "Source track")
DEVICES = (("auto", "Auto"), ("cpu", "CPU"), ("cuda", "CUDA"))
THEMES = (("light", "Light"), ("dark", "Dark"))


def _spin(lo: float, hi: float, step: float, value: float, decimals: int = 2) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    spin.setValue(value)
    return spin


def _combo(items, current) -> QComboBox:
    combo = QComboBox()
    for value, label in items:
        combo.addItem(label, value)
    index = combo.findData(current)
    combo.setCurrentIndex(index if index >= 0 else 0)
    return combo


def _row(*widgets, stretch_first: bool = False) -> QWidget:
    row = QWidget()
    box = QHBoxLayout(row)
    box.setContentsMargins(0, 0, 0, 0)
    for i, widget in enumerate(widgets):
        box.addWidget(widget, 1 if (stretch_first and i == 0) else 0)
    if not stretch_first:
        box.addStretch(1)
    return row


class SettingsWindow(QDialog):
    def __init__(self, app, parent=None):
        super().__init__(parent or app)
        self.setObjectName("settings_window")
        self.setWindowTitle("Settings")
        self.app = app
        self.widgets: dict = {}
        self.engine_forms: dict = {}
        self._remove_source = False

        self.page_list = QListWidget()
        self.page_list.addItems(PAGES)
        self.page_list.setFixedWidth(170)
        self.page_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.pages = QStackedWidget()
        self.page_list.currentRowChanged.connect(self.pages.setCurrentIndex)

        body = QHBoxLayout()
        body.addWidget(self.page_list)
        body.addWidget(self.pages, 1)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
                                        | QDialogButtonBox.StandardButton.Apply)
        self.apply_button = self.buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.buttons.accepted.connect(self._ok)
        self.buttons.rejected.connect(self.reject)
        self.apply_button.clicked.connect(self.apply)

        outer = QVBoxLayout(self)
        outer.addLayout(body, 1)
        outer.addWidget(self.buttons)
        self.resize(760, 520)

        self._build()
        self.page_list.setCurrentRow(0)

    # -- building ---------------------------------------------------------------

    def show_page(self, name: str) -> None:
        if name in PAGES:
            self.page_list.setCurrentRow(PAGES.index(name))

    def _build(self) -> None:
        row = max(0, self.page_list.currentRow())
        while self.pages.count():
            page = self.pages.widget(0)
            self.pages.removeWidget(page)
            page.setParent(None)
            page.deleteLater()
        self.widgets = {}
        self.engine_forms = {}
        self._remove_source = False
        for build in (self._build_general, self._build_generation, self._build_performance,
                      self._build_project, self._build_timeline, self._build_source_track):
            content = QWidget()
            layout = QVBoxLayout(content)
            build(layout)
            layout.addStretch(1)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(content)
            self.pages.addWidget(scroll)
        self.pages.setCurrentIndex(row)
        self._initial = self._values()
        self._watch_all()
        self._sync_apply()

    def _form(self, layout: QVBoxLayout, title: str) -> QFormLayout:
        box = QGroupBox(title)
        form = QFormLayout(box)
        layout.addWidget(box)
        return form

    def _build_general(self, layout: QVBoxLayout) -> None:
        form = self._form(layout, "General")
        theme_combo = _combo(THEMES, self.app.settings.get("theme", theme.DEFAULT_THEME))
        form.addRow("Theme:", theme_combo)
        device = _combo(DEVICES, self.app.settings.get("device", "auto"))
        cuda = device.findData("cuda")
        if not self.app._cuda_available():
            device.model().item(cuda).setEnabled(False)
            device.setItemData(cuda, "torch reports no CUDA device", Qt.ItemDataRole.ToolTipRole)
        form.addRow("Device:", device)
        self.widgets.update({"theme": theme_combo, "device": device})

    def _build_generation(self, layout: QVBoxLayout) -> None:
        schema = self.app.backend.get_config_schema()
        fields = [dataclasses.replace(f, group="Segmentation" if f.key in spec.SEGMENTATION_KEYS else "Output")
                  for f in schema if f.key in spec.PROGRAM_SCHEMA_KEYS]
        values = {f.key: self.app.settings.get(f.key, spec.SETTINGS_DEFAULTS.get(f.key, f.default)) for f in fields}
        self.generation_form = SchemaFormWidget(fields, values)
        layout.addWidget(self.generation_form)

    def _build_performance(self, layout: QVBoxLayout) -> None:
        schema = self.app.backend.get_config_schema()
        shared = [dataclasses.replace(f, group="All engines") for f in schema
                  if spec.is_program_field(f) and f.group == spec.PROGRAM_SCHEMA_GROUP
                  and f.key not in self._per_engine_keys(schema)]
        self.shared_performance_form = SchemaFormWidget(
            shared, {f.key: self.app.settings.get(f.key, spec.SETTINGS_DEFAULTS.get(f.key, f.default))
                     for f in shared})
        layout.addWidget(self.shared_performance_form)
        for engine_id in engine_registry.list_engines():
            schema = engine_registry.get_config_schema(engine_id)
            name = engine_registry.get_display_name(engine_id)
            fields = [dataclasses.replace(f, group=name) for f in per_engine_fields(schema)
                      if spec.is_program_field(f) and f.type != ConfigFieldType.TEXT]
            if not fields:
                continue
            form = SchemaFormWidget(fields, self.app.engine_settings(engine_id))
            layout.addWidget(form)
            self.engine_forms[engine_id] = form

    @staticmethod
    def _per_engine_keys(schema) -> set:
        return {f.key for f in per_engine_fields(schema)}

    def _build_project(self, layout: QVBoxLayout) -> None:
        settings = self.app.document.settings
        form = self._form(layout, "Project")
        focus = getattr(self.app, "focus", None)
        if focus is not None and focus.parent_id is not None:
            title = QLineEdit(focus.title())
            title.setToolTip("The subproject's name in its parent.")
            form.addRow("Subproject title:", title)
            self.widgets["title"] = title
        gap = _spin(0.0, 10.0, 0.05, float(settings.get("gap_s", DEFAULT_GAP_S)))
        para = _spin(0.0, 10.0, 0.05, float(settings.get("paragraph_gap_s", DEFAULT_PARAGRAPH_GAP_S)))
        gap.setToolTip("Silence between clips placed one after another.")
        para.setToolTip("Silence between clips across a blank line.")
        form.addRow("Gap (s):", gap)
        form.addRow("Paragraph gap (s):", para)
        chapter = _spin(0.0, 10.0, 0.05, self._setting_s("chapter_gap_s", float(para.value())))
        chapter.setToolTip("Silence before a subproject, such as the next chapter. Follows the paragraph gap "
                           "until you set it.")
        heading_gap = _spin(0.0, 10.0, 0.05, self._setting_s("heading_gap_after_s", DEFAULT_HEADING_GAP_S))
        heading_gap.setToolTip("Silence after the heading: the first clip, when it is one short line that "
                               "doesn't end like a sentence.")
        heading_speed = _spin(0.25, 2.0, 0.05, self._setting_s("heading_speed", 1.0))
        heading_speed.setSuffix("x")
        heading_speed.setToolTip("Reads the heading at this multiple of its speed. 1.00 leaves it as is. "
                                 "Changing it makes the heading stale.")
        form.addRow("Chapter gap (s):", chapter)
        form.addRow("Gap after a heading (s):", heading_gap)
        form.addRow("Heading speed:", heading_speed)
        jitter_range = gap_jitter_range(self.app.document)
        jitter = QCheckBox("Vary the gap when the speaker changes")
        jitter.setChecked(jitter_range is not None)
        jitter.setToolTip("Each speaker change draws its gap between these two values, the same draw every "
                          "time, so the pacing sounds less mechanical. Same-speaker and paragraph gaps "
                          "stay as set.")
        jitter_min = _spin(0.0, 10.0, 0.05, jitter_range[0] if jitter_range else 0.2)
        jitter_max = _spin(0.0, 10.0, 0.05, jitter_range[1] if jitter_range else 0.8)
        for spin in (jitter_min, jitter_max):
            spin.setEnabled(jitter_range is not None)
            jitter.toggled.connect(spin.setEnabled)
        form.addRow("", jitter)
        form.addRow("Between (s):", _row(jitter_min, QLabel("and"), jitter_max))

        crossfade = QCheckBox("Auto-crossfade overlapping clips")
        crossfade.setChecked(bool(settings.get("auto_crossfade", False)))
        form.addRow("", crossfade)
        ripple = QCheckBox("Ripple on regenerate")
        ripple.setChecked(bool(settings.get("ripple", True)))
        ripple.setToolTip("When a regenerated clip changes length, move the clips placed after it by the "
                          "difference. A clip locked in time stays put.")
        form.addRow("", ripple)
        align = QCheckBox("Align locked clips to their first word")
        align.setChecked(align_onset_enabled(self.app.document))
        align.setToolTip("Start each clip locked in time a little early, by the silence before its first "
                         "word, so the word lands on the clip's time. Skipped for a clip with trim on. "
                         "On by default when the project has a locked clip.")
        form.addRow("", align)
        self.widgets.update({"gap_s": gap, "paragraph_gap_s": para, "chapter_gap_s": chapter,
                             "heading_gap_after_s": heading_gap, "heading_speed": heading_speed,
                             "gap_jitter": jitter, "gap_jitter_min": jitter_min, "gap_jitter_max": jitter_max,
                             "auto_crossfade": crossfade, "ripple": ripple, "align_onset": align})

    def _setting_s(self, key: str, default: float) -> float:
        try:
            return float(self.app.document.settings.get(key, default))
        except (TypeError, ValueError):
            return default

    def _build_timeline(self, layout: QVBoxLayout) -> None:
        settings = self.app.document.settings
        form = self._form(layout, "Tracks")
        track_layout = self.app.document.track_layout()
        layout_combo = _combo((("character", "One per character"), ("unified", "Unified")), track_layout["mode"])
        layout_combo.setToolTip("Unified puts every clip on a few lanes and moves to the next lane "
                                "whenever the speaker changes.")
        lanes = QSpinBox()
        lanes.setRange(1, 16)
        lanes.setValue(int(track_layout.get("lanes", 3)))
        lanes.setSuffix(" lanes")
        lanes.setEnabled(track_layout["mode"] == "unified")
        layout_combo.currentIndexChanged.connect(lambda _i: lanes.setEnabled(layout_combo.currentData() == "unified"))
        form.addRow("Track layout:", _row(layout_combo, lanes, stretch_first=True))
        try:
            duck_db = float(settings.get("duck_db", DEFAULT_DUCK_DB))
        except (TypeError, ValueError):
            duck_db = DEFAULT_DUCK_DB
        duck = _spin(-40.0, 0.0, 1.0, max(-40.0, min(0.0, duck_db)), decimals=1)
        duck.setSuffix(" dB")
        duck.setToolTip("How far a track with D (duck) on goes down while other clips play.")
        form.addRow("Ducking:", duck)

        form = self._form(layout, "Timecode")
        tc = timecode_settings(settings)
        enabled = QCheckBox("Show timecode")
        enabled.setChecked(bool(tc["enabled"]))
        drop = QCheckBox("Drop-frame")
        drop.setChecked(bool(tc["drop_frame"]))
        drop.setToolTip("29.97 and 59.94 only.")
        fps = QComboBox()
        for rate in FRAME_RATES:
            fps.addItem(f"{rate:g} fps", rate)
        index = fps.findData(float(tc["frame_rate"]))
        fps.setCurrentIndex(index if index >= 0 else FRAME_RATES.index(25.0))
        start = QLineEdit(str(tc["start"]))
        start.setToolTip("Timecode of the project's first frame, HH:MM:SS:FF.")
        form.addRow("", _row(enabled, drop))
        form.addRow("Frame rate:", fps)
        form.addRow("Start:", start)
        self.widgets.update({"track_layout": layout_combo, "track_lanes": lanes, "duck_db": duck,
                             "tc_enabled": enabled, "tc_drop": drop, "tc_fps": fps, "tc_start": start})

    def _build_source_track(self, layout: QVBoxLayout) -> None:
        form = self._form(layout, "Source track")
        track = source_track_settings(self.app.document.settings)
        label = QLabel(self._source_track_name(track))
        label.setToolTip("The original dialogue. The transport's Original and Both play it under each "
                         "clip that has a reference range.")
        remove = QPushButton("Remove")
        remove.setEnabled(track is not None)
        remove.clicked.connect(self._remove_source_track)
        offset = _spin(-3600.0, 3600.0, 0.1, track["offset_s"] if track else 0.0, decimals=3)
        offset.setEnabled(track is not None)
        offset.setToolTip("Where reference time zero is in the source track, in seconds.")
        form.addRow("Track:", _row(label, remove, stretch_first=True))
        form.addRow("Offset (s):", offset)
        self.widgets.update({"source_track": label, "source_remove": remove, "source_offset": offset})

    def _source_track_name(self, track) -> str:
        if track is None:
            return "None (File > Import Source Track)"
        name = os.path.basename(track["path"])
        if project_io.source_track_path(self.app.document, self.app.project_dir) is None:
            name += " (missing)"
        return name

    def _remove_source_track(self) -> None:
        """Staged like everything else: the setting clears on Apply. The
        copy stays in the project dir, so an undo brings the track back."""
        self._remove_source = True
        self.widgets["source_track"].setText("None (removed on Apply)")
        self.widgets["source_remove"].setEnabled(False)
        self.widgets["source_offset"].setEnabled(False)
        self._sync_apply()

    # -- reading the widgets ----------------------------------------------------

    def _values(self) -> dict:
        """Every value the window shows, keyed `(where, key)`: "app" an app
        setting, ("engine", id) a per-engine one, "doc" a `Document.settings`
        entry, "title" the subproject's name."""
        w = self.widgets
        values = {
            ("app", "theme"): w["theme"].currentData(),
            ("app", "device"): w["device"].currentData(),
        }
        for key, value in {**self.generation_form.values(), **self.shared_performance_form.values()}.items():
            values[("app", key)] = value
        for engine_id, form in self.engine_forms.items():
            for key, value in form.values().items():
                values[(("engine", engine_id), key)] = value
        if "title" in w:
            values[("title", None)] = w["title"].text().strip()
        for key in ("gap_s", "paragraph_gap_s", "chapter_gap_s", "heading_gap_after_s", "heading_speed"):
            values[("doc", key)] = round(w[key].value(), 2)
        values[("doc", "gap_jitter_s")] = ([round(w["gap_jitter_min"].value(), 2), round(w["gap_jitter_max"].value(), 2)]
                                           if w["gap_jitter"].isChecked() else None)
        for key in ("auto_crossfade", "ripple", "align_onset"):
            values[("doc", key)] = w[key].isChecked()
        mode = w["track_layout"].currentData()
        values[("doc", "track_layout")] = ({"mode": "unified", "lanes": w["track_lanes"].value()}
                                           if mode == "unified" else {"mode": "character"})
        values[("doc", "duck_db")] = round(w["duck_db"].value(), 1)
        values[("doc", "timecode")] = self._timecode_value()
        track = source_track_settings(self.app.document.settings)
        if self._remove_source:
            values[("doc", SOURCE_TRACK_KEY)] = None
        elif track is not None:
            values[("doc", SOURCE_TRACK_KEY)] = {**track, "offset_s": round(w["source_offset"].value(), 3)}
        return values

    def _timecode_value(self) -> dict:
        w = self.widgets
        start = w["tc_start"].text().strip() or "00:00:00:00"
        rate = float(w["tc_fps"].currentData())
        drop = w["tc_drop"].isChecked() and rate in (29.97, 59.94)
        try:
            tc_to_frames(start, rate, drop)
        except ValueError:
            start = timecode_settings(self.app.document.settings)["start"]
        return {"enabled": w["tc_enabled"].isChecked(), "frame_rate": rate, "start": start, "drop_frame": drop}

    def pending(self) -> dict:
        """The values that differ from what the window opened with."""
        current = self._values()
        return {k: v for k, v in current.items() if self._initial.get(k, object()) != v}

    def _watch_all(self) -> None:
        for widget in self.pages.findChildren(QWidget):
            for name in ("currentIndexChanged", "toggled", "valueChanged", "textChanged"):
                signal = getattr(widget, name, None)
                if signal is not None and hasattr(signal, "connect"):
                    try:
                        signal.connect(self._sync_apply)
                    except (RuntimeError, TypeError):
                        pass
                    break

    def _sync_apply(self, *_args) -> None:
        self.apply_button.setEnabled(bool(self.pending()))

    # -- writing ----------------------------------------------------------------

    def _ok(self) -> None:
        self.apply()
        self.accept()

    def apply(self) -> None:
        changes = self.pending()
        if not changes:
            return
        app = self.app
        rehighlight = False
        for (where, key), value in changes.items():
            if where == "app":
                if key == "theme":
                    app.set_theme(value)
                elif key == "device":
                    app.set_device(value)
                else:
                    app.settings[key] = value
                    rehighlight = True
            elif isinstance(where, tuple) and where[0] == "engine":
                app.set_engine_setting(where[1], key, value)
                rehighlight = True
        title = changes.get(("title", None))
        if title is not None and title:
            app.rename_subproject(app.focus, title)
        stack = app.document.undo_stack
        doc_changed = False
        for (where, key), value in changes.items():
            if where != "doc" or app.document.settings.get(key) == value:
                continue
            stack.push(SetFieldCommand("document", None, "settings", value, key=key))
            doc_changed = True
        if ("doc", "timecode") in changes and getattr(app, "transport_dock", None) is not None:
            app.transport_dock.set_position(app.transport.position(), app.transport.duration())
        app.schedule_save()
        if rehighlight and app.editor is not None:
            # Segmentation and engine settings are in the segment key:
            # stale clips show now, not on the next edit.
            app.editor.rehighlight()
        if doc_changed or rehighlight:
            app.refresh_timeline()
        if app.settings_dock is not None:
            app.settings_dock.refresh_scope_fields()
        self._build()
