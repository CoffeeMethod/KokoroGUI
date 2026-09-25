"""Audio FX dock: builds the FX controls from `kokoro_gui.qt.spec.FX_FIELD_SPECS`
and loads/saves FX presets under `presets/fx/`.

UI6 (Claude/PLAN_ui_shell_redesign.md section 3): the tab follows the
selection the way the Settings tab does, with the same three modes:

- "none": the project-wide FX state, persisted in `config_qt.json`, fed
  into whole-document generation and the bottom layer of every clip's
  resolved stack. Edits schedule an autosave and a debounced timeline
  re-render.
- "clip": the selected clip's resolved FX (project state, then the
  character's preset, then `clip.fx_override` on top). Edits are collected
  and pushed as one `SetClipFxCommand` per 300ms of quiet, so a slider drag
  is one undo step and the override is a full resolved-values dict.
- "character": the character's attached FX preset file. The first edit per
  session asks "This changes the preset for every clip using X. Continue?";
  a character with no preset yet gets one named after it.

`project_fx_state()` always returns the "none" values regardless of what's
rendered (what `_assemble_config`/`preview_conversion` need); `get_state()`
is whatever the widgets currently show. The preset combo names the preset
the current scope resolves to. Scope resolution is
`kokoro_gui.qt.fx_resolve.resolve_fx`, shared with `_assemble_clip_config`.

FX are read-time post-processing (kokoro_gui/audio/post.py): every edit here
is audible on the next transport rebuild and never dirties a clip.

Seven FX_PRESET_KEYS fields have no widget here (see spec.py's docstring);
their values live in `self._hidden_values` and only change via preset load.

A `spec.FXFileSpec` field (the convolution reverb's impulse response) is a
combo of names from its store, project-local first (`fx_presets.list_ir_names`),
with a "None" entry that stores "", and an "Add..." button that copies a wav
into the global store (`fx_presets.import_ir_file`).
"""
from __future__ import annotations

import json
import os
import re

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QInputDialog, QLabel, QMessageBox, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

import kokoro_gui.qt.app as qt_app_module
from kokoro_gui.daw.undo import SetClipFxCommand
from kokoro_gui.engine.presets import filter_fx_preset_values
from kokoro_gui.qt import fx_resolve, spec
from kokoro_gui.qt.fx_presets import import_ir_file, list_fx_preset_names, list_ir_names
from kokoro_gui.qt.fx_resolve import PLACEHOLDER as _PLACEHOLDER

CLIP_EDIT_DEBOUNCE_MS = 300
PROJECT_EDIT_DEBOUNCE_MS = 300
FILE_FIELD_NONE = "None"


class FXDock(QDockWidget):
    def __init__(self, app, parent=None):
        super().__init__("Audio FX", parent)
        self.setObjectName("dock_fx")
        self.app = app

        self._value_widgets: dict[str, QDoubleSpinBox] = {}
        self._enabled_checks: dict[str, QCheckBox] = {}
        self._file_combos: dict[str, QComboBox] = {}
        self._hidden_values: dict[str, float] = {
            k: spec.SETTINGS_DEFAULTS[k] for k in spec.FX_KEYS_WITHOUT_WIDGET
        }
        self._mode = "none"
        self._target = None
        self._none_values: dict = {k: self.app.settings.get(k, spec.SETTINGS_DEFAULTS[k]) for k in spec.FX_PRESET_KEYS}
        self._loading = False
        self._character_confirmed: set = set()

        self._clip_timer = QTimer(self)
        self._clip_timer.setSingleShot(True)
        self._clip_timer.setInterval(CLIP_EDIT_DEBOUNCE_MS)
        self._clip_timer.timeout.connect(self._flush_clip_edit)

        # Project-scope edits re-render the timeline/transport (read-time
        # FX); debounced so a run of spinbox steps is one re-render.
        self._project_timer = QTimer(self)
        self._project_timer.setSingleShot(True)
        self._project_timer.setInterval(PROJECT_EDIT_DEBOUNCE_MS)
        self._project_timer.timeout.connect(self.app.refresh_timeline)

        content = QWidget()
        outer = QVBoxLayout(content)

        self.scope_label = QLabel("Project FX")
        outer.addWidget(self.scope_label)

        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.activated.connect(self._on_preset_activated)
        save_btn = QPushButton("Save FX Preset...")
        save_btn.clicked.connect(self._save_preset_dialog)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_presets)
        preset_row.addWidget(QLabel("FX Preset:"))
        preset_row.addWidget(self.preset_combo, 1)
        preset_row.addWidget(save_btn)
        preset_row.addWidget(refresh_btn)
        outer.addLayout(preset_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        specs_by_group: dict[str, list[spec.FXSliderSpec]] = {g: [] for g in spec.FX_GROUP_ORDER}
        for s in spec.FX_FIELD_SPECS:
            specs_by_group[s.group].append(s)

        for group in spec.FX_GROUP_ORDER:
            box = QGroupBox(group.replace("&", "&&"))
            box_layout = QVBoxLayout(box)
            self._build_group(box_layout, specs_by_group[group])
            for key, label, toggle_group in spec.FX_STANDALONE_TOGGLES:
                if toggle_group == group:
                    check = QCheckBox(label)
                    check.setChecked(self.app.settings.get(key, False))
                    check.toggled.connect(self._on_widget_changed)
                    self._enabled_checks[key] = check
                    box_layout.addWidget(check)
            inner_layout.addWidget(box)

        inner_layout.addStretch(1)
        self.setWidget(content)

        self._loading = True
        self.set_values(self.app.settings)
        self._loading = False
        self.refresh_presets()
        self.app.selection.changed.connect(self.refresh_for_selection)
        self.refresh_for_selection()

    def _build_group(self, box_layout: QVBoxLayout, specs: list) -> None:
        sections: dict[str, list] = {}
        section_order: list[str] = []
        for s in specs:
            if s.section not in sections:
                sections[s.section] = []
                section_order.append(s.section)
            sections[s.section].append(s)

        for section in section_order:
            section_specs = sections[section]
            enabled_key = section_specs[0].enabled_key
            if enabled_key:
                header = QCheckBox(section)
                header.setChecked(self.app.settings.get(enabled_key, False))
                header.toggled.connect(self._on_widget_changed)
                self._enabled_checks[enabled_key] = header
                box_layout.addWidget(header)
            else:
                box_layout.addWidget(QLabel(f"<b>{section}</b>"))

            form = QFormLayout()
            for s in section_specs:
                if isinstance(s, spec.FXFileSpec):
                    form.addRow(s.label + ":", self._build_file_field(s))
                    continue
                spin = QDoubleSpinBox()
                spin.setRange(s.minimum, s.maximum)
                span = s.maximum - s.minimum
                spin.setSingleStep(span / s.steps if s.steps else 0.1)
                spin.setDecimals(s.decimals)
                spin.setSuffix(f" {s.unit}" if s.unit else "")
                spin.setValue(self.app.settings.get(s.key, s.minimum))
                spin.valueChanged.connect(self._on_widget_changed)
                form.addRow(s.label + ":", spin)
                self._value_widgets[s.key] = spin
            box_layout.addLayout(form)

    def _build_file_field(self, s) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        combo = QComboBox()
        combo.addItem(FILE_FIELD_NONE, "")
        combo.currentIndexChanged.connect(self._on_widget_changed)
        add_btn = QPushButton("Add...")
        add_btn.setToolTip("Copy a wav file into the global impulse response store (presets/fx/ir)")
        add_btn.clicked.connect(lambda _checked=False, key=s.key: self._add_ir_file(key))
        layout.addWidget(combo, 1)
        layout.addWidget(add_btn)
        self._file_combos[s.key] = combo
        self._fill_file_combo(combo, self.app.settings.get(s.key, ""))
        return row

    def _fill_file_combo(self, combo: QComboBox, current) -> None:
        """Lists the store's names under "None" and selects `current`. A name
        that resolves nowhere is still listed (marked missing) so showing a
        preset or override that names it doesn't drop the name."""
        current = current if isinstance(current, str) else ""
        was_loading = self._loading
        self._loading = True
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem(FILE_FIELD_NONE, "")
            names = list_ir_names(getattr(self.app, "project_dir", None))
            for name in names:
                combo.addItem(name, name)
            if current and current not in names:
                combo.addItem(f"{current} (missing)", current)
            index = combo.findData(current)
            combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            combo.blockSignals(False)
            self._loading = was_loading

    def refresh_ir_choices(self) -> None:
        """Re-lists every file field's names (the project dir changed, or a
        file was added), keeping each selection."""
        for combo in self._file_combos.values():
            self._fill_file_combo(combo, combo.currentData())

    def _add_ir_file(self, key: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Add impulse response", "", "WAV files (*.wav)")
        if not path:
            return
        try:
            name = import_ir_file(path)
        except (OSError, ValueError) as e:
            QMessageBox.critical(self, "Error", f"Couldn't add the impulse response: {e}")
            return
        self.refresh_ir_choices()
        combo = self._file_combos[key]
        combo.setCurrentIndex(max(0, combo.findData(name)))

    # --- scope ---------------------------------------------------------------

    def _resolve_mode(self):
        kind = self.app.selection.kind
        if kind == "clip":
            clip = self.app.document.get_clip(self.app.selection.selected_clip_id)
            if clip is not None:
                return "clip", clip
        elif kind == "character":
            character = self.app.document.get_character(self.app.selection.selected_character_id)
            if character is not None:
                return "character", character
        return "none", None

    @property
    def mode(self) -> str:
        return self._mode

    def refresh_for_selection(self) -> None:
        if self._mode == "none" and not self._loading:
            self._none_values = self.get_state()
        if self._clip_timer.isActive():
            self._flush_clip_edit()
        self._mode, self._target = self._resolve_mode()
        self._loading = True
        try:
            values, preset_name = self._resolved_values()
            self.set_values(values)
            self._set_combo_text(preset_name)
        finally:
            self._loading = False
        if self._mode == "clip":
            text = "Clip FX (override for the selected clip)"
        elif self._mode == "character":
            text = f"Character FX: {self._target.name}"
        else:
            text = "Project FX"
        scope = self.app.scope_text() if hasattr(self.app, "scope_text") else None
        self.scope_label.setText(f"{scope}: {text}" if scope else text)

    def _character_preset_name(self, character):
        return fx_resolve.real_preset_name(character.preset_data.get("fx_preset")) if character is not None else None

    def _resolved_values(self):
        """`(values, preset_name)` for the current scope - the same
        resolution `_assemble_clip_config` generates and plays with."""
        if self._mode == "none":
            return dict(self._none_values), self.app.settings.get("fx_preset") or None
        if self._mode == "character":
            resolution = fx_resolve.resolve_fx(self.app, character=self._target)
        else:
            resolution = fx_resolve.resolve_fx(self.app, clip=self._target)
        return resolution.values, resolution.preset_name

    def _set_combo_text(self, name) -> None:
        self.preset_combo.blockSignals(True)
        try:
            if name and self.preset_combo.findText(name) < 0 and name != "custom":
                self.preset_combo.addItem(name)
            if name == "custom":
                if self.preset_combo.findText("(custom)") < 0:
                    self.preset_combo.addItem("(custom)")
                self.preset_combo.setCurrentText("(custom)")
            else:
                self.preset_combo.setCurrentText(name or _PLACEHOLDER)
        finally:
            self.preset_combo.blockSignals(False)

    # --- edits -------------------------------------------------------------------

    def _on_widget_changed(self, *_args) -> None:
        if self._loading:
            return
        if self._mode == "none":
            self.app.schedule_save()
            self._project_timer.start()
            return
        if self._mode == "clip":
            self._clip_timer.start()
            return
        self._apply_character_edit()

    def _flush_clip_edit(self) -> None:
        self._clip_timer.stop()
        if self._mode != "clip" or self._target is None:
            return
        values = {k: self.get_state()[k] for k in spec.FX_PRESET_KEYS}
        self.app.document.undo_stack.push(SetClipFxCommand(self._target.id, values, preset_name=None))
        self._set_combo_text("custom")
        self.app.editor.rehighlight()
        self.app.schedule_save()
        self.app.refresh_timeline()

    def _confirm_character_edit(self, character) -> bool:
        if character.id in self._character_confirmed:
            return True
        answer = QMessageBox.question(
            self, "Edit character FX",
            f"This changes the preset for every clip using {character.name}. Continue?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False
        self._character_confirmed.add(character.id)
        return True

    def _apply_character_edit(self) -> None:
        character = self._target
        if character is None:
            return
        if not self._confirm_character_edit(character):
            self.refresh_for_selection()
            return
        name = self._character_preset_name(character)
        if not name:
            name = re.sub(r'[<>:"/\\|?*]', "", character.name).strip() or "character"
            character.preset_data["fx_preset"] = name
        data = {k: self.get_state()[k] for k in spec.FX_PRESET_KEYS}
        if self._write_preset_file(name, data):
            self._set_combo_text(name)
            self.app.editor.rehighlight()
            self.app.schedule_save()
            self.app.refresh_timeline()

    # --- state ---------------------------------------------------------------

    def get_state(self) -> dict:
        state = dict(self._hidden_values)
        for key, spin in self._value_widgets.items():
            state[key] = spin.value()
        for key, check in self._enabled_checks.items():
            state[key] = check.isChecked()
        for key, combo in self._file_combos.items():
            state[key] = combo.currentData() or ""
        return state

    def project_fx_state(self) -> dict:
        """The project-wide ("none") FX values regardless of what's rendered."""
        if self._mode == "none":
            return self.get_state()
        return dict(self._none_values)

    def set_values(self, data: dict) -> None:
        was_loading = self._loading
        self._loading = True
        try:
            for key, spin in self._value_widgets.items():
                if key in data:
                    spin.setValue(data[key])
            for key, check in self._enabled_checks.items():
                if key in data:
                    check.setChecked(bool(data[key]))
            for key, combo in self._file_combos.items():
                if key in data:
                    self._fill_file_combo(combo, data[key])
            for key in self._hidden_values:
                if key in data:
                    self._hidden_values[key] = data[key]
        finally:
            self._loading = was_loading

    # --- presets (presets/fx/*.json) --------------------------------------

    def refresh_presets(self) -> None:
        self.refresh_ir_choices()
        current = self.preset_combo.currentText()
        presets = [_PLACEHOLDER] + list_fx_preset_names(self.app.project_dir)
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItems(presets)
        self.preset_combo.setCurrentText(current if current in presets else _PLACEHOLDER)
        self.preset_combo.blockSignals(False)
        if getattr(self.app, "settings_dock", None) is not None:
            self.app.settings_dock.refresh_fx_presets()
        if getattr(self.app, "transcript_dock", None) is not None:
            self.app.transcript_dock.refresh_fx_choices()

    def _write_preset_file(self, name: str, data: dict) -> bool:
        fpath = os.path.join(qt_app_module.FX_PRESETS_DIR, f"{name}.json")
        try:
            os.makedirs(qt_app_module.FX_PRESETS_DIR, exist_ok=True)
            with open(fpath, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=4)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save FX preset: {e}")
            return False
        return True

    def _save_preset_dialog(self) -> None:
        name, ok = QInputDialog.getText(self, "Save FX Preset", "Enter FX preset name:")
        if not ok or not name:
            return
        name = re.sub(r'[<>:"/\\|?*]', "", name).strip()
        if not name:
            return
        data = {k: self.get_state()[k] for k in spec.FX_PRESET_KEYS}
        if self._write_preset_file(name, data):
            QMessageBox.information(self, "Saved", f"FX Preset '{name}' saved.")
            self.refresh_presets()
            self._set_combo_text(name)

    def load_preset(self, name: str) -> None:
        """Project-scope load: applies the preset's values to the "none"
        state (and the widgets, when that's what's rendered)."""
        if not name or name == _PLACEHOLDER:
            return
        safe_name = os.path.basename(name)
        if not safe_name:
            return
        fpath = os.path.join(qt_app_module.FX_PRESETS_DIR, f"{safe_name}.json")
        if not os.path.exists(fpath):
            return
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                data = filter_fx_preset_values(json.load(fh))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load FX preset: {e}")
            return
        self.app.settings["fx_preset"] = safe_name
        if self._mode == "none":
            self.set_values(data)
            self._set_combo_text(safe_name)
        else:
            self._none_values.update(data)
        if getattr(self.app, "settings_dock", None) is not None:
            self.app.settings_dock.set_fx_preset_display(safe_name)
        self.app.schedule_save()
        self.app.refresh_timeline()

    def _on_preset_activated(self, index: int) -> None:
        name = self.preset_combo.itemText(index)
        if not name or name == _PLACEHOLDER or name == "(custom)":
            return
        if self._mode == "clip" and self._target is not None:
            self.app.transcript_dock.apply_fx_preset_to_clip(self._target.id, name)
            self.refresh_for_selection()
            return
        if self._mode == "character" and self._target is not None:
            self._target.preset_data["fx_preset"] = name
            self.app.editor.rehighlight()
            self.app.schedule_save()
            self.app.refresh_timeline()
            self.refresh_for_selection()
            return
        self.load_preset(name)

    def _on_preset_selected(self, name: str) -> None:
        """Kept for callers that used the old currentTextChanged slot."""
        self.load_preset(name)
