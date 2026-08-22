"""Audio FX dock: builds the FX controls from `kokoro_gui.qt.spec.FX_FIELD_SPECS`
and loads/saves FX presets under `presets/fx/`.

Seven FX_PRESET_KEYS fields have no widget here (see spec.py's docstring) -
their values are tracked in `self._hidden_values` and only ever change via
preset load.
"""
from __future__ import annotations

import json
import os
import re

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QInputDialog, QLabel, QMessageBox, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

import kokoro_gui.qt.app as qt_app_module
from kokoro_gui.qt import spec


class FXDock(QDockWidget):
    def __init__(self, app, parent=None):
        super().__init__("Audio FX", parent)
        self.setObjectName("dock_fx")
        self.app = app

        self._value_widgets: dict[str, QDoubleSpinBox] = {}
        self._enabled_checks: dict[str, QCheckBox] = {}
        self._hidden_values: dict[str, float] = {
            k: spec.SETTINGS_DEFAULTS[k] for k in spec.FX_KEYS_WITHOUT_WIDGET
        }

        content = QWidget()
        outer = QVBoxLayout(content)

        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.currentTextChanged.connect(self._on_preset_selected)
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
            box = QGroupBox(group)
            box_layout = QVBoxLayout(box)
            self._build_group(box_layout, specs_by_group[group])
            for key, label, toggle_group in spec.FX_STANDALONE_TOGGLES:
                if toggle_group == group:
                    check = QCheckBox(label)
                    check.setChecked(self.app.settings.get(key, False))
                    check.toggled.connect(lambda _v: self.app.schedule_save())
                    self._enabled_checks[key] = check
                    box_layout.addWidget(check)
            inner_layout.addWidget(box)

        inner_layout.addStretch(1)
        self.setWidget(content)

        self.set_values(self.app.settings)
        self.refresh_presets()

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
                header.toggled.connect(lambda _v: self.app.schedule_save())
                self._enabled_checks[enabled_key] = header
                box_layout.addWidget(header)
            else:
                box_layout.addWidget(QLabel(f"<b>{section}</b>"))

            form = QFormLayout()
            for s in section_specs:
                spin = QDoubleSpinBox()
                spin.setRange(s.minimum, s.maximum)
                span = s.maximum - s.minimum
                spin.setSingleStep(span / s.steps if s.steps else 0.1)
                spin.setDecimals(s.decimals)
                spin.setSuffix(f" {s.unit}" if s.unit else "")
                spin.setValue(self.app.settings.get(s.key, s.minimum))
                spin.valueChanged.connect(lambda _v: self.app.schedule_save())
                form.addRow(s.label + ":", spin)
                self._value_widgets[s.key] = spin
            box_layout.addLayout(form)

    # --- state -----------------------------------------------------------

    def get_state(self) -> dict:
        state = dict(self._hidden_values)
        for key, spin in self._value_widgets.items():
            state[key] = spin.value()
        for key, check in self._enabled_checks.items():
            state[key] = check.isChecked()
        return state

    def set_values(self, data: dict) -> None:
        for key, spin in self._value_widgets.items():
            if key in data:
                spin.setValue(data[key])
        for key, check in self._enabled_checks.items():
            if key in data:
                check.setChecked(bool(data[key]))
        for key in self._hidden_values:
            if key in data:
                self._hidden_values[key] = data[key]

    # --- presets (presets/fx/*.json, shared with Tk) ----------------------

    def refresh_presets(self) -> None:
        presets = ["Select FX Preset..."]
        if os.path.exists(qt_app_module.FX_PRESETS_DIR):
            files = [f for f in os.listdir(qt_app_module.FX_PRESETS_DIR) if f.endswith(".json")]
            presets.extend(f[:-5] for f in files)
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItems(presets)
        self.preset_combo.setCurrentText("Select FX Preset...")
        self.preset_combo.blockSignals(False)
        if hasattr(self.app, "generation_dock") and self.app.generation_dock is not None:
            self.app.generation_dock.refresh_fx_presets()

    def _save_preset_dialog(self) -> None:
        name, ok = QInputDialog.getText(self, "Save FX Preset", "Enter FX preset name:")
        if not ok or not name:
            return
        name = re.sub(r'[<>:"/\\|?*]', "", name).strip()
        if not name:
            return
        data = {k: self.get_state()[k] for k in spec.FX_PRESET_KEYS}
        fpath = os.path.join(qt_app_module.FX_PRESETS_DIR, f"{name}.json")
        try:
            os.makedirs(qt_app_module.FX_PRESETS_DIR, exist_ok=True)
            with open(fpath, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=4)
            QMessageBox.information(self, "Saved", f"FX Preset '{name}' saved.")
            self.refresh_presets()
            self.preset_combo.setCurrentText(name)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save FX preset: {e}")

    def load_preset(self, name: str) -> None:
        if not name or name == "Select FX Preset...":
            return
        safe_name = os.path.basename(name)
        if not safe_name:
            return
        fpath = os.path.join(qt_app_module.FX_PRESETS_DIR, f"{safe_name}.json")
        if not os.path.exists(fpath):
            return
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.set_values(data)
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText(safe_name)
            self.preset_combo.blockSignals(False)
            if hasattr(self.app, "generation_dock") and self.app.generation_dock is not None:
                self.app.generation_dock.fx_preset_combo.blockSignals(True)
                self.app.generation_dock.fx_preset_combo.setCurrentText(safe_name)
                self.app.generation_dock.fx_preset_combo.blockSignals(False)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load FX preset: {e}")

    def _on_preset_selected(self, name: str) -> None:
        self.load_preset(name)
