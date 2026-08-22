"""Generation dock: input source, voice/speed/output config, and the speaker
presets (`presets/*.json`, shared with the Tk frontend) that snapshot that
config. Mirrors kokoro_gui/ui/generation_tab.py + the relevant slice of
gui.py's `start_conversion`/`preview_conversion` config assembly.

Reads `kokoro_gui.qt.app.PRESETS_DIR` qualified at call time (not imported by
name) so tests can monkeypatch it into a tmp_path, same convention
generation_tab.py already uses for `gui.PRESETS_DIR`.
"""
from __future__ import annotations

import json
import os
import re

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox,
    QTabWidget, QVBoxLayout, QWidget,
)

import kokoro_gui.qt.app as qt_app_module
from kokoro_gui.qt import spec
from kokoro_gui.qt.schema_form import SchemaFormWidget


class GenerationDock(QDockWidget):
    def __init__(self, app, parent=None):
        super().__init__("Generate Audio", parent)
        self.setObjectName("dock_generation")
        self.app = app

        content = QWidget()
        outer = QVBoxLayout(content)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        # --- Input source ---
        input_group = QGroupBox("Input Source")
        input_layout = QVBoxLayout(input_group)
        self.tabs = QTabWidget()
        input_layout.addWidget(self.tabs)

        self.text_entry = QPlainTextEdit()
        self.tabs.addTab(self.text_entry, "Direct Text")

        file_tab = QWidget()
        file_layout = QHBoxLayout(file_tab)
        self.file_path_edit = QLineEdit()
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_file)
        file_layout.addWidget(QLabel("File Path:"))
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(browse_btn)
        self.tabs.addTab(file_tab, "Load File")
        layout.addWidget(input_group)

        # --- Presets row ---
        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.currentTextChanged.connect(self._on_preset_selected)
        save_btn = QPushButton("Save Preset...")
        save_btn.clicked.connect(self._save_preset_dialog)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_presets)
        preset_row.addWidget(QLabel("Preset:"))
        preset_row.addWidget(self.preset_combo, 1)
        preset_row.addWidget(save_btn)
        preset_row.addWidget(refresh_btn)
        layout.addLayout(preset_row)

        # --- Schema-driven config (workstream 1's payoff) ---
        self.schema_group = QGroupBox("Configuration")
        self.schema_layout = QVBoxLayout(self.schema_group)
        self.schema_form: SchemaFormWidget | None = None
        layout.addWidget(self.schema_group)
        self._build_schema_form()

        # --- Output (not schema-covered, hand-built same as gui.py/generation_tab.py) ---
        out_group = QGroupBox("Output")
        out_form = QFormLayout(out_group)
        dir_row = QWidget()
        dir_layout = QHBoxLayout(dir_row)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        self.out_dir_edit = QLineEdit(self.app.settings.get("out_dir", "audio_output"))
        dir_browse = QPushButton("...")
        dir_browse.clicked.connect(self._browse_dir)
        dir_layout.addWidget(self.out_dir_edit)
        dir_layout.addWidget(dir_browse)
        out_form.addRow("Output Folder:", dir_row)

        self.filename_edit = QLineEdit(self.app.settings.get("filename", "output"))
        out_form.addRow("Base Filename:", self.filename_edit)
        layout.addWidget(out_group)

        # --- Audio control (volume/pitch - not schema-covered, matches gui.py) ---
        audio_group = QGroupBox("Audio Control")
        audio_form = QFormLayout(audio_group)
        self.volume_spin = QDoubleSpinBox()
        self.volume_spin.setRange(0.1, 2.0)
        self.volume_spin.setSingleStep(0.1)
        self.volume_spin.setValue(self.app.settings.get("volume", 1.0))
        audio_form.addRow("Volume:", self.volume_spin)

        self.pitch_spin = QDoubleSpinBox()
        self.pitch_spin.setRange(-12, 12)
        self.pitch_spin.setSingleStep(1)
        self.pitch_spin.setValue(self.app.settings.get("pitch", 0.0))
        audio_form.addRow("Pitch (st):", self.pitch_spin)

        fx_row = QWidget()
        fx_row_layout = QHBoxLayout(fx_row)
        fx_row_layout.setContentsMargins(0, 0, 0, 0)
        self.fx_preset_combo = QComboBox()
        self.fx_preset_combo.currentTextChanged.connect(self._on_fx_preset_selected)
        self.apply_fx_check = QCheckBox("Apply")
        self.apply_fx_check.setChecked(self.app.settings.get("apply_fx", True))
        fx_row_layout.addWidget(self.fx_preset_combo, 1)
        fx_row_layout.addWidget(self.apply_fx_check)
        audio_form.addRow("FX Preset:", fx_row)

        self.normalize_check = QCheckBox("Normalize")
        self.normalize_check.setChecked(self.app.settings.get("normalize", False))
        self.trim_check = QCheckBox("Trim Silence")
        self.trim_check.setChecked(self.app.settings.get("trim", False))
        toggles_row = QWidget()
        toggles_layout = QHBoxLayout(toggles_row)
        toggles_layout.setContentsMargins(0, 0, 0, 0)
        toggles_layout.addWidget(self.normalize_check)
        toggles_layout.addWidget(self.trim_check)
        audio_form.addRow("", toggles_row)
        layout.addWidget(audio_group)

        # --- Processing options ---
        proc_group = QGroupBox("Processing Options")
        proc_layout = QVBoxLayout(proc_group)
        chk_row = QHBoxLayout()
        self.separate_check = QCheckBox("Keep Segments")
        self.separate_check.setChecked(self.app.settings.get("separate", True))
        self.combine_check = QCheckBox("Combine Output")
        self.combine_check.setChecked(self.app.settings.get("combine", True))
        self.subtitles_check = QCheckBox("Export Subtitles (.srt)")
        self.subtitles_check.setChecked(self.app.settings.get("export_subtitles", False))
        chk_row.addWidget(self.separate_check)
        chk_row.addWidget(self.combine_check)
        chk_row.addWidget(self.subtitles_check)
        proc_layout.addLayout(chk_row)

        thread_row = QHBoxLayout()
        thread_row.addWidget(QLabel("Parallel Threads:"))
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 16)
        self.threads_spin.setValue(self.app.settings.get("num_threads", 1))
        thread_row.addWidget(self.threads_spin)
        thread_row.addWidget(QLabel("(More threads = High RAM usage)"))
        thread_row.addStretch(1)
        proc_layout.addLayout(thread_row)
        layout.addWidget(proc_group)

        layout.addStretch(1)
        self.setWidget(content)

        for w in (self.out_dir_edit, self.filename_edit):
            w.textChanged.connect(lambda _v: self.app.schedule_save())
        for w in (self.volume_spin, self.pitch_spin, self.threads_spin):
            w.valueChanged.connect(lambda _v: self.app.schedule_save())
        for w in (self.apply_fx_check, self.normalize_check, self.trim_check,
                  self.separate_check, self.combine_check, self.subtitles_check):
            w.toggled.connect(lambda _v: self.app.schedule_save())

        self.refresh_presets()
        self.refresh_fx_presets()

    # --- schema form (rebuilt on engine switch) ------------------------

    def _build_schema_form(self) -> None:
        if self.schema_form is not None:
            self.schema_layout.removeWidget(self.schema_form)
            self.schema_form.deleteLater()

        schema = self.app.backend.get_config_schema()
        lang_code = self.app.settings.get("lang_code", "a")
        voice_choices = [(v, v) for v in self.app.get_all_voices(lang_code)]
        lang_choices = [(label, code) for label, code in spec.LANGUAGES.items()]
        values = {
            "lang_code": self.app.settings.get("lang_code", "a"),
            "voice": self.app.settings.get("voice", "af_heart"),
            "speed": self.app.settings.get("speed", 1.0),
            "split_pattern": self.app.settings.get("split_pattern", r"\n+"),
            "format": self.app.settings.get("format", "wav"),
            "num_threads": self.app.settings.get("num_threads", 1),
            "caching": self.app.settings.get("caching", True),
        }
        self.schema_form = SchemaFormWidget(
            schema, values,
            choices_overrides={"voice": voice_choices, "lang_code": lang_choices},
            skip_keys={"lexicon"},
            on_change=self._on_schema_field_changed,
        )
        self.schema_layout.addWidget(self.schema_form)

    def rebuild_schema_form(self) -> None:
        """Called by app.py's switch_engine - re-renders this dock's schema
        fields for the newly-active backend. This is the fix for gui.py's
        own switch_engine docstring gap (see gui.py:530-538)."""
        self._build_schema_form()

    def refresh_voice_choices(self) -> None:
        lang_code = self.schema_form.values().get("lang_code", "a")
        voices = self.app.get_all_voices(lang_code)
        current = self.schema_form.values().get("voice")
        self.schema_form.set_choices("voice", [(v, v) for v in voices], current)

    def _on_schema_field_changed(self, key: str, _value) -> None:
        if key == "lang_code":
            self.refresh_voice_choices()
        self.app.schedule_save()

    # --- output/text helpers --------------------------------------------

    def _browse_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.out_dir_edit.setText(d)

    def _browse_file(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Select input file", filter="Documents (*.txt *.pdf *.epub)")
        if f:
            self.file_path_edit.setText(f)

    def get_text(self) -> str:
        if self.tabs.currentIndex() == 0:
            return self.text_entry.toPlainText().strip()
        fpath = self.file_path_edit.text()
        if os.path.exists(fpath):
            try:
                return self.app.engine.extract_text_from_file(fpath)
            except Exception:
                return ""
        return ""

    def get_file_path(self) -> str:
        return self.file_path_edit.text()

    def using_file_tab(self) -> bool:
        return self.tabs.currentIndex() == 1

    # --- state (feeds app._assemble_config) ------------------------------

    def get_state(self) -> dict:
        state = dict(self.schema_form.values())
        state.update({
            "out_dir": self.out_dir_edit.text(),
            "filename": self.filename_edit.text(),
            "volume": self.volume_spin.value(),
            "pitch": self.pitch_spin.value(),
            "normalize": self.normalize_check.isChecked(),
            "trim_silence": self.trim_check.isChecked(),
            "separate": self.separate_check.isChecked(),
            "combine": self.combine_check.isChecked(),
            "export_subtitles": self.subtitles_check.isChecked(),
        })
        return state

    def apply_fx_enabled(self) -> bool:
        return self.apply_fx_check.isChecked()

    # --- presets (presets/*.json, shared with Tk) -------------------------

    def refresh_presets(self) -> None:
        presets = ["Select Preset..."]
        if os.path.exists(qt_app_module.PRESETS_DIR):
            files = [f for f in os.listdir(qt_app_module.PRESETS_DIR) if f.endswith(".json")]
            presets.extend(f[:-5] for f in files)
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItems(presets)
        self.preset_combo.setCurrentText("Select Preset...")
        self.preset_combo.blockSignals(False)

    def _save_preset_dialog(self) -> None:
        name, ok = QInputDialog.getText(self, "Save Preset", "Enter preset name:")
        if not ok or not name:
            return
        name = re.sub(r'[<>:"/\\|?*]', "", name).strip()
        if not name:
            return

        state = self.get_state()
        data = {
            "voice": state.get("voice"),
            "speed": state.get("speed"),
            "volume": state.get("volume"),
            "pitch": state.get("pitch"),
            "split_pattern": state.get("split_pattern"),
            "normalize": state.get("normalize"),
            "trim": state.get("trim_silence"),
            "format": state.get("format"),
            "apply_fx": self.apply_fx_enabled(),
            "fx_preset": self.fx_preset_combo.currentText(),
        }
        fpath = os.path.join(qt_app_module.PRESETS_DIR, f"{name}.json")
        try:
            os.makedirs(qt_app_module.PRESETS_DIR, exist_ok=True)
            with open(fpath, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=4)
            QMessageBox.information(self, "Saved", f"Preset '{name}' saved successfully.")
            self.refresh_presets()
            self.preset_combo.setCurrentText(name)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save preset: {e}")

    def _on_preset_selected(self, name: str) -> None:
        if not name or name == "Select Preset...":
            return
        fpath = os.path.join(qt_app_module.PRESETS_DIR, f"{name}.json")
        if not os.path.exists(fpath):
            return
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load preset: {e}")
            return

        values = {}
        if "voice" in data:
            values["voice"] = data["voice"]
        if "speed" in data:
            values["speed"] = data["speed"]
        if "split_pattern" in data:
            values["split_pattern"] = data["split_pattern"]
        if "format" in data:
            values["format"] = data["format"]
        if values:
            self.schema_form.set_values(values)
        if "volume" in data:
            self.volume_spin.setValue(data["volume"])
        if "pitch" in data:
            self.pitch_spin.setValue(data["pitch"])
        if "normalize" in data:
            self.normalize_check.setChecked(data["normalize"])
        if "trim" in data:
            self.trim_check.setChecked(data["trim"])
        if "apply_fx" in data:
            self.apply_fx_check.setChecked(data["apply_fx"])
        fx_name = data.get("fx_preset")
        if fx_name and fx_name != "Select FX Preset...":
            self.app.fx_dock.load_preset(fx_name)
            self.fx_preset_combo.setCurrentText(fx_name)

    # --- FX preset combo mirror (kept in sync with the FX dock's own combo) --

    def refresh_fx_presets(self) -> None:
        presets = ["Select FX Preset..."]
        if os.path.exists(qt_app_module.FX_PRESETS_DIR):
            files = [f for f in os.listdir(qt_app_module.FX_PRESETS_DIR) if f.endswith(".json")]
            presets.extend(f[:-5] for f in files)
        self.fx_preset_combo.blockSignals(True)
        self.fx_preset_combo.clear()
        self.fx_preset_combo.addItems(presets)
        self.fx_preset_combo.setCurrentText("Select FX Preset...")
        self.fx_preset_combo.blockSignals(False)

    def _on_fx_preset_selected(self, name: str) -> None:
        if not name or name == "Select FX Preset...":
            return
        self.app.fx_dock.load_preset(name)
