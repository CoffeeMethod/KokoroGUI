"""Generation dock: input source, output config, Processing Options, and
the speaker presets (`presets/*.json`) that snapshot the project-wide
config.

The schema-driven config fields and the hand-built Audio Control widgets
(volume/pitch/FX-preset-combo/apply_fx/normalize/trim) moved to
`kokoro_gui.qt.docks.settings_dock.SettingsDock` (item 2, "Settings panel
rescoping", of the DAW-for-text redesign's remaining-work roadmap) - they're
exactly the surface that needs to vary per clip/character, which this dock
knows nothing about. This dock keeps the transcript editor/file-path tabs,
the legacy `presets/*.json` combo (a distinct feature from
`kokoro_gui.daw.models.Character` - don't conflate them), the Output group,
and Processing Options, none of which is meaningfully scoped to a selection.
Loading/saving a generation preset now reaches into `self.app.settings_dock`
for the fields it actually touches, since the widgets live there - the
preset combo always targets the project-wide ("none") state, same as it did
before this dock had any notion of per-clip/character scoping.

Reads `kokoro_gui.qt.app.PRESETS_DIR` qualified at call time (not imported by
name) so tests can monkeypatch it into a tmp_path.
"""
from __future__ import annotations

import json
import os
import re

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDockWidget, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QMessageBox, QPushButton, QScrollArea,
    QTabWidget, QVBoxLayout, QWidget,
)

import kokoro_gui.qt.app as qt_app_module
from kokoro_gui.qt.transcript_editor import TranscriptEditor


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

        self.text_entry = TranscriptEditor(self.app)
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

        # --- Presets row (presets/*.json, always project-wide) ---
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

        # --- Output (not schema-covered, hand-built) ---
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
        layout.addWidget(proc_group)

        # --- Auto-split (item 7, "Auto-split on generation + combined-vs-
        # separate clip generation") ---
        auto_split_row = QHBoxLayout()
        self.auto_split_paragraph_check = QCheckBox("Split by paragraph")
        self.auto_split_paragraph_check.setChecked(self.app.settings.get("auto_split_by_paragraph", False))
        self.auto_split_btn = QPushButton("Auto-Split && Generate")
        self.auto_split_btn.clicked.connect(self.app.auto_split_and_generate)
        auto_split_row.addWidget(self.auto_split_paragraph_check)
        auto_split_row.addWidget(self.auto_split_btn)
        layout.addLayout(auto_split_row)

        layout.addStretch(1)
        self.setWidget(content)

        for w in (self.out_dir_edit, self.filename_edit):
            w.textChanged.connect(lambda _v: self.app.schedule_save())
        for w in (self.separate_check, self.combine_check, self.subtitles_check):
            w.toggled.connect(lambda _v: self.app.schedule_save())
        # auto_split_by_paragraph isn't read back through get_state() the way
        # the checkboxes above are (auto_split_and_generate reads it straight
        # off self.app.settings, since it's not part of the per-generation
        # config dict) - written immediately rather than left to the debounced
        # schedule_save(), so a toggle-then-click in the same instant sees the
        # new value rather than a stale one.
        self.auto_split_paragraph_check.toggled.connect(self._on_auto_split_paragraph_toggled)

        self.refresh_presets()

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
        """Project-wide ("none") config values, merging `SettingsDock`'s
        schema/Audio-Control fields (now owned by that dock) with this
        dock's own Output/Processing Options widgets - same key shape this
        method produced before those fields moved out."""
        state = dict(self.app.settings_dock.get_state())
        state.update({
            "out_dir": self.out_dir_edit.text(),
            "filename": self.filename_edit.text(),
            "separate": self.separate_check.isChecked(),
            "combine": self.combine_check.isChecked(),
            "export_subtitles": self.subtitles_check.isChecked(),
        })
        return state

    def apply_fx_enabled(self) -> bool:
        return self.app.settings_dock.apply_fx_enabled()

    def _on_auto_split_paragraph_toggled(self, checked: bool) -> None:
        self.app.settings["auto_split_by_paragraph"] = checked
        self.app.schedule_save()

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

        data = self.app.settings_dock.get_none_preset_values()
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

        self.app.settings_dock.apply_none_preset_values(data)

        fx_name = data.get("fx_preset")
        if fx_name and fx_name != "Select FX Preset...":
            self.app.fx_dock.load_preset(fx_name)
            self.app.settings_dock.set_fx_preset_display(fx_name)
