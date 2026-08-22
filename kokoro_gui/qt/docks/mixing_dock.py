"""Custom Voice (mixing) dock: blends two voice tensors via
`self.app.engine.mix_voices` and previews/saves the result. Shown only when
`app.backend.capabilities.supports_voice_mixing` is true - see app.py's
`_sync_mixing_dock`.

Uses the literal relative "custom_voices" path (not
`kokoro_engine.CUSTOM_VOICES_DIR`) - relies on the process cwd for this,
which is why the test fixtures `monkeypatch.chdir(tmp_path)`.
"""
from __future__ import annotations

import os
import re
import tempfile

import playback
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QDockWidget, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea, QSlider,
    QVBoxLayout, QWidget,
)

from kokoro_gui.qt import spec

CUSTOM_VOICES_DIR = "custom_voices"


class MixingDock(QDockWidget):
    previewFinished = Signal(bool, str)
    mixFinished = Signal(bool, str)

    def __init__(self, app, parent=None):
        super().__init__("Custom Voice", parent)
        self.setObjectName("dock_mixing")
        self.app = app
        self.previewFinished.connect(self._on_preview_finished)
        self.mixFinished.connect(self._on_mix_finished)

        content = QWidget()
        layout = QVBoxLayout(content)

        lang_items = list(spec.LANGUAGES.items())

        sel_grid = QGridLayout()
        sel_grid.addWidget(QLabel("Voice A:"), 0, 0)
        self.lang_a_combo = QComboBox()
        self.voice_a_combo = QComboBox()
        for label, code in lang_items:
            self.lang_a_combo.addItem(label, code)
        self.lang_a_combo.currentIndexChanged.connect(lambda _i: self._refresh_voice_list(self.lang_a_combo, self.voice_a_combo))
        sel_grid.addWidget(self.lang_a_combo, 0, 1)
        sel_grid.addWidget(self.voice_a_combo, 0, 2)

        sel_grid.addWidget(QLabel("Voice B:"), 1, 0)
        self.lang_b_combo = QComboBox()
        self.voice_b_combo = QComboBox()
        for label, code in lang_items:
            self.lang_b_combo.addItem(label, code)
        self.lang_b_combo.setCurrentIndex(0)
        self.lang_b_combo.currentIndexChanged.connect(lambda _i: self._refresh_voice_list(self.lang_b_combo, self.voice_b_combo))
        sel_grid.addWidget(self.lang_b_combo, 1, 1)
        sel_grid.addWidget(self.voice_b_combo, 1, 2)
        layout.addLayout(sel_grid)
        self._refresh_voice_list(self.lang_a_combo, self.voice_a_combo)
        self._refresh_voice_list(self.lang_b_combo, self.voice_b_combo)
        if self.voice_b_combo.count() > 1:
            self.voice_b_combo.setCurrentIndex(1)

        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Operation:"))
        self.op_combo = QComboBox()
        self.op_combo.addItems(["mix", "add", "subtract", "multiply", "divide"])
        self.op_combo.currentTextChanged.connect(self._update_ratio_label)
        op_row.addWidget(self.op_combo)
        op_row.addStretch(1)
        layout.addLayout(op_row)

        self.ratio_label = QLabel("Mix: 50% A / 50% B")
        layout.addWidget(self.ratio_label)
        self.ratio_slider = QSlider(Qt.Orientation.Horizontal)
        self.ratio_slider.setRange(0, 100)
        self.ratio_slider.setValue(50)
        self.ratio_slider.valueChanged.connect(self._update_ratio_label)
        layout.addWidget(self.ratio_slider)
        self._update_ratio_label()

        prev_row = QHBoxLayout()
        prev_row.addWidget(QLabel("Preview Language:"))
        self.preview_lang_combo = QComboBox()
        for label, code in lang_items:
            self.preview_lang_combo.addItem(label, code)
        prev_row.addWidget(self.preview_lang_combo)
        preview_btn = QPushButton("\U0001F50A Preview")
        preview_btn.clicked.connect(self.preview_mix)
        prev_row.addWidget(preview_btn)
        prev_row.addStretch(1)
        layout.addLayout(prev_row)

        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("New Voice Name:"))
        self.mix_name_edit = QLineEdit()
        save_row.addWidget(self.mix_name_edit, 1)
        save_btn = QPushButton("Create && Save")
        save_btn.clicked.connect(self.mix_voice_action)
        save_row.addWidget(save_btn)
        layout.addLayout(save_row)

        self.mix_status_label = QLabel("")
        layout.addWidget(self.mix_status_label)

        layout.addWidget(QLabel("<b>Custom Voices:</b>"))
        self.list_scroll = QScrollArea()
        self.list_scroll.setWidgetResizable(True)
        self.list_scroll.setFixedHeight(200)
        self._list_container = QWidget()
        self._list_layout = QVBoxLayout(self._list_container)
        self.list_scroll.setWidget(self._list_container)
        layout.addWidget(self.list_scroll)

        layout.addStretch(1)
        self.setWidget(content)
        self.refresh_voice_lists()

    def _ratio_value(self) -> float:
        return self.ratio_slider.value() / 100.0

    def _update_ratio_label(self, *_args) -> None:
        p = self.ratio_slider.value()
        op = self.op_combo.currentText()
        if op == "mix":
            self.ratio_label.setText(f"Mix: {100 - p}% A / {p}% B")
        elif op == "divide":
            self.ratio_label.setText(f"Op: Divide | Influence: {p}% (unstable and VERY LOUD)")
        else:
            self.ratio_label.setText(f"Op: {op.capitalize()} | Influence: {p}%")

    def _refresh_voice_list(self, lang_combo: QComboBox, voice_combo: QComboBox) -> None:
        code = lang_combo.currentData()
        voices = self.app.get_all_voices(code)
        current = voice_combo.currentText()
        voice_combo.blockSignals(True)
        voice_combo.clear()
        voice_combo.addItems(voices)
        if current in voices:
            voice_combo.setCurrentText(current)
        elif voices:
            voice_combo.setCurrentIndex(0)
        voice_combo.blockSignals(False)

    def refresh_voice_lists(self) -> None:
        if hasattr(self.app, "generation_dock") and self.app.generation_dock is not None:
            self.app.generation_dock.refresh_voice_choices()
        self._refresh_voice_list(self.lang_a_combo, self.voice_a_combo)
        self._refresh_voice_list(self.lang_b_combo, self.voice_b_combo)

        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        custom = []
        if os.path.exists(CUSTOM_VOICES_DIR):
            custom = sorted(f[:-3] for f in os.listdir(CUSTOM_VOICES_DIR) if f.endswith(".pt"))
        if not custom:
            self._list_layout.addWidget(QLabel("No custom voices found."))
        else:
            for cv in custom:
                row = QFrame()
                row_layout = QHBoxLayout(row)
                row_layout.addWidget(QLabel(cv))
                row_layout.addStretch(1)
                del_btn = QPushButton("X")
                del_btn.clicked.connect(lambda _c=False, v=cv: self.delete_custom_voice(v))
                row_layout.addWidget(del_btn)
                self._list_layout.addWidget(row)

    def delete_custom_voice(self, name: str) -> None:
        if QMessageBox.question(self, "Confirm", f"Delete voice '{name}'?") != QMessageBox.StandardButton.Yes:
            return
        try:
            path = os.path.join(CUSTOM_VOICES_DIR, f"{name}.pt")
            if os.path.exists(path):
                os.remove(path)
                self.refresh_voice_lists()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete: {e}")

    def preview_mix(self) -> None:
        v1 = self.voice_a_combo.currentText()
        v2 = self.voice_b_combo.currentText()
        ratio = self._ratio_value()
        op = self.op_combo.currentText()
        preview_lang = self.preview_lang_combo.currentData()

        preview_text = spec.MIX_PREVIEW_TEXT.get(preview_lang, spec.MIX_PREVIEW_TEXT_DEFAULT)
        tmp_voice_name = "_tmp_mix_preview"
        tmp_audio_path = os.path.join(tempfile.gettempdir(), "kokoro_mix_preview.wav")

        self.mix_status_label.setText("Generating preview...")

        async def _run_preview():
            success, msg, tensor = await self.app.engine.mix_voices(v1, v2, ratio, tmp_voice_name, op=op)
            if not success:
                return False, msg
            success = await self.app.engine.generate_preview(
                preview_text, tmp_voice_name, 1.0, tmp_audio_path, voice_tensor=tensor, lang_code=preview_lang,
            )
            try:
                p = os.path.join(CUSTOM_VOICES_DIR, f"{tmp_voice_name}.pt")
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
            return success, ""

        def _done(future):
            try:
                success, err = future.result()
            except Exception as e:
                success, err = False, str(e)
            self.previewFinished.emit(success, err)
            if success:
                playback.play(tmp_audio_path)

        future = self.app.engine.worker.run_coro(_run_preview())
        future.add_done_callback(_done)

    def _on_preview_finished(self, success: bool, err: str) -> None:
        if success:
            self.mix_status_label.setText("Playing preview...")
        else:
            self.mix_status_label.setText(f"Preview failed: {err}")

    def mix_voice_action(self) -> None:
        v1 = self.voice_a_combo.currentText()
        v2 = self.voice_b_combo.currentText()
        ratio = self._ratio_value()
        op = self.op_combo.currentText()
        name = self.mix_name_edit.text().strip()

        if not name:
            QMessageBox.warning(self, "Error", "Please enter a name for the new voice.")
            return
        if not re.match(r"^[a-zA-Z0-9_-]+$", name):
            QMessageBox.warning(self, "Error", "Invalid name. Use alphanumeric, _, - only.")
            return
        if name in self.app.get_all_voices():
            if QMessageBox.question(self, "Overwrite", f"Voice '{name}' exists. Overwrite?") != QMessageBox.StandardButton.Yes:
                return

        self.mix_status_label.setText("Mixing...")
        self.app.set_ui_state(True)
        self._pending_name = name

        def _done(future):
            try:
                success, msg, _tensor = future.result()
            except Exception as e:
                success, msg = False, str(e)
            self.mixFinished.emit(success, msg)

        future = self.app.engine.worker.run_coro(self.app.engine.mix_voices(v1, v2, ratio, name, op=op))
        future.add_done_callback(_done)

    def _on_mix_finished(self, success: bool, msg: str) -> None:
        self.app.set_ui_state(False)
        if success:
            self.mix_status_label.setText(f"Saved: {self._pending_name}")
            self.refresh_voice_lists()
        else:
            self.mix_status_label.setText(f"Error: {msg}")
