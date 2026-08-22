"""Voice Reference dock: browse a reference WAV, get (and edit) an
auto-transcript of it via kokoro_gui/engine/asr.py, and save it under a name
so it shows up as a selectable "voice" for any backend whose
`capabilities.supports_voice_cloning` is true (today: Audio8BackendAdapter -
kokoro_gui/engines/audio8_tts.py). Shown only for such a backend - see
app.py's `_sync_voice_clone_dock`, the same show/hide-on-engine-switch
pattern `_sync_mixing_dock` uses for the Mixing dock.

Saving is required before a reference can be used for generation - there is
no "generate with an unsaved wav" path, deliberately: the Generation dock's
Voice dropdown is the single source of truth for which reference gets used
(populated from `Audio8ReferenceStore.list_references()` via
`app.backend.get_voices()`), so there is never a question of whether a
freshly-browsed-but-unsaved wav or the dropdown's selection "wins".
"""
from __future__ import annotations

import asyncio
import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDockWidget, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from kokoro_gui.engine.asr import transcribe_wav
from kokoro_gui.engines import audio8_tts
from kokoro_gui.engines.audio8_tts import Audio8ReferenceStore


class VoiceCloneDock(QDockWidget):
    transcribeFinished = Signal(bool, str)
    saveFinished = Signal(bool, str)

    def __init__(self, app, parent=None):
        super().__init__("Voice Reference", parent)
        self.setObjectName("dock_voice_clone")
        self.app = app
        self.transcribeFinished.connect(self._on_transcribe_finished)
        self.saveFinished.connect(self._on_save_finished)

        content = QWidget()
        layout = QVBoxLayout(content)

        layout.addWidget(QLabel("<b>Reference Audio</b>"))
        wav_row = QHBoxLayout()
        self.wav_path_edit = QLineEdit()
        wav_row.addWidget(self.wav_path_edit, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_wav)
        wav_row.addWidget(browse_btn)
        layout.addLayout(wav_row)

        layout.addWidget(QLabel("<b>Transcript</b> (what's said in the audio)"))
        self.transcript_edit = QPlainTextEdit()
        self.transcript_edit.setFixedHeight(100)
        layout.addWidget(self.transcript_edit)

        self.transcribe_btn = QPushButton("\U0001F3A4 Auto-Transcribe")
        self.transcribe_btn.clicked.connect(self._on_transcribe_clicked)
        layout.addWidget(self.transcribe_btn)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("Save As:"))
        self.name_edit = QLineEdit()
        save_row.addWidget(self.name_edit, 1)
        save_btn = QPushButton("Save Reference")
        save_btn.clicked.connect(self._on_save_clicked)
        save_row.addWidget(save_btn)
        layout.addLayout(save_row)

        layout.addWidget(QLabel("<b>Saved References:</b>"))
        self.list_scroll = QScrollArea()
        self.list_scroll.setWidgetResizable(True)
        self.list_scroll.setFixedHeight(180)
        self._list_container = QWidget()
        self._list_layout = QVBoxLayout(self._list_container)
        self.list_scroll.setWidget(self._list_container)
        layout.addWidget(self.list_scroll)

        layout.addStretch(1)
        self.setWidget(content)
        self.refresh_list()

    # --- reference audio / transcript -----------------------------------

    def _browse_wav(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select reference audio", filter="Audio (*.wav)")
        if path:
            self.wav_path_edit.setText(path)

    def _on_transcribe_clicked(self) -> None:
        wav_path = self.wav_path_edit.text().strip()
        if not wav_path or not os.path.exists(wav_path):
            QMessageBox.warning(self, "Error", "Select a reference audio file first.")
            return

        self.transcribe_btn.setEnabled(False)
        self.status_label.setText("Transcribing...")

        def _done(future):
            try:
                text = future.result()
                self.transcribeFinished.emit(True, text)
            except Exception as e:
                self.transcribeFinished.emit(False, str(e))

        future = self.app.engine.worker.run_coro(asyncio.to_thread(transcribe_wav, wav_path))
        future.add_done_callback(_done)

    def _on_transcribe_finished(self, success: bool, payload: str) -> None:
        self.transcribe_btn.setEnabled(True)
        if success:
            self.transcript_edit.setPlainText(payload)
            self.status_label.setText("Transcribed - review/edit before saving.")
        else:
            self.status_label.setText(f"Transcription failed: {payload}")

    # --- saved references (name -> wav+transcript sidecar pair) ---------

    def _on_save_clicked(self) -> None:
        name = self.name_edit.text().strip()
        wav_path = self.wav_path_edit.text().strip()
        transcript = self.transcript_edit.toPlainText().strip()

        if not name:
            QMessageBox.warning(self, "Error", "Enter a name for this voice reference.")
            return
        if not wav_path or not os.path.exists(wav_path):
            QMessageBox.warning(self, "Error", "Select a reference audio file first.")
            return
        if not transcript:
            QMessageBox.warning(self, "Error", "Enter or auto-transcribe a transcript first.")
            return
        if name in Audio8ReferenceStore.list_references():
            if QMessageBox.question(self, "Overwrite", f"Reference '{name}' exists. Overwrite?") != QMessageBox.StandardButton.Yes:
                return

        try:
            Audio8ReferenceStore.save_reference(name, wav_path, transcript)
            self.saveFinished.emit(True, name)
        except Exception as e:
            self.saveFinished.emit(False, str(e))

    def _on_save_finished(self, success: bool, payload: str) -> None:
        if success:
            self.status_label.setText(f"Saved: {payload}")
            self.refresh_list()
        else:
            self.status_label.setText(f"Save failed: {payload}")

    def _load_reference(self, name: str) -> None:
        """Loads a saved reference back into the editable fields above, for
        review/edit/re-save (the user's "edit after if needed" path)."""
        self.name_edit.setText(name)
        self.wav_path_edit.setText(os.path.abspath(os.path.join(audio8_tts.AUDIO8_REFS_DIR, f"{name}.wav")))
        self.transcript_edit.setPlainText(Audio8ReferenceStore.get_transcript(name))

    def refresh_list(self) -> None:
        if hasattr(self.app, "generation_dock") and self.app.generation_dock is not None:
            self.app.generation_dock.refresh_voice_choices()

        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        names = Audio8ReferenceStore.list_references()
        if not names:
            self._list_layout.addWidget(QLabel("No saved voice references yet."))
            return

        for name in names:
            row = QFrame()
            row_layout = QHBoxLayout(row)
            load_btn = QPushButton(name)
            load_btn.setFlat(True)
            load_btn.clicked.connect(lambda _c=False, n=name: self._load_reference(n))
            row_layout.addWidget(load_btn, 1)
            del_btn = QPushButton("✕")
            del_btn.clicked.connect(lambda _c=False, n=name: self.delete_reference(n))
            row_layout.addWidget(del_btn)
            self._list_layout.addWidget(row)

    def delete_reference(self, name: str) -> None:
        if QMessageBox.question(self, "Confirm", f"Delete voice reference '{name}'?") != QMessageBox.StandardButton.Yes:
            return
        try:
            Audio8ReferenceStore.delete_reference(name)
            self.refresh_list()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete: {e}")
