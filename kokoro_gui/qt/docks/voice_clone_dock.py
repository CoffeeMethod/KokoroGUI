"""Voice Reference dock: browse a reference WAV, get (and edit) an
auto-transcript of it via kokoro_gui/engine/asr.py, and save it under a name
so it shows up as a selectable "voice" for any backend whose
`capabilities.supports_voice_cloning` is true (today: Audio8BackendAdapter -
kokoro_gui/engines/audio8_tts.py). Shown only for such a backend - see
app.py's `_sync_voice_clone_dock`, the same show/hide-on-engine-switch
pattern `_sync_mixing_dock` uses for the Mixing dock.

The auto-transcribe step itself can run on either of `kokoro_gui.engine.asr`'s
two registered engines (`ASR_ENGINES`) - the default "Audio8-ASR-0.1B"
(online, higher quality) or "Vosk" (fully offline, needs a model folder
downloaded by hand). The engine choice is persisted in `app.settings`
(`asr_engine`, pulled via `get_state()` the same way `FXDock`/
`GenerationDock` persist their own widget state) - but Vosk's model folder
is *not*: it lives in the `VOSK_MODEL_PATH` environment variable, normally
via a `.env` file at the project root, rather than in `config_qt.json`,
since it's a one-time deployment detail rather than a per-session GUI
preference like every other setting this dock/`FXDock`/`GenerationDock`
persist. It's still editable from here though - Browse or type a path and
click Save to write it into `.env` (`kokoro_gui.engine.asr.set_vosk_model_path`),
or Reload to discard an unsaved edit and re-read whatever's actually in
`.env` right now (picks up a change made by hand while the app was already
running).

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
    QComboBox, QDockWidget, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from kokoro_gui.engine.asr import (
    ASR_ENGINES, get_vosk_model_path, reload_vosk_model_path, set_vosk_model_path, transcribe_wav,
)
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

        engine_row = QHBoxLayout()
        engine_row.addWidget(QLabel("ASR Engine:"))
        self.asr_engine_combo = QComboBox()
        for info in ASR_ENGINES:
            self.asr_engine_combo.addItem(info.display_name, info.id)
        self.asr_engine_combo.setToolTip("\n\n".join(f"{i.display_name}: {i.description}" for i in ASR_ENGINES))
        saved_engine_idx = self.asr_engine_combo.findData(self.app.settings.get("asr_engine", ASR_ENGINES[0].id))
        self.asr_engine_combo.setCurrentIndex(saved_engine_idx if saved_engine_idx >= 0 else 0)
        self.asr_engine_combo.currentIndexChanged.connect(self._on_asr_engine_changed)
        engine_row.addWidget(self.asr_engine_combo, 1)
        layout.addLayout(engine_row)

        self.vosk_row = QWidget()
        vosk_row_layout = QHBoxLayout(self.vosk_row)
        vosk_row_layout.setContentsMargins(0, 0, 0, 0)
        vosk_row_layout.addWidget(QLabel("Vosk Model:"))
        self.vosk_model_edit = QLineEdit()
        self.vosk_model_edit.setToolTip(
            "Folder of an unzipped model from https://alphacephei.com/vosk/models.\n"
            "Save writes this to VOSK_MODEL_PATH in a .env file at the project root."
        )
        vosk_row_layout.addWidget(self.vosk_model_edit, 1)
        vosk_browse_btn = QPushButton("Browse...")
        vosk_browse_btn.clicked.connect(self._browse_vosk_model)
        vosk_row_layout.addWidget(vosk_browse_btn)
        vosk_save_btn = QPushButton("Save")
        vosk_save_btn.setToolTip("Write this path to VOSK_MODEL_PATH in .env.")
        vosk_save_btn.clicked.connect(self._save_vosk_model_path)
        vosk_row_layout.addWidget(vosk_save_btn)
        vosk_reload_btn = QPushButton("Reload")
        vosk_reload_btn.setToolTip("Discard unsaved edits and re-read VOSK_MODEL_PATH from .env.")
        vosk_reload_btn.clicked.connect(self._reload_vosk_model_path)
        vosk_row_layout.addWidget(vosk_reload_btn)
        layout.addWidget(self.vosk_row)

        self._sync_vosk_model_edit()
        self.vosk_row.setVisible(self.asr_engine_combo.currentData() == "vosk")

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

    # --- ASR engine picker -------------------------------------------------

    def _on_asr_engine_changed(self, _index: int) -> None:
        self.vosk_row.setVisible(self.asr_engine_combo.currentData() == "vosk")
        self.app.schedule_save()

    def _sync_vosk_model_edit(self) -> None:
        """Fills the Vosk model field from whatever's currently in
        `VOSK_MODEL_PATH` - used at dock construction and by Reload, both of
        which mean "discard any unsaved edit and show what's really there"."""
        self.vosk_model_edit.setText(get_vosk_model_path())

    def _browse_vosk_model(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Vosk model folder")
        if path:
            self.vosk_model_edit.setText(path)
            self._save_vosk_model_path()

    def _save_vosk_model_path(self) -> None:
        set_vosk_model_path(self.vosk_model_edit.text())
        self.status_label.setText("Saved VOSK_MODEL_PATH to .env.")

    def _reload_vosk_model_path(self) -> None:
        reload_vosk_model_path()
        self._sync_vosk_model_edit()

    def get_state(self) -> dict:
        """Pulled into `app.settings` by `QtTTSApp.save_settings` so the
        chosen ASR engine survives a restart, mirroring how
        `FXDock.get_state()`/`GenerationDock.get_state()` are pulled. The
        Vosk model path isn't part of this - see this module's docstring."""
        return {"asr_engine": self.asr_engine_combo.currentData() or ASR_ENGINES[0].id}

    def _on_transcribe_clicked(self) -> None:
        wav_path = self.wav_path_edit.text().strip()
        if not wav_path or not os.path.exists(wav_path):
            QMessageBox.warning(self, "Error", "Select a reference audio file first.")
            return

        engine = self.asr_engine_combo.currentData() or ASR_ENGINES[0].id
        vosk_model_path = self.vosk_model_edit.text().strip()
        if engine == "vosk" and not vosk_model_path:
            QMessageBox.warning(self, "Error", "Enter a Vosk model folder first.")
            return

        self.transcribe_btn.setEnabled(False)
        self.status_label.setText("Transcribing...")

        def _done(future):
            try:
                text = future.result()
                self.transcribeFinished.emit(True, text)
            except Exception as e:
                self.transcribeFinished.emit(False, str(e))

        # Uses whatever's currently typed in the Vosk model field, whether or
        # not it's been Saved yet - transcribing shouldn't require a save
        # first, only persisting the path for next run/the standalone CLI does.
        future = self.app.engine.worker.run_coro(
            asyncio.to_thread(transcribe_wav, wav_path, engine=engine, model_path=vosk_model_path or None)
        )
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
