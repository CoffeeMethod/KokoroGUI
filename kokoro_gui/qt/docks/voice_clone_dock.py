"""Voice Reference dock: browse a reference WAV, get (and edit) an
auto-transcript of it via kokoro_gui/engine/asr.py, and save it under a name
so it shows up as a selectable "voice" for any backend whose
`capabilities.supports_voice_cloning` is true. Shown only for such a backend - see
app.py's `_sync_voice_clone_dock`, the same show/hide-on-engine-switch
pattern `_sync_mixing_dock` uses for the Mixing dock. It edits the Voices
tab's engine (`app.voices_backend()`), which an Engine row on top can move
to any engine with a voice editor without touching a character (grill EN3).

The auto-transcribe step itself can run on any of `kokoro_gui.engine.asr`'s
registered engines (`ASR_ENGINES`): the default "Whisper" (local, downloads
its model on first use after asking, see `kokoro_gui/qt/asr_prompt.py`),
"Audio8-ASR-0.1B" or "Vosk" (fully offline, needs a model folder downloaded
by hand). The engine choice is persisted in `app.settings`
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

Saving is required before a Reference can be used for generation. Save applies
it to the active character when the editor and character use the same engine;
Use explicitly assigns an existing Reference. Settings shows the assignment
without a second picker. Editors and assignments use backend voice kinds and
hooks, never engine IDs. Engines that use only an excerpt report its duration,
so Auto-Transcribe covers the same audio that generation uses.
"""
from __future__ import annotations

import asyncio
import os
import tempfile

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QDockWidget, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from kokoro_gui.engine.asr import (
    ASR_ENGINES, get_vosk_model_path, reload_vosk_model_path, set_vosk_model_path, transcribe_wav,
)
from kokoro_gui.qt import asr_prompt
from kokoro_gui.qt.docks.scrolling import scrollable
from kokoro_gui.qt.docks.voice_header import engine_header


# A reference clip is a few seconds to a few minutes of speech. Anything past
# this is the wrong file, and ASR and the copy into the store read all of it.
MAX_REFERENCE_BYTES = 200 * 1024 * 1024


def _is_network_path(path: str) -> bool:
    r"""A UNC path (`\\server\share\x.wav` or `//server/x.wav`), which makes
    Windows connect to that host just to look at the file. The extended form
    `\\?\C:\...` that `realpath` can give a local drive is local."""
    if path[:2] not in ("\\\\", "//"):
        return False
    return path[2:4] not in ("?\\", "?/") or path[5:6] != ":"


def _transcribe_reference(wav_path, duration, engine, model_path):
    """Transcribe the model's excerpt off the GUI thread; clean up the copy."""
    if duration is None:
        return transcribe_wav(wav_path, engine=engine, model_path=model_path)
    import soundfile as sf

    with sf.SoundFile(wav_path) as source:
        audio = source.read(frames=int(float(duration) * source.samplerate), dtype="float32")
        sample_rate = source.samplerate
    with tempfile.TemporaryDirectory(prefix="kokorogui-reference-") as directory:
        excerpt = os.path.join(directory, "reference.wav")
        sf.write(excerpt, audio, sample_rate)
        return transcribe_wav(excerpt, engine=engine, model_path=model_path)


class VoiceCloneDock(QDockWidget):
    transcribeFinished = Signal(bool, str, int)  # success, text or error, token
    saveFinished = Signal(bool, str)

    def __init__(self, app, parent=None):
        super().__init__("Voice Reference", parent)
        self.setObjectName("dock_voice_clone")
        self.app = app
        # The engine this editor was built for, and its reference store;
        # app.py rebuilds the dock when the Voices tab moves to another
        # cloning engine.
        self.backend_id = app.voices_backend().id
        # Bumped when a transcription starts and when the dock is retired.
        # A result carrying an older token is dropped.
        self._transcribe_token = 0
        self.store = app.voices_backend().voice_store
        self.transcribeFinished.connect(self._on_transcribe_finished)
        self.saveFinished.connect(self._on_save_finished)

        content = QWidget()
        layout = QVBoxLayout(content)
        header, self.engine_combo = engine_header(app, self.backend_id)
        layout.addLayout(header)

        layout.addWidget(QLabel("<b>Reference Audio</b>"))
        wav_row = QHBoxLayout()
        self.wav_path_edit = QLineEdit()
        wav_row.addWidget(self.wav_path_edit, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_wav)
        wav_row.addWidget(browse_btn)
        layout.addLayout(wav_row)

        transcript_label = QLabel("<b>Transcript</b> (review/edit before saving)")
        layout.addWidget(transcript_label)
        self.transcript_edit = QPlainTextEdit()
        self.transcript_edit.setFixedHeight(100)
        self.transcript_edit.setPlaceholderText("Type the Reference transcript or click Auto-Transcribe.")
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
        self.list_scroll.setMinimumHeight(120)
        self._list_container = QWidget()
        self._list_layout = QVBoxLayout(self._list_container)
        self.list_scroll.setWidget(self._list_container)
        layout.addWidget(self.list_scroll, 1)

        self.setWidget(scrollable(content))
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

    def _checked_wav_path(self, path: str) -> str | None:
        """The real path of the reference wav in the path field, or None
        after a warning. Refuses a network path (checked before anything
        touches the file, since `realpath` would connect to the host), a
        non-file, a file over `MAX_REFERENCE_BYTES` and a non-.wav name."""
        path = path.strip()
        if not path:
            QMessageBox.warning(self, "Error", "Select a reference audio file first.")
            return None
        if _is_network_path(path):
            QMessageBox.warning(self, "Error", "Network paths aren't supported; copy the file locally first.")
            return None
        real = os.path.realpath(path)
        if _is_network_path(real):
            QMessageBox.warning(self, "Error", "Network paths aren't supported; copy the file locally first.")
            return None
        if not os.path.isfile(real):
            QMessageBox.warning(self, "Error", "Select a reference audio file first.")
            return None
        if os.path.splitext(real)[1].lower() != ".wav":
            QMessageBox.warning(self, "Error", "The reference audio must be a .wav file.")
            return None
        if os.path.getsize(real) > MAX_REFERENCE_BYTES:
            QMessageBox.warning(self, "Error", f"The reference audio is over {MAX_REFERENCE_BYTES // (1024 * 1024)} MB.")
            return None
        return real

    def _on_transcribe_clicked(self) -> None:
        wav_path = self._checked_wav_path(self.wav_path_edit.text())
        if wav_path is None:
            return

        engine = self.asr_engine_combo.currentData() or ASR_ENGINES[0].id
        vosk_model_path = self.vosk_model_edit.text().strip()
        if engine == "vosk" and not vosk_model_path:
            QMessageBox.warning(self, "Error", "Enter a Vosk model folder first.")
            return

        downloading = False
        if engine == "whisper":
            choice, downloading = asr_prompt.confirm_whisper_download(self)
            if choice == asr_prompt.CANCEL:
                return
            if choice == asr_prompt.OTHER_ENGINE:
                other = next(i for i in range(self.asr_engine_combo.count())
                             if self.asr_engine_combo.itemData(i) != "whisper")
                self.asr_engine_combo.setCurrentIndex(other)
                self.status_label.setText(
                    f"Switched to {self.asr_engine_combo.currentText()}. Click Auto-Transcribe to use it.")
                return

        self.transcribe_btn.setEnabled(False)
        self.status_label.setText("Downloading Whisper model..." if downloading else "Transcribing...")
        # The worker thread can't be interrupted. Retiring the dock or
        # starting another run bumps the token, and `_on_transcribe_finished`
        # drops this run's result when it finally arrives.
        self._transcribe_token += 1
        token = self._transcribe_token

        def _done(future):
            try:
                outcome = (True, future.result())
            except Exception as e:
                outcome = (False, str(e))
            try:
                self.transcribeFinished.emit(*outcome, token)
            except RuntimeError:
                pass  # the dock was deleted while the thread ran

        # Uses whatever's currently typed in the Vosk model field, whether or
        # not it's been Saved yet - transcribing shouldn't require a save
        # first, only persisting the path for next run/the standalone CLI does.
        future = self.app.voices_backend().run(
            asyncio.to_thread(_transcribe_reference, wav_path,
                              self._transcription_duration(), engine, vosk_model_path or None)
        )
        future.add_done_callback(_done)

    def _transcription_duration(self):
        backend = self.app.voices_backend()
        hook = getattr(backend, "reference_transcription_duration", None)
        return hook(self.app.engine_settings(backend.id)) if hook is not None else None

    def retire(self) -> None:
        """Called by `QtTTSApp._drop_voices_dock` when the Voices tab moves
        to another engine: a transcription still running won't write its
        result into this dock's fields."""
        self._transcribe_token += 1

    def _on_transcribe_finished(self, success: bool, payload: str, token: int) -> None:
        if token != self._transcribe_token:
            return
        self.transcribe_btn.setEnabled(True)
        if success:
            self.transcript_edit.setPlainText(payload)
            self.status_label.setText("Transcribed - review/edit before saving.")
        else:
            # The asr module's own errors already say "Transcription failed".
            prefix = "" if payload.startswith("Transcription failed") else "Transcription failed: "
            self.status_label.setText(f"{prefix}{payload}")

    # --- saved references (name -> wav+transcript sidecar pair) ---------

    def _on_save_clicked(self) -> None:
        name = self.name_edit.text().strip()
        transcript = self.transcript_edit.toPlainText().strip()

        if not name:
            QMessageBox.warning(self, "Error", "Enter a name for this voice reference.")
            return
        wav_path = self._checked_wav_path(self.wav_path_edit.text())
        if wav_path is None:
            return
        if not transcript and getattr(self.store, "requires_transcript", True):
            QMessageBox.warning(self, "Error", "Enter or auto-transcribe a transcript first.")
            return
        if name in self.store.list_references():
            if QMessageBox.question(self, "Overwrite", f"Reference '{name}' exists. Overwrite?") != QMessageBox.StandardButton.Yes:
                return

        try:
            self.store.save_reference(name, wav_path, transcript)
            self.saveFinished.emit(True, name)
        except Exception as e:
            self.saveFinished.emit(False, str(e))

    def _on_save_finished(self, success: bool, payload: str) -> None:
        if success:
            self.status_label.setText(f"Saved: {payload}")
            self.refresh_list()
            character = self.app.active_character()
            if character is not None and character.backend_id == self.backend_id:
                self._use_reference(payload)
        else:
            self.status_label.setText(f"Save failed: {payload}")

    def _use_reference(self, name: str) -> None:
        """Explicitly assign a saved Reference to the active character."""
        if self.app.is_busy():
            QMessageBox.warning(self, "Busy", "Cancel the current job before changing a voice.")
            return
        character = self.app.active_character()
        if character is None:
            QMessageBox.warning(self, "No character", "Add a character in Edit > Characters first.")
            return
        if character.backend_id != self.backend_id:
            if not self.app.set_character_engine(character, self.backend_id):
                return
        character.preset_data["voice"] = name
        self.app.set_engine_setting(self.backend_id, "voice", name)
        self.app.commit_character_edit(character)
        self.status_label.setText(f"Using {name} for {character.name}.")

    def _load_reference(self, name: str) -> None:
        """Loads a saved reference back into the editable fields above, for
        review/edit/re-save (the user's "edit after if needed" path)."""
        self.name_edit.setText(name)
        project_dir = getattr(self.app, "project_dir", None)
        wav = self.store.find_wav(name, project_dir) or self.store.global_wav_path(name)
        self.wav_path_edit.setText(wav)
        self.transcript_edit.setPlainText(self.store.get_transcript(name, project_dir))

    def refresh_list(self) -> None:
        self.app.refresh_voice_choices()

        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        # Project-local references (a .tbaw's engines/audio8/refs/) show
        # alongside the global store; a name in both is the project's.
        names = self.store.list_references(getattr(self.app, "project_dir", None))
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
            use_btn = QPushButton("Use")
            use_btn.setToolTip("Assign this saved Reference to the active character.")
            use_btn.clicked.connect(lambda _c=False, n=name: self._use_reference(n))
            row_layout.addWidget(use_btn)
            del_btn = QPushButton("✕")
            del_btn.clicked.connect(lambda _c=False, n=name: self.delete_reference(n))
            row_layout.addWidget(del_btn)
            self._list_layout.addWidget(row)

    def delete_reference(self, name: str) -> None:
        if QMessageBox.question(self, "Confirm", f"Delete voice reference '{name}'?") != QMessageBox.StandardButton.Yes:
            return
        try:
            self.store.delete_reference(name)
            self.refresh_list()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete: {e}")
