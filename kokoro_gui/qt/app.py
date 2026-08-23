"""QtTTSApp: the PySide6 shell (workstream 3a of PLAN_qt_and_engine_abstraction.md),
now the sole GUI frontend - the former CustomTkinter app (`gui.py`,
`kokoro_gui/ui/*.py`) was retired once this reached parity.

A `QMainWindow` + `QDockWidget` shell. Talks to `KokoroEngine` and the
`kokoro_gui.engines` registry the same way the retired Tk frontend did.

`CONFIG_FILE`/`PRESETS_DIR`/`FX_PRESETS_DIR` are defined here, at module
level, before the `kokoro_gui.qt.docks` import below - the dock modules do
`import kokoro_gui.qt.app as qt_app_module` and read `qt_app_module.PRESETS_DIR`
etc. qualified at call time, which makes this a circular import; defining
these names before triggering that import keeps it safe (Python binds the
dock modules' `qt_app_module` name to this already-partially-initialized
module, and by the time any dock function actually reads
`qt_app_module.PRESETS_DIR` the whole package has finished importing anyway).
"""
from __future__ import annotations

import os
import tempfile
import time

import playback
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QLabel, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from kokoro_engine import KokoroEngine
from kokoro_gui.engine.time_utils import format_duration
from kokoro_gui.engines import registry as engine_registry
from kokoro_gui.qt import spec
from kokoro_gui.qt import settings as qt_settings
from kokoro_gui.qt.signals import EngineSignalBridge, wire_engine

CONFIG_FILE = "config_qt.json"
PRESETS_DIR = "presets"
FX_PRESETS_DIR = os.path.join(PRESETS_DIR, "fx")

from kokoro_gui.qt.docks import FXDock, GenerationDock, LexiconDock, MixingDock, VoiceCloneDock  # noqa: E402


class QtTTSApp(QMainWindow):
    previewFinished = Signal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Kokoro TTS (Qt)")
        self.resize(1100, 800)

        os.makedirs(PRESETS_DIR, exist_ok=True)
        os.makedirs(FX_PRESETS_DIR, exist_ok=True)

        self.settings = qt_settings.load_settings(CONFIG_FILE)
        self.jit_enabled = self.settings.get("jit_enabled", False)
        self.timecode_format = "%Y%m%d%H%M%S"

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self.save_settings)

        self.mixing_dock: MixingDock | None = None
        self.voice_clone_dock: VoiceCloneDock | None = None
        self.generation_dock: GenerationDock | None = None

        # --- Engine / backend ---
        self.engine = KokoroEngine()
        self.bridge = EngineSignalBridge()
        wire_engine(self.engine, self.bridge)
        self.backend = engine_registry.get_engine("kokoro", engine=self.engine)
        self._connect_bridge(self.bridge)

        self.previewFinished.connect(self._on_preview_finished)

        self._build_toolbar()
        self._build_docks()
        self._build_action_bar()

        qt_settings.restore_window_state(self, self.settings)

        self.status_label.setText("Initializing engine...")
        # Read back through the Generation dock rather than raw
        # self.settings["lang_code"]: that setting is shared across engine
        # backends whose lang_code value spaces don't overlap (Kokoro's
        # single-letter codes vs. e.g. Audio8's full language names), and a
        # value saved while a different backend was active would otherwise
        # be fed straight into this (now-Kokoro) pipeline init unvalidated.
        # The schema form's combo already reconciled it to a valid default
        # for the active backend when it was built in _build_docks() above.
        init_lang_code = self.generation_dock.get_state().get("lang_code", "a")
        self.engine.worker.run_coro(self.engine.init_pipeline_async(init_lang_code))

    # --- construction -----------------------------------------------------

    def _connect_bridge(self, bridge: EngineSignalBridge) -> None:
        bridge.status.connect(self.on_engine_status)
        bridge.progress.connect(self.on_engine_progress)
        bridge.finished.connect(self.on_engine_finish)

    def _disconnect_bridge(self, bridge: EngineSignalBridge) -> None:
        try:
            bridge.status.disconnect(self.on_engine_status)
            bridge.progress.disconnect(self.on_engine_progress)
            bridge.finished.disconnect(self.on_engine_finish)
        except Exception:
            pass

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.addWidget(QLabel(" Engine: "))

        self._engine_ids_by_display_name = {
            engine_registry.get_display_name(eid): eid for eid in engine_registry.list_engines()
        }
        from PySide6.QtWidgets import QComboBox
        self.engine_picker = QComboBox()
        self.engine_picker.addItems(list(self._engine_ids_by_display_name.keys()))
        self.engine_picker.setCurrentText(engine_registry.get_display_name(self.backend.id))
        self.engine_picker.currentTextChanged.connect(self.on_engine_picker_change)
        toolbar.addWidget(self.engine_picker)

        settings_btn = QPushButton("⚙ Settings")
        settings_btn.clicked.connect(self.open_settings_dialog)
        toolbar.addWidget(settings_btn)

    def _build_docks(self) -> None:
        self.generation_dock = GenerationDock(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.generation_dock)

        self.fx_dock = FXDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.fx_dock)

        self.lexicon_dock = LexiconDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.lexicon_dock)
        self.tabifyDockWidget(self.fx_dock, self.lexicon_dock)

        self._sync_mixing_dock()
        self._sync_voice_clone_dock()

    def _build_action_bar(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

        self.detail_label = QLabel("...")
        self.detail_label.setStyleSheet("color: gray;")
        layout.addWidget(self.detail_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.info_label = QLabel("Time: 00:00 / ETA: --:-- | 0%")
        layout.addWidget(self.info_label)

        from PySide6.QtWidgets import QHBoxLayout
        btn_row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview Audio")
        self.preview_btn.clicked.connect(self.preview_conversion)
        self.start_btn = QPushButton("Start Generation")
        self.start_btn.clicked.connect(self.start_conversion)
        self._update_start_btn_text()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_conversion)
        self.cancel_btn.setEnabled(False)
        btn_row.addWidget(self.preview_btn)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        layout.addStretch(1)
        self.setCentralWidget(central)

    # --- voice listing --

    def get_all_voices(self, lang_code: str | None = None) -> list:
        """`spec.VOICE_DB` is Kokoro's built-in named-voice table specifically
        (empty for any lang_code Kokoro doesn't define, e.g. Audio8's
        language names) - the "custom"/backend-provided half comes from
        `self.backend.get_voices(...)` generically, so this works for
        whichever engine is active rather than always scanning Kokoro's
        `.pt` directory (see `Audio8BackendAdapter.get_voices`, which lists
        saved wav+transcript references instead)."""
        if lang_code is None:
            lang_code = self.settings.get("lang_code", "a")
        standard = spec.VOICE_DB.get(lang_code, [])
        custom = [v.id for v in self.backend.get_voices(lang_code)]
        return sorted(set(standard + custom))

    # --- settings persistence -

    def schedule_save(self) -> None:
        self._save_timer.start(1000)

    def save_settings(self) -> None:
        self._save_timer.stop()

        if self.generation_dock is not None:
            gen_state = self.generation_dock.get_state()
            self.settings["lang_code"] = gen_state["lang_code"]
            self.settings["voice"] = gen_state["voice"]
            self.settings["filename"] = gen_state["filename"]
            self.settings["format"] = gen_state["format"]
            self.settings["out_dir"] = gen_state["out_dir"]
            self.settings["speed"] = gen_state["speed"]
            self.settings["volume"] = gen_state["volume"]
            self.settings["pitch"] = gen_state["pitch"]
            self.settings["num_threads"] = gen_state["num_threads"]
            self.settings["split_pattern"] = gen_state["split_pattern"]
            self.settings["separate"] = gen_state["separate"]
            self.settings["combine"] = gen_state["combine"]
            self.settings["export_subtitles"] = gen_state["export_subtitles"]
            self.settings["caching"] = gen_state["caching"]
            self.settings["normalize"] = gen_state["normalize"]
            self.settings["trim"] = gen_state["trim_silence"]
            self.settings["apply_fx"] = self.generation_dock.apply_fx_enabled()
            self.settings["jit_enabled"] = self.jit_enabled
            self.settings["engine_id"] = self.backend.id
            self.settings.update(self.fx_dock.get_state())
            qt_settings.save_window_state(self, self.settings)

        qt_settings.save_settings(CONFIG_FILE, self.settings)

    # --- config assembly ----------------

    def _assemble_config(self) -> dict:
        gen_state = self.generation_dock.get_state()
        config = {
            "engine_id": self.backend.id,
            "lang_code": gen_state["lang_code"],
            "voice": gen_state["voice"],
            "speed": gen_state["speed"],
            "split_pattern": gen_state["split_pattern"],
            # Sanitize the free-text filename field the same way voice/preset
            # names are sanitized elsewhere - it flows unvalidated into an
            # os.path.join sink in caching.py otherwise (see
            # Claude/SECURITY_AUDIT.md). No-op for a normal base filename
            # (no path separators).
            "filename": os.path.basename(gen_state["filename"]),
            "format": gen_state["format"],
            "out_dir": gen_state["out_dir"],
            "separate": gen_state["separate"],
            "combine": gen_state["combine"],
            "export_subtitles": gen_state["export_subtitles"],
            "caching": gen_state["caching"],
            "time_id": time.strftime(self.timecode_format),
            "num_threads": gen_state["num_threads"],
            "volume": gen_state["volume"],
            "pitch": gen_state["pitch"],
            "normalize": gen_state["normalize"],
            "trim_silence": gen_state["trim_silence"],
            "lexicon": self.settings.get("lexicon", {}),
        }
        if self.generation_dock.apply_fx_enabled():
            config.update(self.fx_dock.get_state())
        return config

    # --- engine picker / switch --------------------

    def on_engine_picker_change(self, display_name: str) -> None:
        engine_id = self._engine_ids_by_display_name.get(display_name)
        if engine_id is None or engine_id == self.backend.id:
            return
        self.switch_engine(engine_id)

    def switch_engine(self, engine_id: str) -> None:
        if self.cancel_btn.isEnabled():
            QMessageBox.warning(self, "Busy", "Cancel the current job before switching engines.")
            self.engine_picker.setCurrentText(engine_registry.get_display_name(self.backend.id))
            return

        old_engine = self.engine
        self._disconnect_bridge(self.bridge)

        new_backend = engine_registry.get_engine(engine_id)
        new_engine = new_backend.engine
        new_bridge = EngineSignalBridge()
        wire_engine(new_engine, new_bridge)
        self._connect_bridge(new_bridge)

        self.engine = new_engine
        self.backend = new_backend
        self.bridge = new_bridge

        # Rebuild the Generation dock's schema-driven fields for the new
        # backend and show/hide the Mixing dock.
        self.generation_dock.rebuild_schema_form()
        self._sync_mixing_dock()
        self._sync_voice_clone_dock()
        self._update_start_btn_text()

        try:
            old_engine.worker.stop()
        except Exception:
            pass

        self.status_label.setText(f"Switched engine to {new_backend.display_name}. Initializing...")
        self.status_label.setStyleSheet("color: gray;")
        # Same reasoning as __init__: read the value rebuild_schema_form()
        # just reconciled for new_backend, not the raw (possibly
        # foreign-format, e.g. Audio8's "English") self.settings value.
        new_lang_code = self.generation_dock.get_state().get("lang_code", "a")
        self.settings["lang_code"] = new_lang_code
        self.engine.worker.run_coro(self.engine.init_pipeline_async(new_lang_code))

    def _update_start_btn_text(self) -> None:
        will_stream = self.jit_enabled and self.backend.capabilities.supports_jit_streaming
        self.start_btn.setText("Start Real-time JIT" if will_stream else "Start Generation")

    def _sync_mixing_dock(self) -> None:
        wants = self.backend.capabilities.supports_voice_mixing
        if wants and self.mixing_dock is None:
            self.mixing_dock = MixingDock(self)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.mixing_dock)
            self.tabifyDockWidget(self.fx_dock, self.mixing_dock)
        elif not wants and self.mixing_dock is not None:
            self.removeDockWidget(self.mixing_dock)
            self.mixing_dock.deleteLater()
            self.mixing_dock = None

    def _sync_voice_clone_dock(self) -> None:
        wants = self.backend.capabilities.supports_voice_cloning
        if wants and self.voice_clone_dock is None:
            self.voice_clone_dock = VoiceCloneDock(self)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.voice_clone_dock)
            self.tabifyDockWidget(self.fx_dock, self.voice_clone_dock)
        elif not wants and self.voice_clone_dock is not None:
            self.removeDockWidget(self.voice_clone_dock)
            self.voice_clone_dock.deleteLater()
            self.voice_clone_dock = None

    # --- settings dialog -

    def open_settings_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Settings")
        layout = QVBoxLayout(dialog)
        jit_check = QCheckBox("Enable JIT Generation (Streaming)")
        jit_check.setChecked(self.jit_enabled)
        if not self.backend.capabilities.supports_jit_streaming:
            jit_check.setEnabled(False)
            layout.addWidget(QLabel(f"({self.backend.display_name} doesn't support streaming - runs as Standard.)"))
        layout.addWidget(jit_check)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.exec()

        self.jit_enabled = jit_check.isChecked()
        self._update_start_btn_text()
        self.save_settings()

    # --- engine callbacks (queued automatically across threads - see signals.py) -

    def on_engine_status(self, msg: str, is_error: bool) -> None:
        self.status_label.setText(msg.split("\n")[0])
        self.status_label.setStyleSheet(f"color: {'#ff5555' if is_error else 'gray'};")
        if is_error and "pip install" in msg:
            QMessageBox.critical(self, "Missing Dependencies", msg)

    def on_engine_progress(self, percent: float, elapsed: float, eta: str, detail: str) -> None:
        self.progress_bar.setValue(int(percent))
        # format_duration, not time.strftime("%M:%S", time.gmtime(elapsed)):
        # gmtime's %M is minutes-mod-60 with no %H alongside it, so a run
        # past the one-hour mark looked like it "reset" back to 00:00/59:59
        # instead of counting into a second hour.
        elapsed_str = format_duration(elapsed)
        self.info_label.setText(f"Time: {elapsed_str} / ETA: {eta} | {int(percent)}%")
        self.detail_label.setText(detail)

    def on_engine_finish(self) -> None:
        self.set_ui_state(False)

    def set_ui_state(self, is_running: bool) -> None:
        self.start_btn.setEnabled(not is_running)
        self.preview_btn.setEnabled(not is_running)
        self.cancel_btn.setEnabled(is_running)
        threads_widget = self.generation_dock.schema_form.widget_for("num_threads")
        if threads_widget is not None:
            threads_widget.setEnabled(not is_running)
        self.generation_dock.volume_spin.setEnabled(not is_running)
        self.generation_dock.pitch_spin.setEnabled(not is_running)
        if not is_running:
            self.progress_bar.setValue(0 if self.engine.cancel_event.is_set() else 100)

    # --- preview -----------------------------------

    def preview_conversion(self) -> None:
        if not self.engine.pipeline:
            QMessageBox.information(self, "Wait", "Engine is initializing... please wait 2 seconds and try again.")
            return

        text_data = self.generation_dock.get_text()
        if not text_data:
            text_data = ("This is a sample audio preview using the Koh-koh-ro Tea-Tea-S engine. "
                         "It demonstrates the voice quality and speed settings.")
        preview_text = text_data[:1000]

        state = self.generation_dock.get_state()
        extra_config = {
            "volume": state["volume"],
            "pitch": state["pitch"],
            "normalize": state["normalize"],
            "trim_silence": state["trim_silence"],
            "lexicon": self.settings.get("lexicon", {}),
        }
        if self.generation_dock.apply_fx_enabled():
            extra_config.update(self.fx_dock.get_state())

        tmp_path = os.path.join(tempfile.gettempdir(), "kokoro_preview.wav")
        self.status_label.setText("Generating preview...")
        self.status_label.setStyleSheet("color: blue;")

        def _done(future):
            try:
                success = future.result()
                payload = tmp_path if success else "Preview failed."
            except Exception as e:
                success = False
                payload = f"Preview error: {e}"
            self.previewFinished.emit(success, payload)

        future = self.engine.worker.run_coro(
            self.engine.generate_preview(preview_text, state["voice"], state["speed"], tmp_path,
                                          extra_config, lang_code=state["lang_code"])
        )
        future.add_done_callback(_done)

    def _on_preview_finished(self, success: bool, payload: str) -> None:
        if success:
            self.status_label.setText("Playing preview...")
            self.status_label.setStyleSheet("color: green;")
            playback.play(payload)
            QTimer.singleShot(3000, lambda: (self.status_label.setText("Ready"), self.status_label.setStyleSheet("color: gray;")))
        else:
            self.status_label.setText(payload)
            self.status_label.setStyleSheet("color: red;")

    # --- start/cancel ------------------------------

    def start_conversion(self) -> None:
        if self.generation_dock.using_file_tab():
            fpath = self.generation_dock.get_file_path()
            if not os.path.exists(fpath):
                QMessageBox.critical(self, "Error", "File not found.")
                return
            try:
                text_data = self.engine.extract_text_from_file(fpath)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Read failed: {e}")
                return
        else:
            text_data = self.generation_dock.get_text()

        if not text_data:
            QMessageBox.warning(self, "Empty", "No text to process.")
            return

        if not self.engine.pipeline:
            QMessageBox.information(self, "Wait", "Engine is initializing... please wait 2 seconds and try again.")
            return

        config = self._assemble_config()

        self.set_ui_state(True)
        self.progress_bar.setValue(0)

        if self.jit_enabled and self.backend.capabilities.supports_jit_streaming:
            self.engine.start_jit_conversion(text_data, config)
        else:
            self.engine.start_conversion(text_data, config)

    def cancel_conversion(self) -> None:
        self.engine.cancel()
        self.status_label.setText("Cancelling... waiting for workers...")
        self.status_label.setStyleSheet("color: orange;")

    # --- lifecycle -----------------------------------------------------------

    def closeEvent(self, event) -> None:
        self.save_settings()
        super().closeEvent(event)
