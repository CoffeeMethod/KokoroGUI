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
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QLabel, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from kokoro_engine import KokoroEngine
from kokoro_gui.daw import serialization as document_serialization
from kokoro_gui.daw.auto_split import plan_auto_split_clips
from kokoro_gui.daw.undo import AssignCharacterCommand
from kokoro_gui.engine.presets import ALLOWED_FX_PRESET_KEYS, filter_allowed_keys
from kokoro_gui.engine.time_utils import format_duration
from kokoro_gui.engines import registry as engine_registry
from kokoro_gui.qt import document_state
from kokoro_gui.qt import spec
from kokoro_gui.qt import settings as qt_settings
from kokoro_gui.qt.selection import SelectionModel
from kokoro_gui.qt.signals import EngineSignalBridge, wire_engine

CONFIG_FILE = "config_qt.json"
PRESETS_DIR = "presets"
FX_PRESETS_DIR = os.path.join(PRESETS_DIR, "fx")
DOCUMENT_FILE = "document.json"

from kokoro_gui.qt.docks import (  # noqa: E402
    FXDock, GenerationDock, LexiconDock, MixingDock, SettingsDock, TimelineDock, VoiceCloneDock,
)


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

        # Workstream 2 (Claude/PLAN_daw_ui_ux_redesign.md): the DAW document
        # model backing the transcript panel. Must exist before
        # _build_docks() below, since GenerationDock's TranscriptEditor reads
        # it at construction time.
        self.document = document_state.load_or_create_document(DOCUMENT_FILE, self.settings, PRESETS_DIR)

        # Item 1 ("Sync layer") of the DAW-for-text remaining-work roadmap:
        # the shared clip/character/range selection, read by TranscriptEditor
        # and TimelineView alike. Constructed before _build_docks() - those
        # docks' widgets (TimelineDock's TimelineView) read self.selection at
        # construction time.
        self.selection = SelectionModel()

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self.save_settings)

        self.mixing_dock: MixingDock | None = None
        self.voice_clone_dock: VoiceCloneDock | None = None
        self.generation_dock: GenerationDock | None = None
        self.settings_dock: SettingsDock | None = None
        self.timeline_dock: TimelineDock | None = None

        # --- Engine / backend ---
        self.engine = KokoroEngine()
        self.bridge = EngineSignalBridge()
        wire_engine(self.engine, self.bridge)
        self.backend = engine_registry.get_engine("kokoro", engine=self.engine)
        self._connect_bridge(self.bridge)

        self.previewFinished.connect(self._on_preview_finished)

        self._build_menu_bar()
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

    def _build_menu_bar(self) -> None:
        """The app's first menu bar (item 4, "Undo/redo", of the DAW-for-text
        roadmap's cross-workstream resolutions - nothing in this app used
        `QMainWindow.menuBar()` before this). A minimal Edit menu for now;
        expected to grow a File menu when item 10 ("ASR-anchored audio
        import") needs a discoverable "Import Audio" action."""
        edit_menu = self.menuBar().addMenu("&Edit")

        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.triggered.connect(self.undo)
        edit_menu.addAction(self.undo_action)

        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_action.triggered.connect(self.redo)
        edit_menu.addAction(self.redo_action)

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

        self.settings_dock = SettingsDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.settings_dock)

        self.fx_dock = FXDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.fx_dock)
        self.tabifyDockWidget(self.settings_dock, self.fx_dock)

        self.lexicon_dock = LexiconDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.lexicon_dock)
        self.tabifyDockWidget(self.fx_dock, self.lexicon_dock)

        self._sync_mixing_dock()
        self._sync_voice_clone_dock()

        # Workstream 3 (Claude/PLAN_daw_ui_ux_redesign.md): unconditional,
        # not capability-gated - renders Document state, which is
        # engine-independent. Bottom area (no existing dock uses it) since a
        # timeline is a wide, horizontally-scrolling strip rather than
        # something to squeeze into the already-tabbed Right column.
        self.timeline_dock = TimelineDock(self)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.timeline_dock)
        self.timeline_dock.batchGenerationProgress.connect(self.on_batch_generation_progress)
        self.timeline_dock.batchGenerationFinished.connect(self.on_batch_generation_finished)

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
        self.start_btn.clicked.connect(self.on_generate_clicked)
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

    def refresh_timeline(self) -> None:
        """App-owned cross-dock coordination point (same precedent as the
        FX-preset-combo mirroring between docks) rather than a signal/event
        bus - Workstream 4's real sync layer may replace this outright, so
        nothing more elaborate is built here yet. No-ops if the timeline
        dock doesn't exist (defensive; it's constructed unconditionally in
        _build_docks(), same guard style save_settings() already uses for
        generation_dock)."""
        if self.timeline_dock is not None:
            self.timeline_dock.refresh()

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
            if self.voice_clone_dock is not None:
                self.settings.update(self.voice_clone_dock.get_state())
            qt_settings.save_window_state(self, self.settings)

        qt_settings.save_settings(CONFIG_FILE, self.settings)
        document_serialization.save_document(self.document, DOCUMENT_FILE)

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

    def _assemble_clip_config(self, clip) -> dict:
        """The config dict for a per-clip Generate action (Workstream 3 of
        Claude/PLAN_daw_ui_ux_redesign.md) - generic app-level defaults
        (out_dir/filename/lang_code/engine_id/caching/num_threads/etc, the
        same source `_assemble_config` reads) with `clip`'s
        character/override settings merged on top, so the clip's own values
        win. Defaults must come first: `process_chunk_task` reads
        `config['voice']`/`config['split_pattern']` via direct dict
        indexing, not `.get`, so a clip with no character (an empty
        `effective_config_for_clip()`) must still end up with usable
        defaults rather than a KeyError deep in a background thread.
        """
        gen_state = self.generation_dock.get_state()
        config = {
            "engine_id": self.backend.id,
            "lang_code": gen_state["lang_code"],
            "voice": gen_state["voice"],
            "speed": gen_state["speed"],
            "split_pattern": gen_state["split_pattern"],
            "format": gen_state["format"],
            "out_dir": gen_state["out_dir"],
            "caching": gen_state["caching"],
            "time_id": time.strftime(self.timecode_format),
            "num_threads": gen_state["num_threads"],
            "volume": gen_state["volume"],
            "pitch": gen_state["pitch"],
            "normalize": gen_state["normalize"],
            "trim_silence": gen_state["trim_silence"],
            "filename": os.path.basename(gen_state["filename"]),
        }

        clip_config = dict(self.document.effective_config_for_clip(clip))
        # ALLOWED_PRESET_KEYS (what effective_config_for_clip can return)
        # whitelists "trim", but process_audio actually reads
        # "trim_silence" - every other caller of process_chunk_task
        # (start_conversion, generate_preview, jit.py) does this same
        # rename inline; Document/models.py doesn't know process_audio's
        # key names, so it's this GUI-side assembly's job.
        if "trim" in clip_config:
            clip_config["trim_silence"] = clip_config.pop("trim")
        config.update(clip_config)

        # effective_config_for_clip only ever returns the FX preset's
        # *name* (that's all ALLOWED_PRESET_KEYS permits) - resolve it into
        # actual FX values the same way _process_text_async/generate_preview
        # already do, or a character's attached FX preset would silently
        # have no audible effect.
        if config.get("apply_fx") and config.get("fx_preset"):
            fx_preset = self.engine.load_fx_preset(config["fx_preset"])
            if fx_preset:
                config.update(filter_allowed_keys(fx_preset, ALLOWED_FX_PRESET_KEYS))

        # Item 5 ("Per-clip FX button"): a clip's own fx_override - a
        # resolved FX-values dict, not a preset name - always wins over
        # whatever the character's fx_preset resolved to above, mirroring
        # the existing "clip overrides beat character preset" rule
        # effective_config_for_clip already applies for ALLOWED_PRESET_KEYS
        # fields. Merged last, deliberately.
        if clip.fx_override:
            config.update(filter_allowed_keys(clip.fx_override, ALLOWED_FX_PRESET_KEYS))

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

        # Rebuild the Settings dock's schema-driven fields for the new
        # backend and show/hide the Mixing dock.
        self.settings_dock.rebuild_schema_form()
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
        # Item 3 ("Consolidated action bar"): once the document has any
        # clips at all, Generate always runs the dirty-scoped batch path
        # (see on_generate_clicked) regardless of the JIT setting below, so
        # the button label stops describing JIT/Standard mode entirely.
        if self.document.clips:
            self.start_btn.setText("Generate")
            return
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

        paste_split_check = QCheckBox("Paste splits character/FX")
        paste_split_check.setChecked(self.settings.get("character_fx_paste_splits", True))
        layout.addWidget(paste_split_check)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        dialog.exec()

        self.jit_enabled = jit_check.isChecked()
        self.settings["character_fx_paste_splits"] = paste_split_check.isChecked()
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
        threads_widget = self.settings_dock.schema_form.widget_for("num_threads")
        if threads_widget is not None:
            threads_widget.setEnabled(not is_running)
        self.settings_dock.volume_spin.setEnabled(not is_running)
        self.settings_dock.pitch_spin.setEnabled(not is_running)
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

    def on_generate_clicked(self) -> None:
        """The action bar's single Generate entry point (item 3,
        "Consolidated action bar + batch dirty-scoped generation"). A
        document with no clips yet (nobody's ever assigned a character to
        any text) falls back to today's whole-document `start_conversion()`
        unchanged; a document with clips dispatches the dirty-scoped batch
        path instead, regardless of whether text has ever been generated
        for it before."""
        if not self.document.clips:
            self.start_conversion()
            return

        dirty = self.document.dirty_clips()
        if not dirty:
            QMessageBox.information(self, "Up to date", "All clips are already generated.")
            return

        # Flagged product decision (DAW-for-text roadmap, cross-workstream
        # resolutions): once a document has any clips, Generate always runs
        # the dirty-scoped batch path below - even if JIT streaming is
        # enabled in Settings. JIT has no per-clip/Segment output shape, so
        # it stays reachable only via the no-clips-yet fallback above. This
        # is a deliberate, confirmed behavior change for JIT users, not an
        # oversight.
        self.timeline_dock.generate_dirty_clips_requested()

    def on_batch_generation_progress(self, completed: int, total: int, current_clip_label: str) -> None:
        self.progress_bar.setValue(int((completed / total) * 100) if total else 0)
        if current_clip_label:
            self.detail_label.setText(f"Generated {completed}/{total} clips ({current_clip_label})")
        else:
            self.detail_label.setText(f"Generating {total} clip(s)...")

    def on_batch_generation_finished(self, succeeded: int, failed: int, failed_clip_ids: list) -> None:
        total = succeeded + failed
        if failed == 0:
            self.status_label.setText(f"Generated {succeeded} clip(s).")
            self.status_label.setStyleSheet("color: gray;")
        elif succeeded == 0:
            # Total failure - nothing succeeded at all - uses the same red
            # already used for a critical/error status elsewhere in this
            # file (on_engine_status, _on_clip_generation_finished).
            self.status_label.setText(f"Batch generation failed for all {failed} clip(s).")
            self.status_label.setStyleSheet("color: #ff5555;")
        else:
            # Partial failure - reuse cancel_conversion's existing warning
            # color rather than inventing a new one.
            self.status_label.setText(f"Generated {succeeded} of {total} clips ({failed} failed)")
            self.status_label.setStyleSheet("color: orange;")

    def auto_split_and_generate(self) -> None:
        """Item 7 ("Auto-split on generation + combined-vs-separate clip
        generation") of the DAW-for-text remaining-work roadmap: turns every
        `[Speaker:FX]:`-tagged span in the document (and, if
        `auto_split_by_paragraph` is on, each span's paragraph-separated
        sub-ranges too) into clips, then batch-generates them via item 3's
        machinery, reused unmodified. Same one-job-at-a-time guard every
        other generation trigger already uses."""
        if self.cancel_btn.isEnabled():
            QMessageBox.warning(self, "Busy", "Finish or cancel the current job before auto-splitting.")
            return

        triples, unmatched = plan_auto_split_clips(
            self.document, split_by_paragraph=self.settings.get("auto_split_by_paragraph", False)
        )

        if unmatched:
            names = ", ".join(sorted(set(unmatched)))
            QMessageBox.warning(
                self, "Unmatched speaker names",
                f"No character found for: {names}. Those blocks were skipped.",
            )

        if not triples:
            QMessageBox.information(self, "Nothing to split", "No taggable text found to auto-split.")
            return

        # Ascending start order, applied against the same unchanging
        # document.text - safe per assign_character_to_range's docstring,
        # since none of these commands are text edits (see auto_split.py's
        # module docstring for the full reasoning).
        for start, end, character_id in triples:
            self.document.undo_stack.push(AssignCharacterCommand(start, end, character_id))

        # Native highlighting (Claude/PLAN_text_editor_redesign.md): each
        # push above only changed app.document.runs - the live editor's
        # QTextCharFormat needs an explicit repaint to catch up, same as
        # every other undo_stack.push site.
        self.generation_dock.text_entry.rehighlight()
        self.schedule_save()
        self.refresh_timeline()

        self.timeline_dock.generate_dirty_clips_requested()

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

    # --- undo/redo (item 4, "Undo/redo") -----------------------------------
    # Delegates to the transcript editor's UndoCoordinator
    # (kokoro_gui.qt.undo_coordinator) rather than touching
    # document.undo_stack directly, so the app-wide Undo/Redo menu actions
    # behave identically to pressing Ctrl+Z/Ctrl+Shift+Z with the editor
    # focused - both pop whichever of the native-typing / custom-tagging
    # histories acted most recently (see the coordinator's module docstring
    # and Claude/PLAN_text_editor_redesign.md's undo-granularity grill
    # answer). The coordinator's own callbacks handle re-syncing the
    # editor's text/highlighting and calling schedule_save/refresh_timeline
    # - nothing left for this method to do afterward.

    def undo(self) -> None:
        self.generation_dock.text_entry.undo_coordinator.undo()

    def redo(self) -> None:
        self.generation_dock.text_entry.undo_coordinator.redo()

    # --- lifecycle -----------------------------------------------------------

    def closeEvent(self, event) -> None:
        self.save_settings()
        super().closeEvent(event)
