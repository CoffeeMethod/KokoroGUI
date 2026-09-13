"""QtTTSApp: the PySide6 shell, the sole GUI frontend since the CustomTkinter
app (`gui.py`, `kokoro_gui/ui/*.py`) was retired.

Reshaped by Claude/PLAN_ui_shell_redesign.md into the wireframe's 2x2 grid:
Transcript (top-left) | Settings / Audio FX / Lexicon / Voices tabs
(top-right), Timeline (bottom-left) | Transport (bottom-right). Every panel
is still a `QDockWidget`; `arrange_docks_default()` builds the grid and
`kokoro_gui.qt.workspace` saves/restores named layouts (Workspace menu).
The old toolbar and the central action bar are gone - engine/device/theme
live under Options, generate/preview/cancel and the progress line live in
the Transport dock. A File menu (`kokoro_gui.qt.project`) replaced the
implicit single `document.json`.

`CONFIG_FILE`/`PRESETS_DIR`/`FX_PRESETS_DIR`/`DOCUMENT_FILE` are defined
here, at module level, before the `kokoro_gui.qt.docks` import below - the
dock modules do `import kokoro_gui.qt.app as qt_app_module` and read
`qt_app_module.PRESETS_DIR` etc. qualified at call time, which makes this a
circular import; defining these names before triggering that import keeps
it safe.
"""
from __future__ import annotations

import os
import tempfile
import time

import playback
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox, QSizePolicy, QWidget

from kokoro_engine import KokoroEngine
from kokoro_gui.daw.arrangement import compute_arrangement
from kokoro_gui.daw.auto_split import plan_auto_split_clips
from kokoro_gui.daw.undo import AssignCharacterCommand
from kokoro_gui.engines import registry as engine_registry
from kokoro_gui.qt import document_state, fx_resolve, project as project_io, spec, theme
from kokoro_gui.qt import settings as qt_settings
from kokoro_gui.qt.selection import SelectionModel
from kokoro_gui.qt.signals import EngineSignalBridge, wire_engine
from kokoro_gui.qt.workspace import ADVANCED, SIMPLE, WorkspaceManager

CONFIG_FILE = "config_qt.json"
PRESETS_DIR = "presets"
FX_PRESETS_DIR = os.path.join(PRESETS_DIR, "fx")
# The project a fresh install (or a config with no last_project) opens.
DOCUMENT_FILE = "document.json"

from kokoro_gui.audio import post  # noqa: E402
from kokoro_gui.audio.transport import ScheduledClip, Transport  # noqa: E402
from kokoro_gui.daw.arrangement import clip_audio_duration_s  # noqa: E402
from kokoro_gui.qt.characters_dialog import CharactersDialog  # noqa: E402
from kokoro_gui.qt.docks import (  # noqa: E402
    FXDock, LexiconDock, MixingDock, SettingsDock, TimelineDock, TranscriptDock, TransportDock,
    VoiceCloneDock,
)
from kokoro_gui.qt.docks.export_dialog import ExportDialog, run_export  # noqa: E402
from kokoro_gui.qt.welcome_dialog import WelcomeDialog  # noqa: E402

APP_NAME = "KokoroGUI"
SCHEDULE_REBUILD_DEBOUNCE_MS = 100


class QtTTSApp(QMainWindow):
    previewFinished = Signal(bool, str)
    themeChanged = Signal()
    exportProgress = Signal(float, str)
    exportFinished = Signal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(1600, 1000)

        os.makedirs(PRESETS_DIR, exist_ok=True)
        os.makedirs(FX_PRESETS_DIR, exist_ok=True)

        self.settings = qt_settings.load_settings(CONFIG_FILE)
        self.jit_enabled = self.settings.get("jit_enabled", False)
        self.timecode_format = "%Y%m%d%H%M%S"

        # Project (section 7): the last-opened project, else the classic
        # document.json next to the config, else a fresh migration.
        self.project_path: str | None = None
        self.project_settings: dict = {}
        self.document = self._load_initial_document()

        self.selection = SelectionModel()

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self.save_settings)

        self._schedule_timer = QTimer(self)
        self._schedule_timer.setSingleShot(True)
        self._schedule_timer.setInterval(SCHEDULE_REBUILD_DEBOUNCE_MS)
        self._schedule_timer.timeout.connect(self._rebuild_transport_schedule)

        self.welcome_dialog: WelcomeDialog | None = None
        self.transcript_dock: TranscriptDock | None = None
        self.settings_dock: SettingsDock | None = None
        self.fx_dock: FXDock | None = None
        self.lexicon_dock: LexiconDock | None = None
        self.mixing_dock: MixingDock | None = None
        self.voice_clone_dock: VoiceCloneDock | None = None
        self.timeline_dock: TimelineDock | None = None
        self.transport_dock: TransportDock | None = None

        # --- Engine / backend ---
        self.engine = KokoroEngine()
        self.bridge = EngineSignalBridge()
        wire_engine(self.engine, self.bridge)
        self.backend = engine_registry.get_engine("kokoro", engine=self.engine)
        self._connect_bridge(self.bridge)

        self.previewFinished.connect(self._on_preview_finished)
        self.exportProgress.connect(self._on_export_progress)
        self.exportFinished.connect(self._on_export_finished)

        # Theme before any custom-painted widget exists, so their first
        # paint already reads the right palette.
        theme.apply(QApplication.instance(), self.settings.get("theme", theme.DEFAULT_THEME))

        # Transport (section 5) before the docks: the Timeline dock wires
        # its playhead to transport.positionChanged at construction.
        self.transport = Transport(self)
        self.transport.positionChanged.connect(self._on_transport_position)
        self.transport.stateChanged.connect(self._on_transport_state)

        self._build_menu_bar()
        self._build_docks()
        self._build_shortcuts()

        self.workspaces = WorkspaceManager(self, self.settings)
        self.workspaces.restore_on_launch()
        self._sync_workspace_actions()

        self._arrangement = None
        self._rebuild_transport_schedule()
        self._update_window_title()

        self.set_status("Initializing engine...")
        init_lang_code = self.settings_dock.get_state().get("lang_code", "a")
        self.engine.worker.run_coro(self.engine.init_pipeline_async(init_lang_code, device=self.settings.get("device", "auto")))

    # --- project bootstrap ------------------------------------------------

    def _load_initial_document(self):
        candidates = []
        last = self.settings.get("last_project")
        if last:
            candidates.append(last)
        candidates.append(DOCUMENT_FILE)
        for path in candidates:
            try:
                loaded = project_io.load_project(path)
            except NotImplementedError:
                loaded = None
            if loaded is not None:
                self.project_path = os.path.abspath(path)
                self.project_settings = loaded.project_settings
                project_io.remember_recent(self.settings, self.project_path)
                return loaded.document
        # Nothing on disk: the classic first-run migration from presets.
        self.project_path = os.path.abspath(DOCUMENT_FILE)
        self.project_settings = {}
        project_io.remember_recent(self.settings, self.project_path)
        return document_state.load_or_create_document(DOCUMENT_FILE, self.settings, PRESETS_DIR)

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
        bar = self.menuBar()

        # File
        self.file_menu = bar.addMenu("&File")
        self.new_action = self._action("&New", self.new_project, QKeySequence.StandardKey.New)
        self.open_action = self._action("&Open...", self.open_project_dialog, QKeySequence.StandardKey.Open)
        self.recent_menu = self.file_menu.addMenu("Recent")
        self.welcome_action = self._action("&Welcome...", self.show_welcome)
        self.save_action = self._action("&Save", self.save_project, QKeySequence.StandardKey.Save)
        self.save_as_action = self._action("Save &As...", self.save_project_as_dialog, QKeySequence.StandardKey.SaveAs)
        self.file_menu.insertAction(self.recent_menu.menuAction(), self.new_action)
        self.file_menu.insertAction(self.recent_menu.menuAction(), self.open_action)
        self.file_menu.addAction(self.welcome_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.save_action)
        self.file_menu.addAction(self.save_as_action)
        self.file_menu.addSeparator()
        self.import_text_action = self._action("Import &Text...", self.import_text_dialog)
        self.file_menu.addAction(self.import_text_action)
        self.import_audio_action = QAction("Import Audio...", self)
        self.import_audio_action.setEnabled(False)
        self.import_audio_action.setToolTip("coming with ASR-anchored import")
        self.file_menu.addAction(self.import_audio_action)
        self.export_action = self._action("&Export...", self.export_dialog, "Ctrl+E")
        self.file_menu.addAction(self.export_action)
        self.file_menu.addSeparator()
        self.quit_action = self._action("&Quit", self.close, QKeySequence.StandardKey.Quit)
        self.file_menu.addAction(self.quit_action)
        self._rebuild_recent_menu()

        # Edit
        self.edit_menu = bar.addMenu("&Edit")
        self.undo_action = self._action("Undo", self.undo, QKeySequence.StandardKey.Undo)
        self.redo_action = self._action("Redo", self.redo, QKeySequence.StandardKey.Redo)
        self.edit_menu.addAction(self.undo_action)
        self.edit_menu.addAction(self.redo_action)
        self.edit_menu.addSeparator()
        self.cut_action = self._action("Cu&t", lambda: self._editor_call("cut"))
        self.copy_action = self._action("&Copy", lambda: self._editor_call("copy"))
        self.paste_action = self._action("&Paste", lambda: self._editor_call("paste"))
        for a in (self.cut_action, self.copy_action, self.paste_action):
            self.edit_menu.addAction(a)
        self.edit_menu.addSeparator()
        self.characters_action = self._action("&Characters...", self.open_characters_dialog)
        self.edit_menu.addAction(self.characters_action)

        # Options
        self.options_menu = bar.addMenu("&Options")
        self.engine_menu = self.options_menu.addMenu("Engine")
        self.engine_group = QActionGroup(self)
        self.engine_group.setExclusive(True)
        self.engine_actions: dict = {}
        for engine_id in engine_registry.list_engines():
            action = QAction(engine_registry.get_display_name(engine_id), self)
            action.setCheckable(True)
            action.setChecked(engine_id == self.backend.id)
            action.triggered.connect(lambda checked=False, eid=engine_id: self.on_engine_action(eid))
            self.engine_group.addAction(action)
            self.engine_menu.addAction(action)
            self.engine_actions[engine_id] = action

        self.device_menu = self.options_menu.addMenu("Device")
        self.device_group = QActionGroup(self)
        self.device_group.setExclusive(True)
        self.device_actions: dict = {}
        cuda_ok = self._cuda_available()
        for device_id, label in (("auto", "Auto"), ("cpu", "CPU"), ("cuda", "CUDA")):
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(self.settings.get("device", "auto") == device_id)
            if device_id == "cuda" and not cuda_ok:
                action.setEnabled(False)
                action.setToolTip("torch reports no CUDA device")
            action.triggered.connect(lambda checked=False, d=device_id: self.set_device(d))
            self.device_group.addAction(action)
            self.device_menu.addAction(action)
            self.device_actions[device_id] = action

        self.theme_menu = self.options_menu.addMenu("Theme")
        self.theme_group = QActionGroup(self)
        self.theme_group.setExclusive(True)
        self.theme_actions: dict = {}
        for theme_id, label in (("light", "Light"), ("dark", "Dark")):
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(self.settings.get("theme", theme.DEFAULT_THEME) == theme_id)
            action.triggered.connect(lambda checked=False, t=theme_id: self.set_theme(t))
            self.theme_group.addAction(action)
            self.theme_menu.addAction(action)
            self.theme_actions[theme_id] = action

        self.options_menu.addSeparator()
        self.copy_carries_action = QAction("Copy carries character/FX", self)
        self.copy_carries_action.setCheckable(True)
        self.copy_carries_action.setChecked(bool(self.settings.get("character_fx_copy", True)))
        self.copy_carries_action.toggled.connect(lambda v: self._set_setting("character_fx_copy", v))
        self.options_menu.addAction(self.copy_carries_action)

        self.paste_splits_action = QAction("Paste splits character/FX", self)
        self.paste_splits_action.setCheckable(True)
        self.paste_splits_action.setChecked(bool(self.settings.get("character_fx_paste_splits", True)))
        self.paste_splits_action.toggled.connect(lambda v: self._set_setting("character_fx_paste_splits", v))
        self.options_menu.addAction(self.paste_splits_action)

        self.jit_action = QAction("JIT streaming (no-clips fallback only)", self)
        self.jit_action.setCheckable(True)
        self.jit_action.setChecked(bool(self.jit_enabled))
        self.jit_action.toggled.connect(self._on_jit_toggled)
        self.options_menu.addAction(self.jit_action)
        self._sync_jit_action_enabled()

        # Workspace
        self.workspace_menu = bar.addMenu("&Workspace")
        self.workspace_group = QActionGroup(self)
        self.workspace_group.setExclusive(True)
        self.workspace_actions: dict = {}
        for name in (ADVANCED, SIMPLE):
            action = QAction(name, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, n=name: self.activate_workspace(n))
            self.workspace_group.addAction(action)
            self.workspace_menu.addAction(action)
            self.workspace_actions[name] = action
        self.workspace_menu.addSeparator()
        self.reset_layout_action = self._action("Reset layout", self.reset_workspace)
        self.workspace_menu.addAction(self.reset_layout_action)

    def _action(self, text: str, slot, shortcut=None) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut) if isinstance(shortcut, str) else shortcut)
        action.triggered.connect(slot)
        return action

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _build_docks(self) -> None:
        self.transcript_dock = TranscriptDock(self)
        self.settings_dock = SettingsDock(self)
        self.fx_dock = FXDock(self)
        self.lexicon_dock = LexiconDock(self)
        self.timeline_dock = TimelineDock(self)
        self.transport_dock = TransportDock(self)

        self.timeline_dock.batchGenerationProgress.connect(self.on_batch_generation_progress)
        self.timeline_dock.batchGenerationFinished.connect(self.on_batch_generation_finished)
        self.timeline_dock.timeline_view.seekRequested.connect(self.transport.seek)

        self.transport_dock.playRequested.connect(self.transport.play)
        self.transport_dock.pauseRequested.connect(self.transport.pause)
        self.transport_dock.stopRequested.connect(self.transport.stop)
        self.transport_dock.loopToggled.connect(self._on_loop_toggled)

        # A QMainWindow needs a central widget; the docks fill everything.
        # Hidden with an Ignored size policy, NOT setFixedSize(0, 0): a fixed
        # 0x0 central widget caps the maximum height of the row it sits in,
        # so the docks sharing that row (the timeline) could never be made
        # taller than their minimum by dragging the separator above them.
        central = QWidget()
        central.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        central.hide()
        self.setCentralWidget(central)
        self.setDockNestingEnabled(True)

        self.arrange_docks_default()

    def arrange_docks_default(self) -> None:
        """The drawing's 2x2 grid. Also what Workspace > Reset rebuilds.

        Top row (transcript | settings tabs) in the Top dock area, bottom row
        (timeline | transport) in the Left area next to the hidden central
        widget. Two areas, one separator between the rows, both rows
        resizable - the arrangement the maintainer settled on by dragging."""
        for dock in self._all_docks():
            self.removeDockWidget(dock)

        top = Qt.DockWidgetArea.TopDockWidgetArea
        bottom = Qt.DockWidgetArea.LeftDockWidgetArea
        self.addDockWidget(top, self.transcript_dock)
        self.addDockWidget(top, self.settings_dock)
        for dock in (self.fx_dock, self.lexicon_dock):
            self.addDockWidget(top, dock)
            self.tabifyDockWidget(self.settings_dock, dock)
        self._sync_mixing_dock()
        self._sync_voice_clone_dock()
        for voices_dock in (self.mixing_dock, self.voice_clone_dock):
            if voices_dock is not None:
                self._place_voices_dock(voices_dock)
        self.addDockWidget(bottom, self.timeline_dock)
        self.addDockWidget(bottom, self.transport_dock)
        self.splitDockWidget(self.timeline_dock, self.transport_dock, Qt.Orientation.Horizontal)

        for dock in self._all_docks():
            dock.setFloating(False)
            dock.setVisible(True)
        self.settings_dock.raise_()
        self.apply_default_proportions()

    def apply_default_proportions(self) -> None:
        """Left column ~65% of the width, top row ~65% of the height.
        `resizeDocks` only sticks once the dock layout is active, so this
        runs again from the first `showEvent`."""
        width = max(self.width(), 800)
        height = max(self.height(), 600)
        left_w, right_w = int(width * 0.65), int(width * 0.35)
        top_h, bottom_h = int(height * 0.65), int(height * 0.35)
        self.resizeDocks([self.transcript_dock, self.settings_dock], [left_w, right_w], Qt.Orientation.Horizontal)
        self.resizeDocks([self.timeline_dock, self.transport_dock], [left_w, right_w], Qt.Orientation.Horizontal)
        self.resizeDocks([self.transcript_dock, self.timeline_dock], [top_h, bottom_h], Qt.Orientation.Vertical)

    def apply_simple_proportions(self) -> None:
        """Workspace > Simple: the timeline is hidden, so the bottom row only
        needs the transport's three rows."""
        height = max(self.height(), 600)
        bottom_h = 140
        self.resizeDocks([self.transcript_dock, self.transport_dock], [height - bottom_h, bottom_h],
                         Qt.Orientation.Vertical)

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        if not getattr(self, "_shown_once", False):
            self._shown_once = True
            if hasattr(self, "workspaces") and self.workspaces.saved(self.workspaces.active) is None:
                QTimer.singleShot(0, lambda: self.workspaces.apply_default(self.workspaces.active))

    def _all_docks(self) -> list:
        docks = [self.transcript_dock, self.settings_dock, self.fx_dock, self.lexicon_dock,
                 self.mixing_dock, self.voice_clone_dock, self.timeline_dock, self.transport_dock]
        return [d for d in docks if d is not None]

    def _build_shortcuts(self) -> None:
        # UI12: plain Space toggles playback anywhere the focus widget
        # doesn't claim it (the editor and line edits accept it as text via
        # ShortcutOverride); Ctrl+Space always toggles.
        self.space_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self.space_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.space_shortcut.activated.connect(self.transport.toggle)
        self.ctrl_space_shortcut = QShortcut(QKeySequence("Ctrl+Space"), self)
        self.ctrl_space_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.ctrl_space_shortcut.activated.connect(self.transport.toggle)

    # --- status helpers ---------------------------------------------------

    def set_status(self, message: str, kind: str = "info") -> None:
        if self.transport_dock is not None:
            self.transport_dock.set_status(message, kind)

    def is_busy(self) -> bool:
        return self.transport_dock is not None and self.transport_dock.is_busy()

    @property
    def editor(self):
        return self.transcript_dock.editor if self.transcript_dock is not None else None

    def _editor_call(self, method: str) -> None:
        editor = self.editor
        if editor is not None:
            getattr(editor, method)()

    def raise_fx_tab(self) -> None:
        if self.fx_dock is not None:
            self.fx_dock.show()
            self.fx_dock.raise_()

    # --- voice listing --

    def get_all_voices(self, lang_code: str | None = None) -> list:
        """`spec.VOICE_DB` is Kokoro's built-in named-voice table; the
        backend-provided half comes from `self.backend.get_voices(...)`."""
        if lang_code is None:
            lang_code = self.settings.get("lang_code", "a")
        standard = spec.VOICE_DB.get(lang_code, [])
        custom = [v.id for v in self.backend.get_voices(lang_code)]
        return sorted(set(standard + custom))

    # --- settings persistence -

    def schedule_save(self) -> None:
        self._save_timer.start(1000)
        self._update_window_title(pending=True)

    def refresh_timeline(self) -> None:
        """App-owned cross-dock coordination point: re-render the timeline
        and (debounced) rebuild the transport's schedule."""
        if self.timeline_dock is not None:
            self.timeline_dock.refresh()
        self._schedule_timer.start()
        if self.transcript_dock is not None:
            self.transcript_dock.sync_header()

    def save_settings(self) -> None:
        self._save_timer.stop()

        if self.settings_dock is not None:
            gen_state = self.settings_dock.get_state()
            for key in ("lang_code", "voice", "speed", "volume", "pitch", "num_threads", "split_pattern",
                        "caching", "normalize", "format"):
                if key in gen_state:
                    self.settings[key] = gen_state[key]
            self.settings["trim"] = gen_state.get("trim_silence", self.settings.get("trim", False))
            self.settings["apply_fx"] = self.settings_dock.apply_fx_enabled()
            self.settings["jit_enabled"] = self.jit_enabled
            self.settings["engine_id"] = self.backend.id
            self.settings.update(self.fx_dock.project_fx_state())
            if self.voice_clone_dock is not None:
                self.settings.update(self.voice_clone_dock.get_state())
            if hasattr(self, "workspaces"):
                self.workspaces.capture()

        qt_settings.save_settings(CONFIG_FILE, self.settings)
        if self.project_path:
            try:
                project_io.save_project(self.document, self.project_path, self.project_settings)
            except Exception as e:  # noqa: BLE001 - autosave must never crash the UI
                self.set_status(f"Autosave failed: {e}", "error")
        self._update_window_title(pending=False)

    def _set_setting(self, key: str, value) -> None:
        self.settings[key] = value
        self.schedule_save()

    def _update_window_title(self, pending: bool = False) -> None:
        name = project_io.project_title(self.project_path)
        self.setWindowTitle(f"{name}{'*' if pending else ''} - {APP_NAME}")

    # --- config assembly ----------------

    def _export_values(self) -> dict:
        from kokoro_gui.qt.docks.export_dialog import export_defaults

        return export_defaults(self)

    def _assemble_config(self) -> dict:
        gen_state = self.settings_dock.get_state()
        export = self._export_values()
        config = {
            "engine_id": self.backend.id,
            "lang_code": gen_state["lang_code"],
            "voice": gen_state["voice"],
            "speed": gen_state["speed"],
            "split_pattern": gen_state["split_pattern"],
            # Sanitize the free-text filename field the same way voice/preset
            # names are sanitized elsewhere - it flows unvalidated into an
            # os.path.join sink in caching.py otherwise.
            "filename": os.path.basename(export["filename"]),
            "format": export["format"],
            "out_dir": export["out_dir"],
            "separate": export["keep_clip_files"],
            "combine": True,
            "export_subtitles": export["srt"],
            "caching": gen_state["caching"],
            "time_id": time.strftime(self.timecode_format),
            "num_threads": gen_state["num_threads"],
            "volume": gen_state["volume"],
            "pitch": gen_state["pitch"],
            "normalize": gen_state["normalize"],
            "trim_silence": gen_state["trim_silence"],
            "lexicon": self.settings.get("lexicon", {}),
        }
        if self.settings_dock.apply_fx_enabled():
            config.update(self.fx_dock.project_fx_state())
        return config

    def _assemble_clip_config(self, clip) -> dict:
        """The config dict for a per-clip Generate action and, through
        `post_config_for_clip`, for read-time post-processing - app-level
        defaults with `clip`'s character/override settings merged on top,
        so the clip's own values win. Defaults must come first:
        `process_chunk_task` reads `config['voice']`/`config['split_pattern']`
        by direct indexing, so a clip with no character must still end up
        with usable defaults. FX come from `fx_resolve.resolve_fx`, the same
        resolver the Audio FX tab renders."""
        gen_state = self.settings_dock.get_state()
        export = self._export_values()
        config = {
            "engine_id": self.backend.id,
            "lang_code": gen_state["lang_code"],
            "voice": gen_state["voice"],
            "speed": gen_state["speed"],
            "split_pattern": gen_state["split_pattern"],
            "format": export["format"],
            "out_dir": export["out_dir"],
            "caching": gen_state["caching"],
            "time_id": time.strftime(self.timecode_format),
            "num_threads": gen_state["num_threads"],
            "volume": gen_state["volume"],
            "pitch": gen_state["pitch"],
            "normalize": gen_state["normalize"],
            "trim_silence": gen_state["trim_silence"],
            "filename": os.path.basename(export["filename"]),
        }

        clip_config = dict(self.document.effective_config_for_clip(clip))
        # ALLOWED_PRESET_KEYS whitelists "trim", but process_audio reads
        # "trim_silence" - same inline rename every other caller does.
        if "trim" in clip_config:
            clip_config["trim_silence"] = clip_config.pop("trim")
        config.update(clip_config)

        # effective_config_for_clip only ever carries the FX preset's *name*;
        # the resolver turns project values + character preset + clip preset
        # + clip.fx_override into the actual FX keys and the ANDed apply_fx.
        resolution = fx_resolve.resolve_fx(self, clip=clip)
        config.update(resolution.values)
        config["apply_fx"] = resolution.apply_fx
        return config

    # --- read-time post-processing (kokoro_gui/audio/post.py) ---------------

    def post_config_for_clip(self, clip) -> dict:
        """The `POST_KEYS` subset of the clip's resolved config: what the
        transport, the exporter and the timeline waveform apply on top of
        the raw segment files. Changing any of it never dirties the clip."""
        return post.extract_post_config(self._assemble_clip_config(clip))

    def clip_duration_s(self, clip):
        """`compute_arrangement`'s `clip_duration`: the clip's rendered
        length (trim and pitch change it), or the raw `Segment.duration`
        for a file that can't be read, or None with no audio at all. Falls
        back to the raw durations while the docks are still being built."""
        segments = [s for s in clip.segments if s.audio_path]
        if not segments:
            return None
        if self.settings_dock is None or self.fx_dock is None:
            return clip_audio_duration_s(clip)
        post_config = self.post_config_for_clip(clip)
        rate = self.project_sample_rate()
        total = 0.0
        for segment in segments:
            try:
                total += post.rendered_duration_s(segment.audio_path, post_config, rate)
            except Exception:
                total += segment.duration or 0.0
        return total

    def rendered_clip_samples(self, clip):
        """`(samples, rate)` for the clip's segments concatenated and
        post-processed, or None. The timeline draws its waveform from this
        so it shows what the transport plays."""
        segments = sorted((s for s in clip.segments if s.audio_path), key=lambda s: s.order_index)
        if not segments:
            return None
        post_config = self.post_config_for_clip(clip)
        rate = self.project_sample_rate()
        parts = []
        for segment in segments:
            try:
                parts.append(post.render(segment.audio_path, post_config, rate))
            except Exception:
                continue
        if not parts:
            return None
        import numpy as np

        return np.concatenate(parts), rate

    def build_arrangement(self):
        """Every `compute_arrangement` call for the live document goes
        through here so they all measure clips the same way."""
        return compute_arrangement(self.document, engine_id=self.backend.id, clip_duration=self.clip_duration_s)

    # --- Options: engine / device / theme ---------------------------------

    def on_engine_action(self, engine_id: str) -> None:
        if engine_id == self.backend.id:
            return
        self.switch_engine(engine_id)

    def switch_engine(self, engine_id: str) -> None:
        if self.is_busy():
            QMessageBox.warning(self, "Busy", "Cancel the current job before switching engines.")
            self._sync_engine_actions()
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

        self.settings_dock.rebuild_schema_form()
        self._sync_mixing_dock()
        self._sync_voice_clone_dock()
        self._sync_engine_actions()
        self._sync_jit_action_enabled()

        try:
            old_engine.worker.stop()
        except Exception:
            pass

        self.set_status(f"Switched engine to {new_backend.display_name}. Initializing...")
        new_lang_code = self.settings_dock.get_state().get("lang_code", "a")
        self.settings["lang_code"] = new_lang_code
        self.engine.worker.run_coro(self.engine.init_pipeline_async(new_lang_code, device=self.settings.get("device", "auto")))
        self.schedule_save()

    def _sync_engine_actions(self) -> None:
        for engine_id, action in self.engine_actions.items():
            action.setChecked(engine_id == self.backend.id)

    def _sync_jit_action_enabled(self) -> None:
        supported = self.backend.capabilities.supports_jit_streaming
        self.jit_action.setEnabled(supported)
        self.jit_action.setToolTip("" if supported else f"{self.backend.display_name} doesn't support streaming - runs as Standard.")

    def _on_jit_toggled(self, checked: bool) -> None:
        self.jit_enabled = checked
        self.settings["jit_enabled"] = checked
        self.schedule_save()

    def set_device(self, device: str) -> None:
        self.settings["device"] = device
        for device_id, action in self.device_actions.items():
            action.setChecked(device_id == device)
        self.schedule_save()
        if self.is_busy():
            return
        lang_code = self.settings_dock.get_state().get("lang_code", "a")
        self.set_status(f"Re-initializing engine on {device}...")
        self.engine.worker.run_coro(self.engine.init_pipeline_async(lang_code, device=device))

    def set_theme(self, name: str) -> None:
        self.settings["theme"] = name
        theme.apply(QApplication.instance(), name)
        for theme_id, action in self.theme_actions.items():
            action.setChecked(theme_id == name)
        self.themeChanged.emit()
        self.schedule_save()

    def _sync_mixing_dock(self) -> None:
        wants = self.backend.capabilities.supports_voice_mixing
        if wants and self.mixing_dock is None:
            self.mixing_dock = MixingDock(self)
            self._place_voices_dock(self.mixing_dock)
        elif not wants and self.mixing_dock is not None:
            self.removeDockWidget(self.mixing_dock)
            self.mixing_dock.deleteLater()
            self.mixing_dock = None
        elif wants and self.mixing_dock is not None and self.mixing_dock.parent() is None:
            self._place_voices_dock(self.mixing_dock)

    def _sync_voice_clone_dock(self) -> None:
        wants = self.backend.capabilities.supports_voice_cloning
        if wants and self.voice_clone_dock is None:
            self.voice_clone_dock = VoiceCloneDock(self)
            self._place_voices_dock(self.voice_clone_dock)
        elif not wants and self.voice_clone_dock is not None:
            self.removeDockWidget(self.voice_clone_dock)
            self.voice_clone_dock.deleteLater()
            self.voice_clone_dock = None
        elif wants and self.voice_clone_dock is not None and self.voice_clone_dock.parent() is None:
            self._place_voices_dock(self.voice_clone_dock)

    def _place_voices_dock(self, dock) -> None:
        """Both capability-gated voice docks share the "Voices" tab title and
        objectName, so the tab strip doesn't jump when the engine changes
        and a saved layout places either one in the same slot."""
        dock.setWindowTitle("Voices")
        dock.setObjectName("dock_voices")
        self.addDockWidget(self.dockWidgetArea(self.settings_dock), dock)
        self.tabifyDockWidget(self.lexicon_dock, dock)
        if self.settings_dock is not None:
            self.settings_dock.raise_()

    # --- Workspace ----------------------------------------------------------

    def activate_workspace(self, name: str) -> None:
        self.workspaces.activate(name)
        self._sync_workspace_actions()
        self.schedule_save()

    def reset_workspace(self) -> None:
        self.workspaces.reset()
        self._sync_workspace_actions()
        self.schedule_save()

    def _sync_workspace_actions(self) -> None:
        for name, action in self.workspace_actions.items():
            action.setChecked(name == self.workspaces.active)

    # --- File menu ----------------------------------------------------------

    def _rebuild_recent_menu(self) -> None:
        self.recent_menu.clear()
        recent = [p for p in self.settings.get("recent_projects", []) if isinstance(p, str)]
        if not recent:
            empty = self.recent_menu.addAction("(empty)")
            empty.setEnabled(False)
            return
        for path in recent:
            action = self.recent_menu.addAction(project_io.project_title(path))
            action.setToolTip(path)
            action.triggered.connect(lambda checked=False, p=path: self.open_project(p))

    def show_welcome(self) -> WelcomeDialog:
        """Window-modal via `open()`, not `exec()`, so engine init keeps
        reporting underneath and tests can drive it."""
        if self.welcome_dialog is None:
            self.welcome_dialog = WelcomeDialog(self)
        else:
            self.welcome_dialog.reload()
        if self.welcome_dialog.isVisible():
            self.welcome_dialog.raise_()
        else:
            self.welcome_dialog.open()
        return self.welcome_dialog

    def show_welcome_if_enabled(self) -> WelcomeDialog | None:
        """The launch-time trigger; `main.py` is its only caller, so the
        test fixture and the screenshot script never get a dialog."""
        if not self.settings.get("show_welcome", True):
            return None
        return self.show_welcome()

    def _switch_document(self, document, path: str | None, project_settings: dict | None = None) -> None:
        self.transport.stop()
        self.document = document
        self.project_path = os.path.abspath(path) if path else None
        self.project_settings = dict(project_settings or {})
        self.selection.clear()
        self.selection.set_playing_clip(None)
        if self.project_path:
            project_io.remember_recent(self.settings, self.project_path)
        else:
            self.settings["last_project"] = None
        self._rebuild_recent_menu()
        if self.editor is not None:
            self.editor.rebind_document()
        if self.transcript_dock is not None:
            self.transcript_dock.refresh_character_choices()
        if self.settings_dock is not None:
            self.settings_dock.rebuild_schema_form()
        if self.fx_dock is not None:
            self.fx_dock.refresh_for_selection()
        self.refresh_timeline()
        self._update_window_title()
        self.schedule_save()

    def new_project(self) -> None:
        document = project_io.new_document_from(self.document)
        self._switch_document(document, None)
        self.set_status("New project (characters inherited from the previous one). Save As to name it.")

    def open_project(self, path: str) -> None:
        try:
            loaded = project_io.load_project(path)
        except NotImplementedError as e:
            QMessageBox.information(self, "Not yet", str(e))
            return
        if loaded is None:
            QMessageBox.warning(self, "Open failed", f"Couldn't read {path}.")
            project_io.forget_recent(self.settings, path)
            self._rebuild_recent_menu()
            return
        self._switch_document(loaded.document, path, loaded.project_settings)
        self.set_status(f"Opened {project_io.project_title(path)}.")

    def open_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", project_io.PROJECT_FILTER)
        if path:
            self.open_project(path)

    def save_project(self) -> None:
        if not self.project_path:
            self.save_project_as_dialog()
            return
        self.save_settings()
        self.set_status(f"Saved {project_io.project_title(self.project_path)}.")

    def save_project_as(self, path: str) -> None:
        if not path.lower().endswith((".json", ".tbaw")):
            path += project_io.DEFAULT_EXTENSION
        self.project_path = os.path.abspath(path)
        project_io.remember_recent(self.settings, self.project_path)
        self._rebuild_recent_menu()
        self.save_settings()
        self.set_status(f"Saved as {project_io.project_title(self.project_path)}.")

    def save_project_as_dialog(self) -> None:
        start = self.project_path or ""
        path, _ = QFileDialog.getSaveFileName(self, "Save project as", start, project_io.PROJECT_FILTER)
        if path:
            self.save_project_as(path)

    def import_text_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import text", filter="Documents (*.txt *.pdf *.epub)")
        if path:
            self.import_text(path)

    def import_text(self, path: str, target: str | None = None) -> None:
        """WF10: prompts "Add to current project" / "New project" unless
        `target` ("add" | "new") is given."""
        try:
            text = self.engine.extract_text_from_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Import failed", f"Read failed: {e}")
            return
        if not text:
            QMessageBox.warning(self, "Empty", "No text found in that file.")
            return
        if target is None:
            box = QMessageBox(self)
            box.setWindowTitle("Import text")
            box.setText("Add the text to the current project, or start a new project from it?")
            add_btn = box.addButton("Add to current project", QMessageBox.ButtonRole.AcceptRole)
            new_btn = box.addButton("New project", QMessageBox.ButtonRole.ActionRole)
            box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            clicked = box.clickedButton()
            if clicked is add_btn:
                target = "add"
            elif clicked is new_btn:
                target = "new"
            else:
                return
        if target == "new":
            self.new_project()
        editor = self.editor
        cursor = editor.textCursor()
        # A real editor insert: goes through contentsChange -> replace_text
        # and lands on the native undo stack like a paste would. The edit
        # block keeps Qt from coalescing it with whatever was typed just
        # before, so one Ctrl+Z removes exactly the import.
        cursor.beginEditBlock()
        cursor.insertText(text)
        cursor.endEditBlock()
        editor.setTextCursor(cursor)
        self.set_status(f"Imported {os.path.basename(path)}.")

    def export_dialog(self) -> None:
        dialog = ExportDialog(self)
        if dialog.exec() != ExportDialog.DialogCode.Accepted:
            return
        run_export(self, dialog.values(), parent=self)

    def _on_export_progress(self, percent: float, detail: str) -> None:
        self.transport_dock.set_progress(percent, detail)

    def _on_export_finished(self, success: bool, message: str) -> None:
        self.transport_dock.set_busy(False)
        self.transport_dock.set_progress_value(100 if success else 0)
        self.set_status(message, "success" if success else "error")

    def project_sample_rate(self) -> int:
        """The mix rate for transport and export: the active engine's
        output rate (44.1k for Audio8, 24k otherwise). Clips rendered at
        another rate are resampled once on load."""
        return int(getattr(self.engine, "SAMPLE_RATE", 24000) or 24000)

    # --- Edit menu ------------------------------------------------------------

    def open_characters_dialog(self) -> None:
        dialog = CharactersDialog(self)
        dialog.exec()

    def on_characters_changed(self) -> None:
        if self.editor is not None:
            self.editor.rehighlight()
        if self.transcript_dock is not None:
            self.transcript_dock.refresh_character_choices()
        self.refresh_timeline()
        self.schedule_save()

    # --- engine callbacks (queued automatically across threads - see signals.py) -

    def on_engine_status(self, msg: str, is_error: bool) -> None:
        self.set_status(msg, "error" if is_error else "info")
        if is_error and "pip install" in msg:
            QMessageBox.critical(self, "Missing Dependencies", msg)

    def on_engine_progress(self, percent: float, elapsed: float, eta: str, detail: str) -> None:
        self.transport_dock.set_progress(percent, detail, elapsed=elapsed, eta=eta)

    def on_engine_finish(self) -> None:
        self.set_ui_state(False)
        self._rebuild_transport_schedule()

    def set_ui_state(self, is_running: bool) -> None:
        self.transport_dock.set_busy(is_running)
        threads_widget = self.settings_dock.schema_form.widget_for("num_threads")
        if threads_widget is not None:
            threads_widget.setEnabled(not is_running)
        self.settings_dock.volume_spin.setEnabled(not is_running)
        self.settings_dock.pitch_spin.setEnabled(not is_running)
        if not is_running:
            self.transport_dock.set_progress_value(0 if self.engine.cancel_event.is_set() else 100)

    # --- preview -----------------------------------

    def preview_conversion(self) -> None:
        if not self.engine.pipeline:
            QMessageBox.information(self, "Wait", "Engine is initializing... please wait 2 seconds and try again.")
            return

        editor = self.editor
        cursor = editor.textCursor()
        text_data = cursor.selectedText().replace(" ", "\n") if cursor.hasSelection() else editor.toPlainText().strip()
        if not text_data:
            text_data = ("This is a sample audio preview using the Koh-koh-ro Tea-Tea-S engine. "
                         "It demonstrates the voice quality and speed settings.")
        preview_text = text_data[:1000]

        state = self.settings_dock.get_state()
        extra_config = {
            "volume": state["volume"],
            "pitch": state["pitch"],
            "normalize": state["normalize"],
            "trim_silence": state["trim_silence"],
            "lexicon": self.settings.get("lexicon", {}),
        }
        if self.settings_dock.apply_fx_enabled():
            extra_config.update(self.fx_dock.project_fx_state())

        tmp_path = os.path.join(tempfile.gettempdir(), "kokoro_preview.wav")
        self.set_status("Generating preview...", "busy")

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
            self.set_status("Playing preview...", "success")
            playback.play(payload)
            QTimer.singleShot(3000, lambda: self.set_status("Ready"))
        else:
            self.set_status(payload, "error")

    # --- start/cancel ------------------------------

    def on_generate_clicked(self) -> None:
        """The Transport dock's Generate button. A document with no clips
        yet falls back to whole-document `start_conversion()`; a document
        with clips dispatches the dirty-scoped batch path."""
        if not self.document.clips:
            self.start_conversion()
            return

        dirty = self.document.dirty_clips()
        if not dirty:
            QMessageBox.information(self, "Up to date", "All clips are already generated.")
            return

        # Once a document has any clips, Generate always runs the
        # dirty-scoped batch path - even with JIT enabled (JIT has no
        # per-clip output shape; it stays reachable via the no-clips
        # fallback above).
        self.timeline_dock.generate_dirty_clips_requested()

    def generate_clip(self, clip_id: str) -> None:
        """UI3: the gutter's per-clip play button and the timeline's
        context menu both land here."""
        self.timeline_dock.on_generate_clip_requested(clip_id)

    def on_batch_generation_progress(self, completed: int, total: int, current_clip_label: str) -> None:
        percent = int((completed / total) * 100) if total else 0
        if current_clip_label:
            detail = f"Generated {completed}/{total} clips"
        else:
            detail = f"Generating {total} clip(s)..."
        self.transport_dock.set_progress(percent, detail)

    def on_batch_generation_finished(self, succeeded: int, failed: int, failed_clip_ids: list) -> None:
        total = succeeded + failed
        if failed == 0:
            self.set_status(f"Generated {succeeded} clip(s).", "success")
        elif succeeded == 0:
            self.set_status(f"Batch generation failed for all {failed} clip(s).", "error")
        else:
            self.set_status(f"Generated {succeeded} of {total} clips ({failed} failed)", "warning")
        self._rebuild_transport_schedule()

    def auto_split_and_generate(self) -> None:
        """Generate menu > "Auto-split then generate": turns every
        `[Speaker:FX]:`-tagged span (and, with "Split by paragraph" on,
        each span's paragraphs) into clips, then batch-generates them."""
        if self.is_busy():
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

        for start, end, character_id in triples:
            self.document.undo_stack.push(AssignCharacterCommand(start, end, character_id))

        self.editor.rehighlight()
        self.schedule_save()
        self.refresh_timeline()

        self.timeline_dock.generate_dirty_clips_requested()

    def start_conversion(self) -> None:
        text_data = self.editor.toPlainText().strip()
        if not text_data:
            QMessageBox.warning(self, "Empty", "No text to process.")
            return

        if not self.engine.pipeline:
            QMessageBox.information(self, "Wait", "Engine is initializing... please wait 2 seconds and try again.")
            return

        config = self._assemble_config()

        self.set_ui_state(True)
        self.transport_dock.set_progress(0, "")

        if self.jit_enabled and self.backend.capabilities.supports_jit_streaming:
            self.engine.start_jit_conversion(text_data, config)
        else:
            self.engine.start_conversion(text_data, config)

    def cancel_conversion(self) -> None:
        self.engine.cancel()
        self.set_status("Cancelling... waiting for workers...", "warning")

    # --- transport / playhead (section 5) ----------------------------------

    def current_arrangement(self):
        if self._arrangement is None:
            self._arrangement = self.build_arrangement()
        return self._arrangement

    def _rebuild_transport_schedule(self) -> None:
        self._arrangement = self.build_arrangement()
        rate = self.project_sample_rate()
        schedule = []
        for placed in self._arrangement.placed:
            if placed.estimated:
                continue
            post_config = self.post_config_for_clip(placed.clip)
            # One ScheduledClip per segment so multi-segment clips play
            # back to back at their real (rendered) offsets.
            offset = placed.start_s
            for segment in sorted(placed.clip.segments, key=lambda s: s.order_index):
                if not segment.audio_path:
                    continue
                schedule.append(ScheduledClip(clip_id=placed.clip.id, start_s=offset, path=segment.audio_path,
                                              post_config=post_config))
                try:
                    offset += post.rendered_duration_s(segment.audio_path, post_config, rate)
                except Exception:
                    offset += segment.duration or 0.0
        self.transport.load(schedule, sample_rate=self.project_sample_rate(),
                            total_duration_s=self._arrangement.total_duration_s)
        if self.timeline_dock is not None:
            self.timeline_dock.timeline_view.set_arrangement(self._arrangement)
        self.transport_dock.set_position(self.transport.position(), self.transport.duration())

    def _on_transport_position(self, seconds: float) -> None:
        self.transport_dock.set_position(seconds, self.transport.duration())
        if self.timeline_dock is not None:
            self.timeline_dock.timeline_view.set_playhead(seconds)
        arrangement = self.current_arrangement()
        playing = None
        if self.transport.is_playing:
            hits = arrangement.at_time(seconds)
            playing = hits[0].clip.id if hits else None
        self.selection.set_playing_clip(playing)

    def _on_transport_state(self, state: str) -> None:
        self.transport_dock.set_playing(state == "playing")
        if state != "playing":
            self.selection.set_playing_clip(None)

    def _on_loop_toggled(self, checked: bool) -> None:
        self.transport.loop = checked

    # --- undo/redo -----------------------------------------------------------

    def undo(self) -> None:
        self.editor.undo_coordinator.undo()

    def redo(self) -> None:
        self.editor.undo_coordinator.redo()

    # --- lifecycle -----------------------------------------------------------

    def closeEvent(self, event) -> None:
        try:
            self.transport.stop()
        except Exception:
            pass
        self.save_settings()
        super().closeEvent(event)
