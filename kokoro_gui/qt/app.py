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

Projects are `.tbaw` bundles (Claude/old/PLAN_tbaw_bundle.md). The live project
is a directory under `cache/projects/<project_id>/` (`self.project_dir`)
that autosave writes JSON into and clips generate straight into; Save
rewrites the zip from it on a background thread behind `is_busy`, Open
extracts into it the same way with the editor read-only, and Close asks
Save / Discard / Cancel when the dir is ahead of the file. `project.py`
holds the steps; this class holds the sequencing, the lock and the dirty
flag.

`CONFIG_FILE`/`PRESETS_DIR`/`FX_PRESETS_DIR`/`DOCUMENT_FILE` are defined
here, at module level, before the `kokoro_gui.qt.docks` import below - the
dock modules do `import kokoro_gui.qt.app as qt_app_module` and read
`qt_app_module.PRESETS_DIR` etc. qualified at call time, which makes this a
circular import; defining these names before triggering that import keeps
it safe.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time

import playback
from PySide6.QtCore import QFileSystemWatcher, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox, QSizePolicy, QWidget

from kokoro_engine import KokoroEngine
from kokoro_gui.daw import library as character_library, wordalign
from kokoro_gui.daw.migration import import_presets_to_library, link_exact_matches
from kokoro_gui.daw import subtitles
from kokoro_gui.daw.models import DEFAULT_HIGHLIGHT_PALETTE, Character, Document
from kokoro_gui.daw.arrangement import compute_arrangement, segment_timeline
from kokoro_gui.daw.mixplan import clip_mixes
from kokoro_gui.daw.auto_split import plan_auto_split_clips, plan_pause_gaps
from kokoro_gui.daw.beds import playable_segments
from kokoro_gui.daw.mixdown import duck_db_setting
from kokoro_gui.daw.undo import AssignCharacterCommand, ImportBedCommand, ImportCuesCommand
from kokoro_gui.engine import caching
from kokoro_gui.engines import registry as engine_registry
from kokoro_gui.qt import document_state, fx_resolve, project as project_io, spec, theme
from kokoro_gui.qt import settings as qt_settings
from kokoro_gui.qt.open_projects import OpenProject
from kokoro_gui.qt.subprojects import ParentStore, SubprojectsMixin
from kokoro_gui.qt.selection import SelectionModel
from kokoro_gui.qt.speaker_mapping_dialog import (
    NARRATOR, NO_SPEAKER, SpeakerMappingDialog, resolve_mapping, speaker_rows,
)
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
    VideoDock, VoiceCloneDock,
)
from kokoro_gui.qt.docks.export_dialog import ExportDialog, run_export  # noqa: E402
from kokoro_gui.qt.welcome_dialog import WelcomeDialog  # noqa: E402

APP_NAME = "KokoroGUI"
SCHEDULE_REBUILD_DEBOUNCE_MS = 100
LIBRARY_WATCH_DEBOUNCE_MS = 150


class QtTTSApp(SubprojectsMixin, QMainWindow):
    previewFinished = Signal(bool, str)
    themeChanged = Signal()
    exportProgress = Signal(float, str)
    exportFinished = Signal(bool, str)
    # Background project I/O (Open's audio extraction, Save's zip write):
    # progress as (percent, detail), completion as (callback, result, error)
    # marshalled onto the GUI thread.
    projectIoProgress = Signal(float, str)
    _projectIoFinished = Signal(object)
    # Word alignment on a worker thread (phase 2, C1): progress as
    # (done, total), the result as [(clip_id, segment_id, words)].
    _wordAlignProgress = Signal(int, int)
    _wordsAligned = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(1600, 1000)

        os.makedirs(PRESETS_DIR, exist_ok=True)
        os.makedirs(FX_PRESETS_DIR, exist_ok=True)

        self.settings = qt_settings.load_settings(CONFIG_FILE)
        # The global character library (phase 3, grill WF4-WF7/WF12).
        # `library_missing` holds the document character ids whose entry
        # the last resolve didn't find (the Characters dialog's "not found
        # here"). `_library_link_ids` is set by the one-time presets import
        # and consumed by the next `_switch_document`.
        self.character_library = character_library.CharacterLibrary()
        os.makedirs(self.character_library.root, exist_ok=True)
        self.library_missing: set = set()
        self._library_link_ids: list | None = None
        self._library_mtime = 0
        self._import_presets_once()
        self.jit_enabled = self.settings.get("jit_enabled", False)
        self.timecode_format = "%Y%m%d%H%M%S"

        # Projects (phase 4): the root the File menu opened (the last-opened
        # project, else the classic document.json next to the config, else a
        # fresh document) and any of its subprojects open beside it
        # (kokoro_gui/qt/open_projects.py). Each has a live project directory
        # (Claude/old/PLAN_tbaw_bundle.md section 3), `cache/projects/<id>/`,
        # held under an OS lock while open; every config the segment key or
        # the engine sees carries it as `project_dir`. `document`,
        # `project_dir` and `project_settings` read the `focus` project;
        # `project_path`, `project_id`, the lock, the manifest and the
        # dirty flag ("dir is ahead of the file", from autosave's digest,
        # cleared by Save) read the root.
        self.root = OpenProject(document=Document(runs=[], clips=[], tracks=[], characters=[], settings={}))
        self.children: dict = {}
        self.focus = self.root
        self.level = self.root
        # Child project ids whose bundle couldn't be opened (painted as
        # missing, with Relink).
        self._missing_children: set = set()
        # child id -> "ok"/"stale" for subprojects that aren't open.
        self._closed_child_states: dict = {}
        # Stale subprojects waiting to be generated and rendered, one at a
        # time behind is_busy: [(nested clip, then)], and the render step a
        # child's batch generate hands back to.
        self._subproject_queue: list = []
        self._pending_render_after_generate: dict = {}
        self._nested_after_batch = None
        self._io_thread: threading.Thread | None = None
        self._pending_open_path: str | None = None
        self._closing_after_save = False
        self._closed = False
        # Every engine a character uses stays resident (grill V3, "engine
        # follows the character"): engine id -> adapter, and its signal
        # bridge. Built on first use by `_backend_for`; the pipeline loads
        # once per backend (`_ensure_backend_ready`).
        self._backends: dict = {}
        self._bridges: dict = {}
        self._ready_backends: set = set()
        self._primary_engine_id = "kokoro"
        self._last_active_engine_id = None
        self.projectIoProgress.connect(self._on_project_io_progress)
        self._projectIoFinished.connect(self._on_project_io_finished)
        self._wordAlignProgress.connect(self._on_word_align_progress)
        self._wordsAligned.connect(self._on_words_aligned)
        self._word_align_thread: threading.Thread | None = None
        # Set when the user turns down the Whisper download for alignment,
        # so the next Generate doesn't ask again this session.
        self._word_align_declined = False
        self.document = self._load_initial_document()

        self.selection = SelectionModel()

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self.save_settings)

        self._schedule_timer = QTimer(self)
        self._schedule_timer.setSingleShot(True)
        self._schedule_timer.setInterval(SCHEDULE_REBUILD_DEBOUNCE_MS)
        self._schedule_timer.timeout.connect(self._rebuild_transport_schedule)

        # Another window (or process) editing the library reaches this one:
        # a directory change re-resolves the open document, debounced so
        # one save's .tmp + os.replace is one pass.
        self._library_timer = QTimer(self)
        self._library_timer.setSingleShot(True)
        self._library_timer.setInterval(LIBRARY_WATCH_DEBOUNCE_MS)
        self._library_timer.timeout.connect(self._on_library_changed_on_disk)
        self._library_watcher = QFileSystemWatcher(self)
        self._library_watcher.addPath(os.path.abspath(self.character_library.root))
        self._library_watcher.directoryChanged.connect(lambda _path: self._library_timer.start())

        self.welcome_dialog: WelcomeDialog | None = None
        self.transcript_dock: TranscriptDock | None = None
        self.settings_dock: SettingsDock | None = None
        self.fx_dock: FXDock | None = None
        self.lexicon_dock: LexiconDock | None = None
        self.mixing_dock: MixingDock | None = None
        self.voice_clone_dock: VoiceCloneDock | None = None
        self.timeline_dock: TimelineDock | None = None
        self.transport_dock: TransportDock | None = None
        self.video_dock: VideoDock | None = None

        # --- Engines: one resident backend per engine id in use ---
        self._add_backend(engine_registry.get_engine("kokoro", engine=KokoroEngine()))
        self._install_segment_key_fn()
        self._install_nested_state_fn(self.root)

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
        # Focus first: a nested block's child becomes the docks' document
        # before the active engine is read.
        self.selection.changed.connect(self._on_selection_for_focus)
        self.selection.changed.connect(self._on_active_backend_maybe_changed)
        self._last_active_engine_id = self.backend.id
        for engine_id in self._document_engine_ids(self.document):
            self._ensure_backend_ready(self._backend_for(engine_id))

        for backend in list(self._backends.values()):
            backend.on_project_opened(self.project_dir, {})
        if self._pending_open_path:
            path, self._pending_open_path = self._pending_open_path, None
            self.open_project(path)

    # --- the open projects (phase 4) ---------------------------------------

    @property
    def document(self):
        return self.focus.document

    @document.setter
    def document(self, value) -> None:
        self.focus.document = value

    @property
    def project_dir(self):
        return self.focus.project_dir

    @project_dir.setter
    def project_dir(self, value) -> None:
        self.focus.project_dir = value

    @property
    def project_settings(self) -> dict:
        return self.focus.project_settings

    @project_settings.setter
    def project_settings(self, value) -> None:
        self.focus.project_settings = value

    @property
    def project_path(self):
        return self.root.path

    @project_path.setter
    def project_path(self, value) -> None:
        self.root.path = value

    @property
    def project_id(self):
        return self.root.project_id

    @project_id.setter
    def project_id(self, value) -> None:
        self.root.project_id = value

    @property
    def _project_lock(self):
        return self.root.lock

    @_project_lock.setter
    def _project_lock(self, value) -> None:
        self.root.lock = value

    @property
    def _project_manifest(self) -> dict:
        return self.root.manifest

    @_project_manifest.setter
    def _project_manifest(self, value) -> None:
        self.root.manifest = value

    @property
    def _project_dirty(self) -> bool:
        return self.root.dirty

    @_project_dirty.setter
    def _project_dirty(self, value) -> None:
        self.root.dirty = value

    @property
    def projects(self) -> dict:
        """Every open project by id: the root and its open subprojects."""
        out = {self.root.project_id: self.root}
        out.update(self.children)
        return out

    def project_of_clip_id(self, clip_id):
        """The open project whose document has a clip with `clip_id`, or
        None. Focus, level and root first."""
        seen = []
        for project in (self.focus, self.level, self.root, *self.children.values()):
            if project in seen:
                continue
            seen.append(project)
            if project.document.get_clip(clip_id) is not None:
                return project
        return None

    def project_for(self, clip):
        """The open project whose document holds `clip`: focus, level and
        root first (almost always one of them), then the other children."""
        if clip is None:
            return self.focus
        seen = []
        for project in (self.focus, self.level, self.root, *self.children.values()):
            if project in seen:
                continue
            seen.append(project)
            if any(c is clip for c in project.document.clips):
                return project
        return self.focus

    # --- project bootstrap ------------------------------------------------

    def _load_initial_document(self):
        """The window starts on an Untitled project in a fresh project dir.
        The last project (or the 4.0-preview `document.json` next to the
        config, which migrates to `document.tbaw`) is opened right after
        the docks exist, since Open extracts on a thread with progress on
        the transport bar. Characters for a first run come from the
        character library."""
        candidates = []
        last = self.settings.get("last_project")
        if last and os.path.isfile(last):
            candidates.append(last)
        if os.path.isfile(DOCUMENT_FILE):
            candidates.append(DOCUMENT_FILE)
        self._pending_open_path = candidates[0] if candidates else None
        self.project_path = None
        self.project_settings = {}
        if self._pending_open_path:
            # A placeholder until the Open lands.
            document = Document(runs=[], clips=[], tracks=[], characters=[], settings={})
        else:
            document = document_state.load_or_create_document(DOCUMENT_FILE, self.settings, self.character_library)
            self._resolve_library_into(document)
        self._begin_untitled_project_dir(document)
        return document

    def _import_presets_once(self) -> None:
        """Phase 3 migration: the first launch with the library copies every
        `presets/*.json` into it (the files stay) and remembers the ids so
        the project opened next can link its characters that match one
        exactly. `settings["library_imported"]` makes it one-time."""
        if self.settings.get("library_imported"):
            return
        try:
            self._library_link_ids = import_presets_to_library(PRESETS_DIR, self.character_library)
        except OSError:
            # The flag stays unset, so the next launch tries again.
            self._library_link_ids = None
            return
        self.settings["library_imported"] = True

    # --- character library ------------------------------------------------

    def _resolve_library_into(self, document) -> character_library.ResolveReport:
        report = character_library.resolve_characters(document, [self.character_library])
        # A project-scope character in the root is its own store (NP3).
        report.missing = [cid for cid in report.missing
                          if not character_library.is_project_scope_id(
                              getattr(document.get_character(cid), "library_id", None))]
        self.library_missing = set(report.missing)
        self._library_mtime = self.character_library.mtime()
        return report

    def resolve_library(self, refresh: bool = True) -> character_library.ResolveReport:
        """Re-reads every linked character of every open project (the live
        link, WF5): the root from the library, each subproject from the
        root's characters first, then the library (NP3). With `refresh`, a
        change reaches the editor, transcript, timeline and autosave through
        `on_characters_changed`. Returns the focus project's report."""
        root_report = self._resolve_library_into(self.root.document)
        reports = {self.root.project_id: root_report}
        missing = set(root_report.missing)
        changed = bool(root_report.changed)
        for child in self.children.values():
            report = character_library.resolve_characters(
                child.document, [ParentStore(self.root.document), self.character_library])
            reports[child.project_id] = report
            missing |= set(report.missing)
            changed = changed or bool(report.changed)
        self.library_missing = missing
        if changed and refresh:
            self.on_characters_changed()
        return reports.get(self.focus.project_id, root_report)

    def character_scope(self, character) -> str:
        """NP3's three scopes plus the orphan: "global" (the library has its
        `library_id`), "project" (the root document has a character with it,
        minted at project scope), "local" (no `library_id`), or "missing"
        (linked, but neither store has it on this machine)."""
        library_id = character.library_id if character is not None else None
        if not library_id:
            return "local"
        if self.character_library.get(library_id) is not None:
            return "global"
        if character_library.is_project_scope_id(library_id) and \
                any(c.library_id == library_id for c in self.root.document.characters):
            return "project"
        return "missing"

    def _on_library_changed_on_disk(self) -> None:
        if self.character_library.mtime() == self._library_mtime:
            return
        self.resolve_library()

    def _begin_untitled_project_dir(self, document) -> None:
        """New: a fresh dir with an empty `document.json` and the lock, so an
        Untitled project has somewhere to generate into before its first
        Save (Claude/old/PLAN_tbaw_bundle.md section 3)."""
        project_dir, project_id = project_io.create_project_dir()
        root = self.root
        root.lock = project_io.ProjectLock(project_dir).acquire()
        root.project_dir = project_dir
        root.project_id = project_id
        root.manifest = {}
        digest = project_io.autosave_to_dir(document, root.project_settings, project_dir)
        project_io.write_session(project_dir, {
            "source_path": None, "zip_size": None, "zip_mtime": None,
            "saved_digest": digest, "dirty": False, "asset_index": {},
        })
        self._project_dirty = False

    def _backend_for(self, engine_id: str):
        """The resident adapter for `engine_id`, built once and kept, with
        its callbacks wired to the app (building one starts its worker
        thread but loads no model), or None for an engine that isn't
        registered."""
        engine_id = engine_id or self._primary_engine_id
        backend = self._backends.get(engine_id)
        if backend is not None:
            return backend
        try:
            backend = engine_registry.get_engine(engine_id)
        except Exception:  # noqa: BLE001 - an unregistered id is a warning, not a crash
            return None
        self._add_backend(backend)
        backend.on_project_opened(self.project_dir, self._engine_meta(backend.id))
        return backend

    def _add_backend(self, backend) -> None:
        bridge = EngineSignalBridge()
        wire_engine(backend.engine, bridge)
        self._connect_bridge(bridge)
        self._backends[backend.id] = backend
        self._bridges[backend.id] = bridge

    def _ensure_backend_ready(self, backend) -> None:
        """Loads `backend`'s pipeline once (the model, for Audio8), on its
        own worker."""
        if backend is None or backend.id in self._ready_backends:
            return
        self._ready_backends.add(backend.id)
        lang_code = self.settings.get("lang_code", "a")
        if self.settings_dock is not None:
            lang_code = self.settings_dock.get_state().get("lang_code", lang_code)
        backend.engine.worker.run_coro(
            backend.engine.init_pipeline_async(lang_code, device=self.settings.get("device", "auto")))

    @property
    def backends(self) -> dict:
        """Every resident backend, engine id -> adapter."""
        return dict(self._backends)

    def _document_engine_ids(self, document) -> list:
        """The engine ids `document`'s characters use, the primary engine
        first, registered ones only."""
        known = set(engine_registry.list_engines())
        ids = [self._primary_engine_id]
        for character in getattr(document, "characters", []) or []:
            engine_id = character.backend_id or self._primary_engine_id
            if engine_id in known and engine_id not in ids:
                ids.append(engine_id)
        return ids

    def backend_for_character(self, character):
        """The adapter a character generates with: its `backend_id`, or the
        primary engine when that id isn't registered here."""
        engine_id = (character.backend_id if character is not None else None) or self._primary_engine_id
        return self._backend_for(engine_id) or self._backends[self._primary_engine_id]

    def backend_for(self, clip, project=None):
        """The adapter `clip` generates with: its character's engine."""
        if clip is None:
            return self.backend_for_character(None)
        document = (project or self.project_for(clip)).document
        return self.backend_for_character(document.get_character(clip.character_id))

    def active_character(self):
        """The character the selection points at (a clip's, or a lane's),
        else the document's first character, else None."""
        document = self.document
        selection = getattr(self, "selection", None)
        if selection is not None:
            if selection.kind == "clip":
                clip = document.get_clip(selection.selected_clip_id)
                if clip is not None and document.get_character(clip.character_id) is not None:
                    return document.get_character(clip.character_id)
            elif selection.kind == "character":
                character = document.get_character(selection.selected_character_id)
                if character is not None:
                    return character
        return document.characters[0] if document.characters else None

    @property
    def backend(self):
        """The active backend: the engine of `active_character()`. What the
        Settings tab's schema form, the Voices tab, Preview and the
        whole-document path use; a clip always generates with
        `backend_for(clip)`."""
        return self.backend_for_character(self.active_character())

    @property
    def engine(self):
        return self.backend.engine

    @property
    def bridge(self):
        return self._bridges[self.backend.id]

    def set_character_engine(self, character, engine_id: str) -> bool:
        """The Characters dialog's engine picker: `character` generates with
        `engine_id` from now on. The backend is made resident and its
        pipeline loads. Refused while a job runs or for an unknown engine."""
        if self.is_busy():
            QMessageBox.warning(self, "Busy", "Cancel the current job before changing a character's engine.")
            return False
        backend = self._backend_for(engine_id)
        if backend is None:
            return False
        character.backend_id = engine_id
        self._ensure_backend_ready(backend)
        self._on_active_backend_maybe_changed(force=True)
        self.on_characters_changed()
        return True

    def _on_active_backend_maybe_changed(self, force: bool = False) -> None:
        """The selection moved: when the active engine changed, the
        Settings schema follows (the dock rebuilds on selection anyway),
        and the Voices tab and the JIT action follow here."""
        engine_id = self.backend.id
        if not force and engine_id == self._last_active_engine_id:
            return
        self._last_active_engine_id = engine_id
        if self.settings_dock is not None and force:
            self.settings_dock.rebuild_schema_form()
        if self.settings_dock is not None:
            self._sync_mixing_dock()
            self._sync_voice_clone_dock()
            self._sync_jit_action_enabled()

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
        self.new_subproject_action = self._action("New &Subproject", self.new_subproject_from_selection)
        self.open_action = self._action("&Open...", self.open_project_dialog, QKeySequence.StandardKey.Open)
        self.recent_menu = self.file_menu.addMenu("Recent")
        self.welcome_action = self._action("&Welcome...", self.show_welcome)
        self.save_action = self._action("&Save", self.save_project, QKeySequence.StandardKey.Save)
        self.save_as_action = self._action("Save &As...", self.save_project_as_dialog, QKeySequence.StandardKey.SaveAs)
        self.file_menu.insertAction(self.recent_menu.menuAction(), self.new_action)
        self.file_menu.insertAction(self.recent_menu.menuAction(), self.new_subproject_action)
        self.add_subproject_action = self._action("Add Su&bproject...", self.add_subproject_dialog)
        self.file_menu.insertAction(self.recent_menu.menuAction(), self.add_subproject_action)
        self.file_menu.insertAction(self.recent_menu.menuAction(), self.open_action)
        self.file_menu.addAction(self.welcome_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.save_action)
        self.file_menu.addAction(self.save_as_action)
        self.file_menu.addSeparator()
        self.import_text_action = self._action("Import &Text...", self.import_text_dialog)
        self.file_menu.addAction(self.import_text_action)
        self.import_subtitles_action = self._action("Import S&ubtitles...", self.import_subtitles_dialog)
        self.file_menu.addAction(self.import_subtitles_action)
        self.import_audio_action = self._action("Import Au&dio...", self.import_audio_dialog)
        self.import_audio_action.setToolTip("Add an audio file as a music bed on the Music track.")
        self.file_menu.addAction(self.import_audio_action)
        self.load_video_action = self._action("Load &Video...", self.load_video_dialog)
        self.file_menu.addAction(self.load_video_action)
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
        # No Engine menu: each character picks its engine in Edit >
        # Characters (grill V3).
        self.options_menu = bar.addMenu("&Options")
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
        self.video_dock = VideoDock(self)

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
        # The reference video (TB16) is the last tab of the right-hand group;
        # Load Video raises it.
        self.addDockWidget(top, self.video_dock)
        self.tabifyDockWidget(self.settings_dock, self.video_dock)
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
                 self.mixing_dock, self.voice_clone_dock, self.timeline_dock, self.transport_dock,
                 self.video_dock]
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

    def get_all_voices(self, lang_code: str | None = None, backend=None) -> list:
        """`spec.VOICE_DB` is Kokoro's built-in named-voice table; the
        backend-provided half comes from `backend.get_voices(...)` (the
        active backend by default)."""
        if lang_code is None:
            lang_code = self.settings.get("lang_code", "a")
        backend = backend or self.backend
        standard = spec.VOICE_DB.get(lang_code, [])
        custom = [v.id for v in backend.get_voices(lang_code)]
        return sorted(set(standard + custom))

    # --- settings persistence -

    def schedule_save(self) -> None:
        self.invalidate_child_states()
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
            for key in ("lang_code", "voice", "speed", "volume", "pitch", "num_threads",
                        "caching", "normalize", "format", *spec.SEGMENTATION_KEYS):
                if key in gen_state:
                    self.settings[key] = gen_state[key]
            self.settings["trim"] = gen_state.get("trim_silence", self.settings.get("trim", False))
            self.settings["apply_fx"] = self.settings_dock.apply_fx_enabled()
            self.settings["jit_enabled"] = self.jit_enabled
            self.settings.update(self.fx_dock.project_fx_state())
            if self.voice_clone_dock is not None:
                self.settings.update(self.voice_clone_dock.get_state())
            if hasattr(self, "workspaces"):
                self.workspaces.capture()

        qt_settings.save_settings(CONFIG_FILE, self.settings)
        self._autosave_project_dir()
        self._update_window_title(pending=False)

    def _autosave_project_dir(self) -> None:
        """Writes `document.json`/`project.json` into the project dir and
        sets the session's `dirty` iff the digest differs from what the
        last Save or Open recorded. A content comparison, not an mtime:
        `_switch_document`'s trailing `schedule_save` would otherwise dirty
        every project a second after Open. Never touches the `.tbaw`. Runs
        for the root and every open subproject."""
        for project in self.open_projects():
            self._autosave_one(project)

    def is_project_dirty(self) -> bool:
        """True when a project dir is ahead of its `.tbaw` on disk (or an
        Untitled project has edits): the root, or any open subproject, whose
        bundle rides inside the root's. Flushes a pending autosave first so
        a keystroke a moment ago counts."""
        if self._save_timer.isActive():
            self.save_settings()
        return self.any_project_dirty()

    def _set_setting(self, key: str, value) -> None:
        self.settings[key] = value
        self.schedule_save()

    def _update_window_title(self, pending: bool = False) -> None:
        name = project_io.project_title(self.project_path)
        star = "*" if (pending or self.any_project_dirty()) else ""
        self.setWindowTitle(f"{name}{star} - {APP_NAME}")

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
            **self._segmentation_config(gen_state),
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

    @staticmethod
    def _segmentation_config(gen_state: dict) -> dict:
        """The segmentation keys (`spec.SEGMENTATION_KEYS`) from the
        Settings tab, defaulted: where every path cuts text into pieces."""
        return {key: gen_state.get(key, spec.SETTINGS_DEFAULTS[key]) for key in spec.SEGMENTATION_KEYS}

    def _assemble_generation_config(self, clip, project=None) -> dict:
        """Exactly the inputs that decide what a clip's audio *is*: the
        segment key hashes these and nothing else, and `_assemble_clip_config`
        is built on top. App defaults from the Settings tab, the backend's
        "Model" schema group (Audio8's sampling knobs), the Lexicon (the
        key is over the text after substitution), the segmentation settings
        (they decide the pieces), then the clip's
        `effective_config_for_clip` (character preset, then overrides) on
        top, then `project_dir` and the clip's take. The take is read off
        `clip.overrides` directly, not through the `ALLOWED_PRESET_KEYS`
        whitelist, which is for untrusted preset files and shouldn't widen
        for a runtime counter. One method decides what both the dirty check
        and a Generate hash, so they can't drift."""
        project = project or self.project_for(clip)
        gen_state = self.settings_dock.get_state()
        backend = self.backend_for(clip, project)
        config = {
            "engine_id": backend.id,
            "lang_code": gen_state["lang_code"],
            "voice": gen_state["voice"],
            "speed": gen_state["speed"],
            "pitch": gen_state["pitch"],
            "lexicon": dict(self.settings.get("lexicon", {})),
            **self._segmentation_config(gen_state),
        }
        for field in backend.get_config_schema():
            if field.group == "Model" and field.key in gen_state:
                config[field.key] = gen_state[field.key]
        clip_config = dict(project.document.effective_config_for_clip(clip))
        for key in ("voice", "speed", "pitch", "lang_code"):
            if key in clip_config:
                config[key] = clip_config[key]
        variant_voice = self._variant_voice(clip, project)
        if variant_voice:
            config["voice"] = variant_voice
        config["project_dir"] = project.project_dir
        config["take"] = int(clip.overrides.get("take", 0) or 0)
        return config

    def _variant_voice(self, clip, project=None):
        """The reference name `clip.overrides["variant"]` picks from its
        character's `variants`, or None. Only a cloning backend has
        variants; on any other the override is ignored."""
        project = project or self.project_for(clip)
        variant = (clip.overrides or {}).get("variant")
        if not variant or not getattr(self.backend_for(clip, project).capabilities, "supports_voice_cloning", False):
            return None
        character = project.document.get_character(clip.character_id)
        if character is None:
            return None
        return (character.variants or {}).get(variant) or None

    def _assemble_clip_config(self, clip, project=None) -> dict:
        """The config dict for a per-clip Generate action and, through
        `post_config_for_clip`, for read-time post-processing:
        `_assemble_generation_config` plus everything that doesn't change
        the audio (output location and naming, threads, caching, the post
        keys) with `clip`'s character/override settings merged on top, so
        the clip's own values win. Defaults must come first:
        `process_chunk_task` reads `config['voice']` by direct indexing, so a clip with no character must still end up
        with usable defaults. FX come from `fx_resolve.resolve_fx`, the same
        resolver the Audio FX tab renders."""
        project = project or self.project_for(clip)
        gen_state = self.settings_dock.get_state()
        export = self._export_values()
        config = {
            "format": export["format"],
            "out_dir": export["out_dir"],
            "caching": gen_state["caching"],
            "time_id": time.strftime(self.timecode_format),
            "num_threads": gen_state["num_threads"],
            "volume": gen_state["volume"],
            "normalize": gen_state["normalize"],
            "trim_silence": gen_state["trim_silence"],
            "filename": os.path.basename(export["filename"]),
        }
        for key in ("cache_reference_codes",):
            if key in gen_state:
                config[key] = gen_state[key]

        clip_config = dict(project.document.effective_config_for_clip(clip))
        # ALLOWED_PRESET_KEYS whitelists "trim", but process_audio reads
        # "trim_silence" - same inline rename every other caller does.
        if "trim" in clip_config:
            clip_config["trim_silence"] = clip_config.pop("trim")
        config.update(clip_config)
        config.update(self._assemble_generation_config(clip, project))
        if project.project_dir:
            # Clips generate straight into the project dir, once, named by
            # their segment key (Claude/old/PLAN_tbaw_bundle.md section 3). The
            # bundle's audio format decides the extension of new segments,
            # over a preset's `format` (that one is for the export path).
            config["out_dir"] = os.path.join(project.project_dir, *project_io.AUDIO_GENERATED.split("/"))
            config["segment_naming"] = "cache_key"
            config["format"] = project_io.bundle_options(project.project_settings)["audio_format"]

        # effective_config_for_clip only ever carries the FX preset's *name*;
        # the resolver turns project values + character preset + clip preset
        # + clip.fx_override into the actual FX keys and the ANDed apply_fx.
        resolution = fx_resolve.resolve_fx(self, clip=clip, project=project)
        config.update(resolution.values)
        config["apply_fx"] = resolution.apply_fx
        return config

    def _install_segment_key_fn(self, project=None) -> None:
        """Sets `Document.segment_key_fn` to a closure that keys each clip
        with its own character's backend (`backend_for`) and the project dir (Claude/old/PLAN_tbaw_bundle.md section 2.3):
        `key_fn(text, clip, engine_version=None) -> caching.segment_key`
        over `_assemble_generation_config(clip)`. Memoized on the text, the
        config and the version; a hit re-checks the voice file's mtime and
        the backend's `cache_key_extra` (Audio8's transcript, itself cached
        by mtime), which is the only way a key changes without its inputs
        changing (a re-saved mix, a re-recorded reference). So a rehighlight
        of a book costs about two stats per clip and no hashing or reads.
        Also sets `Document.generation_config_fn`, so the dirty check reads
        the lexicon and segmentation settings the generation will use."""
        project = project or self.focus
        memo: dict = {}

        def key_fn(text, clip, engine_version=None):
            backend = self.backend_for(clip, project)
            config = self._assemble_generation_config(clip, project)
            memo_key = (text, json.dumps(config, sort_keys=True, default=str), engine_version)
            name, _fp = caching.normalize_voice(config.get("voice"), backend, config.get("project_dir"))
            voice_file = backend.resolve_voice_file(name, config.get("project_dir")) if name else None
            try:
                stamp = os.path.getmtime(voice_file) if voice_file else None
            except OSError:
                stamp = None
            extra = backend.cache_key_extra(config)
            hit = memo.get(memo_key)
            if hit is not None and hit[0] == stamp and hit[1] == extra:
                return hit[2]
            key = caching.segment_key(text, config, backend, engine_version)
            memo[memo_key] = (stamp, extra, key)
            return key

        def config_fn(clip):
            return self._dirty_check_config(clip, project)

        project.document.segment_key_fn = key_fn
        project.document.generation_config_fn = config_fn

    def _dirty_check_config(self, clip, project=None) -> dict:
        """`Document.generation_config_fn`: `_assemble_generation_config`,
        or the clip's own config while the Settings tab is still being
        built (the editor can rehighlight before it exists)."""
        project = project or self.project_for(clip)
        if self.settings_dock is None:
            return project.document.effective_config_for_clip(clip)
        return self._assemble_generation_config(clip, project)

    # --- read-time post-processing (kokoro_gui/audio/post.py) ---------------

    def post_config_for_clip(self, clip, project=None) -> dict:
        """The `POST_KEYS` subset of the clip's resolved config: what the
        transport, the exporter and the timeline waveform apply on top of
        the raw segment files. Changing any of it never dirties the clip.
        A nested clip or a music bed gets only the FX set on it
        (`nested_post_config`): the project's volume, trim and normalize are
        for speech."""
        if clip.has_placeholder:
            return self.nested_post_config(clip, project)
        return post.extract_post_config(self._assemble_clip_config(clip, project))

    def clip_duration_s(self, clip, project=None):
        """`compute_arrangement`'s `clip_duration`: the clip's rendered
        length (trim and pitch change it), or the raw `Segment.duration`
        for a file that can't be read, or None with no audio at all. A
        segment with a stored onset and tail is measured from those, without
        reading its file (`post.duration_hint`), and a segment with a
        `range` as `end - start`. Falls back to the raw
        durations while the docks are still being built. A nested clip is
        its child's mixdown length, from `mixdown.json` (no read). A music
        bed is its trim or loop length, from the file's header
        (`beds.bed_segments`)."""
        if clip.is_nested:
            return self.nested_duration_s(clip, project)
        segments = playable_segments(clip)
        if not segments:
            return None
        if self.settings_dock is None or self.fx_dock is None:
            return clip_audio_duration_s(clip)
        post_config = self.post_config_for_clip(clip, project)
        rate = self.project_sample_rate()
        total = 0.0
        for segment in segments:
            try:
                total += post.rendered_duration_s(segment.audio_path, post_config, rate,
                                                  hint=post.duration_hint(segment, post_config),
                                                  range_s=post.segment_range(segment))
            except Exception:
                total += segment.duration or 0.0
        return total

    def rendered_clip_samples(self, clip, project=None):
        """`(samples, rate)` for the clip's segments concatenated and
        post-processed, or None. The timeline draws its waveform from this
        so it shows what the transport plays."""
        if clip.is_nested:
            path = self.nested_audio_path(clip, project)
            if not path:
                return None
            rate = self.project_sample_rate()
            try:
                return post.render(path, self.post_config_for_clip(clip, project), rate), rate
            except Exception:
                return None
        segments = playable_segments(clip)
        if not segments:
            return None
        post_config = self.post_config_for_clip(clip, project)
        rate = self.project_sample_rate()
        parts = []
        for segment in segments:
            try:
                parts.append(post.render(segment.audio_path, post_config, rate, post.segment_range(segment)))
            except Exception:
                continue
        if not parts:
            return None
        import numpy as np

        return np.concatenate(parts), rate

    def build_arrangement(self, project=None):
        """Every `compute_arrangement` call goes through here so they all
        measure clips the same way. Default: the `level` project, the one
        the timeline and transport show. The clip's post config decides
        whether onset alignment applies (trim already cut the silence)."""
        project = project or self.level
        docks_ready = self.settings_dock is not None and self.fx_dock is not None
        return compute_arrangement(project.document, engine_id=self.backend.id,
                                   clip_duration=lambda clip: self.clip_duration_s(clip, project),
                                   clip_estimate=self.nested_estimate_s,
                                   clip_post_config=(lambda clip: self.post_config_for_clip(clip, project))
                                   if docks_ready else None)

    # --- Options: engine / device / theme ---------------------------------

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
        self.set_status(f"Re-initializing engines on {device}...")
        for engine_id in sorted(self._ready_backends):
            engine = self._backends[engine_id].engine
            engine.worker.run_coro(engine.init_pipeline_async(lang_code, device=device))

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
        """The root project now holds `document` (New, Open, a migration);
        focus and level go back to it."""
        self.transport.stop()
        self.focus = self.level = self.root
        self.root.document = document
        if self._library_link_ids is not None:
            # First launch with the library: link what the presets import
            # made an exact copy of (one time only).
            entries = [e for e in self.character_library.list() if e.library_id in set(self._library_link_ids)]
            link_exact_matches(document, entries)
            self._library_link_ids = None
        self._resolve_library_into(document)
        self.project_path = os.path.abspath(path) if path else None
        self.project_settings = dict(project_settings or {})
        self._install_segment_key_fn()
        self._install_nested_state_fn(self.root)
        for engine_id in self._document_engine_ids(document):
            self._ensure_backend_ready(self._backend_for(engine_id))
        for backend in list(self._backends.values()):
            backend.on_project_opened(self.project_dir, self._engine_meta(backend.id))
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
            self.transcript_dock.refresh_scope()
        if self.settings_dock is not None:
            self.settings_dock.rebuild_schema_form()
            self.settings_dock.refresh_scope_fields()
        if self.fx_dock is not None:
            self.fx_dock.refresh_for_selection()
        self._sync_video_dock()
        self.refresh_timeline()
        session = project_io.read_session(self.root.project_dir) if self.root.project_dir else None
        loop = (session or {}).get("loop_s")
        if isinstance(loop, list) and len(loop) == 2:
            self.set_loop_range(loop[0], loop[1], remember=False)
        else:
            self.set_loop_range(None, None, remember=False)
        self._update_window_title()
        self.schedule_save()

    def set_loop_range(self, start_s, end_s, remember: bool = True) -> None:
        """The transport's loop region (phase 2, A3), shaded on the ruler.
        Runtime state: it goes into the project dir's `session.json`, never
        into the document or the bundle. `None` clears it."""
        if start_s is None or end_s is None or abs(float(end_s) - float(start_s)) < 1e-3:
            loop = None
        else:
            loop = tuple(sorted((max(0.0, float(start_s)), max(0.0, float(end_s)))))
        self.transport.set_loop_range_s(*(loop or (None, None)))
        if self.timeline_dock is not None:
            self.timeline_dock.timeline_view.set_loop_s(loop)
        if remember and self.root.project_dir:
            session = project_io.read_session(self.root.project_dir) or {}
            session["loop_s"] = list(loop) if loop else None
            try:
                project_io.write_session(self.root.project_dir, session)
            except OSError:
                pass

    def loop_range(self):
        """`(start_s, end_s)` of the loop region, or None."""
        loop = self.transport.loop_range
        if loop is None:
            return None
        rate = float(self.transport.sample_rate)
        return loop[0] / rate, loop[1] / rate

    def _engine_meta(self, engine_id: str) -> dict:
        engines = self._project_manifest.get("engines") if isinstance(self._project_manifest, dict) else None
        block = (engines or {}).get(engine_id) if isinstance(engines, dict) else None
        meta = block.get("meta") if isinstance(block, dict) else None
        return dict(meta) if isinstance(meta, dict) else {}

    # -- closing the current project ----------------------------------------

    def _ask_close_choice(self) -> str:
        """Save / Discard / Cancel for a dirty project (grill TB12). A
        method so tests can replace it."""
        box = QMessageBox(self)
        box.setWindowTitle("Unsaved changes")
        box.setText(self._unsaved_changes_text())
        save_btn = box.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
        discard_btn = box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(save_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is save_btn:
            return "save"
        if clicked is discard_btn:
            return "discard"
        return "cancel"

    def _unsaved_changes_text(self) -> str:
        """The close prompt's line: the root, and by name any subproject
        whose unsaved edits it carries (one prompt for the whole tree)."""
        text = f"{project_io.project_title(self.project_path)} has unsaved changes."
        children = [c.title() for c in self.children.values() if c.dirty]
        if children:
            text += " Changed subprojects: " + ", ".join(children) + "."
        return text

    def _close_current_project(self, then) -> None:
        """Runs `then()` once the open project is put away: a clean project
        is GC'd and released; a dirty one asks Save / Discard / Cancel, and
        Save continues after the background Save succeeds."""
        if not self.root.project_dir:
            then()
            return
        if not self.is_project_dirty():
            self._teardown_project(discard=False)
            then()
            return
        choice = self._ask_close_choice()
        if choice == "cancel":
            return
        if choice == "discard":
            self._teardown_project(discard=True)
            then()
            return
        path = self.project_path
        if not path:
            path = self._save_as_path_dialog()
            if not path:
                return
            self.project_path = path

        def _after_save():
            self._teardown_project(discard=False)
            then()

        self._save_bundle(self.project_path, then=_after_save)

    def _teardown_project(self, discard: bool) -> None:
        """Releases the locks (every open subproject's, then the root's); on
        Discard deletes the dirs (the zip has the last saved state),
        otherwise GCs orphaned segments (TB11: only at close, when no undo
        history can point at them any more)."""
        self._teardown_children(discard)
        root = self.root
        if root.lock is not None:
            if not discard:
                try:
                    project_io.gc_project_dir(root.project_dir, root.document)
                except OSError:
                    pass
            root.lock.release()
            root.lock = None
        if discard and root.project_dir:
            project_io.delete_project_dir(root.project_dir)
        root.project_dir = None
        root.project_id = None
        root.manifest = {}
        root.dirty = False

    def _evict_other_project_dirs(self) -> None:
        """TB13: only the open project's dir stays; every other clean,
        unlocked dir under `cache/projects/` goes."""
        try:
            project_io.evict_project_dirs(self.root.project_dir)
        except OSError:
            pass

    # -- new ------------------------------------------------------------------

    def new_project(self, then=None) -> None:
        """File > New. `then()` runs once the new project is in place."""
        def _start():
            document = project_io.new_document_from(self.character_library, self.settings)
            self.project_settings = {}
            self._begin_untitled_project_dir(document)
            self._switch_document(document, None)
            self._evict_other_project_dirs()
            self.set_status("New project with the library's characters. Save As to name it.")
            if then is not None:
                then()

        self._close_current_project(_start)

    # -- open -----------------------------------------------------------------

    def open_project(self, path: str) -> None:
        if self.is_busy():
            QMessageBox.warning(self, "Busy", "Finish or cancel the current job before opening a project.")
            return
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Open failed", f"Couldn't read {path}.")
            project_io.forget_recent(self.settings, path)
            self._rebuild_recent_menu()
            return
        if project_io.format_for_path(path) != "tbaw":
            self._open_json_project(path)
            return
        try:
            info = project_io.inspect_bundle(path)
        except project_io.ProjectError as e:
            QMessageBox.warning(self, "Open failed", str(e))
            return
        if self.project_path and os.path.abspath(path) == self.project_path and self.project_id == info.project_id:
            return  # already open: the welcome dialog's Resume
        self._close_current_project(lambda: self._open_bundle(info))

    def _ask_recover_choice(self, session: dict, info: project_io.BundleInfo, project_dir: str) -> str:
        """Recover prompt (grill TB15): "keep" the unsaved session, "take"
        the file (wipe and extract fresh) or "cancel". A method so tests can
        replace it."""
        stamp = ""
        try:
            stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(
                os.path.getmtime(os.path.join(project_dir, project_io.DOCUMENT))))
        except OSError:
            pass
        lines = [f"Recover unsaved changes{' from ' + stamp if stamp else ''}?"]
        theirs = session.get("source_path")
        if theirs and os.path.abspath(theirs) != os.path.abspath(info.path):
            lines.append(f"They were made on {theirs}.")
        changed = not project_io.session_matches_file(session, info)
        if changed:
            lines.append("The file was changed outside KokoroGUI since.")
        box = QMessageBox(self)
        box.setWindowTitle("Recover project")
        box.setText("\n".join(lines))
        keep_btn = box.addButton("Keep session", QMessageBox.ButtonRole.AcceptRole)
        take_btn = box.addButton("Take file" if changed else "Discard session", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(keep_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is keep_btn:
            return "keep"
        if clicked is take_btn:
            return "take"
        return "cancel"

    def _open_bundle(self, info: project_io.BundleInfo) -> None:
        """Steps 1-5 of Open (Claude/old/PLAN_tbaw_bundle.md section 6): the
        project dir and its lock, the recover prompt, the free-space check,
        the small entries on this thread, the audio on a background thread
        behind `is_busy` with the editor read-only."""
        project_dir = project_io.choose_project_dir(info.project_id, info.path)
        try:
            lock = project_io.ProjectLock(project_dir).acquire()
        except project_io.ProjectLockedError as e:
            QMessageBox.warning(self, "Already open", str(e))
            self._start_untitled_after_failed_open()
            return
        try:
            project_io.sweep_orphan_dirs()
        except OSError:
            pass

        session = project_io.read_session(project_dir)
        recovered = False
        extract = True
        # Unsaved subproject edits (phase 4) count as the project's own: the
        # recover prompt covers them, and taking the file drops them too.
        child_dirs = [d for d in project_io.child_dirs_of(info.path) if not project_io.is_locked(d)]
        dirty_children = [d for d in child_dirs if (project_io.read_session(d) or {}).get("dirty")]
        has_document = os.path.isfile(os.path.join(project_dir, project_io.DOCUMENT))
        if session and has_document and (session.get("dirty") or dirty_children):
            choice = self._ask_recover_choice(session, info, project_dir)
            if choice == "cancel":
                lock.release()
                self._start_untitled_after_failed_open()
                return
            if choice == "keep":
                recovered = True
                extract = False
            else:
                project_io.wipe_project_dir(project_dir)
                for child_dir in child_dirs:
                    project_io.delete_project_dir(child_dir)
        elif session and project_io.session_matches_file(session, info) and has_document \
                and not project_io.video_extract_pending(info, project_dir):
            extract = False  # a clean extraction of this very file: Resume is fast
        else:
            project_io.wipe_project_dir(project_dir)

        if not extract:
            self._finish_open(info, project_dir, lock, recovered)
            return

        heavy_bytes = project_io.heavy_bytes(info)
        try:
            project_io.check_free_space(project_io.projects_root(), heavy_bytes, "open the project")
            project_io.extract_small(info, project_dir)
        except (project_io.ProjectError, OSError) as e:
            lock.release()
            QMessageBox.warning(self, "Open failed", str(e))
            self._start_untitled_after_failed_open()
            return

        if heavy_bytes == 0:
            self._finish_open(info, project_dir, lock, recovered)
            return

        self._begin_project_io(f"Opening {project_io.project_title(info.path)}...", read_only=True)

        def _work():
            project_io.extract_audio(info, project_dir, progress=self._io_progress("Extracting audio"))

        def _done(_result, error):
            self._end_project_io(read_only=True)
            if error is not None:
                lock.release()
                QMessageBox.warning(self, "Open failed", str(error))
                self._start_untitled_after_failed_open()
                return
            self._finish_open(info, project_dir, lock, recovered)

        self._run_project_io(_work, _done)

    def _start_untitled_after_failed_open(self) -> None:
        """The previous project was already put away when an Open fails
        partway; the window can't sit on a document with no dir."""
        if self.root.project_dir:
            return
        document = project_io.new_document_from(self.character_library, self.settings)
        self.project_settings = {}
        self._begin_untitled_project_dir(document)
        self._switch_document(document, None)

    def _finish_open(self, info, project_dir: str, lock, recovered: bool) -> None:
        """Steps 6 and 7: the document, the session record, the TB9 status
        line, and the backend's `on_project_opened`."""
        versions = self._engine_versions(info.manifest)
        try:
            loaded = project_io.finish_open(info, project_dir, versions, recovered=recovered)
        except (OSError, ValueError, KeyError) as e:
            lock.release()
            QMessageBox.warning(self, "Open failed", f"Couldn't read the project: {e}")
            self._start_untitled_after_failed_open()
            return
        root = self.root
        root.lock = lock
        root.project_dir = project_dir
        root.project_id = info.project_id
        root.manifest = dict(info.manifest)
        root.dirty = bool(recovered)
        self._switch_document(loaded.document, info.path, loaded.project_settings)
        self._evict_other_project_dirs()
        for notice in loaded.notices:
            self.set_status(notice, "warning")
        if not loaded.notices:
            self.set_status(f"Opened {project_io.project_title(info.path)}.")
        if project_io.video_settings(self.root.project_settings) is not None and self.video_path() is None:
            self._offer_video_relink()

    def _open_json_project(self, path: str) -> None:
        """TB6: a 4.0-preview `.json` project opens, gets a project id and a dir,
        has its segments rekeyed and copied in (`migrate_segments`), and is
        saved as `<name>.tbaw` next to the `.json`, which then becomes the
        recent entry. A `.tbaw` already there from an earlier migration is
        opened instead. An unwritable directory falls through to Save As."""
        target = project_io.bundle_path_for(path)
        if os.path.isfile(target):
            self.open_project(target)
            return
        loaded = project_io.load_json_project(path)
        if loaded is None:
            QMessageBox.warning(self, "Open failed", f"Couldn't read {path}.")
            project_io.forget_recent(self.settings, path)
            self._rebuild_recent_menu()
            return

        def _migrate():
            self.project_settings = dict(loaded.project_settings)
            self._begin_untitled_project_dir(loaded.document)
            self.document = loaded.document
            self._install_segment_key_fn()
            audio_format = project_io.bundle_options(self.project_settings)["audio_format"]
            counts = project_io.migrate_segments(loaded.document, self.project_dir, self._assemble_generation_config,
                                                 self.document.segment_key_fn, audio_format)
            self._switch_document(loaded.document, None, self.project_settings)
            project_io.forget_recent(self.settings, path)
            self._rebuild_recent_menu()
            if counts["adopted"] or counts["dropped"]:
                self.set_status(f"Migrated {os.path.basename(path)}: {counts['adopted']} segment(s) kept, "
                                f"{counts['dropped']} will regenerate.", "info")
            save_to = target
            if not os.access(os.path.dirname(os.path.abspath(target)) or ".", os.W_OK):
                save_to = self._save_as_path_dialog()
                if not save_to:
                    return
            self.project_path = os.path.abspath(save_to)
            project_io.remember_recent(self.settings, self.project_path)
            self._rebuild_recent_menu()
            self._update_window_title()
            self._save_bundle(self.project_path)

        self._close_current_project(_migrate)

    def open_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", project_io.PROJECT_FILTER)
        if path:
            self.open_project(path)

    # -- save -----------------------------------------------------------------

    def save_project(self) -> None:
        if not self.project_path:
            self.save_project_as_dialog()
            return
        if self.is_busy():
            QMessageBox.warning(self, "Busy", "Finish or cancel the current job before saving.")
            return
        self._save_bundle(self.project_path)

    def save_project_as(self, path: str) -> None:
        """Save As keeps the project dir (it's keyed by `project_id`), so no
        `audio_path` changes; only `source_path` moves."""
        if self.is_busy():
            QMessageBox.warning(self, "Busy", "Finish or cancel the current job before saving.")
            return
        new_path = os.path.abspath(project_io.bundle_path_for(path))
        self._rebase_linked_paths(new_path)
        self._rebase_video_path(new_path)
        self.project_path = new_path
        project_io.remember_recent(self.settings, self.project_path)
        self._rebuild_recent_menu()
        self._update_window_title()
        self._save_bundle(self.project_path)

    def _save_as_path_dialog(self) -> str | None:
        start = self.project_path or ""
        path, _ = QFileDialog.getSaveFileName(self, "Save project as", start, project_io.PROJECT_FILTER)
        return os.path.abspath(project_io.bundle_path_for(path)) if path else None

    def save_project_as_dialog(self) -> None:
        path = self._save_as_path_dialog()
        if path:
            self.save_project_as(path)

    def _save_bundle(self, path: str, then=None) -> None:
        """Save (Claude/old/PLAN_tbaw_bundle.md section 3): the document and
        assets are planned on this thread (a snapshot), the zip is written
        on a background thread behind `is_busy`, and the session record is
        written back here. `then()` runs after a successful save."""
        if self._save_timer.isActive():
            self.save_settings()
        root = self.root
        try:
            # Open subprojects first (deepest first): an embedded child's
            # bundle goes into its parent's project dir, where the parent's
            # plan picks it up; a linked child's to its own file.
            child_plans, warnings = self._plan_child_saves(FX_PRESETS_DIR)
            plan, root_warnings = project_io.plan_save(
                root.document, root.project_settings, path, root.project_dir, root.project_id,
                self._backend_for, FX_PRESETS_DIR, project_io.read_session(root.project_dir), root.manifest,
                pending_children={c.project_id for c, _p, _s in child_plans if c.parent_id == root.project_id},
            )
            warnings = [*warnings, *root_warnings]
        except Exception as e:  # noqa: BLE001 - surfaced, never a crash
            self.set_status(f"Save failed: {e}", "error")
            return
        for warning in warnings:
            self.set_status(f"Save: {warning}", "warning")
        known_ids = list(engine_registry.list_engines())
        self._begin_project_io(f"Saving {project_io.project_title(path)}...", read_only=False)

        def _work():
            child_results = [project_io.write_bundle(child_plan, known_ids) for _c, child_plan, _s in child_plans]
            # The parent's plan listed its children's bundles by path; their
            # bytes are the ones just written.
            return child_results, project_io.write_bundle(plan, known_ids,
                                                          progress=self._io_progress("Writing bundle"))

        def _done(result, error):
            self._end_project_io(read_only=False)
            if error is not None:
                self.set_status(f"Save failed: {error}", "error")
                return
            child_results, result = result
            self._record_child_saves(child_plans, child_results)
            try:
                project_io.record_save(root.project_dir, path, result)
            except OSError as e:
                self.set_status(f"Saved, but couldn't record the session: {e}", "warning")
            root.manifest = dict(plan.manifest)
            root.dirty = False
            self._update_window_title()
            self.set_status(f"Saved {project_io.project_title(path)}.", "success")
            if then is not None:
                then()

        self._run_project_io(_work, _done)

    # -- reference video (phase 5, TB16) -----------------------------------------

    def video_path(self) -> str | None:
        """The file the video dock plays: the root project's
        `project_settings["video"]["path"]` when that file exists, else the
        copy Open extracted from the bundle, else None."""
        root = self.root
        return project_io.video_source(root.project_settings, root.path, root.project_dir, root.manifest)

    def _sync_video_dock(self) -> None:
        if self.video_dock is None:
            return
        block = project_io.video_settings(self.root.project_settings)
        if block is None:
            self.video_dock.set_video(None)
            return
        path = self.video_path()
        missing = None if path else (project_io.resolve_video_path(self.root.project_settings, self.root.path)
                                     or block["path"])
        self.video_dock.set_video(path, block["offset_s"], missing=missing)

    def load_video(self, path: str) -> bool:
        """File > Load Video...: the root project's `project_settings["video"]`
        names `path`, relative to the project's file when it has one on the
        same drive. The offset is kept. Returns False when `path` isn't a file."""
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Load video", f"Couldn't read {path}.")
            return False
        block = project_io.video_settings(self.root.project_settings)
        self.root.project_settings["video"] = {
            "path": project_io.video_path_for(path, self.root.path),
            "offset_s": block["offset_s"] if block else 0.0,
        }
        self._sync_video_dock()
        if self.video_dock is not None and self.workspaces.active != SIMPLE:
            self.video_dock.show()
            self.video_dock.raise_()
        self.schedule_save()
        self.set_status(f"Reference video: {os.path.basename(path)}.")
        return True

    def load_video_dialog(self) -> bool:
        path, _ = QFileDialog.getOpenFileName(self, "Load video", "", project_io.VIDEO_FILTER)
        return bool(path) and self.load_video(path)

    def set_video_offset(self, offset_s: float) -> None:
        """The video dock's offset box: video time is transport time plus
        `offset_s`."""
        block = self.root.project_settings.get("video")
        if not isinstance(block, dict):
            return
        block["offset_s"] = round(float(offset_s), 3)
        if self.video_dock is not None:
            self.video_dock.set_offset(block["offset_s"])
        self.schedule_save()

    def _offer_video_relink(self) -> bool:
        """The reference video is neither at its path nor in the bundle: ask
        for it, the way a missing linked subproject asks for its file (NP4)."""
        block = project_io.video_settings(self.root.project_settings) or {}
        missing = project_io.resolve_video_path(self.root.project_settings, self.root.path) or block.get("path")
        answer = QMessageBox.question(self, "Video not found",
                                      f"The reference video {missing} isn't there.\nFind the video file?")
        if answer == QMessageBox.StandardButton.Yes:
            return self.load_video_dialog()
        return False

    def _rebase_video_path(self, new_root_path: str) -> None:
        """Save As moved the root: a relative video path is rewritten for the
        new location (the video didn't move)."""
        absolute = project_io.resolve_video_path(self.root.project_settings, self.root.path)
        if absolute:
            self.root.project_settings["video"]["path"] = project_io.video_path_for(absolute, new_root_path)

    # -- background I/O plumbing ------------------------------------------------

    def _begin_project_io(self, status: str, read_only: bool) -> None:
        self.set_ui_state(True)
        self.set_status(status, "busy")
        self.transport_dock.set_progress(0, "")
        if read_only and self.editor is not None:
            self.editor.setReadOnly(True)

    def _end_project_io(self, read_only: bool) -> None:
        if read_only and self.editor is not None:
            self.editor.setReadOnly(False)
        self.set_ui_state(False)

    def _io_progress(self, label: str):
        def _progress(done, total):
            percent = (done / total * 100.0) if total else 100.0
            self.projectIoProgress.emit(percent, f"{label} {int(percent)}%")

        return _progress

    def _on_project_io_progress(self, percent: float, detail: str) -> None:
        self.transport_dock.set_progress(percent, detail)

    def _run_project_io(self, work, done) -> None:
        """`work()` on a plain thread (independent of the engine's worker
        loop, so an engine switch mid-save can't strand it); `done(result,
        error)` back on the GUI thread."""

        def _target():
            try:
                result = work()
                self._projectIoFinished.emit((done, result, None))
            except BaseException as e:  # noqa: BLE001 - delivered to the GUI thread
                self._projectIoFinished.emit((done, None, e))

        self._io_thread = threading.Thread(target=_target, name="project-io", daemon=True)
        self._io_thread.start()

    def _on_project_io_finished(self, payload) -> None:
        done, result, error = payload
        self._io_thread = None
        done(result, error)

    def wait_for_project_io(self, timeout_s: float = 60.0) -> None:
        """Blocks until the background Open/Save (if any) has finished and
        its completion has run on this thread. For tests and scripts."""
        deadline = time.time() + timeout_s
        while self._io_thread is not None and time.time() < deadline:
            thread = self._io_thread
            if thread is not None:
                thread.join(0.02)
            QApplication.processEvents()
        QApplication.processEvents()

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

    def import_subtitles_dialog(self) -> None:
        patterns = " ".join("*" + ext for ext in subtitles.SUBTITLE_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(self, "Import subtitles", filter=f"Subtitles ({patterns})")
        if path:
            self.import_subtitles(path)

    def _ask_speaker_mapping(self, speakers: list, characters: list):
        """The speaker mapping dialog: `{speaker: choice}` or None on
        Cancel. Its own method so tests answer it without a modal."""
        return SpeakerMappingDialog.ask(speakers, characters, self)

    def _new_speaker_character(self, name: str, index: int):
        """A local character for a speaker the mapping dialog marked new,
        made the way Edit > Characters' Add makes one. No track until a
        clip uses it (grill PR4)."""
        color = DEFAULT_HIGHLIGHT_PALETTE[index % len(DEFAULT_HIGHLIGHT_PALETTE)]
        return Character.from_preset_dict(name, {"voice": self.settings.get("voice", "af_heart")},
                                          highlight_color=color, backend_id=self.backend.id)

    def import_subtitles(self, path: str) -> list:
        """File > Import Subtitles (phase 5 D2): every cue of an SRT, VTT or
        ASS file becomes a paragraph at the end of the focus project's
        transcript and a clip locked in time at the cue's start, with the
        cue as its `source_text` and the cue's length as its target
        duration (`kokoro_gui.daw.undo.ImportCuesCommand`). When the file
        names speakers, the mapping dialog asks once which character voices
        each; Cancel imports nothing. One undo step. Returns the new clips'
        ids ([] when nothing was imported)."""
        try:
            cues = subtitles.parse(path)
        except (OSError, subtitles.SubtitleError) as e:
            QMessageBox.critical(self, "Import failed", f"Read failed: {e}")
            return []
        if not cues:
            QMessageBox.warning(self, "Empty", "No subtitle cues found in that file.")
            return []
        document = self.document
        rows = speaker_rows(cues)
        mapping = {row: NARRATOR for row in rows or [NO_SPEAKER]}
        if rows:
            answer = self._ask_speaker_mapping(rows, list(document.characters))
            if answer is None:
                return []
            mapping.update({k: v for k, v in answer.items() if k in mapping})
        character_ids, new_characters = resolve_mapping(mapping, document.characters, self._new_speaker_character)
        command = ImportCuesCommand(cues, [character_ids[cue.speaker or NO_SPEAKER] for cue in cues],
                                    new_characters)
        document.undo_stack.push(command)
        if self.editor is not None:
            self.editor.load_text(document.text)
        self.on_characters_changed()
        self.set_status(f"Imported {len(cues)} subtitle cue(s) from {os.path.basename(path)}.")
        return command.clip_ids

    def import_audio_dialog(self) -> None:
        """File > Import Audio...: picks a file and adds it as a music bed.
        Recording import (phase 5 P3) adds its own choice ahead of this
        one; a bed is the only kind of audio import so far."""
        path, _ = QFileDialog.getOpenFileName(self, "Import audio", "", project_io.AUDIO_FILTER)
        if not path:
            return
        at_s = self._ask_bed_placement(self.transport.position())
        if at_s is None:
            return
        self.import_music_bed(path, at_s)

    def _ask_bed_placement(self, playhead_s: float):
        """Where a new bed starts: 0.0, or the playhead when it isn't at
        the start and the user picks it. None on Cancel. Its own method so
        tests answer it without a modal."""
        if playhead_s <= 0.0:
            return 0.0
        box = QMessageBox(self)
        box.setWindowTitle("Import audio")
        box.setText("Where should the audio start?")
        start = box.addButton("At the start", QMessageBox.ButtonRole.AcceptRole)
        playhead = box.addButton(f"At the playhead ({playhead_s:.1f}s)", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(start)
        box.exec()
        clicked = box.clickedButton()
        if clicked is start:
            return 0.0
        if clicked is playhead:
            return float(playhead_s)
        return None

    def import_music_bed(self, path: str, at_s: float = 0.0):
        """A music bed (phase 5 P2, grill Q30): the file is copied into the
        focus project's `audio/imported/` (`project.import_audio_file`) and
        becomes an imported clip on the "Music" track, pinned at `at_s`,
        with its file name as a read-only line at the end of the
        transcript (`kokoro_gui.daw.undo.ImportBedCommand`). One undo step.
        Returns the new clip's id, or None when the file can't be read or
        copied."""
        try:
            import soundfile as sf

            sf.info(path)
        except Exception as e:
            QMessageBox.critical(self, "Import failed", f"Can't read {os.path.basename(path)} as audio: {e}")
            return None
        try:
            stored = project_io.import_audio_file(path, self.project_dir)
        except (OSError, project_io.ProjectError) as e:
            QMessageBox.critical(self, "Import failed", str(e))
            return None
        title = os.path.splitext(os.path.basename(path))[0].strip() or "Audio"
        command = ImportBedCommand(stored, title, at_s)
        self.document.undo_stack.push(command)
        if self.editor is not None:
            self.editor.load_text(self.document.text)
            self.editor.rehighlight()
        self.schedule_save()
        self.refresh_timeline()
        self.set_status(f"Imported {os.path.basename(path)} as a music bed.")
        return command.clip_id

    def export_dialog(self) -> None:
        dialog = ExportDialog(self)
        if dialog.exec() != ExportDialog.DialogCode.Accepted:
            return
        run_export(self, dialog.values(), parent=self, bundle=dialog.bundle_values(), range_s=dialog.range_s())

    def _on_export_progress(self, percent: float, detail: str) -> None:
        self.transport_dock.set_progress(percent, detail)

    def _on_export_finished(self, success: bool, message: str) -> None:
        self.transport_dock.set_busy(False)
        self.transport_dock.set_progress_value(100 if success else 0)
        self.set_status(message, "success" if success else "error")

    def project_sample_rate(self) -> int:
        """The mix rate for transport and export: the highest output rate
        among the engines the open projects' characters use (44.1k with an
        Audio8 character, 24k otherwise). Clips rendered at another rate
        are resampled once on load."""
        engine_ids = set()
        for project in self.open_projects():
            engine_ids.update(self._document_engine_ids(project.document))
        rates = [int(getattr(self._backends[eid].engine, "SAMPLE_RATE", 24000) or 24000)
                 for eid in engine_ids if eid in self._backends]
        return max(rates, default=24000)

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
        if not is_running and self._subproject_queue:
            # The next stale subproject, once this job's handler is done.
            QTimer.singleShot(0, self._advance_subproject_queue)
        self.transport_dock.set_busy(is_running)
        threads_widget = self.settings_dock.schema_form.widget_for("num_threads")
        if threads_widget is not None:
            threads_widget.setEnabled(not is_running)
        self.settings_dock.volume_spin.setEnabled(not is_running)
        self.settings_dock.pitch_spin.setEnabled(not is_running)
        if not is_running:
            cancelled = any(b.engine.cancel_event.is_set() for b in self._backends.values())
            self.transport_dock.set_progress_value(0 if cancelled else 100)

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

        # The active character's voice on its own engine (the selected
        # clip's character, else the first), over the project defaults.
        state = dict(self.settings_dock.get_state())
        character = self.active_character()
        if character is not None:
            for key in ("voice", "speed", "lang_code"):
                if character.preset_data.get(key) not in (None, ""):
                    state[key] = character.preset_data[key]
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
        level = self.level
        if not level.document.clips:
            self.start_conversion()
            return

        dirty = level.document.dirty_clips()
        if not dirty:
            QMessageBox.information(self, "Up to date", "All clips are already generated.")
            return

        # Once a document has any clips, Generate always runs the
        # dirty-scoped batch path - even with JIT enabled (JIT has no
        # per-clip output shape; it stays reachable via the no-clips
        # fallback above). Stale subprojects follow, one at a time (phase 4).
        if any(not clip.is_nested for clip in dirty):
            self._nested_after_batch = level if any(clip.is_nested for clip in dirty) else None
            self.timeline_dock.generate_dirty_clips_requested(level)
        else:
            self.generate_stale_subprojects(level)

    def generate_clip(self, clip_id: str) -> None:
        """UI3: the gutter's per-clip play button and the timeline's
        context menu both land here. On a clip that is already clean the
        request means "regenerate": the engine bumps the clip's take and
        writes fresh audio under a new key instead of serving the cached
        file (grill TB8). A nested clip generates its subproject's stale
        clips and renders its mixdown."""
        project = self.project_of_clip_id(clip_id)
        nested = project.document.get_clip(clip_id) if project is not None else None
        if nested is not None and nested.is_nested:
            self.generate_subproject(nested)
            return
        clip = self.document.get_clip(clip_id)
        regenerate = clip is not None and clip not in self.document.dirty_clips()
        self.timeline_dock.on_generate_clip_requested(clip_id, regenerate=regenerate)

    def on_project_generated(self, project) -> None:
        """A generate finished and its results are in `project`'s document.
        A subproject's mixdown follows (phase 4); the root has none."""
        self._after_project_generated(project)

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
        project, self._nested_after_batch = self._nested_after_batch, None
        if project is not None:
            self.generate_stale_subprojects(project)

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

        gaps = plan_pause_gaps(self.document, triples)
        for start, end, character_id in triples:
            fields = {"gap_before_s": gaps[start]} if start in gaps else None
            self.document.undo_stack.push(AssignCharacterCommand(start, end, character_id, clip_fields=fields))

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
        for backend in self._backends.values():
            backend.engine.cancel()
        self.set_status("Cancelling... waiting for workers...", "warning")

    # --- transport / playhead (section 5) ----------------------------------

    def current_arrangement(self):
        if self._arrangement is None:
            self._arrangement = self.build_arrangement()
        return self._arrangement

    def _rebuild_transport_schedule(self) -> None:
        """The transport's schedule from the arrangement and the mix plan
        (`daw/mixplan.py`): muted and soloed-out tracks are left out; the
        track's gain, pan and automation and the clip's fades ride each
        entry. A clip's fade-in goes on its first segment and its fade-out
        on its last. A segment with a `range` becomes a sliced entry; a
        music bed plays its trim range, once per loop pass
        (`beds.playable_segments`). A clip on a ducked track carries `duck`,
        and speech carries `sidechain`, for the mixer's ducking."""
        level = self.level
        self._arrangement = self.build_arrangement(level)
        rate = self.project_sample_rate()
        mixes = clip_mixes(level.document, self._arrangement)
        schedule = []
        for placed in self._arrangement.placed:
            mix = mixes.get(placed.clip.id)
            if placed.estimated or mix is None:
                continue
            post_config = self.post_config_for_clip(placed.clip, level)
            if placed.clip.is_nested:
                # A subproject plays its mixdown (NP2).
                path = self.nested_audio_path(placed.clip, level)
                if path:
                    schedule.append(ScheduledClip(
                        clip_id=placed.clip.id, start_s=placed.start_s, path=path, post_config=post_config,
                        gain=mix.gain, pan=mix.pan, automation=mix.automation,
                        fade_in_s=mix.fade_in_s, fade_out_s=mix.fade_out_s,
                        duck=mix.duck, sidechain=mix.sidechain,
                    ))
                continue
            # One ScheduledClip per segment so multi-segment clips play
            # back to back at their real (rendered) offsets.
            offset = placed.start_s
            segments = playable_segments(placed.clip)
            for index, segment in enumerate(segments):
                range_s = post.segment_range(segment)
                schedule.append(ScheduledClip(
                    clip_id=placed.clip.id, start_s=offset, path=segment.audio_path, post_config=post_config,
                    gain=mix.gain, pan=mix.pan, automation=mix.automation,
                    fade_in_s=mix.fade_in_s if index == 0 else 0.0,
                    fade_out_s=mix.fade_out_s if index == len(segments) - 1 else 0.0,
                    slice=range_s, duck=mix.duck, sidechain=mix.sidechain,
                ))
                try:
                    offset += post.rendered_duration_s(segment.audio_path, post_config, rate,
                                                       hint=post.duration_hint(segment, post_config),
                                                       range_s=range_s)
                except Exception:
                    offset += segment.duration or 0.0
        self.transport.load(schedule, sample_rate=self.project_sample_rate(),
                            total_duration_s=self._arrangement.total_duration_s,
                            duck_db=duck_db_setting(level.document))
        if self.timeline_dock is not None:
            self.timeline_dock.timeline_view.set_arrangement(self._arrangement)
        self.transport_dock.set_position(self.transport.position(), self.transport.duration())

    def _on_transport_position(self, seconds: float) -> None:
        self.transport_dock.set_position(seconds, self.transport.duration())
        if self.timeline_dock is not None:
            self.timeline_dock.timeline_view.set_playhead(seconds)
        arrangement = self.current_arrangement()
        playing = None
        word = None
        if self.transport.is_playing:
            hits = arrangement.at_time(seconds)
            # The line being read, not the music under it.
            hits = [h for h in hits if not h.clip.is_bed] or hits
            playing = hits[0].clip.id if hits else None
            if hits:
                word = self.word_at(hits[0], seconds)
        self.selection.set_playing_clip(playing)
        if self.editor is not None:
            # The transcript shows the focus; the transport plays the level.
            self.editor.set_playing_word(word if self.focus is self.level else None)

    def word_at(self, placed, seconds: float):
        """`(start, end)` document offsets of the word the playhead is on
        inside `placed`, or None. Segment by cumulative duration, word by
        time, then the word's offset in the clip's spoken text mapped back
        through the lexicon (`apply_lexicon(..., with_spans=True)`) to the
        transcript."""
        for segment, seg_start, scale in segment_timeline(placed):
            words = segment.words or []
            seg_end = seg_start + float(segment.duration or 0.0) * scale
            if not words or not (seg_start <= seconds < seg_end):
                continue
            local = (seconds - seg_start) / scale if scale else 0.0
            index = next((i for i, w in enumerate(words) if w[1] <= local < w[2]), None)
            if index is None:
                return None
            return self._word_offsets(placed.clip, segment, index)
        return None

    def _word_offsets(self, clip, segment, word_index: int):
        """Document offsets of word `word_index` of `segment`: the n-th
        whitespace word of the segment's text, found in the clip's spoken
        text from where the segment's text starts."""
        from kokoro_gui.engine.lexicon import apply_lexicon, original_span

        document = self.level.document
        extent = document.clip_extent(clip.id)
        if extent is None:
            return None
        clip_text = document.clip_text(clip)
        spoken, spans = apply_lexicon(clip_text, self.settings.get("lexicon", {}), with_spans=True)
        seg_words = (segment.text or "").split()
        if word_index >= len(seg_words):
            return None
        cursor = spoken.find(seg_words[0]) if seg_words else -1
        cursor = max(cursor, 0)
        for i, token in enumerate(seg_words):
            found = spoken.find(token, cursor)
            if found < 0:
                return None
            if i == word_index:
                start, end = original_span(spans, found, found + len(token))
                return extent[0] + start, extent[0] + max(end, start + 1)
            cursor = found + len(token)
        return None

    def seek_to_offset(self, offset: int) -> bool:
        """Ctrl+click in the transcript: seek the transport to the word at
        document `offset` (the clip's start when it has no word times).
        False when the offset isn't inside a placed clip, or when the
        transcript shows a subproject the transport isn't playing."""
        if self.focus is not self.level:
            return False
        clip = self.document.clip_covering(offset)
        if clip is None:
            return False
        placed = self.current_arrangement().by_clip_id().get(clip.id)
        if placed is None:
            return False
        target = placed.start_s
        for segment, seg_start, scale in segment_timeline(placed):
            for index, word in enumerate(segment.words or []):
                span = self._word_offsets(clip, segment, index)
                if span is not None and span[0] <= offset < span[1]:
                    target = seg_start + float(word[1]) * scale
                    self.transport.seek(target)
                    return True
        self.transport.seek(target)
        return True

    # --- word alignment (phase 2, C1) ------------------------------------------

    def _engine_has_word_timing(self, clip) -> bool:
        """True when generation already stamps word times for `clip`:
        Kokoro and Dummy in English (KPipeline's other languages yield no
        tokens), never Audio8."""
        backend = self.backend_for(clip)
        if not getattr(backend.capabilities, "supports_word_timing", False):
            return False
        if backend.id != "kokoro":
            return True
        return self._assemble_generation_config(clip).get("lang_code") in ("a", "b")

    def schedule_word_alignment(self, clip_ids, force: bool = False) -> bool:
        """Runs Whisper over each listed clip's segments that have no word
        times, on a worker thread, and stores the aligned words
        (`daw/wordalign.py`). Automatic after a generate only for engines
        without token timing; `force` (the timeline's "Align words") runs
        it for any clip and re-aligns segments that have words. Never
        dirties a clip: words aren't a generation input. False when nothing
        was scheduled."""
        from kokoro_gui.qt import asr_prompt

        if self._word_align_thread is not None:
            return False
        jobs = []
        for clip_id in clip_ids:
            project = self.project_of_clip_id(clip_id)
            clip = project.document.get_clip(clip_id) if project is not None else None
            if clip is None or clip.is_nested or (not force and self._engine_has_word_timing(clip)):
                continue
            for segment in clip.segments:
                if segment.audio_path and (force or not segment.words):
                    jobs.append((clip.id, segment.id, segment.audio_path, segment.text))
        if not jobs or (self._word_align_declined and not force):
            return False
        choice, _downloading = asr_prompt.confirm_whisper_download(self)
        if choice != asr_prompt.PROCEED:
            self._word_align_declined = True
            return False

        def _work():
            from kokoro_gui.engine import asr

            out = []
            for done, (clip_id, segment_id, path, text) in enumerate(jobs, start=1):
                self._wordAlignProgress.emit(done, len(jobs))
                try:
                    heard = asr.transcribe_wav_words(path, "whisper")
                except Exception:  # noqa: BLE001 - one bad file skips one segment
                    continue
                out.append((clip_id, segment_id, wordalign.align(text, heard)))
            self._wordsAligned.emit(out)

        self._word_align_thread = threading.Thread(target=_work, name="word-align", daemon=True)
        self._word_align_thread.start()
        return True

    def wait_for_word_alignment(self, timeout_s: float = 30.0) -> None:
        """Test hook: blocks until the alignment thread finishes and its
        result has been applied."""
        thread = self._word_align_thread
        if thread is not None:
            thread.join(timeout_s)
        QApplication.processEvents()

    def _on_word_align_progress(self, done: int, total: int) -> None:
        self.set_status(f"Aligning words {done}/{total}", "busy")

    def _on_words_aligned(self, results) -> None:
        self._word_align_thread = None
        by_id = {}
        for clip in (c for project in self.projects.values() for c in project.document.clips):
            for segments in (clip.segments, *clip.takes.values()):
                for segment in segments:
                    by_id[segment.id] = segment
        applied = 0
        for _clip_id, segment_id, words in results:
            segment = by_id.get(segment_id)
            if segment is not None:
                segment.words = words
                applied += 1
        if applied:
            self.schedule_save()
        self.set_status(f"Aligned words for {applied} segment(s).", "success" if applied else "info")

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
        """Grill TB12: a dirty project asks Save / Discard / Cancel. Save runs
        in the background and closes the window when it succeeds; a failure
        keeps the window open with the error. Then close-time GC, and
        eviction of every project dir but the one launch resumes (TB13)."""
        try:
            self.transport.stop()
        except Exception:
            pass
        if self._closed:
            super().closeEvent(event)
            return
        if self._io_thread is not None and not self._closing_after_save:
            event.ignore()
            return
        self.save_settings()
        if self.root.project_dir and not self._closing_after_save and self.any_project_dirty():
            choice = self._ask_close_choice()
            if choice == "cancel":
                event.ignore()
                return
            if choice == "save":
                path = self.project_path or self._save_as_path_dialog()
                if not path:
                    event.ignore()
                    return
                if not self.project_path:
                    self.project_path = path
                    project_io.remember_recent(self.settings, path)
                event.ignore()

                def _then():
                    self._closing_after_save = True
                    self.close()

                self._save_bundle(path, then=_then)
                return
            self._teardown_project(discard=True)
        keep = None
        last = self.settings.get("last_project")
        if self.root.project_dir and last and self.project_path and os.path.abspath(last) == self.project_path:
            keep = self.root.project_dir
        if self.root.project_dir:
            self._teardown_project(discard=False)
        try:
            project_io.evict_project_dirs(keep)
        except OSError:
            pass
        qt_settings.save_settings(CONFIG_FILE, self.settings)
        self._closed = True
        for engine_id, backend in self._backends.items():
            if engine_id == self._primary_engine_id:
                continue
            try:
                backend.engine.worker.stop()
            except Exception:
                pass
        super().closeEvent(event)
