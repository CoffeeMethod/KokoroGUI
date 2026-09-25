"""Subprojects (phase 4, grill NP1-NP8): the app half.

A subproject is a `.tbaw` placed on another one's timeline as a nested clip
(`Clip.source == "nested"`, `Clip.child`). `SubprojectsMixin` gives
`QtTTSApp` the tree of open projects (`root`, `children`, `focus`,
`level`, kokoro_gui/qt/open_projects.py) and what it takes to work in it:

- Opening a child lazily, the first time its block is selected or entered,
  through the same steps a root takes (`inspect_bundle`,
  `choose_project_dir`, the lock, extraction on the worker, `finish_open`).
  An embedded child's bundle is `<parent project dir>/projects/<id>.tbaw`;
  its session names its source `<parent source>#<id>`. A child made this
  session has only a project dir and opens from it (`load_project_dir`).
  One that can't be opened (a linked file gone, the wrong project, a lock
  held elsewhere) is "missing": its block paints crossed out and offers
  Relink.
- Saving: an open, dirty embedded child writes its bundle into its
  parent's project dir before the parent is planned (deepest first), so
  the parent's `plan_save` carries the new bytes; a linked child saves to
  its own file.
- Autosave, the dirty flag and close-time teardown over every open project.

Kept out of app.py, which already sequences the root's Open and Save.
"""
from __future__ import annotations

import os

from PySide6.QtWidgets import QMessageBox

from kokoro_gui.daw import library as character_library
from kokoro_gui.qt import project as project_io
from kokoro_gui.qt.open_projects import OpenProject


class ParentStore:
    """A character store over the root document (NP3): `get(library_id)`
    answers from the root's characters whose `library_id` matches, so a
    subproject's characters linked at project scope follow the book's."""

    def __init__(self, document):
        self.document = document

    def get(self, library_id):
        if not library_id:
            return None
        return next((c for c in self.document.characters if c.library_id == library_id), None)


class SubprojectsMixin:
    """Mixed into `QtTTSApp`; expects `root`, `children`, `focus`, `level`,
    `_backend_for`, `_run_project_io` and the rest of the app."""

    # -- the tree ------------------------------------------------------------------

    def parent_of(self, project):
        if project is None or project.parent_id is None:
            return None
        if self.root.project_id == project.parent_id:
            return self.root
        return self.children.get(project.parent_id)

    def depth_of(self, project) -> int:
        depth = 0
        while project is not None and project.parent_id is not None:
            project = self.parent_of(project)
            depth += 1
        return depth

    def chain_of(self, project) -> list:
        """`[root, ..., project]`: the breadcrumb."""
        chain = []
        while project is not None:
            chain.append(project)
            project = self.parent_of(project)
        return list(reversed(chain))

    def source_of(self, project):
        """What `session.json`'s `source_path` names for `project`: the root's
        file, a linked child's file, `<parent source>#<id>` for an embedded
        child (None while the root is Untitled)."""
        if project is None:
            return None
        if project.parent_id is None or (project.path and project.kind == "linked"):
            return project.path
        return project_io.child_source_path(self.source_of(self.parent_of(project)), project.project_id)

    @staticmethod
    def child_id_of(clip):
        child = clip.child if clip is not None and clip.is_nested and isinstance(clip.child, dict) else None
        return project_io.safe_child_id(child.get("id")) if child else None

    def child_project(self, clip):
        child_id = self.child_id_of(clip)
        return self.children.get(child_id) if child_id else None

    def is_child_missing(self, clip) -> bool:
        return self.child_id_of(clip) in self._missing_children

    def _parent_file(self, parent):
        """The `.tbaw` a linked path is relative to: the nearest project up
        the chain that has a file of its own."""
        while parent is not None:
            if parent.path:
                return parent.path
            parent = self.parent_of(parent)
        return None

    def child_bundle_path(self, parent, clip):
        """Where the child's bundle is now: in `parent`'s project dir for an
        embedded child, the linked file (relative to the parent's file) for
        a linked one. None when that can't be said."""
        child = clip.child if isinstance(clip.child, dict) else {}
        if child.get("kind") == "linked":
            path = child.get("path")
            if not isinstance(path, str) or not path:
                return None
            if os.path.isabs(path):
                return os.path.normpath(path)
            base = self._parent_file(parent)
            if not base:
                return None
            return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(base)), path))
        return project_io.embedded_child_path(parent.project_dir, child.get("id"))

    # -- opening a child -----------------------------------------------------------

    def open_child(self, clip, then=None):
        """The child project for nested `clip`, opened if it isn't yet. Returns
        it when it's ready now; when its audio has to be extracted first,
        returns None and calls `then(child)` once it is (`then(None)` on
        failure, with the block marked missing)."""
        existing = self.child_project(clip)
        if existing is not None:
            if then is not None:
                then(existing)
            return existing
        child_id = self.child_id_of(clip)
        parent = self.project_for(clip)
        if child_id is None or parent is None or not parent.project_dir:
            return self._child_failed(clip, "The subproject reference is broken.", then)
        kind = (clip.child or {}).get("kind", "embedded")
        bundle = self.child_bundle_path(parent, clip)
        source = bundle if kind == "linked" else project_io.child_source_path(self.source_of(parent), child_id)
        project_dir = project_io.choose_project_dir(child_id, source)
        try:
            lock = project_io.ProjectLock(project_dir).acquire()
        except project_io.ProjectLockedError as e:
            return self._child_failed(clip, str(e), then)

        session = project_io.read_session(project_dir)
        has_document = os.path.isfile(os.path.join(project_dir, project_io.DOCUMENT))
        info = None
        if bundle and os.path.isfile(bundle):
            try:
                info = project_io.inspect_bundle(bundle)
            except project_io.ProjectError as e:
                lock.release()
                return self._child_failed(clip, str(e), then)
            if info.project_id != child_id:
                lock.release()
                return self._child_failed(
                    clip, f"{os.path.basename(bundle)} is project {info.project_id}, not {child_id}.", then)

        # The dir is this child's and usable as it is: ahead of the bundle
        # (dirty), made this session (no bundle yet), or a clean extraction
        # of the bundle as it is now.
        if has_document and (info is None or (session and (session.get("dirty")
                                                           or project_io.session_matches_file(session, info)))):
            loaded = project_io.load_project_dir(project_dir, child_id, info.manifest if info else {})
            dirty = bool(session and session.get("dirty"))
            child = self._attach_child(parent, clip, loaded, lock, bundle if kind == "linked" else None, dirty)
            if then is not None:
                then(child)
            return child
        if info is None:
            lock.release()
            what = bundle or "the subproject's file"
            return self._child_failed(clip, f"Couldn't find {what}.", then)

        project_io.wipe_project_dir(project_dir)
        try:
            project_io.check_free_space(project_io.projects_root(), info.audio_bytes, "open the subproject")
            project_io.extract_small(info, project_dir)
        except (project_io.ProjectError, OSError) as e:
            lock.release()
            return self._child_failed(clip, str(e), then)

        def _finish():
            try:
                loaded = project_io.finish_open(info, project_dir, self._engine_versions(info.manifest),
                                                source_path=source)
            except (OSError, ValueError, KeyError) as e:
                lock.release()
                return self._child_failed(clip, f"Couldn't read the subproject: {e}", then)
            child = self._attach_child(parent, clip, loaded, lock, bundle if kind == "linked" else None, False)
            if then is not None:
                then(child)
            return child

        if info.audio_bytes == 0:
            return _finish()

        self._begin_project_io("Opening subproject...", read_only=False)

        def _work():
            project_io.extract_audio(info, project_dir, progress=self._io_progress("Extracting subproject"))

        def _done(_result, error):
            self._end_project_io(read_only=False)
            if error is not None:
                lock.release()
                self._child_failed(clip, str(error), then)
                return
            _finish()

        self._run_project_io(_work, _done)
        return None

    def _engine_versions(self, manifest: dict) -> dict:
        versions = {}
        for engine_id in [*self._backends, *((manifest or {}).get("engines") or {})]:
            backend = self._backend_for(engine_id)
            if backend is not None:
                versions[backend.id] = backend.engine_version()
        return versions

    def _child_failed(self, clip, message: str, then):
        child_id = self.child_id_of(clip)
        if child_id:
            self._missing_children.add(child_id)
        self.set_status(f"Subproject unavailable: {message}", "warning")
        self.refresh_timeline()
        if (clip.child or {}).get("kind") == "linked" and not getattr(self, "_offering_relink", False):
            # NP4: a missing or wrong linked file asks for the right one.
            self._offering_relink = True
            try:
                answer = QMessageBox.question(self, "Subproject not found",
                                              f"{message}\nFind the subproject's file?")
                if answer == QMessageBox.StandardButton.Yes:
                    self.relink_subproject_dialog(clip)
            finally:
                self._offering_relink = False
        if then is not None:
            then(self.child_project(clip))
        return self.child_project(clip)

    def _attach_child(self, parent, clip, loaded, lock, path, dirty: bool) -> OpenProject:
        child = OpenProject(
            document=loaded.document, project_dir=loaded.project_dir, project_id=self.child_id_of(clip), lock=lock,
            project_settings=dict(loaded.project_settings or {}), path=path, parent_id=parent.project_id,
            clip_id=clip.id, manifest=dict(loaded.manifest or {}), dirty=dirty,
            kind=(clip.child or {}).get("kind", "embedded"),
        )
        self.children[child.project_id] = child
        self._missing_children.discard(child.project_id)
        self._closed_child_states.pop(child.project_id, None)
        self._install_segment_key_fn(child)
        self._install_nested_state_fn(child)
        self._resolve_child_characters(child)
        for engine_id in self._document_engine_ids(child.document):
            self._ensure_backend_ready(self._backend_for(engine_id))
        # The placeholder shows the child's title as the child names itself.
        title = child.title()
        if parent.document.clip_text(clip) != title:
            parent.document.set_placeholder_text(clip.id, title)
        for notice in loaded.notices:
            self.set_status(notice, "warning")
        return child

    def _resolve_child_characters(self, child) -> None:
        """NP3: a child's linked characters look in the root's characters
        first (project scope), then the global library."""
        report = character_library.resolve_characters(
            child.document, [ParentStore(self.root.document), self.character_library])
        self.library_missing |= set(report.missing)

    def _install_nested_state_fn(self, project) -> None:
        """`Document.nested_state_fn` for `project`'s document: a nested clip
        is stale unless its child's mixdown is current (NP2)."""
        project.document.nested_state_fn = lambda clip: self.nested_state(clip, project) != "ok"

    # -- subproject mixdown and staleness (NP2) -------------------------------------

    def _closed_child_dir(self, parent, clip):
        """The project dir a not-yet-open child would open into, when one
        exists already (a previous session's)."""
        child_id = self.child_id_of(clip)
        if child_id is None:
            return None
        kind = (clip.child or {}).get("kind", "embedded")
        source = self.child_bundle_path(parent, clip) if kind == "linked" else \
            project_io.child_source_path(self.source_of(parent), child_id)
        project_dir = project_io.choose_project_dir(child_id, source)
        return project_dir if os.path.isfile(os.path.join(project_dir, project_io.DOCUMENT)) else None

    def child_state(self, child) -> str:
        """"ok" when the child has no stale clip and its mixdown was
        rendered from its document as it is now; else "stale". Cached until
        the next edit (`schedule_save` drops every child's cache)."""
        if child.state_cache is None:
            info = project_io.read_mixdown_info(child.project_dir)
            if child.digest is None:
                child.digest = project_io.project_digest(child.document, child.project_settings, child.project_dir)
            fresh = info is not None and info.get("digest") == child.digest
            child.state_cache = "ok" if fresh and not child.document.dirty_clips() else "stale"
        return child.state_cache

    def nested_state(self, clip, project=None) -> str:
        """"ok", "stale" or "missing" for a nested clip. A child that isn't
        open is "ok" when its project dir from an earlier session holds a
        mixdown of its document as last saved there."""
        if clip is None or not clip.is_nested:
            return "ok"
        child = self.child_project(clip)
        if child is not None:
            return self.child_state(child)
        if self.is_child_missing(clip):
            return "missing"
        # A closed child's dir can't change until it's opened or rendered,
        # so its answer is kept (dropped by `open_child` and teardown).
        child_id = self.child_id_of(clip)
        cached = self._closed_child_states.get(child_id)
        if cached is not None:
            return cached
        parent = project or self.project_for(clip)
        project_dir = self._closed_child_dir(parent, clip)
        info = project_io.read_mixdown_info(project_dir)
        session = project_io.read_session(project_dir) if project_dir else None
        state = "ok" if (info and session and not session.get("dirty")
                         and info.get("digest") == session.get("saved_digest")) else "stale"
        self._closed_child_states[child_id] = state
        return state

    def invalidate_child_states(self) -> None:
        for child in self.children.values():
            child.state_cache = None

    def nested_audio_path(self, clip, project=None):
        """The child's mixdown file for a nested clip (the last one rendered,
        current or not, the way a stale clip still plays its old audio), or
        None."""
        child = self.child_project(clip)
        if child is not None:
            project_dir = child.project_dir
        else:
            project_dir = self._closed_child_dir(project or self.project_for(clip), clip)
        info = project_io.read_mixdown_info(project_dir)
        return info["file"] if info else None

    def nested_duration_s(self, clip, project=None):
        child = self.child_project(clip)
        if child is not None:
            project_dir = child.project_dir
        else:
            project_dir = self._closed_child_dir(project or self.project_for(clip), clip)
        info = project_io.read_mixdown_info(project_dir)
        return info["duration_s"] if info else None

    def nested_estimate_s(self, clip):
        """An unrendered subproject's estimated length: its open document's
        arrangement, so the block is sized by its content rather than its
        title. None for any other clip, or a child that isn't open."""
        if clip is None or not clip.is_nested:
            return None
        child = self.child_project(clip)
        if child is None:
            return None
        return self.build_arrangement(child).total_duration_s

    def nested_post_config(self, clip, project=None) -> dict:
        """What the parent applies over a child's mixdown: nothing but FX
        the parent set on the nested clip itself. The mixdown already has
        the child's volume, trim, normalize and FX; applying the project's
        again would double them (PHASE_4, Risks)."""
        from kokoro_gui.audio import post
        from kokoro_gui.qt import fx_resolve

        own = fx_resolve.real_preset_name(clip.overrides.get("fx_preset"))
        if not clip.fx_override and not own:
            return {}
        values = {}
        preset = fx_resolve.load_fx_preset_values(self, own, project or self.project_for(clip)) if own else None
        if preset:
            values.update(preset)
        if clip.fx_override:
            values.update(clip.fx_override)
        values["apply_fx"] = True
        return post.extract_post_config(values)

    def render_subproject(self, child, then=None) -> bool:
        """Renders `child`'s mixdown on the worker (the same `mixdown()`
        Export runs, its own subprojects as their mixdowns). Refused while
        the child has stale clips. `then(ok)` runs on the GUI thread."""
        from kokoro_gui.daw.mixdown import mixdown

        if child is None or child.parent_id is None:
            return False
        if [c for c in child.document.dirty_clips() if not c.is_nested]:
            self.set_status(f"{child.title()} has clips to generate first.", "warning")
            return False
        self._autosave_one(child)
        digest = child.digest
        arrangement = self.build_arrangement(child)
        post_configs = {p.clip.id: self.post_config_for_clip(p.clip, child) for p in arrangement.placed}
        nested_paths = {p.clip.id: self.nested_audio_path(p.clip, child)
                        for p in arrangement.placed if p.clip.is_nested}
        rate = self.project_sample_rate()
        fmt = project_io.bundle_options(child.project_settings)["audio_format"]
        target = project_io.mixdown_file(child.project_dir, fmt)
        tmp = os.path.join(child.project_dir, f"{project_io.MIXDOWN}.tmp.{fmt}")
        document = child.document
        self._begin_project_io(f"Rendering {child.title()}...", read_only=False)

        def _work():
            result = mixdown(document, tmp, fmt, rate, arrangement=arrangement,
                             post_config_for_clip=lambda clip: post_configs.get(clip.id),
                             nested_audio_path=lambda clip: nested_paths.get(clip.id))
            os.replace(tmp, target)
            return result

        def _done(result, error):
            self._end_project_io(read_only=False)
            if error is not None:
                self.set_status(f"Rendering {child.title()} failed: {error}", "error")
                if then is not None:
                    then(False)
                return
            project_io.write_mixdown_info(child.project_dir, target, digest, result.duration_s, rate)
            child.mixdown_path, child.mixdown_digest = target, digest
            child.state_cache = None
            self.set_status(f"Rendered {child.title()}.", "success")
            parent = self.parent_of(child)
            if parent is not None and parent.parent_id is not None:
                parent.state_cache = None
            if self.editor is not None:
                self.editor.rehighlight()
            self.refresh_timeline()
            if then is not None:
                then(True)

        self._run_project_io(_work, _done)
        return True

    def generate_subproject(self, clip, then=None) -> None:
        """A stale nested clip's play button: open the child, generate its
        stale clips, then render its mixdown (NP2)."""
        self._subproject_queue.append((clip, then))
        if len(self._subproject_queue) == 1 and not self.is_busy():
            self._advance_subproject_queue()

    def generate_stale_subprojects(self, project) -> int:
        """Queues every stale nested clip of `project`'s document."""
        stale = [c for c in project.document.nested_clips() if self.nested_state(c, project) == "stale"]
        for clip in stale:
            self._subproject_queue.append((clip, None))
        if stale and not self.is_busy():
            self._advance_subproject_queue()
        return len(stale)

    def _advance_subproject_queue(self) -> None:
        if not self._subproject_queue or self.is_busy():
            return
        clip, then = self._subproject_queue[0]

        def _finished(ok):
            self._subproject_queue.pop(0)
            if then is not None:
                then(ok)
            self._advance_subproject_queue()

        def _opened(child):
            if child is None:
                _finished(False)
                return
            stale = [c for c in child.document.dirty_clips() if not c.is_nested]
            if stale:
                self._pending_render_after_generate[child.project_id] = _finished
                self.timeline_dock.generate_dirty_clips_requested(child)
                return
            nested_stale = [c for c in child.document.nested_clips() if self.nested_state(c, child) == "stale"]
            if nested_stale:
                # Grandchildren first, then this child again.
                self._subproject_queue[1:1] = [(c, None) for c in nested_stale]
                self._subproject_queue.append((clip, then))
                self._subproject_queue.pop(0)
                self._advance_subproject_queue()
                return
            if not self.render_subproject(child, then=_finished):
                _finished(False)

        self.open_child(clip, then=_opened)

    def _after_project_generated(self, project) -> None:
        """A generate finished in `project`: a child with nothing stale left
        renders its mixdown (the queue's own step, or on its own)."""
        if project.parent_id is None:
            return
        project.state_cache = None
        waiting = self._pending_render_after_generate.pop(project.project_id, None)
        if [c for c in project.document.dirty_clips() if not c.is_nested]:
            if waiting is not None:
                waiting(False)
            return
        if not self.render_subproject(project, then=waiting) and waiting is not None:
            waiting(False)

    # -- focus: what the transcript, Settings and FX docks show (NP1) --------------

    def scope_text(self):
        """"Subproject: <title>" while the docks show a subproject the
        timeline isn't in, else None."""
        if self.focus is self.level:
            return None
        return f"Subproject: {self.focus.title()}"

    def set_focus(self, project) -> None:
        """Points the transcript, Settings and Audio FX docks at `project`'s
        document. Undo follows (each document has its own stack)."""
        if project is None or project is self.focus:
            return
        self.focus = project
        self.selection.project_id = project.project_id
        self._refresh_focus_docks()

    def _refresh_focus_docks(self) -> None:
        self._focus_switching = True
        try:
            editor = self.editor
            if editor is not None:
                editor._updating_from_model = True
                try:
                    editor.rebind_document()
                finally:
                    editor._updating_from_model = False
            for backend in list(self._backends.values()):
                backend.on_project_opened(self.project_dir, self._engine_meta(backend.id))
            if self.transcript_dock is not None:
                self.transcript_dock.refresh_character_choices()
                self.transcript_dock.refresh_scope()
            if self.settings_dock is not None:
                self.settings_dock._build_for_selection()
            if self.fx_dock is not None:
                self.fx_dock.refresh_for_selection()
            self._on_active_backend_maybe_changed(force=True)
        finally:
            self._focus_switching = False

    def _on_selection_for_focus(self) -> None:
        """NP1: selecting a nested block (in the timeline, or its placeholder
        line in the transcript) points the docks at its child, opening it
        the first time; selecting another clip points them at the clip's
        own project; a lane label at the level. A text range or an empty
        selection leaves the focus where it is, so editing inside a
        subproject stays there."""
        if getattr(self, "_focus_switching", False):
            return
        selection = self.selection
        if selection.kind == "clip":
            owner = self.project_of_clip_id(selection.selected_clip_id)
            if owner is None:
                return
            selection.project_id = owner.project_id
            clip = owner.document.get_clip(selection.selected_clip_id)
            if clip is not None and clip.is_nested:
                self.note_nested_selected(clip.id)
                if self.is_child_missing(clip) and self.child_project(clip) is None:
                    self.set_focus(owner)
                    return
                self.open_child(clip, then=lambda child: self.set_focus(child or owner))
                return
            self.set_focus(owner)
        elif selection.kind == "character":
            selection.project_id = self.level.project_id
            self.set_focus(self.level)

    def rename_subproject(self, child, title: str) -> None:
        """Sets a subproject's title (`project_settings["title"]`, in its
        own `project.json`) and rewrites its placeholder line in the parent."""
        title = (title or "").strip()
        if child is None or child.parent_id is None or not title or title == child.title():
            return
        child.project_settings["title"] = title
        parent = self.parent_of(child)
        if parent is not None and child.clip_id:
            parent.document.set_placeholder_text(child.clip_id, child.title())
            if parent is self.focus and self.editor is not None:
                self.editor.load_text(parent.document.text)
        if self.transcript_dock is not None:
            self.transcript_dock.refresh_scope()
        self.schedule_save()
        self.refresh_timeline()

    def on_subproject_action(self, clip_id: str, action: str) -> None:
        """A nested block's context menu (or double-click, "enter")."""
        project = self.project_of_clip_id(clip_id)
        clip = project.document.get_clip(clip_id) if project is not None else None
        if clip is None or not clip.is_nested:
            return
        if action == "render":
            self.generate_subproject(clip)
        elif action == "enter":
            self.enter_subproject(clip)
        elif action == "relink":
            self.relink_subproject_dialog(clip)
        elif action == "detach":
            self.detach_subproject_dialog(clip)
        elif action == "embed":
            self.embed_subproject(clip)
        elif action == "remove":
            self.remove_subproject(clip)

    # -- level: what the timeline and transport show (NP5) --------------------------

    def set_level(self, project) -> None:
        """The timeline and transport show `project`; the docks follow.
        The breadcrumb is `chain_of(level)`."""
        if project is None:
            return
        if project is self.level and project is self.focus:
            return
        self.transport.stop()
        self.level = project
        self._focus_switching = True
        try:
            self.selection.clear()
        finally:
            self._focus_switching = False
        self._set_focus_forced(project)
        if self.timeline_dock is not None:
            self.timeline_dock.refresh_breadcrumb()
        self.refresh_timeline()
        self._rebuild_transport_schedule()

    def _set_focus_forced(self, project) -> None:
        self.focus = project
        self.selection.project_id = project.project_id
        self._refresh_focus_docks()

    def enter_subproject(self, clip) -> None:
        """Double-click on a nested block or its placeholder: the timeline
        shows the child (opened if needed)."""
        self.open_child(clip, then=lambda child: self.set_level(child) if child is not None else None)

    def note_nested_selected(self, clip_id) -> None:
        import time

        self._last_nested_selected = (clip_id, time.monotonic())

    def enter_recently_selected_subproject(self, window_s: float = 0.8) -> bool:
        """The transcript's double-click on a placeholder line: its first
        click already put the child in the docks; the second enters it."""
        import time

        last = getattr(self, "_last_nested_selected", None)
        if not last or time.monotonic() - last[1] > window_s:
            return False
        project = self.project_of_clip_id(last[0])
        clip = project.document.get_clip(last[0]) if project is not None else None
        if clip is None or not clip.is_nested:
            return False
        self._last_nested_selected = None
        self.enter_subproject(clip)
        return True

    # -- linked children, relink, detach, embed, remove (NP4) ---------------------

    def _linked_path_for(self, parent, file_path: str) -> str:
        """`file_path` as `Clip.child["path"]`: relative to the parent's
        file when there is one on the same drive, else absolute."""
        file_path = os.path.abspath(file_path)
        base = self._parent_file(parent)
        if base:
            base_dir = os.path.dirname(os.path.abspath(base))
            if os.path.splitdrive(base_dir)[0].lower() == os.path.splitdrive(file_path)[0].lower():
                return os.path.relpath(file_path, base_dir).replace("\\", "/")
        return file_path

    def add_subproject(self, path: str, position: int | None = None):
        """File > Add Subproject...: links an existing `.tbaw` into the focus
        project as a nested clip (NP4: "add an existing project" links).
        Returns the child, or None when refused."""
        from kokoro_gui.daw.models import _new_id
        from kokoro_gui.daw.undo import ReplaceWithNestedCommand

        parent = self.focus
        try:
            info = project_io.inspect_bundle(path)
        except project_io.ProjectError as e:
            QMessageBox.warning(self, "Add subproject", str(e))
            return None
        tree_ids = {p.project_id for p in self.open_projects()}
        tree_ids |= {self.child_id_of(c) for p in self.open_projects() for c in p.document.nested_clips()}
        if info.project_id in tree_ids:
            QMessageBox.warning(self, "Add subproject",
                                f"{os.path.basename(path)} (project {info.project_id}) is already in this project.")
            return None
        document = parent.document
        position = len(document.text) if position is None else max(0, min(int(position), len(document.text)))
        loaded = project_io.load_project(path)
        title = project_io.display_title(loaded.project_settings if loaded else {}, path, "Subproject")
        child_ref = {"kind": "linked", "id": info.project_id, "path": self._linked_path_for(parent, path)}
        clip_id = _new_id()
        document.undo_stack.push(ReplaceWithNestedCommand(position, position, child_ref, title, clip_id))
        if parent is self.focus and self.editor is not None:
            self.editor.load_text(document.text)
        clip = document.get_clip(clip_id)
        child = self.open_child(clip)
        self.wait_for_project_io()
        child = child or self.child_project(clip)
        self.on_characters_changed()
        return child

    def add_subproject_dialog(self):
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(self, "Add subproject", "", "KokoroGUI project (*.tbaw)")
        if path:
            editor = self.editor
            position = None
            if editor is not None:
                cursor = editor.textCursor()
                position = cursor.block().position() + cursor.block().length() - 1
            return self.add_subproject(path, position)
        return None

    def relink_subproject(self, clip, path: str) -> bool:
        """Points a nested clip at `path`, refused unless the file's
        `project_id` is the one the clip names."""
        from kokoro_gui.daw.undo import SetFieldCommand

        child_id = self.child_id_of(clip)
        try:
            info = project_io.inspect_bundle(path)
        except project_io.ProjectError as e:
            QMessageBox.warning(self, "Relink", str(e))
            return False
        if info.project_id != child_id:
            QMessageBox.warning(self, "Relink", f"{os.path.basename(path)} is project {info.project_id}; "
                                                f"this subproject is {child_id}.")
            return False
        parent = self.project_for(clip)
        new_child = dict(clip.child or {})
        new_child.update({"kind": "linked", "path": self._linked_path_for(parent, path)})
        parent.document.undo_stack.push(SetFieldCommand("clip", clip.id, "child", new_child))
        self._missing_children.discard(child_id)
        self.open_child(clip)
        self.wait_for_project_io()
        self.refresh_timeline()
        return True

    def relink_subproject_dialog(self, clip) -> bool:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(self, "Relink subproject", "", "KokoroGUI project (*.tbaw)")
        return bool(path) and self.relink_subproject(clip, path)

    def _write_child_bundle(self, child, target: str) -> None:
        """Writes `child`'s bundle to `target` now (on this thread) and
        records it in the child's session."""
        plan, _warnings = project_io.plan_save(
            child.document, child.project_settings, target, child.project_dir, child.project_id,
            self._backend_for, self._fx_presets_dir(), project_io.read_session(child.project_dir), child.manifest,
        )
        import kokoro_gui.engines.registry as engine_registry

        result = project_io.write_bundle(plan, list(engine_registry.list_engines()))
        return plan, result

    @staticmethod
    def _fx_presets_dir() -> str:
        import kokoro_gui.qt.app as qt_app_module

        return qt_app_module.FX_PRESETS_DIR

    def detach_subproject(self, clip, path: str) -> bool:
        """Detach to file...: the embedded child is written to `path` and the
        clip links to it from now on (an undoable parent edit). The child's
        project dir stays where it is (it's keyed by id)."""
        from kokoro_gui.daw.undo import SetFieldCommand

        child = self.child_project(clip) or self.open_child(clip)
        self.wait_for_project_io()
        child = child or self.child_project(clip)
        if child is None:
            return False
        path = os.path.abspath(project_io.bundle_path_for(path))
        self._autosave_one(child)
        plan, result = self._write_child_bundle(child, path)
        project_io.record_save(child.project_dir, path, result)
        child.kind, child.path, child.dirty, child.manifest = "linked", path, False, dict(plan.manifest)
        parent = self.project_for(clip)
        new_child = {"kind": "linked", "id": child.project_id, "path": self._linked_path_for(parent, path)}
        parent.document.undo_stack.push(SetFieldCommand("clip", clip.id, "child", new_child))
        self.schedule_save()
        self.refresh_timeline()
        self.set_status(f"{child.title()} is now {os.path.basename(path)}.")
        return True

    def detach_subproject_dialog(self, clip) -> bool:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(self, "Detach subproject to", "", "KokoroGUI project (*.tbaw)")
        return bool(path) and self.detach_subproject(clip, path)

    def embed_subproject(self, clip) -> bool:
        """Embed: the linked child's bundle is copied into the parent (its
        project dir's `projects/`) and the clip embeds it from now on."""
        from kokoro_gui.daw.undo import SetFieldCommand

        child = self.child_project(clip) or self.open_child(clip)
        self.wait_for_project_io()
        child = child or self.child_project(clip)
        if child is None:
            return False
        parent = self.project_for(clip)
        target = project_io.embedded_child_path(parent.project_dir, child.project_id)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        self._autosave_one(child)
        plan, result = self._write_child_bundle(child, target)
        new_child = {"kind": "embedded", "id": child.project_id}
        parent.document.undo_stack.push(SetFieldCommand("clip", clip.id, "child", new_child))
        child.kind, child.path, child.manifest = "embedded", None, dict(plan.manifest)
        project_io.record_save(child.project_dir, target, result, source_path=self.source_of(child))
        child.dirty = False
        self.schedule_save()
        self.refresh_timeline()
        self.set_status(f"{child.title()} is embedded in {parent.title()}.")
        return True

    def remove_subproject(self, clip) -> bool:
        """Remove: the placeholder line (and with it the nested clip) leaves
        the parent, one undo step. An open child is put away, its dir left
        for close-time eviction."""
        from kokoro_gui.daw.undo import TextEditCommand

        parent = self.project_for(clip)
        extent = parent.document.clip_extent(clip.id)
        if extent is None:
            return False
        child = self.child_project(clip)
        if child is not None:
            if child is self.level or self.level in self._descendants_of(child):
                self.set_level(parent)
            if self.focus is child:
                self._set_focus_forced(parent)
            self._autosave_one(child)
            for gone in [*self._descendants_of(child), child]:
                if gone.lock is not None:
                    gone.lock.release()
                    gone.lock = None
                self.children.pop(gone.project_id, None)
        start, end = extent
        text = parent.document.text
        parent.document.undo_stack.push(TextEditCommand(start, end - start, 0, text[:start] + text[end:]))
        if parent is self.focus and self.editor is not None:
            self.editor.load_text(parent.document.text)
        self.schedule_save()
        self.refresh_timeline()
        return True

    def _descendants_of(self, project) -> list:
        out = []
        for child in self.children.values():
            parent = self.parent_of(child)
            while parent is not None:
                if parent is project:
                    out.append(child)
                    break
                parent = self.parent_of(parent)
        return out

    def _rebase_linked_paths(self, new_root_path: str) -> None:
        """Save As moved the root: every linked path relative to it is
        rewritten for the new location (the files didn't move)."""
        old_root_path = self.root.path
        for project in self.open_projects():
            # Only paths relative to the root's file; a linked child's own
            # links are relative to that child's file, which didn't move.
            if self._parent_file(project) != old_root_path:
                continue
            for clip in project.document.nested_clips():
                child = clip.child or {}
                if child.get("kind") != "linked" or not child.get("path"):
                    continue
                absolute = self.child_bundle_path(project, clip)
                if not absolute:
                    continue
                base_dir = os.path.dirname(os.path.abspath(new_root_path))
                if os.path.splitdrive(base_dir)[0].lower() == os.path.splitdrive(absolute)[0].lower():
                    child["path"] = os.path.relpath(absolute, base_dir).replace("\\", "/")
                else:
                    child["path"] = absolute

    # -- making a child -----------------------------------------------------------

    def _next_subproject_title(self, document) -> str:
        titles = {document.clip_text(c) for c in document.nested_clips()}
        n = len(titles) + 1
        while f"Subproject {n}" in titles:
            n += 1
        return f"Subproject {n}"

    def _seed_child_characters(self, source_document, moved_clips) -> list:
        """NP3: a new child starts with the merged view: every root character
        linked at project scope (a root-local one gets a project-scope
        `library_id` minted first), every global entry not already there,
        and any other character a moved clip uses. Records keep the root's
        ids, so moved clips' `character_id`s still resolve."""
        import copy

        root_doc = self.root.document
        characters = []
        for character in root_doc.characters:
            if not character.library_id:
                character.library_id = character_library.new_project_scope_id()
            characters.append(copy.deepcopy(character))
        present = {c.library_id for c in characters}
        for entry in self.character_library.list():
            if entry.library_id not in present:
                characters.append(character_library.linked_copy(entry))
                present.add(entry.library_id)
        ids = {c.id for c in characters}
        for clip in moved_clips:
            character = source_document.get_character(clip.character_id)
            if character is not None and character.id not in ids:
                characters.append(copy.deepcopy(character))
                ids.add(character.id)
        return characters

    def new_subproject(self, start: int | None = None, end: int | None = None, title: str | None = None):
        """File > New Subproject: a new embedded child of the focus project.
        With a range, the text in it (widened to whole clips) and its clips
        move into the child, their audio copied along, and a placeholder
        line for the child takes its place; without one, an empty child's
        placeholder goes in at `start` (default: the end). One undo step on
        the parent. Returns the child `OpenProject`, or None."""
        import copy
        import shutil

        from kokoro_gui.daw import markers as marker_ops
        from kokoro_gui.daw.models import Document, _new_id
        from kokoro_gui.daw.undo import ReplaceWithNestedCommand

        parent = self.focus
        if not parent.project_dir:
            return None
        document = parent.document
        text_len = len(document.text)
        if start is None:
            start = text_len
        start = max(0, min(int(start), text_len))
        end = start if end is None else max(start, min(int(end), text_len))
        if end > start:
            if document.overlaps_nested(start, end):
                self.set_status("A selection holding a subproject can't become one.", "warning")
                return None
            start, end = _snap_to_clips(document, start, end)
        moved = _moved_clips(document, start, end) if end > start else []
        title = (title or "").strip() or self._next_subproject_title(document)

        project_dir, child_id = project_io.create_project_dir()
        try:
            lock = project_io.ProjectLock(project_dir).acquire()
        except project_io.ProjectLockedError as e:
            self.set_status(str(e), "error")
            return None

        # The child's document: the moved runs (tags kept) and clips, the
        # tracks they sit on, the merged characters, the parent's pacing.
        runs = []
        pos = 0
        for run in document.runs:
            r_start, r_end = pos, pos + len(run.text)
            pos = r_end
            lo, hi = max(r_start, start), min(r_end, end)
            if hi > lo:
                piece = copy.deepcopy(run)
                piece.text = run.text[lo - r_start:hi - r_start]
                runs.append(piece)
        clips = [copy.deepcopy(c) for c in moved]
        track_ids = {c.track_id for c in clips if c.track_id}
        tracks = [copy.deepcopy(t) for t in document.tracks if t.id in track_ids]
        settings = {k: copy.deepcopy(v) for k, v in document.settings.items() if k != marker_ops.MARKERS_KEY}
        child_document = Document(runs=runs, clips=clips, tracks=tracks,
                                  characters=self._seed_child_characters(document, moved), settings=settings)
        child_document._normalize_runs()
        generated = os.path.join(project_dir, *project_io.AUDIO_GENERATED.split("/"))
        os.makedirs(generated, exist_ok=True)
        for clip in child_document.clips:
            for segments in (clip.segments, *clip.takes.values()):
                for segment in segments:
                    if segment.audio_path and os.path.isfile(segment.audio_path):
                        target = os.path.join(generated, os.path.basename(segment.audio_path))
                        if not os.path.exists(target):
                            shutil.copyfile(segment.audio_path, target)
                        segment.audio_path = os.path.abspath(target)

        child_settings = {"title": title}
        project_io.autosave_to_dir(child_document, child_settings, project_dir)
        project_io.write_session(project_dir, {
            "source_path": project_io.child_source_path(self.source_of(parent), child_id),
            "zip_size": None, "zip_mtime": None, "saved_digest": None, "dirty": True, "asset_index": {},
        })

        clip_id = _new_id()
        child_ref = {"kind": "embedded", "id": child_id}
        document.undo_stack.push(ReplaceWithNestedCommand(start, end, child_ref, title, clip_id))
        clip = document.get_clip(clip_id)
        loaded = project_io.load_project_dir(project_dir, child_id, {})
        child = self._attach_child(parent, clip, loaded, lock, None, True)
        if parent is self.focus and self.editor is not None:
            self.editor.load_text(document.text)
        self.on_characters_changed()
        self.set_status(f"New subproject: {title}.")
        return child

    def new_subproject_from_selection(self):
        """The File menu's New Subproject: the transcript's selection (or an
        empty child at the caret's paragraph end when nothing is selected)."""
        if self.is_busy():
            QMessageBox.warning(self, "Busy", "Finish or cancel the current job first.")
            return None
        editor = self.editor
        if editor is None:
            return self.new_subproject()
        cursor = editor.textCursor()
        if cursor.hasSelection():
            return self.new_subproject(cursor.selectionStart(), cursor.selectionEnd())
        return self.new_subproject(cursor.block().position() + cursor.block().length() - 1)

    def new_from_sections(self, sections) -> list:
        """New from eBook, one subproject per chapter (NP8): a new project
        whose text is one placeholder line per `(title, text)` section, a
        blank line between them, each an embedded subproject holding that
        chapter's text. Returns the children. Runs after the current
        project is put away (the close prompt may cancel it: then [])."""
        made = []

        def _build():
            document = self.root.document
            for index, (title, text) in enumerate(sections):
                if index:
                    document.replace_text(len(document.text), 0, 2, document.text + "\n\n")
                child = self.new_subproject(len(document.text), title=title)
                if child is None:
                    continue
                child.document.set_plain_text(text)
                self._autosave_one(child)
                made.append(child)
            if self.editor is not None:
                self.editor.load_text(document.text)
            self.refresh_timeline()
            self.set_status(f"New project with {len(made)} subproject(s). Save As to name it.")

        self.new_project(then=_build)
        return made

    def new_from_ebook(self, path: str, per_chapter: bool = True) -> list:
        """The welcome dialog's New from text: an EPUB or a PDF with an
        outline becomes one subproject per chapter when `per_chapter`,
        anything else (or a book with one section) plain text as before."""
        from kokoro_gui.engine.text_extraction import extract_sections

        if per_chapter and path.lower().endswith((".epub", ".pdf")):
            try:
                sections = extract_sections(path)
            except Exception as e:  # noqa: BLE001 - reported, never a crash
                QMessageBox.critical(self, "Import failed", f"Read failed: {e}")
                return []
            if len(sections) > 1:
                return self.new_from_sections(sections)
        self.import_text(path, target="new")
        return []

    # -- autosave, dirty, save ------------------------------------------------------

    def open_projects(self) -> list:
        return [self.root, *self.children.values()]

    def any_project_dirty(self) -> bool:
        return any(p.dirty for p in self.open_projects())

    def _autosave_one(self, project) -> None:
        if not project.project_dir:
            return
        try:
            digest = project_io.autosave_to_dir(project.document, project.project_settings, project.project_dir)
        except Exception as e:  # noqa: BLE001 - autosave must never crash the UI
            self.set_status(f"Autosave failed: {e}", "error")
            return
        if digest != project.digest:
            project.digest = digest
            project.state_cache = None
        session = project_io.read_session(project.project_dir) or {}
        dirty = digest != session.get("saved_digest")
        if bool(session.get("dirty")) != dirty:
            session["dirty"] = dirty
            try:
                project_io.write_session(project.project_dir, session)
            except OSError as e:
                self.set_status(f"Autosave failed: {e}", "error")
        project.dirty = dirty

    def _children_to_save(self) -> list:
        """Open children whose bundle is out of date, deepest first, so a
        parent's bundle is written after every child it carries."""
        out = []
        for child in self.children.values():
            parent = self.parent_of(child)
            if parent is None:
                continue
            target = child.path if child.kind == "linked" else \
                project_io.embedded_child_path(parent.project_dir, child.project_id)
            if not target:
                continue
            if child.dirty or not os.path.isfile(target):
                out.append((child, target))
        out.sort(key=lambda pair: self.depth_of(pair[0]), reverse=True)
        return out

    def _plan_child_saves(self, fx_presets_dir: str) -> tuple:
        """`([(child, plan, source), ...], warnings)` on the GUI thread."""
        plans = []
        warnings = []
        to_save = self._children_to_save()
        for child, target in to_save:
            pending = {c.project_id for c, _t in to_save if c.parent_id == child.project_id}
            plan, child_warnings = project_io.plan_save(
                child.document, child.project_settings, target, child.project_dir, child.project_id,
                self._backend_for, fx_presets_dir, project_io.read_session(child.project_dir), child.manifest,
                pending_children=pending,
            )
            source = target if child.kind == "linked" else self.source_of(child)
            plans.append((child, plan, source))
            warnings.extend(f"{child.title()}: {w}" for w in child_warnings)
        return plans, warnings

    def _record_child_saves(self, plans, results) -> None:
        for (child, plan, source), result in zip(plans, results):
            try:
                project_io.record_save(child.project_dir, plan.path, result, source_path=source)
            except OSError as e:
                self.set_status(f"Saved, but couldn't record {child.title()}'s session: {e}", "warning")
            child.manifest = dict(plan.manifest)
            child.dirty = False

    # -- teardown --------------------------------------------------------------------

    def _teardown_children(self, discard: bool) -> None:
        """Releases every open child's lock (deepest first); on Discard its
        dir goes (the parent's bundle, or the linked file, has the last
        saved state), otherwise it's GC'd."""
        for child in sorted(self.children.values(), key=self.depth_of, reverse=True):
            if child.lock is not None:
                if not discard:
                    try:
                        project_io.gc_project_dir(child.project_dir, child.document)
                    except OSError:
                        pass
                child.lock.release()
                child.lock = None
            if discard and child.project_dir:
                project_io.delete_project_dir(child.project_dir)
        self.children.clear()
        self._missing_children.clear()
        self._closed_child_states.clear()
        self.focus = self.level = self.root


def _snap_to_clips(document, start: int, end: int) -> tuple:
    """`[start, end)` widened so no clip is cut: a clip straddling either
    edge comes along whole."""
    for clip in document.clips:
        extent = document.clip_extent(clip.id)
        if extent is None:
            continue
        c_start, c_end = extent
        if c_start < start < c_end:
            start = c_start
        if c_start < end < c_end:
            end = c_end
    return start, end


def _moved_clips(document, start: int, end: int) -> list:
    out = []
    for clip in document.clips:
        extent = document.clip_extent(clip.id)
        if extent is not None and start <= extent[0] and extent[1] <= end:
            out.append(clip)
    return out
