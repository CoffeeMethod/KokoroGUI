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
        if then is not None:
            then(None)
        return None

    def _attach_child(self, parent, clip, loaded, lock, path, dirty: bool) -> OpenProject:
        child = OpenProject(
            document=loaded.document, project_dir=loaded.project_dir, project_id=self.child_id_of(clip), lock=lock,
            project_settings=dict(loaded.project_settings or {}), path=path, parent_id=parent.project_id,
            clip_id=clip.id, manifest=dict(loaded.manifest or {}), dirty=dirty,
            kind=(clip.child or {}).get("kind", "embedded"),
        )
        self.children[child.project_id] = child
        self._missing_children.discard(child.project_id)
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
        """Placeholder until the mixdown lands: a nested clip is stale."""
        project.document.nested_state_fn = lambda clip: True

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

        from kokoro_gui.daw.models import _new_id

        root_doc = self.root.document
        characters = []
        for character in root_doc.characters:
            if not character.library_id:
                character.library_id = _new_id()
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
