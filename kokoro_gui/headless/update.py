"""Opening the `.tbaw` an update re-renders, and what the rebuilt document
keeps of the old one (plan section 4, "update").

The document comes from the new script. It keeps the old settings, each
character's id, color and variants (matched by name), and the tracks with
their mixer settings. The voice and the rest come from the new cast; a speaker
it doesn't name keeps what the project has. Clip edits made in the app are
not kept, so a project with audio a script can't make again (recordings, music,
subprojects, guide takes) is refused.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import zipfile
from collections.abc import Mapping
from dataclasses import replace
from typing import Optional

from kokoro_gui.daw import dirty, lanes, revision, serialization
from kokoro_gui.daw.models import Document, _new_id
from kokoro_gui.daw.share import clean_preset_data
from kokoro_gui.qt import project as project_io

from . import paths
from .errors import HeadlessError
from .script import CAST_KEYS

log = logging.getLogger(__name__)


def check_update(update) -> Optional[str]:
    """The real path of an existing `.tbaw` under an allowed root."""
    if update is None:
        return None
    real = paths.check_out_path(update)
    if os.path.splitext(real)[1].lower() != project_io.DEFAULT_EXTENSION or not os.path.isfile(real):
        raise HeadlessError("bad_arguments", f"update must be an existing {project_io.DEFAULT_EXTENSION} file: {real}")
    return real


def _known_source(project_id: str, path: str) -> str:
    """`path` as a working folder's session spells it. The app may have opened
    the file through a link, and a folder (and so its lock) is found by that
    spelling."""
    root = project_io.projects_root()
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if name != project_id and not name.startswith(project_id + "-"):
            continue
        folder = os.path.join(root, name)
        source = (project_io.read_session(folder) or {}).get("source_path")
        try:
            same = isinstance(source, str) and os.path.samefile(source, path)
        except (OSError, ValueError):
            same = False
        if same:
            return source
    return path


def _refuse_lost_audio(document: Document) -> None:
    recorded = any(dirty.is_recorded_take(segments) for clip in document.clips
                   for segments in (clip.segments, *clip.takes.values()))
    if recorded or document.sources or any(clip.source != "generated" for clip in document.clips):
        raise HeadlessError("bad_arguments",
                            "this project has recordings, music or subprojects, which an update would drop")


def _document_in(info: project_io.BundleInfo) -> Document:
    with zipfile.ZipFile(info.path) as zf:
        return serialization.document_from_dict(json.loads(zf.read(project_io.DOCUMENT)))


class Opened:
    """The project an update opened: locked, its old document and settings,
    and the working folder's own files as bytes."""

    _FILES = (project_io.DOCUMENT, project_io.PROJECT_JSON, project_io.SESSION)

    def __init__(self, path: str, discard_session: bool):
        self.info = project_io.inspect_bundle(path)
        self.path = os.path.abspath(_known_source(self.info.project_id, path))
        self.info.path = self.path
        self.project_dir = project_io.choose_project_dir(self.info.project_id, self.path)
        self.lock = project_io.ProjectLock(self.project_dir).acquire()
        try:
            self._open(discard_session)
        except BaseException:
            self.lock.release()
            raise

    def _open(self, discard_session: bool) -> None:
        """The steps of the app's Open, with `unsaved_session` for its
        recover prompt and `discard_session` for "Take file"."""
        session = project_io.read_session(self.project_dir)
        children = [d for d in project_io.child_dirs_of(self.path) if not project_io.is_locked(d)]
        unsaved = [d for d in children if (project_io.read_session(d) or {}).get("dirty")]
        action = project_io.open_action(session, self.info, self.project_dir, unsaved)
        if action == "recover" and not discard_session:
            raise HeadlessError("unsaved_session", "this project has unsaved changes from the app; save or discard "
                                                   "them there, or pass discard_session")
        # Before the wipe: a project that would be refused keeps its session.
        _refuse_lost_audio(_document_in(self.info))
        if action != "reuse":
            project_io.check_free_space(project_io.projects_root(), project_io.heavy_bytes(self.info),
                                        "open the project")
            project_io.wipe_project_dir(self.project_dir)
            if action == "recover":
                for folder in children:
                    project_io.delete_project_dir(folder)
            project_io.extract_small(self.info, self.project_dir)
            project_io.extract_audio(self.info, self.project_dir)
        loaded = project_io.finish_open(self.info, self.project_dir)
        revision.bump_files()  # files were extracted
        _refuse_lost_audio(loaded.document)
        self.document, self.settings = loaded.document, loaded.project_settings
        self._bytes = {}
        for name in self._FILES:
            try:
                with open(os.path.join(self.project_dir, name), "rb") as f:
                    self._bytes[name] = f.read()
            except FileNotFoundError:
                self._bytes[name] = None

    def mark_dirty(self) -> None:
        """Before the folder's document changes: a crash from here on leaves
        a folder Open offers to recover, not one it takes for a clean copy of
        the file. Save clears it."""
        session = project_io.read_session(self.project_dir) or {}
        project_io.write_session(self.project_dir, {**session, "dirty": True})

    def restore(self) -> None:
        """The folder's own files as they were, the session last."""
        for name in self._FILES:
            path = os.path.join(self.project_dir, name)
            try:
                if self._bytes[name] is None:
                    if os.path.exists(path):
                        os.remove(path)
                    continue
                with open(path + ".tmp", "wb") as f:
                    f.write(self._bytes[name])
                os.replace(path + ".tmp", path)
            except OSError:
                log.exception("couldn't put %s back", path)
                return


def _key(name) -> str:
    return str(name or "").strip().lower()


def merged_cast(cast, old: Document, backend) -> tuple:
    """`(cast, kept)`: the new cast plus the old project's settings for each
    speaker it doesn't name, on the same engine. `kept` is the name keys of
    those. The project file is untrusted, so its values are cleaned."""
    if cast is not None and not isinstance(cast, Mapping):
        return cast, set()  # `build_document` says what is wrong with it
    merged = dict(cast or {})
    named = {_key(name) for name in merged}
    kept = set()
    for character in old.characters:
        key = _key(character.name)
        if not isinstance(character.name, str) or not key or key in named or key in kept \
                or character.backend_id != backend.id:
            continue
        preset = clean_preset_data(dict(character.preset_data or {}))
        if preset.get("voice"):
            merged[character.name] = {k: preset[k] for k in CAST_KEYS if k in preset}
            kept.add(key)
    return merged, kept


def carry_over(document: Document, old: Document, kept: set) -> None:
    """Puts the old settings, character identities and tracks on a freshly
    built document."""
    document.settings = dict(old.settings)
    previous: dict = {}
    for character in old.characters:
        previous.setdefault(_key(character.name), character)
    for character in document.characters:
        before = previous.get(_key(character.name))
        if before is None:
            continue
        for clip in document.clips:
            if clip.character_id == character.id:
                clip.character_id = before.id
        character.id = before.id
        character.highlight_color = before.highlight_color
        character.variants = dict(before.variants or {})
        character.extra = dict(before.extra or {})
        if _key(character.name) in kept:
            character.preset_data = {**clean_preset_data(dict(before.preset_data or {})), **character.preset_data}
            character.library_id = before.library_id
    known = {character.id for character in document.characters}
    document.tracks = [copy.deepcopy(track) for track in old.tracks
                       if track.character_id is None or track.character_id in known]
    for clip in document.clips:
        clip.track_id = None
    plan = lanes.plan_relane(document)
    document.tracks = [*document.tracks, *plan.new_tracks]
    for clip in document.clips:
        clip.track_id = plan.assignments.get(clip.id)


def _pieces(texts) -> tuple:
    return tuple(" ".join((text or "").split()) for text in texts)


def reuse_segments(document: Document, old: Document, config_for_clip) -> None:
    """Gives each clip the old segments that are right for it: the same pieces
    of text, under the key it has now, with their files there. Parked takes
    count, since a rebuilt clip is at take 0."""
    pool: dict = {}
    for clip in old.clips:
        for segments in (clip.segments, *clip.takes.values()):
            if segments:
                ordered = sorted(segments, key=lambda s: s.order_index)
                pool.setdefault(_pieces(s.text for s in ordered), []).append(ordered)
    for clip in document.clips:
        config = config_for_clip(clip)
        text = document.clip_text(clip)
        pieces = dirty.predict_segment_texts(dirty.spoken_text(text, config), config)
        for segments in pool.get(_pieces(pieces), ()):
            clip.segments = [replace(s, id=_new_id(), words=[list(w) for w in s.words or []]) for s in segments]
            if all(s.audio_path for s in segments) \
                    and not dirty.is_clip_dirty(clip, text, config, document.segment_key_fn):
                clip.status = "generated"
                break
            clip.segments = []
