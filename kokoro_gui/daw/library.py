"""The global character library (grill WF4-WF7, data shape WF12).

A store outside any project: one JSON file per character at
`<root>/<library_id>.json`, holding the dict `serialization.character_to_dict`
writes, with `library_id` (and `id`) equal to the file stem. A project's
`Character` links to an entry through `Character.library_id` and keeps its
own `id` and `name`; the record stays inlined in `document.json`, so a
bundle opens on a machine without the entry and plays the snapshot.

`resolve_characters` is the live link (WF5): it copies an entry's voice
settings, colour, engine and variants onto every linked record. The app calls
it when a document is switched in and again when the library directory
changes on disk.

Qt-free like the rest of `kokoro_gui/daw/`.
"""
from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field

from kokoro_gui.daw.models import Character, _new_id
from kokoro_gui.daw.serialization import character_from_dict, character_to_dict

# `characters/` beside `presets/`, relative to the working directory like
# the other stores. Read at call time, so tests monkeypatch it the way they
# do `kokoro_engine.CUSTOM_VOICES_DIR`.
LIBRARY_DIR = "characters"

# What resolution copies from an entry onto a linked record. Never `id`
# (the document's own) and never `name` (renamed per project, WF12).
RESOLVED_FIELDS = ("preset_data", "highlight_color", "backend_id", "variants")

_log = logging.getLogger(__name__)


def _safe_id(library_id) -> str | None:
    """`library_id` as a bare file stem, or None when nothing usable is
    left. Same `os.path.basename` rule the voice and preset stores use."""
    if not isinstance(library_id, str):
        return None
    stem = os.path.basename(library_id.strip())
    if not stem or stem.startswith("."):
        return None
    return stem


class CharacterLibrary:
    """One directory of character files. `root=None` reads `LIBRARY_DIR`
    each time, so a monkeypatched module constant takes effect."""

    def __init__(self, root: str | None = None):
        self._root = root

    @property
    def root(self) -> str:
        return self._root if self._root is not None else LIBRARY_DIR

    def _path(self, library_id: str) -> str:
        return os.path.join(self.root, f"{library_id}.json")

    def _read(self, path: str, stem: str) -> Character | None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("not a JSON object")
            character = character_from_dict(data)
        except (OSError, ValueError, TypeError) as e:
            _log.warning("Skipping unreadable character library file %s: %s", path, e)
            return None
        # The file name is the id; a hand-edited file can't claim another.
        character.library_id = stem
        character.id = stem
        return character

    def list(self) -> list:
        """Every readable entry, sorted by name. A corrupt file is skipped
        and logged."""
        root = self.root
        if not os.path.isdir(root):
            return []
        entries = []
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if not name.endswith(".json") or not os.path.isfile(path):
                continue
            stem = _safe_id(name[: -len(".json")])
            if stem is None:
                continue
            character = self._read(path, stem)
            if character is not None:
                entries.append(character)
        entries.sort(key=lambda c: (c.name.casefold(), c.library_id))
        return entries

    def get(self, library_id) -> Character | None:
        stem = _safe_id(library_id)
        if stem is None:
            return None
        path = self._path(stem)
        if not os.path.isfile(path):
            return None
        return self._read(path, stem)

    def save(self, character: Character) -> str:
        """Writes `character` as an entry and returns its library id: the
        record's `library_id` when set, else a freshly minted one (promote,
        WF7). The file gets `id == library_id`; `character` itself is not
        changed. Written to a `.tmp` and `os.replace`d, so a reader never
        sees half a file."""
        library_id = _safe_id(character.library_id) if character.library_id else None
        if library_id is None:
            library_id = _new_id()
        entry = copy.deepcopy(character)
        entry.library_id = library_id
        entry.id = library_id
        root = self.root
        os.makedirs(root, exist_ok=True)
        path = self._path(library_id)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(character_to_dict(entry), f, indent=2)
        os.replace(tmp, path)
        return library_id

    def delete(self, library_id) -> bool:
        stem = _safe_id(library_id)
        if stem is None:
            return False
        try:
            os.remove(self._path(stem))
        except FileNotFoundError:
            return False
        return True

    def mtime(self) -> int:
        """A change stamp for the whole store: the newest `st_mtime_ns` of
        the directory and its files, 0 when the directory is missing.
        Adding, removing or rewriting an entry moves it."""
        root = self.root
        try:
            newest = os.stat(root).st_mtime_ns
        except OSError:
            return 0
        try:
            names = os.listdir(root)
        except OSError:
            return newest
        for name in names:
            if not name.endswith(".json"):
                continue
            try:
                newest = max(newest, os.stat(os.path.join(root, name)).st_mtime_ns)
            except OSError:
                continue
        return newest


def linked_copy(entry: Character) -> Character:
    """A document record linked to library `entry`: the entry's name and
    settings, `library_id` set, and a fresh document `id`."""
    return Character(
        name=entry.name,
        preset_data=copy.deepcopy(entry.preset_data),
        highlight_color=entry.highlight_color,
        backend_id=entry.backend_id,
        variants=copy.deepcopy(entry.variants),
        library_id=entry.library_id or entry.id,
        extra=copy.deepcopy(entry.extra),
    )


def write_through(character: Character, library: CharacterLibrary) -> bool:
    """The live link's write half (WF5): copies a linked record's resolved
    fields onto its library entry and saves it, keeping the entry's own
    name. Returns False, writing nothing, for a local character or one
    whose entry isn't in `library` (a bundle from another machine)."""
    if not character.library_id:
        return False
    entry = library.get(character.library_id)
    if entry is None:
        return False
    for name in RESOLVED_FIELDS:
        setattr(entry, name, copy.deepcopy(getattr(character, name)))
    library.save(entry)
    return True


@dataclass
class ResolveReport:
    """What `resolve_characters` did: document character ids whose record
    changed, and linked ones no store had (the snapshot stands in)."""

    changed: list = field(default_factory=list)
    missing: list = field(default_factory=list)


def resolve_characters(document, stores) -> ResolveReport:
    """Refreshes every linked character in `document` from the first store
    in `stores` holding its `library_id`. `stores` is `[global library]`
    for now; phase 4 puts the top-level project's store ahead of it. Each
    store needs only a `get(library_id)`."""
    report = ResolveReport()
    for character in document.characters:
        if not character.library_id:
            continue
        entry = None
        for store in stores:
            entry = store.get(character.library_id)
            if entry is not None:
                break
        if entry is None:
            report.missing.append(character.id)
            continue
        changed = False
        for name in RESOLVED_FIELDS:
            value = getattr(entry, name)
            if getattr(character, name) != value:
                setattr(character, name, copy.deepcopy(value))
                changed = True
        if changed:
            report.changed.append(character.id)
    return report
