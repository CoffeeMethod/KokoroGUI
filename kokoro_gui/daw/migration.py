"""First-launch migration into the character library, and the characters a
new document starts with.

`import_presets_to_library` runs once, on the first launch after the
library shipped (`settings["library_imported"]` unset): every
`presets/*.json` becomes a library entry and the files stay where they
are. `link_exact_matches` then links the open project's characters that
equal an imported entry by name and `preset_data`; the rest stay local.

`migrate_legacy_settings_to_document` builds the document a first run (or
File > New) starts on: every library entry, linked, and no tracks (a track
is made the first time the transcript uses its character, grill PR4). An
empty library gives one local "Default" character seeded from the app
settings, so a returning user's last-used voice isn't dropped.

There is no document *text* to migrate: the old generation text box was
never written to `config_qt.json`.
"""
import json
import os

from kokoro_gui.daw.library import CharacterLibrary, linked_copy
from kokoro_gui.daw.models import DEFAULT_HIGHLIGHT_PALETTE, Character, Document
from kokoro_gui.engine.presets import ALLOWED_PRESET_KEYS, filter_allowed_keys

DEFAULT_CHARACTER_NAME = "Default"


def _load_preset_files(presets_dir: str) -> dict:
    """Reads every `<presets_dir>/*.json` file into `{name: preset_dict}`,
    skipping the `fx/` subdirectory (FX presets, not speaker presets) and
    any file that fails to parse. Plain `json.load` rather than
    `PresetsMixin.load_preset`, which needs an engine instance and
    hardcodes `"presets"` as a relative path."""
    presets = {}
    if not os.path.isdir(presets_dir):
        return presets

    for entry in sorted(os.listdir(presets_dir)):
        full_path = os.path.join(presets_dir, entry)
        if not entry.endswith(".json") or not os.path.isfile(full_path):
            continue
        name = entry[: -len(".json")]
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            presets[name] = data
    return presets


def _same_character(entry: Character, character: Character) -> bool:
    return entry.name == character.name and entry.preset_data == character.preset_data


def import_presets_to_library(presets_dir: str, library: CharacterLibrary) -> list:
    """Adds one library entry per `presets_dir/*.json` and returns their
    library ids, in file-name order. A preset equal (name and settings) to
    an entry the library already has reuses that entry instead of adding a
    duplicate, so running this twice is harmless. The files are left in
    place."""
    existing = library.list()
    imported = []
    for index, (name, preset_data) in enumerate(sorted(_load_preset_files(presets_dir).items())):
        color = DEFAULT_HIGHLIGHT_PALETTE[index % len(DEFAULT_HIGHLIGHT_PALETTE)]
        character = Character.from_preset_dict(name, preset_data, highlight_color=color)
        match = next((e for e in existing if _same_character(e, character)), None)
        if match is not None:
            imported.append(match.library_id)
            continue
        library_id = library.save(character)
        existing.append(library.get(library_id) or character)
        imported.append(library_id)
    return imported


def link_exact_matches(document: Document, entries: list) -> list:
    """Links each local character in `document` whose name and
    `preset_data` equal one of `entries` (library `Character`s) to that
    entry. A one-time step after the presets import; returns the linked
    records."""
    linked = []
    for character in document.characters:
        if character.library_id:
            continue
        match = next((e for e in entries if _same_character(e, character)), None)
        if match is not None:
            character.library_id = match.library_id
            linked.append(character)
    return linked


def seed_characters(library: CharacterLibrary, settings: dict | None = None) -> list:
    """The characters a new document starts with: every library entry,
    linked, or one local "Default" from `settings` when the library is
    empty."""
    entries = library.list()
    if entries:
        return [linked_copy(entry) for entry in entries]
    seeded = filter_allowed_keys(settings or {}, ALLOWED_PRESET_KEYS)
    return [Character.from_preset_dict(DEFAULT_CHARACTER_NAME, seeded, highlight_color=DEFAULT_HIGHLIGHT_PALETTE[0])]


def migrate_legacy_settings_to_document(settings: dict, library: CharacterLibrary) -> Document:
    """A fresh, empty `Document` whose characters come from the library
    (see `seed_characters`). No tracks: `Document.assign_character_to_range`
    makes a character's track on first use."""
    return Document(runs=[], clips=[], tracks=[], characters=seed_characters(library, settings), settings={})
