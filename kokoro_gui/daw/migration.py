"""First-load migration from today's `presets/*.json` + `config_qt.json`
settings into a `Document`.

There is no document *text* to migrate: `GenerationDock.text_entry`'s
content was never written to `config_qt.json` (`save_settings` never
persists the text box), so a fresh `Document` always starts with empty text
and no clips - only the presets/settings side has anything to carry
forward.

Not called anywhere yet: nothing in `kokoro_gui/qt` consumes `Document` until
the transcript-panel/timeline workstreams land. This module exists now so
its behavior is locked in and tested before the GUI depends on it.
"""
import json
import os

from kokoro_gui.daw.models import DEFAULT_HIGHLIGHT_PALETTE, Character, Document, Track
from kokoro_gui.engine.presets import ALLOWED_PRESET_KEYS, filter_allowed_keys

DEFAULT_CHARACTER_NAME = "Default"


def _load_preset_files(presets_dir: str) -> dict:
    """Reads every `<presets_dir>/*.json` file into `{name: preset_dict}`,
    skipping the `fx/` subdirectory (FX presets, not speaker presets) and
    any file that fails to parse. Deliberately plain `json.load` rather than
    `PresetsMixin.load_preset` - that method requires an engine instance and
    hardcodes `"presets"` as a relative path, neither of which fits a
    migration helper that needs to run standalone against an arbitrary
    directory in tests."""
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
                presets[name] = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
    return presets


def migrate_legacy_settings_to_document(settings: dict, presets_dir: str) -> Document:
    """Builds a fresh `Document` from today's presets directory and app
    settings:

    1. Every `presets/*.json` file becomes a `Character`, via
       `Character.from_preset_dict` - the JSON files themselves are left
       untouched (a safe downgrade path).
    2. If no presets exist yet, one "Default" `Character` is seeded from
       `settings`'s current voice/speed/etc. so a returning user's
       last-used config isn't silently discarded.
    3. One default `Track` is created per `Character` (Q8's auto-placement
       default - a character gets an obvious lane to land clips on before
       any manual re-sorting happens).
    """
    characters = []
    for index, (name, preset_data) in enumerate(sorted(_load_preset_files(presets_dir).items())):
        color = DEFAULT_HIGHLIGHT_PALETTE[index % len(DEFAULT_HIGHLIGHT_PALETTE)]
        characters.append(Character.from_preset_dict(name, preset_data, highlight_color=color))

    if not characters:
        seeded = filter_allowed_keys(settings or {}, ALLOWED_PRESET_KEYS)
        characters.append(
            Character.from_preset_dict(
                DEFAULT_CHARACTER_NAME, seeded, highlight_color=DEFAULT_HIGHLIGHT_PALETTE[0]
            )
        )

    tracks = [
        Track(name=character.name, character_id=character.id, order_index=i)
        for i, character in enumerate(characters)
    ]

    return Document(text="", clips=[], tracks=tracks, characters=characters, settings={})
