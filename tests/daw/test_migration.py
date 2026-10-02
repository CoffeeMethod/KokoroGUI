"""Tests for kokoro_gui/daw/migration.py: the one-time `presets/*.json` import
into the character library, the exact-match link, and the characters a new
document is seeded with. tmp_path-isolated preset and library directories,
mirroring the isolation convention tests/conftest.py's isolated_dirs fixture
uses for kokoro_engine's storage dirs."""
import json

import pytest

from kokoro_gui.daw.library import CharacterLibrary
from kokoro_gui.daw.migration import (
    DEFAULT_CHARACTER_NAME, import_presets_to_library, link_exact_matches, migrate_legacy_settings_to_document,
)
from kokoro_gui.daw.models import DEFAULT_HIGHLIGHT_PALETTE, Character, Document


def _write_preset(presets_dir, name, data):
    presets_dir.mkdir(parents=True, exist_ok=True)
    (presets_dir / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def lib(tmp_path):
    return CharacterLibrary(str(tmp_path / "characters"))


# -- seeding a new document --------------------------------------------------------

def test_empty_library_seeds_default_character_from_settings(lib):
    settings = {"voice": "af_bella", "speed": 1.2, "out_dir": "/should/be/dropped"}
    doc = migrate_legacy_settings_to_document(settings, lib)

    assert len(doc.characters) == 1
    character = doc.characters[0]
    assert character.name == DEFAULT_CHARACTER_NAME
    assert character.preset_data == {"voice": "af_bella", "speed": 1.2}
    assert character.library_id is None  # local
    assert doc.tracks == []  # tracks are made on first use (grill PR4)
    assert doc.text == ""
    assert doc.clips == []


def test_library_entries_seed_linked_characters_without_tracks(lib):
    alice = lib.save(Character.from_preset_dict("Alice", {"voice": "af_bella"}))
    bob = lib.save(Character.from_preset_dict("Bob", {"voice": "am_michael"}))

    doc = migrate_legacy_settings_to_document({"voice": "ignored"}, lib)

    assert [c.name for c in doc.characters] == ["Alice", "Bob"]
    assert [c.library_id for c in doc.characters] == [alice, bob]
    assert all(c.id not in (alice, bob) for c in doc.characters)
    assert doc.tracks == []


# -- the presets import --------------------------------------------------------------

def test_import_wraps_each_preset_file_as_a_library_entry(tmp_path, lib):
    presets_dir = tmp_path / "presets"
    _write_preset(presets_dir, "Alice", {"voice": "af_bella", "speed": 1.0})
    _write_preset(presets_dir, "Bob", {"voice": "am_michael", "speed": 0.9})

    ids = import_presets_to_library(str(presets_dir), lib)

    assert len(ids) == 2
    assert {c.name for c in lib.list()} == {"Alice", "Bob"}
    assert (presets_dir / "Alice.json").exists()  # files stay in place


def test_import_with_nonexistent_presets_dir_imports_nothing(tmp_path, lib):
    assert import_presets_to_library(str(tmp_path / "does_not_exist"), lib) == []
    assert lib.list() == []


def test_import_assigns_distinct_highlight_colors_from_palette(tmp_path, lib):
    presets_dir = tmp_path / "presets"
    for i in range(len(DEFAULT_HIGHLIGHT_PALETTE) + 1):
        _write_preset(presets_dir, f"Character{i}", {"voice": "af_bella"})

    import_presets_to_library(str(presets_dir), lib)
    by_name = {c.name: c.highlight_color for c in lib.list()}
    colors = [by_name[f"Character{i}"] for i in range(len(DEFAULT_HIGHLIGHT_PALETTE) + 1)]
    # Cycles back around once there are more characters than palette entries.
    assert colors[0] == colors[len(DEFAULT_HIGHLIGHT_PALETTE)]
    assert len(set(colors[: len(DEFAULT_HIGHLIGHT_PALETTE)])) == len(DEFAULT_HIGHLIGHT_PALETTE)


def test_import_ignores_fx_subdirectory(tmp_path, lib):
    presets_dir = tmp_path / "presets"
    _write_preset(presets_dir, "Alice", {"voice": "af_bella"})
    _write_preset(presets_dir / "fx", "reverb_heavy", {"reverb_enabled": True})

    import_presets_to_library(str(presets_dir), lib)

    assert [c.name for c in lib.list()] == ["Alice"]


def test_import_skips_unparseable_preset_file(tmp_path, lib):
    presets_dir = tmp_path / "presets"
    _write_preset(presets_dir, "Alice", {"voice": "af_bella"})
    (presets_dir / "Broken.json").write_text("{not valid json", encoding="utf-8")

    import_presets_to_library(str(presets_dir), lib)

    assert [c.name for c in lib.list()] == ["Alice"]


def test_import_twice_does_not_duplicate(tmp_path, lib):
    presets_dir = tmp_path / "presets"
    _write_preset(presets_dir, "Alice", {"voice": "af_bella"})

    first = import_presets_to_library(str(presets_dir), lib)
    second = import_presets_to_library(str(presets_dir), lib)

    assert first == second
    assert len(lib.list()) == 1


def test_import_filters_preset_keys(tmp_path, lib):
    presets_dir = tmp_path / "presets"
    _write_preset(presets_dir, "Alice", {"voice": "af_bella", "out_dir": "/etc"})
    import_presets_to_library(str(presets_dir), lib)
    assert lib.list()[0].preset_data == {"voice": "af_bella"}


# -- the exact-match link --------------------------------------------------------------

def test_exact_match_links_and_the_rest_stay_local(tmp_path, lib):
    presets_dir = tmp_path / "presets"
    _write_preset(presets_dir, "Alice", {"voice": "af_bella", "speed": 1.0})
    _write_preset(presets_dir, "Bob", {"voice": "am_michael"})
    import_presets_to_library(str(presets_dir), lib)

    same = Character.from_preset_dict("Alice", {"voice": "af_bella", "speed": 1.0})
    edited = Character.from_preset_dict("Bob", {"voice": "am_adam"})  # name matches, settings don't
    other = Character.from_preset_dict("Carol", {"voice": "af_bella", "speed": 1.0})  # settings match, name doesn't
    doc = Document(characters=[same, edited, other])

    linked = link_exact_matches(doc, lib.list())

    alice_entry = next(e for e in lib.list() if e.name == "Alice")
    assert linked == [same]
    assert same.library_id == alice_entry.library_id
    assert edited.library_id is None
    assert other.library_id is None


def test_exact_match_leaves_already_linked_characters_alone(lib):
    entry_id = lib.save(Character.from_preset_dict("Alice", {"voice": "af_bella"}))
    record = Character.from_preset_dict("Alice", {"voice": "af_bella"}, library_id="elsewhere")
    doc = Document(characters=[record])
    assert link_exact_matches(doc, lib.list()) == []
    assert record.library_id == "elsewhere"
    assert entry_id
