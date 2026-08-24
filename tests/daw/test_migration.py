"""Tests for kokoro_gui/daw/migration.py's presets/settings -> Document
first-load migration. tmp_path-isolated preset directories, mirroring the
isolation convention tests/conftest.py's isolated_dirs fixture uses for
kokoro_engine's storage dirs."""
import json

from kokoro_gui.daw.migration import DEFAULT_CHARACTER_NAME, migrate_legacy_settings_to_document
from kokoro_gui.daw.models import DEFAULT_HIGHLIGHT_PALETTE


def _write_preset(presets_dir, name, data):
    presets_dir.mkdir(parents=True, exist_ok=True)
    (presets_dir / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")


def test_migration_with_no_presets_seeds_default_character_from_settings(tmp_path):
    settings = {"voice": "af_bella", "speed": 1.2, "out_dir": "/should/be/dropped"}
    doc = migrate_legacy_settings_to_document(settings, str(tmp_path / "presets"))

    assert len(doc.characters) == 1
    character = doc.characters[0]
    assert character.name == DEFAULT_CHARACTER_NAME
    assert character.preset_data == {"voice": "af_bella", "speed": 1.2}
    assert len(doc.tracks) == 1
    assert doc.tracks[0].character_id == character.id
    assert doc.text == ""
    assert doc.clips == []


def test_migration_with_nonexistent_presets_dir_seeds_default(tmp_path):
    doc = migrate_legacy_settings_to_document({"voice": "af_bella"}, str(tmp_path / "does_not_exist"))
    assert len(doc.characters) == 1
    assert doc.characters[0].name == DEFAULT_CHARACTER_NAME


def test_migration_wraps_each_preset_file_as_a_character(tmp_path):
    presets_dir = tmp_path / "presets"
    _write_preset(presets_dir, "Alice", {"voice": "af_bella", "speed": 1.0})
    _write_preset(presets_dir, "Bob", {"voice": "am_michael", "speed": 0.9})

    doc = migrate_legacy_settings_to_document({}, str(presets_dir))

    names = {c.name for c in doc.characters}
    assert names == {"Alice", "Bob"}
    assert len(doc.tracks) == 2
    track_character_ids = {t.character_id for t in doc.tracks}
    assert track_character_ids == {c.id for c in doc.characters}


def test_migration_assigns_distinct_highlight_colors_from_palette(tmp_path):
    presets_dir = tmp_path / "presets"
    for i in range(len(DEFAULT_HIGHLIGHT_PALETTE) + 1):
        _write_preset(presets_dir, f"Character{i}", {"voice": "af_bella"})

    doc = migrate_legacy_settings_to_document({}, str(presets_dir))
    colors = [c.highlight_color for c in doc.characters]
    # Cycles back around once there are more characters than palette entries.
    assert colors[0] == colors[len(DEFAULT_HIGHLIGHT_PALETTE)]
    assert len(set(colors[: len(DEFAULT_HIGHLIGHT_PALETTE)])) == len(DEFAULT_HIGHLIGHT_PALETTE)


def test_migration_ignores_fx_subdirectory(tmp_path):
    presets_dir = tmp_path / "presets"
    _write_preset(presets_dir, "Alice", {"voice": "af_bella"})
    _write_preset(presets_dir / "fx", "reverb_heavy", {"reverb_enabled": True})

    doc = migrate_legacy_settings_to_document({}, str(presets_dir))

    assert [c.name for c in doc.characters] == ["Alice"]


def test_migration_skips_unparseable_preset_file(tmp_path):
    presets_dir = tmp_path / "presets"
    _write_preset(presets_dir, "Alice", {"voice": "af_bella"})
    presets_dir.mkdir(parents=True, exist_ok=True)
    (presets_dir / "Broken.json").write_text("{not valid json", encoding="utf-8")

    doc = migrate_legacy_settings_to_document({}, str(presets_dir))

    assert [c.name for c in doc.characters] == ["Alice"]
