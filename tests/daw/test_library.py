"""Tests for kokoro_gui/daw/library.py: the global character library store
(phase 3 step 2) and `resolve_characters` (step 3)."""
import json
import logging
import os

import pytest

from kokoro_gui.daw import library as library_mod
from kokoro_gui.daw.library import (
    CharacterLibrary, linked_copy, resolve_characters, write_through,
)
from kokoro_gui.daw.models import Character, Document


@pytest.fixture
def lib(tmp_path):
    return CharacterLibrary(str(tmp_path / "characters"))


def _entry(name="Narrator", voice="af_bella", color="#4f8fe6", **kwargs):
    return Character.from_preset_dict(name, {"voice": voice, "speed": 1.0}, highlight_color=color, **kwargs)


# -- the store -----------------------------------------------------------------

def test_empty_or_missing_library_lists_nothing(lib):
    assert lib.list() == []
    assert lib.get("anything") is None
    assert lib.mtime() == 0


def test_save_mints_an_id_and_writes_one_file(lib):
    character = _entry()
    library_id = lib.save(character)

    assert library_id
    assert character.library_id is None  # save doesn't stamp the caller's record
    path = os.path.join(lib.root, f"{library_id}.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["library_id"] == library_id
    assert data["id"] == library_id
    assert data["name"] == "Narrator"
    assert data["preset_data"] == {"voice": "af_bella", "speed": 1.0}


def test_save_get_list_delete(lib):
    a = lib.save(_entry("Zed", "am_adam"))
    b = lib.save(_entry("alice", "af_bella"))

    got = lib.get(a)
    assert got.name == "Zed"
    assert got.library_id == a
    assert got.preset_data["voice"] == "am_adam"
    assert [c.name for c in lib.list()] == ["alice", "Zed"]  # case-insensitive name order

    assert lib.delete(b) is True
    assert lib.delete(b) is False
    assert [c.library_id for c in lib.list()] == [a]


def test_save_with_existing_library_id_overwrites_that_entry(lib):
    library_id = lib.save(_entry())
    entry = lib.get(library_id)
    entry.preset_data["voice"] = "bf_emma"
    assert lib.save(entry) == library_id
    assert lib.get(library_id).preset_data["voice"] == "bf_emma"
    assert len(lib.list()) == 1


def test_save_is_atomic_and_leaves_no_tmp_file(lib, monkeypatch):
    library_id = lib.save(_entry())
    replaced = []
    real_replace = os.replace
    monkeypatch.setattr(library_mod.os, "replace", lambda src, dst: (replaced.append((src, dst)), real_replace(src, dst)))

    entry = lib.get(library_id)
    lib.save(entry)

    target = os.path.join(lib.root, f"{library_id}.json")
    assert replaced == [(target + ".tmp", target)]
    assert not any(name.endswith(".tmp") for name in os.listdir(lib.root))


def test_corrupt_file_is_skipped_and_logged(lib, caplog):
    good = lib.save(_entry())
    with open(os.path.join(lib.root, "broken.json"), "w", encoding="utf-8") as f:
        f.write("{not json")
    with open(os.path.join(lib.root, "list.json"), "w", encoding="utf-8") as f:
        f.write("[1, 2]")

    with caplog.at_level(logging.WARNING, logger="kokoro_gui.daw.library"):
        entries = lib.list()

    assert [c.library_id for c in entries] == [good]
    assert "broken.json" in caplog.text
    assert "list.json" in caplog.text
    assert lib.get("broken") is None


def test_ids_are_sanitised_with_basename(lib, tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"name": "Evil", "preset_data": {}}), encoding="utf-8")

    assert lib.get("../outside") is None
    assert lib.get("") is None
    assert lib.get(None) is None
    assert lib.delete("../outside") is False
    assert outside.exists()

    library_id = lib.save(_entry(library_id="../../escape"))
    assert library_id == "escape"
    assert os.path.isfile(os.path.join(lib.root, "escape.json"))
    assert not (tmp_path.parent / "escape.json").exists()


def test_file_stem_wins_over_ids_inside_the_file(lib):
    os.makedirs(lib.root)
    with open(os.path.join(lib.root, "abc.json"), "w", encoding="utf-8") as f:
        json.dump({"name": "X", "id": "other", "library_id": "other", "preset_data": {}}, f)
    entry = lib.get("abc")
    assert entry.library_id == "abc"
    assert entry.id == "abc"


def test_preset_whitelist_applies_to_library_files(lib):
    os.makedirs(lib.root)
    with open(os.path.join(lib.root, "abc.json"), "w", encoding="utf-8") as f:
        json.dump({"name": "X", "preset_data": {"voice": "af_bella", "out_dir": "/etc"}}, f)
    assert lib.get("abc").preset_data == {"voice": "af_bella"}


def test_mtime_moves_on_save_and_delete(lib):
    library_id = lib.save(_entry())
    first = lib.mtime()
    assert first > 0
    os.utime(os.path.join(lib.root, f"{library_id}.json"), ns=(first + 10_000_000, first + 10_000_000))
    assert lib.mtime() > first


def test_root_none_reads_the_module_constant_at_call_time(tmp_path, monkeypatch):
    monkeypatch.setattr(library_mod, "LIBRARY_DIR", str(tmp_path / "patched"))
    store = CharacterLibrary()
    store.save(_entry())
    assert os.path.isdir(tmp_path / "patched")
    assert len(store.list()) == 1


def test_linked_copy_has_a_fresh_document_id(lib):
    entry = lib.get(lib.save(_entry()))
    record = linked_copy(entry)
    assert record.library_id == entry.library_id
    assert record.id != entry.library_id
    assert record.name == entry.name
    assert record.preset_data == entry.preset_data
    record.preset_data["voice"] = "changed"
    assert entry.preset_data["voice"] == "af_bella"


# -- resolution ------------------------------------------------------------------

def test_resolve_hit_refreshes_voice_colour_engine_not_name_or_id(lib):
    library_id = lib.save(_entry("Library name", "am_adam", "#e0655c", backend_id="audio8"))
    record = Character.from_preset_dict("Project name", {"voice": "af_bella"}, highlight_color="#000000",
                                        library_id=library_id)
    original_id = record.id
    doc = Document(characters=[record])

    report = resolve_characters(doc, [lib])

    assert report.changed == [original_id]
    assert report.missing == []
    assert record.preset_data == {"voice": "am_adam", "speed": 1.0}
    assert record.highlight_color == "#e0655c"
    assert record.backend_id == "audio8"
    assert record.name == "Project name"
    assert record.id == original_id


def test_resolve_twice_reports_no_change(lib):
    library_id = lib.save(_entry())
    doc = Document(characters=[linked_copy(lib.get(library_id))])
    assert resolve_characters(doc, [lib]).changed == []


def test_resolve_miss_keeps_the_snapshot(lib):
    record = Character.from_preset_dict("Narrator", {"voice": "af_bella"}, library_id="gone")
    doc = Document(characters=[record])

    report = resolve_characters(doc, [lib])

    assert report.missing == [record.id]
    assert report.changed == []
    assert record.preset_data == {"voice": "af_bella"}


def test_resolve_leaves_local_characters_alone(lib):
    local = Character.from_preset_dict("Guest", {"voice": "am_adam"})
    doc = Document(characters=[local])
    report = resolve_characters(doc, [lib])
    assert report.changed == [] and report.missing == []
    assert local.preset_data == {"voice": "am_adam"}


def test_resolve_asks_stores_in_order(tmp_path):
    project_store = CharacterLibrary(str(tmp_path / "project"))
    global_store = CharacterLibrary(str(tmp_path / "global"))
    shared = _entry(voice="af_bella", library_id="shared")
    global_store.save(shared)
    project_only = _entry(voice="am_adam", library_id="only-global")
    global_store.save(project_only)
    shadow = _entry(voice="bf_emma", library_id="shared")
    project_store.save(shadow)

    a = Character.from_preset_dict("A", {}, library_id="shared")
    b = Character.from_preset_dict("B", {}, library_id="only-global")
    doc = Document(characters=[a, b])

    resolve_characters(doc, [project_store, global_store])

    assert a.preset_data["voice"] == "bf_emma"  # the first store wins
    assert b.preset_data["voice"] == "am_adam"  # falls through to the second


def test_write_through_updates_the_entry_but_keeps_its_name(lib):
    library_id = lib.save(_entry("Library name", "af_bella"))
    record = linked_copy(lib.get(library_id))
    record.name = "Renamed here"
    record.preset_data["voice"] = "am_adam"
    record.highlight_color = "#123456"

    assert write_through(record, lib) is True

    entry = lib.get(library_id)
    assert entry.name == "Library name"
    assert entry.preset_data["voice"] == "am_adam"
    assert entry.highlight_color == "#123456"


def test_write_through_skips_local_and_missing(lib):
    local = _entry()
    assert write_through(local, lib) is False
    missing = _entry(library_id="gone")
    assert write_through(missing, lib) is False
    assert lib.list() == []
