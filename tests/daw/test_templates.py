"""Tests for kokoro_gui/daw/templates.py: the template store (plan 25) and the
audiobook credit lines."""
import json
import os

import pytest

from kokoro_gui.daw import templates as templates_mod
from kokoro_gui.daw.arrangement import is_heading_text
from kokoro_gui.daw.templates import (
    credit_texts, delete_template, list_templates, load_template, safe_name, save_template,
)


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(templates_mod, "TEMPLATES_DIR", str(tmp_path / "templates"))
    return tmp_path / "templates"


SETTINGS = {"gap_s": 0.4, "paragraph_gap_s": 0.9, "duck_db": -12.0, "markers": [{"id": "m", "seconds": 3.0}],
            "sources": {"a": {"path": "C:/x.wav"}}, "source_track": {"path": "/a.wav", "offset_s": 1.0},
            "ripple": False, "video_path": "x.mp4", "gap_jitter_s": [0.1, 0.3],
            "nested": {"path": "/etc/passwd", "keep": 1}, "note": "C:\\Users\\me\\x.wav"}


def test_round_trip(store):
    stem = save_template("Episode", SETTINGS, ["lib1", "lib2"], [("Cold open", "Hi."), ("Intro", "")])
    assert stem == "Episode"
    assert os.path.isfile(store / "Episode.json")
    loaded = load_template("Episode")
    assert loaded.name == "Episode"
    assert loaded.characters == ["lib1", "lib2"]
    assert [(s.title, s.text) for s in loaded.sections] == [("Cold open", "Hi."), ("Intro", "")]
    assert loaded.settings["gap_s"] == 0.4 and loaded.settings["gap_jitter_s"] == [0.1, 0.3]
    assert loaded.settings["ripple"] is False


def test_settings_drop_markers_sources_and_paths(store):
    save_template("T", SETTINGS, [], [])
    settings = load_template("T").settings
    for key in ("markers", "sources", "source_track", "video_path", "note"):
        assert key not in settings
    assert settings["nested"] == {"keep": 1}


def test_names_are_sanitised(store):
    assert safe_name("../x") == "x"
    assert safe_name("a:b") == "a_b"
    assert safe_name("..\\..\\evil") == "evil"
    assert safe_name("..") is None and safe_name("") is None and safe_name("  ") is None
    assert safe_name("CON") is None and safe_name(".hidden") is None and safe_name(None) is None
    assert save_template("../x", {}, [], []) == "x"
    assert save_template("a:b", {}, [], []) == "a_b"
    assert sorted(os.listdir(store)) == ["a_b.json", "x.json"]
    assert save_template("..", {}, [], []) is None
    assert load_template("../x").name == "../x"  # read through the same rule: the file is x.json


def test_invalid_files_are_ignored_by_list(store):
    save_template("Good", {}, [], [("A", "")])
    store.mkdir(exist_ok=True)
    (store / "broken.json").write_text("{not json", encoding="utf-8")
    (store / "list.json").write_text("[1, 2]", encoding="utf-8")
    (store / "other.json").write_text(json.dumps({"format": "something-else", "version": 1}), encoding="utf-8")
    (store / "future.json").write_text(json.dumps({"format": "kokorogui-template", "version": 99}), encoding="utf-8")
    (store / "notes.txt").write_text("hi", encoding="utf-8")
    assert [t.name for t in list_templates()] == ["Good"]
    assert load_template("broken") is None


def test_missing_store_lists_nothing():
    assert list_templates() == []
    assert load_template("nothing") is None
    assert delete_template("nothing") is False


def test_wrong_types_fall_back_and_unknown_keys_are_ignored(store):
    store.mkdir()
    data = {"format": "kokorogui-template", "version": 1, "name": 5, "extra": 1,
            "settings": [1], "characters": ["a", 3, "../b", "a", ".c"],
            "sections": ["x", {"title": 4, "text": None}, {"title": " Ad   break ", "text": "Buy."}]}
    (store / "odd.json").write_text(json.dumps(data), encoding="utf-8")
    loaded = load_template("odd")
    assert loaded.name == "odd"
    assert loaded.settings == {}
    assert loaded.characters == ["a", "b"]
    assert [(s.title, s.text) for s in loaded.sections] == [("", ""), ("Ad break", "Buy.")]


def test_limits_on_sections_and_text(store):
    store.mkdir()
    many = [{"title": f"S{i}", "text": "x"} for i in range(templates_mod.MAX_SECTIONS + 20)]
    (store / "many.json").write_text(json.dumps(
        {"format": "kokorogui-template", "version": 1, "name": "many", "sections": many}), encoding="utf-8")
    assert len(load_template("many").sections) == templates_mod.MAX_SECTIONS

    big = "y" * (templates_mod.MAX_TEXT_CHARS // 2 + 10)
    (store / "big.json").write_text(json.dumps(
        {"format": "kokorogui-template", "version": 1, "name": "big",
         "sections": [{"title": "a", "text": big}, {"title": "b", "text": big}, {"title": "c", "text": big}]}),
        encoding="utf-8")
    total = sum(len(s.text) for s in load_template("big").sections)
    assert total == templates_mod.MAX_TEXT_CHARS


def test_an_oversize_file_is_skipped(store, monkeypatch):
    save_template("Fat", {}, [], [("A", "")])
    monkeypatch.setattr(templates_mod, "MAX_FILE_BYTES", 10)
    assert list_templates() == []
    assert load_template("Fat") is None


def test_saving_again_replaces_and_delete_removes(store):
    save_template("T", {}, [], [("One", "")])
    save_template("T", {}, [], [("Two", "")])
    assert [s.title for s in load_template("T").sections] == ["Two"]
    assert not any(n.endswith(".tmp") for n in os.listdir(store))
    assert delete_template("T") is True
    assert list_templates() == []


# -- credits -------------------------------------------------------------------

FULL = {"title": "Moby Dick", "subtitle": "or, The Whale", "author": "Herman Melville", "narrator": "A. Reader",
        "publisher": "Acme Audio", "year": "2026"}


def test_credit_texts_with_every_field():
    opening, closing = credit_texts(FULL)
    assert opening == "Moby Dick, or, The Whale. Written by Herman Melville. Narrated by A. Reader."
    assert closing == ("You have been listening to Moby Dick. Written by Herman Melville. Narrated by A. Reader. "
                       "Copyright 2026 by Acme Audio. The end.")


def test_credit_texts_drop_empty_fields():
    opening, closing = credit_texts({"title": "Moby Dick", "author": "Herman Melville"})
    assert opening == "Moby Dick. Written by Herman Melville."
    assert closing == "You have been listening to Moby Dick. Written by Herman Melville. The end."
    assert credit_texts({"title": "T", "year": "2026"})[1] == "You have been listening to T. Copyright 2026. The end."
    assert credit_texts({"title": "T", "publisher": "P"})[1] == "You have been listening to T. Copyright by P. The end."
    assert credit_texts({"subtitle": "Only"})[0] == "Only."


def test_credit_texts_with_nothing_has_no_opening():
    assert credit_texts({}) == ("", "The end.")
    assert credit_texts(None) == ("", "The end.")
    # A number is read as text, any other wrong type as empty.
    assert "Copyright 2026" in credit_texts({"title": ["x"], "year": 2026})[1]
    assert credit_texts({"title": ["x"], "author": True}) == ("", "The end.")


def test_credit_fields_are_flattened_and_punctuation_is_not_doubled():
    opening, _ = credit_texts({"title": "Why?", "subtitle": "A  Question", "author": "J. Doe."})
    assert opening == "Why? A Question. Written by J. Doe."


def test_no_credit_line_reads_as_a_heading():
    # Plan 24: the first line of a subproject that is a short title with no ending
    # punctuation becomes a heading. Every credit sentence ends in a full stop.
    for fields in (FULL, {"title": "Moby Dick"}, {"author": "X"}, {"subtitle": "S"}, {"narrator": "N"}):
        for text in credit_texts(fields):
            if text:
                assert not is_heading_text(text), text
