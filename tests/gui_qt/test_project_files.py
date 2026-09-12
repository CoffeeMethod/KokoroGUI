"""Tests for the File menu / project lifecycle (section 7 of
Claude/PLAN_ui_shell_redesign.md) and kokoro_gui/qt/project.py."""
import json
import os

from PySide6.QtGui import QTextCursor

from kokoro_gui.daw.models import Character
from kokoro_gui.qt import project as project_io


def _type(editor, text):
    cursor = editor.textCursor()
    cursor.select(QTextCursor.SelectionType.Document)
    cursor.insertText(text)


# -- project.py ---------------------------------------------------------------------


def test_remember_recent_moves_to_front_dedupes_and_caps(tmp_path):
    settings = {}
    for i in range(12):
        project_io.remember_recent(settings, str(tmp_path / f"p{i}.json"))
    project_io.remember_recent(settings, str(tmp_path / "p3.json"))

    recent = settings["recent_projects"]
    assert len(recent) == project_io.MAX_RECENT
    assert recent[0].endswith("p3.json")
    assert sum(1 for p in recent if p.endswith("p3.json")) == 1
    assert settings["last_project"].endswith("p3.json")


def test_save_and_load_roundtrip_with_project_settings(tmp_path):
    from kokoro_gui.daw.models import Document

    doc = Document.from_plain_text("hello", characters=[Character.from_preset_dict("A", {})])
    path = str(tmp_path / "proj.json")

    project_io.save_project(doc, path, {"export": {"format": "flac"}})
    loaded = project_io.load_project(path)

    assert loaded.document.text == "hello"
    assert loaded.document.characters[0].name == "A"
    assert loaded.project_settings == {"export": {"format": "flac"}}
    assert project_io.load_project(str(tmp_path / "missing.json")) is None


def test_tbaw_is_recognized_but_not_implemented(tmp_path):
    import pytest

    assert project_io.format_for_path("x.tbaw") == "tbaw"
    (tmp_path / "x.tbaw").write_bytes(b"PK")
    with pytest.raises(NotImplementedError):
        project_io.load_project(str(tmp_path / "x.tbaw"))


def test_new_document_inherits_characters_as_copies():
    from kokoro_gui.daw.models import Document

    alice = Character.from_preset_dict("Alice", {"voice": "af_heart"})
    previous = Document.from_plain_text("old text", characters=[alice])

    fresh = project_io.new_document_from(previous)

    assert fresh.text == ""
    assert [c.name for c in fresh.characters] == ["Alice"]
    assert fresh.characters[0] is not alice
    assert fresh.tracks[0].character_id == fresh.characters[0].id


# -- app-level --------------------------------------------------------------------------


def test_launch_opens_document_json_and_records_it_as_last_project(qt_app):
    import kokoro_gui.qt.app as qt_app_module

    assert qt_app.project_path == os.path.abspath(qt_app_module.DOCUMENT_FILE)
    assert qt_app.settings["last_project"] == qt_app.project_path
    assert qt_app.settings["recent_projects"][0] == qt_app.project_path


def test_save_as_then_open_restores_text_and_clips(qt_app, tmp_path):
    _type(qt_app.editor, "hello world")
    alice = qt_app.document.characters[0]
    clip = qt_app.document.assign_character_to_range(0, 5, alice.id)
    target = str(tmp_path / "story")

    qt_app.save_project_as(target)

    assert qt_app.project_path.endswith("story.json")
    assert os.path.exists(qt_app.project_path)
    assert qt_app.windowTitle().startswith("story")
    assert qt_app.settings["recent_projects"][0] == qt_app.project_path

    qt_app.new_project()
    assert qt_app.document.text == ""
    assert qt_app.editor.toPlainText() == ""
    assert qt_app.project_path is None

    qt_app.open_project(target + ".json")
    assert qt_app.document.text == "hello world"
    assert qt_app.editor.toPlainText() == "hello world"
    assert qt_app.document.clip_covering(2).id == clip.id


def test_new_project_inherits_characters_and_clears_selection(qt_app):
    bob = Character.from_preset_dict("Bob", {})
    qt_app.document.characters.append(bob)
    _type(qt_app.editor, "hello")
    qt_app.document.assign_character_to_range(0, 5, bob.id)
    qt_app.editor.rehighlight()
    qt_app.selection.select_clip(qt_app.document.clips[0].id)

    qt_app.new_project()

    assert [c.name for c in qt_app.document.characters] == ["Default", "Bob"]
    assert qt_app.document.clips == []
    assert qt_app.selection.kind == "none"
    assert qt_app.transcript_dock.character_combo.findData(qt_app.document.characters[1].id) >= 0


def test_recent_menu_lists_projects_and_opens_them(qt_app, tmp_path):
    qt_app.save_project_as(str(tmp_path / "one.json"))
    qt_app.save_project_as(str(tmp_path / "two.json"))

    texts = [a.text() for a in qt_app.recent_menu.actions()]
    assert texts[:2] == ["two", "one"]

    next(a for a in qt_app.recent_menu.actions() if a.text() == "one").trigger()
    assert qt_app.project_path.endswith("one.json")


def test_open_missing_project_warns_and_forgets_it(qt_app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a)))
    ghost = str(tmp_path / "ghost.json")
    project_io.remember_recent(qt_app.settings, ghost)
    qt_app._rebuild_recent_menu()

    qt_app.open_project(ghost)

    assert warned
    assert not any(p.endswith("ghost.json") for p in qt_app.settings["recent_projects"])


def test_import_text_add_inserts_at_caret_on_native_undo(qt_app, tmp_path):
    _type(qt_app.editor, "start ")
    qt_app.engine.extract_text_from_file.return_value = "imported words"
    src = tmp_path / "in.txt"
    src.write_text("imported words", encoding="utf-8")

    qt_app.import_text(str(src), target="add")

    assert qt_app.document.text == "start imported words"
    assert qt_app.editor.toPlainText() == "start imported words"
    qt_app.undo()
    assert qt_app.document.text == "start "


def test_import_text_new_starts_a_fresh_project_with_the_text(qt_app, tmp_path):
    _type(qt_app.editor, "old")
    qt_app.engine.extract_text_from_file.return_value = "chapter one"
    src = tmp_path / "in.txt"
    src.write_text("chapter one", encoding="utf-8")

    qt_app.import_text(str(src), target="new")

    assert qt_app.document.text == "chapter one"
    assert qt_app.project_path is None


def test_autosave_writes_the_current_project_path(qt_app, tmp_path):
    qt_app.save_project_as(str(tmp_path / "auto.json"))
    _type(qt_app.editor, "typed later")

    qt_app.save_settings()

    with open(tmp_path / "auto.json", encoding="utf-8") as f:
        data = json.load(f)
    assert "".join(r["text"] for r in data["runs"]) == "typed later"
    assert "project_settings" in data
