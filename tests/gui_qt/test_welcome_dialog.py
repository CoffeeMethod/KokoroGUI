"""Welcome dialog (grill WF2, revised): kokoro_gui/qt/welcome_dialog.py and
the `show_welcome` / File > Welcome... wiring in app.py."""
import os

from PySide6.QtGui import QTextCursor

from kokoro_gui.daw.models import Character
from kokoro_gui.qt.welcome_dialog import MISSING_SUFFIX


def _type(editor, text):
    cursor = editor.textCursor()
    cursor.select(QTextCursor.SelectionType.Document)
    cursor.insertText(text)


def test_default_on_and_checkbox_turns_it_off(qt_app):
    assert qt_app.settings["show_welcome"] is True
    dialog = qt_app.show_welcome_if_enabled()
    assert dialog is not None and dialog.isVisible()

    dialog.show_at_startup.setChecked(False)
    dialog.reject()
    assert qt_app.settings["show_welcome"] is False
    assert qt_app.show_welcome_if_enabled() is None
    # The menu action still brings it back regardless.
    assert qt_app.show_welcome().isVisible()


def test_fixture_never_opens_it(qt_app):
    assert qt_app.welcome_dialog is None


def test_lists_recent_with_current_first_and_resume_default(qt_app, tmp_path):
    qt_app.save_project_as(str(tmp_path / "one.json"))
    qt_app.save_project_as(str(tmp_path / "two.json"))
    dialog = qt_app.show_welcome()

    assert dialog.paths()[:2] == [qt_app.project_path, str(tmp_path / "one.json")]
    assert dialog.selected_path() == qt_app.project_path
    assert dialog.open_btn.text() == "Resume"
    assert dialog.characters_label.text() == str(len(qt_app.document.characters))

    dialog.list.setCurrentRow(1)
    assert dialog.open_btn.text() == "Open"
    assert dialog.path_label.text() == str(tmp_path / "one.json")


def test_missing_row_is_disabled_and_can_be_removed(qt_app, tmp_path):
    from kokoro_gui.qt import project as project_io

    ghost = str(tmp_path / "ghost.json")
    project_io.remember_recent(qt_app.settings, ghost)
    qt_app.settings["last_project"] = qt_app.project_path
    dialog = qt_app.show_welcome()

    row = dialog.paths().index(ghost)
    item = dialog.list.item(row)
    assert item.text().endswith(MISSING_SUFFIX)
    assert not item.flags() & item.flags().ItemIsEnabled

    dialog.remove_from_recent(ghost)
    assert ghost not in dialog.paths()
    assert ghost not in qt_app.settings["recent_projects"]
    assert all(a.text() != "ghost" for a in qt_app.recent_menu.actions())


def test_clear_list_keeps_only_current_project(qt_app, tmp_path):
    qt_app.save_project_as(str(tmp_path / "one.json"))
    qt_app.save_project_as(str(tmp_path / "two.json"))
    dialog = qt_app.show_welcome()
    dialog.clear_recent()
    assert qt_app.settings["recent_projects"] == []
    assert dialog.paths() == [qt_app.project_path]


def test_choose_opens_other_and_resume_is_noop(qt_app, tmp_path):
    _type(qt_app.editor, "story text")
    qt_app.save_project_as(str(tmp_path / "story.json"))
    story = qt_app.project_path
    qt_app.new_project()
    qt_app.save_project_as(str(tmp_path / "other.json"))
    other_doc = qt_app.document

    dialog = qt_app.show_welcome()
    dialog.choose(qt_app.project_path)
    assert not dialog.isVisible()
    assert qt_app.document is other_doc

    dialog = qt_app.show_welcome()
    dialog.choose(story)
    assert qt_app.project_path == story
    assert qt_app.editor.toPlainText() == "story text"
    assert qt_app.windowTitle().startswith("story")


def test_new_project_inherits_characters(qt_app):
    qt_app.document.characters.append(Character.from_preset_dict("Bob", {}))
    dialog = qt_app.show_welcome()
    dialog.new_project()
    assert not dialog.isVisible()
    assert qt_app.project_path is None
    assert [c.name for c in qt_app.document.characters] == ["Default", "Bob"]


def test_new_from_text_starts_fresh_project_with_the_text(qt_app, tmp_path):
    _type(qt_app.editor, "old")
    qt_app.engine.extract_text_from_file.return_value = "Once upon a time."
    src = tmp_path / "chapter.txt"
    src.write_text("Once upon a time.", encoding="utf-8")
    dialog = qt_app.show_welcome()
    dialog.new_from_text(str(src))
    assert qt_app.project_path is None
    assert qt_app.editor.toPlainText() == "Once upon a time."


def test_welcome_action_sits_in_file_menu(qt_app):
    texts = [a.text() for a in qt_app.file_menu.actions()]
    assert "&Welcome..." in texts
    assert texts.index("&Welcome...") > texts.index("Recent")
