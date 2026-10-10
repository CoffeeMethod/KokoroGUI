"""Project templates and credits (plan 25): Save as Template, New from
Template, the welcome button, and Add Credits."""
import os

import pytest
from PySide6.QtWidgets import QDialog, QMessageBox

from kokoro_gui.daw import library as library_module, templates
from kokoro_gui.daw.models import Character
from kokoro_gui.qt.template_dialogs import CreditsDialog, NewFromTemplateDialog, SaveTemplateDialog

LONG = "word " * 600  # 3,000 characters: over the keep-text limit


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(templates, "TEMPLATES_DIR", str(tmp_path / "templates"))


def _show(qt_app):
    """A project with three subprojects: a short intro, a long chapter, an empty outro,
    and its own pacing settings."""
    document = qt_app.document
    document.settings["gap_s"] = 0.7
    document.settings["chapter_gap_s"] = 3.0
    document.settings["markers"] = [{"id": "m1", "seconds": 4.0, "name": "Here"}]
    for title in ("Cold open", "Chapter", "Outro"):
        qt_app.new_subproject(len(qt_app.document.text), title=title)
        if title != "Outro":
            document.replace_text(len(document.text), 0, 2, document.text + "\n\n")
    kids = {c.title(): c for c in qt_app.children.values()}
    kids["Cold open"].document.set_plain_text("Welcome back.")
    kids["Chapter"].document.set_plain_text(LONG)
    return kids


def test_save_as_template_defaults_keep_short_text_only(qt_app):
    _show(qt_app)
    dialog = SaveTemplateDialog(qt_app, "Show", qt_app.template_sections())
    assert [box.isChecked() for box in dialog.keep_boxes] == [True, False, False]
    assert [box.isEnabled() for box in dialog.keep_boxes] == [True, True, False]
    name, rows = dialog.values()
    assert name == "Show"
    assert rows == [("Cold open", "Welcome back."), ("Chapter", ""), ("Outro", "")]


def test_save_button_needs_a_usable_name(qt_app):
    dialog = SaveTemplateDialog(qt_app, "", [])
    from PySide6.QtWidgets import QDialogButtonBox
    save = dialog.buttons.button(QDialogButtonBox.StandardButton.Save)
    assert not save.isEnabled()
    dialog.name_edit.setText("..")
    assert not save.isEnabled()
    dialog.name_edit.setText("Weekly")
    assert save.isEnabled()


def test_a_saved_template_makes_the_same_structure(qt_app):
    _show(qt_app)
    rows = SaveTemplateDialog(qt_app, "Show", qt_app.template_sections()).values()[1]
    assert qt_app.save_as_template("Show", rows) == "Show"

    saved = templates.load_template("Show")
    assert saved.settings["gap_s"] == 0.7 and "markers" not in saved.settings
    assert [s.title for s in saved.sections] == ["Cold open", "Chapter", "Outro"]

    old_root = qt_app.root.project_id
    kids = qt_app.new_from_template(saved)
    assert qt_app.root.project_id != old_root
    assert [c.title() for c in kids] == ["Cold open", "Chapter", "Outro"]
    document = qt_app.document
    assert [document.clip_text(c) for c in document.nested_clips()] == ["Cold open", "Chapter", "Outro"]
    assert [c.document.text for c in kids] == ["Welcome back.", "", ""]
    # The same gap settings, on the project and on each subproject.
    assert document.settings["gap_s"] == 0.7 and document.settings["chapter_gap_s"] == 3.0
    assert "markers" not in document.settings
    assert all(c.document.settings["gap_s"] == 0.7 for c in kids)


def test_a_template_with_no_subprojects_only_sets_the_project_up(qt_app):
    qt_app.document.settings["paragraph_gap_s"] = 1.5
    assert qt_app.save_as_template("Plain", []) == "Plain"
    qt_app.document.settings["paragraph_gap_s"] = 0.2
    assert qt_app.new_from_template(templates.load_template("Plain")) == []
    assert qt_app.document.settings["paragraph_gap_s"] == 1.5
    assert qt_app.document.nested_clips() == []


def test_a_section_without_a_title_gets_a_numbered_one(qt_app):
    kids = qt_app.new_from_template(templates.Template("T", sections=[templates.Section(""), templates.Section("")]))
    assert [c.title() for c in kids] == ["Subproject 1", "Subproject 2"]


def test_linked_library_characters_come_along(qt_app):
    library = qt_app.character_library
    narrator = Character.from_preset_dict("Narrator", {"voice": "af_bella", "speed": 1.0})
    library_id = library.save(narrator)
    qt_app.new_project()
    template = templates.Template("T", characters=[library_id, "gone-from-the-library"])
    qt_app.new_from_template(template)
    linked = [c.library_id for c in qt_app.document.characters]
    assert linked.count(library_id) == 1
    assert "gone-from-the-library" not in linked


def test_save_template_stores_only_linked_library_characters(qt_app):
    library_id = qt_app.character_library.save(Character.from_preset_dict("N", {"voice": "af_bella"}))
    qt_app.new_project()
    qt_app.document.characters.append(Character.from_preset_dict("Local", {"voice": "am_adam"}))
    qt_app.document.characters.append(Character.from_preset_dict(
        "Scoped", {"voice": "am_adam"}, library_id=library_module.new_project_scope_id()))
    qt_app.save_as_template("Cast", [])
    assert templates.load_template("Cast").characters == [library_id]


def test_saving_over_a_template_asks(qt_app, monkeypatch):
    qt_app.save_as_template("T", [("One", "")])
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    assert qt_app.save_as_template("T", [("Two", "")]) is None
    assert [s.title for s in templates.load_template("T").sections] == ["One"]
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    assert qt_app.save_as_template("T", [("Two", "")]) == "T"
    assert [s.title for s in templates.load_template("T").sections] == ["Two"]


def test_an_unusable_name_saves_nothing(qt_app):
    assert qt_app.save_as_template("..", []) is None
    assert templates.list_templates() == []


def test_new_from_template_dialog_lists_picks_and_deletes(qt_app, monkeypatch):
    templates.save_template("Beta", {}, [], [("A", "")])
    templates.save_template("Alpha", {}, [], [("A", ""), ("B", "")])
    dialog = NewFromTemplateDialog(qt_app)
    assert [dialog.list.item(i).text() for i in range(dialog.list.count())] == [
        "Alpha  (2 subprojects)", "Beta  (1 subproject)"]
    assert dialog.selected().name == "Alpha"
    assert dialog.delete_selected() is True
    assert [t.name for t in templates.list_templates()] == ["Beta"]
    assert dialog.selected().name == "Beta"
    dialog.delete_selected()
    assert dialog.selected() is None
    from PySide6.QtWidgets import QDialogButtonBox
    assert not dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()


def test_the_menu_action_builds_from_the_picked_template(qt_app, monkeypatch):
    templates.save_template("Weekly", {"gap_s": 0.55}, [], [("Intro", "Hello."), ("Main", "")])
    monkeypatch.setattr(NewFromTemplateDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
    qt_app.new_from_template_action.trigger()
    assert [c.title() for c in qt_app.children.values()] == ["Intro", "Main"]
    assert qt_app.document.settings["gap_s"] == 0.55


def test_cancelling_the_template_dialog_changes_nothing(qt_app, monkeypatch):
    templates.save_template("Weekly", {}, [], [("Intro", "")])
    root = qt_app.root.project_id
    monkeypatch.setattr(NewFromTemplateDialog, "exec", lambda self: QDialog.DialogCode.Rejected)
    assert qt_app.new_from_template_dialog() == []
    assert qt_app.root.project_id == root


def test_file_menu_has_the_template_and_credit_entries(qt_app):
    texts = [a.text().replace("&", "") for a in qt_app.file_menu.actions()]
    assert "New from Template..." in texts and "Save as Template..." in texts and "Add Credits..." in texts


def test_welcome_dialog_has_new_from_template(qt_app, monkeypatch):
    calls = []
    monkeypatch.setattr(type(qt_app), "new_from_template_dialog", lambda self: calls.append(1))
    welcome = qt_app.show_welcome()
    welcome.new_from_template_btn.click()
    assert calls == [1]


# -- credits -----------------------------------------------------------------------

FIELDS = {"title": "Moby Dick", "author": "Herman Melville", "narrator": "A. Reader", "year": "2026",
          "publisher": "Acme"}


def _book(qt_app, text="Call me Ishmael."):
    qt_app.document.text = text
    qt_app.editor.load_text(text)


def _credits(qt_app):
    document = qt_app.document
    return {document.clip_text(c): qt_app.children[c.child["id"]] for c in document.nested_clips()}


def test_add_credits_puts_two_subprojects_at_the_ends(qt_app):
    _book(qt_app)
    made = qt_app.add_credits(FIELDS)
    assert len(made) == 2
    document = qt_app.document
    assert document.text == "Opening Credits\n\nCall me Ishmael.\n\nClosing Credits"
    found = _credits(qt_app)
    opening, closing = templates.credit_texts(FIELDS)
    assert found["Opening Credits"].document.text == opening
    assert found["Closing Credits"].document.text == closing
    assert qt_app.editor.toPlainText() == document.text
    # The first clip of the book is still the book's own text, not a heading.
    assert qt_app.project_settings["credits"]["narrator"] == "A. Reader"


def test_credits_leave_the_books_generated_clips_clean(qt_app):
    from tests.gui_qt.test_subprojects import _generated_clip

    _book(qt_app, "First line. Last line.")
    first = _generated_clip(qt_app, 0, 11)
    last = _generated_clip(qt_app, 12, 22)
    assert qt_app.document.dirty_ids() == set()
    qt_app.add_credits(FIELDS)
    document = qt_app.document
    assert document.get_clip(first.id) is not None and document.get_clip(last.id) is not None
    assert document.clip_text(last) == "Last line."
    assert document.dirty_ids() - {c.id for c in document.nested_clips()} == set()


def test_running_add_credits_again_updates_instead_of_duplicating(qt_app):
    _book(qt_app)
    qt_app.add_credits(FIELDS)
    text_before = qt_app.document.text
    qt_app.add_credits({**FIELDS, "narrator": "B. Voice"})
    assert qt_app.document.text == text_before
    assert len(qt_app.document.nested_clips()) == 2
    found = _credits(qt_app)
    assert "Narrated by B. Voice." in found["Opening Credits"].document.text
    assert "Narrated by B. Voice." in found["Closing Credits"].document.text
    assert qt_app.project_settings["credits"]["narrator"] == "B. Voice"
    assert len(qt_app.children) == 2


def test_an_unchanged_form_changes_nothing(qt_app):
    _book(qt_app)
    qt_app.add_credits(FIELDS)
    assert qt_app.add_credits(FIELDS) == []


def test_credits_work_around_existing_subprojects(qt_app):
    qt_app.new_subproject(0, title="Chapter 1")
    qt_app.document.replace_text(len(qt_app.document.text), 0, 2, qt_app.document.text + "\n\n")
    qt_app.new_subproject(len(qt_app.document.text), title="Chapter 2")
    qt_app.add_credits(FIELDS)
    document = qt_app.document
    assert [document.clip_text(c) for c in document.nested_clips()].count("Opening Credits") == 1
    assert document.text == "Opening Credits\n\nChapter 1\n\nChapter 2\n\nClosing Credits"


def test_without_a_title_author_or_narrator_only_the_closing_is_added(qt_app):
    _book(qt_app)
    made = qt_app.add_credits({"year": "2026"})
    assert [c.title() for c in made] == ["Closing Credits"]
    assert [qt_app.document.clip_text(c) for c in qt_app.document.nested_clips()] == ["Closing Credits"]
    assert _credits(qt_app)["Closing Credits"].document.text == "Copyright 2026. The end."


def test_the_credits_dialog_previews_and_returns_its_fields(qt_app):
    dialog = CreditsDialog(qt_app, FIELDS)
    assert "Moby Dick. Written by Herman Melville." in dialog.opening_label.text()
    dialog.edits["title"].setText("")
    dialog.edits["author"].setText("")
    dialog.edits["narrator"].setText("")
    assert "none" in dialog.opening_label.text()
    assert dialog.values()["publisher"] == "Acme"


def test_add_credits_dialog_remembers_the_form(qt_app, monkeypatch):
    _book(qt_app)
    seen = []

    def _exec(self):
        seen.append(self.values())
        self.edits["author"].setText("Herman Melville")
        self.edits["title"].setText("Moby Dick")
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(CreditsDialog, "exec", _exec)
    qt_app.add_credits_action.trigger()
    assert seen[0]["year"].isdigit()  # the form starts with this year
    assert qt_app.project_settings["credits"]["author"] == "Herman Melville"
    seen.clear()
    monkeypatch.setattr(CreditsDialog, "exec", lambda self: seen.append(self.values()) or QDialog.DialogCode.Rejected)
    qt_app.add_credits_action.trigger()
    assert seen[0]["author"] == "Herman Melville"


def test_add_credits_needs_text_or_a_subproject(qt_app):
    qt_app.document.text = ""
    qt_app._sync_add_credits_action()
    assert not qt_app.add_credits_action.isEnabled()
    _book(qt_app)
    qt_app._sync_add_credits_action()
    assert qt_app.add_credits_action.isEnabled()


def test_the_remembered_form_is_in_project_json_not_the_document(qt_app):
    _book(qt_app)
    qt_app.add_credits(FIELDS)
    from kokoro_gui.qt import project as project_io
    qt_app._autosave_one(qt_app.root)
    loaded = project_io.load_project_dir(qt_app.root.project_dir)
    assert loaded.project_settings["credits"]["title"] == "Moby Dick"
    assert "credits" not in loaded.document.settings
