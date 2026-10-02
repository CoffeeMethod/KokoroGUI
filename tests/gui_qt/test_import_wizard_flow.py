"""The import wizard wired into File > Import Text and the welcome dialog's
New from text (kokoro_gui/qt/import_dialog.py, `QtTTSApp.import_book`)."""
from PySide6.QtWidgets import QDialog

from kokoro_gui.engine import text_extraction
from kokoro_gui.qt.import_dialog import TARGET_ADD, TARGET_SECTIONS, TARGET_TRANSCRIPT, ImportDialog

THREE = [
    ("Copyright", "Copyright 2021. All rights reserved."),
    ("Arrival", "She came home.\n3\nThe road was long.\n\n* * *\n\nMorning."),
    ("The Storm", "Rain fell on the roof all night."),
]


def _book(tmp_path, monkeypatch, sections=THREE, name="novel.epub"):
    path = tmp_path / name
    path.write_bytes(b"fake")
    monkeypatch.setattr(text_extraction, "extract_sections", lambda p, **k: list(sections))
    return str(path)


def _accept(monkeypatch):
    """The wizard answers OK on whatever it starts with; returns its dialogs."""
    seen = []

    def _exec(self):
        seen.append(self)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(ImportDialog, "exec", _exec)
    return seen


def _type_old(app, text="old"):
    app.editor.setPlainText(text)


def test_new_from_an_epub_skips_the_copyright_section(qt_app, tmp_path, monkeypatch):
    path = _book(tmp_path, monkeypatch)
    seen = _accept(monkeypatch)
    qt_app.show_welcome().new_from_text(path)
    qt_app.wait_for_text_import()

    assert [d.is_section_checked(r) for d in seen for r in range(3)] == [False, True, True]
    children = list(qt_app.children.values())
    assert [c.title() for c in children] == ["Arrival", "The Storm"]
    # The cleanup came along: the page number is gone, the scene break is a pause.
    assert children[0].document.text == "She came home.\nThe road was long.\n\n[pause:1.5]\n\nMorning."


def test_welcome_wizard_offers_only_the_new_project_targets(qt_app, tmp_path, monkeypatch):
    path = _book(tmp_path, monkeypatch)
    seen = _accept(monkeypatch)
    qt_app.show_welcome().new_from_text(path)
    qt_app.wait_for_text_import()
    assert set(seen[0].target_radios) == {TARGET_SECTIONS, TARGET_TRANSCRIPT}
    assert seen[0].target() == TARGET_SECTIONS


def test_welcome_one_transcript_target_makes_plain_text(qt_app, tmp_path, monkeypatch):
    path = _book(tmp_path, monkeypatch)
    monkeypatch.setattr(qt_app, "_ask_import_choices", lambda *a: (TARGET_TRANSCRIPT, [("A", "one"), ("B", "two")]))
    qt_app.new_from_ebook(path)
    qt_app.wait_for_text_import()
    assert qt_app.document.text == "one\n\ntwo"
    assert qt_app.children == {}


def test_file_menu_import_adds_the_joined_text_at_the_caret(qt_app, tmp_path, monkeypatch):
    path = _book(tmp_path, monkeypatch)
    _type_old(qt_app, "old ")
    cursor = qt_app.editor.textCursor()
    cursor.movePosition(cursor.MoveOperation.End)
    qt_app.editor.setTextCursor(cursor)
    seen = []

    def _ask(p, sections, targets, default_target):
        seen.append((targets, default_target))
        return TARGET_ADD, sections[1:]

    monkeypatch.setattr(qt_app, "_ask_import_choices", _ask)
    qt_app.import_text(path)
    qt_app.wait_for_text_import()
    assert qt_app.document.text == "old " + THREE[1][1] + "\n\n" + THREE[2][1]
    assert set(seen[0][0]) == {TARGET_SECTIONS, TARGET_TRANSCRIPT, TARGET_ADD}


def test_file_menu_import_of_a_txt_is_one_section_and_still_asks(qt_app, tmp_path, monkeypatch):
    src = tmp_path / "notes.txt"
    src.write_text("Plain words.\n7\nMore words.", encoding="utf-8")
    seen = _accept(monkeypatch)
    qt_app.import_text(str(src))
    qt_app.wait_for_text_import()
    assert len(seen) == 1 and seen[0].section_list.count() == 1
    assert not seen[0].target_radios[TARGET_SECTIONS].isEnabled()
    assert seen[0].target() == TARGET_TRANSCRIPT
    # The page-number rule ran: the "7" line is gone from the new project's text.
    assert qt_app.document.text == "Plain words.\nMore words."


def test_cancelling_the_wizard_changes_nothing(qt_app, tmp_path, monkeypatch):
    path = _book(tmp_path, monkeypatch)
    _type_old(qt_app)
    monkeypatch.setattr(ImportDialog, "exec", lambda self: QDialog.DialogCode.Rejected)
    qt_app.import_text(path)
    qt_app.wait_for_text_import()
    qt_app.new_from_ebook(path)
    qt_app.wait_for_text_import()
    assert qt_app.document.text == "old"
    assert qt_app.children == {}


def test_an_empty_book_says_so_and_skips_the_wizard(qt_app, tmp_path, monkeypatch):
    path = _book(tmp_path, monkeypatch, sections=[])
    seen = _accept(monkeypatch)
    qt_app.import_text(path)
    qt_app.wait_for_text_import()
    assert seen == []


def test_import_text_with_a_target_still_inserts_the_whole_text_without_the_wizard(qt_app, tmp_path, monkeypatch):
    src = tmp_path / "plain.txt"
    src.write_text("Line one.\n12\nLine two.", encoding="utf-8")
    monkeypatch.setattr(ImportDialog, "exec", lambda self: (_ for _ in ()).throw(AssertionError("wizard opened")))
    _type_old(qt_app)
    qt_app.engine.extract_text_from_file.return_value = "Line one.
12
Line two."  # the fixture's stub reader
    qt_app.import_text(str(src), target="new")
    qt_app.wait_for_text_import()
    assert qt_app.document.text == "Line one.\n12\nLine two."


def test_the_wizard_saves_the_ticked_rules_for_next_time(qt_app, tmp_path, monkeypatch):
    path = _book(tmp_path, monkeypatch)

    def _exec(self):
        self.rule_checks["page_numbers"].setChecked(False)
        self.accept()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(ImportDialog, "exec", _exec)
    qt_app.new_from_ebook(path)
    qt_app.wait_for_text_import()
    assert qt_app.settings["import_rules"]["page_numbers"] is False
    assert list(qt_app.children.values())[0].document.text.startswith("She came home.\n3\n")
