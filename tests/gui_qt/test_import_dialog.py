"""kokoro_gui/qt/import_dialog.py: the import wizard dialog."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog

from kokoro_gui.qt.import_dialog import (
    PREVIEW_CHARS, TARGET_ADD, TARGET_SECTIONS, TARGET_TRANSCRIPT, ImportDialog)

BOOK = [
    ("Copyright", "Copyright 2021. All rights reserved. " * 3),
    ("Contents", "1 Arrival\n2 The Storm"),
    ("Arrival", "She came home.\n12\nThe road was long and exam-\nple of a wander.\n\n\n\n* * *\n\nMorning."),
    ("The Storm", "Rain fell on the roof all night."),
]


def _dialog(qt_app, qtbot, sections=BOOK, **kwargs):
    dialog = ImportDialog(qt_app, "novel.epub", sections, **kwargs)
    qtbot.addWidget(dialog)
    return dialog


def _titles(dialog):
    return [dialog.section_list.item(r).text() for r in range(dialog.section_list.count())]


def test_sections_are_listed_with_word_counts_and_guessed_defaults(qt_app, qtbot):
    dialog = _dialog(qt_app, qtbot)
    assert _titles(dialog)[3] == "The Storm  (7 words)"
    assert [dialog.is_section_checked(r) for r in range(4)] == [False, False, True, True]


def test_all_and_none_buttons(qt_app, qtbot):
    dialog = _dialog(qt_app, qtbot)
    dialog.all_btn.click()
    assert dialog.checked_rows() == [0, 1, 2, 3]
    dialog.none_btn.click()
    assert dialog.checked_rows() == []
    assert not dialog.button_box.button(dialog.button_box.StandardButton.Ok).isEnabled()
    dialog.all_btn.click()
    assert dialog.button_box.button(dialog.button_box.StandardButton.Ok).isEnabled()


def test_toggling_a_rule_changes_the_preview_after_the_debounce(qt_app, qtbot):
    dialog = _dialog(qt_app, qtbot)
    dialog.section_list.setCurrentRow(2)
    assert dialog.before_edit.toPlainText() == BOOK[2][1]
    assert "\n12\n" not in dialog.after_edit.toPlainText()
    assert "[pause:1.5]" in dialog.after_edit.toPlainText()

    dialog.rule_checks["page_numbers"].setChecked(False)
    assert dialog._preview_timer.isActive()
    assert "\n12\n" not in dialog.after_edit.toPlainText()  # not refreshed yet
    qtbot.waitUntil(lambda: "\n12\n" in dialog.after_edit.toPlainText(), timeout=2000)

    dialog.rule_checks["soft_wraps"].setChecked(True)
    dialog.refresh_preview()
    assert "The road was long and example of a wander." in dialog.after_edit.toPlainText()


def test_preview_of_a_long_section_is_cut_and_says_so(qt_app, qtbot):
    dialog = _dialog(qt_app, qtbot, sections=[("Long", "word " * PREVIEW_CHARS)])
    assert len(dialog.before_edit.toPlainText()) == PREVIEW_CHARS
    assert "first" in dialog.preview_note.text()
    # The result cleans the whole text, not the preview.
    assert len(dialog.choices()[1][0][1]) > PREVIEW_CHARS


def test_choices_return_only_checked_sections_cleaned(qt_app, qtbot):
    dialog = _dialog(qt_app, qtbot)
    target, parts = dialog.choices()
    assert target == TARGET_SECTIONS
    assert [title for title, _ in parts] == ["Arrival", "The Storm"]
    assert parts[0][1] == "She came home.\nThe road was long and example of a wander.\n\n[pause:1.5]\n\nMorning."
    dialog.section_list.item(0).setCheckState(Qt.CheckState.Checked)
    assert [t for t, _ in dialog.choices()[1]] == ["Copyright", "Arrival", "The Storm"]


def test_a_section_with_nothing_left_after_cleanup_is_dropped(qt_app, qtbot):
    dialog = _dialog(qt_app, qtbot, sections=[("Pages", "1\n2\n3"), ("Story", "Words here.")])
    dialog.all_btn.click()
    assert dialog.choices() == (TARGET_SECTIONS, [("Story", "Words here.")])


def test_target_radios_and_default(qt_app, qtbot):
    dialog = _dialog(qt_app, qtbot)
    assert dialog.target() == TARGET_SECTIONS
    dialog.target_radios[TARGET_ADD].setChecked(True)
    assert dialog.choices()[0] == TARGET_ADD
    dialog = _dialog(qt_app, qtbot, default_target=TARGET_TRANSCRIPT)
    assert dialog.target() == TARGET_TRANSCRIPT


def test_targets_can_be_limited_and_a_one_section_file_has_no_split_radio(qt_app, qtbot):
    dialog = _dialog(qt_app, qtbot, targets=(TARGET_SECTIONS, TARGET_TRANSCRIPT))
    assert set(dialog.target_radios) == {TARGET_SECTIONS, TARGET_TRANSCRIPT}
    one = _dialog(qt_app, qtbot, sections=[("notes", "Plain words.")], default_target=TARGET_SECTIONS)
    assert not one.target_radios[TARGET_SECTIONS].isEnabled()
    assert one.target() == TARGET_TRANSCRIPT
    assert one.choices() == (TARGET_TRANSCRIPT, [("notes", "Plain words.")])


def test_rules_start_from_the_saved_setting_and_are_saved_on_accept(qt_app, qtbot):
    qt_app.settings["import_rules"] = {"soft_wraps": True, "page_numbers": False}
    dialog = _dialog(qt_app, qtbot)
    assert dialog.enabled_rules() == ["hyphen_joins", "soft_wraps", "scene_breaks",
                                      "collapse_blank_lines", "strip_whitespace"]
    dialog.rule_checks["scene_breaks"].setChecked(False)
    dialog.accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert qt_app.settings["import_rules"] == {
        "page_numbers": False, "hyphen_joins": True, "soft_wraps": True, "scene_breaks": False,
        "collapse_blank_lines": True, "strip_whitespace": True}
    # A second dialog opens on what was saved.
    assert "scene_breaks" not in _dialog(qt_app, qtbot).enabled_rules()


def test_cancel_saves_nothing(qt_app, qtbot):
    dialog = _dialog(qt_app, qtbot)
    dialog.rule_checks["page_numbers"].setChecked(False)
    dialog.reject()
    assert qt_app.settings["import_rules"] == {}
