"""Lexicon rules roundtrip into settings["lexicon"] with eager save (bypasses
the debounced autosave), and the dock edits, orders and tests them."""
import json

from PySide6.QtCore import Qt


def _rule(find, replace, mode="literal", case=False):
    return {"find": find, "replace": replace, "mode": mode, "case": case}


def test_add_rule_updates_settings_and_saves_eagerly(qt_app):
    import kokoro_gui.qt.app as qt_app_module
    qt_app.lexicon_dock.orig_edit.setText("API")
    qt_app.lexicon_dock.replace_edit.setText("A P I")
    qt_app.lexicon_dock.add_rule()

    assert qt_app.settings["lexicon"] == [_rule("API", "A P I")]
    # Eager save - config_qt.json exists immediately, no need to advance a timer.
    with open(qt_app_module.CONFIG_FILE, encoding="utf-8") as f:
        data = json.load(f)
    assert data["lexicon"] == [_rule("API", "A P I")]


def test_add_rule_rejects_empty_original(qt_app):
    qt_app.lexicon_dock.orig_edit.setText("")
    qt_app.lexicon_dock.replace_edit.setText("something")
    qt_app.lexicon_dock.add_rule()
    assert qt_app.settings.get("lexicon", []) == []


def test_delete_rule_removes_the_entry_at_that_index(qt_app):
    qt_app.settings["lexicon"] = [_rule("foo", "bar"), _rule("baz", "qux")]
    qt_app.lexicon_dock.refresh_list()
    qt_app.lexicon_dock.delete_rule(0)
    assert qt_app.settings["lexicon"] == [_rule("baz", "qux")]
    assert qt_app.lexicon_dock.table.rowCount() == 1


def test_add_rule_clears_input_fields(qt_app):
    qt_app.lexicon_dock.orig_edit.setText("x")
    qt_app.lexicon_dock.replace_edit.setText("y")
    qt_app.lexicon_dock.add_rule()
    assert qt_app.lexicon_dock.orig_edit.text() == ""
    assert qt_app.lexicon_dock.replace_edit.text() == ""


def test_lexicon_feeds_into_assembled_config(qt_app):
    qt_app.settings["lexicon"] = [_rule("TTS", "Tee Tee Ess")]
    config = qt_app._assemble_config()
    assert config["lexicon"] == [_rule("TTS", "Tee Tee Ess")]


def test_the_dock_converts_a_dict_set_in_settings(qt_app):
    qt_app.settings["lexicon"] = {"foo": "bar"}
    qt_app.lexicon_dock.refresh_list()
    assert qt_app.settings["lexicon"] == [_rule("foo", "bar")]
    assert qt_app.lexicon_dock.table.rowCount() == 1


def test_rules_show_user_markup_as_literal_text(qt_app):
    from PySide6.QtWidgets import QLabel

    dock = qt_app.lexicon_dock
    qt_app.settings["lexicon"] = [_rule("<b>x</b>", "<img src=http://example.invalid/a.png>")]
    dock.refresh_list()

    # Table items are plain text, and the result label never renders markup.
    assert dock.table.item(0, 0).text() == "<b>x</b>"
    assert dock.table.item(0, 1).text() == "<img src=http://example.invalid/a.png>"
    dock.test_edit.setText("<i>y</i>")
    assert dock.after_label.textFormat() == Qt.TextFormat.PlainText
    assert dock.after_label.text() == "After your rules: <i>y</i>"
    assert dock.error_label.textFormat() == Qt.TextFormat.PlainText
    assert all(isinstance(label, QLabel) for label in dock.findChildren(QLabel))


def test_add_a_whole_word_rule_leaves_a_longer_word_alone(qt_app):
    dock = qt_app.lexicon_dock
    dock.orig_edit.setText("Al")
    dock.replace_edit.setText("Albert")
    dock.mode_combo.setCurrentIndex(dock.mode_combo.findData("word"))
    dock.add_rule()

    assert qt_app.settings["lexicon"] == [_rule("Al", "Albert", "word")]
    dock.test_edit.setText("Al said Also")
    assert dock.after_label.text() == "After your rules: Albert said Also"


def test_a_typed_backslash_is_spoken_as_typed_and_shown_as_typed(qt_app):
    dock = qt_app.lexicon_dock
    dock.orig_edit.setText("path")
    dock.replace_edit.setText("C:\\dir")
    dock.add_rule()

    assert qt_app.settings["lexicon"][0]["replace"] == "C:\\\\dir"
    assert dock.table.item(0, 1).text() == "C:\\dir"
    dock.test_edit.setText("the path")
    assert dock.after_label.text() == "After your rules: the C:\\dir"


def test_a_pattern_rule_reads_groups(qt_app):
    dock = qt_app.lexicon_dock
    dock.orig_edit.setText(r"(\d+)(st|nd|rd|th)")
    dock.replace_edit.setText(r"\1 \2")
    dock.mode_combo.setCurrentIndex(dock.mode_combo.findData("regex"))
    dock.add_rule()

    dock.test_edit.setText("the 3rd")
    assert dock.after_label.text() == "After your rules: the 3 rd"


def test_an_invalid_pattern_is_refused_with_the_error(qt_app):
    dock = qt_app.lexicon_dock
    dock.orig_edit.setText("(")
    dock.replace_edit.setText("x")
    dock.mode_combo.setCurrentIndex(dock.mode_combo.findData("regex"))
    dock.add_rule()

    assert qt_app.settings["lexicon"] == []
    assert "missing )" in dock.error_label.text()
    assert dock.orig_edit.text() == "("  # kept for the user to fix


def test_a_reply_naming_a_missing_group_is_refused(qt_app):
    dock = qt_app.lexicon_dock
    dock.orig_edit.setText("(a)")
    dock.replace_edit.setText(r"\2")
    dock.mode_combo.setCurrentIndex(dock.mode_combo.findData("regex"))
    dock.add_rule()

    assert qt_app.settings["lexicon"] == []
    assert dock.error_label.text()


def test_editing_a_cell_to_an_invalid_pattern_puts_the_old_rule_back(qt_app):
    dock = qt_app.lexicon_dock
    qt_app.settings["lexicon"] = [_rule("a+", "x", "regex")]
    dock.refresh_list()

    dock.table.item(0, 0).setText("a(")

    assert qt_app.settings["lexicon"] == [_rule("a+", "x", "regex")]
    assert dock.table.item(0, 0).text() == "a+"
    assert dock.error_label.text().startswith("Not saved")


def test_editing_a_cell_saves_the_rule(qt_app):
    dock = qt_app.lexicon_dock
    qt_app.settings["lexicon"] = [_rule("a", "x")]
    dock.refresh_list()

    dock.table.item(0, 0).setText("b")
    dock.table.item(0, 1).setText("y")
    dock.table.item(0, 3).setCheckState(Qt.CheckState.Checked)
    combo = dock.table.cellWidget(0, 2)
    combo.setCurrentIndex(combo.findData("word"))

    assert qt_app.settings["lexicon"] == [_rule("b", "y", "word", True)]
    assert dock.error_label.text() == ""


def test_an_empty_find_cell_is_refused(qt_app):
    dock = qt_app.lexicon_dock
    qt_app.settings["lexicon"] = [_rule("a", "x")]
    dock.refresh_list()
    dock.table.item(0, 0).setText("  ")
    assert qt_app.settings["lexicon"] == [_rule("a", "x")]
    assert dock.table.item(0, 0).text() == "a"


def test_reordering_two_rules_changes_the_result(qt_app):
    dock = qt_app.lexicon_dock
    qt_app.settings["lexicon"] = [_rule("a", "b"), _rule("b", "c")]
    dock.refresh_list()
    dock.test_edit.setText("a")
    assert dock.after_label.text() == "After your rules: c"

    dock.move_rule(1, -1)

    assert qt_app.settings["lexicon"] == [_rule("b", "c"), _rule("a", "b")]
    assert dock.after_label.text() == "After your rules: b"
    assert dock.table.item(0, 0).text() == "b"


def test_dragging_a_row_number_reorders_the_rules(qt_app, qapp):
    dock = qt_app.lexicon_dock
    qt_app.settings["lexicon"] = [_rule("a", "b"), _rule("b", "c"), _rule("c", "d")]
    dock.refresh_list()

    dock.table.verticalHeader().moveSection(2, 0)
    qapp.processEvents()

    assert [r["find"] for r in qt_app.settings["lexicon"]] == ["c", "a", "b"]
    assert [dock.table.item(row, 0).text() for row in range(3)] == ["c", "a", "b"]
    header = dock.table.verticalHeader()
    assert [header.logicalIndex(v) for v in range(3)] == [0, 1, 2]


def test_a_rule_change_refreshes_the_timeline(qt_app):
    """The dock refreshes the timeline, so the dirty check sees a new rule."""
    seen = []
    qt_app.refresh_timeline = lambda *a, **k: seen.append(list(qt_app.settings["lexicon"]))
    qt_app.lexicon_dock.orig_edit.setText("x")
    qt_app.lexicon_dock.add_rule()
    assert seen == [[_rule("x", "")]]


def test_the_test_field_shows_the_sentence_after_the_rules(qt_app):
    dock = qt_app.lexicon_dock
    qt_app.settings["lexicon"] = [_rule("Dr.", "Doctor", "word")]
    dock.refresh_list()
    dock.test_edit.setText("Dr. Who [pause:1] knows")
    # Generation strips the pause marker first, so the test field does too.
    assert dock.after_label.text().startswith("After your rules: Doctor Who")
    assert "[pause" not in dock.after_label.text()
    dock.test_edit.setText("")
    assert dock.after_label.text() == ""


def test_show_text_fills_the_test_field_and_raises_the_tab(qt_app):
    dock = qt_app.lexicon_dock
    dock.show_text("in 1999\nDr. Who")
    assert dock.test_edit.text() == "in 1999 Dr. Who"
    assert not dock.isHidden()
