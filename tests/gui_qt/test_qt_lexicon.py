"""Lexicon CRUD roundtrips into settings["lexicon"] with eager save (bypasses
the debounced autosave)."""
import json


def test_add_rule_updates_settings_and_saves_eagerly(qt_app):
    import kokoro_gui.qt.app as qt_app_module
    qt_app.lexicon_dock.orig_edit.setText("API")
    qt_app.lexicon_dock.replace_edit.setText("A P I")
    qt_app.lexicon_dock.add_rule()

    assert qt_app.settings["lexicon"] == {"API": "A P I"}
    # Eager save - config_qt.json exists immediately, no need to advance a timer.
    with open(qt_app_module.CONFIG_FILE, encoding="utf-8") as f:
        data = json.load(f)
    assert data["lexicon"] == {"API": "A P I"}


def test_add_rule_rejects_empty_original(qt_app):
    qt_app.lexicon_dock.orig_edit.setText("")
    qt_app.lexicon_dock.replace_edit.setText("something")
    qt_app.lexicon_dock.add_rule()
    assert qt_app.settings.get("lexicon", {}) == {}


def test_delete_rule_removes_entry(qt_app):
    qt_app.settings["lexicon"] = {"foo": "bar", "baz": "qux"}
    qt_app.lexicon_dock.refresh_list()
    qt_app.lexicon_dock.delete_rule("foo")
    assert qt_app.settings["lexicon"] == {"baz": "qux"}


def test_add_rule_clears_input_fields(qt_app):
    qt_app.lexicon_dock.orig_edit.setText("x")
    qt_app.lexicon_dock.replace_edit.setText("y")
    qt_app.lexicon_dock.add_rule()
    assert qt_app.lexicon_dock.orig_edit.text() == ""
    assert qt_app.lexicon_dock.replace_edit.text() == ""


def test_lexicon_feeds_into_assembled_config(qt_app):
    qt_app.settings["lexicon"] = {"TTS": "Tee Tee Ess"}
    config = qt_app._assemble_config()
    assert config["lexicon"] == {"TTS": "Tee Tee Ess"}


def test_rule_labels_show_user_markup_as_literal_text(qt_app):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel

    qt_app.settings["lexicon"] = {"<b>x</b>": "<img src=http://example.invalid/a.png>"}
    qt_app.lexicon_dock.refresh_list()

    row = qt_app.lexicon_dock._list_layout.itemAt(0).widget()
    labels = {label.text(): label for label in row.findChildren(QLabel)}
    for text in ("<b>x</b>", "<img src=http://example.invalid/a.png>"):
        assert labels[text].textFormat() == Qt.TextFormat.PlainText
