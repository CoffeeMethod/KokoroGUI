"""Tests confirming kokoro_gui.daw.models.Document is actually wired into a
running QtTTSApp (Workstream 2's document_state.load_or_create_document call
in app.py's __init__, and the save_settings()/document.json round trip)."""
import kokoro_gui.qt.app as qt_app_module
from kokoro_gui.daw.serialization import load_document


def test_fresh_app_has_a_populated_document(qt_app):
    assert qt_app.document is not None
    # A brand-new tmp_path has no presets/document.json, so migration seeds
    # exactly one "Default" character (see migration.py).
    assert len(qt_app.document.characters) == 1


def test_save_settings_writes_document_json_matching_current_state(qt_app):
    qt_app.document.text = "hello world"
    qt_app.save_settings()

    loaded = load_document(qt_app_module.DOCUMENT_FILE)
    assert loaded is not None
    assert loaded.text == "hello world"


def test_document_persists_across_app_construction(qt_app):
    clip = qt_app.document.assign_character_to_range(0, 5, qt_app.document.characters[0].id)
    qt_app.document.text = "hello world"
    qt_app.save_settings()

    # Constructing a second QtTTSApp against the same (monkeypatched) paths
    # should load the just-saved document back, not re-migrate.
    second_app = qt_app_module.QtTTSApp()
    try:
        assert second_app.document.text == "hello world"
        assert second_app.document.get_clip(clip.id) is not None
    finally:
        second_app.close()
