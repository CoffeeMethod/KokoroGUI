"""Tests confirming kokoro_gui.daw.models.Document is actually wired into a
running QtTTSApp (Workstream 2's document_state.load_or_create_document call
in app.py's __init__, and the autosave -> project dir -> resume round trip)."""
import os

import kokoro_gui.qt.app as qt_app_module
from kokoro_gui.daw.serialization import load_document


def test_fresh_app_has_a_populated_document(qt_app):
    assert qt_app.document is not None
    # A brand-new tmp_path has no presets/document.json, so migration seeds
    # exactly one "Default" character (see migration.py).
    assert len(qt_app.document.characters) == 1


def test_save_settings_writes_document_json_into_the_project_dir(qt_app):
    """Autosave writes the live project dir, never a file at DOCUMENT_FILE
    (that path is only read, for the one-time migration to .tbaw)."""
    qt_app.document.text = "hello world"
    qt_app.save_settings()

    loaded = load_document(os.path.join(qt_app.project_dir, "document.json"))
    assert loaded is not None
    assert loaded.text == "hello world"
    assert not os.path.exists(qt_app_module.DOCUMENT_FILE)


def test_document_persists_across_app_construction(qt_app, tmp_path):
    qt_app.document.text = "hello world"
    clip = qt_app.document.assign_character_to_range(0, 5, qt_app.document.characters[0].id)
    qt_app.save_project_as(str(tmp_path / "persist"))
    qt_app.wait_for_project_io()
    qt_app.close()  # releases the lock, keeps the last project's dir

    # A second QtTTSApp against the same (monkeypatched) config resumes the
    # last project from its kept dir (TB13), not a re-migration.
    second_app = qt_app_module.QtTTSApp()
    try:
        second_app.wait_for_project_io()
        assert second_app.project_path == qt_app.project_path
        assert second_app.document.text == "hello world"
        assert second_app.document.get_clip(clip.id) is not None
    finally:
        second_app.close()


def test_legacy_document_json_next_to_the_config_migrates_on_launch(qt_app, tmp_path):
    """The pre-4.2 implicit `document.json` becomes `document.tbaw` on the
    first launch that finds it."""
    from kokoro_gui.daw.models import Character, Document
    from kokoro_gui.qt import project as project_io

    qt_app.close()
    doc = Document.from_plain_text("legacy words", characters=[Character.from_preset_dict("L", {})])
    project_io.save_json_project(doc, qt_app_module.DOCUMENT_FILE, {})
    qt_app.settings["last_project"] = None
    import kokoro_gui.qt.settings as qt_settings

    qt_settings.save_settings(qt_app_module.CONFIG_FILE, qt_app.settings)

    second_app = qt_app_module.QtTTSApp()
    try:
        second_app.wait_for_project_io()
        assert second_app.document.text == "legacy words"
        assert second_app.project_path == project_io.bundle_path_for(os.path.abspath(qt_app_module.DOCUMENT_FILE))
        assert os.path.isfile(second_app.project_path)
    finally:
        second_app.close()
