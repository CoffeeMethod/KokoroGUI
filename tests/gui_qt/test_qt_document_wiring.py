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
    """The 4.0-preview implicit `document.json` becomes `document.tbaw` on the
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


# -- the character library (phase 3 step 4, and the phase's "done when") -----------------


def test_switching_in_a_document_resolves_linked_characters(qt_app):
    from kokoro_gui.daw.models import Character, Document

    library = qt_app.character_library
    library_id = library.save(Character.from_preset_dict("Narrator", {"voice": "am_adam"}, highlight_color="#e0655c"))
    stale = Character.from_preset_dict("My narrator", {"voice": "af_bella"}, library_id=library_id)
    doc = Document.from_plain_text("", characters=[stale])

    qt_app._switch_document(doc, None)

    assert stale.preset_data == {"voice": "am_adam"}
    assert stale.highlight_color == "#e0655c"
    assert stale.name == "My narrator"
    assert qt_app.library_missing == set()


def test_a_library_change_on_disk_re_resolves_the_open_document(qt_app, qtbot):
    from kokoro_gui.daw.models import Character

    library = qt_app.character_library
    character = qt_app.document.characters[0]
    character.library_id = library.save(character)
    qt_app.resolve_library()

    changed = []
    real = qt_app.on_characters_changed
    qt_app.on_characters_changed = lambda: (changed.append(True), real())

    # Another window edits the entry.
    entry = library.get(character.library_id)
    entry.preset_data["voice"] = "bm_george"
    library.save(entry)

    qtbot.waitUntil(lambda: character.preset_data.get("voice") == "bm_george", timeout=5000)
    assert changed
    assert isinstance(entry, Character)


def test_two_projects_share_a_library_character(qt_app, tmp_path):
    """The phase's "done when": a voice change in one project's Characters
    dialog reaches the other on its next open; a machine without the entry
    plays the inlined voice and shows "not found here"."""
    from kokoro_gui.qt.characters_dialog import SCOPE_MISSING, CharactersDialog

    dialog = CharactersDialog(qt_app)
    library_id = dialog.promote_current()
    qt_app.save_project_as(str(tmp_path / "one"))
    qt_app.wait_for_project_io()
    one = qt_app.project_path

    qt_app.new_project()
    assert qt_app.document.characters[0].library_id == library_id
    qt_app.save_project_as(str(tmp_path / "two"))
    qt_app.wait_for_project_io()
    two = qt_app.project_path

    dialog = CharactersDialog(qt_app)
    dialog.voice_combo.setCurrentText("bf_emma")
    qt_app.save_project()
    qt_app.wait_for_project_io()

    qt_app.open_project(one)
    qt_app.wait_for_project_io()
    assert qt_app.project_path == one
    assert qt_app.document.characters[0].preset_data["voice"] == "bf_emma"

    # A machine where the library doesn't have the entry: project two was
    # saved after the change, so its snapshot has the new voice.
    assert qt_app.character_library.delete(library_id)
    qt_app.open_project(two)
    qt_app.wait_for_project_io()
    assert qt_app.project_path == two
    character = qt_app.document.characters[0]
    assert character.preset_data["voice"] == "bf_emma"  # the snapshot plays
    assert character.id in qt_app.library_missing
    dialog = CharactersDialog(qt_app)
    assert dialog.scope_label.text() == SCOPE_MISSING


def test_first_launch_imports_presets_and_links_exact_matches(qt_app, tmp_path):
    """Step 7: presets/*.json become library entries once, and the project
    opened at launch links its characters that match one exactly."""
    import json

    from kokoro_gui.daw.models import Character
    import kokoro_gui.qt.settings as qt_settings

    match = Character.from_preset_dict("Alice", {"voice": "af_bella"})
    differs = Character.from_preset_dict("Bob", {"voice": "am_adam"})
    qt_app.document.characters = [match, differs]
    qt_app.save_project_as(str(tmp_path / "old"))
    qt_app.wait_for_project_io()
    path = qt_app.project_path
    qt_app.close()

    presets = tmp_path / "presets"
    (presets / "Alice.json").write_text(json.dumps({"voice": "af_bella"}), encoding="utf-8")
    (presets / "Bob.json").write_text(json.dumps({"voice": "am_michael"}), encoding="utf-8")
    settings = qt_settings.load_settings(qt_app_module.CONFIG_FILE)
    settings.pop("library_imported", None)
    settings["last_project"] = path
    qt_settings.save_settings(qt_app_module.CONFIG_FILE, settings)

    second_app = qt_app_module.QtTTSApp()
    try:
        second_app.wait_for_project_io()
        assert second_app.project_path == path
        assert {e.name for e in second_app.character_library.list()} == {"Alice", "Bob"}
        assert (presets / "Alice.json").exists()
        by_name = {c.name: c for c in second_app.document.characters}
        assert by_name["Alice"].library_id is not None
        assert by_name["Bob"].library_id is None  # the preset's voice differs
        assert second_app.settings["library_imported"] is True
        second_app.save_settings()
    finally:
        second_app.close()

    # Once only: a later launch doesn't import again.
    (presets / "Carol.json").write_text(json.dumps({"voice": "af_heart"}), encoding="utf-8")
    third_app = qt_app_module.QtTTSApp()
    try:
        third_app.wait_for_project_io()
        assert {e.name for e in third_app.character_library.list()} == {"Alice", "Bob"}
    finally:
        third_app.close()
