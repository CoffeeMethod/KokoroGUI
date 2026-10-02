"""ENGINE_AGNOSTIC plan, C6 "done when": a test-only engine registered by an
entry point (tests/plugins/toneclone.py) shows in Edit > Characters and the
Settings row, gets the Voice Reference editor, generates through its own
model, and saves and reopens a bundle - with no code for it under `qt/`,
`daw/` or `audio/`."""
import zipfile

import numpy as np
import soundfile as sf


def _ref_wav(tmp_path):
    path = tmp_path / "ref.wav"
    sf.write(str(path), (0.1 * np.sin(np.arange(1600) / 5)).astype(np.float32), 16000)
    return str(path)


def test_a_plugin_engine_works_end_to_end(qt_app, qtbot, toneclone_plugin, tmp_path):
    from kokoro_gui.qt.characters_dialog import CharactersDialog

    # It's listed wherever an engine is picked.
    assert ("ToneClone (test)", "toneclone") in qt_app.engine_choices()
    dialog = CharactersDialog(qt_app)
    assert dialog.engine_combo.findData("toneclone") >= 0
    qt_app.settings_dock._build_for_selection()
    assert "toneclone" in qt_app.voice_editor_engines()

    # Its voice editor, from the Voices tab, saves into its own store.
    qt_app.set_voices_engine("toneclone")
    dock = qt_app.voice_clone_dock
    assert dock is not None and dock.backend_id == "toneclone"
    dock.wav_path_edit.setText(_ref_wav(tmp_path))
    dock.transcript_edit.setPlainText("A reference line.")
    dock.name_edit.setText("Echo")
    dock._on_save_clicked()
    assert qt_app.voices_backend().voice_store.list_references() == ["Echo"]

    # A character on it generates through its model.
    character = qt_app.document.characters[0]
    assert qt_app.set_character_engine(character, "toneclone")
    assert character.preset_data["voice"] == "Echo"
    qt_app.document.text = "one two three"
    clip = qt_app.document.assign_character_to_range(0, len(qt_app.document.text), character.id)
    assert clip in qt_app.document.dirty_clips()
    assert qt_app.timeline_dock.on_generate_clip_requested(clip.id)
    qtbot.waitUntil(lambda: bool(clip.segments) and not qt_app.is_busy(), timeout=10000)
    segment = clip.segments[0]
    data, rate = sf.read(segment.audio_path)
    assert rate == 16000 and len(data) == int(16000 * 0.02 * 3)
    assert clip not in qt_app.document.dirty_clips()

    # Save bundles its reference; the reopened project still uses it.
    path = str(tmp_path / "plugin.tbaw")
    qt_app.save_project_as(path)
    qt_app.wait_for_project_io()
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
    assert {"engines/toneclone/refs/Echo.wav", "engines/toneclone/refs/Echo.txt"} <= names

    qt_app.open_project(path)
    qt_app.wait_for_project_io()
    reopened = qt_app.document.characters[0]
    assert reopened.backend_id == "toneclone"
    assert qt_app.backend_for_character(reopened).id == "toneclone"
    assert qt_app.document.dirty_clips() == []
