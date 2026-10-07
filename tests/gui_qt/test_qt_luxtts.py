"""LuxTTS editing, settings and project portability without downloads."""
import concurrent.futures
import zipfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from kokoro_gui.engines import luxtts


@pytest.fixture
def lux_app(qt_app, monkeypatch):
    fake = SimpleNamespace(encode_prompt=MagicMock(return_value={}),
                           generate_speech=MagicMock(return_value=np.ones(4800, dtype=np.float32)))
    monkeypatch.setattr(luxtts, "_load_lux", lambda device: fake)
    qt_app.set_voices_engine("luxtts")
    return qt_app


def _save(app, tmp_path):
    wav = tmp_path / "reference.wav"
    sf.write(str(wav), np.full(3 * 16000, 0.1), 16000)
    dock = app.voice_clone_dock
    dock.wav_path_edit.setText(str(wav))
    dock.name_edit.setText("Lux Voice")
    dock._on_save_clicked()
    return wav


def test_save_without_transcript_and_select_for_character(lux_app, tmp_path):
    app = lux_app
    _save(app, tmp_path)
    dock = app.voice_clone_dock
    assert not dock.transcript_edit.isHidden() and not dock.transcribe_btn.isHidden()
    assert luxtts.LuxTTSReferenceStore.list_references() == ["Lux Voice"]
    character = app.document.characters[0]
    assert app.set_character_engine(character, "luxtts")
    assert character.preset_data["voice"] == "Lux Voice"
    assert not app.jit_action.isEnabled()


def test_model_settings_reach_document_and_clip_generation(lux_app, tmp_path):
    app = lux_app
    _save(app, tmp_path)
    character = app.document.characters[0]
    app.set_character_engine(character, "luxtts")
    app.editor.setPlainText("Hello there.")
    app.flush_updates()
    app.document.assign_character_to_range(0, len(app.document.text), character.id)
    app.settings["engines"]["luxtts"].update({"num_steps": 8, "ref_rms": 0.02, "return_smooth": True})
    clip = app.document.clips[0]
    for config in (app._assemble_config(), app._assemble_generation_config(clip)):
        assert config["num_steps"] == 8 and config["ref_rms"] == 0.02
        assert config["return_smooth"] is True and config["engine_id"] == "luxtts"


def test_preview_uses_model_settings_and_project_reference(lux_app, monkeypatch, tmp_path):
    app = lux_app
    _save(app, tmp_path)
    app.set_character_engine(app.document.characters[0], "luxtts")
    app.settings["engines"]["luxtts"]["num_steps"] = 9
    future = concurrent.futures.Future()
    future.set_result(False)
    preview = MagicMock(return_value=future)
    monkeypatch.setattr(app.backend, "preview", preview)
    app.backend.engine.pipeline = True
    app.preview_conversion()
    config = preview.call_args.args[4]
    assert config["num_steps"] == 9
    assert config["project_dir"] == app.project_dir


def test_project_round_trip_preserves_audio_only_voice(lux_app, tmp_path):
    app = lux_app
    _save(app, tmp_path)
    app.set_character_engine(app.document.characters[0], "luxtts")
    app.settings["engines"]["luxtts"]["num_steps"] = 8
    path = str(tmp_path / "lux.tbaw")
    app.save_project_as(path)
    app.wait_for_project_io()
    with zipfile.ZipFile(path) as bundle:
        assert "engines/luxtts/refs/Lux Voice.wav" in bundle.namelist()
    luxtts.LuxTTSReferenceStore.delete_reference("Lux Voice")
    # Extract on a fresh machine; opening an already-open path only focuses it.
    from kokoro_gui.qt import project as project_io
    info = project_io.inspect_bundle(path)
    directory = str(tmp_path / "fresh-project")
    project_io.extract_small(info, directory)
    project_io.extract_audio(info, directory)
    loaded = project_io.finish_open(info, directory)
    assert loaded.document.characters[0].backend_id == "luxtts"
    assert luxtts.LuxTTSReferenceStore.find_wav("Lux Voice", directory)


def test_project_bundles_and_reopens_reviewed_transcript(lux_app, tmp_path):
    app = lux_app
    _save(app, tmp_path)
    luxtts.LuxTTSReferenceStore.save_reference("Reviewed", str(tmp_path / "reference.wav"), "Reviewed words.")
    app.voice_clone_dock._use_reference("Reviewed")
    path = str(tmp_path / "reviewed.tbaw")
    app.save_project_as(path)
    app.wait_for_project_io()
    with zipfile.ZipFile(path) as bundle:
        assert bundle.read("engines/luxtts/refs/Reviewed.txt").decode() == "Reviewed words."
    luxtts.LuxTTSReferenceStore.delete_reference("Reviewed")
    from kokoro_gui.qt import project as project_io
    info = project_io.inspect_bundle(path)
    directory = str(tmp_path / "fresh-reviewed")
    project_io.extract_small(info, directory)
    assert luxtts.LuxTTSReferenceStore.get_transcript("Reviewed", directory) == "Reviewed words."


def test_engine_settings_persist_in_app_config(lux_app, tmp_path):
    import json

    app = lux_app
    app.set_engine_setting("luxtts", "num_steps", 8)
    app.set_engine_setting("luxtts", "return_smooth", True)
    app.save_settings()
    saved = json.loads((tmp_path / "config_qt.json").read_text(encoding="utf-8"))
    assert saved["engines"]["luxtts"]["num_steps"] == 8
    assert saved["engines"]["luxtts"]["return_smooth"] is True


def test_save_and_use_reference_assigns_character_without_settings_picker(lux_app, tmp_path):
    app = lux_app
    character = app.document.characters[0]
    app.set_character_engine(character, "luxtts")
    _save(app, tmp_path)
    dock = app.voice_clone_dock
    app.selection.select_character(character.id)
    assert character.preset_data["voice"] == "Lux Voice"
    assert app.settings_dock.schema_form.widget_for("voice") is None
    assert "Lux Voice" in app.settings_dock.reference_voice_label.text()
    luxtts.LuxTTSReferenceStore.save_reference("Other", str(tmp_path / "reference.wav"), "Edited words.")
    dock._use_reference("Other")
    assert character.preset_data["voice"] == "Other"
    assert app.engine_settings("luxtts")["voice"] == "Other"
    dock._load_reference("Other")
    assert dock.transcript_edit.toPlainText() == "Edited words."


def test_auto_transcribe_uses_model_excerpt_and_allows_edits(lux_app, tmp_path, monkeypatch, qtbot):
    from kokoro_gui.qt.docks import voice_clone_dock

    app = lux_app
    wav = tmp_path / "long.wav"
    sf.write(str(wav), np.ones(10 * 16000) * 0.1, 16000)
    seen = []

    def transcribe(path, **kwargs):
        seen.append((path, sf.info(path).duration))
        return "Heard words."

    monkeypatch.setattr(voice_clone_dock, "transcribe_wav", transcribe)
    dock = app.voice_clone_dock
    dock.wav_path_edit.setText(str(wav))
    dock._on_transcribe_clicked()
    qtbot.waitUntil(lambda: dock.transcript_edit.toPlainText() == "Heard words.", timeout=5000)
    assert seen[0][1] == 5.0
    from pathlib import Path
    assert not Path(seen[0][0]).exists()
    dock.transcript_edit.setPlainText("Corrected words.")
    dock.name_edit.setText("Reviewed")
    dock._on_save_clicked()
    assert luxtts.LuxTTSReferenceStore.get_transcript("Reviewed") == "Corrected words."
