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
    assert dock.transcript_edit.isHidden() and dock.transcribe_btn.isHidden()
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


def test_engine_settings_persist_in_app_config(lux_app, tmp_path):
    import json

    app = lux_app
    app.set_engine_setting("luxtts", "num_steps", 8)
    app.set_engine_setting("luxtts", "return_smooth", True)
    app.save_settings()
    saved = json.loads((tmp_path / "config_qt.json").read_text(encoding="utf-8"))
    assert saved["engines"]["luxtts"]["num_steps"] == 8
    assert saved["engines"]["luxtts"]["return_smooth"] is True
