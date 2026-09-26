"""Voice Reference dock: shown/hidden per-engine (supports_voice_cloning),
save/delete of wav+transcript references, and the auto-transcribe button.

Switching to the "audio8" engine builds a *real* `Audio8Engine` (the
registry factory has no stub-swapping hook the way `qt_app`'s fixture
patches `KokoroEngine` -> `StubEngine`), so its model load
(`kokoro_gui.engines.audio8_tts._get_model`) is monkeypatched to a fast fake
before every switch - never touches `transformers`/downloads a model.
"""
import os

import numpy as np
import soundfile as sf

from kokoro_gui.engines import audio8_tts
from kokoro_gui.engines.audio8_tts import Audio8ReferenceStore

# `kokoro_gui.qt.docks.voice_clone_dock` is deliberately never imported at
# this module's top level - `kokoro_gui.qt.app`/`kokoro_gui.qt.docks` have a
# documented circular-import relationship (see app.py's module docstring)
# that only resolves when `kokoro_gui.qt.app` is imported first, which the
# `qt_app` fixture guarantees but a bare top-level import here would not.
# `monkeypatch.setattr("module.path.attr", ...)` (string target) below
# imports the module lazily, at test-run time, after `qt_app` has already
# done so.
_TRANSCRIBE_TARGET = "kokoro_gui.qt.docks.voice_clone_dock.transcribe_wav"


def _switch_to_audio8(qt_app, monkeypatch):
    """The first character generates with Audio8 (engine follows the
    character), which makes Audio8 the active backend."""
    monkeypatch.setattr(audio8_tts, "_get_model", lambda: (object(), object()))
    assert qt_app.set_character_engine(qt_app.document.characters[0], "audio8")


def _write_wav(path):
    audio = (0.1 * np.sin(2 * np.pi * 220 * np.arange(1600) / 16000)).astype(np.float32)
    sf.write(str(path), audio, 16000)
    return str(path)


# --- show/hide on engine switch ---------------------------------------------

def test_audio8_backend_shows_voice_clone_dock_and_hides_mixing(qt_app, monkeypatch):
    _switch_to_audio8(qt_app, monkeypatch)
    assert qt_app.backend.id == "audio8"
    assert qt_app.voice_clone_dock is not None
    assert qt_app.mixing_dock is None


def test_switch_back_to_kokoro_hides_voice_clone_dock_and_restores_mixing(qt_app, monkeypatch):
    _switch_to_audio8(qt_app, monkeypatch)
    assert qt_app.set_character_engine(qt_app.document.characters[0], "kokoro")
    assert qt_app.voice_clone_dock is None
    assert qt_app.mixing_dock is not None


def test_jit_streaming_disabled_falls_back_to_standard_start(qt_app, monkeypatch):
    _switch_to_audio8(qt_app, monkeypatch)
    qt_app.jit_enabled = True
    # The Options menu's JIT toggle greys out for a backend without
    # streaming, and start_conversion() takes the Standard path.
    assert qt_app.jit_action.isEnabled() is False
    from unittest.mock import MagicMock
    monkeypatch.setattr(qt_app.engine, "start_conversion", MagicMock())
    monkeypatch.setattr(qt_app.engine, "start_jit_conversion", MagicMock())
    monkeypatch.setattr(qt_app.engine, "pipeline", True)
    qt_app.editor.setPlainText("hello")
    qt_app.start_conversion()
    assert qt_app.engine.start_conversion.called
    assert not qt_app.engine.start_jit_conversion.called


# --- save / delete reference -------------------------------------------------

def test_save_reference_appears_in_generation_voice_dropdown(qt_app, monkeypatch, tmp_path):
    _switch_to_audio8(qt_app, monkeypatch)
    wav_path = _write_wav(tmp_path / "ref.wav")

    dock = qt_app.voice_clone_dock
    dock.wav_path_edit.setText(wav_path)
    dock.transcript_edit.setPlainText("Hello world reference.")
    dock.name_edit.setText("Fred")
    dock._on_save_clicked()

    assert Audio8ReferenceStore.list_references() == ["Fred"]

    combo = qt_app.settings_dock.schema_form.widget_for("voice")
    items = [combo.itemData(i) for i in range(combo.count())]
    assert "Fred" in items


def test_save_reference_rejects_missing_name(qt_app, monkeypatch, tmp_path):
    _switch_to_audio8(qt_app, monkeypatch)
    dock = qt_app.voice_clone_dock
    dock.wav_path_edit.setText(_write_wav(tmp_path / "ref.wav"))
    dock.transcript_edit.setPlainText("Some transcript.")
    dock.name_edit.setText("")

    dock._on_save_clicked()

    assert Audio8ReferenceStore.list_references() == []


def test_delete_reference_removes_it_and_dropdown_entry(qt_app, monkeypatch, tmp_path):
    _switch_to_audio8(qt_app, monkeypatch)
    dock = qt_app.voice_clone_dock
    dock.wav_path_edit.setText(_write_wav(tmp_path / "ref.wav"))
    dock.transcript_edit.setPlainText("Some transcript.")
    dock.name_edit.setText("Ghost")
    dock._on_save_clicked()
    assert Audio8ReferenceStore.list_references() == ["Ghost"]

    # qt_app fixture patches QMessageBox.question -> Yes globally.
    dock.delete_reference("Ghost")

    assert Audio8ReferenceStore.list_references() == []
    combo = qt_app.settings_dock.schema_form.widget_for("voice")
    items = [combo.itemData(i) for i in range(combo.count())]
    assert "Ghost" not in items


def test_load_reference_populates_editable_fields(qt_app, monkeypatch, tmp_path):
    _switch_to_audio8(qt_app, monkeypatch)
    wav_path = _write_wav(tmp_path / "ref.wav")
    Audio8ReferenceStore.save_reference("Loaded", wav_path, "Original text.")

    dock = qt_app.voice_clone_dock
    dock.refresh_list()
    dock._load_reference("Loaded")

    assert dock.name_edit.text() == "Loaded"
    assert dock.transcript_edit.toPlainText() == "Original text."
    assert dock.wav_path_edit.text().endswith("Loaded.wav")


# --- auto-transcribe ----------------------------------------------------

def test_auto_transcribe_populates_transcript(qt_app, monkeypatch, tmp_path, qtbot):
    _switch_to_audio8(qt_app, monkeypatch)
    monkeypatch.setattr(_TRANSCRIBE_TARGET, lambda path, **kwargs: "Fake transcript text.")

    dock = qt_app.voice_clone_dock
    dock.wav_path_edit.setText(_write_wav(tmp_path / "ref.wav"))

    dock._on_transcribe_clicked()

    qtbot.waitUntil(lambda: dock.transcript_edit.toPlainText() != "", timeout=5000)
    assert dock.transcript_edit.toPlainText() == "Fake transcript text."
    assert dock.transcribe_btn.isEnabled()


def test_auto_transcribe_failure_shows_status_and_reenables_button(qt_app, monkeypatch, tmp_path, qtbot):
    _switch_to_audio8(qt_app, monkeypatch)

    def _boom(path, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(_TRANSCRIBE_TARGET, _boom)

    dock = qt_app.voice_clone_dock
    dock.wav_path_edit.setText(_write_wav(tmp_path / "ref.wav"))

    dock._on_transcribe_clicked()

    qtbot.waitUntil(lambda: dock.transcribe_btn.isEnabled(), timeout=5000)
    assert "failed" in dock.status_label.text().lower()
    assert dock.transcript_edit.toPlainText() == ""


def test_auto_transcribe_without_wav_selected_is_a_noop(qt_app, monkeypatch):
    _switch_to_audio8(qt_app, monkeypatch)
    dock = qt_app.voice_clone_dock
    dock.wav_path_edit.setText("")

    dock._on_transcribe_clicked()

    assert dock.transcript_edit.toPlainText() == ""


# --- ASR engine picker ----------------------------------------------------

def test_default_engine_is_whisper_and_vosk_row_hidden(qt_app, monkeypatch):
    _switch_to_audio8(qt_app, monkeypatch)
    dock = qt_app.voice_clone_dock

    assert dock.asr_engine_combo.currentData() == "whisper"
    # isHidden() (not isVisible()) - the dock's never .show()n in this
    # offscreen test, so isVisible() would be False for everything
    # regardless of our explicit setVisible() calls.
    assert dock.vosk_row.isHidden()


def test_selecting_vosk_shows_model_row(qt_app, monkeypatch):
    _switch_to_audio8(qt_app, monkeypatch)
    dock = qt_app.voice_clone_dock

    idx = dock.asr_engine_combo.findData("vosk")
    dock.asr_engine_combo.setCurrentIndex(idx)

    assert not dock.vosk_row.isHidden()


def test_transcribe_with_vosk_passes_engine_and_typed_model_path(qt_app, monkeypatch, tmp_path, qtbot):
    monkeypatch.delenv("VOSK_MODEL_PATH", raising=False)
    _switch_to_audio8(qt_app, monkeypatch)
    dock = qt_app.voice_clone_dock
    dock.wav_path_edit.setText(_write_wav(tmp_path / "ref.wav"))
    dock.asr_engine_combo.setCurrentIndex(dock.asr_engine_combo.findData("vosk"))
    # Typed but not Saved - transcribing shouldn't require a save first.
    dock.vosk_model_edit.setText(str(tmp_path / "some-vosk-model"))

    seen = {}

    def _fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return "Vosk transcript."

    monkeypatch.setattr(_TRANSCRIBE_TARGET, _fake_transcribe)
    dock._on_transcribe_clicked()

    qtbot.waitUntil(lambda: dock.transcript_edit.toPlainText() != "", timeout=5000)
    assert dock.transcript_edit.toPlainText() == "Vosk transcript."
    assert seen == {"engine": "vosk", "model_path": str(tmp_path / "some-vosk-model")}


def test_transcribe_with_vosk_and_empty_model_field_is_a_noop(qt_app, monkeypatch, tmp_path):
    monkeypatch.delenv("VOSK_MODEL_PATH", raising=False)
    _switch_to_audio8(qt_app, monkeypatch)
    dock = qt_app.voice_clone_dock
    dock.wav_path_edit.setText(_write_wav(tmp_path / "ref.wav"))
    dock.asr_engine_combo.setCurrentIndex(dock.asr_engine_combo.findData("vosk"))
    dock.vosk_model_edit.setText("")

    dock._on_transcribe_clicked()

    assert dock.transcript_edit.toPlainText() == ""


def test_vosk_model_field_reflects_env_var_on_open(qt_app, monkeypatch, tmp_path):
    monkeypatch.setenv("VOSK_MODEL_PATH", str(tmp_path / "my-model"))
    _switch_to_audio8(qt_app, monkeypatch)
    dock = qt_app.voice_clone_dock

    assert dock.vosk_model_edit.text() == str(tmp_path / "my-model")


def test_vosk_model_field_empty_when_env_var_unset(qt_app, monkeypatch):
    monkeypatch.delenv("VOSK_MODEL_PATH", raising=False)
    _switch_to_audio8(qt_app, monkeypatch)
    dock = qt_app.voice_clone_dock

    assert dock.vosk_model_edit.text() == ""


def test_save_writes_typed_path_to_dotenv(qt_app, monkeypatch, tmp_path):
    monkeypatch.delenv("VOSK_MODEL_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    _switch_to_audio8(qt_app, monkeypatch)
    dock = qt_app.voice_clone_dock
    dock.vosk_model_edit.setText(str(tmp_path / "typed-model"))

    dock._save_vosk_model_path()

    assert os.environ["VOSK_MODEL_PATH"] == str(tmp_path / "typed-model")
    assert "typed-model" in (tmp_path / ".env").read_text()
    assert "Saved" in dock.status_label.text()


def test_browse_sets_field_and_saves(qt_app, monkeypatch, tmp_path):
    monkeypatch.delenv("VOSK_MODEL_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    _switch_to_audio8(qt_app, monkeypatch)
    dock = qt_app.voice_clone_dock

    browsed = tmp_path / "browsed-model"
    monkeypatch.setattr(
        "kokoro_gui.qt.docks.voice_clone_dock.QFileDialog.getExistingDirectory",
        lambda *a, **k: str(browsed),
    )
    dock._browse_vosk_model()

    assert dock.vosk_model_edit.text() == str(browsed)
    assert os.environ["VOSK_MODEL_PATH"] == str(browsed)
    assert "browsed-model" in (tmp_path / ".env").read_text()


def test_reload_discards_unsaved_edit_and_rereads_env(qt_app, monkeypatch, tmp_path):
    # No .env file in this cwd - reload_vosk_model_path() finds nothing to
    # re-read and leaves os.environ (set below) alone, so this isolates the
    # test from whatever real .env file the repo root might actually have.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VOSK_MODEL_PATH", str(tmp_path / "saved-model"))
    _switch_to_audio8(qt_app, monkeypatch)
    dock = qt_app.voice_clone_dock
    assert dock.vosk_model_edit.text() == str(tmp_path / "saved-model")

    dock.vosk_model_edit.setText(str(tmp_path / "unsaved-typed-path"))

    dock._reload_vosk_model_path()

    assert dock.vosk_model_edit.text() == str(tmp_path / "saved-model")


def test_asr_engine_choice_persists_via_get_state(qt_app, monkeypatch):
    _switch_to_audio8(qt_app, monkeypatch)
    dock = qt_app.voice_clone_dock
    dock.asr_engine_combo.setCurrentIndex(dock.asr_engine_combo.findData("vosk"))

    assert dock.get_state() == {"asr_engine": "vosk"}

    qt_app.save_settings()
    assert qt_app.settings["asr_engine"] == "vosk"


# --- Whisper first-use download prompt (grill PR5) --------------------------

def _whisper_not_cached(monkeypatch, answer):
    from kokoro_gui.engine import asr
    from kokoro_gui.qt import asr_prompt

    asked = []
    monkeypatch.setattr(asr, "whisper_model_cached", lambda name=None: False)
    monkeypatch.setattr(asr_prompt, "ask_whisper_download",
                        lambda parent, name, size: (asked.append((name, size)), answer)[1])
    return asked


def test_whisper_download_prompt_names_the_size_and_says_downloading(qt_app, monkeypatch, tmp_path, qtbot):
    from kokoro_gui.qt import asr_prompt

    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    _switch_to_audio8(qt_app, monkeypatch)
    asked = _whisper_not_cached(monkeypatch, asr_prompt.PROCEED)
    statuses = []
    monkeypatch.setattr(_TRANSCRIBE_TARGET,
                        lambda path, **kwargs: (statuses.append(dock.status_label.text()), "Words.")[1])
    dock = qt_app.voice_clone_dock
    dock.wav_path_edit.setText(_write_wav(tmp_path / "ref.wav"))

    dock._on_transcribe_clicked()

    qtbot.waitUntil(lambda: dock.transcript_edit.toPlainText() == "Words.", timeout=5000)
    assert asked == [("large-v3-turbo", "1.6 GB")]
    assert statuses == ["Downloading Whisper model..."]


def test_whisper_download_prompt_use_another_engine_switches_without_transcribing(qt_app, monkeypatch, tmp_path):
    from kokoro_gui.qt import asr_prompt

    _switch_to_audio8(qt_app, monkeypatch)
    _whisper_not_cached(monkeypatch, asr_prompt.OTHER_ENGINE)
    called = []
    monkeypatch.setattr(_TRANSCRIBE_TARGET, lambda path, **kwargs: called.append(path) or "x")
    dock = qt_app.voice_clone_dock
    dock.wav_path_edit.setText(_write_wav(tmp_path / "ref.wav"))

    dock._on_transcribe_clicked()

    assert dock.asr_engine_combo.currentData() != "whisper"
    assert called == []
    assert dock.transcribe_btn.isEnabled()


def test_whisper_download_prompt_cancel_does_nothing(qt_app, monkeypatch, tmp_path):
    from kokoro_gui.qt import asr_prompt

    _switch_to_audio8(qt_app, monkeypatch)
    _whisper_not_cached(monkeypatch, asr_prompt.CANCEL)
    called = []
    monkeypatch.setattr(_TRANSCRIBE_TARGET, lambda path, **kwargs: called.append(path) or "x")
    dock = qt_app.voice_clone_dock
    dock.wav_path_edit.setText(_write_wav(tmp_path / "ref.wav"))

    dock._on_transcribe_clicked()

    assert dock.asr_engine_combo.currentData() == "whisper"
    assert called == []


# --- the Voices tab's own engine (grill EN3) ----------------------------------

def test_the_audio8_editor_opens_with_only_kokoro_characters(qt_app, monkeypatch, tmp_path):
    monkeypatch.setattr(audio8_tts, "_get_model", lambda: (object(), object()))
    character = qt_app.document.characters[0]
    assert qt_app.voice_clone_dock is None and qt_app.mixing_dock is not None

    qt_app.set_voices_engine("audio8")

    dock = qt_app.voice_clone_dock
    assert dock is not None and qt_app.mixing_dock is None
    assert dock.engine_combo.currentData() == "audio8"
    assert character.backend_id == "kokoro" and qt_app.backend.id == "kokoro"

    dock.wav_path_edit.setText(_write_wav(tmp_path / "ref.wav"))
    dock.transcript_edit.setPlainText("Hello world reference.")
    dock.name_edit.setText("Fred")
    dock._on_save_clicked()
    assert Audio8ReferenceStore.list_references() == ["Fred"]

    assert qt_app.set_character_engine(character, "audio8")
    assert character.preset_data["voice"] == "Fred"  # the one reference it lists
    combo = qt_app.settings_dock.schema_form.widget_for("voice")
    assert "Fred" in [combo.itemData(i) for i in range(combo.count())]


def test_the_next_selection_puts_the_voices_tab_back_on_the_active_engine(qt_app):
    character = qt_app.document.characters[0]
    qt_app.set_voices_engine("audio8")
    assert qt_app.voice_clone_dock is not None

    qt_app.selection.select_character(character.id)

    assert qt_app.voices_engine_id == "kokoro"
    assert qt_app.mixing_dock is not None and qt_app.voice_clone_dock is None


def test_picking_in_the_voices_engine_combo_is_deferred(qt_app, qtbot):
    combo = qt_app.mixing_dock.engine_combo
    assert [combo.itemData(i) for i in range(combo.count())] == qt_app.voice_editor_engines() == ["audio8", "kokoro"]

    combo.setCurrentIndex(combo.findData("audio8"))
    combo.activated.emit(combo.currentIndex())
    assert qt_app.voices_engine_id == "kokoro"  # not from inside the signal

    qtbot.waitUntil(lambda: qt_app.voice_clone_dock is not None)
    assert qt_app.voices_engine_id == "audio8"


def test_voice_tools_run_on_the_voices_engines_worker(qt_app, monkeypatch, tmp_path):
    qt_app.set_voices_engine("audio8")
    dock = qt_app.voice_clone_dock
    audio8_worker = qt_app.voices_backend().engine.worker
    calls = []
    monkeypatch.setattr(audio8_worker, "run_coro", lambda coro: (calls.append(coro), coro.close())[0] or _Done())
    dock.wav_path_edit.setText(_write_wav(tmp_path / "ref.wav"))
    dock.asr_engine_combo.setCurrentIndex(dock.asr_engine_combo.findData("vosk"))
    dock.vosk_model_edit.setText(str(tmp_path))

    dock._on_transcribe_clicked()

    assert len(calls) == 1


class _Done:
    def add_done_callback(self, _cb):
        pass
