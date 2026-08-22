"""Voice Reference dock: shown/hidden per-engine (supports_voice_cloning),
save/delete of wav+transcript references, and the auto-transcribe button.

Switching to the "audio8" engine builds a *real* `Audio8Engine` (the
registry factory has no stub-swapping hook the way `qt_app`'s fixture
patches `KokoroEngine` -> `StubEngine`), so its model load
(`kokoro_gui.engines.audio8_tts._get_model`) is monkeypatched to a fast fake
before every switch - never touches `transformers`/downloads a model.
"""
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
    monkeypatch.setattr(audio8_tts, "_get_model", lambda: (object(), object()))
    qt_app.switch_engine("audio8")


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
    qt_app.switch_engine("kokoro")
    assert qt_app.voice_clone_dock is None
    assert qt_app.mixing_dock is not None


def test_jit_streaming_disabled_falls_back_to_standard_start(qt_app, monkeypatch):
    _switch_to_audio8(qt_app, monkeypatch)
    qt_app.jit_enabled = True
    qt_app._update_start_btn_text()
    assert qt_app.start_btn.text() == "Start Generation"


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

    combo = qt_app.generation_dock.schema_form.widget_for("voice")
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
    combo = qt_app.generation_dock.schema_form.widget_for("voice")
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
    monkeypatch.setattr(_TRANSCRIBE_TARGET, lambda path: "Fake transcript text.")

    dock = qt_app.voice_clone_dock
    dock.wav_path_edit.setText(_write_wav(tmp_path / "ref.wav"))

    dock._on_transcribe_clicked()

    qtbot.waitUntil(lambda: dock.transcript_edit.toPlainText() != "", timeout=5000)
    assert dock.transcript_edit.toPlainText() == "Fake transcript text."
    assert dock.transcribe_btn.isEnabled()


def test_auto_transcribe_failure_shows_status_and_reenables_button(qt_app, monkeypatch, tmp_path, qtbot):
    _switch_to_audio8(qt_app, monkeypatch)

    def _boom(path):
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
