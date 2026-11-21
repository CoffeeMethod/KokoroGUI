"""Save to library (plan 41) in the GUI: the right-click menus of the Audio FX
list, the Mixing and Voice Reference lists, and the Characters dialog's button.
The planning and copying are tested in tests/daw/test_library_copy.py. The
modal dialog is answered by patching `exec`."""
import json
import os

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtWidgets import QDialog, QLabel, QMessageBox, QPushButton

from kokoro_gui.daw import share
from kokoro_gui.engines import audio8_tts
from kokoro_gui.engines.audio8_tts import Audio8ReferenceStore


@pytest.fixture
def stores(qt_app, tmp_path, monkeypatch):
    from kokoro_gui.engine import runtime
    from kokoro_gui.qt import app as qt_app_module

    monkeypatch.setattr(runtime, "CUSTOM_VOICES_DIR", str(tmp_path / "custom_voices"))
    monkeypatch.setattr(audio8_tts, "AUDIO8_REFS_DIR", str(tmp_path / "audio8_refs"))
    return {"fx": qt_app_module.FX_PRESETS_DIR, "mixes": str(tmp_path / "custom_voices"),
            "refs": str(tmp_path / "audio8_refs")}


@pytest.fixture
def messages(monkeypatch):
    seen = []
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda p, t, text, *a, **k: seen.append(text)))
    return seen


def put(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data if isinstance(data, bytes) else data.encode("utf-8"))


def project_file(qt_app, *parts, data=b"x"):
    path = os.path.join(qt_app.project_dir, *parts)
    put(path, data)
    return path


def row_texts(layout):
    out = []
    for i in range(layout.count()):
        widget = layout.itemAt(i).widget()
        out.append([label.text() for label in widget.findChildren(QLabel)] if widget is not None else [])
    return out


def trigger_save(menu):
    action = menu.actions()[0]
    assert action.isEnabled(), action.text()
    action.trigger()


# -- the Audio FX list ------------------------------------------------------------------


def test_the_fx_menu_saves_a_project_fx_and_its_impulse_response(qt_app, stores, messages):
    project_file(qt_app, "fx", "Radio.json", data=json.dumps({"gain_db": 3, "convolution_ir": "Hall"}))
    project_file(qt_app, "fx", "ir", "Hall.wav", data=b"ir")
    dock = qt_app.fx_dock
    dock.refresh_presets()
    index = dock.preset_combo.findData("Radio")
    assert dock.preset_combo.itemData(index, 3) == "In this project"  # the tooltip marker

    trigger_save(dock.preset_menu("Radio"))

    assert os.path.isfile(os.path.join(stores["fx"], "Radio.json"))
    assert os.path.isfile(os.path.join(stores["fx"], "ir", "Hall.wav"))
    assert "Saved 2 items" in messages[-1]
    # Saved: the menu now says the library has it.
    assert not dock.preset_menu("Radio").actions()[0].isEnabled()


def test_the_fx_menu_is_off_for_a_library_fx_and_absent_for_no_fx(qt_app, stores):
    put(os.path.join(stores["fx"], "Warm.json"), json.dumps({"gain_db": 1}))
    dock = qt_app.fx_dock
    dock.refresh_presets()
    assert not dock.preset_menu("Warm").actions()[0].isEnabled()
    assert dock.preset_menu(dock.preset_combo.itemData(0)) is None
    assert dock.preset_menu(None) is None


def test_a_clashing_fx_asks_and_keep_both_numbers_it(qt_app, stores, monkeypatch, messages):
    from kokoro_gui.qt.share_dialogs import SaveToLibraryDialog

    project_file(qt_app, "fx", "Radio.json", data=json.dumps({"gain_db": 3}))
    put(os.path.join(stores["fx"], "Radio.json"), json.dumps({"gain_db": -1}))
    seen = {}

    def answer(self):
        seen["choices"] = self.choices()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(SaveToLibraryDialog, "exec", answer)
    qt_app.fx_dock.save_fx_to_library("Radio")

    assert seen["choices"] == {"fx:Radio": share.KEEP}
    assert os.path.isfile(os.path.join(stores["fx"], "Radio (2).json"))
    with open(os.path.join(stores["fx"], "Radio.json"), encoding="utf-8") as f:
        assert json.load(f)["gain_db"] == -1


def test_cancelling_the_clash_dialog_saves_nothing(qt_app, stores, monkeypatch):
    from kokoro_gui.qt.share_dialogs import SaveToLibraryDialog

    project_file(qt_app, "fx", "Radio.json", data=json.dumps({"gain_db": 3}))
    put(os.path.join(stores["fx"], "Radio.json"), json.dumps({"gain_db": -1}))
    monkeypatch.setattr(SaveToLibraryDialog, "exec", lambda self: QDialog.DialogCode.Rejected)
    assert qt_app.fx_dock.save_fx_to_library("Radio") is None
    assert not os.path.exists(os.path.join(stores["fx"], "Radio (2).json"))


# -- the Mixing list --------------------------------------------------------------------


def test_the_mixing_list_shows_a_project_mix_and_saves_it(qt_app, stores, messages):
    put(os.path.join(stores["mixes"], "own.pt"), b"own")
    project_file(qt_app, "engines", "kokoro", "voices", "carried.pt", data=b"carried")
    dock = qt_app.mixing_dock
    dock.refresh_voice_lists()

    rows = row_texts(dock._list_layout)
    assert ["own"] in rows and ["carried", "in this project"] in rows
    assert dock.mix_menu("own").actions()[0].isEnabled() is False

    trigger_save(dock.mix_menu("carried"))

    with open(os.path.join(stores["mixes"], "carried.pt"), "rb") as f:
        assert f.read() == b"carried"


def test_a_project_only_mix_row_has_no_delete_button(qt_app, stores):
    project_file(qt_app, "engines", "kokoro", "voices", "carried.pt")
    dock = qt_app.mixing_dock
    dock.refresh_voice_lists()
    buttons = [b for i in range(dock._list_layout.count())
               for b in dock._list_layout.itemAt(i).widget().findChildren(QPushButton)]
    assert buttons == []


# -- the Voice Reference list -----------------------------------------------------------


def _audio8_dock(qt_app, monkeypatch):
    monkeypatch.setattr(audio8_tts, "_get_model", lambda: (object(), object()))
    assert qt_app.set_character_engine(qt_app.document.characters[0], "audio8")
    return qt_app.voice_clone_dock


def _wav(path):
    sf.write(str(path), (0.1 * np.sin(np.arange(1600) / 5)).astype(np.float32), 16000)
    return str(path)


def _project_reference(qt_app, tmp_path, name, text):
    wav = _wav(tmp_path / f"{name}-src.wav")
    with open(wav, "rb") as f:
        project_file(qt_app, "engines", "audio8", "refs", f"{name}.wav", data=f.read())
    project_file(qt_app, "engines", "audio8", "refs", f"{name}.txt", data=text)
    return wav


def test_the_reference_menu_saves_a_project_reference(qt_app, stores, monkeypatch, messages, tmp_path):
    dock = _audio8_dock(qt_app, monkeypatch)
    _project_reference(qt_app, tmp_path, "ann", "the words")
    dock.refresh_list()

    assert any("in this project" in row for row in row_texts(dock._list_layout))
    trigger_save(dock.reference_menu("ann"))

    assert Audio8ReferenceStore.list_references() == ["ann"]
    assert os.path.isfile(os.path.join(stores["refs"], "ann.wav"))
    with open(os.path.join(stores["refs"], "ann.txt"), encoding="utf-8") as f:
        assert f.read() == "the words"


def test_saving_a_reference_the_project_also_carries_says_the_project_copy_is_used(qt_app, stores, monkeypatch,
                                                                                    tmp_path):
    dock = _audio8_dock(qt_app, monkeypatch)
    wav = _project_reference(qt_app, tmp_path, "ann", "project words")

    dock.wav_path_edit.setText(wav)
    dock.transcript_edit.setPlainText("edited words")
    dock.name_edit.setText("ann")
    dock._on_save_clicked()

    assert "own copy" in dock.status_label.text()
    # The save goes to the library; the project dir is not written.
    with open(os.path.join(qt_app.project_dir, "engines", "audio8", "refs", "ann.txt"), encoding="utf-8") as f:
        assert f.read() == "project words"


def test_a_plain_reference_save_has_no_shadow_note(qt_app, stores, monkeypatch, tmp_path):
    dock = _audio8_dock(qt_app, monkeypatch)
    dock.wav_path_edit.setText(_wav(tmp_path / "ref.wav"))
    dock.transcript_edit.setPlainText("words")
    dock.name_edit.setText("fresh")
    dock._on_save_clicked()
    assert "own copy" not in dock.status_label.text()


# -- the Characters dialog --------------------------------------------------------------


def test_the_characters_dialog_saves_everything_a_character_uses(qt_app, stores, messages):
    from kokoro_gui.qt.characters_dialog import CharactersDialog

    project_file(qt_app, "engines", "kokoro", "voices", "deep.pt", data=b"deep")
    project_file(qt_app, "engines", "kokoro", "voices", "soft.pt", data=b"soft")
    project_file(qt_app, "fx", "Radio.json", data=json.dumps({"convolution_ir": "Hall"}))
    project_file(qt_app, "fx", "ir", "Hall.wav", data=b"ir")
    dialog = CharactersDialog(qt_app)
    character = dialog._current
    character.preset_data.update({"voice": "deep", "fx_preset": "Radio"})
    character.variants = {"soft": "soft"}

    result = dialog.save_assets_to_library()

    assert sorted(result.saved) == ["FX Radio", "Impulse response Hall", "Mix deep (kokoro)", "Mix soft (kokoro)"]
    assert os.path.isfile(os.path.join(stores["mixes"], "soft.pt"))
    assert os.path.isfile(os.path.join(stores["fx"], "ir", "Hall.wav"))
    assert "Saved 4 items" in messages[-1]


def test_the_characters_dialog_says_when_nothing_is_project_only(qt_app, stores, messages):
    from kokoro_gui.qt.characters_dialog import CharactersDialog

    dialog = CharactersDialog(qt_app)
    assert dialog.save_assets_to_library() is None
    assert "already stored outside this project" in messages[-1]
