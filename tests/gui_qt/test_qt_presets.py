"""Generation + FX preset save/load (presets/*.json, presets/fx/*.json -
shared with the Tk frontend, see spec.py's module docstring), and the
cross-frontend load that proves that sharing actually holds."""
import json
import os

from PySide6.QtWidgets import QInputDialog


def _stub_get_text(monkeypatch, value):
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: (value, True)))


def test_save_generation_preset_writes_expected_keys(qt_app, monkeypatch):
    _stub_get_text(monkeypatch, "MyPreset")
    qt_app.generation_dock.schema_form.set_values({"voice": "af_bella", "speed": 1.4})
    qt_app.generation_dock._save_preset_dialog()

    import kokoro_gui.qt.app as qt_app_module
    fpath = os.path.join(qt_app_module.PRESETS_DIR, "MyPreset.json")
    assert os.path.exists(fpath)
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)
    assert data["voice"] == "af_bella"
    assert data["speed"] == 1.4
    assert set(data.keys()) == {"voice", "speed", "volume", "pitch", "split_pattern",
                                 "normalize", "trim", "format", "apply_fx", "fx_preset"}


def test_load_generation_preset_applies_values(qt_app, monkeypatch):
    _stub_get_text(monkeypatch, "SpeedyBella")
    qt_app.generation_dock.schema_form.set_values({"voice": "af_bella", "speed": 1.6})
    qt_app.generation_dock._save_preset_dialog()

    qt_app.generation_dock.schema_form.set_values({"voice": "af_heart", "speed": 1.0})
    qt_app.generation_dock.refresh_presets()
    qt_app.generation_dock._on_preset_selected("SpeedyBella")

    state = qt_app.generation_dock.get_state()
    assert state["voice"] == "af_bella"
    assert state["speed"] == 1.6


def test_save_fx_preset_writes_all_43_keys(qt_app, monkeypatch):
    from kokoro_gui.qt import spec
    _stub_get_text(monkeypatch, "MyFX")
    qt_app.fx_dock._value_widgets["gain_db"].setValue(3.0)
    qt_app.fx_dock._save_preset_dialog()

    import kokoro_gui.qt.app as qt_app_module
    fpath = os.path.join(qt_app_module.FX_PRESETS_DIR, "MyFX.json")
    assert os.path.exists(fpath)
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)
    assert set(data.keys()) == set(spec.FX_PRESET_KEYS)
    assert data["gain_db"] == 3.0


def test_load_fx_preset_applies_values_and_syncs_gen_combo(qt_app, monkeypatch):
    _stub_get_text(monkeypatch, "LoudFX")
    qt_app.fx_dock._value_widgets["gain_db"].setValue(9.0)
    qt_app.fx_dock._save_preset_dialog()

    qt_app.fx_dock._value_widgets["gain_db"].setValue(0.0)
    qt_app.fx_dock.load_preset("LoudFX")

    assert qt_app.fx_dock._value_widgets["gain_db"].value() == 9.0
    assert qt_app.generation_dock.fx_preset_combo.currentText() == "LoudFX"


def test_preset_saved_by_tk_loads_correctly_in_qt(tts_app, qt_app, monkeypatch):
    """The shared-presets design decision (see the plan): a preset written by
    the Tk frontend must load correctly through Qt, and vice versa, since
    both point at the same presets/*.json directory for a shared tmp_path.

    tts_app must be requested before qt_app - see the comment in
    test_qt_config_assembly.py's equivalent test."""
    import gui

    class FakeDialog:
        def __init__(self, *a, **kw):
            pass

        def get_input(self):
            return "FromTk"

    monkeypatch.setattr(gui.ctk, "CTkInputDialog", FakeDialog)

    tts_app.voice_var.set("af_bella")
    tts_app.speed_var.set(1.25)
    tts_app.save_preset_dialog()

    qt_app.generation_dock.refresh_presets()
    qt_app.generation_dock._on_preset_selected("FromTk")

    state = qt_app.generation_dock.get_state()
    assert state["voice"] == "af_bella"
    assert state["speed"] == 1.25


def test_fx_preset_saved_by_qt_loads_correctly_in_tk(tts_app, qt_app, monkeypatch):
    _stub_get_text(monkeypatch, "FromQt")
    qt_app.fx_dock._value_widgets["gain_db"].setValue(4.5)
    qt_app.fx_dock._save_preset_dialog()

    tts_app.load_fx_preset("FromQt")
    assert tts_app.gain_db.get() == 4.5
