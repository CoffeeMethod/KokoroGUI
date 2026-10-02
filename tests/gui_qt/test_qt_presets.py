"""FX preset save/load (presets/fx/*.json). The legacy generation-preset
row left the transcript panel with the UI shell redesign (Characters
replaced it), so only FX presets have GUI save/load now."""
import json
import os

from PySide6.QtWidgets import QInputDialog


def _stub_get_text(monkeypatch, value):
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: (value, True)))


def test_save_fx_preset_writes_every_fx_key(qt_app, monkeypatch):
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
    assert qt_app.settings_dock.fx_preset_combo.currentText() == "LoudFX"
