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


def test_loading_a_preset_with_wrong_typed_values_keeps_the_good_ones(qt_app):
    import kokoro_gui.qt.app as qt_app_module

    os.makedirs(qt_app_module.FX_PRESETS_DIR, exist_ok=True)
    with open(os.path.join(qt_app_module.FX_PRESETS_DIR, "Odd.json"), "w", encoding="utf-8") as f:
        json.dump({"gain_db": "loud", "eq_bass": 5, "comp_threshold": [1], "reverb_room_size": 40,
                   "comp_enabled": "yes", "convolution_ir": {"a": 1}}, f)
    qt_app.fx_dock._value_widgets["gain_db"].setValue(2.0)

    qt_app.fx_dock.load_preset("Odd")

    widgets = qt_app.fx_dock._value_widgets
    assert widgets["gain_db"].value() == 2.0       # the bad value left the widget alone
    assert widgets["eq_bass"].value() == 5.0
    assert widgets["reverb_room_size"].value() == 1.0  # clamped to the slider's top
    assert not qt_app.fx_dock._enabled_checks["comp_enabled"].isChecked()


def test_the_fx_dock_survives_a_non_object_preset_file(qt_app):
    import kokoro_gui.qt.app as qt_app_module

    os.makedirs(qt_app_module.FX_PRESETS_DIR, exist_ok=True)
    with open(os.path.join(qt_app_module.FX_PRESETS_DIR, "List.json"), "w", encoding="utf-8") as f:
        f.write("[1, 2, 3]")
    qt_app.fx_dock.load_preset("List")


OLD_SENTINEL = "Select FX Preset..."  # what config_qt.json and old projects store


def test_the_no_fx_item_shows_no_fx_and_stores_the_old_sentinel(qt_app):
    for combo in (qt_app.settings_dock.fx_preset_combo, qt_app.fx_dock.preset_combo):
        assert combo.itemText(0) == "No FX"
        assert combo.itemData(0) == OLD_SENTINEL
        assert combo.findText(OLD_SENTINEL) < 0
    assert qt_app.settings_dock._snapshot_none_values()["fx_preset"] == OLD_SENTINEL


def test_a_character_holding_the_old_sentinel_shows_no_fx_and_resolves_to_no_fx(qt_app):
    from kokoro_gui.qt import fx_resolve

    character = qt_app.document.characters[0]
    character.preset_data["fx_preset"] = OLD_SENTINEL

    qt_app.selection.select_character(character.id)

    for combo in (qt_app.settings_dock.fx_preset_combo, qt_app.fx_dock.preset_combo):
        assert combo.currentText() == "No FX"
        assert combo.currentData() == OLD_SENTINEL
    resolution = fx_resolve.resolve_fx(qt_app, character=character)
    assert resolution.preset_name is None
    assert character.preset_data["fx_preset"] == OLD_SENTINEL  # showing it doesn't rewrite it


def test_choosing_a_named_fx_in_the_settings_combo_stores_its_name(qt_app):
    from kokoro_gui.qt import fx_resolve

    import kokoro_gui.qt.app as qt_app_module
    os.makedirs(qt_app_module.FX_PRESETS_DIR, exist_ok=True)
    with open(os.path.join(qt_app_module.FX_PRESETS_DIR, "Radio.json"), "w", encoding="utf-8") as f:
        json.dump({"gain_db": 1.0}, f)
    qt_app.fx_dock.refresh_presets()
    character = qt_app.document.characters[0]
    qt_app.selection.select_character(character.id)
    combo = qt_app.settings_dock.fx_preset_combo

    combo.setCurrentIndex(combo.findData("Radio"))

    assert character.preset_data["fx_preset"] == "Radio"
    assert fx_resolve.real_preset_name(character.preset_data["fx_preset"]) == "Radio"
