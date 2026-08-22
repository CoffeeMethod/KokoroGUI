"""config_qt.json load/save roundtrip, debounce, and dock-state persistence."""
import json
import os

from kokoro_gui.qt import settings as qt_settings


def test_save_settings_writes_config_qt_json(qt_app):
    import kokoro_gui.qt.app as qt_app_module
    qt_app.generation_dock.filename_edit.setText("my_output")
    qt_app.save_settings()

    assert os.path.exists(qt_app_module.CONFIG_FILE)
    with open(qt_app_module.CONFIG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["filename"] == "my_output"


def test_save_settings_persists_fx_state(qt_app):
    import kokoro_gui.qt.app as qt_app_module
    qt_app.fx_dock._value_widgets["gain_db"].setValue(6.5)
    qt_app.save_settings()

    with open(qt_app_module.CONFIG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["gain_db"] == 6.5


def test_save_settings_stores_dock_state_and_geometry(qt_app):
    qt_app.save_settings()
    assert qt_app.settings["dock_state"]
    assert qt_app.settings["geometry"]


def test_schedule_save_debounces(qt_app, qtbot):
    calls = []
    qt_app.save_settings = lambda: calls.append(1)
    qt_app._save_timer.timeout.disconnect()
    qt_app._save_timer.timeout.connect(qt_app.save_settings)

    qt_app.schedule_save()
    qt_app.schedule_save()  # restarting the timer shouldn't double-fire
    qtbot.wait(1300)
    assert calls == [1]


def test_load_settings_defaults_when_no_file(tmp_path):
    cfg = str(tmp_path / "does_not_exist.json")
    settings = qt_settings.load_settings(cfg)
    from kokoro_gui.qt import spec
    assert settings == spec.SETTINGS_DEFAULTS


def test_load_settings_merges_over_defaults(tmp_path):
    cfg = tmp_path / "config_qt.json"
    cfg.write_text(json.dumps({"voice": "af_bella"}), encoding="utf-8")
    settings = qt_settings.load_settings(str(cfg))
    assert settings["voice"] == "af_bella"
    assert settings["speed"] == 1.0  # untouched default survives the merge


def test_save_then_load_roundtrip(tmp_path):
    cfg = str(tmp_path / "config_qt.json")
    data = {"voice": "af_bella", "speed": 1.3}
    qt_settings.save_settings(cfg, data)
    loaded = qt_settings.load_settings(cfg)
    assert loaded["voice"] == "af_bella"
    assert loaded["speed"] == 1.3
