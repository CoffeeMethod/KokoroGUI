"""config_qt.json load/save roundtrip, debounce, and dock-state persistence."""
import json
import os

from kokoro_gui.qt import settings as qt_settings


def test_save_settings_writes_config_qt_json(qt_app):
    import kokoro_gui.qt.app as qt_app_module
    qt_app.settings_dock.volume_spin.setValue(1.7)
    qt_app.save_settings()

    assert os.path.exists(qt_app_module.CONFIG_FILE)
    with open(qt_app_module.CONFIG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["volume"] == 1.7


def test_export_settings_persist_in_the_project_file(qt_app, tmp_path):
    import os

    from kokoro_gui.qt import project as project_io

    qt_app.project_settings["export"] = {"filename": "my_output", "format": "flac"}
    qt_app.save_settings()

    with open(os.path.join(qt_app.project_dir, "project.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["export"]["filename"] == "my_output"
    assert qt_app._assemble_config()["filename"] == "my_output"
    assert qt_app._assemble_config()["format"] == "flac"

    qt_app.save_project_as(str(tmp_path / "exp"))
    qt_app.wait_for_project_io()
    assert project_io.load_project(qt_app.project_path).project_settings["export"]["filename"] == "my_output"


def test_save_settings_persists_fx_state(qt_app):
    import kokoro_gui.qt.app as qt_app_module
    qt_app.fx_dock._value_widgets["gain_db"].setValue(6.5)
    qt_app.save_settings()

    with open(qt_app_module.CONFIG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["gain_db"] == 6.5


def test_save_settings_stores_active_workspace_layout(qt_app):
    qt_app.save_settings()
    entry = qt_app.settings["workspaces"][qt_app.settings["active_workspace"]]
    assert entry["state"]
    assert entry["geometry"]


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


# --- per-engine settings (grill EN5) -----------------------------------------

def test_flat_language_and_threads_move_into_each_engines_bucket():
    settings = {"lang_code": "b", "num_threads": 3, "voice": "bm_daniel", "engines": {}}
    qt_settings.migrate_engine_settings(settings, engine_ids=["kokoro", "audio8", "dummy"])

    assert not {"lang_code", "num_threads", "voice"} & set(settings)
    # "b" is a Kokoro (and Dummy) language, not an Audio8 one; the voice was
    # the default engine's.
    assert settings["engines"]["kokoro"] == {"lang_code": "b", "num_threads": 3, "voice": "bm_daniel"}
    assert settings["engines"]["dummy"] == {"lang_code": "b", "num_threads": 3}
    assert settings["engines"]["audio8"] == {"num_threads": 3}


def test_migration_keeps_a_value_an_engine_already_has():
    settings = {"lang_code": "a", "engines": {"kokoro": {"lang_code": "j"}}}
    qt_settings.migrate_engine_settings(settings, engine_ids=["kokoro"])
    assert settings["engines"]["kokoro"] == {"lang_code": "j"}


def test_an_old_config_opens_with_its_language_on_kokoro(tmp_path, monkeypatch, qtbot):
    import kokoro_gui.qt.app as qt_app_module

    config = tmp_path / "config_qt.json"
    config.write_text(json.dumps({"lang_code": "b", "num_threads": 2}), encoding="utf-8")
    monkeypatch.setattr(qt_app_module, "CONFIG_FILE", str(config))
    settings = qt_settings.load_settings(str(config))
    qt_settings.migrate_engine_settings(settings)

    assert settings["engines"]["kokoro"]["lang_code"] == "b"
    assert "lang_code" not in settings["engines"].get("audio8", {})


def test_a_per_engine_edit_lands_in_that_engines_bucket(qt_app):
    import kokoro_gui.qt.app as qt_app_module

    qt_app.selection.clear()
    qt_app.settings_dock.schema_form.widget_for("num_threads").setValue(4)
    qt_app.save_settings()

    with open(qt_app_module.CONFIG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["engines"]["kokoro"]["num_threads"] == 4
    assert "num_threads" not in data and "lang_code" not in data
    assert qt_app.engine_settings("kokoro")["num_threads"] == 4
    assert qt_app.engine_settings("audio8")["num_threads"] == 1


def test_engine_settings_default_from_the_schema(qt_app):
    assert qt_app.engine_settings("kokoro") == {"lang_code": "a", "voice": "af_heart", "num_threads": 1}
    audio8 = qt_app.engine_settings("audio8")
    assert audio8["lang_code"] == "English" and audio8["temperature"] == 0.8
    assert audio8["voice"] is None and "caching" not in audio8 and "speed" not in audio8
