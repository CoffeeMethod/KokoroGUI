"""config_qt.json load/save roundtrip, debounce, and dock-state persistence."""
import json
import os

import pytest

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

    window = qt_app.open_settings_window("Performance")
    window.engine_forms["kokoro"].widget_for("num_threads").setValue(4)
    window.apply()
    window.reject()
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


# --- untrusted config_qt.json: wrong-typed values, atomic write -----------------

def _write_config(tmp_path, content):
    path = tmp_path / "config_qt.json"
    path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return str(path)


def test_load_settings_resets_a_wrong_typed_value_to_its_default(tmp_path, capsys):
    from kokoro_gui.qt import spec

    cfg = _write_config(tmp_path, {"speed": "fast", "lexicon": "none", "num_threads": 2.5, "caching": "false",
                                   "volume": 2, "voice_note": "unknown keys pass through"})
    settings = qt_settings.load_settings(cfg)

    assert settings["speed"] == spec.SETTINGS_DEFAULTS["speed"]
    assert settings["lexicon"] == []
    assert settings["num_threads"] == spec.SETTINGS_DEFAULTS["num_threads"]
    assert settings["caching"] == spec.SETTINGS_DEFAULTS["caching"]
    assert settings["volume"] == 2  # an int where a float is expected is fine
    assert settings["voice_note"] == "unknown keys pass through"
    assert "caching, lexicon, num_threads, speed" in capsys.readouterr().out


def test_load_settings_rejects_a_bool_for_a_number_and_a_non_finite_float(tmp_path):
    from kokoro_gui.qt import spec

    cfg = _write_config(tmp_path, '{"speed": true, "pitch": NaN, "num_threads": true, "last_project": 5}')
    settings = qt_settings.load_settings(cfg)
    assert settings["speed"] == spec.SETTINGS_DEFAULTS["speed"]
    assert settings["pitch"] == spec.SETTINGS_DEFAULTS["pitch"]
    assert settings["num_threads"] == spec.SETTINGS_DEFAULTS["num_threads"]
    assert settings["last_project"] is None


def test_load_settings_keeps_a_last_project_path(tmp_path):
    cfg = _write_config(tmp_path, {"last_project": "C:/books/novel.tbaw"})
    assert qt_settings.load_settings(cfg)["last_project"] == "C:/books/novel.tbaw"


def test_a_config_holding_a_list_loads_the_defaults(tmp_path):
    from kokoro_gui.qt import spec

    assert qt_settings.load_settings(_write_config(tmp_path, "[]")) == spec.SETTINGS_DEFAULTS


def test_a_config_that_is_not_json_loads_the_defaults(tmp_path):
    from kokoro_gui.qt import spec

    assert qt_settings.load_settings(_write_config(tmp_path, "{not json")) == spec.SETTINGS_DEFAULTS


@pytest.fixture
def wrong_typed_config(tmp_path):
    """On disk before `qt_app` builds (a fixture listed first is set up first)."""
    return _write_config(tmp_path, {"speed": "fast", "lexicon": "none", "volume": "loud", "reverb_room_size": "big",
                                    "gain_db": None, "comp_enabled": "yes", "convolution_ir": ["x"],
                                    "highpass_freq": {"a": 1}, "gsm_enabled": 3})


def test_the_app_starts_with_wrong_typed_settings(wrong_typed_config, qt_app):
    """Each of these used to raise `TypeError` in a widget constructor."""
    from kokoro_gui.qt import spec

    assert qt_app.settings["speed"] == spec.SETTINGS_DEFAULTS["speed"]
    assert qt_app.settings["lexicon"] == []
    assert qt_app.settings_dock.volume_spin.value() == spec.SETTINGS_DEFAULTS["volume"]
    assert qt_app.fx_dock._value_widgets["gain_db"].value() == spec.SETTINGS_DEFAULTS["gain_db"]


def test_save_settings_leaves_no_temp_file(tmp_path):
    cfg = str(tmp_path / "config_qt.json")
    qt_settings.save_settings(cfg, {"speed": 1.2})
    assert sorted(os.listdir(tmp_path)) == ["config_qt.json"]


def test_a_failed_replace_keeps_the_old_file_and_removes_the_temp(tmp_path, monkeypatch):
    cfg = str(tmp_path / "config_qt.json")
    qt_settings.save_settings(cfg, {"speed": 1.2})

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(qt_settings.os, "replace", boom)
    qt_settings.save_settings(cfg, {"speed": 9.9})

    with open(cfg, "r", encoding="utf-8") as f:
        assert json.load(f) == {"speed": 1.2}
    assert sorted(os.listdir(tmp_path)) == ["config_qt.json"]


def test_a_write_that_fails_midway_keeps_the_old_file(tmp_path, monkeypatch):
    cfg = str(tmp_path / "config_qt.json")
    qt_settings.save_settings(cfg, {"speed": 1.2})

    def boom(obj, fp, **kwargs):
        fp.write('{"speed": ')
        raise OSError("disk full")

    monkeypatch.setattr(qt_settings.json, "dump", boom)
    qt_settings.save_settings(cfg, {"speed": 9.9})

    with open(cfg, "r", encoding="utf-8") as f:
        assert json.load(f) == {"speed": 1.2}
    assert sorted(os.listdir(tmp_path)) == ["config_qt.json"]


def test_an_old_dict_lexicon_loads_as_the_same_rules_in_a_list(tmp_path):
    cfg = _write_config(tmp_path, {"lexicon": {"Dr": "Doctor", "": "dropped", "TTS": "Tee Tee Ess"}})
    settings = qt_settings.load_settings(cfg)
    assert settings["lexicon"] == [
        {"find": "Dr", "replace": "Doctor", "mode": "literal", "case": False},
        {"find": "TTS", "replace": "Tee Tee Ess", "mode": "literal", "case": False},
    ]


def test_a_list_lexicon_is_cleaned_on_load(tmp_path):
    cfg = _write_config(tmp_path, {"lexicon": [{"find": "a", "replace": "b", "mode": "word", "case": True},
                                               "junk", {"replace": "x"}, {"find": "c"}]})
    assert qt_settings.load_settings(cfg)["lexicon"] == [
        {"find": "a", "replace": "b", "mode": "word", "case": True},
        {"find": "c", "replace": "", "mode": "literal", "case": False},
    ]


def test_the_default_lexicon_is_not_shared_between_loads(tmp_path):
    first = qt_settings.load_settings(str(tmp_path / "missing.json"))
    first["lexicon"].append({"find": "a", "replace": "b", "mode": "literal", "case": False})
    assert qt_settings.load_settings(str(tmp_path / "missing.json"))["lexicon"] == []
