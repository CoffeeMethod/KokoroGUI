"""Tests for TTSApp.load_settings/save_settings/apply_settings
(gui.py:263-436)."""
import json


def test_load_settings_defaults_when_no_config_file(tts_app):
    settings = tts_app.load_settings()
    assert settings["voice"] == "af_heart"
    assert settings["lexicon"] == {}
    assert settings["caching"] is True


def test_load_settings_merges_existing_config_json(tts_app):
    import gui
    with open(gui.CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"voice": "am_adam"}, f)

    settings = tts_app.load_settings()

    assert settings["voice"] == "am_adam"
    assert settings["format"] == "wav"  # untouched default still present


def test_load_settings_corrupt_json_falls_back_to_defaults(tts_app):
    import gui
    with open(gui.CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write("{not valid json")

    settings = tts_app.load_settings()

    assert settings["voice"] == "af_heart"


def test_save_settings_writes_json_with_current_vars(tts_app):
    import gui
    tts_app.voice_var.set("am_liam")
    tts_app.save_settings()

    with open(gui.CONFIG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["voice"] == "am_liam"


def test_change_appearance_and_scaling_persist_to_settings(tts_app):
    tts_app.change_appearance("Light")
    tts_app.change_scaling("120%")

    assert tts_app.settings["appearance"] == "Light"
    assert tts_app.settings["scaling"] == "120%"
