"""Tests for TTSApp.load_settings/save_settings/apply_settings
(gui.py:263-436)."""
import json
from unittest.mock import call, MagicMock


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


def test_load_settings_migrates_legacy_font_scaling(tts_app):
    import gui
    with open(gui.CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"font_scaling": "300%"}, f)

    settings = tts_app.load_settings()

    assert settings["font_size"] == 40
    assert "font_scaling" not in settings


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


def test_change_scaling_uses_requested_widget_scale_without_cap(tts_app, monkeypatch):
    import gui
    set_widget_scaling = MagicMock()
    monkeypatch.setattr(gui.ctk, "set_widget_scaling", set_widget_scaling)

    tts_app.change_scaling("250%")
    tts_app.change_scaling("300%")

    assert set_widget_scaling.call_args_list == [call(2.5), call(3.0)]


def test_change_font_size_resizes_text_without_widget_scaling(tts_app, monkeypatch):
    import gui
    set_widget_scaling = MagicMock()
    monkeypatch.setattr(gui.ctk, "set_widget_scaling", set_widget_scaling)
    status_font = tts_app.status_label.cget("font")
    rendered_widgets = [
        tts_app.status_label._label,
        tts_app.preview_btn._text_label,
        tts_app.text_entry._textbox,
    ]
    initial_sizes = [
        abs(int(widget.tk.call("font", "actual", widget.cget("font"), "-size")))
        for widget in rendered_widgets
    ]

    tts_app.change_font_size("32 px")

    assert tts_app.settings["font_size"] == 32
    assert status_font.cget("size") == 32
    assert gui.ctk.ThemeManager.theme["CTkFont"]["size"] == 32
    assert tts_app.ui_font("Roboto", 14, "bold").cget("size") == 34
    for widget, initial_size in zip(rendered_widgets, initial_sizes):
        rendered_size = widget.tk.call("font", "actual", widget.cget("font"), "-size")
        rendered_family = widget.tk.call("font", "actual", widget.cget("font"), "-family")
        assert rendered_family.casefold() == "liberation sans"
        assert abs(int(rendered_size)) > initial_size
    set_widget_scaling.assert_not_called()


def test_apply_settings_falls_back_to_default_for_invalid_scale(tts_app, monkeypatch):
    import gui
    set_widget_scaling = MagicMock()
    monkeypatch.setattr(gui.ctk, "set_widget_scaling", set_widget_scaling)
    tts_app.settings["scaling"] = "invalid"

    tts_app.apply_settings()

    set_widget_scaling.assert_called_once_with(1.0)


def test_accessibility_fonts_are_larger_than_control_base_fonts(tts_app):
    import gui

    assert gui.ctk.ThemeManager.theme["CTkFont"]["size"] == 24
    assert tts_app.ui_font("Roboto", 14, "bold").cget("size") == 26
    assert gui.ctk.DrawEngine.preferred_drawing_method == "polygon_shapes"
