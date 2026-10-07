"""The Timeline dock's Snap to grid button: a view setting kept in
config_qt.json, never in the project (grill PG4)."""
import json

from kokoro_gui.qt import settings as qt_settings


def test_the_button_is_off_by_default_and_the_view_agrees(qt_app):
    dock = qt_app.timeline_dock
    assert dock.snap_button.isCheckable()
    assert dock.snap_button.isChecked() is False
    assert dock.timeline_view.snap_to_grid is False
    assert qt_app.settings["snap_to_grid"] is False


def test_clicking_the_button_turns_the_view_snap_on_and_saves_the_setting(qt_app):
    import kokoro_gui.qt.app as qt_app_module

    dock = qt_app.timeline_dock
    dock.snap_button.click()
    assert dock.timeline_view.snap_to_grid is True
    assert qt_app.settings["snap_to_grid"] is True

    qt_app.save_settings()
    with open(qt_app_module.CONFIG_FILE, "r", encoding="utf-8") as f:
        assert json.load(f)["snap_to_grid"] is True

    dock.snap_button.click()
    assert dock.timeline_view.snap_to_grid is False
    assert qt_app.settings["snap_to_grid"] is False


def test_a_saved_choice_comes_back_on_the_next_launch(qt_app, tmp_path):
    import kokoro_gui.qt.app as qt_app_module

    qt_app.timeline_dock.snap_button.click()
    qt_app.save_settings()

    again = qt_settings.load_settings(qt_app_module.CONFIG_FILE)
    assert again["snap_to_grid"] is True


def test_a_junk_value_in_the_config_falls_back_to_off(tmp_path):
    config = tmp_path / "config_qt.json"
    config.write_text(json.dumps({"snap_to_grid": "yes please"}), encoding="utf-8")
    assert qt_settings.load_settings(str(config))["snap_to_grid"] is False


def test_snap_to_grid_is_not_project_data(qt_app):
    qt_app.timeline_dock.snap_button.click()
    qt_app.save_settings()
    assert "snap_to_grid" not in qt_app.document.settings
    assert "snap_to_grid" not in qt_app.project_settings
