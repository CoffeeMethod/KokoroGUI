"""File > Show in Folder, File > Recent > Clear list, and the Help menu
(plan 06)."""
import os

import pytest

import kokoro_gui.qt.app as app_module
from kokoro_gui.qt import project as project_io


@pytest.fixture
def revealed(monkeypatch):
    paths = []
    monkeypatch.setattr(app_module, "reveal", lambda path: paths.append(path) or True)
    return paths


# -- File > Show in Folder --------------------------------------------------------------


def test_show_in_folder_entries_are_disabled_until_their_target_exists(qt_app):
    qt_app.project_path = None
    qt_app._sync_show_in_folder_actions()
    assert not qt_app.show_project_file_action.isEnabled()
    assert not qt_app.show_last_export_action.isEnabled()


def test_show_project_file_reveals_the_saved_project(qt_app, tmp_path, revealed):
    path = tmp_path / "story.tbaw"
    qt_app.save_project_as(str(path))
    qt_app.wait_for_project_io()
    qt_app.show_in_folder_menu.aboutToShow.emit()
    assert qt_app.show_project_file_action.isEnabled()

    qt_app.show_project_file_action.trigger()

    assert revealed == [qt_app.project_path]


def test_show_working_folder_reveals_the_root_project_dir(qt_app, tmp_path, revealed):
    qt_app.root.project_dir = str(tmp_path)
    qt_app.show_in_folder_menu.aboutToShow.emit()
    assert qt_app.show_working_folder_action.isEnabled()

    qt_app.show_working_folder_action.trigger()

    assert revealed == [str(tmp_path)]


def test_last_export_is_remembered_when_an_export_finishes(qt_app, tmp_path, revealed):
    mix = tmp_path / "mix.wav"
    mix.write_bytes(b"x")
    qt_app.exportWrote.emit(str(mix))
    qt_app.exportFinished.emit(True, f"Exported {mix}")
    qt_app.show_in_folder_menu.aboutToShow.emit()
    assert qt_app.show_last_export_action.isEnabled()

    qt_app.show_last_export_action.trigger()

    assert revealed == [str(mix)]


def test_a_vanished_export_disables_its_entry_on_the_next_open(qt_app, tmp_path):
    mix = tmp_path / "mix.wav"
    mix.write_bytes(b"x")
    qt_app.exportWrote.emit(str(mix))
    mix.unlink()
    qt_app.show_in_folder_menu.aboutToShow.emit()
    assert not qt_app.show_last_export_action.isEnabled()


def test_revealing_a_missing_path_says_so_in_the_status_line(qt_app, tmp_path):
    qt_app._reveal_path(str(tmp_path / "gone.wav"))
    assert "gone" in qt_app.transport_dock.status_text()


# -- File > Recent > Clear list ---------------------------------------------------------


def test_recent_menu_has_no_clear_entry_when_empty(qt_app):
    qt_app.settings["recent_projects"] = []
    qt_app._rebuild_recent_menu()
    assert [a.text() for a in qt_app.recent_menu.actions()] == ["(empty)"]


def test_recent_clear_list_empties_the_menu_and_the_setting(qt_app, tmp_path):
    project_io.remember_recent(qt_app.settings, str(tmp_path / "one.tbaw"))
    project_io.remember_recent(qt_app.settings, str(tmp_path / "two.tbaw"))
    qt_app._rebuild_recent_menu()
    actions = qt_app.recent_menu.actions()
    assert actions[-1].text() == "Clear list"
    assert actions[-2].isSeparator()

    actions[-1].trigger()

    assert qt_app.settings["recent_projects"] == []
    assert [a.text() for a in qt_app.recent_menu.actions()] == ["(empty)"]
