"""Named custom workspaces, the Panels submenu and hardened layout restore
(plan 47 pass 1, grill UI17 and PG31)."""
import base64
import json
import os

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QInputDialog, QMessageBox

import kokoro_gui.qt.app as qt_app_module
from kokoro_gui.qt import settings as qt_settings
from kokoro_gui.qt.workspace import (
    ADVANCED,
    SIMPLE,
    WorkspaceExistsError,
    WorkspaceManager,
    WorkspaceNameError,
    menu_label,
    validate_name,
)


def _answer_text(monkeypatch, text, ok=True):
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: (text, ok)))


def _answer_question(monkeypatch, yes):
    answer = QMessageBox.StandardButton.Yes if yes else QMessageBox.StandardButton.No
    asked = []

    def question(*args, **kwargs):
        asked.append(args[2] if len(args) > 2 else "")
        return answer

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    return asked


def _save_as(qt_app, monkeypatch, name):
    _answer_text(monkeypatch, name)
    qt_app.save_workspace_as()


def _menu_texts(menu):
    return [action.text() for action in menu.actions() if not action.isSeparator()]


def _config():
    with open(qt_app_module.CONFIG_FILE, encoding="utf-8") as f:
        return json.load(f)


# -- names -----------------------------------------------------------------------


def test_a_name_is_trimmed_and_unicode_is_kept():
    assert validate_name("  Mixing ") == "Mixing"
    assert validate_name("Café — récit") == "Café — récit"  # NFC
    assert validate_name("Dialogue & FX") == "Dialogue & FX"


@pytest.mark.parametrize("name", ["", "   ", "a" * 65, "two\nlines", "tab\there", "advanced", " SIMPLE "])
def test_a_blank_long_control_or_built_in_name_is_refused(name):
    with pytest.raises(WorkspaceNameError):
        validate_name(name)


def test_a_duplicate_name_is_refused_ignoring_case_except_the_one_being_renamed():
    with pytest.raises(WorkspaceExistsError) as clash:
        validate_name("mixing", ["Mixing"])
    assert clash.value.name == "Mixing"
    assert validate_name("MIXING", ["Mixing"], current="Mixing") == "MIXING"


def test_a_name_with_an_ampersand_shows_the_ampersand_in_the_menu(qt_app, monkeypatch):
    assert menu_label("Dialogue & FX") == "Dialogue && FX"
    _save_as(qt_app, monkeypatch, "Dialogue & FX")
    assert "Dialogue && FX" in _menu_texts(qt_app.workspace_menu)
    assert qt_app.settings["custom_workspaces"].keys() == {"Dialogue & FX"}


def test_a_name_that_looks_like_a_path_is_only_a_key(qt_app, monkeypatch, tmp_path):
    before = sorted(os.listdir(tmp_path))
    _save_as(qt_app, monkeypatch, "../../evil")
    assert "../../evil" in qt_app.settings["custom_workspaces"]
    after = sorted(os.listdir(tmp_path))
    assert set(after) - set(before) <= {"config_qt.json"}


def test_save_as_refuses_a_bad_name_with_a_message(qt_app, monkeypatch):
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a[2])))
    _save_as(qt_app, monkeypatch, "Simple")
    assert qt_app.settings["custom_workspaces"] == {}
    assert qt_app.workspaces.active == ADVANCED
    assert warned and "built-in" in warned[0]


# -- save as, update, revert -------------------------------------------------------


def test_save_as_creates_and_activates_a_snapshot_of_the_current_layout(qt_app, monkeypatch):
    qt_app.timeline_dock.hide()
    _save_as(qt_app, monkeypatch, "Narration")

    assert qt_app.workspaces.active == "Narration"
    assert qt_app.workspace_actions["Narration"].isChecked()
    assert "Narration" in _menu_texts(qt_app.workspace_menu)
    assert not qt_app.workspaces.is_modified()
    assert not qt_app.update_workspace_action.isEnabled()
    assert not qt_app.revert_workspace_action.isEnabled()
    saved = _config()
    assert saved["active_workspace"] == "Narration"
    assert "Narration" in saved["custom_workspaces"]


def test_moving_panels_marks_the_workspace_modified_and_never_touches_the_snapshot(qt_app, monkeypatch):
    _save_as(qt_app, monkeypatch, "Narration")
    snapshot = dict(qt_app.settings["custom_workspaces"]["Narration"])

    qt_app.timeline_dock.hide()
    qt_app.save_settings()  # a general settings save
    qt_app.save_layout()    # the layout save

    assert qt_app.workspaces.is_modified()
    assert qt_app.settings["custom_workspaces"]["Narration"] == snapshot
    assert qt_app.settings["workspace_live"]["workspace"] == "Narration"
    assert qt_app.settings["workspace_live"]["modified"] is True
    assert "Narration (modified)" in _menu_texts(qt_app.workspace_menu)
    assert qt_app.update_workspace_action.isEnabled() and qt_app.revert_workspace_action.isEnabled()


def test_raising_another_tab_is_not_a_modification(qt_app, monkeypatch):
    _save_as(qt_app, monkeypatch, "Narration")
    qt_app.lexicon_dock.raise_()
    assert not qt_app.workspaces.is_modified()


def test_update_replaces_the_snapshot_and_revert_brings_it_back(qt_app, monkeypatch):
    _save_as(qt_app, monkeypatch, "Narration")
    qt_app.timeline_dock.hide()
    qt_app.update_workspace()
    assert not qt_app.workspaces.is_modified()

    qt_app.timeline_dock.show()
    assert qt_app.workspaces.is_modified()
    asked = _answer_question(monkeypatch, yes=True)
    qt_app.revert_workspace()

    assert asked  # a modified arrangement is confirmed before it is discarded
    assert qt_app.timeline_dock.isHidden()
    assert not qt_app.workspaces.is_modified()


def test_declining_revert_keeps_the_live_arrangement(qt_app, monkeypatch):
    _save_as(qt_app, monkeypatch, "Narration")
    qt_app.timeline_dock.hide()
    _answer_question(monkeypatch, yes=False)
    qt_app.revert_workspace()
    assert qt_app.timeline_dock.isHidden()
    assert qt_app.workspaces.is_modified()


def test_switching_away_and_back_loads_the_saved_snapshot(qt_app, monkeypatch):
    qt_app.timeline_dock.hide()
    _save_as(qt_app, monkeypatch, "Narration")
    qt_app.timeline_dock.show()
    qt_app.lexicon_dock.hide()
    snapshot = dict(qt_app.settings["custom_workspaces"]["Narration"])

    qt_app.activate_workspace(ADVANCED)
    assert qt_app.settings["custom_workspaces"]["Narration"] == snapshot  # switching never writes it
    qt_app.activate_workspace("Narration")

    assert qt_app.timeline_dock.isHidden()
    assert not qt_app.lexicon_dock.isHidden()
    assert not qt_app.workspaces.is_modified()


def test_save_as_over_an_existing_name_asks_first(qt_app, monkeypatch):
    _save_as(qt_app, monkeypatch, "Narration")
    first = dict(qt_app.settings["custom_workspaces"]["Narration"])
    qt_app.timeline_dock.hide()

    _answer_question(monkeypatch, yes=False)
    _save_as(qt_app, monkeypatch, "narration")
    assert qt_app.settings["custom_workspaces"]["Narration"] == first

    _answer_question(monkeypatch, yes=True)
    _save_as(qt_app, monkeypatch, "narration")
    assert list(qt_app.settings["custom_workspaces"]) == ["Narration"]
    assert qt_app.settings["custom_workspaces"]["Narration"] != first


def test_floating_tabbed_and_closed_panels_come_back_from_a_snapshot(qt_app, qtbot, monkeypatch):
    qt_app.show()
    qtbot.waitExposed(qt_app)
    qt_app.fx_dock.setFloating(True)
    qt_app.removeDockWidget(qt_app.lexicon_dock)
    qt_app.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, qt_app.lexicon_dock)
    qt_app.tabifyDockWidget(qt_app.transcript_dock, qt_app.lexicon_dock)
    qt_app.lexicon_dock.show()
    qt_app.outline_dock.hide()
    _save_as(qt_app, monkeypatch, "Edit")

    qt_app.activate_workspace(ADVANCED)
    qt_app.reset_workspace()
    assert not qt_app.fx_dock.isFloating() and not qt_app.outline_dock.isHidden()
    qt_app.activate_workspace("Edit")

    assert qt_app.fx_dock.isFloating()
    assert qt_app.lexicon_dock in qt_app.tabifiedDockWidgets(qt_app.transcript_dock)
    assert qt_app.outline_dock.isHidden()
    assert not qt_app.workspaces.is_modified()


# -- rename, delete, restore default --------------------------------------------


def test_rename_keeps_the_workspace_active_and_refuses_duplicates(qt_app, monkeypatch):
    _save_as(qt_app, monkeypatch, "One")
    _save_as(qt_app, monkeypatch, "Two")
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a[2])))

    _answer_text(monkeypatch, "one")
    qt_app.rename_workspace()  # renames the active one, Two
    assert warned and list(qt_app.settings["custom_workspaces"]) == ["One", "Two"]

    _answer_text(monkeypatch, "Three")
    qt_app.rename_workspace()
    assert list(qt_app.settings["custom_workspaces"]) == ["One", "Three"]
    assert qt_app.workspaces.active == "Three"
    assert qt_app.workspace_actions["Three"].isChecked()
    assert "Two" not in qt_app.workspace_actions
    assert _config()["active_workspace"] == "Three"


def test_built_ins_cannot_be_renamed_or_deleted(qt_app, monkeypatch):
    assert not qt_app.rename_workspace_action.isEnabled()
    assert not qt_app.delete_workspace_action.isEnabled()
    with pytest.raises(WorkspaceNameError):
        qt_app.workspaces.rename(ADVANCED, "Mine")
    assert not qt_app.workspaces.delete(SIMPLE)


def test_rename_or_delete_from_a_built_in_asks_which_workspace(qt_app, monkeypatch):
    _save_as(qt_app, monkeypatch, "One")
    qt_app.activate_workspace(ADVANCED)
    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(lambda *a, **k: ("One", True)))
    _answer_question(monkeypatch, yes=True)
    qt_app.delete_workspace()
    assert qt_app.settings["custom_workspaces"] == {}
    assert qt_app.workspaces.active == ADVANCED


def test_deleting_the_active_workspace_confirms_then_loads_advanced(qt_app, monkeypatch):
    qt_app.activate_workspace(SIMPLE)
    _save_as(qt_app, monkeypatch, "Narration")
    advanced_before = qt_app.settings["workspaces"].get(ADVANCED)

    _answer_question(monkeypatch, yes=False)
    qt_app.delete_workspace()
    assert "Narration" in qt_app.settings["custom_workspaces"]

    asked = _answer_question(monkeypatch, yes=True)
    qt_app.delete_workspace()

    assert "Narration" in asked[0]
    assert qt_app.settings["custom_workspaces"] == {}
    assert qt_app.workspaces.active == ADVANCED
    assert qt_app.workspace_actions[ADVANCED].isChecked()
    assert "Narration" not in _menu_texts(qt_app.workspace_menu)
    assert not qt_app.timeline_dock.isHidden()  # Advanced, not the Simple layout left on screen
    if advanced_before is not None:
        assert qt_app.settings["workspaces"][ADVANCED]["state"]


def test_restore_default_on_a_custom_workspace_keeps_its_snapshot(qt_app, monkeypatch):
    qt_app.timeline_dock.hide()
    _save_as(qt_app, monkeypatch, "Narration")
    snapshot = dict(qt_app.settings["custom_workspaces"]["Narration"])

    asked = _answer_question(monkeypatch, yes=True)
    qt_app.reset_workspace()

    assert not asked  # nothing modified to discard
    assert not qt_app.timeline_dock.isHidden()
    assert qt_app.settings["custom_workspaces"]["Narration"] == snapshot
    assert qt_app.workspaces.is_modified()

    qt_app.lexicon_dock.hide()
    asked = _answer_question(monkeypatch, yes=False)
    qt_app.reset_workspace()
    assert asked and qt_app.lexicon_dock.isHidden()


# -- restart (PG31) ----------------------------------------------------------------


def _relaunch(qt_app, qtbot):
    qt_app.close()
    second = qt_app_module.QtTTSApp()
    qtbot.addWidget(second)
    return second


def test_restart_restores_the_live_arrangement_with_its_modified_mark(qt_app, qtbot, monkeypatch):
    _save_as(qt_app, monkeypatch, "Narration")
    qt_app.timeline_dock.hide()

    second = _relaunch(qt_app, qtbot)

    assert second.workspaces.active == "Narration"
    assert second.timeline_dock.isHidden()
    assert second.workspaces.is_modified()
    second._sync_workspace_actions()
    assert "Narration (modified)" in _menu_texts(second.workspace_menu)

    second.activate_workspace(ADVANCED)
    second.activate_workspace("Narration")  # the menu loads the snapshot
    assert not second.timeline_dock.isHidden()
    assert not second.workspaces.is_modified()


def test_restart_of_an_unmodified_workspace_is_unmodified(qt_app, qtbot, monkeypatch):
    qt_app.lexicon_dock.hide()
    _save_as(qt_app, monkeypatch, "Narration")
    second = _relaunch(qt_app, qtbot)
    assert second.workspaces.active == "Narration"
    assert second.lexicon_dock.isHidden()
    assert not second.workspaces.is_modified()


# -- panels ------------------------------------------------------------------------


def test_panels_lists_every_dock_with_its_own_toggle_action(qt_app):
    qt_app._rebuild_panels_menu()
    actions = qt_app.panels_menu.actions()
    assert [a.text() for a in actions] == ["Transcript", "Settings", "Audio FX", "Lexicon", "Outline", "Proof",
                                           "Queue", "Exports", "Voices", "Video", "Timeline", "Transport"]
    assert qt_app.timeline_dock.toggleViewAction() in actions
    assert qt_app.mixing_dock.toggleViewAction() in actions


def test_a_closed_panel_reopens_from_panels_and_the_rest_stays_put(qt_app):
    from kokoro_gui.qt.workspace import layout_fingerprint

    others = [d for d in qt_app._all_docks() if d is not qt_app.timeline_dock]
    before = layout_fingerprint(qt_app, others)[0]
    widget = qt_app.timeline_dock

    qt_app.timeline_dock.close()
    assert qt_app.timeline_dock.isHidden()
    assert not qt_app.timeline_dock.toggleViewAction().isChecked()
    qt_app._rebuild_panels_menu()
    qt_app.timeline_dock.toggleViewAction().trigger()

    assert qt_app.timeline_dock is widget and not qt_app.timeline_dock.isHidden()
    assert layout_fingerprint(qt_app, others)[0] == before


def test_the_voices_entry_follows_an_editor_replacement(qt_app):
    old = qt_app.mixing_dock.toggleViewAction()
    qt_app.set_voices_engine("audio8")
    qt_app._rebuild_panels_menu()
    actions = qt_app.panels_menu.actions()

    assert old not in actions
    assert qt_app.voice_clone_dock.toggleViewAction() in actions
    assert [a.text() for a in actions].count("Voices") == 1


def test_a_closed_voices_panel_stays_closed_after_an_engine_change(qt_app):
    qt_app.mixing_dock.close()
    qt_app.voices_engine_id = "audio8"
    qt_app._sync_mixing_dock()
    qt_app._sync_voice_clone_dock()
    assert qt_app.voice_clone_dock.isHidden()
    assert not qt_app.voice_clone_dock.toggleViewAction().isChecked()


def test_a_floating_voices_panel_stays_floating_after_an_engine_change(qt_app):
    qt_app.mixing_dock.setFloating(True)
    qt_app.voices_engine_id = "audio8"
    qt_app._sync_mixing_dock()
    qt_app._sync_voice_clone_dock()
    assert qt_app.voice_clone_dock.isFloating()


def test_an_engine_with_no_voice_editor_explains_the_missing_entry(qt_app, monkeypatch):
    monkeypatch.setattr(qt_app, "mixing_dock", None)
    monkeypatch.setattr(qt_app, "voice_clone_dock", None)
    qt_app._rebuild_panels_menu()
    voices = [a for a in qt_app.panels_menu.actions() if a.text() == "Voices"]
    assert len(voices) == 1 and not voices[0].isEnabled()
    assert "no voice editor" in voices[0].toolTip()


def test_a_custom_workspace_survives_an_engine_change(qt_app, monkeypatch):
    _save_as(qt_app, monkeypatch, "Narration")
    qt_app.set_voices_engine("audio8")
    assert qt_app.workspaces.active == "Narration"
    assert "Narration" in qt_app.settings["custom_workspaces"]
    qt_app.activate_workspace(ADVANCED)
    qt_app.activate_workspace("Narration")
    assert qt_app.voice_clone_dock.objectName() == "dock_voices"
    assert not qt_app.voice_clone_dock.isFloating()


# -- layout only: no project change ------------------------------------------------


def test_a_layout_edit_never_marks_the_project_unsaved(qt_app, monkeypatch):
    qt_app.save_settings()
    assert "*" not in qt_app.windowTitle()

    qt_app.timeline_dock.hide()
    qt_app.activate_workspace(SIMPLE)
    _save_as(qt_app, monkeypatch, "Narration")
    qt_app.lexicon_dock.hide()
    qt_app.save_layout()

    assert not qt_app._save_timer.isActive()
    assert "*" not in qt_app.windowTitle()
    assert not qt_app.any_project_dirty()


def test_a_dock_closed_by_the_user_schedules_a_layout_save(qt_app):
    assert not qt_app._layout_timer.isActive()
    qt_app.timeline_dock.close()
    assert qt_app._layout_timer.isActive()
    assert not qt_app._save_timer.isActive()


def test_restoring_a_layout_does_not_schedule_a_save(qt_app):
    qt_app._layout_timer.stop()
    qt_app.workspaces.activate(SIMPLE, save_outgoing=False)
    assert not qt_app._layout_timer.isActive()


# -- damaged settings --------------------------------------------------------------


def _blob(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def test_damaged_workspace_entries_are_dropped_with_a_notice(qt_app):
    good = {"state": qt_settings.encode_bytes(qt_app.saveState()), "geometry": None}
    settings = {
        "workspaces": {ADVANCED: ["not", "a", "dict"], SIMPLE: {"state": 5}},
        "custom_workspaces": {"": good, "Ok": good, "Bad": {"state": None}, "Bad2": "x", "Simple": good,
                              " Spaced ": good},
        "workspace_live": {"workspace": 3},
        "active_workspace": "Gone",
    }
    manager = WorkspaceManager(qt_app, settings)

    assert settings["workspaces"] == {}
    assert list(settings["custom_workspaces"]) == ["Ok", "Spaced"]
    assert settings["workspace_live"] == {}
    assert manager.active == ADVANCED
    assert any("damaged" in notice for notice in manager.notices)


def test_a_custom_workspaces_value_of_the_wrong_type_is_reset(qt_app):
    settings = {"custom_workspaces": ["Narration"], "active_workspace": "Narration"}
    manager = WorkspaceManager(qt_app, settings)
    assert settings["custom_workspaces"] == {}
    assert manager.active == ADVANCED
    assert manager.notices


def test_a_snapshot_that_cannot_restore_falls_back_with_a_readable_notice(qt_app, monkeypatch):
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a[2])))
    _save_as(qt_app, monkeypatch, "Narration")
    qt_app.activate_workspace(ADVANCED)
    good = bytes(qt_app.saveState())
    qt_app.settings["custom_workspaces"]["Narration"]["state"] = _blob(good[:4] + os.urandom(2))
    qt_app.timeline_dock.hide()

    qt_app.activate_workspace("Narration")

    assert not qt_app.timeline_dock.isHidden()  # the default layout, not an empty window
    assert all(not dock.isHidden() for dock in (qt_app.transcript_dock, qt_app.settings_dock,
                                                 qt_app.transport_dock))
    assert warned and "Narration" in warned[0] and "Update workspace" in warned[0]
    assert "Narration" in qt_app.transport_dock.status_text()
    assert qt_app.workspaces.active == "Narration"
    assert qt_app.workspaces.is_modified()  # so Update can replace the broken snapshot


def test_a_launch_with_a_damaged_custom_workspace_shows_the_notice(qt_app, qtbot):
    qt_app.close()
    with open(qt_app_module.CONFIG_FILE, encoding="utf-8") as f:
        data = json.load(f)
    data["custom_workspaces"] = {"Narration": {"state": _blob(os.urandom(64)), "geometry": None}}
    data["active_workspace"] = "Narration"
    data["workspace_live"] = {}
    with open(qt_app_module.CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f)

    second = qt_app_module.QtTTSApp()
    qtbot.addWidget(second)

    assert "Narration" in second.transport_dock.status_text()
    assert not second.timeline_dock.isHidden() and not second.transcript_dock.isHidden()


def test_a_floating_panel_off_every_screen_comes_back(qt_app):
    from PySide6.QtGui import QGuiApplication

    qt_app.fx_dock.setFloating(True)
    qt_app.fx_dock.show()
    qt_app.fx_dock.setGeometry(100000, 100000, 300, 200)
    qt_app.workspaces.recover_offscreen_docks()

    frame = qt_app.fx_dock.frameGeometry()
    assert any(screen.availableGeometry().intersects(frame) for screen in QGuiApplication.screens())
