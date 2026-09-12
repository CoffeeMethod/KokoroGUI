"""Tests for the reshaped shell (Claude/PLAN_ui_shell_redesign.md section
1): menu bar, 2x2 dock grid, Transport dock, workspaces, theme, Options."""
from PySide6.QtCore import Qt

from kokoro_gui.qt import theme
from kokoro_gui.qt.workspace import ADVANCED, SIMPLE


def _menu_titles(app):
    return [a.text().replace("&", "") for a in app.menuBar().actions()]


def _action_texts(menu):
    return [a.text().replace("&", "") for a in menu.actions() if not a.isSeparator()]


def test_menu_bar_has_the_four_menus(qt_app):
    assert _menu_titles(qt_app) == ["File", "Edit", "Options", "Workspace"]


def test_file_menu_actions(qt_app):
    texts = _action_texts(qt_app.file_menu)
    assert texts == ["New", "Open...", "Recent", "Save", "Save As...", "Import Text...", "Import Audio...",
                     "Export...", "Quit"]
    assert qt_app.import_audio_action.isEnabled() is False
    assert "ASR" in qt_app.import_audio_action.toolTip()


def test_edit_menu_actions(qt_app):
    assert _action_texts(qt_app.edit_menu) == ["Undo", "Redo", "Cut", "Copy", "Paste", "Characters..."]


def test_options_menu_holds_engine_device_theme_and_toggles(qt_app):
    texts = _action_texts(qt_app.options_menu)
    assert texts[:3] == ["Engine", "Device", "Theme"]
    assert "Copy carries character/FX" in texts
    assert "Paste splits character/FX" in texts
    assert any(t.startswith("JIT streaming") for t in texts)
    assert set(qt_app.device_actions) == {"auto", "cpu", "cuda"}
    assert set(qt_app.theme_actions) == {"light", "dark"}


def test_no_toolbar_and_hidden_central_widget(qt_app):
    from PySide6.QtWidgets import QToolBar

    assert qt_app.findChildren(QToolBar) == []
    central = qt_app.centralWidget()
    assert central.isHidden()
    # Not fixed to 0x0: a fixed central widget caps the height of the row it
    # sits in, and the timeline could never be dragged taller.
    assert central.maximumHeight() > 0


def test_timeline_row_can_be_made_taller(qt_app, qtbot):
    qt_app.resize(1600, 1000)
    qt_app.show()
    qtbot.waitExposed(qt_app)
    qtbot.wait(50)  # let the first-show default-proportions pass run
    before = qt_app.timeline_dock.height()
    qt_app.resizeDocks([qt_app.transcript_dock, qt_app.timeline_dock], [300, 650], Qt.Orientation.Vertical)
    qtbot.wait(20)
    assert qt_app.timeline_dock.height() > before + 100


def test_all_panels_are_docks_in_the_grid(qt_app):
    top, bottom = Qt.DockWidgetArea.TopDockWidgetArea, Qt.DockWidgetArea.LeftDockWidgetArea
    for dock in (qt_app.transcript_dock, qt_app.settings_dock, qt_app.fx_dock, qt_app.lexicon_dock):
        assert not dock.isFloating()
        assert qt_app.dockWidgetArea(dock) == top
    for dock in (qt_app.timeline_dock, qt_app.transport_dock):
        assert not dock.isFloating()
        assert qt_app.dockWidgetArea(dock) == bottom
    tabbed = qt_app.tabifiedDockWidgets(qt_app.settings_dock)
    assert qt_app.fx_dock in tabbed and qt_app.lexicon_dock in tabbed
    assert qt_app.mixing_dock in tabbed  # Kokoro -> Mixing behind the "Voices" tab
    assert qt_app.mixing_dock.windowTitle() == "Voices"
    assert qt_app.mixing_dock.objectName() == "dock_voices"


def test_voices_tab_keeps_its_title_across_engines(qt_app):
    qt_app.switch_engine("dummy")
    assert qt_app.mixing_dock is None and qt_app.voice_clone_dock is None
    qt_app.switch_engine("kokoro")
    assert qt_app.mixing_dock.windowTitle() == "Voices"
    assert qt_app.mixing_dock in qt_app.tabifiedDockWidgets(qt_app.settings_dock)


# -- transport dock -----------------------------------------------------------


def test_transport_dock_carries_generate_menu_and_status(qt_app):
    dock = qt_app.transport_dock
    assert [a.text() for a in dock.generate_menu.actions() if not a.isSeparator()] == [
        "Generate dirty clips", "Auto-split then generate", "Split by paragraph",
    ]
    dock.set_status("Hello", "error")
    assert dock.status_text() == "Hello"
    assert "#ff5555" in dock.progress_bar.styleSheet()
    dock.set_progress(42, "clip 3/5", elapsed=61, eta="00:10")
    assert dock.progress_bar.value() == 42
    assert "clip 3/5" in dock.progress_bar.format()
    assert "01:01" in dock.progress_bar.format()


def test_busy_state_disables_generate_and_enables_cancel(qt_app):
    qt_app.set_ui_state(True)
    assert qt_app.is_busy()
    assert not qt_app.transport_dock.generate_btn.isEnabled()
    assert qt_app.transport_dock.cancel_btn.isEnabled()
    qt_app.set_ui_state(False)
    assert not qt_app.is_busy()
    assert qt_app.transport_dock.generate_btn.isEnabled()


def test_split_by_paragraph_action_writes_the_setting(qt_app):
    qt_app.transport_dock.split_paragraph_action.setChecked(True)
    assert qt_app.settings["auto_split_by_paragraph"] is True
    qt_app.transport_dock.split_paragraph_action.setChecked(False)
    assert qt_app.settings["auto_split_by_paragraph"] is False


# -- workspaces ----------------------------------------------------------------


def test_default_workspace_is_advanced_with_timeline_visible(qt_app):
    assert qt_app.workspaces.active == ADVANCED
    assert qt_app.workspace_actions[ADVANCED].isChecked()
    assert not qt_app.timeline_dock.isHidden()


def test_simple_workspace_hides_the_timeline_only(qt_app):
    qt_app.activate_workspace(SIMPLE)
    assert qt_app.timeline_dock.isHidden()
    assert not qt_app.transport_dock.isHidden()
    assert not qt_app.transcript_dock.isHidden()
    assert qt_app.settings["active_workspace"] == SIMPLE
    assert qt_app.workspace_actions[SIMPLE].isChecked()

    qt_app.activate_workspace(ADVANCED)
    assert not qt_app.timeline_dock.isHidden()


def test_workspace_edits_are_captured_per_workspace_and_reset_forgets_them(qt_app):
    qt_app.timeline_dock.hide()  # a "drag edit" in Advanced
    qt_app.save_settings()
    assert qt_app.settings["workspaces"][ADVANCED]["state"]

    qt_app.activate_workspace(SIMPLE)
    qt_app.activate_workspace(ADVANCED)
    assert qt_app.timeline_dock.isHidden()  # the saved edit came back

    qt_app.reset_workspace()
    assert not qt_app.timeline_dock.isHidden()
    assert ADVANCED not in qt_app.settings["workspaces"] or qt_app.settings["workspaces"][ADVANCED]["state"]


def test_legacy_dock_state_keys_migrate_into_advanced(qt_app):
    from kokoro_gui.qt.workspace import WorkspaceManager

    settings = {"dock_state": "AAAA", "geometry": "BBBB"}
    WorkspaceManager(qt_app, settings)
    assert "dock_state" not in settings and "geometry" not in settings
    assert settings["workspaces"][ADVANCED] == {"state": "AAAA", "geometry": "BBBB"}
    assert settings["active_workspace"] == ADVANCED


# -- theme --------------------------------------------------------------------------


def test_theme_switch_updates_palette_setting_and_signals(qt_app):
    fired = []
    qt_app.themeChanged.connect(lambda: fired.append(True))

    qt_app.set_theme("dark")

    assert qt_app.settings["theme"] == "dark"
    assert theme.current() is theme.DARK
    assert qt_app.theme_actions["dark"].isChecked()
    assert fired == [True]
    qt_app.set_theme("light")
    assert theme.current() is theme.LIGHT


def test_theme_tokens_are_complete_in_both_palettes():
    from dataclasses import fields

    for pal in (theme.LIGHT, theme.DARK):
        for f in fields(theme.Palette):
            value = getattr(pal, f.name)
            assert value, f"{pal.name}.{f.name} is empty"


# -- options ---------------------------------------------------------------------------


def test_device_action_writes_setting_and_reinitializes(qt_app):
    qt_app.engine.init_pipeline_async.reset_mock()
    qt_app.set_device("cpu")
    assert qt_app.settings["device"] == "cpu"
    assert qt_app.device_actions["cpu"].isChecked()
    qt_app.engine.init_pipeline_async.assert_called_with("a", device="cpu")


def test_copy_carries_toggle_controls_the_character_mime_type(qt_app):
    from PySide6.QtGui import QTextCursor

    editor = qt_app.editor
    cursor = editor.textCursor()
    cursor.insertText("hello world")
    alice = qt_app.document.characters[0]
    qt_app.document.assign_character_to_range(0, 5, alice.id)
    cursor = editor.textCursor()
    cursor.setPosition(0)
    cursor.setPosition(5, QTextCursor.MoveMode.KeepAnchor)
    editor.setTextCursor(cursor)

    assert editor.createMimeDataFromSelection().hasFormat(editor.CHARACTER_ID_MIME_TYPE)
    qt_app.copy_carries_action.setChecked(False)
    assert qt_app.settings["character_fx_copy"] is False
    assert not editor.createMimeDataFromSelection().hasFormat(editor.CHARACTER_ID_MIME_TYPE)


def test_space_shortcuts_toggle_the_transport(qt_app):
    toggles = []
    qt_app.transport.toggle = lambda: toggles.append(True)
    qt_app.space_shortcut.activated.disconnect()
    qt_app.ctrl_space_shortcut.activated.disconnect()
    qt_app.space_shortcut.activated.connect(qt_app.transport.toggle)
    qt_app.ctrl_space_shortcut.activated.connect(qt_app.transport.toggle)
    qt_app.space_shortcut.activated.emit()
    qt_app.ctrl_space_shortcut.activated.emit()
    assert toggles == [True, True]
    assert qt_app.space_shortcut.key().toString() == "Space"
    assert qt_app.ctrl_space_shortcut.key().toString() == "Ctrl+Space"


def test_window_title_names_the_project_and_marks_pending_saves(qt_app):
    assert qt_app.windowTitle() == "document - KokoroGUI"
    qt_app.schedule_save()
    assert qt_app.windowTitle() == "document* - KokoroGUI"
    qt_app.save_settings()
    assert qt_app.windowTitle() == "document - KokoroGUI"
