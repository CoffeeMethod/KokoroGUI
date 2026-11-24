"""The right-hand dock group's tabs sit on top, stay readable and stay there
(plan 47 pass 3, grill PG32). Native Qt only: `setTabPosition`, no elision,
scroll buttons, a tooltip with the full name."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTabBar, QTabWidget

import kokoro_gui.qt.app as qt_app_module
from kokoro_gui.qt import settings as qt_settings
from kokoro_gui.qt.workspace import ADVANCED, SIMPLE

NORTH = QTabBar.Shape.RoundedNorth


def _settle(app, qtbot):
    app.show()
    qtbot.wait(30)
    app.tune_dock_tab_bars()


def _right_bar(app):
    """The tab bar of the settings group: the one that lists Voice & Clip."""
    for bar in app.findChildren(QTabBar, options=Qt.FindChildOption.FindDirectChildrenOnly):
        if any(bar.tabText(i).startswith("Voice") for i in range(bar.count())):
            return bar
    raise AssertionError("no tab bar holds the settings group")


def _assert_readable_top(app):
    bars = [b for b in app.findChildren(QTabBar, options=Qt.FindChildOption.FindDirectChildrenOnly) if b.count()]
    assert bars
    for bar in bars:
        assert bar.shape() == NORTH
        assert bar.elideMode() == Qt.TextElideMode.ElideNone
        assert bar.usesScrollButtons()
    bar = _right_bar(app)
    assert bar.count() >= 8  # nine, or eight with a panel closed
    assert app.tabPosition(Qt.DockWidgetArea.TopDockWidgetArea) == QTabWidget.TabPosition.North


def _relaunch(qt_app, qtbot):
    qt_app.close()
    second = qt_app_module.QtTTSApp()
    qtbot.addWidget(second)
    return second


def test_tabs_are_on_top_and_unelided(qt_app, qtbot):
    _settle(qt_app, qtbot)
    _assert_readable_top(qt_app)


def test_every_tab_has_its_full_name_as_a_tooltip(qt_app, qtbot):
    _settle(qt_app, qtbot)
    bar = _right_bar(qt_app)
    tips = [bar.tabToolTip(i) for i in range(bar.count())]
    for name in ("Voice & Clip", "Audio FX", "Lexicon", "Outline", "Proof", "Queue", "Exports", "Voices", "Video"):
        assert name in tips
    assert not any("&&" in t for t in tips)


def test_tabs_stay_on_top_after_switching_workspaces(qt_app, qtbot):
    _settle(qt_app, qtbot)
    qt_app.activate_workspace(SIMPLE)
    _settle(qt_app, qtbot)
    qt_app.activate_workspace(ADVANCED)
    _settle(qt_app, qtbot)
    _assert_readable_top(qt_app)


def test_tabs_are_on_top_after_restore_default_layout(qt_app, qtbot):
    _settle(qt_app, qtbot)
    qt_app.workspaces.reset()
    _settle(qt_app, qtbot)
    _assert_readable_top(qt_app)


def test_tabs_are_on_top_after_a_restart_restore(qt_app, qtbot):
    _settle(qt_app, qtbot)
    qt_app.lexicon_dock.hide()
    qt_app.save_layout()
    second = _relaunch(qt_app, qtbot)
    assert second.lexicon_dock.isHidden()
    _settle(second, qtbot)
    _assert_readable_top(second)


def test_tabs_are_readable_after_a_voices_editor_replacement(qt_app, qtbot):
    _settle(qt_app, qtbot)
    qt_app.set_voices_engine("audio8")
    _settle(qt_app, qtbot)
    _assert_readable_top(qt_app)
    bar = _right_bar(qt_app)
    assert "Voices" in [bar.tabToolTip(i) for i in range(bar.count())]
    qt_app.set_voices_engine("kokoro")
    _settle(qt_app, qtbot)
    _assert_readable_top(qt_app)


def test_a_rebuilt_tab_bar_is_tuned_without_a_call(qt_app, qtbot):
    _settle(qt_app, qtbot)
    qt_app.removeDockWidget(qt_app.queue_dock)
    qt_app.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, qt_app.queue_dock)
    qt_app.tabifyDockWidget(qt_app.settings_dock, qt_app.queue_dock)
    qt_app.queue_dock.show()
    qtbot.wait(60)  # the zero-delay tune timer, no explicit call
    for bar in qt_app.findChildren(QTabBar, options=Qt.FindChildOption.FindDirectChildrenOnly):
        if bar.count():
            assert bar.elideMode() == Qt.TextElideMode.ElideNone


def test_a_layout_saved_with_bottom_tabs_still_restores(qt_app, qtbot):
    """A layout from before this change was saved while the tabs sat at the
    bottom. `saveState` doesn't carry the tab position, so the blob restores
    as it was and the tabs come up on top."""
    qt_app.setTabPosition(Qt.DockWidgetArea.AllDockWidgetAreas, QTabWidget.TabPosition.South)
    _settle_without_tune(qt_app, qtbot)
    qt_app.lexicon_dock.hide()
    qt_app.workspaces.capture()
    entry = qt_app.settings["workspaces"][ADVANCED]
    assert entry["state"]
    qt_app.close()
    path = qt_app_module.CONFIG_FILE
    settings = qt_settings.load_settings(path)
    settings["workspaces"] = {ADVANCED: entry}
    settings["active_workspace"] = ADVANCED
    qt_settings.save_settings(path, settings)

    second = qt_app_module.QtTTSApp()
    qtbot.addWidget(second)
    assert second.workspaces.notices == []
    assert second.lexicon_dock.isHidden()
    _settle(second, qtbot)
    _assert_readable_top(second)


def _settle_without_tune(app, qtbot):
    app.show()
    qtbot.wait(30)
