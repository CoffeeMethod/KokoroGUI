"""The right-hand dock group's tabs sit on top, stay readable and stay there
(plan 47 pass 3, grill PG32). Native Qt only: `setTabPosition`, no elision,
scroll buttons, a tooltip with the full name."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTabBar, QTabWidget, QToolButton

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


def _scrollers(app):
    bar = _right_bar(app)
    return [b for b in bar.findChildren(QToolButton) if b.isVisible()]


def _colours_inside_border(button):
    image = button.grab().toImage()
    seen = set()
    for x in range(2, image.width() - 2):
        for y in range(2, image.height() - 2):
            seen.add(image.pixel(x, y))
    return len(seen)


def _narrow(app, qtbot):
    app.resize(900, 700)
    _settle(app, qtbot)
    qtbot.wait(30)


def test_the_tab_scroll_buttons_show_arrows_in_both_themes(qt_app, qtbot):
    """A QToolButton rule with 8 px side padding left the 16 px scroller
    buttons no room for the arrow: they were empty boxes."""
    for name in ("dark", "light"):
        qt_app.set_theme(name)
        _narrow(qt_app, qtbot)
        buttons = _scrollers(qt_app)
        assert len(buttons) == 2, name
        for button in buttons:
            assert button.width() >= 18
            assert _colours_inside_border(button) > 1, (name, button.arrowType())


def test_the_scroll_buttons_have_names_and_tooltips(qt_app, qtbot):
    _narrow(qt_app, qtbot)
    names = sorted(b.accessibleName() for b in _scrollers(qt_app))
    assert names == ["Scroll tabs left", "Scroll tabs right"]
    assert all(b.toolTip() == b.accessibleName() for b in _scrollers(qt_app))
