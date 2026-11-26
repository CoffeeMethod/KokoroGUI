"""Options > Language, the window under a translation, and that everything
you can click or type in has a tooltip."""
import json

import pytest
from PySide6.QtCore import QLocale, Qt
from PySide6.QtWidgets import (
    QAbstractButton, QAbstractItemView, QAbstractSlider, QAbstractSpinBox, QComboBox, QDialogButtonBox, QFormLayout,
    QHeaderView, QLabel, QLineEdit, QListView, QMenu, QPlainTextEdit, QScrollBar, QSpinBox, QTabBar, QWidget,
)

from kokoro_gui import i18n
from kokoro_gui.qt import localize, tips
from kokoro_gui.qt.characters_dialog import CharactersDialog
from kokoro_gui.qt.settings_window import SettingsWindow
from kokoro_gui.qt.storage_dialog import StorageDialog
from kokoro_gui.qt.welcome_dialog import WelcomeDialog


@pytest.fixture(autouse=True)
def _english():
    yield
    i18n.set_language("en")


# -- language -------------------------------------------------------------------

def test_the_language_setting_is_kept_for_the_next_launch(qt_app):
    assert qt_app.settings["language"] == "system"
    assert qt_app.language_actions["system"].isChecked()
    qt_app.set_language("en")
    assert qt_app.settings["language"] == "en"
    assert qt_app.language_actions["en"].isChecked() and not qt_app.language_actions["system"].isChecked()
    assert "restart" in qt_app.transport_dock.progress_bar.format()


def test_the_settings_window_changes_it_too(qt_app, qtbot):
    window = SettingsWindow(qt_app)
    qtbot.addWidget(window)
    combo = window.widgets["language"]
    assert combo.currentData() == "system"
    combo.setCurrentIndex(combo.findData("en"))
    window.apply()
    assert qt_app.settings["language"] == "en"


def test_resolve_takes_only_a_language_that_has_a_file(monkeypatch):
    monkeypatch.setattr(i18n, "available_languages", lambda directory=None: [("en", "English"), ("fr", "Français")])
    assert localize.resolve("fr") == "fr"
    for setting in ("de", "../fr", None, 3, ["fr"]):
        assert localize.resolve(setting) == "en"
    monkeypatch.setattr(QLocale, "uiLanguages", lambda self: ["fr-CA", "en"])
    assert localize.resolve("system") == "fr"


def test_apply_loads_the_file_and_sets_the_direction(tmp_path, monkeypatch, qapp):
    (tmp_path / "ar.json").write_text(json.dumps({"name": "العربية", "strings": {"Play": "تشغيل"}}), encoding="utf-8")
    monkeypatch.setattr(i18n, "LOCALES_DIR", str(tmp_path))
    before = qapp.layoutDirection()
    try:
        assert localize.apply(qapp, "ar") == "ar"
        assert i18n.tr("Play") == "تشغيل"
        assert qapp.layoutDirection() == Qt.LayoutDirection.RightToLeft
    finally:
        qapp.setLayoutDirection(before)


def test_the_window_opens_in_another_language(tmp_path, qt_app, qtbot):
    """Dialogs built after the language is set show it; the keys a project
    or a setting stores stay as they are."""
    (tmp_path / "xx.json").write_text(json.dumps({"strings": {
        "General": "Algemeen", "Light": "Licht", "Generation": "Opwekken", "Advanced": "Uitgebreid"}}), encoding="utf-8")
    i18n.set_language("xx", str(tmp_path))
    window = SettingsWindow(qt_app)
    qtbot.addWidget(window)
    assert window.page_list.item(0).text() == "Algemeen"
    theme = window.widgets["theme"]
    assert theme.itemText(theme.findData("light")) == "Licht"
    window.show_page("Generation")
    assert window.page_list.currentItem().text() == "Opwekken"
    assert qt_app.workspaces.active == "Advanced"


# -- tooltips -------------------------------------------------------------------

_CONTROLS = (QAbstractButton, QComboBox, QAbstractSpinBox, QLineEdit, QPlainTextEdit, QAbstractItemView,
             QAbstractSlider)
_PARTS = (QAbstractSpinBox, QComboBox, QTabBar, QAbstractItemView, QHeaderView, QLineEdit)


def _is_part(widget) -> bool:
    """A piece Qt builds inside another control, or a dialog's standard
    button, which says what it does."""
    if type(widget) is QListView:
        return True  # a combo box's popup
    parent = widget.parent()
    while parent is not None:
        if isinstance(parent, (*_PARTS, QDialogButtonBox)):
            return True
        parent = parent.parent() if not parent.isWindow() else None
    return isinstance(widget, (QScrollBar, QHeaderView)) or widget.objectName().startswith("qt_")


def _without_tooltip(root) -> list:
    missing = []
    for widget in root.findChildren(QWidget):
        if isinstance(widget, _CONTROLS) and not _is_part(widget) and not widget.toolTip():
            text = widget.text() if isinstance(widget, QAbstractButton) else ""
            missing.append(f"{type(widget).__name__} {widget.objectName() or text}")
    return missing


def test_the_docks_have_tooltips(qt_app):
    docks = [d for d in qt_app._all_docks() if d is not None]
    assert docks
    assert [m for dock in docks for m in _without_tooltip(dock)] == []


@pytest.mark.parametrize("build", [SettingsWindow, CharactersDialog, StorageDialog, WelcomeDialog])
def test_the_dialogs_have_tooltips(qt_app, qtbot, build):
    dialog = build(qt_app)
    qtbot.addWidget(dialog)
    assert _without_tooltip(dialog) == []


def test_the_export_dialog_has_tooltips(qt_app, qtbot):
    from kokoro_gui.qt.docks.export_dialog import ExportDialog  # imports app.py, so not at module level

    dialog = ExportDialog(qt_app)
    qtbot.addWidget(dialog)
    assert _without_tooltip(dialog) == []


def test_the_menus_have_tooltips(qt_app):
    """Every entry of the menu bar, with the menu set to show it. An entry
    whose tooltip is only its own text has none."""
    missing = []

    def look(menu: QMenu, trail: str):
        if not menu.toolTipsVisible():
            missing.append(f"{trail}: tooltips are off")
        for action in menu.actions():
            if action.isSeparator():
                continue
            if action.menu() is not None:
                look(action.menu(), f"{trail} > {action.text()}")
            elif action.toolTip().rstrip(".") in ("", action.text().replace("&", "").rstrip(".")):
                missing.append(f"{trail} > {action.text()}")

    for top in qt_app.menuBar().actions():
        look(top.menu(), top.text())
    assert missing == []


def test_the_block_menu_has_tooltips(qt_app):
    document = qt_app.document
    document.text = "hello world"
    clip = document.assign_character_to_range(0, 5, document.characters[0].id)
    qt_app.refresh_timeline()
    view = qt_app.timeline_dock.timeline_view
    menu = view._build_batch_menu(QMenu(view), [clip.id])
    assert [a.text() for a in menu.actions() if not a.isSeparator() and a.menu() is None
            and a.toolTip() == a.text()] == []


def test_a_row_label_shares_its_controls_tooltip(qtbot):
    root = QWidget()
    qtbot.addWidget(root)
    form = QFormLayout(root)
    spin, own = QSpinBox(), QSpinBox()
    spin.setToolTip("How many.")
    own.setToolTip("Control.")
    form.addRow("Count:", spin)
    label = QLabel("Own:")
    label.setToolTip("Label.")
    form.addRow(label, own)
    tips.label_rows(root)
    assert form.labelForField(spin).toolTip() == "How many."
    assert label.toolTip() == "Label."
