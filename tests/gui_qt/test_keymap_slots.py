"""keymap.KEYS against the app: every slot exists, every row becomes a shortcut."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence

from kokoro_gui.qt import keymap
from kokoro_gui.qt.app import QtTTSApp

CONTEXTS = {
    keymap.WINDOW: Qt.ShortcutContext.WindowShortcut,
    keymap.APP: Qt.ShortcutContext.ApplicationShortcut,
    keymap.TIMELINE: Qt.ShortcutContext.WidgetWithChildrenShortcut,
}


def test_every_slot_name_is_a_method_of_the_app():
    for binding in keymap.KEYS:
        assert callable(getattr(QtTTSApp, binding.slot_name, None)), binding.slot_name


def test_every_binding_becomes_a_shortcut_with_its_sequence_and_context(qt_app):
    for binding in keymap.KEYS:
        shortcut = getattr(qt_app, f"{binding.id}_shortcut")
        assert shortcut.key() == QKeySequence(binding.sequence), binding.id
        assert shortcut.context() == CONTEXTS[binding.scope], binding.id


def test_timeline_keys_live_on_the_timeline_view_and_the_rest_on_the_window(qt_app):
    view = qt_app.timeline_dock.timeline_view
    for binding in keymap.KEYS:
        shortcut = getattr(qt_app, f"{binding.id}_shortcut")
        expected = view if binding.scope == keymap.TIMELINE else qt_app
        assert shortcut.parent() is expected, binding.id


def test_only_the_generate_keys_start_disabled(qt_app):
    for binding in keymap.KEYS:
        shortcut = getattr(qt_app, f"{binding.id}_shortcut")
        assert shortcut.isEnabled() is (not binding.while_generating), binding.id
