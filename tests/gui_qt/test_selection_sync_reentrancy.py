"""Regression test for item 1 ("Sync layer"): a single user action must not
ping-pong between TranscriptEditor and TimelineView's selection-changed
handlers. Full qt_app fixture; uses the same call-counting idiom already used
elsewhere in this suite (e.g. tests/gui_qt/test_qt_timeline_dock.py's
`monkeypatch.setattr(qt_app, "refresh_timeline", lambda: calls.append(True))`),
adapted here via disconnect/reconnect since the handlers under test are
already bound as signal slots at qt_app construction time (a class-level
monkeypatch after construction would not affect an already-bound
connection)."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor

from kokoro_gui.qt.timeline_view import ClipBlockItem


def _editor(qt_app):
    return qt_app.generation_dock.text_entry


def _set_text_via_real_edit(editor, text):
    cursor = editor.textCursor()
    cursor.select(QTextCursor.SelectionType.Document)
    cursor.insertText(text)


def _clip_block_items(qt_app):
    return [item for item in qt_app.timeline_dock.timeline_view._scene.items() if isinstance(item, ClipBlockItem)]


def _instrument(signal, original_slot):
    """Disconnects `original_slot` from `signal` and reconnects a
    call-counting wrapper that still invokes it, returning the calls list."""
    calls = []
    signal.disconnect(original_slot)

    def wrapper():
        calls.append(True)
        original_slot()

    signal.connect(wrapper)
    return calls


def test_timeline_click_fires_each_handler_at_most_once(qt_app, qtbot):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    character = qt_app.document.characters[0]
    clip = qt_app.document.assign_character_to_range(6, 11, character.id)  # "world"
    qt_app.refresh_timeline()

    view = qt_app.timeline_dock.timeline_view
    editor_calls = _instrument(qt_app.selection.changed, editor._on_selection_model_changed)
    timeline_calls = _instrument(qt_app.selection.changed, view._on_selection_changed)

    block = _clip_block_items(qt_app)[0]
    assert block.clip_id == clip.id
    pos = view.mapFromScene(block.mapToScene(5, 5))
    qtbot.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=pos)

    assert len(editor_calls) == 1
    assert len(timeline_calls) == 1


def test_transcript_cursor_move_into_clip_fires_each_handler_at_most_once(qt_app, qtbot):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    character = qt_app.document.characters[0]
    qt_app.document.assign_character_to_range(6, 11, character.id)  # "world"
    qt_app.refresh_timeline()

    view = qt_app.timeline_dock.timeline_view
    editor_calls = _instrument(qt_app.selection.changed, editor._on_selection_model_changed)
    timeline_calls = _instrument(qt_app.selection.changed, view._on_selection_changed)

    cursor = editor.textCursor()
    cursor.setPosition(8)  # inside "world"
    editor.setTextCursor(cursor)

    assert len(editor_calls) == 1
    assert len(timeline_calls) == 1
