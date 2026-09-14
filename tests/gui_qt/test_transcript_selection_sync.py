"""Tests for the transcript <-> timeline selection sync (item 1, "Sync
layer") - full qt_app fixture, covering TranscriptEditor's
_on_cursor_position_changed/_on_selection_model_changed and the round trip
through TimelineView's click-to-select."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor

from kokoro_gui.qt.timeline_view import ClipBlockItem


def _editor(qt_app):
    return qt_app.editor


def _set_text_via_real_edit(editor, text):
    cursor = editor.textCursor()
    cursor.select(QTextCursor.SelectionType.Document)
    cursor.insertText(text)


def _clip_block_items(qt_app):
    return [item for item in qt_app.timeline_dock.timeline_view._scene.items() if isinstance(item, ClipBlockItem)]


def test_caret_inside_clip_range_selects_that_clip(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    character = qt_app.document.characters[0]
    clip = qt_app.document.assign_character_to_range(6, 11, character.id)  # "world"

    cursor = editor.textCursor()
    cursor.setPosition(8)
    editor.setTextCursor(cursor)

    assert qt_app.selection.selected_clip_id == clip.id
    assert qt_app.selection.kind == "clip"


def test_drag_selection_starting_inside_clip_selects_that_clip(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    character = qt_app.document.characters[0]
    clip = qt_app.document.assign_character_to_range(6, 11, character.id)  # "world"

    cursor = editor.textCursor()
    cursor.setPosition(6)
    cursor.setPosition(9, QTextCursor.MoveMode.KeepAnchor)
    editor.setTextCursor(cursor)

    assert qt_app.selection.selected_clip_id == clip.id
    assert qt_app.selection.kind == "clip"


def test_caret_in_plain_text_with_no_covering_clip_yields_none(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")

    cursor = editor.textCursor()
    cursor.setPosition(2)
    editor.setTextCursor(cursor)

    assert qt_app.selection.kind == "none"


def test_drag_selection_over_plain_text_yields_exact_range(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")

    cursor = editor.textCursor()
    cursor.setPosition(0)
    cursor.setPosition(5, QTextCursor.MoveMode.KeepAnchor)
    editor.setTextCursor(cursor)

    assert qt_app.selection.kind == "range"
    assert qt_app.selection.selected_range == (0, 5)


def test_clicking_timeline_clip_block_moves_transcript_cursor_to_clip_range(qt_app, qtbot):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    character = qt_app.document.characters[0]
    clip = qt_app.document.assign_character_to_range(6, 11, character.id)  # "world"
    qt_app.refresh_timeline()

    view = qt_app.timeline_dock.timeline_view
    block = _clip_block_items(qt_app)[0]
    assert block.clip_id == clip.id
    pos = view.mapFromScene(block.mapToScene(5, 5))
    qtbot.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=pos)

    cursor = editor.textCursor()
    expected_start, expected_end = qt_app.document.clip_extent(clip.id)
    assert cursor.selectionStart() == expected_start
    assert cursor.selectionEnd() == expected_end
