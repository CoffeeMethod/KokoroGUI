"""GUI-level tests for item 4 ("Undo/redo") - full qt_app fixture, covering
the app's undo()/redo() methods, its first menu bar (Edit > Undo/Redo), and
the transcript editor pushing commands instead of mutating the document
directly."""
from PySide6.QtGui import QKeySequence, QTextCursor

from kokoro_gui.qt.timeline_view import ClipBlockItem


def _editor(qt_app):
    return qt_app.generation_dock.text_entry


def _set_text_via_real_edit(editor, text):
    cursor = editor.textCursor()
    cursor.select(QTextCursor.SelectionType.Document)
    cursor.insertText(text)


def _clip_block_items(qt_app):
    return [item for item in qt_app.timeline_dock.timeline_view._scene.items() if isinstance(item, ClipBlockItem)]


# ---------------------------------------------------------------------------
# Typing + undo resyncs both the document and the visible widget
# ---------------------------------------------------------------------------

def test_typing_then_undo_restores_previous_text_in_document_and_widget(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    assert qt_app.document.text == "hello world"

    cursor = editor.textCursor()
    cursor.setPosition(11)
    editor.setTextCursor(cursor)
    cursor.insertText("!")
    assert qt_app.document.text == "hello world!"

    qt_app.undo()

    assert qt_app.document.text == "hello world"
    assert editor.toPlainText() == "hello world"


def test_undo_then_redo_reapplies_the_typed_text(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    cursor = editor.textCursor()
    cursor.setPosition(11)
    editor.setTextCursor(cursor)
    cursor.insertText("!")

    qt_app.undo()
    qt_app.redo()

    assert qt_app.document.text == "hello world!"
    assert editor.toPlainText() == "hello world!"


# ---------------------------------------------------------------------------
# Characters menu + undo
# ---------------------------------------------------------------------------

def test_assign_character_then_undo_removes_the_clip(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    character = qt_app.document.characters[0]

    editor._assign_character(character.id)
    assert len(qt_app.document.clips) == 0  # cursor had no selection - no-op

    cursor = editor.textCursor()
    cursor.setPosition(6)
    cursor.setPosition(11, QTextCursor.MoveMode.KeepAnchor)
    editor.setTextCursor(cursor)
    editor._assign_character(character.id)
    assert len(qt_app.document.clips) == 1

    qt_app.undo()

    assert qt_app.document.clips == []
    assert _clip_block_items(qt_app) == []


def test_assign_character_undo_redo_restores_clip(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    character = qt_app.document.characters[0]
    cursor = editor.textCursor()
    cursor.setPosition(6)
    cursor.setPosition(11, QTextCursor.MoveMode.KeepAnchor)
    editor.setTextCursor(cursor)
    editor._assign_character(character.id)

    qt_app.undo()
    qt_app.redo()

    assert len(qt_app.document.clips) == 1
    clip = qt_app.document.clips[0]
    assert (clip.start_offset, clip.end_offset) == (6, 11)
    assert clip.character_id == character.id


# ---------------------------------------------------------------------------
# Menu bar
# ---------------------------------------------------------------------------

def test_menu_bar_has_edit_menu_with_undo_redo_actions(qt_app):
    menu_bar = qt_app.menuBar()
    assert menu_bar is not None

    menu_titles = [action.text() for action in menu_bar.actions()]
    assert any("Edit" in title for title in menu_titles)

    assert qt_app.undo_action.text() == "Undo"
    assert qt_app.redo_action.text() == "Redo"
    assert qt_app.undo_action.shortcut() == QKeySequence(QKeySequence.StandardKey.Undo)
    assert qt_app.redo_action.shortcut() == QKeySequence(QKeySequence.StandardKey.Redo)


def test_undo_action_trigger_calls_undo(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")

    qt_app.undo_action.trigger()

    assert qt_app.document.text == ""
    assert editor.toPlainText() == ""


def test_redo_action_trigger_calls_redo(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    qt_app.undo_action.trigger()

    qt_app.redo_action.trigger()

    assert qt_app.document.text == "hello world"
    assert editor.toPlainText() == "hello world"
