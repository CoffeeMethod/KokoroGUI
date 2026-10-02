"""GUI-level tests for item 4 ("Undo/redo") - full qt_app fixture, covering
the app's undo()/redo() methods, its first menu bar (Edit > Undo/Redo), and
the transcript editor pushing commands instead of mutating the document
directly."""
from PySide6.QtGui import QKeySequence, QTextCursor

from kokoro_gui.qt.timeline_view import ClipBlockItem


def _editor(qt_app):
    return qt_app.editor


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
    # One bulk edit, not two adjacent inserts - Qt's native undo merges
    # adjacent same-position insertions into a single command (verified
    # directly against QTextDocument), so this exercises exactly one
    # native-stack undo, cleanly.
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    assert qt_app.document.text == "hello world"

    qt_app.undo()

    assert qt_app.document.text == ""
    assert editor.toPlainText() == ""


def test_reloading_the_editor_text_forgets_the_native_steps_it_wiped(qt_app):
    """setPlainText clears Qt's own undo history; the coordinator's log
    drops those steps too, so Ctrl+Z never lands on a step that is gone."""
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    assert editor.undo_coordinator.can_undo()

    editor.load_text(qt_app.document.text)

    assert not editor.undo_coordinator.can_undo()


def test_undo_then_redo_reapplies_the_typed_text(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")

    qt_app.undo()
    qt_app.redo()

    assert qt_app.document.text == "hello world"
    assert editor.toPlainText() == "hello world"


def test_typing_then_assigning_then_undo_twice_reverts_assignment_then_typing(qt_app):
    """The coordinated-dual-stack behavior this whole rebuild grilled for:
    a native (typing) edit followed by a custom-stack (character
    assignment) action undoes in the right order - the more recent action
    (the assignment, on the custom stack) first, then the older one (the
    typing, on the native stack) - regardless of which stack each came
    from."""
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    character = qt_app.document.characters[0]
    cursor = editor.textCursor()
    cursor.setPosition(6)
    cursor.setPosition(11, QTextCursor.MoveMode.KeepAnchor)
    editor.setTextCursor(cursor)
    editor._assign_character(character.id)
    assert len(qt_app.document.clips) == 1

    qt_app.undo()
    assert qt_app.document.clips == []
    assert qt_app.document.text == "hello world"  # typing survives this first undo

    qt_app.undo()
    assert qt_app.document.text == ""


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
    assert qt_app.document.clip_extent(clip.id) == (6, 11)
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
