"""Tests for kokoro_gui/qt/docks/timeline_dock.py's TimelineDock, wired into
the running app - qt_app fixture. Reuses test_transcript_editor.py's
editor-driving helpers to confirm the refresh mechanism end-to-end (via the
real editor, not just calling Document methods directly)."""
from PySide6.QtCore import QMimeData, Qt
from PySide6.QtGui import QTextCursor

from kokoro_gui.qt.timeline_view import ClipBlockItem


def _editor(qt_app):
    return qt_app.generation_dock.text_entry


def _set_text_via_real_edit(editor, text):
    cursor = editor.textCursor()
    cursor.select(QTextCursor.SelectionType.Document)
    cursor.insertText(text)


def _mime_with_character(text, character_id):
    mime = QMimeData()
    mime.setText(text)
    mime.setData("application/x-kokorogui-character-id", character_id.encode("utf-8"))
    return mime


def _clip_block_items(dock):
    return [item for item in dock.timeline_view._scene.items() if isinstance(item, ClipBlockItem)]


def test_timeline_dock_constructed_and_registered(qt_app):
    assert qt_app.timeline_dock is not None
    assert qt_app.timeline_dock.objectName() == "dock_timeline"
    assert qt_app.dockWidgetArea(qt_app.timeline_dock) == Qt.DockWidgetArea.BottomDockWidgetArea


def test_dock_renders_clips_already_present_at_startup(qt_app):
    character = qt_app.document.characters[0]
    qt_app.document.text = "hello world"
    qt_app.document.assign_character_to_range(0, 5, character.id)

    qt_app.refresh_timeline()

    assert len(_clip_block_items(qt_app.timeline_dock)) == 1


def test_real_edit_triggers_timeline_refresh(qt_app, monkeypatch):
    calls = []
    monkeypatch.setattr(qt_app, "refresh_timeline", lambda: calls.append(True))
    editor = _editor(qt_app)

    _set_text_via_real_edit(editor, "hello world")

    assert calls


def test_assign_character_triggers_timeline_refresh_and_renders_block(qt_app, monkeypatch):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    character = qt_app.document.characters[0]

    calls = []
    monkeypatch.setattr(qt_app, "refresh_timeline", lambda: calls.append(True))
    cursor = editor.textCursor()
    cursor.setPosition(0)
    cursor.setPosition(5, QTextCursor.MoveMode.KeepAnchor)
    editor.setTextCursor(cursor)

    editor._assign_character(character.id)

    assert calls
    # refresh_timeline is monkeypatched to a no-op above, so drive a real
    # refresh now to confirm the underlying clip is actually there.
    qt_app.timeline_dock.refresh()
    assert len(_clip_block_items(qt_app.timeline_dock)) == 1


def test_paste_with_split_triggers_timeline_refresh(qt_app, monkeypatch):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    character = qt_app.document.characters[0]
    qt_app.settings["character_fx_paste_splits"] = True

    calls = []
    monkeypatch.setattr(qt_app, "refresh_timeline", lambda: calls.append(True))
    cursor = editor.textCursor()
    cursor.setPosition(11)
    editor.setTextCursor(cursor)

    editor.insertFromMimeData(_mime_with_character(" PASTED", character.id))

    assert calls


def test_paste_without_split_still_refreshes_via_text_edit_path(qt_app, monkeypatch):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    character = qt_app.document.characters[0]
    qt_app.settings["character_fx_paste_splits"] = False

    calls = []
    monkeypatch.setattr(qt_app, "refresh_timeline", lambda: calls.append(True))
    cursor = editor.textCursor()
    cursor.setPosition(11)
    editor.setTextCursor(cursor)

    editor.insertFromMimeData(_mime_with_character(" PASTED", character.id))

    # No clip-metadata mutation happens (splits disabled), but the plain
    # text insertion still flows through _on_contents_change, which already
    # calls refresh_timeline() unconditionally.
    assert calls
