"""Tests for kokoro_gui/qt/transcript_editor.py's TranscriptEditor and
CharacterFxHighlighter - Workstream 2 of the DAW-for-text redesign."""
from PySide6.QtCore import QMimeData
from PySide6.QtGui import QTextCursor

from kokoro_gui.daw.models import Character


def _editor(qt_app):
    return qt_app.generation_dock.text_entry


def _set_text_via_real_edit(editor, text):
    """Drives a real user-style edit (goes through contentsChange), unlike
    load_text/setPlainText which is the deliberately-suppressed path."""
    cursor = editor.textCursor()
    cursor.select(QTextCursor.SelectionType.Document)
    cursor.insertText(text)


def _highlight_color_at(editor, position):
    """QSyntaxHighlighter's setFormat() calls land in the block's QTextLayout
    format overlay, not in QTextCursor.charFormat() (that reads the
    document's "real" character formatting, a separate store) and, in this
    PySide6 version, not reliably in QTextBlock.textFormats() either (it was
    observed to return one merged, un-highlighted range) - so reading a
    highlighter's actual output back for assertions means walking
    QTextBlock.layout().formats() instead, which does reflect it."""
    block = editor.document().findBlock(position)
    offset_in_block = position - block.position()
    for fmt_range in block.layout().formats():
        if fmt_range.start <= offset_in_block < fmt_range.start + fmt_range.length:
            return fmt_range.format.background().color().name()
    return None


# ---------------------------------------------------------------------------
# Document sync
# ---------------------------------------------------------------------------

def test_real_edit_updates_document_text(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    assert qt_app.document.text == "hello world"


def test_real_edit_shifts_existing_clip_offsets(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    character_id = qt_app.document.characters[0].id
    clip = qt_app.document.assign_character_to_range(6, 11, character_id)  # "world"

    cursor = editor.textCursor()
    cursor.setPosition(0)
    cursor.insertText("XYZ ")  # insert 4 chars before "hello world"

    assert qt_app.document.text == "XYZ hello world"
    assert clip.start_offset == 10
    assert clip.end_offset == 15


def test_load_text_does_not_call_apply_text_change(qt_app, monkeypatch):
    editor = _editor(qt_app)
    calls = []
    monkeypatch.setattr(
        qt_app.document, "apply_text_change",
        lambda *a, **k: calls.append(a) or []
    )
    editor.load_text("some new text")
    assert calls == []
    assert editor.toPlainText() == "some new text"


def test_load_text_does_not_disturb_clip_offsets(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    character_id = qt_app.document.characters[0].id
    clip = qt_app.document.assign_character_to_range(6, 11, character_id)

    editor.load_text("hello world")  # same text, reloaded programmatically

    assert clip.start_offset == 6
    assert clip.end_offset == 11


# ---------------------------------------------------------------------------
# Highlighting
# ---------------------------------------------------------------------------

def test_clip_range_is_highlighted_with_character_color(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    character = qt_app.document.characters[0]
    character.highlight_color = "#123456"
    qt_app.document.assign_character_to_range(6, 11, character.id)
    editor._highlighter.rehighlight()

    assert _highlight_color_at(editor, 7) == "#123456"


def test_text_outside_any_clip_is_not_highlighted(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    character = qt_app.document.characters[0]
    qt_app.document.assign_character_to_range(6, 11, character.id)
    editor._highlighter.rehighlight()

    assert _highlight_color_at(editor, 1) is None  # inside "hello", not covered by any clip


def test_inline_tag_matching_a_character_name_is_highlighted(qt_app):
    editor = _editor(qt_app)
    alice = Character.from_preset_dict("Alice", {"voice": "af_bella"}, highlight_color="#abcdef")
    qt_app.document.characters.append(alice)

    _set_text_via_real_edit(editor, "[Alice]: hello there")

    assert _highlight_color_at(editor, 10) == "#abcdef"  # inside "hello"


def test_inline_tag_with_no_matching_character_is_not_highlighted(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "[NobodyHome]: hello there")

    assert _highlight_color_at(editor, 15) is None


# ---------------------------------------------------------------------------
# Characters menu
# ---------------------------------------------------------------------------

def _characters_submenu(menu):
    return next(a.menu() for a in menu.actions() if a.menu() is not None and a.text() == "Characters")


def test_context_menu_lists_characters_and_disables_without_selection(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    editor.textCursor()  # no selection made

    menu = editor._build_context_menu()
    characters_menu = _characters_submenu(menu)
    names = [a.text() for a in characters_menu.actions()]
    assert names == [c.name for c in qt_app.document.characters]
    assert characters_menu.isEnabled() is False


def test_context_menu_characters_submenu_enabled_with_selection(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    cursor = editor.textCursor()
    cursor.setPosition(0)
    cursor.setPosition(5, QTextCursor.MoveMode.KeepAnchor)
    editor.setTextCursor(cursor)

    menu = editor._build_context_menu()
    characters_menu = _characters_submenu(menu)
    assert characters_menu.isEnabled() is True


def test_assign_character_directly_creates_clip_and_rehighlights(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    character = qt_app.document.characters[0]
    cursor = editor.textCursor()
    cursor.setPosition(0)
    cursor.setPosition(5, QTextCursor.MoveMode.KeepAnchor)
    editor.setTextCursor(cursor)

    editor._assign_character(character.id)

    clip = qt_app.document.clip_covering(0)
    assert clip is not None
    assert clip.character_id == character.id
    assert clip.start_offset == 0
    assert clip.end_offset == 5


# ---------------------------------------------------------------------------
# Copy/paste split-vs-inherit
# ---------------------------------------------------------------------------

def _mime_with_character(text, character_id):
    mime = QMimeData()
    mime.setText(text)
    mime.setData(
        "application/x-kokorogui-character-id",
        character_id.encode("utf-8"),
    )
    return mime


def test_paste_with_splits_enabled_creates_clip_for_source_character(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    character = qt_app.document.characters[0]
    qt_app.settings["character_fx_paste_splits"] = True

    cursor = editor.textCursor()
    cursor.setPosition(11)
    editor.setTextCursor(cursor)
    editor.insertFromMimeData(_mime_with_character(" PASTED", character.id))

    assert qt_app.document.text == "hello world PASTED"
    clip = qt_app.document.clip_covering(11)
    assert clip is not None
    assert clip.character_id == character.id


def test_paste_with_splits_disabled_does_not_create_clip(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    character = qt_app.document.characters[0]
    qt_app.settings["character_fx_paste_splits"] = False

    cursor = editor.textCursor()
    cursor.setPosition(11)
    editor.setTextCursor(cursor)
    editor.insertFromMimeData(_mime_with_character(" PASTED", character.id))

    assert qt_app.document.text == "hello world PASTED"
    assert qt_app.document.clip_covering(11) is None


def test_create_mime_data_from_selection_tags_source_character(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    character = qt_app.document.characters[0]
    qt_app.document.assign_character_to_range(6, 11, character.id)  # "world"

    cursor = editor.textCursor()
    cursor.setPosition(6)
    cursor.setPosition(11, QTextCursor.MoveMode.KeepAnchor)
    editor.setTextCursor(cursor)

    mime = editor.createMimeDataFromSelection()
    tagged_id = bytes(mime.data(editor.CHARACTER_ID_MIME_TYPE)).decode("utf-8")
    assert tagged_id == character.id
