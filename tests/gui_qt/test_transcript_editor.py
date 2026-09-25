"""Tests for kokoro_gui/qt/transcript_editor.py's TranscriptEditor - a
`QTextEdit` synced to a `kokoro_gui.daw.models.Document`'s run list
(Claude/PLAN_text_editor_redesign.md). `ClipHighlighter` replaces the
retired `CharacterFxHighlighter`'s two-pass reconciliation with a single
pass over `app.document.runs`, triggered via `TranscriptEditor.rehighlight()`."""
from PySide6.QtCore import QMimeData, Qt
from PySide6.QtGui import QFocusEvent, QTextCursor

from kokoro_gui.daw.models import Character


def _editor(qt_app):
    return qt_app.editor


def _set_text_via_real_edit(editor, text):
    """Drives a real user-style edit (goes through contentsChange), unlike
    load_text/setPlainText which is the deliberately-suppressed path."""
    cursor = editor.textCursor()
    cursor.select(QTextCursor.SelectionType.Document)
    cursor.insertText(text)


def _type_line_and_press_enter(qtbot, editor, line_text):
    """Simulates real keystrokes (unlike _set_text_via_real_edit's bulk
    cursor.insertText) so TranscriptEditor.keyPressEvent's Enter-triggered
    [Speaker:FX]: shorthand recognition (TE6's "on completing the line"
    grill answer) actually fires."""
    editor.setFocus()
    qtbot.keyClicks(editor, line_text)
    qtbot.keyClick(editor, Qt.Key.Key_Return)


def _highlight_color_at(editor, position):
    """ClipHighlighter's setFormat() calls land in the block's QTextLayout
    format overlay, not in QTextCursor.charFormat() (that reads the
    document's "real" character formatting, a separate store the
    highlighter deliberately never touches - see transcript_editor.py's
    module docstring for why) and, in this PySide6 version, not reliably in
    QTextBlock.textFormats() either (it was observed to return one merged,
    un-highlighted range) - so reading a highlighter's actual output back
    for assertions means walking QTextBlock.layout().formats() instead,
    which does reflect it."""
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


def test_real_edit_extends_existing_clip_run(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    character_id = qt_app.document.characters[0].id
    clip = qt_app.document.assign_character_to_range(6, 11, character_id)  # "world"

    cursor = editor.textCursor()
    cursor.setPosition(0)
    cursor.insertText("XYZ ")  # insert 4 chars before "hello world"

    assert qt_app.document.text == "XYZ hello world"
    assert qt_app.document.clip_extent(clip.id) == (10, 15)


def test_load_text_does_not_call_replace_text(qt_app, monkeypatch):
    editor = _editor(qt_app)
    calls = []
    monkeypatch.setattr(
        qt_app.document, "replace_text",
        lambda *a, **k: calls.append(a) or []
    )
    editor.load_text("some new text")
    assert calls == []
    assert editor.toPlainText() == "some new text"


def test_load_text_does_not_disturb_clip_tagging(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    character_id = qt_app.document.characters[0].id
    clip = qt_app.document.assign_character_to_range(6, 11, character_id)

    editor.load_text("hello world")  # same text, reloaded programmatically

    assert qt_app.document.clip_extent(clip.id) == (6, 11)


# ---------------------------------------------------------------------------
# Highlighting
# ---------------------------------------------------------------------------

def test_clip_range_is_highlighted_with_character_color(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    character = qt_app.document.characters[0]
    character.highlight_color = "#123456"
    qt_app.document.assign_character_to_range(6, 11, character.id)
    editor.rehighlight()

    assert _highlight_color_at(editor, 7) == "#123456"


def test_text_outside_any_clip_is_not_highlighted(qt_app):
    editor = _editor(qt_app)
    _set_text_via_real_edit(editor, "hello world")
    character = qt_app.document.characters[0]
    qt_app.document.assign_character_to_range(6, 11, character.id)
    editor.rehighlight()

    assert _highlight_color_at(editor, 1) is None  # inside "hello", not covered by any clip


def test_shorthand_line_is_recognized_and_highlighted_on_enter(qtbot, qt_app):
    editor = _editor(qt_app)
    alice = Character.from_preset_dict("Alice", {"voice": "af_bella"}, highlight_color="#abcdef")
    qt_app.document.characters.append(alice)

    _type_line_and_press_enter(qtbot, editor, "[Alice]: hello there")

    assert _highlight_color_at(editor, 10) == "#abcdef"  # inside "hello"
    clip = qt_app.document.clip_covering(10)
    assert clip is not None
    assert clip.character_id == alice.id


def test_shorthand_line_with_no_matching_character_is_not_recognized(qtbot, qt_app):
    editor = _editor(qt_app)

    _type_line_and_press_enter(qtbot, editor, "[NobodyHome]: hello there")

    assert _highlight_color_at(editor, 15) is None
    assert qt_app.document.clip_covering(15) is None


def test_shorthand_line_not_yet_completed_is_not_recognized(qt_app):
    """No Enter pressed and no focus lost - matches the recommended "on
    completing the line" behavior's other half: a still-being-typed line
    stays plain text."""
    editor = _editor(qt_app)
    alice = Character.from_preset_dict("Alice", {"voice": "af_bella"})
    qt_app.document.characters.append(alice)

    _set_text_via_real_edit(editor, "[Alice]: hello the")

    assert qt_app.document.clip_covering(10) is None


def test_shorthand_line_is_recognized_on_focus_out_without_enter(qt_app):
    """Catches a last line with no trailing Enter, per the grill answer."""
    editor = _editor(qt_app)
    alice = Character.from_preset_dict("Alice", {"voice": "af_bella"})
    qt_app.document.characters.append(alice)
    _set_text_via_real_edit(editor, "[Alice]: hello there")

    editor.focusOutEvent(QFocusEvent(QFocusEvent.Type.FocusOut, Qt.FocusReason.OtherFocusReason))

    clip = qt_app.document.clip_covering(10)
    assert clip is not None
    assert clip.character_id == alice.id


def test_already_tagged_line_is_not_reassigned_on_revisit(qtbot, qt_app):
    editor = _editor(qt_app)
    alice = Character.from_preset_dict("Alice", {"voice": "af_bella"})
    qt_app.document.characters.append(alice)
    _type_line_and_press_enter(qtbot, editor, "[Alice]: hello there")
    first_clip = qt_app.document.clip_covering(10)

    # Revisiting (focus-out again, cursor still on/near that already-tagged
    # line) must not mint a second clip for the same text.
    editor.focusOutEvent(QFocusEvent(QFocusEvent.Type.FocusOut, Qt.FocusReason.OtherFocusReason))

    assert qt_app.document.clip_covering(10).id == first_clip.id
    assert len(qt_app.document.clips) == 1


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
    character.highlight_color = "#654321"
    cursor = editor.textCursor()
    cursor.setPosition(0)
    cursor.setPosition(5, QTextCursor.MoveMode.KeepAnchor)
    editor.setTextCursor(cursor)

    editor._assign_character(character.id)

    clip = qt_app.document.clip_covering(0)
    assert clip is not None
    assert clip.character_id == character.id
    assert qt_app.document.clip_extent(clip.id) == (0, 5)
    assert _highlight_color_at(editor, 0) == "#654321"


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



def test_playing_word_is_a_second_extra_selection(qt_app):
    editor = qt_app.editor
    editor.load_text("hello brave world")
    editor.set_playing_word((6, 11))
    selections = editor.extraSelections()
    assert (selections[-1].cursor.selectionStart(), selections[-1].cursor.selectionEnd()) == (6, 11)
    editor.set_playing_word(None)
    assert editor.playing_word() is None


def test_word_at_maps_the_playhead_through_the_lexicon(qt_app):
    from kokoro_gui.daw.models import Segment

    qt_app.document.text = "Dr Who arrives."
    clip = qt_app.document.assign_character_to_range(0, 15, qt_app.document.characters[0].id)
    qt_app.settings["lexicon"] = {"Dr": "Doctor"}
    clip.segments = [Segment(text="Doctor Who arrives.", audio_path="a.wav", duration=3.0,
                             words=[["Doctor", 0.0, 1.0], ["Who", 1.0, 2.0], ["arrives.", 2.0, 3.0]])]
    placed = qt_app.build_arrangement().by_clip_id()[clip.id]

    assert qt_app.word_at(placed, placed.start_s + 1.5) == (3, 6)
    assert qt_app.word_at(placed, placed.start_s + 0.5) == (0, 2)


def test_ctrl_click_seeks_to_the_word(qt_app, qtbot):
    from PySide6.QtCore import Qt

    from kokoro_gui.daw.models import Segment

    qt_app.document.text = "hello brave world"
    clip = qt_app.document.assign_character_to_range(0, 17, qt_app.document.characters[0].id)
    clip.segments = [Segment(text="hello brave world", audio_path="a.wav", duration=3.0,
                             words=[["hello", 0.0, 1.0], ["brave", 1.2, 2.0], ["world", 2.1, 3.0]])]
    qt_app._rebuild_transport_schedule()  # seek goes by what the transport plays
    seeks = []
    qt_app.transport.seek = seeks.append
    placed = qt_app.current_arrangement().by_clip_id()[clip.id]

    assert qt_app.seek_to_offset(8) is True
    assert seeks == [placed.start_s + 1.2]
    assert qt_app.seek_to_offset(1000) is False


def test_gutter_tooltip_shows_the_source_text(qt_app):
    qt_app.document.text = "Hello."
    clip = qt_app.document.assign_character_to_range(0, 6, qt_app.document.characters[0].id)
    clip.source_text = "Bonjour."
    qt_app.editor.load_text(qt_app.document.text)
    gutter = qt_app.editor._gutter
    from PySide6.QtGui import QPaintEvent
    gutter.paintEvent(QPaintEvent(gutter.rect()))
    rect, _start, _end = gutter._label_rects[0]
    assert gutter.tooltip_at(rect.center()) == "Source: Bonjour."
