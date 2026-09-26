"""Tests for kokoro_gui/qt/transcript_editor.py's TranscriptEditor - a
`QTextEdit` synced to a `kokoro_gui.daw.models.Document`'s run list
(Claude/PLAN_text_editor_redesign.md). `ClipHighlighter` replaces the
retired `CharacterFxHighlighter`'s two-pass reconciliation with a single
pass over `app.document.runs`, triggered via `TranscriptEditor.rehighlight()`."""
import pytest
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


# ---------------------------------------------------------------------------
# Imported recording text (phase 5 P3): cut, paste, drag, undo, visuals
# ---------------------------------------------------------------------------

RATE = 16000


def _recording(qt_app, tmp_path, rows, name="talk.wav"):
    """Imports a 10 s file into the project and appends one recording clip
    per row of `(word, start_s, end_s)`, as File > Import Audio's commit
    does. Returns `(source, clip_ids)`."""
    import numpy as np
    import soundfile as sf

    from kokoro_gui.daw import imported
    from kokoro_gui.daw.undo import ImportRecordingCommand
    from kokoro_gui.qt import project as project_io

    path = str(tmp_path / name)
    sf.write(path, np.linspace(-0.2, 0.2, RATE * 10).astype(np.float32), RATE)
    stored = project_io.import_audio_file(path, qt_app.project_dir)
    source, entry = imported.source_entry(stored)
    host = qt_app.document.characters[0]
    command = ImportRecordingCommand(
        [dict(zip(("text", "words"), imported.run_from_asr_words(words, source)), character_id=host.id)
         for words in rows], {source: entry})
    qt_app.document.undo_stack.push(command)
    qt_app.editor.load_text(qt_app.document.text)
    return source, command.clip_ids


def _select(editor, start, end):
    # The caret goes in first, as a click would: selecting straight into a
    # clip that isn't selected yet selects the whole clip.
    _caret(editor, start)
    cursor = editor.textCursor()
    cursor.setPosition(start)
    cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
    editor.setTextCursor(cursor)


def _caret(editor, position):
    cursor = editor.textCursor()
    cursor.setPosition(position)
    editor.setTextCursor(cursor)


def _cursor_at(editor, position):
    cursor = QTextCursor(editor.document())
    cursor.setPosition(position)
    return cursor


def _format_at(editor, position):
    from PySide6.QtGui import QTextCharFormat

    block = editor.document().findBlock(position)
    offset = position - block.position()
    formats = block.layout().formats()
    for fmt_range in formats:
        if fmt_range.start <= offset < fmt_range.start + fmt_range.length:
            return QTextCharFormat(fmt_range.format)
    return None


def _word_times(document, start):
    from kokoro_gui.daw import imported

    for clip in document.clips:
        for w_start, _w_end, _source, start_s, end_s in imported.clip_words(document, clip):
            if w_start == start:
                return [start_s, end_s]
    return None


def _words_mime(text, words, sources):
    import json

    from kokoro_gui.daw import imported

    mime = QMimeData()
    mime.setText(text)
    mime.setData(imported.WORDS_MIME_TYPE, json.dumps({"words": words, "sources": sources}).encode("utf-8"))
    return mime


HELLO = [("Hello", 0.0, 0.4), ("there", 0.5, 0.9), ("friend.", 1.0, 1.5)]
SECOND = [("Second", 2.0, 2.5), ("line.", 2.5, 3.0)]


def test_cut_a_word_and_paste_it_two_paragraphs_later_keeps_its_audio(qt_app, tmp_path, qtbot):
    from kokoro_gui.daw import imported

    document, editor = qt_app.document, qt_app.editor
    _source, (first_id, second_id) = _recording(qt_app, tmp_path, [HELLO, SECOND])
    _caret(editor, len(document.text))
    editor.textCursor().insertText("\n\nNotes: ")
    there = document.text.index("there")
    times = _word_times(document, there)
    assert times == [0.45, 0.95]  # the pauses either side split at their midpoints

    _select(editor, there, there + 5)
    qtbot.keyClick(editor, Qt.Key.Key_X, Qt.KeyboardModifier.ControlModifier)
    assert document.text.startswith("Hello  friend.")
    assert [s.range for s in document.get_clip(first_id).segments] == [[0.0, 0.45], [0.95, 1.5]]

    _caret(editor, len(document.text))
    qtbot.keyClick(editor, Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier)
    assert document.text.endswith("Notes: there")
    pasted = document.clip_covering(len(document.text) - 1)
    assert imported.is_recording_clip(pasted) and pasted.id not in (first_id, second_id)
    assert [s.range for s in pasted.segments] == [times]
    assert pasted.character_id == document.characters[0].id

    # One undo takes the paste away (text and clip), the next gives the
    # word back where it was, with its audio.
    qt_app.undo()
    assert document.text.endswith("Notes: ") and document.get_clip(pasted.id) is None
    qt_app.undo()
    assert document.text.startswith("Hello there friend.") and editor.toPlainText() == document.text
    assert [s.range for s in document.get_clip(first_id).segments] == [[0.0, 1.5]]

    qt_app.redo()
    qt_app.redo()
    assert document.text.endswith("Notes: there") and editor.toPlainText() == document.text
    assert [s.range for s in document.clip_covering(len(document.text) - 1).segments] == [times]


def test_paste_timed_text_whose_recording_is_gone_lands_untimed(qt_app, tmp_path):
    document, editor = qt_app.document, qt_app.editor
    _set_text_via_real_edit(editor, "Notes: ")
    source = "0123456789abcdef"
    gone = str(tmp_path / f"{source}.wav")
    _caret(editor, 7)

    editor.insertFromMimeData(_words_mime("there", [[0, 5, source, 1.0, 1.5]], {source: {"path": gone}}))

    assert document.text == "Notes: there"
    assert document.clips == [] and document.sources == {}
    assert "without timing" in qt_app.transport_dock.status_text()


def test_paste_from_another_project_imports_its_recording(qt_app, tmp_path):
    import os

    import numpy as np
    import soundfile as sf

    from kokoro_gui.daw import imported
    from kokoro_gui.qt import project as project_io

    other_dir, _other_id = project_io.create_project_dir()
    wav = str(tmp_path / "elsewhere.wav")
    sf.write(wav, np.full(RATE * 3, 0.1, dtype=np.float32), RATE)
    source, entry = imported.source_entry(project_io.import_audio_file(wav, other_dir))
    document, editor = qt_app.document, qt_app.editor
    _set_text_via_real_edit(editor, "Notes: ")
    _caret(editor, 7)

    editor.insertFromMimeData(_words_mime("there", [[0, 5, source, 1.0, 1.5]], {source: entry}))

    local = document.source_path(source)
    assert local.startswith(os.path.join(qt_app.project_dir, "audio", "imported")) and os.path.isfile(local)
    clip = document.clip_covering(8)
    assert [s.range for s in clip.segments] == [[1.0, 1.5]] and clip.segments[0].audio_path == local

    qt_app.undo()
    assert document.text == "Notes: " and document.sources == {}


def test_paste_from_another_project_relinks_a_recording_missing_here(qt_app, tmp_path):
    import numpy as np
    import soundfile as sf

    from kokoro_gui.daw import imported
    from kokoro_gui.qt import project as project_io

    other_dir, _other_id = project_io.create_project_dir()
    wav = str(tmp_path / "elsewhere.wav")
    sf.write(wav, np.full(RATE * 3, 0.1, dtype=np.float32), RATE)
    source, entry = imported.source_entry(project_io.import_audio_file(wav, other_dir))
    document, editor = qt_app.document, qt_app.editor
    # This project knew the recording, but its file was missing on open.
    document.settings["sources"] = {source: {"path": None, "sample_rate": RATE, "duration_s": 3.0}}
    _set_text_via_real_edit(editor, "Notes: ")
    _caret(editor, 7)

    editor.insertFromMimeData(_words_mime("there", [[0, 5, source, 1.0, 1.5]], {source: entry}))

    local = document.source_path(source)
    assert local is not None and local.startswith(qt_app.project_dir)
    clip = document.clip_covering(8)
    assert clip is not None and [s.range for s in clip.segments] == [[1.0, 1.5]]

    qt_app.undo()
    assert document.text == "Notes: " and document.source_path(source) is None


def test_paste_of_malformed_timed_words_keeps_the_good_ones(qt_app, tmp_path):
    document, editor = qt_app.document, qt_app.editor
    source, _ids = _recording(qt_app, tmp_path, [HELLO])
    entry = dict(document.sources[source])
    _caret(editor, len(document.text))
    editor.textCursor().insertText("\n\nNotes: ")
    at = len(document.text)
    words = [[0, 3, [source], 1.0, 1.2], [0, 3, {"a": 1}, 1.0, 1.2], "junk", [4, 7, source, 1.3, 1.5],
             [0, 3], None, [4, 7, source, "x", 1.5]]

    editor.insertFromMimeData(_words_mime("one two", words, {source: entry, "bad": "not a dict"}))

    assert document.text.endswith("Notes: one two")
    clip = document.clip_covering(at + 4)
    assert clip is not None and [s.range for s in clip.segments] == [[1.3, 1.5]]
    assert set(document.sources) == {source}


def test_paste_of_only_malformed_timed_words_lands_untimed(qt_app):
    document, editor = qt_app.document, qt_app.editor
    _set_text_via_real_edit(editor, "Notes: ")
    _caret(editor, 7)

    editor.insertFromMimeData(_words_mime("there", [[0, 5, ["x"], 1.0, 1.5]], {}))

    assert document.text == "Notes: there" and document.clips == []


def test_deleting_imported_words_undoes_with_their_timing(qt_app, tmp_path, qtbot):
    import copy

    document, editor = qt_app.document, qt_app.editor
    _source, (clip_id,) = _recording(qt_app, tmp_path, [HELLO])
    before = copy.deepcopy(document.runs)
    there = document.text.index("there")

    _select(editor, there, there + 6)
    qtbot.keyClick(editor, Qt.Key.Key_Delete)
    assert document.text == "Hello friend."
    assert [s.range for s in document.get_clip(clip_id).segments] == [[0.0, 0.45], [0.95, 1.5]]

    qt_app.undo()
    assert document.runs == before and editor.toPlainText() == document.text
    qt_app.redo()
    assert document.text == "Hello friend." and editor.toPlainText() == document.text


def test_typing_inside_a_recording_splits_it_and_undo_joins_it_back(qt_app, tmp_path, qtbot):
    import copy

    from kokoro_gui.daw import imported
    from kokoro_gui.qt import theme

    document, editor = qt_app.document, qt_app.editor
    _source, (clip_id,) = _recording(qt_app, tmp_path, [HELLO])
    before = copy.deepcopy(document.runs)
    at = document.text.index(" friend")
    _caret(editor, at)
    editor.setFocus()

    qtbot.keyClicks(editor, " my")
    assert document.text == "Hello there my friend."
    typed = document._run_covering(at + 1)
    assert typed.clip_id is None and typed.text == " my"
    assert len([c for c in document.clips if imported.is_recording_clip(c)]) == 2
    # The typed text has no character: no tint, greyed.
    fmt = _format_at(editor, at + 2)
    assert fmt.background().style() == Qt.BrushStyle.NoBrush
    assert fmt.foreground().color().name() == theme.current().text_muted

    # Qt coalesces "my" into the space's undo step, so one undo takes back
    # the typing and joins the clip again, words and all.
    qt_app.undo()
    assert document.runs == before and [c.id for c in document.clips] == [clip_id]
    assert editor.toPlainText() == document.text

    qt_app.redo()
    assert document.text == "Hello there my friend." and editor.toPlainText() == document.text
    assert document._run_covering(at + 1).text == " my"


def _edit_state(qt_app):
    """The document's text, the editor's, and its runs with clip ids
    numbered in text order (a redone split makes a clip with a new id)."""
    document = qt_app.document
    names: dict = {}
    runs = []
    for run in document.runs:
        name = names.setdefault(run.clip_id, len(names)) if run.clip_id else None
        runs.append((run.text, name, run.kind, [list(w) for w in run.words]))
    clips = sorted((names[c.id], [s.range for s in c.segments]) for c in document.clips if c.id in names)
    return document.text, qt_app.editor.toPlainText(), runs, clips


@pytest.mark.parametrize("typed", [" x", "x"])
def test_typing_then_backspacing_inside_a_recording_undoes_and_redoes_step_by_step(qt_app, tmp_path, qtbot, typed):
    """Type inside a recording (it splits), Backspace it all away (the
    last one joins the halves again), then undo and redo all of it, twice:
    each step gives back the same text, runs, words and clips, and no undo
    or redo pushes a command of its own."""
    document, editor = qt_app.document, qt_app.editor
    _recording(qt_app, tmp_path, [HELLO])
    states = [_edit_state(qt_app)]
    at = document.text.index(" friend")
    _caret(editor, at)
    editor.setFocus()

    qtbot.keyClicks(editor, typed)
    states.append(_edit_state(qt_app))
    assert document.text == f"Hello there{typed} friend." and len(document.clips) == 2
    for _char in typed:
        qtbot.keyClick(editor, Qt.Key.Key_Backspace)
        states.append(_edit_state(qt_app))
    assert document.text == "Hello there friend." and len(document.clips) == 1
    stack = document.undo_stack
    commands = len(stack._undo) + len(stack._redo)

    for _round in range(2):
        for expected in reversed(states[:-1]):
            qt_app.undo()
            assert _edit_state(qt_app) == expected
            assert len(stack._undo) + len(stack._redo) == commands
        for expected in states[1:]:
            qt_app.redo()
            assert _edit_state(qt_app) == expected
            assert len(stack._undo) + len(stack._redo) == commands


def test_a_native_undo_or_redo_never_pushes_a_command(qt_app, qtbot, monkeypatch):
    """Even when the text a native step replays reads as an edit of
    imported text, the replay only syncs the document."""
    document, editor = qt_app.document, qt_app.editor
    _caret(editor, len(document.text))
    editor.setFocus()
    qtbot.keyClicks(editor, "Notes")
    text = document.text
    stack = document.undo_stack
    commands = len(stack._undo) + len(stack._redo)
    monkeypatch.setattr(type(document), "edit_touches_imported", lambda self, position, removed: True)

    qt_app.undo()
    assert editor.toPlainText() == document.text and "Notes" not in document.text
    assert len(stack._undo) + len(stack._redo) == commands
    qt_app.redo()
    assert editor.toPlainText() == document.text == text
    assert len(stack._undo) + len(stack._redo) == commands
    assert not editor.undo_coordinator.replaying


def test_imported_words_get_a_faint_underline(qt_app, tmp_path):
    from PySide6.QtGui import QTextCharFormat

    document, editor = qt_app.document, qt_app.editor
    _set_text_via_real_edit(editor, "Plain text.")
    _recording(qt_app, tmp_path, [HELLO])
    there = document.text.index("there")

    assert _format_at(editor, there + 1).underlineStyle() == QTextCharFormat.UnderlineStyle.SingleUnderline
    plain = _format_at(editor, 2)
    assert plain is None or plain.underlineStyle() == QTextCharFormat.UnderlineStyle.NoUnderline


def test_dragging_imported_words_inside_the_editor_moves_their_audio(qt_app, tmp_path, monkeypatch):
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QDropEvent

    document, editor = qt_app.document, qt_app.editor
    _source, (first_id, _second_id) = _recording(qt_app, tmp_path, [HELLO, SECOND])
    there = document.text.index("there")
    times = _word_times(document, there)
    _select(editor, there, there + 5)
    mime = editor.createMimeDataFromSelection()
    end = len(document.text)
    monkeypatch.setattr(editor, "_is_own_drag", lambda event: True)
    monkeypatch.setattr(editor, "cursorForPosition", lambda point: _cursor_at(editor, end))
    event = QDropEvent(QPointF(5, 5), Qt.DropAction.MoveAction | Qt.DropAction.CopyAction, mime,
                       Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    event.setDropAction(Qt.DropAction.MoveAction)

    editor.dropEvent(event)

    assert document.text == "Hello  friend.\n\nSecond line.there"
    assert event.dropAction() == Qt.DropAction.CopyAction
    moved = document.clip_covering(len(document.text) - 1)
    assert [s.range for s in moved.segments][-1] == times
    assert [s.range for s in document.get_clip(first_id).segments] == [[0.0, 0.45], [0.95, 1.5]]
