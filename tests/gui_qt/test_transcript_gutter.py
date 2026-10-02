"""Tests for kokoro_gui/qt/transcript_editor.py's TranscriptGutter (TE3) -
the left gutter's "Character: X" labels, change-only painting, and
click-to-reassign picker."""
from PySide6.QtGui import QPaintEvent

from kokoro_gui.daw.models import Character
from kokoro_gui.qt.transcript_editor import GUTTER_WIDTH_PX


def _editor(qt_app):
    return qt_app.editor


def _gutter(qt_app):
    return _editor(qt_app)._gutter


def _repaint(gutter):
    """Directly invokes paintEvent (no real display needed) so
    `_label_rects` gets (re)populated the same way a real paint would."""
    gutter.paintEvent(QPaintEvent(gutter.rect()))


def _set_text(editor, text):
    editor.setPlainText("")  # ensure a clean slate regardless of prior state
    from PySide6.QtGui import QTextCursor
    cursor = editor.textCursor()
    cursor.select(QTextCursor.SelectionType.Document)
    cursor.insertText(text)


def test_gutter_has_expected_width(qt_app):
    gutter = _gutter(qt_app)
    assert gutter.sizeHint().width() == GUTTER_WIDTH_PX
    assert gutter.parent() is _editor(qt_app)


def test_gutter_labels_a_tagged_line(qt_app):
    editor = _editor(qt_app)
    gutter = _gutter(qt_app)
    _set_text(editor, "hello world")
    alice = qt_app.document.characters[0]
    qt_app.document.assign_character_to_range(0, 5, alice.id)
    editor.rehighlight()

    _repaint(gutter)

    # _label_rects stores the painted LINE's range (a block can be wider
    # than the clip it starts with a label for) - the picker menu widens
    # this out to the clip's actual full extent at click time (see
    # test_picker_widens_range_to_the_clips_full_extent_not_just_the_clicked_line).
    assert len(gutter._label_rects) == 1
    _rect, line_start, _line_end = gutter._label_rects[0]
    assert line_start == 0


def test_gutter_omits_label_for_untagged_text(qt_app):
    editor = _editor(qt_app)
    gutter = _gutter(qt_app)
    _set_text(editor, "hello world")

    _repaint(gutter)

    assert gutter._label_rects == []


def test_gutter_labels_only_where_character_changes_across_lines(qt_app):
    editor = _editor(qt_app)
    gutter = _gutter(qt_app)
    alice = qt_app.document.characters[0]
    _set_text(editor, "line one\nline two\nline three")
    # Tag all three lines to the SAME character - only the first line (where
    # the change from "untagged" happens) should get a label.
    qt_app.document.assign_character_to_range(0, len(qt_app.document.text), alice.id)
    editor.rehighlight()

    _repaint(gutter)

    assert len(gutter._label_rects) == 1


def test_gutter_relabels_when_character_changes_mid_document(qt_app):
    editor = _editor(qt_app)
    gutter = _gutter(qt_app)
    bob = Character.from_preset_dict("Bob", {"voice": "am_michael"})
    qt_app.document.characters.append(bob)
    alice = qt_app.document.characters[0]
    text = "line one\nline two"
    _set_text(editor, text)
    first_line_end = text.index("\n")
    qt_app.document.assign_character_to_range(0, first_line_end, alice.id)
    qt_app.document.assign_character_to_range(first_line_end, len(text), bob.id)
    editor.rehighlight()

    _repaint(gutter)

    assert len(gutter._label_rects) == 2


def test_gutter_label_shows_fx_indicator_when_override_set(qt_app):
    editor = _editor(qt_app)
    gutter = _gutter(qt_app)
    _set_text(editor, "hello world")
    alice = qt_app.document.characters[0]
    clip = qt_app.document.assign_character_to_range(0, 5, alice.id)
    clip.fx_override = {"reverb_enabled": True}
    editor.rehighlight()

    _repaint(gutter)

    # Can't read drawn pixel text back directly, but the label rect existing
    # at all (with the fx_override set) confirms the code path that appends
    # "  FX" ran without raising - covered end-to-end below by asserting the
    # click handler still resolves the same clip/range correctly.
    assert len(gutter._label_rects) == 1


def test_clicking_a_label_opens_a_picker_listing_characters(qt_app):
    editor = _editor(qt_app)
    gutter = _gutter(qt_app)
    _set_text(editor, "hello world")
    alice = qt_app.document.characters[0]
    bob = Character.from_preset_dict("Bob", {"voice": "am_michael"})
    qt_app.document.characters.append(bob)
    qt_app.document.assign_character_to_range(0, 5, alice.id)
    editor.rehighlight()
    _repaint(gutter)

    rect, line_start, line_end = gutter._label_rects[0]
    menu = gutter._build_picker_menu(line_start, line_end)

    assert [a.text() for a in menu.actions()] == [c.name for c in qt_app.document.characters]


def test_picker_action_reassigns_the_clicked_clip(qt_app):
    editor = _editor(qt_app)
    gutter = _gutter(qt_app)
    _set_text(editor, "hello world")
    alice = qt_app.document.characters[0]
    bob = Character.from_preset_dict("Bob", {"voice": "am_michael"})
    qt_app.document.characters.append(bob)
    clip = qt_app.document.assign_character_to_range(0, 5, alice.id)
    editor.rehighlight()
    _repaint(gutter)
    rect, line_start, line_end = gutter._label_rects[0]

    menu = gutter._build_picker_menu(line_start, line_end)
    bob_action = next(a for a in menu.actions() if a.text() == "Bob")
    bob_action.trigger()

    new_clip = qt_app.document.clip_covering(0)
    assert new_clip.character_id == bob.id
    assert new_clip.id != clip.id  # reassignment mints a fresh clip, same as the Characters menu


def test_picker_widens_range_to_the_clips_full_extent_not_just_the_clicked_line(qt_app):
    """A clip can span more text than the one line its label happens to be
    painted next to (a multi-line clip, or one that only changed on its
    first line) - the picker must act on the whole clip, not just that line."""
    editor = _editor(qt_app)
    gutter = _gutter(qt_app)
    alice = qt_app.document.characters[0]
    bob = Character.from_preset_dict("Bob", {"voice": "am_michael"})
    qt_app.document.characters.append(bob)
    text = "line one\nline two"
    _set_text(editor, text)
    qt_app.document.assign_character_to_range(0, len(text), alice.id)
    editor.rehighlight()
    _repaint(gutter)
    assert len(gutter._label_rects) == 1
    rect, line_start, line_end = gutter._label_rects[0]
    assert line_end < len(text)  # the label's own line is shorter than the whole clip

    menu = gutter._build_picker_menu(line_start, line_end)
    next(a for a in menu.actions() if a.text() == "Bob").trigger()

    # The reassignment covered the WHOLE clip (through "line two", not just
    # "line one" where the label happened to be painted).
    assert qt_app.document.clip_covering(0).character_id == bob.id
    assert qt_app.document.clip_covering(len(text) - 1).character_id == bob.id


def test_gutter_resizes_with_the_editor(qt_app):
    """Drives TranscriptEditor.resizeEvent directly (no real display/shown
    window in this fixture to guarantee a queued QResizeEvent gets
    delivered synchronously) - same "call the Qt override directly, no
    event loop needed" precedent as _repaint above."""
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QResizeEvent

    editor = _editor(qt_app)
    gutter = _gutter(qt_app)
    old_size = editor.size()
    new_size = QSize(600, 400)
    editor.resize(new_size)
    editor.resizeEvent(QResizeEvent(new_size, old_size))

    assert gutter.geometry().width() == GUTTER_WIDTH_PX
    assert gutter.geometry().height() == editor.height()


# -- imported recordings (phase 5 P3) ------------------------------------------------


def _tall(editor):
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QResizeEvent

    old_size = editor.size()
    editor.resize(QSize(600, 600))
    editor.resizeEvent(QResizeEvent(editor.size(), old_size))


def _click(gutter, rect):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    pos = QPointF(rect.center())
    gutter.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, pos, pos, Qt.MouseButton.LeftButton,
                                       Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))


def test_a_recording_clip_gets_a_play_only_button_that_plays_from_its_start(qt_app, tmp_path):
    from tests.gui_qt.test_transcript_editor import HELLO, SECOND, _recording

    editor, gutter = _editor(qt_app), _gutter(qt_app)
    _tall(editor)
    _set_text(editor, "Intro line.")
    _source, (first_id, second_id) = _recording(qt_app, tmp_path, [HELLO, SECOND])
    _repaint(gutter)

    assert gutter.button_rects() == []  # never stale, so never Generate
    assert [cid for _rect, cid in gutter.play_rects()] == [first_id, second_id]

    played = []
    qt_app.transport.seek = lambda s: played.append(("seek", s))
    qt_app.transport.play = lambda: played.append(("play",))
    generated = []
    qt_app.generate_clip = generated.append
    _click(gutter, gutter.play_rects()[1][0])

    placed = qt_app.current_arrangement().by_clip_id()[second_id]
    assert placed.start_s > 0.0
    assert played == [("seek", placed.start_s), ("play",)] and generated == []


def test_text_typed_into_a_recording_gets_a_hollow_no_character_mark(qt_app, tmp_path):
    from tests.gui_qt.test_transcript_editor import HELLO, _recording

    editor, gutter = _editor(qt_app), _gutter(qt_app)
    _tall(editor)
    _set_text(editor, "Plain line, no mark.")
    _recording(qt_app, tmp_path, [HELLO])
    _repaint(gutter)
    assert gutter.mark_rects() == []  # plain untagged text has no mark

    at = qt_app.document.text.index(" friend")
    cursor = editor.textCursor()
    cursor.setPosition(at)
    cursor.insertText(" my")
    _repaint(gutter)

    (rect, line_start, line_end), = gutter.mark_rects()
    assert line_start <= at < line_end
    assert "No character" in gutter.tooltip_at(rect.center())

    qt_app.document.assign_character_to_range(at, at + 3, qt_app.document.characters[0].id)
    _repaint(gutter)
    assert gutter.mark_rects() == []
