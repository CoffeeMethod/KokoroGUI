"""Split and join a clip at the playhead (plan 10): Edit menu entries, the
timeline block menu, the timeline-scoped S key, and one-step undo."""
import numpy as np
import soundfile as sf
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtTest import QTest

from kokoro_gui.daw.models import Character, Segment, Track
from kokoro_gui.qt.about_dialog import shortcuts_text

LINE = "alpha beta gamma delta"


def _type(editor, text):
    cursor = editor.textCursor()
    cursor.select(QTextCursor.SelectionType.Document)
    cursor.insertText(text)


def _one_clip(qt_app, text=LINE):
    _type(qt_app.editor, text)
    alice = qt_app.document.characters[0]
    clip = qt_app.document.assign_character_to_range(0, len(text), alice.id)
    qt_app.editor.rehighlight()
    qt_app.refresh_timeline()
    return clip


def _playhead(qt_app, seconds):
    qt_app.transport.position = lambda: seconds


def _placed(qt_app, clip):
    return qt_app.build_arrangement().by_clip_id()[clip.id]


def _split_at_fraction(qt_app, clip, fraction):
    placed = _placed(qt_app, clip)
    _playhead(qt_app, placed.start_s + placed.duration_s * fraction)


def test_split_at_the_playhead_makes_two_clips_and_one_undo_restores_one(qt_app):
    clip = _one_clip(qt_app)
    text = qt_app.document.text
    _split_at_fraction(qt_app, clip, 0.55)  # inside "gamma"; the next word start is "delta"

    assert qt_app.split_clip_at_playhead() is True

    doc = qt_app.document
    assert doc.text == text and qt_app.editor.toPlainText() == text
    assert len(doc.clips) == 2
    assert doc.clip_text(doc.clips[0]) == "alpha beta gamma " and doc.clip_text(doc.clips[1]) == "delta"
    assert doc.clips[1].gap_before_s == 0.0

    qt_app.undo()
    assert len(doc.clips) == 1 and doc.clip_text(doc.clips[0]) == text
    qt_app.redo()
    assert len(doc.clips) == 2


def test_split_uses_the_word_under_the_playhead_when_the_clip_has_word_times(qt_app, tmp_path):
    clip = _one_clip(qt_app)
    path = str(tmp_path / "a.wav")
    sf.write(path, np.zeros(24000 * 4, dtype=np.float32), 24000)
    clip.segments = [Segment(order_index=0, text=LINE, audio_path=path, duration=4.0,
                             words=[["alpha", 0.0, 1.0], ["beta", 1.0, 2.0], ["gamma", 2.0, 3.0],
                                    ["delta", 3.0, 4.0]])]
    qt_app.refresh_timeline()
    placed = _placed(qt_app, clip)
    assert abs(placed.duration_s - 4.0) < 1e-6
    _playhead(qt_app, placed.start_s + 1.4)  # on "beta"

    assert qt_app.split_clip_at_playhead() is True

    first, second = qt_app.document.clips
    assert qt_app.document.clip_text(first) == "alpha " and qt_app.document.clip_text(second) == "beta gamma delta"
    # The audio stays with the first half, which no longer matches its text.
    assert first.segments and not second.segments
    assert first in qt_app.document.dirty_clips() and second in qt_app.document.dirty_clips()


def test_split_in_a_gap_off_every_clip_says_so_and_changes_nothing(qt_app):
    _one_clip(qt_app)
    _playhead(qt_app, 500.0)
    assert qt_app.split_clip_at_playhead() is False
    assert len(qt_app.document.clips) == 1
    assert not qt_app.document.undo_stack.can_undo()


def test_split_refuses_a_clip_placed_on_the_timeline(qt_app):
    clip = _one_clip(qt_app)
    clip.timeline_timestamp = 0.0
    qt_app.refresh_timeline()
    _split_at_fraction(qt_app, clip, 0.55)
    assert qt_app.split_clip_at_playhead() is False
    assert len(qt_app.document.clips) == 1


def test_join_with_next_merges_two_clips_and_one_undo_splits_them_again(qt_app):
    clip = _one_clip(qt_app)
    _split_at_fraction(qt_app, clip, 0.55)
    qt_app.split_clip_at_playhead()
    first, second = qt_app.document.clips
    qt_app.selection.selected_clip_id = first.id

    assert qt_app.join_selected_with_next() is True

    assert len(qt_app.document.clips) == 1
    assert qt_app.document.clip_text(qt_app.document.clips[0]) == LINE
    qt_app.undo()
    assert len(qt_app.document.clips) == 2
    qt_app.undo()
    assert len(qt_app.document.clips) == 1


def test_join_across_two_characters_is_refused(qt_app):
    bob = Character.from_preset_dict("Bob", {})
    qt_app.document.characters.append(bob)
    qt_app.document.tracks.append(Track(name="Bob", character_id=bob.id, order_index=1))
    _type(qt_app.editor, "aaa bbb")
    alice = qt_app.document.characters[0]
    first = qt_app.document.assign_character_to_range(0, 3, alice.id)
    qt_app.document.assign_character_to_range(4, 7, bob.id)
    qt_app.refresh_timeline()
    assert qt_app.join_clip_with_next(first.id) is False
    assert len(qt_app.document.clips) == 2


def test_edit_menu_entries_follow_the_playhead_and_the_selection(qt_app):
    clip = _one_clip(qt_app)
    _split_at_fraction(qt_app, clip, 0.55)
    qt_app._sync_split_join_actions()
    assert qt_app.split_clip_action.isEnabled()
    assert not qt_app.join_clip_action.isEnabled()  # no clip after it

    _playhead(qt_app, 500.0)
    qt_app._sync_split_join_actions()
    assert not qt_app.split_clip_action.isEnabled()

    _split_at_fraction(qt_app, clip, 0.55)
    qt_app.split_clip_at_playhead()
    first = qt_app.document.clips[0]
    qt_app.selection.selected_clip_id = first.id
    qt_app._sync_split_join_actions()
    assert qt_app.join_clip_action.isEnabled()
    texts = [a.text() for a in qt_app.edit_menu.actions()]
    assert "&Split Clip at Playhead" in texts and "&Join with Next Clip" in texts


def _block_menu(qt_app, clip, x_fraction=0.5):
    view = qt_app.timeline_dock.timeline_view
    block = next(item for item in view.items() if getattr(item, "clip_id", None) == clip.id)
    rect = block.boundingRect()
    point = block.mapToScene(QPointF(rect.width() * x_fraction, rect.height() / 2))
    return view, view._build_context_menu(view.mapFromScene(point))


def test_block_menu_split_here_cuts_at_the_clicked_time(qt_app):
    clip = _one_clip(qt_app)
    view, menu = _block_menu(qt_app, clip, 0.55)
    actions = {a.text(): a for a in menu.actions() if a.text()}
    assert actions["Split here"].isEnabled()
    assert not actions["Join with next"].isEnabled()  # no clip follows

    actions["Split here"].trigger()

    first, second = qt_app.document.clips
    assert qt_app.document.clip_text(first) == "alpha beta gamma " and qt_app.document.clip_text(second) == "delta"

    _view, menu = _block_menu(qt_app, first, 0.5)
    actions = {a.text(): a for a in menu.actions() if a.text()}
    assert actions["Join with next"].isEnabled()
    actions["Join with next"].trigger()
    assert len(qt_app.document.clips) == 1


def test_block_menu_split_is_off_for_a_pinned_clip(qt_app):
    clip = _one_clip(qt_app)
    clip.timeline_timestamp = 0.0
    qt_app.refresh_timeline()
    _view, menu = _block_menu(qt_app, clip)
    actions = {a.text(): a for a in menu.actions() if a.text()}
    assert not actions["Split here"].isEnabled()
    assert "Unpin" in actions["Split here"].toolTip()


def _show_active(qt_app, qtbot):
    qt_app.show()
    qtbot.waitExposed(qt_app)
    qt_app.activateWindow()
    qtbot.waitUntil(qt_app.isActiveWindow, timeout=2000)


def test_the_s_key_with_the_transcript_focused_types_an_s_and_does_not_split(qt_app, qtbot):
    clip = _one_clip(qt_app)
    _show_active(qt_app, qtbot)
    editor = qt_app.editor
    editor.setFocus()
    cursor = editor.textCursor()
    cursor.movePosition(QTextCursor.MoveOperation.End)
    editor.setTextCursor(cursor)
    _split_at_fraction(qt_app, clip, 0.55)

    QTest.keyClick(editor, Qt.Key.Key_S)

    assert editor.toPlainText() == LINE + "s"
    assert len(qt_app.document.clips) == 1


def test_the_s_key_with_the_timeline_focused_splits_at_the_playhead(qt_app, qtbot):
    clip = _one_clip(qt_app)
    _show_active(qt_app, qtbot)
    view = qt_app.timeline_dock.timeline_view
    view.setFocus()
    _split_at_fraction(qt_app, clip, 0.55)
    assert qt_app.split_shortcut.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut

    QTest.keyClick(view, Qt.Key.Key_S)

    assert len(qt_app.document.clips) == 2
    assert qt_app.editor.toPlainText() == LINE


def test_the_shortcut_sheet_lists_the_split_key(qt_app):
    assert qt_app.split_shortcut.key().toString() == "S"
    assert "Split clip at playhead" in shortcuts_text(qt_app)
