"""Tests for TimelineDock.on_clip_moved / on_clip_unpin_requested (UI9 and
grill Q13): a horizontal drop pins a timestamp, and a drop before the
text-order predecessor also moves the clip's text."""
from PySide6.QtGui import QTextCursor

from kokoro_gui.daw.arrangement import compute_arrangement
from kokoro_gui.daw.models import Character


def _type(editor, text):
    cursor = editor.textCursor()
    cursor.select(QTextCursor.SelectionType.Document)
    cursor.insertText(text)


def _two_clips(qt_app):
    bob = Character.from_preset_dict("Bob", {})
    qt_app.document.characters.append(bob)
    from kokoro_gui.daw.models import Track

    qt_app.document.tracks.append(Track(name="Bob", character_id=bob.id, order_index=1))
    _type(qt_app.editor, "aaaaaaaaaa bbbbbbbbbb")
    alice = qt_app.document.characters[0]
    a = qt_app.document.assign_character_to_range(0, 10, alice.id)
    b = qt_app.document.assign_character_to_range(11, 21, bob.id)
    qt_app.editor.rehighlight()
    qt_app.refresh_timeline()
    return a, b


def test_drop_after_the_predecessor_only_pins_a_timestamp(qt_app):
    a, b = _two_clips(qt_app)
    text_before = qt_app.document.text

    qt_app.timeline_dock.on_clip_moved(b.id, 9.0)

    assert qt_app.document.text == text_before
    assert b.timeline_timestamp == 9.0
    assert compute_arrangement(qt_app.document, chars_per_second=10.0).placed[1].start_s == 9.0
    qt_app.undo()
    assert b.timeline_timestamp is None


def test_drop_before_the_predecessor_moves_the_text_too(qt_app):
    a, b = _two_clips(qt_app)

    qt_app.timeline_dock.on_clip_moved(b.id, 0.0)  # a starts at 0.0 - b now lands before it

    assert qt_app.document.text == "bbbbbbbbbbaaaaaaaaaa "
    assert qt_app.editor.toPlainText() == qt_app.document.text
    assert b.timeline_timestamp == 0.0
    placed = compute_arrangement(qt_app.document, chars_per_second=10.0).placed
    assert [p.clip.id for p in placed] == [b.id, a.id]

    qt_app.undo()
    assert qt_app.document.text == "aaaaaaaaaa bbbbbbbbbb"
    # The command restores a snapshot, so re-fetch the clip by id.
    assert qt_app.document.get_clip(b.id).timeline_timestamp is None


def test_unpin_clears_the_timestamp(qt_app):
    a, _b = _two_clips(qt_app)
    qt_app.timeline_dock.on_clip_moved(a.id, 5.0)
    assert a.timeline_timestamp == 5.0

    qt_app.timeline_dock.on_clip_unpin_requested(a.id)

    assert a.timeline_timestamp is None


def test_timeline_context_menu_offers_unpin_only_for_pinned_clips(qt_app):
    a, _b = _two_clips(qt_app)
    view = qt_app.timeline_dock.timeline_view
    block = view._blocks_by_clip_id[a.id]
    pos = view.mapFromScene(block.mapToScene(2, 2))
    assert "Unpin from timeline" not in [x.text() for x in view._build_context_menu(pos).actions()]

    qt_app.timeline_dock.on_clip_moved(a.id, 5.0)
    block = view._blocks_by_clip_id[a.id]
    pos = view.mapFromScene(block.mapToScene(2, 2))
    assert "Unpin from timeline" in [x.text() for x in view._build_context_menu(pos).actions()]
