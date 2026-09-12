"""Tests for item 8 ("Drag-to-reassign a clip to a different track"):
dragging a clip block onto a different track's lane, Q9's reassign-vs-move
prompt when the target lane's character differs from the clip's own, and
`TimelineDock.on_clip_drag_reassigned` re-resolving ids off
`TimelineView.clipDragReassigned` and pushing the resulting
`MoveClipCommand`/`ReassignTrackCommand` (kokoro_gui/daw/undo.py) - mirrors
test_timeline_clip_fx.py's conventions (qt_app fixture, driving the real
dock/view rather than only unit-level document mutation).
"""
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QMessageBox

from kokoro_gui.daw.models import Character, Track
from kokoro_gui.qt.timeline_view import LANE_HEIGHT_PX, lane_top


def _setup_two_tracks(qt_app, same_character: bool):
    """Replaces qt_app.document's tracks with two lanes ordered [0, 1] -
    track_a (the clip's starting lane) and track_b (the drop target).
    `same_character=False` gives track_b a distinct Character, engaging
    Q9's reassign-vs-move prompt on drop."""
    document = qt_app.document
    alice = document.characters[0]
    if same_character:
        bob = alice
    else:
        bob = Character.from_preset_dict("Bob", {})
        document.characters.append(bob)

    track_a = Track(name="Track A", character_id=alice.id, order_index=0)
    track_b = Track(name="Track B", character_id=bob.id, order_index=1)
    document.tracks = [track_a, track_b]
    return track_a, track_b, alice, bob


def _make_clip_on_track(qt_app, track, start=0, end=5, text="hello world"):
    document = qt_app.document
    document.text = text
    clip = document.assign_character_to_range(start, end, track.character_id)
    clip.track_id = track.id
    return clip


def _drag_clip_onto_track_b(qt_app, qtbot, clip):
    """Simulates the full mouse gesture: press on the clip's block, release
    over track_b's lane (lane index 1, below the ruler)."""
    qt_app.timeline_dock.refresh()
    view = qt_app.timeline_dock.timeline_view
    block = view._blocks_by_clip_id[clip.id]

    press_pos = view.mapFromScene(block.mapToScene(2, 2))
    release_scene_x = block.mapToScene(2, 2).x()
    release_pos = view.mapFromScene(QPointF(release_scene_x, lane_top(1) + 10))

    qtbot.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=press_pos)
    qtbot.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=release_pos)


def _clicked_button_named(button_text):
    """A QMessageBox.clickedButton replacement that always resolves to the
    button whose text matches `button_text`, regardless of which QMessageBox
    instance called it - used with QMessageBox.exec patched to a no-op, so
    tests can drive each of the three prompt outcomes (Reassign/Just Move/
    Cancel) without a real, blocking modal dialog."""
    def _clicked_button(self):
        return next(b for b in self.buttons() if b.text() == button_text)
    return _clicked_button


# ---------------------------------------------------------------------------
# No ambiguity: target lane's character already matches - straight move,
# no prompt.
# ---------------------------------------------------------------------------

def test_drag_to_track_with_matching_character_moves_without_prompt(qt_app, qtbot, monkeypatch):
    track_a, track_b, alice, _bob = _setup_two_tracks(qt_app, same_character=True)
    clip = _make_clip_on_track(qt_app, track_a)

    def _fail_exec(self):
        raise AssertionError("no prompt expected when the target character already matches")
    monkeypatch.setattr(QMessageBox, "exec", _fail_exec)

    _drag_clip_onto_track_b(qt_app, qtbot, clip)

    assert clip.track_id == track_b.id
    assert clip.character_id == alice.id
    assert qt_app.document.undo_stack.can_undo() is True

    qt_app.document.undo_stack.undo()
    assert clip.track_id == track_a.id


# ---------------------------------------------------------------------------
# Ambiguity: target lane's character differs - the three-way prompt.
# ---------------------------------------------------------------------------

def test_drag_reassign_choice_updates_track_and_character(qt_app, qtbot, monkeypatch):
    track_a, track_b, alice, bob = _setup_two_tracks(qt_app, same_character=False)
    clip = _make_clip_on_track(qt_app, track_a)

    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    monkeypatch.setattr(QMessageBox, "clickedButton", _clicked_button_named("Reassign"))

    _drag_clip_onto_track_b(qt_app, qtbot, clip)

    assert clip.track_id == track_b.id
    assert clip.character_id == bob.id
    assert qt_app.document.undo_stack.can_undo() is True

    qt_app.document.undo_stack.undo()
    assert clip.track_id == track_a.id
    assert clip.character_id == alice.id


def test_drag_just_move_choice_updates_track_only(qt_app, qtbot, monkeypatch):
    track_a, track_b, alice, _bob = _setup_two_tracks(qt_app, same_character=False)
    clip = _make_clip_on_track(qt_app, track_a)

    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    monkeypatch.setattr(QMessageBox, "clickedButton", _clicked_button_named("Just Move"))

    _drag_clip_onto_track_b(qt_app, qtbot, clip)

    assert clip.track_id == track_b.id
    assert clip.character_id == alice.id  # unchanged
    assert qt_app.document.undo_stack.can_undo() is True

    qt_app.document.undo_stack.undo()
    assert clip.track_id == track_a.id
    assert clip.character_id == alice.id


def test_drag_cancel_choice_pushes_nothing(qt_app, qtbot, monkeypatch):
    track_a, _track_b, alice, _bob = _setup_two_tracks(qt_app, same_character=False)
    clip = _make_clip_on_track(qt_app, track_a)
    can_undo_before = qt_app.document.undo_stack.can_undo()

    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    monkeypatch.setattr(QMessageBox, "clickedButton", _clicked_button_named("Cancel"))

    _drag_clip_onto_track_b(qt_app, qtbot, clip)

    assert clip.track_id == track_a.id
    assert clip.character_id == alice.id
    assert qt_app.document.undo_stack.can_undo() is can_undo_before


# ---------------------------------------------------------------------------
# Busy/no-op cases leave the undo stack untouched.
# ---------------------------------------------------------------------------

def test_drag_along_same_track_pins_a_timestamp_and_keeps_the_track(qt_app, qtbot):
    """UI9: a horizontal drag on the same lane is a move on the seconds
    axis (an undoable SetClipTimestampCommand), never a track change."""
    track_a, _track_b, _alice, _bob = _setup_two_tracks(qt_app, same_character=True)
    clip = _make_clip_on_track(qt_app, track_a)
    assert clip.timeline_timestamp is None

    qt_app.timeline_dock.refresh()
    view = qt_app.timeline_dock.timeline_view
    block = view._blocks_by_clip_id[clip.id]
    press_pos = view.mapFromScene(block.mapToScene(2, 2))
    release_pos = view.mapFromScene(block.mapToScene(60, 2))  # same lane, big x movement

    qtbot.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=press_pos)
    qtbot.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=release_pos)

    assert clip.track_id == track_a.id
    assert clip.timeline_timestamp is not None
    assert clip.timeline_timestamp > 0
    qt_app.undo()
    assert clip.timeline_timestamp is None


def test_drag_off_all_lanes_pushes_no_command(qt_app, qtbot):
    track_a, _track_b, _alice, _bob = _setup_two_tracks(qt_app, same_character=True)
    clip = _make_clip_on_track(qt_app, track_a)
    can_undo_before = qt_app.document.undo_stack.can_undo()

    qt_app.timeline_dock.refresh()
    view = qt_app.timeline_dock.timeline_view
    block = view._blocks_by_clip_id[clip.id]
    press_pos = view.mapFromScene(block.mapToScene(2, 2))
    release_scene_x = block.mapToScene(2, 2).x()
    release_pos = view.mapFromScene(QPointF(release_scene_x, LANE_HEIGHT_PX * 100))

    qtbot.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=press_pos)
    qtbot.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=release_pos)  # must not crash

    assert clip.track_id == track_a.id
    assert qt_app.document.undo_stack.can_undo() is can_undo_before


def test_clip_drag_reassigned_signal_is_connected_to_the_dock_handler(qt_app):
    track_a, track_b, alice, _bob = _setup_two_tracks(qt_app, same_character=True)
    clip = _make_clip_on_track(qt_app, track_a)

    qt_app.timeline_dock.timeline_view.clipDragReassigned.emit(clip.id, track_b.id, False)

    assert clip.track_id == track_b.id
    assert clip.character_id == alice.id
