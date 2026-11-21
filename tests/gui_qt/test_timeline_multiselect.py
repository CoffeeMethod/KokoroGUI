"""Plan 35: several clips selected on the timeline. Ctrl+click toggles,
Shift+click extends in text order, a drag from empty lane space is a rubber
band, and the block menu's batch entries apply to every selected clip as one
undo step."""
from PySide6.QtCore import QPoint, QRectF, Qt
from PySide6.QtGui import QTextCursor

from kokoro_gui.daw.models import Character, Clip, Document, Run, Track
from kokoro_gui.qt.selection import SelectionModel
from kokoro_gui.qt.timeline_view import ClipBlockItem, TimelineView

CTRL = Qt.KeyboardModifier.ControlModifier
SHIFT = Qt.KeyboardModifier.ShiftModifier


def _doc_with_clips(n=4):
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="Alice", character_id=alice.id)
    clips = [Clip(character_id=alice.id, track_id=track.id) for _ in range(n)]
    runs = []
    for clip in clips:
        runs.append(Run(text="x" * 20, clip_id=clip.id, kind=clip.source))
        runs.append(Run(text=" "))
    return Document(runs=runs, characters=[alice], tracks=[track], clips=clips), clips


def _view(qtbot, n=4):
    selection = SelectionModel()
    view = TimelineView(selection_model=selection)
    qtbot.addWidget(view)
    view.resize(1600, 300)
    doc, clips = _doc_with_clips(n)
    view.render_document(doc)
    return view, selection, clips


def _block(view, clip_id) -> ClipBlockItem:
    return view._blocks_by_clip_id[clip_id]


def _pos(view, clip_id) -> QPoint:
    return view.mapFromScene(_block(view, clip_id).mapToScene(5, 5))


def _click(view, pos, qtbot, modifier=Qt.KeyboardModifier.NoModifier):
    qtbot.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, modifier, pos)


def test_ctrl_click_adds_and_drops_a_block(qtbot):
    view, selection, clips = _view(qtbot)
    _click(view, _pos(view, clips[0].id), qtbot)
    _click(view, _pos(view, clips[2].id), qtbot, CTRL)
    assert selection.selected_clip_ids == (clips[0].id, clips[2].id)
    assert _block(view, clips[0].id)._selected and _block(view, clips[2].id)._selected
    assert not _block(view, clips[1].id)._selected

    _click(view, _pos(view, clips[0].id), qtbot, CTRL)
    assert selection.selected_clip_ids == (clips[2].id,)
    assert not _block(view, clips[0].id)._selected


def test_shift_click_extends_in_text_order(qtbot):
    view, selection, clips = _view(qtbot)
    _click(view, _pos(view, clips[1].id), qtbot)
    _click(view, _pos(view, clips[3].id), qtbot, SHIFT)
    assert set(selection.selected_clip_ids) == {clips[1].id, clips[2].id, clips[3].id}
    assert selection.selected_clip_id == clips[1].id
    assert all(_block(view, c.id)._selected for c in clips[1:])


def test_a_plain_click_collapses_a_several_clip_selection(qtbot):
    view, selection, clips = _view(qtbot)
    selection.select_clips([c.id for c in clips])
    _click(view, _pos(view, clips[1].id), qtbot)
    assert selection.selected_clip_ids == (clips[1].id,)
    assert [b for b in view._selected_blocks] == [_block(view, clips[1].id)]


def test_rubber_band_from_empty_lane_space_selects_the_blocks_it_touches(qtbot):
    view, selection, clips = _view(qtbot)
    first, third = _block(view, clips[0].id), _block(view, clips[2].id)
    lane_y = first.sceneBoundingRect().center().y()
    # Start below the blocks, in the lane's empty space past the last clip.
    start = view.mapFromScene(_block(view, clips[3].id).sceneBoundingRect().right() + 200, lane_y)
    assert view._clip_block_at(start) is None
    end = view.mapFromScene(third.sceneBoundingRect().center().x(), lane_y)
    viewport = view.viewport()
    qtbot.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(viewport, pos=end)
    qtbot.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=end)
    assert set(selection.selected_clip_ids) == {clips[2].id, clips[3].id}
    assert view._band_item is None


def test_select_blocks_in_adds_with_ctrl(qtbot):
    view, selection, clips = _view(qtbot)
    selection.select_clip(clips[0].id)
    view.select_blocks_in(QRectF(_block(view, clips[3].id).sceneBoundingRect()), add=True)
    assert selection.selected_clip_ids == (clips[3].id, clips[0].id)
    view.select_blocks_in(QRectF(-100, -100, 1, 1))
    assert selection.kind == "none"


def test_block_menu_with_several_selected_is_the_batch_menu(qtbot):
    view, selection, clips = _view(qtbot)
    selection.select_clips([clips[0].id, clips[1].id])
    menu = view._build_context_menu(_pos(view, clips[1].id))
    labels = [a.text() for a in menu.actions() if a.text()]
    assert labels[0] == "2 clips selected"
    assert {"Generate stale clips", "Status", "FX", "Remove gap override", "Align words"} <= set(labels)

    emitted = []
    view.batchActionRequested.connect(lambda ids, action, value: emitted.append((ids, action, value)))
    status_menu = next(a.menu() for a in menu.actions() if a.text() == "Status")
    next(a for a in status_menu.actions() if a.text() == "Approved").trigger()
    assert emitted == [([clips[0].id, clips[1].id], "status", "approved")]

    # A block outside the selection gets its own menu.
    single = view._build_context_menu(_pos(view, clips[3].id))
    assert "2 clips selected" not in [a.text() for a in single.actions()]


# -- the app side ---------------------------------------------------------------

def _three_clips(qt_app):
    cursor = qt_app.editor.textCursor()
    cursor.select(QTextCursor.SelectionType.Document)
    cursor.insertText("one. two. three.")
    character = qt_app.document.characters[0]
    return [qt_app.document.assign_character_to_range(s, e, character.id) for s, e in ((0, 4), (5, 9), (10, 16))]


def test_batch_status_on_three_clips_is_one_undo_step(qt_app):
    clips = _three_clips(qt_app)
    qt_app.timeline_dock.on_batch_action_requested([c.id for c in clips], "status", "approved")
    assert all(qt_app.document.get_clip(c.id).status == "approved" for c in clips)
    qt_app.document.undo_stack.undo()
    assert all(qt_app.document.get_clip(c.id).status != "approved" for c in clips)


def test_batch_clear_gap_and_fx(qt_app):
    clips = _three_clips(qt_app)
    for clip in clips:
        clip.gap_before_s = 1.0
    qt_app.timeline_dock.on_batch_action_requested([c.id for c in clips], "clear_gap", None)
    assert all(c.gap_before_s is None for c in clips)
    qt_app.engine.load_fx_preset.return_value = {"reverb_enabled": True}
    qt_app.timeline_dock.on_batch_action_requested([c.id for c in clips], "fx", "Warm")
    assert all(c.fx_override and c.overrides.get("fx_preset") == "Warm" for c in clips)
    qt_app.document.undo_stack.undo()
    assert all(not c.fx_override for c in clips)


def test_app_selected_clip_ids_prefers_several_selected(qt_app):
    clips = _three_clips(qt_app)
    qt_app.selection.select_clips([clips[0].id, clips[2].id])
    assert qt_app.selected_clip_ids() == {clips[0].id, clips[2].id}


def test_a_text_range_selects_the_clips_it_spans(qt_app):
    clips = _three_clips(qt_app)
    editor = qt_app.editor
    cursor = editor.textCursor()
    cursor.setPosition(1)
    cursor.setPosition(12, cursor.MoveMode.KeepAnchor)
    editor.setTextCursor(cursor)
    assert set(qt_app.selection.selected_clip_ids) == {c.id for c in clips}
    assert qt_app.selection.selected_clip_id == clips[0].id
    assert editor.textCursor().selectionEnd() == 12  # the text selection stays
