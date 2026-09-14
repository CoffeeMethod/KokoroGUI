"""Tests for kokoro_gui/qt/timeline_view.py's click-to-select wiring (item 1,
"Sync layer") - standalone TimelineView + a bare SelectionModel, no full
qt_app, mirroring test_timeline_view.py's app-independence pattern."""
from PySide6.QtCore import QPoint, Qt

from kokoro_gui.daw.models import Character, Clip, Document, Run, Track
from kokoro_gui.qt.selection import SelectionModel
from kokoro_gui.qt.timeline_view import ClipBlockItem, TimelineView


def _clip_block_items(view):
    return [item for item in view._scene.items() if isinstance(item, ClipBlockItem)]


def _click(view, pos: QPoint, qtbot):
    qtbot.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=pos)


def _build_doc_with_one_clip():
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="Alice", character_id=alice.id)
    clip = Clip(character_id=alice.id, track_id=track.id)
    doc = Document(
        runs=[Run(text="x" * 10, clip_id=clip.id, kind=clip.source), Run(text="x" * 30)],
        characters=[alice], tracks=[track], clips=[clip],
    )
    return doc, clip, track


def test_clicking_clip_block_selects_that_clip(qtbot):
    selection = SelectionModel()
    view = TimelineView(selection_model=selection)
    qtbot.addWidget(view)
    doc, clip, _track = _build_doc_with_one_clip()
    view.render_document(doc)

    block = _clip_block_items(view)[0]
    pos = view.mapFromScene(block.mapToScene(5, 5))
    _click(view, pos, qtbot)

    assert selection.selected_clip_id == clip.id
    assert selection.kind == "clip"


def test_clicking_empty_lane_space_clears_selection(qtbot):
    selection = SelectionModel()
    view = TimelineView(selection_model=selection)
    qtbot.addWidget(view)
    doc, clip, _track = _build_doc_with_one_clip()
    view.render_document(doc)
    selection.select_clip(clip.id)

    # Far past the clip's end, still inside the lane rect - empty space.
    pos = view.mapFromScene(500, 40)
    _click(view, pos, qtbot)

    assert selection.kind == "none"


def test_clicking_lane_label_selects_character(qtbot):
    """Track labels live in the header column (TimelineWidget.header), a
    separate view so labels never overlap clips."""
    from kokoro_gui.qt.timeline_view import TimelineWidget, lane_top

    selection = SelectionModel()
    widget = TimelineWidget(selection_model=selection)
    qtbot.addWidget(widget)
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="Alice", character_id=alice.id)
    doc = Document.from_plain_text("x" * 40, characters=[alice], tracks=[track], clips=[])
    widget.render_document(doc)

    # The label is drawn at (22, lane_top + 5); click inside its glyph area.
    header = widget.header
    pos = header.mapFromScene(30, lane_top(0) + 10)
    _click(header, pos, qtbot)

    assert selection.kind == "character"
    assert selection.selected_character_id == alice.id


def test_set_selected_flips_internal_attribute():
    block = ClipBlockItem()
    assert block._selected is False
    block.set_selected(True)
    assert block._selected is True
    block.set_selected(False)
    assert block._selected is False


def test_selection_survives_unrelated_rerender(qtbot):
    selection = SelectionModel()
    view = TimelineView(selection_model=selection)
    qtbot.addWidget(view)
    doc, clip, _track = _build_doc_with_one_clip()
    view.render_document(doc)
    selection.select_clip(clip.id)

    # Simulate an unrelated refresh (e.g. a keystroke elsewhere).
    view.render_document(doc)

    block = _clip_block_items(view)[0]
    assert block.clip_id == clip.id
    assert block._selected is True


def test_selection_of_removed_clip_does_not_crash_and_clears_highlight(qtbot):
    selection = SelectionModel()
    view = TimelineView(selection_model=selection)
    qtbot.addWidget(view)
    doc, clip, _track = _build_doc_with_one_clip()
    view.render_document(doc)
    selection.select_clip(clip.id)

    doc.clips = []
    view.render_document(doc)  # must not raise

    assert _clip_block_items(view) == []
    assert view._selected_block is None
