"""File > Import Audio as a music bed (phase 5 P2, grill Q30): the clip
shape, one undo step, the read-only line with no play button, duration and
waveform from the file, edge trims and loops in the transport schedule, the
track's duck toggle and the ducking setting."""
import os

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QPaintEvent
from PySide6.QtWidgets import QFileDialog

from kokoro_gui.daw.models import PLACEHOLDER
from kokoro_gui.qt.timeline_view import ClipBlockItem

RATE = 24000


def _wav(tmp_path, name="Theme Song.wav", seconds=4.0, value=0.1):
    path = str(tmp_path / name)
    sf.write(path, np.full(int(seconds * RATE), value, dtype=np.float32), RATE)
    return path


def _import(qt_app, tmp_path, monkeypatch, at_s=0.0, **wav):
    path = _wav(tmp_path, **wav)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (path, "")))
    monkeypatch.setattr(qt_app, "_ask_audio_import", lambda path: {"kind": "bed"})
    asked = []
    monkeypatch.setattr(qt_app, "_ask_bed_placement", lambda playhead: asked.append(playhead) or at_s)
    before = {c.id for c in qt_app.document.clips}
    qt_app.import_audio_dialog()
    new = [c for c in qt_app.document.clips if c.id not in before]
    assert len(new) == 1
    return new[0], asked


def _blocks(qt_app):
    view = qt_app.timeline_dock.timeline_view
    return {item.clip_id: item for item in view.scene().items() if isinstance(item, ClipBlockItem)}


def _schedule(qt_app, monkeypatch):
    captured = []
    load = qt_app.transport.load

    def _load(schedule, **kwargs):
        captured.append((list(schedule), kwargs))
        return load(schedule, **kwargs)

    monkeypatch.setattr(qt_app.transport, "load", _load)
    qt_app._rebuild_transport_schedule()
    return captured[-1]


def test_import_audio_adds_a_pinned_bed_on_a_music_track(qt_app, tmp_path, monkeypatch):
    document = qt_app.document
    document.text = "Welcome to the show."
    qt_app.editor.load_text(document.text)

    bed, asked = _import(qt_app, tmp_path, monkeypatch)

    assert asked == [0.0]
    assert bed.source == "imported" and bed.is_bed
    assert bed.original_audio_path.startswith(os.path.join(qt_app.project_dir, "audio", "imported") + os.sep)
    assert os.path.isfile(bed.original_audio_path)
    assert bed.segments == [] and bed.pinned and bed.timeline_timestamp == 0.0
    track = document.get_track(bed.track_id)
    assert (track.name, track.role, track.duck) == ("Music", "music", False)
    assert document.text == "Welcome to the show.\n\nTheme Song"
    assert qt_app.editor.toPlainText() == document.text
    runs = [r for r in document.runs if r.clip_id == bed.id]
    assert [(r.text, r.kind) for r in runs] == [("Theme Song", PLACEHOLDER)]


def test_import_is_one_undo_step(qt_app, tmp_path, monkeypatch):
    document = qt_app.document
    document.text = "Hello."
    qt_app.editor.load_text(document.text)
    tracks_before = [t.id for t in document.tracks]

    bed, _asked = _import(qt_app, tmp_path, monkeypatch)
    qt_app.editor.undo_coordinator.undo()

    assert document.get_clip(bed.id) is None
    assert document.text == "Hello."
    assert qt_app.editor.toPlainText() == "Hello."
    assert [t.id for t in document.tracks] == tracks_before


def test_place_at_the_playhead(qt_app, tmp_path, monkeypatch):
    bed, _asked = _import(qt_app, tmp_path, monkeypatch, at_s=2.5)
    assert bed.timeline_timestamp == 2.5
    assert qt_app.build_arrangement().by_clip_id()[bed.id].start_s == 2.5


def test_the_placement_question_is_skipped_at_the_start(qt_app):
    assert qt_app._ask_bed_placement(0.0) == 0.0


def test_an_unreadable_file_imports_nothing(qt_app, tmp_path):
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not audio")
    before = list(qt_app.document.clips)
    assert qt_app.import_music_bed(str(junk)) is None
    assert qt_app.document.clips == before
    assert not os.path.isdir(os.path.join(qt_app.project_dir, "audio", "imported"))


def test_the_bed_line_is_read_only_and_takes_no_character(qt_app, tmp_path, monkeypatch):
    document = qt_app.document
    document.text = "Intro."
    qt_app.editor.load_text(document.text)
    bed, _asked = _import(qt_app, tmp_path, monkeypatch)
    start, end = document.clip_extent(bed.id)
    editor = qt_app.editor

    assert editor.edit_touches_placeholder(start + 2, start + 2, inserting=True)
    assert editor.edit_touches_placeholder(start + 1, start + 3)
    assert not editor.edit_touches_placeholder(start, end)  # the whole line may go

    statuses = []
    monkeypatch.setattr(qt_app, "set_status", lambda message, kind="info": statuses.append(message))
    editor._push_assign_character(start, end, document.characters[0].id)
    assert document.get_clip(bed.id) is bed
    assert "audio file" in statuses[-1]
    editor._placeholder_status(start + 1, start + 3)
    assert "imported audio" in statuses[-1]


def test_the_gutter_labels_the_bed_line_audio_with_no_play_button(qt_app, tmp_path, monkeypatch):
    bed, _asked = _import(qt_app, tmp_path, monkeypatch)
    assert bed not in qt_app.document.dirty_clips()
    gutter = qt_app.editor._gutter
    gutter.paintEvent(QPaintEvent(gutter.rect()))
    assert all(clip_id != bed.id for _rect, clip_id in gutter._button_rects)
    assert gutter._label_key(qt_app.document, bed) == ("bed", bed.id)


def test_duration_and_waveform_come_from_the_file(qt_app, tmp_path, monkeypatch):
    bed, _asked = _import(qt_app, tmp_path, monkeypatch, seconds=3.0)
    assert qt_app.clip_duration_s(bed) == pytest.approx(3.0)
    samples, rate = qt_app.rendered_clip_samples(bed)
    assert len(samples) == pytest.approx(3.0 * rate, abs=2)
    placed = qt_app.build_arrangement().by_clip_id()[bed.id]
    assert placed.duration_s == pytest.approx(3.0) and not placed.estimated

    qt_app.timeline_dock.refresh()
    block = _blocks(qt_app)[bed.id]
    assert block.is_bed and block.audio_path == bed.original_audio_path
    assert block._waveform_item is not None
    assert block._label == "Theme Song"


def test_the_schedule_plays_the_bed_file_on_its_track(qt_app, tmp_path, monkeypatch):
    bed, _asked = _import(qt_app, tmp_path, monkeypatch)
    schedule, kwargs = _schedule(qt_app, monkeypatch)
    entries = [s for s in schedule if s.clip_id == bed.id]
    assert [(e.path, e.slice, e.start_s, e.duck, e.sidechain) for e in entries] == [
        (bed.original_audio_path, (0.0, 4.0), 0.0, False, False)]
    assert kwargs["duck_db"] == -12.0


def test_dragging_a_bed_edge_trims_it_and_the_schedule_slices(qt_app, tmp_path, monkeypatch):
    bed, _asked = _import(qt_app, tmp_path, monkeypatch, at_s=1.0)
    dock = qt_app.timeline_dock

    dock.on_bed_edge_dragged(bed.id, "left", 0.5)
    assert bed.overrides["trim"] == [0.5, 4.0]
    assert bed.timeline_timestamp == 1.5  # the audio stays put on the timeline
    dock.on_bed_edge_dragged(bed.id, "right", -1.0)
    assert bed.overrides["trim"] == [0.5, 3.0]

    schedule, _kwargs = _schedule(qt_app, monkeypatch)
    [entry] = [s for s in schedule if s.clip_id == bed.id]
    assert (entry.slice, entry.start_s) == ((0.5, 3.0), 1.5)
    assert qt_app.clip_duration_s(bed) == pytest.approx(2.5)

    # The left drag was one undo step: trim and start together.
    qt_app.document.undo_stack.undo()
    assert bed.overrides["trim"] == [0.5, 4.0]
    qt_app.document.undo_stack.undo()
    assert "trim" not in bed.overrides and bed.timeline_timestamp == 1.0

    dock.on_bed_edge_dragged(bed.id, "right", -1.0)
    dock.on_bed_edge_dragged(bed.id, "right", 9.0)  # can't run past the file's end
    assert bed.overrides["trim"] == [0.0, 4.0]
    dock.on_bed_action_requested(bed.id, "reset_trim")
    assert "trim" not in bed.overrides


def test_a_looping_bed_repeats_to_its_dragged_length(qt_app, tmp_path, monkeypatch):
    bed, _asked = _import(qt_app, tmp_path, monkeypatch, seconds=2.0)
    dock = qt_app.timeline_dock

    dock.on_bed_action_requested(bed.id, "loop")
    assert bed.overrides["loop"] is True
    dock.on_bed_edge_dragged(bed.id, "right", 3.0)
    assert bed.overrides["loop_length_s"] == 5.0

    schedule, _kwargs = _schedule(qt_app, monkeypatch)
    entries = [s for s in schedule if s.clip_id == bed.id]
    assert [(e.start_s, e.slice) for e in entries] == [(0.0, (0.0, 2.0)), (2.0, (0.0, 2.0)), (4.0, (0.0, 1.0))]
    assert qt_app.clip_duration_s(bed) == pytest.approx(5.0)
    qt_app.timeline_dock.refresh()
    assert len(_blocks(qt_app)[bed.id]._loop_marks_px) == 2

    dock.on_bed_action_requested(bed.id, "loop")
    assert "loop" not in bed.overrides and "loop_length_s" not in bed.overrides
    qt_app.document.undo_stack.undo()
    assert bed.overrides["loop"] is True and bed.overrides["loop_length_s"] == 5.0


def test_the_bed_menu_has_loop_and_reset_trim_and_no_generate(qt_app, tmp_path, monkeypatch):
    bed, _asked = _import(qt_app, tmp_path, monkeypatch)
    qt_app.timeline_dock.refresh()
    view = qt_app.timeline_dock.timeline_view
    block = _blocks(qt_app)[bed.id]
    menu = view._build_context_menu(view.mapFromScene(block.mapToScene(QPointF(30.0, 30.0))))
    texts = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert "Generate" not in texts and "Loop" in texts and "Reset trim" in texts and "Remove" in texts
    reset = next(a for a in menu.actions() if a.text() == "Reset trim")
    assert not reset.isEnabled()


def test_remove_deletes_the_bed_line(qt_app, tmp_path, monkeypatch):
    qt_app.document.text = "Intro."
    qt_app.editor.load_text(qt_app.document.text)
    bed, _asked = _import(qt_app, tmp_path, monkeypatch)
    qt_app.timeline_dock.on_bed_action_requested(bed.id, "remove")
    assert qt_app.document.get_clip(bed.id) is None
    assert "Theme Song" not in qt_app.document.text


def test_edge_drag_on_the_timeline_emits_bed_edge_dragged(qt_app, tmp_path, monkeypatch, qtbot):
    bed, _asked = _import(qt_app, tmp_path, monkeypatch)
    qt_app.timeline_dock.refresh()
    view = qt_app.timeline_dock.timeline_view
    block = _blocks(qt_app)[bed.id]
    received = []
    view.bedEdgeDragged.connect(lambda cid, edge, delta: received.append((cid, edge, delta)))
    monkeypatch.setattr(qt_app.timeline_dock, "on_bed_edge_dragged", lambda *a: None)

    rect = block.right_edge_rect()
    press = view.mapFromScene(block.mapToScene(QPointF(rect.center().x(), 30.0)))
    release = press + (view.mapFromScene(QPointF(view.zoom, 0)) - view.mapFromScene(QPointF(0, 0)))
    qtbot.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=press)
    qtbot.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=release)

    assert received == [(bed.id, "right", 1.0)]


def test_the_track_duck_toggle_is_one_undoable_field(qt_app, qtbot, tmp_path, monkeypatch):
    bed, _asked = _import(qt_app, tmp_path, monkeypatch)
    qt_app.timeline_dock.refresh()
    track = qt_app.document.get_track(bed.track_id)
    header = qt_app.timeline_dock.timeline_widget.header

    header.controls[track.id]["duck"].click()
    qtbot.waitUntil(lambda: track.duck is True)
    schedule, _kwargs = _schedule(qt_app, monkeypatch)
    [entry] = [s for s in schedule if s.clip_id == bed.id]
    assert entry.duck is True and entry.sidechain is False

    qt_app.document.undo_stack.undo()
    assert track.duck is False


def test_the_ducking_setting_is_a_project_field(qt_app):
    qt_app.selection.clear()
    fields = qt_app.settings_dock.scope_fields
    fields.build_project()
    spin = fields.widgets["duck_db"]
    assert spin.value() == -12.0
    spin.setValue(-18.0)
    spin.editingFinished.emit()
    assert qt_app.document.settings["duck_db"] == -18.0
    qt_app.document.undo_stack.undo()
    assert "duck_db" not in qt_app.document.settings
