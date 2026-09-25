"""Tests for kokoro_gui/qt/timeline_view.py's TimelineView/ClipBlockItem -
qtbot-only, no full qt_app (QtTTSApp) fixture, mirroring
test_waveform_view.py's app-independence, since this widget has no
dependency on the running app - it only needs a kokoro_gui.daw.models.Document."""
import numpy as np
import soundfile as sf
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QMessageBox

from kokoro_gui.daw.models import Character, Clip, Document, Run, Segment, Track
from kokoro_gui.qt.selection import SelectionModel
from kokoro_gui.daw.arrangement import compute_arrangement
from kokoro_gui.qt.timeline_view import (
    ClipBlockItem, DEFAULT_PIXELS_PER_SECOND, FX_BUTTON_HEIGHT_PX, FX_BUTTON_WIDTH_PX, LANE_HEIGHT_PX,
    MIN_CLIP_WIDTH_PX, RULER_HEIGHT_PX, TimelineView, TimelineWidget, lane_top, seconds_to_x,
)

# Every ungenerated clip is estimated at this rate, so a clip's width is
# predictable without generation_stats.json (see kokoro_gui/daw/arrangement.py).
CPS = 10.0


def _render(view, doc):
    view.render_document(doc, compute_arrangement(doc, chars_per_second=CPS))


def _write_tone_wav(path, sample_rate=8000, seconds=0.25, freq=440):
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    data = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), data, sample_rate)


def _clip_block_items(view):
    return [item for item in view._scene.items() if isinstance(item, ClipBlockItem)]


def _tagged_doc(text, tagged=(), **kwargs):
    """Builds a Document whose clips are placed at specific text offsets -
    a test-only convenience, since Document has no offsets to set directly
    any more (Claude/PLAN_text_editor_redesign.md's run-list rework). Pass
    `tagged` as `[(start, end, clip), ...]`."""
    runs = []
    cursor = 0
    for start, end, clip in sorted(tagged, key=lambda t: t[0]):
        if start > cursor:
            runs.append(Run(text=text[cursor:start]))
        runs.append(Run(text=text[start:end], clip_id=clip.id, kind=clip.source))
        cursor = end
    if cursor < len(text):
        runs.append(Run(text=text[cursor:]))
    clips = kwargs.pop("clips", None)
    if clips is None:
        clips = [clip for _start, _end, clip in tagged]
    # Placement tests here predate gaps; test_arrangement.py covers those.
    kwargs.setdefault("settings", {"gap_s": 0.0, "paragraph_gap_s": 0.0})
    return Document(runs=runs, clips=clips, **kwargs)


def test_lanes_match_track_count_and_order_index_ordering(qtbot):
    widget = TimelineWidget()
    qtbot.addWidget(widget)
    alice = Character.from_preset_dict("Alice", {})
    bob = Character.from_preset_dict("Bob", {})
    # Deliberately built out of order_index order to prove sorting, not
    # insertion order, drives lane position.
    track_bob = Track(name="Bob", character_id=bob.id, order_index=1)
    track_alice = Track(name="Alice", character_id=alice.id, order_index=0)
    clip_bob = Clip(character_id=bob.id, track_id=track_bob.id)
    clip_alice = Clip(character_id=alice.id, track_id=track_alice.id)
    doc = _tagged_doc("bob alice", [(0, 3, clip_bob), (4, 9, clip_alice)], characters=[alice, bob],
                      tracks=[track_bob, track_alice])

    widget.render_document(doc)

    labels = [item for item in widget.header._scene.items() if hasattr(item, "text") and item.text() in ("Alice", "Bob")]
    label_by_text = {label.text(): label for label in labels}
    assert label_by_text["Alice"].pos().y() < label_by_text["Bob"].pos().y()
    assert label_by_text["Alice"].pos().y() >= RULER_HEIGHT_PX


def test_a_track_without_clips_is_not_drawn(qtbot):
    """Grill PR4: an unused track stays in the model (with its mixer
    settings) but gets no lane or header row."""
    widget = TimelineWidget()
    qtbot.addWidget(widget)
    alice = Character.from_preset_dict("Alice", {})
    bob = Character.from_preset_dict("Bob", {})
    track_alice = Track(name="Alice", character_id=alice.id, order_index=0)
    track_bob = Track(name="Bob", character_id=bob.id, order_index=1, gain=0.5)
    clip = Clip(character_id=bob.id, track_id=track_bob.id)
    doc = _tagged_doc("bob", [(0, 3, clip)], characters=[alice, bob], tracks=[track_alice, track_bob])

    widget.render_document(doc, compute_arrangement(doc, chars_per_second=CPS))

    names = {item.text() for item in widget.header._scene.items() if hasattr(item, "text")}
    assert "Bob" in names and "Alice" not in names
    assert list(widget.header.controls) == [track_bob.id]
    block = widget.view._blocks_by_clip_id[clip.id]
    assert block.pos().y() < lane_top(1)  # Bob takes the first lane
    assert widget.view._track_at_y(doc, lane_top(0) + 5) is track_bob
    assert widget.view._track_at_y(doc, lane_top(1) + 5) is None
    assert track_alice in doc.tracks


def test_first_clip_starts_at_zero_seconds_and_is_as_wide_as_its_estimate(qtbot):
    view = TimelineView()
    qtbot.addWidget(view)
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="Alice", character_id=alice.id)
    clip = Clip(character_id=alice.id, track_id=track.id)
    doc = _tagged_doc("x" * 40, [(10, 30, clip)], characters=[alice], tracks=[track])

    _render(view, doc)

    blocks = _clip_block_items(view)
    assert len(blocks) == 1
    block = blocks[0]
    # Text offset 10 does not matter any more: the first clip in text order
    # starts at 0s. 20 chars at CPS=10 -> 2.0s estimated.
    assert block.pos().x() == 0
    assert block.estimated is True
    assert block.boundingRect().width() == seconds_to_x(2.0, DEFAULT_PIXELS_PER_SECOND)
    assert block.pos().y() == lane_top(0) + 8


def test_clips_are_laid_end_to_end_in_text_order_across_tracks(qtbot):
    view = TimelineView()
    qtbot.addWidget(view)
    alice = Character.from_preset_dict("Alice", {})
    bob = Character.from_preset_dict("Bob", {})
    track_a = Track(name="Alice", character_id=alice.id, order_index=0)
    track_b = Track(name="Bob", character_id=bob.id, order_index=1)
    clip_a = Clip(character_id=alice.id, track_id=track_a.id)
    clip_b = Clip(character_id=bob.id, track_id=track_b.id)
    doc = _tagged_doc("x" * 40, [(0, 10, clip_a), (10, 40, clip_b)], characters=[alice, bob], tracks=[track_a, track_b])

    _render(view, doc)

    by_id = view._blocks_by_clip_id
    assert by_id[clip_a.id].pos().x() == 0
    # clip_a is 10 chars / 10 cps = 1.0s, so clip_b starts at 1.0s on its own lane.
    assert by_id[clip_b.id].pos().x() == seconds_to_x(1.0, DEFAULT_PIXELS_PER_SECOND)
    assert by_id[clip_b.id].pos().y() == lane_top(1) + 8


def test_pinned_timestamp_positions_the_clip(qtbot):
    view = TimelineView()
    qtbot.addWidget(view)
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="Alice", character_id=alice.id)
    clip = Clip(character_id=alice.id, track_id=track.id, timeline_timestamp=3.5)
    doc = _tagged_doc("x" * 10, [(0, 10, clip)], characters=[alice], tracks=[track])

    _render(view, doc)

    assert _clip_block_items(view)[0].pos().x() == seconds_to_x(3.5, DEFAULT_PIXELS_PER_SECOND)


def test_zoom_rescales_clip_positions(qtbot):
    view = TimelineView()
    qtbot.addWidget(view)
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="Alice", character_id=alice.id)
    clip = Clip(character_id=alice.id, track_id=track.id, timeline_timestamp=2.0)
    doc = _tagged_doc("x" * 10, [(0, 10, clip)], characters=[alice], tracks=[track])
    _render(view, doc)

    view.set_zoom(100.0)

    assert view.zoom == 100.0
    assert _clip_block_items(view)[0].pos().x() == 200.0
    view.set_zoom(5.0)  # clamps to the minimum
    assert view.zoom == 20.0


def test_clip_width_floors_at_min_clip_width(qtbot):
    view = TimelineView()
    qtbot.addWidget(view)
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="Alice", character_id=alice.id)
    clip = Clip(character_id=alice.id, track_id=track.id)  # 1 char, tiny
    doc = _tagged_doc("x", [(0, 1, clip)], characters=[alice], tracks=[track])

    _render(view, doc)

    block = _clip_block_items(view)[0]
    assert block.boundingRect().width() == MIN_CLIP_WIDTH_PX


def test_clip_color_matches_character_highlight_color(qtbot):
    view = TimelineView()
    qtbot.addWidget(view)
    alice = Character.from_preset_dict("Alice", {}, highlight_color="#abcdef")
    track = Track(name="Alice", character_id=alice.id)
    clip = Clip(character_id=alice.id, track_id=track.id)
    doc = _tagged_doc("hello", [(0, 5, clip)], characters=[alice], tracks=[track])

    _render(view, doc)

    block = _clip_block_items(view)[0]
    assert block._color == "#abcdef"


def test_clip_with_unresolvable_character_uses_fallback_color(qtbot):
    view = TimelineView()
    qtbot.addWidget(view)
    track = Track(name="Orphan")
    clip = Clip(character_id="nonexistent", track_id=track.id)
    doc = _tagged_doc("hello", [(0, 5, clip)], characters=[], tracks=[track])

    _render(view, doc)

    block = _clip_block_items(view)[0]
    assert block._color == "#888888"


def test_clip_with_unresolvable_track_is_skipped_not_crashed(qtbot):
    view = TimelineView()
    qtbot.addWidget(view)
    clip = Clip(track_id="nonexistent")
    doc = _tagged_doc("hello", [(0, 5, clip)], tracks=[])

    _render(view, doc)  # must not raise

    assert _clip_block_items(view) == []


def test_bounding_rect_matches_set_geometry():
    block = ClipBlockItem()
    block.set_geometry(5, 10, 100, 50)
    rect = block.boundingRect()
    assert rect.width() == 100
    assert rect.height() == 50


def test_rerender_replaces_previous_clip_items(qtbot):
    view = TimelineView()
    qtbot.addWidget(view)
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="Alice", character_id=alice.id)
    clip_a = Clip(character_id=alice.id, track_id=track.id)
    doc = _tagged_doc("hello", [(0, 5, clip_a)], characters=[alice], tracks=[track])
    _render(view, doc)
    assert len(_clip_block_items(view)) == 1

    doc.clips = []
    _render(view, doc)

    assert _clip_block_items(view) == []


def test_clip_with_real_audio_path_renders_waveform(qtbot, tmp_path):
    view = TimelineView()
    qtbot.addWidget(view)
    wav_path = tmp_path / "tone.wav"
    _write_tone_wav(wav_path)

    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="Alice", character_id=alice.id)
    segment = Segment(audio_path=str(wav_path))
    clip = Clip(character_id=alice.id, track_id=track.id, segments=[segment])
    doc = _tagged_doc("x" * 10, [(0, 10, clip)], characters=[alice], tracks=[track])

    _render(view, doc)

    block = _clip_block_items(view)[0]
    assert block._waveform_item is not None
    assert block._waveform_item._peaks is not None


def test_clip_with_missing_audio_path_falls_back_to_flat_block(qtbot, tmp_path):
    view = TimelineView()
    qtbot.addWidget(view)
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="Alice", character_id=alice.id)
    segment = Segment(audio_path=str(tmp_path / "does_not_exist.wav"))
    clip = Clip(character_id=alice.id, track_id=track.id, segments=[segment])
    doc = _tagged_doc("x" * 10, [(0, 10, clip)], characters=[alice], tracks=[track])

    _render(view, doc)  # must not raise

    block = _clip_block_items(view)[0]
    assert block._waveform_item is None


# ---------------------------------------------------------------------------
# Play action (per-clip Generate's "hear it" follow-up)
# ---------------------------------------------------------------------------

def _menu_action_texts(menu):
    return [a.text() for a in menu.actions()]


def test_context_menu_over_clip_with_audio_shows_generate_and_play(qtbot, tmp_path):
    view = TimelineView()
    qtbot.addWidget(view)
    wav_path = tmp_path / "tone.wav"
    _write_tone_wav(wav_path)
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="Alice", character_id=alice.id)
    segment = Segment(audio_path=str(wav_path))
    clip = Clip(character_id=alice.id, track_id=track.id, segments=[segment])
    doc = _tagged_doc("x" * 10, [(0, 10, clip)], characters=[alice], tracks=[track])
    _render(view, doc)

    block = _clip_block_items(view)[0]
    pos = view.mapFromScene(block.mapToScene(0, 0))
    menu = view._build_context_menu(pos)

    # Status is always there; Align words needs audio.
    assert _menu_action_texts(menu) == ["Generate", "Play", "", "Status", "Align words"]


def test_context_menu_over_clip_without_audio_shows_generate_only(qtbot):
    view = TimelineView()
    qtbot.addWidget(view)
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="Alice", character_id=alice.id)
    clip = Clip(character_id=alice.id, track_id=track.id)
    doc = _tagged_doc("x" * 10, [(0, 10, clip)], characters=[alice], tracks=[track])
    _render(view, doc)

    block = _clip_block_items(view)[0]
    pos = view.mapFromScene(block.mapToScene(0, 0))
    menu = view._build_context_menu(pos)

    assert _menu_action_texts(menu) == ["Generate", "", "Status"]


def test_triggering_play_emits_play_clip_requested(qtbot, tmp_path):
    """Play goes through the owner's transport (seek + play) so the clip is
    heard with its read-time post-processing, not the raw segment file."""
    view = TimelineView()
    qtbot.addWidget(view)
    calls = []
    view.playClipRequested.connect(calls.append)
    wav_path = tmp_path / "tone.wav"
    _write_tone_wav(wav_path)
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="Alice", character_id=alice.id)
    segment = Segment(audio_path=str(wav_path))
    clip = Clip(character_id=alice.id, track_id=track.id, segments=[segment])
    doc = _tagged_doc("x" * 10, [(0, 10, clip)], characters=[alice], tracks=[track])
    _render(view, doc)

    block = _clip_block_items(view)[0]
    pos = view.mapFromScene(block.mapToScene(0, 0))
    menu = view._build_context_menu(pos)
    play_action = next(a for a in menu.actions() if a.text() == "Play")
    play_action.trigger()

    assert calls == [clip.id]


# ---------------------------------------------------------------------------
# Per-clip FX button (item 5, "Per-clip FX button")
# ---------------------------------------------------------------------------

def _click(view, pos, qtbot):
    qtbot.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=pos)


def _build_doc_with_one_clip(fx_override=None):
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="Alice", character_id=alice.id)
    clip = Clip(character_id=alice.id, track_id=track.id, fx_override=fx_override)
    doc = _tagged_doc("x" * 40, [(0, 10, clip)], characters=[alice], tracks=[track])
    return doc, clip, track


def test_clip_with_fx_override_renders_fx_active(qtbot):
    view = TimelineView()
    qtbot.addWidget(view)
    doc, _clip, _track = _build_doc_with_one_clip(fx_override={"reverb_enabled": True})

    _render(view, doc)

    block = _clip_block_items(view)[0]
    assert block._fx_active is True


def test_clip_without_fx_override_renders_fx_inactive(qtbot):
    view = TimelineView()
    qtbot.addWidget(view)
    doc, _clip, _track = _build_doc_with_one_clip(fx_override=None)

    _render(view, doc)

    block = _clip_block_items(view)[0]
    assert block._fx_active is False


def test_fx_button_rect_is_anchored_to_bottom_right_corner():
    block = ClipBlockItem()
    block.set_geometry(0, 0, 100, 50)

    rect = block.fx_button_rect()

    assert rect.width() == FX_BUTTON_WIDTH_PX
    assert rect.height() == FX_BUTTON_HEIGHT_PX
    assert rect.right() == 100
    assert rect.bottom() == 50


def test_fx_button_rect_clamps_to_a_block_smaller_than_the_button():
    block = ClipBlockItem()
    block.set_geometry(0, 0, MIN_CLIP_WIDTH_PX, 10)  # narrower/shorter than the FX button itself

    rect = block.fx_button_rect()

    assert rect.width() <= MIN_CLIP_WIDTH_PX
    assert rect.height() <= 10
    assert rect.x() >= 0
    assert rect.y() >= 0


def test_set_fx_active_flips_internal_attribute():
    block = ClipBlockItem()
    assert block._fx_active is False
    block.set_fx_active(True)
    assert block._fx_active is True
    block.set_fx_active(False)
    assert block._fx_active is False


def test_fx_menu_lists_preset_names_plus_clear_fx(qtbot, monkeypatch):
    view = TimelineView()
    qtbot.addWidget(view)
    monkeypatch.setattr(
        "kokoro_gui.qt.timeline_view.list_fx_preset_names", lambda: ["Warm", "Telephone"]
    )
    doc, clip, _track = _build_doc_with_one_clip()
    _render(view, doc)
    block = _clip_block_items(view)[0]

    menu = view._build_fx_menu(block)

    assert _menu_action_texts(menu) == ["Warm", "Telephone", "", "Clear FX"]  # "" is the separator


def test_fx_menu_preset_action_emits_fx_preset_requested_with_clip_id_and_name(qtbot, monkeypatch):
    view = TimelineView()
    qtbot.addWidget(view)
    monkeypatch.setattr("kokoro_gui.qt.timeline_view.list_fx_preset_names", lambda: ["Warm"])
    doc, clip, _track = _build_doc_with_one_clip()
    _render(view, doc)
    block = _clip_block_items(view)[0]

    menu = view._build_fx_menu(block)
    received = []
    view.fxPresetRequested.connect(lambda cid, name: received.append((cid, name)))
    next(a for a in menu.actions() if a.text() == "Warm").trigger()

    assert received == [(clip.id, "Warm")]


def test_fx_menu_clear_fx_action_emits_fx_preset_requested_with_empty_name(qtbot, monkeypatch):
    view = TimelineView()
    qtbot.addWidget(view)
    monkeypatch.setattr("kokoro_gui.qt.timeline_view.list_fx_preset_names", lambda: [])
    doc, clip, _track = _build_doc_with_one_clip(fx_override={"reverb_enabled": True})
    _render(view, doc)
    block = _clip_block_items(view)[0]

    menu = view._build_fx_menu(block)
    received = []
    view.fxPresetRequested.connect(lambda cid, name: received.append((cid, name)))
    next(a for a in menu.actions() if a.text() == "Clear FX").trigger()

    assert received == [(clip.id, "")]


def test_click_within_fx_button_rect_bypasses_click_to_select(qtbot, monkeypatch):
    selection = SelectionModel()
    view = TimelineView(selection_model=selection)
    qtbot.addWidget(view)
    doc, clip, _track = _build_doc_with_one_clip()
    _render(view, doc)
    block = _clip_block_items(view)[0]

    handled = []
    monkeypatch.setattr(view, "_handle_fx_button_click", lambda blk, pos: handled.append(blk.clip_id))

    fx_rect = block.fx_button_rect()
    pos = view.mapFromScene(block.mapToScene(fx_rect.center()))
    _click(view, pos, qtbot)

    assert handled == [clip.id]
    assert selection.selected_clip_id is None  # click-to-select did NOT run for this click
    assert selection.kind == "none"


# ---------------------------------------------------------------------------
# Real time-based positioning (item 6, "Real time-based positioning")
# ---------------------------------------------------------------------------

def test_clip_with_segments_but_no_audio_is_still_estimated(qtbot):
    """Segments that exist but carry no audio_path (not yet generated) are
    treated as ungenerated: estimated width, dashed outline."""
    view = TimelineView()
    qtbot.addWidget(view)
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="Alice", character_id=alice.id)
    segments = [Segment(duration=None), Segment(duration=None)]
    clip = Clip(character_id=alice.id, track_id=track.id, segments=segments)
    doc = _tagged_doc("x" * 40, [(10, 30, clip)], characters=[alice], tracks=[track])

    _render(view, doc)

    block = _clip_block_items(view)[0]
    assert block.estimated is True
    assert block.boundingRect().width() == seconds_to_x(2.0, DEFAULT_PIXELS_PER_SECOND)


def test_generated_clip_width_is_its_real_duration(qtbot, tmp_path):
    view = TimelineView()
    qtbot.addWidget(view)
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="Alice", character_id=alice.id)
    wav = tmp_path / "a.wav"
    _write_tone_wav(wav)
    segment = Segment(duration=2.0, audio_path=str(wav))
    clip = Clip(character_id=alice.id, track_id=track.id, segments=[segment])
    doc = _tagged_doc("x" * 5, [(0, 5, clip)], characters=[alice], tracks=[track])

    _render(view, doc)

    block = _clip_block_items(view)[0]
    assert block.estimated is False
    assert block.boundingRect().width() == seconds_to_x(2.0, DEFAULT_PIXELS_PER_SECOND)


def test_generated_clip_width_sums_multiple_segment_durations(qtbot, tmp_path):
    view = TimelineView()
    qtbot.addWidget(view)
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="Alice", character_id=alice.id)
    wav = tmp_path / "a.wav"
    _write_tone_wav(wav)
    segments = [Segment(duration=1.0, audio_path=str(wav)), Segment(duration=2.0, audio_path=str(wav)),
                Segment(duration=0.5, audio_path=str(wav))]
    clip = Clip(character_id=alice.id, track_id=track.id, segments=segments)
    doc = _tagged_doc("x" * 10, [(0, 10, clip)], characters=[alice], tracks=[track])

    _render(view, doc)

    block = _clip_block_items(view)[0]
    assert block.boundingRect().width() == seconds_to_x(3.5, DEFAULT_PIXELS_PER_SECOND)


def test_generated_clip_skips_none_duration_segments_without_crashing(qtbot, tmp_path):
    view = TimelineView()
    qtbot.addWidget(view)
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="Alice", character_id=alice.id)
    wav = tmp_path / "a.wav"
    _write_tone_wav(wav)
    segments = [Segment(duration=1.0, audio_path=str(wav)), Segment(duration=None),
                Segment(duration=2.0, audio_path=str(wav))]
    clip = Clip(character_id=alice.id, track_id=track.id, segments=segments)
    doc = _tagged_doc("x" * 10, [(0, 10, clip)], characters=[alice], tracks=[track])

    _render(view, doc)  # must not raise (summing a None duration)

    block = _clip_block_items(view)[0]
    assert block.boundingRect().width() == seconds_to_x(3.0, DEFAULT_PIXELS_PER_SECOND)


def test_overlapping_clips_paint_in_ascending_start_offset_order(qtbot):
    """Clips are added to the scene in ascending start_offset order,
    regardless of document.clips's incidental list order - so the
    later-starting clip is always painted on top of an earlier, overlong one
    (deterministic paint order, not incidental insertion-list order)."""
    view = TimelineView()
    qtbot.addWidget(view)
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="Alice", character_id=alice.id)
    # Pinned so they overlap on the seconds axis (the default layout is
    # end-to-end, which never overlaps).
    clip_a = Clip(character_id=alice.id, track_id=track.id, timeline_timestamp=0.5)
    clip_b = Clip(character_id=alice.id, track_id=track.id, timeline_timestamp=0.0)
    doc = _tagged_doc(
        "x" * 100, [(50, 60, clip_a), (10, 20, clip_b)],
        characters=[alice], tracks=[track], clips=[],
    )
    # Deliberately appended in descending start_offset order: clip_a (larger
    # start_offset) first, clip_b (smaller start_offset) after - proving
    # render order follows start_offset, not list/insertion order.
    doc.clips.append(clip_a)
    doc.clips.append(clip_b)

    _render(view, doc)

    blocks = _clip_block_items(view)
    assert len(blocks) == 2
    # QGraphicsScene.items() returns items topmost-first; clip_a (the later-
    # starting clip, added to the scene last) must be on top.
    assert blocks[0].clip_id == clip_a.id
    assert blocks[1].clip_id == clip_b.id


def test_click_outside_fx_button_rect_still_selects_the_block(qtbot, monkeypatch):
    selection = SelectionModel()
    view = TimelineView(selection_model=selection)
    qtbot.addWidget(view)
    doc, clip, _track = _build_doc_with_one_clip()
    _render(view, doc)
    block = _clip_block_items(view)[0]

    handled = []
    monkeypatch.setattr(view, "_handle_fx_button_click", lambda blk, pos: handled.append(blk.clip_id))

    # Top-left corner of the block - well outside the bottom-right FX rect.
    pos = view.mapFromScene(block.mapToScene(2, 2))
    _click(view, pos, qtbot)

    assert handled == []
    assert selection.selected_clip_id == clip.id
    assert selection.kind == "clip"


# ---------------------------------------------------------------------------
# Drag-to-reassign a clip to a different track (item 8)
# ---------------------------------------------------------------------------

def _build_doc_two_tracks_same_character():
    alice = Character.from_preset_dict("Alice", {})
    track_a = Track(name="Alice A", character_id=alice.id, order_index=0)
    track_b = Track(name="Alice B", character_id=alice.id, order_index=1)
    clip = Clip(character_id=alice.id, track_id=track_a.id)
    # Only a track with clips is drawn (grill PR4), so lane B holds one too,
    # far to the right of where the drags land.
    other = Clip(character_id=alice.id, track_id=track_b.id, timeline_timestamp=20.0)
    doc = _tagged_doc("x" * 40, [(0, 10, clip), (30, 40, other)], characters=[alice], tracks=[track_a, track_b])
    return doc, clip, track_a, track_b


def _press_release(view, qtbot, press_pos, release_pos, modifier=Qt.KeyboardModifier.NoModifier):
    qtbot.mousePress(view.viewport(), Qt.MouseButton.LeftButton, modifier, pos=press_pos)
    qtbot.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, modifier, pos=release_pos)


SHIFT = Qt.KeyboardModifier.ShiftModifier


def test_plain_click_on_clip_block_still_selects_it_unchanged(qtbot):
    """Regression check (sanity check #1, before any drag-specific test):
    a plain click - press and release at the exact same position - must
    still select the clip exactly like item 1 already established, now that
    press events also record drag-tracking state."""
    selection = SelectionModel()
    view = TimelineView(selection_model=selection)
    qtbot.addWidget(view)
    doc, clip, _track = _build_doc_with_one_clip()
    _render(view, doc)
    block = _clip_block_items(view)[0]

    pos = view.mapFromScene(block.mapToScene(2, 2))
    _press_release(view, qtbot, pos, pos)

    assert selection.selected_clip_id == clip.id
    assert selection.kind == "clip"


def test_plain_click_on_fx_button_still_opens_fx_menu_unchanged(qtbot, monkeypatch):
    """Regression check (sanity check #2): a plain click on the FX button
    rect must still open the FX menu instead of engaging click-to-select or
    a drag, exactly like item 5 already established."""
    selection = SelectionModel()
    view = TimelineView(selection_model=selection)
    qtbot.addWidget(view)
    doc, clip, _track = _build_doc_with_one_clip()
    _render(view, doc)
    block = _clip_block_items(view)[0]

    handled = []
    monkeypatch.setattr(view, "_handle_fx_button_click", lambda blk, pos: handled.append(blk.clip_id))

    fx_rect = block.fx_button_rect()
    pos = view.mapFromScene(block.mapToScene(fx_rect.center()))
    _press_release(view, qtbot, pos, pos)

    assert handled == [clip.id]
    assert selection.selected_clip_id is None


def test_sub_threshold_movement_behaves_as_plain_click_not_a_drag(qtbot):
    """A press+release with real movement, but below _drag_threshold_px,
    must not be treated as a drag."""
    selection = SelectionModel()
    view = TimelineView(selection_model=selection)
    qtbot.addWidget(view)
    doc, clip, track = _build_doc_with_one_clip()
    _render(view, doc)
    block = _clip_block_items(view)[0]

    received = []
    view.clipDragReassigned.connect(lambda *a: received.append(a))

    press_pos = view.mapFromScene(block.mapToScene(2, 2))
    assert view._drag_threshold_px == 8
    release_pos = press_pos + type(press_pos)(3, 0)  # 3px, sub-threshold
    _press_release(view, qtbot, press_pos, release_pos)

    assert received == []
    assert selection.selected_clip_id == clip.id  # click-to-select still ran on press
    assert clip.track_id == track.id  # untouched


def test_drag_release_on_different_track_with_matching_character_emits_signal_no_dialog(qtbot, monkeypatch):
    """No ambiguity (target lane's character already matches the clip's) -
    must dispatch without ever constructing/exec'ing a QMessageBox."""
    selection = SelectionModel()
    view = TimelineView(selection_model=selection)
    qtbot.addWidget(view)
    doc, clip, track_a, track_b = _build_doc_two_tracks_same_character()
    _render(view, doc)
    block = view._blocks_by_clip_id[clip.id]

    def _fail_exec(self):
        raise AssertionError("QMessageBox.exec must not be called when there's no ambiguity")
    monkeypatch.setattr(QMessageBox, "exec", _fail_exec)

    received = []
    view.clipDragReassigned.connect(lambda *a: received.append(a))

    press_pos = view.mapFromScene(block.mapToScene(2, 2))
    release_scene_x = block.mapToScene(2, 2).x()
    release_pos = view.mapFromScene(QPointF(release_scene_x, lane_top(1) + 10))
    _press_release(view, qtbot, press_pos, release_pos)

    assert received == [(clip.id, track_b.id, False)]


def test_drag_release_on_same_track_emits_clip_moved_not_reassigned(qtbot):
    """UI9: a horizontal drag on the same lane is a move on the seconds
    axis. The view reports the new start; the dock decides timestamp vs.
    text reorder."""
    selection = SelectionModel()
    view = TimelineView(selection_model=selection)
    qtbot.addWidget(view)
    doc, clip, track_a, _track_b = _build_doc_two_tracks_same_character()
    _render(view, doc)
    block = view._blocks_by_clip_id[clip.id]

    received = []
    moved = []
    view.clipDragReassigned.connect(lambda *a: received.append(a))
    view.clipMoved.connect(lambda *a: moved.append(a))

    press_pos = view.mapFromScene(block.mapToScene(2, 2))
    release_pos = view.mapFromScene(block.mapToScene(60, 2))  # 58px right = 1.16s at 50px/s
    _press_release(view, qtbot, press_pos, release_pos)

    assert received == []
    assert clip.track_id == track_a.id
    assert len(moved) == 1
    assert moved[0][0] == clip.id
    assert abs(moved[0][1] - 58 / DEFAULT_PIXELS_PER_SECOND) < 0.05


def test_ruler_click_emits_seek(qtbot):
    view = TimelineView()
    qtbot.addWidget(view)
    doc, _clip, _track = _build_doc_with_one_clip()
    _render(view, doc)

    seeks = []
    view.seekRequested.connect(seeks.append)
    pos = view.mapFromScene(QPointF(seconds_to_x(2.0, DEFAULT_PIXELS_PER_SECOND), RULER_HEIGHT_PX / 2))
    _click(view, pos, qtbot)

    assert len(seeks) == 1
    assert abs(seeks[0] - 2.0) < 0.05


def test_set_playhead_shows_line_at_the_right_x(qtbot):
    view = TimelineView()
    qtbot.addWidget(view)
    doc, _clip, _track = _build_doc_with_one_clip()
    _render(view, doc)

    view.set_playhead(1.5)

    line = view._playhead_item
    assert line.isVisible()
    assert line.line().x1() == seconds_to_x(1.5, DEFAULT_PIXELS_PER_SECOND)
    view.set_playhead(None)
    assert not line.isVisible()


def test_drag_release_off_all_lanes_is_a_noop(qtbot):
    selection = SelectionModel()
    view = TimelineView(selection_model=selection)
    qtbot.addWidget(view)
    doc, clip, track_a, _track_b = _build_doc_two_tracks_same_character()
    _render(view, doc)
    block = view._blocks_by_clip_id[clip.id]

    received = []
    view.clipDragReassigned.connect(lambda *a: received.append(a))

    press_pos = view.mapFromScene(block.mapToScene(2, 2))
    release_scene_x = block.mapToScene(2, 2).x()
    release_pos = view.mapFromScene(QPointF(release_scene_x, LANE_HEIGHT_PX * 100))
    _press_release(view, qtbot, press_pos, release_pos)  # must not crash

    assert received == []
    assert clip.track_id == track_a.id


# ---------------------------------------------------------------------------
# Sub-range TTS replacement (item 9): Shift+drag that starts and ends within
# one clip block's own x-range. Behind Shift since the UI shell redesign, so
# a plain drag can mean "move on the seconds axis" (clipMoved).
# ---------------------------------------------------------------------------

def test_same_track_drag_within_one_block_emits_sub_range_tts_left_to_right(qtbot):
    selection = SelectionModel()
    view = TimelineView(selection_model=selection)
    qtbot.addWidget(view)
    doc, clip, _track = _build_doc_with_one_clip()  # start=0, end=10, block width 40px
    _render(view, doc)
    block = _clip_block_items(view)[0]

    received = []
    view.subRangeTtsRequested.connect(lambda *a: received.append(a))

    press_pos = view.mapFromScene(block.mapToScene(4, 2))
    release_pos = view.mapFromScene(block.mapToScene(20, 2))
    _press_release(view, qtbot, press_pos, release_pos, SHIFT)

    assert len(received) == 1
    cid, sub_start, sub_end = received[0]
    assert cid == clip.id
    clip_start, clip_end = doc.clip_extent(clip.id)
    assert clip_start <= sub_start < sub_end <= clip_end


def test_same_track_drag_within_one_block_emits_sub_range_tts_right_to_left(qtbot):
    """A right-to-left drag must produce the same, correctly-ordered
    (sub_start < sub_end) result as the equivalent left-to-right drag."""
    selection = SelectionModel()
    view = TimelineView(selection_model=selection)
    qtbot.addWidget(view)
    doc, clip, _track = _build_doc_with_one_clip()
    _render(view, doc)
    block = _clip_block_items(view)[0]

    received = []
    view.subRangeTtsRequested.connect(lambda *a: received.append(a))

    press_pos = view.mapFromScene(block.mapToScene(20, 2))
    release_pos = view.mapFromScene(block.mapToScene(4, 2))
    _press_release(view, qtbot, press_pos, release_pos, SHIFT)

    assert len(received) == 1
    cid, sub_start, sub_end = received[0]
    assert cid == clip.id
    assert sub_start < sub_end
    clip_start, clip_end = doc.clip_extent(clip.id)
    assert clip_start <= sub_start < sub_end <= clip_end


def test_shift_drag_exiting_block_bounds_emits_nothing(qtbot):
    """A Shift+drag whose x-coordinates exit the origin block's own bounds
    is neither a sub-range selection nor a move."""
    selection = SelectionModel()
    view = TimelineView(selection_model=selection)
    qtbot.addWidget(view)
    doc, clip, track_a, _track_b = _build_doc_two_tracks_same_character()
    _render(view, doc)
    block = view._blocks_by_clip_id[clip.id]

    drag_received = []
    sub_range_received = []
    view.clipDragReassigned.connect(lambda *a: drag_received.append(a))
    view.subRangeTtsRequested.connect(lambda *a: sub_range_received.append(a))

    press_pos = view.mapFromScene(block.mapToScene(2, 2))
    release_pos = view.mapFromScene(block.mapToScene(60, 2))  # past the block's own 40px width
    _press_release(view, qtbot, press_pos, release_pos, SHIFT)

    assert drag_received == []
    assert sub_range_received == []
    assert clip.track_id == track_a.id


def test_different_track_drag_emits_drag_reassigned_not_sub_range_tts(qtbot):
    """Regression check: a drag ending on a different track must still
    trigger item 8's clipDragReassigned flow, not this new gesture - the two
    must never fire for the same drag."""
    selection = SelectionModel()
    view = TimelineView(selection_model=selection)
    qtbot.addWidget(view)
    doc, clip, _track_a, track_b = _build_doc_two_tracks_same_character()
    _render(view, doc)
    block = view._blocks_by_clip_id[clip.id]

    drag_received = []
    sub_range_received = []
    view.clipDragReassigned.connect(lambda *a: drag_received.append(a))
    view.subRangeTtsRequested.connect(lambda *a: sub_range_received.append(a))

    press_pos = view.mapFromScene(block.mapToScene(2, 2))
    release_scene_x = block.mapToScene(2, 2).x()
    release_pos = view.mapFromScene(QPointF(release_scene_x, lane_top(1) + 10))
    _press_release(view, qtbot, press_pos, release_pos)

    assert drag_received == [(clip.id, track_b.id, False)]
    assert sub_range_received == []


def test_sub_range_drag_too_small_to_select_a_character_emits_nothing(qtbot):
    """A drag whose x-coordinates round to the same document-text offset
    (too small to select any real character range) must emit nothing at
    all - not even a zero-width subRangeTtsRequested. Vertical movement is
    used to clear the drag-distance threshold while keeping the horizontal
    movement within one character's rounding tolerance and staying on the
    clip's own (single) track lane."""
    selection = SelectionModel()
    view = TimelineView(selection_model=selection)
    qtbot.addWidget(view)
    doc, clip, _track = _build_doc_with_one_clip()
    _render(view, doc)
    block = _clip_block_items(view)[0]

    sub_range_received = []
    drag_received = []
    view.subRangeTtsRequested.connect(lambda *a: sub_range_received.append(a))
    view.clipDragReassigned.connect(lambda *a: drag_received.append(a))

    press_pos = view.mapFromScene(block.mapToScene(4, 2))
    release_pos = view.mapFromScene(block.mapToScene(5, 70))
    _press_release(view, qtbot, press_pos, release_pos, SHIFT)

    assert sub_range_received == []
    assert drag_received == []


# --- phase 2: track header, fades, markers, loop, automation, filter, takes -----


def _generated_doc(tmp_path):
    """One generated 1 s clip on one track (so it's not estimated and has
    fade handles)."""
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="Alice", character_id=alice.id)
    path = tmp_path / "a.wav"
    _write_tone_wav(path, seconds=1.0)
    clip = Clip(character_id=alice.id, track_id=track.id,
                segments=[Segment(order_index=0, text="x" * 10, audio_path=str(path), duration=1.0)])
    doc = _tagged_doc("x" * 10, [(0, 10, clip)], characters=[alice], tracks=[track])
    return doc, clip, track


def test_header_controls_emit_track_field_changes(qtbot):
    widget = TimelineWidget()
    qtbot.addWidget(widget)
    doc, _clip, track = _build_doc_with_one_clip()
    widget.render_document(doc, compute_arrangement(doc, chars_per_second=CPS))
    received = []
    widget.header.trackFieldChanged.connect(lambda tid, field, value: received.append((tid, field, value)))

    controls = widget.header.controls[track.id]
    controls["mute"].click()
    controls["solo"].click()
    controls["gain"].setValue(50)
    controls["gain"].sliderReleased.emit()
    controls["pan"].setValue(3)  # inside the centre detent
    controls["pan"].sliderReleased.emit()

    assert received == [(track.id, "mute", True), (track.id, "solo", True), (track.id, "gain", 0.5),
                        (track.id, "pan", 0.0)]
    assert controls["pan"].value() == 0


def test_header_a_toggle_shows_the_automation_lane(qtbot):
    widget = TimelineWidget()
    qtbot.addWidget(widget)
    doc, _clip, track = _build_doc_with_one_clip()
    track.automation = [[0.0, 1.0], [1.0, 0.5]]
    widget.render_document(doc, compute_arrangement(doc, chars_per_second=CPS))

    widget.header.controls[track.id]["auto"].click()

    lane = widget.view.automation_item(track.id)
    assert lane is not None
    assert lane.points == [[0.0, 1.0], [1.0, 0.5]]
    assert widget.header.controls[track.id]["auto"].isChecked()


def test_dragging_the_fade_in_handle_emits_fade_changed(qtbot, tmp_path):
    view = TimelineView()
    qtbot.addWidget(view)
    view.resize(800, 300)
    doc, clip, _track = _generated_doc(tmp_path)
    _render(view, doc)
    block = _clip_block_items(view)[0]
    received = []
    view.fadeChanged.connect(lambda cid, field, seconds: received.append((cid, field, seconds)))

    press = view.mapFromScene(block.mapToScene(block.fade_in_handle_rect().center()))
    release = view.mapFromScene(block.mapToScene(QPointF(25.0, 4.0)))  # 25 px = 0.5 s at 50 px/s
    _press_release(view, qtbot, press, release)

    assert received == [(clip.id, "fade_in_s", 0.5)]


def test_fades_draw_as_handle_positions(qtbot, tmp_path):
    view = TimelineView()
    qtbot.addWidget(view)
    doc, clip, _track = _generated_doc(tmp_path)
    clip.fade_out_s = 0.2
    _render(view, doc)
    block = _clip_block_items(view)[0]
    width = block.boundingRect().width()
    assert block.fade_out_handle_rect().right() == width - seconds_to_x(0.2, DEFAULT_PIXELS_PER_SECOND)


def test_ruler_menu_adds_a_marker_and_a_flag_menu_offers_rename_and_delete(qtbot):
    from kokoro_gui.daw import markers

    view = TimelineView()
    qtbot.addWidget(view)
    doc, _clip, _track = _build_doc_with_one_clip()
    _render(view, doc)
    added = []
    view.markerAddRequested.connect(added.append)

    menu = view._build_ruler_menu(seconds_to_x(2.0, DEFAULT_PIXELS_PER_SECOND))
    assert [a.text() for a in menu.actions()] == ["Add marker here"]
    menu.actions()[0].trigger()
    assert added == [2.0]

    doc.settings["markers"], m = markers.add_marker(doc.settings, 2.0, name="Intro")
    doc.settings["markers"], _later = markers.add_marker(doc.settings, 4.0)
    _render(view, doc)
    menu = view._build_ruler_menu(seconds_to_x(2.0, DEFAULT_PIXELS_PER_SECOND) + 2)
    assert [a.text() for a in menu.actions()] == ["Rename marker...", "Delete marker", "Loop to next marker"]
    loops = []
    view.loopRangeRequested.connect(lambda a, b: loops.append((a, b)))
    menu.actions()[2].trigger()
    assert loops == [(2.0, 4.0)]


def test_dragging_a_marker_flag_moves_it_and_shift_drag_sets_a_loop(qtbot):
    from kokoro_gui.daw import markers

    view = TimelineView()
    qtbot.addWidget(view)
    view.resize(800, 300)
    doc, _clip, _track = _build_doc_with_one_clip()
    doc.settings["markers"], m = markers.add_marker(doc.settings, 1.0)
    _render(view, doc)
    moved, loops = [], []
    view.markerMoved.connect(lambda mid, s: moved.append((mid, s)))
    view.loopRangeRequested.connect(lambda a, b: loops.append((a, b)))

    y = RULER_HEIGHT_PX / 2
    press = view.mapFromScene(QPointF(seconds_to_x(1.0, DEFAULT_PIXELS_PER_SECOND), y))
    release = view.mapFromScene(QPointF(seconds_to_x(3.0, DEFAULT_PIXELS_PER_SECOND), y))
    _press_release(view, qtbot, press, release)
    assert moved == [(m["id"], 3.0)]

    press = view.mapFromScene(QPointF(seconds_to_x(5.0, DEFAULT_PIXELS_PER_SECOND), y))
    release = view.mapFromScene(QPointF(seconds_to_x(7.0, DEFAULT_PIXELS_PER_SECOND), y))
    _press_release(view, qtbot, press, release, SHIFT)
    assert loops == [(5.0, 7.0)]


def test_automation_double_click_adds_a_point_and_right_click_deletes_it(qtbot):
    widget = TimelineWidget()
    qtbot.addWidget(widget)
    widget.resize(900, 300)
    view = widget.view
    doc, _clip, track = _build_doc_with_one_clip()
    widget.render_document(doc, compute_arrangement(doc, chars_per_second=CPS))
    view.set_automation_visible(track.id, True)
    changes = []
    view.automationChanged.connect(lambda tid, points: changes.append((tid, points)))

    lane = view.automation_item(track.id)
    scene = QPointF(seconds_to_x(2.0, DEFAULT_PIXELS_PER_SECOND), lane.gain_to_y(1.0))
    qtbot.mouseDClick(view.viewport(), Qt.MouseButton.LeftButton, pos=view.mapFromScene(scene))
    assert changes[-1][0] == track.id
    assert changes[-1][1] == [[2.0, 1.0]]

    track.automation = [[2.0, 1.0]]
    widget.render_document(doc, compute_arrangement(doc, chars_per_second=CPS))
    assert view._delete_automation_point_at(scene)
    assert changes[-1][1] == []


def test_dragging_an_automation_point_is_clamped_between_its_neighbours(qtbot):
    widget = TimelineWidget()
    qtbot.addWidget(widget)
    widget.resize(900, 300)
    view = widget.view
    doc, _clip, track = _build_doc_with_one_clip()
    track.automation = [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    view.set_automation_visible(track.id, True)
    widget.render_document(doc, compute_arrangement(doc, chars_per_second=CPS))
    changes = []
    view.automationChanged.connect(lambda tid, points: changes.append(points))

    lane = view.automation_item(track.id)
    press = view.mapFromScene(lane.point_pos([2.0, 1.0]))
    release = view.mapFromScene(QPointF(seconds_to_x(5.0, DEFAULT_PIXELS_PER_SECOND), lane.gain_to_y(2.0)))
    _press_release(view, qtbot, press, release)

    assert changes[-1][1] == [3.0, 2.0]


def test_status_filter_dims_blocks_that_do_not_match(qtbot):
    view = TimelineView()
    qtbot.addWidget(view)
    doc, clip, _track = _build_doc_with_one_clip()
    clip.status = "approved"
    _render(view, doc)
    block = _clip_block_items(view)[0]

    view.set_status_filter("not_approved")
    assert block.opacity() < 1.0
    view.set_status_filter("all")
    assert block.opacity() == 1.0


def test_take_menu_lists_takes_and_marks_old_text(qtbot):
    view = TimelineView()
    qtbot.addWidget(view)
    doc, clip, _track = _build_doc_with_one_clip()
    clip.segments = [Segment(text="x" * 10, audio_path="t1.wav", duration=1.0)]
    clip.overrides["take"] = 1
    clip.takes = {0: [Segment(text="something else", audio_path="t0.wav", duration=2.0)]}
    _render(view, doc)
    block = _clip_block_items(view)[0]
    menu = view._build_context_menu(view.mapFromScene(block.mapToScene(10, 30)))

    take_menu = next(a.menu() for a in menu.actions() if a.text() == "Take")
    labels = [a.text() for a in take_menu.actions()]
    assert labels == ["Take 1 (2.0s) - old text", "Take 2 (1.0s)"]
    picked = []
    view.takeSelected.connect(lambda cid, i: picked.append((cid, i)))
    take_menu.actions()[0].trigger()
    assert picked == [(clip.id, 0)]


def test_ruler_labels_in_timecode_when_enabled(qtbot):
    view = TimelineView()
    qtbot.addWidget(view)
    doc, _clip, _track = _build_doc_with_one_clip()
    doc.settings["timecode"] = {"enabled": True, "frame_rate": 25.0, "start": "01:00:00:00"}
    _render(view, doc)
    assert view._ruler.label_for(2.0) == "01:00:02:00"
