"""Tests for kokoro_gui/qt/timeline_view.py's TimelineView/ClipBlockItem -
qtbot-only, no full qt_app (QtTTSApp) fixture, mirroring
test_waveform_view.py's app-independence, since this widget has no
dependency on the running app - it only needs a kokoro_gui.daw.models.Document."""
import numpy as np
import soundfile as sf

from kokoro_gui.daw.models import Character, Clip, Document, Segment, Track
from kokoro_gui.qt.timeline_view import (
    ClipBlockItem, MIN_CLIP_WIDTH_PX, PLACEHOLDER_PIXELS_PER_CHAR, TimelineView,
)


def _write_tone_wav(path, sample_rate=8000, seconds=0.25, freq=440):
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    data = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), data, sample_rate)


def _clip_block_items(view):
    return [item for item in view._scene.items() if isinstance(item, ClipBlockItem)]


def test_lanes_match_track_count_and_order_index_ordering(qtbot):
    view = TimelineView()
    qtbot.addWidget(view)
    alice = Character.from_preset_dict("Alice", {})
    bob = Character.from_preset_dict("Bob", {})
    # Deliberately built out of order_index order to prove sorting, not
    # insertion order, drives lane position.
    track_bob = Track(name="Bob", character_id=bob.id, order_index=1)
    track_alice = Track(name="Alice", character_id=alice.id, order_index=0)
    doc = Document(text="", characters=[alice, bob], tracks=[track_bob, track_alice])

    view.render_document(doc)

    labels = [item for item in view._scene.items() if hasattr(item, "text") and item.text() in ("Alice", "Bob")]
    label_by_text = {label.text(): label for label in labels}
    assert label_by_text["Alice"].pos().y() < label_by_text["Bob"].pos().y()


def test_clip_position_and_width_match_offsets(qtbot):
    view = TimelineView()
    qtbot.addWidget(view)
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="Alice", character_id=alice.id)
    clip = Clip(start_offset=10, end_offset=30, character_id=alice.id, track_id=track.id)
    doc = Document(text="x" * 40, characters=[alice], tracks=[track], clips=[clip])

    view.render_document(doc)

    blocks = _clip_block_items(view)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.pos().x() == 10 * PLACEHOLDER_PIXELS_PER_CHAR
    assert block.boundingRect().width() == 20 * PLACEHOLDER_PIXELS_PER_CHAR


def test_clip_width_floors_at_min_clip_width(qtbot):
    view = TimelineView()
    qtbot.addWidget(view)
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="Alice", character_id=alice.id)
    clip = Clip(start_offset=0, end_offset=1, character_id=alice.id, track_id=track.id)  # 1 char, tiny
    doc = Document(text="x", characters=[alice], tracks=[track], clips=[clip])

    view.render_document(doc)

    block = _clip_block_items(view)[0]
    assert block.boundingRect().width() == MIN_CLIP_WIDTH_PX


def test_clip_color_matches_character_highlight_color(qtbot):
    view = TimelineView()
    qtbot.addWidget(view)
    alice = Character.from_preset_dict("Alice", {}, highlight_color="#abcdef")
    track = Track(name="Alice", character_id=alice.id)
    clip = Clip(start_offset=0, end_offset=5, character_id=alice.id, track_id=track.id)
    doc = Document(text="hello", characters=[alice], tracks=[track], clips=[clip])

    view.render_document(doc)

    block = _clip_block_items(view)[0]
    assert block._color == "#abcdef"


def test_clip_with_unresolvable_character_uses_fallback_color(qtbot):
    view = TimelineView()
    qtbot.addWidget(view)
    track = Track(name="Orphan")
    clip = Clip(start_offset=0, end_offset=5, character_id="nonexistent", track_id=track.id)
    doc = Document(text="hello", characters=[], tracks=[track], clips=[clip])

    view.render_document(doc)

    block = _clip_block_items(view)[0]
    assert block._color == "#888888"


def test_clip_with_unresolvable_track_is_skipped_not_crashed(qtbot):
    view = TimelineView()
    qtbot.addWidget(view)
    clip = Clip(start_offset=0, end_offset=5, track_id="nonexistent")
    doc = Document(text="hello", tracks=[], clips=[clip])

    view.render_document(doc)  # must not raise

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
    clip_a = Clip(start_offset=0, end_offset=5, character_id=alice.id, track_id=track.id)
    doc = Document(text="hello", characters=[alice], tracks=[track], clips=[clip_a])
    view.render_document(doc)
    assert len(_clip_block_items(view)) == 1

    doc.clips = []
    view.render_document(doc)

    assert _clip_block_items(view) == []


def test_clip_with_real_audio_path_renders_waveform(qtbot, tmp_path):
    view = TimelineView()
    qtbot.addWidget(view)
    wav_path = tmp_path / "tone.wav"
    _write_tone_wav(wav_path)

    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="Alice", character_id=alice.id)
    segment = Segment(audio_path=str(wav_path))
    clip = Clip(start_offset=0, end_offset=10, character_id=alice.id, track_id=track.id, segments=[segment])
    doc = Document(text="x" * 10, characters=[alice], tracks=[track], clips=[clip])

    view.render_document(doc)

    block = _clip_block_items(view)[0]
    assert block._waveform_item is not None
    assert block._waveform_item._peaks is not None


def test_clip_with_missing_audio_path_falls_back_to_flat_block(qtbot, tmp_path):
    view = TimelineView()
    qtbot.addWidget(view)
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="Alice", character_id=alice.id)
    segment = Segment(audio_path=str(tmp_path / "does_not_exist.wav"))
    clip = Clip(start_offset=0, end_offset=10, character_id=alice.id, track_id=track.id, segments=[segment])
    doc = Document(text="x" * 10, characters=[alice], tracks=[track], clips=[clip])

    view.render_document(doc)  # must not raise

    block = _clip_block_items(view)[0]
    assert block._waveform_item is None
