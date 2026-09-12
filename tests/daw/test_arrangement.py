"""Tests for kokoro_gui/daw/arrangement.py - where clips sit on the seconds
axis (UI9/UI11 of Claude/PLAN_ui_shell_redesign.md). Pure Python."""
from kokoro_gui.daw.arrangement import (
    FALLBACK_CHARS_PER_SECOND, compute_arrangement, estimate_duration_s, text_order_predecessor,
)
from kokoro_gui.daw.models import Character, Clip, Document, Run, Segment, Track


def _doc(text, tagged, **kwargs):
    runs, cursor = [], 0
    for start, end, clip in sorted(tagged, key=lambda t: t[0]):
        if start > cursor:
            runs.append(Run(text=text[cursor:start]))
        runs.append(Run(text=text[start:end], clip_id=clip.id, kind=clip.source))
        cursor = end
    if cursor < len(text):
        runs.append(Run(text=text[cursor:]))
    return Document(runs=runs, clips=[c for _s, _e, c in tagged], **kwargs)


def test_estimate_uses_rate_and_speed():
    assert estimate_duration_s("x" * 30, 1.0, 10.0) == 3.0
    assert estimate_duration_s("x" * 30, 2.0, 10.0) == 1.5
    assert estimate_duration_s("   ", 1.0, 10.0) == 0.0


def test_estimate_falls_back_when_no_history():
    assert estimate_duration_s("x" * 30, 1.0, None) == 30 / FALLBACK_CHARS_PER_SECOND
    assert estimate_duration_s("x" * 30, 1.0, 0.0) == 30 / FALLBACK_CHARS_PER_SECOND


def test_clips_are_placed_end_to_end_in_text_order_across_tracks():
    alice = Character.from_preset_dict("Alice", {})
    bob = Character.from_preset_dict("Bob", {})
    track_a = Track(name="A", character_id=alice.id, order_index=0)
    track_b = Track(name="B", character_id=bob.id, order_index=1)
    late = Clip(character_id=bob.id, track_id=track_b.id)
    early = Clip(character_id=alice.id, track_id=track_a.id)
    doc = _doc("x" * 50, [(20, 50, late), (0, 20, early)], characters=[alice, bob], tracks=[track_a, track_b])

    arr = compute_arrangement(doc, chars_per_second=10.0)

    assert [p.clip.id for p in arr.placed] == [early.id, late.id]
    assert arr.placed[0].start_s == 0.0
    assert arr.placed[0].duration_s == 2.0
    assert arr.placed[0].estimated is True
    assert arr.placed[1].start_s == 2.0
    assert arr.placed[1].duration_s == 3.0
    assert arr.total_duration_s == 5.0


def test_generated_clip_uses_segment_durations_and_is_not_estimated():
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="A", character_id=alice.id)
    clip = Clip(character_id=alice.id, track_id=track.id,
                segments=[Segment(duration=1.5, audio_path="a.wav"), Segment(duration=0.5, audio_path="b.wav")])
    doc = _doc("x" * 100, [(0, 100, clip)], characters=[alice], tracks=[track])

    arr = compute_arrangement(doc, chars_per_second=10.0)

    assert arr.placed[0].duration_s == 2.0
    assert arr.placed[0].estimated is False


def test_pinned_timestamp_overrides_sequential_placement_and_shifts_followers():
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="A", character_id=alice.id)
    first = Clip(character_id=alice.id, track_id=track.id, timeline_timestamp=4.0)
    second = Clip(character_id=alice.id, track_id=track.id)
    doc = _doc("x" * 20, [(0, 10, first), (10, 20, second)], characters=[alice], tracks=[track])

    arr = compute_arrangement(doc, chars_per_second=10.0)

    assert arr.placed[0].start_s == 4.0
    assert arr.placed[1].start_s == 5.0  # follows the pinned clip's end


def test_effective_speed_from_character_preset_shortens_estimate():
    fast = Character.from_preset_dict("Fast", {"speed": 2.0})
    track = Track(name="F", character_id=fast.id)
    clip = Clip(character_id=fast.id, track_id=track.id)
    doc = _doc("x" * 20, [(0, 20, clip)], characters=[fast], tracks=[track])

    arr = compute_arrangement(doc, chars_per_second=10.0)

    assert arr.placed[0].duration_s == 1.0


def test_at_time_and_predecessor_lookups():
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="A", character_id=alice.id)
    a = Clip(character_id=alice.id, track_id=track.id)
    b = Clip(character_id=alice.id, track_id=track.id)
    doc = _doc("x" * 20, [(0, 10, a), (10, 20, b)], characters=[alice], tracks=[track])
    arr = compute_arrangement(doc, chars_per_second=10.0)

    assert [p.clip.id for p in arr.at_time(0.5)] == [a.id]
    assert [p.clip.id for p in arr.at_time(1.5)] == [b.id]
    assert arr.at_time(9.0) == []
    assert text_order_predecessor(arr, b.id).clip.id == a.id
    assert text_order_predecessor(arr, a.id) is None


def test_clip_without_runs_is_skipped():
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="A", character_id=alice.id)
    orphan = Clip(character_id=alice.id, track_id=track.id)
    doc = Document.from_plain_text("hello", characters=[alice], tracks=[track], clips=[orphan])

    arr = compute_arrangement(doc, chars_per_second=10.0)

    assert arr.placed == []
    assert arr.total_duration_s == 0.0
