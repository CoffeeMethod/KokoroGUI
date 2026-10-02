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
    # These tests are about order and length; gaps have their own tests.
    kwargs.setdefault("settings", {"gap_s": 0.0, "paragraph_gap_s": 0.0})
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


def test_clip_duration_callable_replaces_segment_durations():
    alice = Character.from_preset_dict("Alice", {})
    clip = Clip(character_id=alice.id, segments=[Segment(order_index=0, duration=4.0, audio_path="x.wav")])
    doc = _doc("hello", [(0, 5, clip)], characters=[alice])

    placed = compute_arrangement(doc, chars_per_second=10.0, clip_duration=lambda c: 1.25).placed[0]
    assert placed.duration_s == 1.25 and not placed.estimated

    placed = compute_arrangement(doc, chars_per_second=10.0, clip_duration=lambda c: None).placed[0]
    assert placed.estimated


# ---------------------------------------------------------------------------
# Gaps (phase 2, A1)
# ---------------------------------------------------------------------------


def _two_clips(text, split, **settings):
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="A", character_id=alice.id)
    first = Clip(character_id=alice.id, track_id=track.id)
    second = Clip(character_id=alice.id, track_id=track.id)
    doc = _doc(text, [(0, split[0], first), (split[1], len(text), second)],
               characters=[alice], tracks=[track], settings=dict(settings))
    return doc, first, second


def test_default_gap_between_clips():
    from kokoro_gui.daw.arrangement import DEFAULT_GAP_S

    doc, first, second = _two_clips("x" * 10 + " " + "y" * 10, (10, 11))
    arr = compute_arrangement(doc, chars_per_second=10.0)
    assert arr.placed[0].start_s == 0.0
    assert arr.placed[1].start_s == 1.0 + DEFAULT_GAP_S


def test_paragraph_gap_across_a_blank_line():
    doc, _first, _second = _two_clips("x" * 10 + "\n  \n" + "y" * 10, (10, 14), gap_s=0.1, paragraph_gap_s=2.0)
    arr = compute_arrangement(doc, chars_per_second=10.0)
    assert arr.placed[1].start_s == 3.0


def test_clip_override_wins_over_the_setting():
    doc, _first, second = _two_clips("x" * 10 + "\n\n" + "y" * 10, (10, 12), gap_s=0.1, paragraph_gap_s=2.0)
    second.gap_before_s = 0.5
    arr = compute_arrangement(doc, chars_per_second=10.0)
    assert arr.placed[1].start_s == 1.5


def test_pinned_clip_ignores_gaps():
    doc, _first, second = _two_clips("x" * 10 + " " + "y" * 10, (10, 11), gap_s=5.0)
    second.gap_before_s = 3.0
    second.timeline_timestamp = 1.0
    arr = compute_arrangement(doc, chars_per_second=10.0)
    assert arr.placed[1].start_s == 1.0


def test_first_clip_gets_only_its_own_override():
    doc, first, _second = _two_clips("x" * 10 + " " + "y" * 10, (10, 11), gap_s=0.0)
    assert compute_arrangement(doc, chars_per_second=10.0).placed[0].start_s == 0.0
    first.gap_before_s = 0.75
    assert compute_arrangement(doc, chars_per_second=10.0).placed[0].start_s == 0.75


# -- overlaps and ripple on regenerate (phase 3, beside the library) -----------------


def _placed(**clips_at):
    """An arrangement from `name=(track_id, start, duration, timestamp, pinned)`."""
    from kokoro_gui.daw.arrangement import Arrangement, PlacedClip

    placed, by_name = [], {}
    for name, (track_id, start, duration, timestamp, pinned) in clips_at.items():
        clip = Clip(track_id=track_id, timeline_timestamp=timestamp, pinned=pinned)
        by_name[name] = clip
        placed.append(PlacedClip(clip=clip, start_s=start, duration_s=duration, estimated=False))
    total = max((p.end_s for p in placed), default=0.0)
    return Arrangement(placed=placed, total_duration_s=total), by_name


def test_overlaps_reports_same_track_intersections_only():
    from kokoro_gui.daw.arrangement import overlaps

    arrangement, c = _placed(
        a=("t1", 0.0, 2.0, None, False),
        b=("t1", 1.5, 1.0, 1.5, False),   # overlaps a
        c=("t2", 0.5, 1.0, 0.5, False),   # other track: fine
        d=("t1", 2.5, 1.0, 2.5, False),   # touches b's end exactly: fine
        e=(None, 0.0, 5.0, 0.0, False),   # no track: ignored
    )
    assert overlaps(arrangement) == [(c["a"].id, c["b"].id)]


def test_overlaps_is_empty_for_a_plain_read_through():
    from kokoro_gui.daw.arrangement import overlaps

    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="A", character_id=alice.id)
    first = Clip(character_id=alice.id, track_id=track.id)
    second = Clip(character_id=alice.id, track_id=track.id)
    doc = _doc("aaaa bbbb", [(0, 4, first), (5, 9, second)], characters=[alice], tracks=[track])
    assert overlaps(compute_arrangement(doc, chars_per_second=10.0)) == []


def test_ripple_moves_later_timestamp_clips_and_skips_pinned_ones():
    from kokoro_gui.daw.arrangement import plan_ripple

    arrangement, c = _placed(
        regen=("t1", 0.0, 2.0, None, False),
        before=("t2", 1.0, 1.0, 1.0, False),     # starts before regen's old end
        after=("t1", 2.5, 1.0, 2.5, False),      # timestamp-placed, after: moves
        locked=("t1", 4.0, 1.0, 4.0, True),      # pinned: never moves
        text_ordered=("t1", 5.0, 1.0, None, False),  # follows on its own
    )
    assert plan_ripple(arrangement, {c["regen"].id: 0.75}) == {c["after"].id: 0.75}
    assert plan_ripple(arrangement, {c["regen"].id: -0.5}) == {c["after"].id: -0.5}
    assert plan_ripple(arrangement, {}) == {}


def test_ripple_from_several_clips_adds_up():
    from kokoro_gui.daw.arrangement import plan_ripple

    arrangement, c = _placed(
        one=("t1", 0.0, 1.0, None, False),
        two=("t1", 2.0, 1.0, 2.0, False),
        three=("t1", 4.0, 1.0, 4.0, False),
    )
    shifts = plan_ripple(arrangement, {c["one"].id: 0.5, c["two"].id: 0.25})
    # two moves for one's change only; three for both.
    assert shifts == {c["two"].id: 0.5, c["three"].id: 0.75}


# -- D3: onset alignment of pinned clips -------------------------------------------


def _cue_doc(timestamp=2.0, pinned=True, words=None, onset_s=0.2, overrides=None, **settings):
    """One clip with 1 s of audio at `timestamp`, and a text-order follower."""
    cue = Clip(timeline_timestamp=timestamp, pinned=pinned, overrides=dict(overrides or {}),
               segments=[Segment(order_index=0, duration=1.0, audio_path="cue.wav",
                                 words=list(words or []), onset_s=onset_s, tail_s=0.1)])
    after = Clip(segments=[Segment(duration=0.5, audio_path="after.wav")])
    doc = _doc("x" * 20, [(0, 10, cue), (10, 20, after)])
    doc.settings.update(settings)
    return doc, cue, after


def test_align_onset_starts_a_pinned_clip_early_by_its_onset():
    doc, cue, after = _cue_doc(align_onset=True)
    placed = compute_arrangement(doc, chars_per_second=10.0).by_clip_id()

    assert abs(placed[cue.id].start_s - 1.8) < 1e-9
    assert abs(placed[cue.id].aligned_onset_s - 0.2) < 1e-9
    # The follower is placed after the audio's real end.
    assert abs(placed[after.id].start_s - 2.8) < 1e-9


def test_align_onset_off_places_the_clip_at_its_timestamp():
    doc, cue, _after = _cue_doc(align_onset=False)
    placed = compute_arrangement(doc, chars_per_second=10.0).by_clip_id()[cue.id]
    assert placed.start_s == 2.0 and placed.aligned_onset_s == 0.0


def test_trim_on_disables_alignment():
    """Trim already cut the leading silence; subtracting it again would
    place the clip early."""
    doc, cue, _after = _cue_doc(align_onset=True, overrides={"trim": True})
    assert compute_arrangement(doc, chars_per_second=10.0).by_clip_id()[cue.id].start_s == 2.0

    doc, cue, _after = _cue_doc(align_onset=True)
    trims = {"trim_silence": True}
    arrangement = compute_arrangement(doc, chars_per_second=10.0, clip_post_config=lambda clip: trims)
    assert arrangement.by_clip_id()[cue.id].start_s == 2.0
    arrangement = compute_arrangement(doc, chars_per_second=10.0, clip_post_config=lambda clip: {})
    assert abs(arrangement.by_clip_id()[cue.id].start_s - 1.8) < 1e-9


def test_first_word_start_wins_over_the_energy_onset():
    from kokoro_gui.daw.arrangement import first_onset_s

    doc, cue, _after = _cue_doc(align_onset=True, words=[["hello", 0.35, 0.6], ["there", 0.7, 0.9]])
    assert first_onset_s(cue) == 0.35
    assert abs(compute_arrangement(doc, chars_per_second=10.0).by_clip_id()[cue.id].start_s - 1.65) < 1e-9

    cue.segments[0].words, cue.segments[0].onset_s = [], None
    assert first_onset_s(cue) is None
    assert compute_arrangement(doc, chars_per_second=10.0).by_clip_id()[cue.id].start_s == 2.0


def test_alignment_needs_a_pinned_clip_placed_by_timestamp():
    doc, cue, _after = _cue_doc(pinned=False, align_onset=True)
    assert compute_arrangement(doc, chars_per_second=10.0).by_clip_id()[cue.id].start_s == 2.0

    doc, cue, _after = _cue_doc(timestamp=None, align_onset=True)
    assert compute_arrangement(doc, chars_per_second=10.0).by_clip_id()[cue.id].start_s == 0.0


def test_aligned_start_never_goes_below_zero():
    doc, cue, _after = _cue_doc(timestamp=0.1, align_onset=True)
    assert compute_arrangement(doc, chars_per_second=10.0).by_clip_id()[cue.id].start_s == 0.0


def test_onset_is_scaled_to_the_rendered_length():
    """A pitch shift that halves the length halves the onset too."""
    doc, cue, _after = _cue_doc(align_onset=True)
    arrangement = compute_arrangement(doc, chars_per_second=10.0, clip_duration=lambda clip: 0.5)
    assert abs(arrangement.by_clip_id()[cue.id].start_s - 1.9) < 1e-9


def test_align_onset_default_follows_the_pinned_clips():
    from kokoro_gui.daw.arrangement import align_onset_enabled

    doc, cue, _after = _cue_doc()  # no align_onset key, one pinned clip
    assert "align_onset" not in doc.settings and align_onset_enabled(doc)
    assert abs(compute_arrangement(doc, chars_per_second=10.0).by_clip_id()[cue.id].start_s - 1.8) < 1e-9

    doc, cue, _after = _cue_doc(pinned=False)
    assert not align_onset_enabled(doc)
    # An explicit value wins either way.
    doc.settings["align_onset"] = True
    assert align_onset_enabled(doc)
    doc, cue, _after = _cue_doc(align_onset=False)
    assert not align_onset_enabled(doc)


def test_a_pinned_music_bed_does_not_turn_align_onset_on():
    from kokoro_gui.daw.arrangement import align_onset_enabled

    doc, _cue, _after = _cue_doc(pinned=False)
    bed = Clip(source="imported", original_audio_path="/p/audio/imported/abc.wav", timeline_timestamp=0.0,
               pinned=True)
    doc.clips.append(bed)
    assert not align_onset_enabled(doc)
