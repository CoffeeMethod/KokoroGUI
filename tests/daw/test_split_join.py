"""Split and join a clip at a point (plan 10): `split_join.offset_at` maps a
playhead time to a text offset, `Document.split_clip` / `join_clips` do the
cut and the merge, and `SplitClipCommand` / `JoinClipsCommand` undo them."""
import dataclasses
import re

from kokoro_gui.daw import split_join
from kokoro_gui.daw.arrangement import PlacedClip
from kokoro_gui.daw.models import Clip, Segment

TEXT = "alpha beta gamma delta"
START = 10  # the clip's document offset


def _spans(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _placed(text=TEXT, timed=True, start_s=5.0, per_word=1.0, gaps=0.0):
    """A clip placed at `start_s`; each word lasts `per_word` seconds, a
    silence of `gaps` between words. Returns (placed, word_offsets)."""
    spans = _spans(text)
    words, t = [], 0.0
    for i, (a, b) in enumerate(spans):
        words.append([text[a:b], t, t + per_word])
        t += per_word + gaps
    segment = Segment(order_index=0, text=text, audio_path="x.wav", duration=t, words=words if timed else [])
    clip = Clip(segments=[segment])
    placed = PlacedClip(clip=clip, start_s=start_s, duration_s=t, estimated=not timed)

    def word_offsets(seg, index):
        a, b = spans[index]
        return START + a, START + b

    return placed, word_offsets


def test_playhead_mid_word_cuts_at_that_words_start():
    placed, offsets = _placed()
    # "gamma" is the third word, 2.0 to 3.0 s into the clip.
    assert split_join.offset_at(placed, 5.0 + 2.5, TEXT, START, offsets) == START + TEXT.index("gamma")


def test_playhead_in_a_gap_cuts_at_the_next_word():
    placed, offsets = _placed(per_word=1.0, gaps=0.5)
    # "beta" spans 1.5 to 2.5 s, "gamma" 3.0 to 4.0 s; 2.75 s is between them.
    assert split_join.offset_at(placed, 5.0 + 2.75, TEXT, START, offsets) == START + TEXT.index("gamma")


def test_playhead_on_the_first_word_cuts_before_the_second():
    placed, offsets = _placed()
    assert split_join.offset_at(placed, 5.0 + 0.2, TEXT, START, offsets) == START + TEXT.index("beta")


def test_playhead_before_the_audio_starts_cuts_before_the_second_word():
    placed, offsets = _placed()
    assert split_join.offset_at(placed, 4.0, TEXT, START, offsets) == START + TEXT.index("beta")


def test_playhead_after_the_last_word_has_no_cut():
    placed, offsets = _placed(per_word=1.0, gaps=0.5)
    assert split_join.offset_at(placed, 5.0 + 99.0, TEXT, START, offsets) is None


def test_leading_whitespace_does_not_make_the_first_word_a_cut():
    text = "  alpha beta"
    placed, offsets = _placed(text)
    assert split_join.offset_at(placed, 5.0 + 0.2, text, START, offsets) == START + text.index("beta")


def test_untimed_clip_cuts_proportionally_and_snaps_to_a_word_start():
    placed, offsets = _placed(timed=False)
    # 55 percent of 22 characters is offset 12, the second letter of
    # "gamma" (11 to 16); the next word start is "delta" at 17.
    placed = dataclasses.replace(placed, duration_s=10.0)
    assert split_join.offset_at(placed, 5.0 + 5.5, TEXT, START, offsets) == START + TEXT.index("delta")


def test_untimed_cut_on_a_word_start_stays_there():
    placed, offsets = _placed(timed=False)
    placed = dataclasses.replace(placed, duration_s=22.0)
    assert split_join.offset_at(placed, 5.0 + 11.0, TEXT, START, offsets) == START + TEXT.index("gamma")


def test_untimed_cut_near_the_start_still_leaves_a_first_half():
    placed, offsets = _placed(timed=False)
    placed = dataclasses.replace(placed, duration_s=10.0)
    assert split_join.offset_at(placed, 5.0, TEXT, START, offsets) == START + TEXT.index("beta")


def test_untimed_cut_inside_the_last_word_has_no_cut():
    placed, offsets = _placed(timed=False)
    placed = dataclasses.replace(placed, duration_s=10.0)
    assert split_join.offset_at(placed, 5.0 + 9.9, TEXT, START, offsets) is None


def test_one_word_clip_has_no_cut():
    placed, offsets = _placed("alpha")
    assert split_join.offset_at(placed, 5.0 + 0.5, "alpha", START, offsets) is None
    untimed, offsets = _placed("alpha", timed=False)
    assert split_join.offset_at(untimed, 5.0 + 0.5, "alpha", START, offsets) is None


def test_words_the_text_cannot_place_fall_back_to_the_proportional_cut():
    placed, _ = _placed()
    placed = dataclasses.replace(placed, duration_s=10.0)
    assert split_join.offset_at(placed, 5.0 + 5.5, TEXT, START, lambda seg, i: None) \
        == START + TEXT.index("delta")


def test_a_word_offset_inside_a_word_snaps_to_the_next_word_start():
    placed, _ = _placed()
    # A lexicon rewrite can map a spoken word to a span that starts mid-word.
    assert split_join.offset_at(placed, 5.0 + 0.2, TEXT, START, lambda seg, i: (START + 2, START + 4)) \
        == START + TEXT.index("beta")


# -- Document.split_clip / join_clips ------------------------------------------------

import copy  # noqa: E402
import dataclasses as _dc  # noqa: E402

import pytest  # noqa: E402

from kokoro_gui.daw.dirty import is_clip_dirty  # noqa: E402
from kokoro_gui.daw.models import Character, Document, Run, Track  # noqa: E402
from kokoro_gui.daw.undo import JoinClipsCommand, SplitClipCommand  # noqa: E402

LINE = "alpha beta gamma delta"
CUT = LINE.index("gamma")


def _walk_clip_text(doc, clip_id):
    return "".join(run.text for run in doc.runs if run.clip_id == clip_id)


def _consistent(doc):
    """The index agrees with a walk of the runs (verify mode is on for the
    whole suite, so reading `index()` also compares it with a fresh build)."""
    index = doc.index()
    pos, extents = 0, {}
    for run in doc.runs:
        if run.clip_id is not None:
            start, _ = extents.get(run.clip_id, (pos, pos))
            extents[run.clip_id] = (start, pos + len(run.text))
        pos += len(run.text)
    assert {c.id for c in doc.clips} == set(extents)
    for clip_id, extent in extents.items():
        assert index.extent(clip_id) == extent
        assert doc.clip_text(doc.get_clip(clip_id)) == _walk_clip_text(doc, clip_id)
    assert "".join(r.text for r in doc.runs) == doc.text


def _generated():
    alice = Character.from_preset_dict("Alice", {"voice": "af_bella"})
    doc = Document.from_plain_text(LINE + "\n\nnext line", characters=[alice],
                                   tracks=[Track(name="Alice", character_id=alice.id)])
    clip = doc.assign_character_to_range(0, len(LINE), alice.id)
    clip.overrides.update({"take": 2, "speed": 1.2, "target_duration_s": 3.0, "reference_range": [1.0, 2.0],
                           "fx_preset": "Radio", "overlap_s": 0.3})
    clip.fx_override = {"reverb": 0.3}
    clip.status, clip.note = "approved", "check the name"
    clip.fade_in_s, clip.fade_out_s = 0.1, 0.4
    clip.segments = [Segment(order_index=0, text=LINE, cache_key="k", audio_path="a.wav", duration=2.0)]
    clip.takes = {0: [Segment(order_index=0, text="old", cache_key="o", audio_path="o.wav", duration=1.0)]}
    clip.gap_before_s = 0.7
    return doc, clip, alice


def test_split_makes_two_clips_and_leaves_the_text_alone():
    doc, clip, alice = _generated()
    text = doc.text
    second = doc.split_clip(clip.id, CUT)

    assert doc.text == text
    assert doc.clip_text(clip) == "alpha beta "
    assert doc.clip_text(second) == "gamma delta"
    assert doc.clips == [clip, second]
    assert doc.clip_extent(second.id) == (CUT, len(LINE))
    _consistent(doc)


def test_split_copies_the_clips_fields_to_the_second_half():
    doc, clip, alice = _generated()
    second = doc.split_clip(clip.id, CUT)

    assert second.character_id == alice.id and second.track_id == clip.track_id
    assert second.overrides == {"speed": 1.2, "fx_preset": "Radio"}
    assert second.fx_override == {"reverb": 0.3} and second.fx_override is not clip.fx_override
    assert (second.status, second.note, second.gap_before_s) == ("approved", "check the name", 0.0)
    assert second.segments == [] and second.takes == {}
    assert (second.fade_in_s, second.fade_out_s) == (0.0, 0.4)
    assert (clip.fade_in_s, clip.fade_out_s) == (0.1, 0.0)
    assert clip.gap_before_s == 0.7
    # The slot the first half was cut to stays with it.
    assert clip.overrides["target_duration_s"] == 3.0 and clip.overrides["take"] == 2
    # So does the overlap with the clip before: the second half follows the first.
    assert clip.overrides["overlap_s"] == 0.3


def test_split_leaves_the_first_half_stale_and_the_second_new():
    doc, clip, _ = _generated()
    second = doc.split_clip(clip.id, CUT)
    assert is_clip_dirty(clip, doc.clip_text(clip), doc.effective_config_for_clip(clip)) is True
    assert is_clip_dirty(second, doc.clip_text(second), doc.effective_config_for_clip(second)) is True
    assert clip.segments and clip.takes  # still there for a regenerate to park


def test_split_keeps_the_clips_after_it_in_place():
    doc, clip, alice = _generated()
    other = doc.assign_character_to_range(len(LINE) + 2, len(doc.text), alice.id)
    second = doc.split_clip(clip.id, CUT)
    assert doc.clips == [clip, second, other]
    _consistent(doc)


@pytest.mark.parametrize("offset", [0, len(LINE), len(LINE) + 5, -3])
def test_split_refuses_a_cut_outside_the_clip_or_at_its_edge(offset):
    doc, clip, _ = _generated()
    with pytest.raises(ValueError):
        doc.split_clip(clip.id, offset)
    assert doc.clips == [clip] and doc.text == LINE + "\n\nnext line"


def test_split_refuses_a_pinned_clip_a_subproject_and_an_unknown_id():
    doc, clip, alice = _generated()
    clip.timeline_timestamp = 4.0
    with pytest.raises(ValueError, match="Unpin"):
        doc.split_clip(clip.id, CUT)
    clip.timeline_timestamp = None
    nested = doc.insert_nested_clip(len(doc.text), {"kind": "embedded", "id": "p"}, "Chapter")
    with pytest.raises(ValueError):
        doc.split_clip(nested.id, len(doc.text) - 2)
    with pytest.raises(ValueError):
        doc.split_clip("nope", 3)


def test_split_refuses_a_cut_with_only_whitespace_on_one_side():
    alice = Character.from_preset_dict("Alice", {})
    doc = Document.from_plain_text("alpha   ", characters=[alice])
    clip = doc.assign_character_to_range(0, 8, alice.id)
    with pytest.raises(ValueError):
        doc.split_clip(clip.id, 6)


def test_split_takes_the_id_a_redo_asks_for():
    doc, clip, _ = _generated()
    second = doc.split_clip(clip.id, CUT, new_id="fixed-id")
    assert second.id == "fixed-id"
    assert doc.get_clip("fixed-id") is second
    assert all(run.clip_id in (None, clip.id, "fixed-id") for run in doc.runs)


def test_join_puts_the_text_back_under_the_first_clip():
    doc, clip, _ = _generated()
    second = doc.split_clip(clip.id, CUT)
    clip.fade_in_s = 0.25
    joined = doc.join_clips(clip.id, second.id)

    assert joined is clip
    assert doc.clips == [clip]
    assert doc.clip_text(clip) == LINE
    assert (clip.fade_in_s, clip.fade_out_s) == (0.25, 0.4)
    assert is_clip_dirty(clip, doc.clip_text(clip), doc.effective_config_for_clip(clip)) is True
    _consistent(doc)


def test_join_takes_the_whitespace_between_the_clips_too():
    alice = Character.from_preset_dict("Alice", {})
    doc = Document.from_plain_text("one two\n\nthree four", characters=[alice])
    first = doc.assign_character_to_range(0, 7, alice.id)
    second = doc.assign_character_to_range(9, 19, alice.id)
    doc.join_clips(first.id, second.id)
    assert doc.clip_extent(first.id) == (0, 19)
    assert doc.clip_text(first) == "one two\n\nthree four"
    assert doc.clips == [first]
    _consistent(doc)


def test_join_refuses_another_character_text_between_and_the_wrong_order():
    alice = Character.from_preset_dict("Alice", {})
    bob = Character.from_preset_dict("Bob", {})
    doc = Document.from_plain_text("one two three four five", characters=[alice, bob])
    a = doc.assign_character_to_range(0, 3, alice.id)
    b = doc.assign_character_to_range(4, 7, bob.id)
    c = doc.assign_character_to_range(8, 13, alice.id)
    d = doc.assign_character_to_range(14, 18, alice.id)
    with pytest.raises(ValueError, match="same character"):
        doc.join_clips(a.id, b.id)
    with pytest.raises(ValueError, match="between"):
        doc.join_clips(a.id, c.id)
    with pytest.raises(ValueError, match="follow"):
        doc.join_clips(d.id, c.id)
    with pytest.raises(ValueError):
        doc.join_clips(a.id, a.id)
    doc.join_clips(c.id, d.id)
    assert doc.clip_text(c) == "three four"
    assert [x.id for x in doc.clips] == [a.id, b.id, c.id]


def test_join_refuses_a_clip_with_another_clip_between_and_one_placed_by_timestamp():
    alice = Character.from_preset_dict("Alice", {})
    bob = Character.from_preset_dict("Bob", {})
    doc = Document.from_plain_text("a b c", characters=[alice, bob])
    a = doc.assign_character_to_range(0, 1, alice.id)
    doc.assign_character_to_range(2, 3, bob.id)
    c = doc.assign_character_to_range(4, 5, alice.id)
    with pytest.raises(ValueError, match="between"):
        doc.join_clips(a.id, c.id)

    doc2 = Document.from_plain_text("one two", characters=[alice])
    x = doc2.assign_character_to_range(0, 3, alice.id)
    y = doc2.assign_character_to_range(4, 7, alice.id)
    y.timeline_timestamp = 9.0
    with pytest.raises(ValueError, match="Unpin"):
        doc2.join_clips(x.id, y.id)


def test_join_refuses_a_generated_clip_and_an_imported_one():
    alice = Character.from_preset_dict("Alice", {})
    doc = Document.from_plain_text("one two", characters=[alice])
    a = doc.assign_character_to_range(0, 3, alice.id)
    b = doc.assign_character_to_range(4, 7, alice.id)
    b.source = "imported"
    with pytest.raises(ValueError, match="same character and kind"):
        doc.join_clips(a.id, b.id)


def test_next_clip_is_the_next_tagged_clip_in_the_text():
    alice = Character.from_preset_dict("Alice", {})
    doc = Document.from_plain_text("one two three", characters=[alice])
    a = doc.assign_character_to_range(0, 3, alice.id)
    c = doc.assign_character_to_range(8, 13, alice.id)
    assert doc.next_clip(a.id) is c
    assert doc.next_clip(c.id) is None
    assert doc.next_clip("nope") is None


# -- imported recordings ------------------------------------------------------------

REC_A = "a" * 16
RECORDING = "one two three four"
RECORDING_WORDS = [[0, 3, REC_A, 1.0, 1.3], [4, 7, REC_A, 1.33, 1.6], [8, 13, REC_A, 1.6, 2.0],
                   [14, 18, REC_A, 2.0, 2.4]]


def _recording():
    clip = Clip(source="imported", fade_out_s=0.2)
    doc = Document(runs=[Run(text=RECORDING, clip_id=clip.id, kind="imported", words=copy.deepcopy(RECORDING_WORDS))],
                   clips=[clip],
                   settings={"sources": {REC_A: {"path": "/p/a.wav", "sample_rate": 24000, "duration_s": 60.0}}})
    doc.refresh_imported_segments()
    return doc, clip


def test_splitting_a_recording_splits_its_words_and_audio():
    doc, clip = _recording()
    second = doc.split_clip(clip.id, 8)

    assert doc.text == RECORDING
    assert doc.clip_text(clip) == "one two " and doc.clip_text(second) == "three four"
    assert second.source == "imported" and second.gap_before_s == 0.0
    assert (clip.fade_out_s, second.fade_out_s) == (0.0, 0.2)
    first_run = next(r for r in doc.runs if r.clip_id == clip.id)
    second_run = next(r for r in doc.runs if r.clip_id == second.id)
    assert first_run.words == RECORDING_WORDS[:2]
    assert second_run.words == [[0, 5, REC_A, 1.6, 2.0], [6, 10, REC_A, 2.0, 2.4]]
    assert clip.segments and second.segments
    assert clip.segments[0].range == [1.0, 1.6] and second.segments[0].range == [1.6, 2.4]
    _consistent(doc)


def test_joining_two_halves_of_a_recording_gives_the_original_back():
    doc, clip = _recording()
    second = doc.split_clip(clip.id, 8)
    doc.join_clips(clip.id, second.id)

    assert doc.clips == [clip]
    assert doc.clip_text(clip) == RECORDING
    assert [r.words for r in doc.runs] == [RECORDING_WORDS]
    assert clip.fade_out_s == 0.2
    assert clip.segments[0].range == [1.0, 2.4]
    _consistent(doc)


# -- the undo commands ---------------------------------------------------------------


def _state(doc):
    clips = [_dc.asdict(c) for c in doc.clips]
    for clip in clips:  # a recording's segments are derived and get new ids
        for segment in clip["segments"]:
            segment.pop("id")
    return ([(r.text, r.clip_id, r.kind, r.words) for r in doc.runs], clips, [_dc.asdict(t) for t in doc.tracks])


def test_one_undo_puts_a_split_clip_back_and_redo_splits_it_again():
    doc, clip, _ = _generated()
    before = _state(doc)
    doc.undo_stack.push(SplitClipCommand(clip.id, CUT))
    after = _state(doc)
    assert len(doc.clips) == 2

    doc.undo_stack.undo()
    assert _state(doc) == before
    assert doc.get_clip(clip.id).segments and doc.get_clip(clip.id).takes
    _consistent(doc)

    doc.undo_stack.redo()
    assert _state(doc) == after  # the second half has the id it had
    _consistent(doc)


def test_one_undo_puts_a_joined_clip_back():
    doc, clip, _ = _generated()
    second = doc.split_clip(clip.id, CUT)
    second.segments = [Segment(order_index=0, text="gamma delta", cache_key="s", audio_path="s.wav", duration=1.0)]
    before = _state(doc)
    doc.undo_stack.push(JoinClipsCommand(clip.id, second.id))
    after = _state(doc)
    assert len(doc.clips) == 1

    doc.undo_stack.undo()
    assert _state(doc) == before
    assert doc.get_clip(second.id).segments[0].audio_path == "s.wav"
    _consistent(doc)

    doc.undo_stack.redo()
    assert _state(doc) == after
    _consistent(doc)


def test_undo_of_a_split_in_a_document_with_a_recording_restores_everything():
    doc, clip = _recording()
    before = _state(doc)
    doc.undo_stack.push(SplitClipCommand(clip.id, 8))
    split = _state(doc)
    doc.undo_stack.undo()
    assert _state(doc) == before
    doc.undo_stack.redo()
    assert _state(doc) == split
    new_id = doc.clips[1].id
    doc.undo_stack.push(JoinClipsCommand(clip.id, new_id))
    doc.undo_stack.undo()
    assert _state(doc) == split


def test_a_refused_split_leaves_nothing_on_the_stack():
    doc, clip, _ = _generated()
    with pytest.raises(ValueError):
        doc.undo_stack.push(SplitClipCommand(clip.id, 0))
    assert not doc.undo_stack.can_undo()
    assert len(doc.clips) == 1
