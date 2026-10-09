"""Pacing rules, conversational gaps and overlap in kokoro_gui/daw/arrangement.py
(plan 24). Pure Python: documents are built from run lists, no Qt."""
import pytest

from kokoro_gui.daw.arrangement import (
    boundary_gap, compute_arrangement, gap_jitter_range, heading_clip_id, heading_speed, is_heading,
    is_heading_text, jitter_gap_s, overlap_s,
)
from kokoro_gui.daw.models import Character, Clip, Document, Run

# One character speaks 10 characters a second at 10 chars/s: every line below is 1.0 s long.


def _line(label: str, length: int = 10) -> str:
    """`length` characters of text ending in a full stop, so it is never a heading."""
    return (label * length)[:length - 1] + "."


def _document(parts, characters=None, **settings):
    """`parts` is `[(clip_or_None, text)]` in order, joined with `separator`
    the caller puts in the text itself. A `None` clip is unclipped text."""
    runs, clips = [], []
    for clip, text in parts:
        if clip is None:
            runs.append(Run(text=text))
        else:
            runs.append(Run(text=text, clip_id=clip.id, kind=clip.source))
            clips.append(clip)
    base = {"gap_s": 0.5, "paragraph_gap_s": 2.0}
    base.update(settings)
    return Document(runs=runs, clips=clips, characters=characters or [], settings=base)


def _starts(doc):
    arrangement = compute_arrangement(doc, chars_per_second=10.0)
    return [p.start_s for p in arrangement.placed]


# -- headings ----------------------------------------------------------------


@pytest.mark.parametrize("text, expected", [
    ("Chapter One", True),
    ("  Chapter One  ", True),
    ("[Narrator]: Chapter One", True),
    ("A Very Long Title That Runs Past The Twelve Word Limit For Sure Yes", False),
    ("one two three four five six seven eight nine ten eleven twelve", True),
    ("one two three four five six seven eight nine ten eleven twelve thirteen", False),
    ("It was a dark night.", False),
    ("Was it dark?", False),
    ("Stop!", False),
    ("First, then", True),
    ("First,", False),
    ("Before;", False),
    ("Two\nlines", False),
    ("", False),
    ("   ", False),
])
def test_is_heading_text(text, expected):
    assert is_heading_text(text) is expected


def test_only_the_first_generated_clip_can_be_the_heading():
    title, body = Clip(), Clip()
    doc = _document([(title, "Chapter One"), (None, "\n\n"), (body, "Short title")])
    assert heading_clip_id(doc) == title.id
    assert is_heading(doc, title)
    assert not is_heading(doc, body)


def test_a_first_clip_that_reads_as_a_sentence_is_no_heading():
    first, second = Clip(), Clip()
    doc = _document([(first, "It was a dark night."), (None, " "), (second, "Chapter Two")])
    assert heading_clip_id(doc) is None
    assert not is_heading(doc, second)


def test_a_subproject_or_a_bed_is_never_the_heading():
    nested = Clip(source="nested")
    doc = _document([(nested, "Chapter One"), (None, "\n"), (Clip(), "Hello there.")])
    assert heading_clip_id(doc) is None
    bed = Clip(source="imported", original_audio_path="/p/audio/imported/abc.wav")
    assert heading_clip_id(_document([(bed, "theme.wav")])) is None


def test_heading_is_the_first_clip_by_text_not_by_list_order():
    title, body = Clip(), Clip()
    doc = _document([(title, "Chapter One"), (None, "\n\n"), (body, _line("b"))])
    doc.clips.reverse()
    assert heading_clip_id(doc) == title.id


def test_a_document_with_no_clip_has_no_heading():
    assert heading_clip_id(_document([(None, "Chapter One")])) is None


@pytest.mark.parametrize("value, expected", [(None, 1.0), (0.9, 0.9), ("0.8", 0.8), (0, 1.0), (-1, 1.0), ("x", 1.0)])
def test_heading_speed_setting(value, expected):
    doc = _document([(Clip(), "Title")], **({} if value is None else {"heading_speed": value}))
    assert heading_speed(doc) == expected


# -- gap order -----------------------------------------------------------------


def test_heading_gap_follows_a_heading_and_defaults_to_one_point_two():
    title, body, more = Clip(), Clip(), Clip()
    doc = _document([(title, "Chapter One"), (None, " "), (body, _line("b")), (None, " "), (more, _line("c"))])
    # "Chapter One" is 11 characters: 1.1 s. 1.1 + 1.2, then the body 1.0 s + the clip gap.
    assert _starts(doc) == pytest.approx([0.0, 1.1 + 1.2, 1.1 + 1.2 + 1.0 + 0.5])


def test_heading_gap_setting_overrides_the_default():
    title, body = Clip(), Clip()
    doc = _document([(title, "Chapter One"), (None, " "), (body, _line("b"))], heading_gap_after_s=3.0)
    assert _starts(doc)[1] == pytest.approx(1.1 + 3.0)


def test_heading_gap_wins_over_the_paragraph_gap_and_loses_to_a_clip_override():
    title, body = Clip(), Clip()
    doc = _document([(title, "Chapter One"), (None, "\n\n"), (body, _line("b"))], heading_gap_after_s=0.7)
    assert _starts(doc)[1] == pytest.approx(1.1 + 0.7)
    body.gap_before_s = 0.1
    assert _starts(doc)[1] == pytest.approx(1.1 + 0.1)


def test_a_paragraph_after_a_body_clip_still_gets_the_paragraph_gap():
    first, second = Clip(), Clip()
    doc = _document([(first, _line("a")), (None, "\n\n"), (second, _line("b"))])
    assert _starts(doc)[1] == pytest.approx(1.0 + 2.0)


def test_chapter_gap_defaults_to_the_paragraph_gap_and_has_its_own_setting():
    body, chapter = Clip(), Clip(source="nested")
    parts = [(body, _line("a")), (None, " "), (chapter, "Chapter Two")]
    assert _starts(_document(parts))[1] == pytest.approx(1.0 + 2.0)
    assert _starts(_document(parts, chapter_gap_s=4.0))[1] == pytest.approx(1.0 + 4.0)


def test_chapter_gap_wins_over_the_heading_gap():
    title, chapter = Clip(), Clip(source="nested")
    doc = _document([(title, "Chapter One"), (None, " "), (chapter, "Chapter Two")], chapter_gap_s=3.5)
    assert _starts(doc)[1] == pytest.approx(1.1 + 3.5)


def test_a_clip_override_wins_over_the_chapter_gap_and_the_first_clip_gets_none():
    chapter = Clip(source="nested")
    doc = _document([(chapter, "Chapter Two"), (None, " "), (Clip(), _line("b"))], chapter_gap_s=3.0)
    assert _starts(doc)[0] == 0.0
    second = Clip(source="nested", gap_before_s=0.25)
    doc = _document([(Clip(), _line("a")), (None, " "), (second, "Chapter Two")], chapter_gap_s=3.0)
    assert _starts(doc)[1] == pytest.approx(1.0 + 0.25)


def test_boundary_gap_names_the_rule_that_decided():
    title, body = Clip(), Clip()
    doc = _document([(title, "Chapter One"), (None, "\n\n"), (body, _line("b"))])
    text = doc.text
    extent = doc.clip_extent(body.id)
    assert boundary_gap(doc, text, None, title, 0) == ("first", 0.0)
    assert boundary_gap(doc, text, 11, body, extent[0], title, title.id) == ("heading", 1.2)
    assert boundary_gap(doc, text, 11, body, extent[0], title, None) == ("paragraph", 2.0)


def test_the_estimate_of_a_heading_uses_heading_speed():
    title, body = Clip(), Clip()
    parts = [(title, "x" * 9 + "z"), (None, " "), (body, _line("b"))]
    assert compute_arrangement(_document(parts), chars_per_second=10.0).placed[0].duration_s == pytest.approx(1.0)
    slow = compute_arrangement(_document(parts, heading_speed=0.5), chars_per_second=10.0)
    assert slow.placed[0].duration_s == pytest.approx(2.0)
    assert slow.placed[1].duration_s == pytest.approx(1.0)


# -- jitter --------------------------------------------------------------------


def _dialogue(count, jitter=None, characters=None, **settings):
    """`count` one-second lines alternating between two speakers, a space apart."""
    alice, bob = characters or (Character.from_preset_dict("Alice", {}), Character.from_preset_dict("Bob", {}))
    parts, clips = [], []
    for index in range(count):
        clip = Clip(character_id=(alice if index % 2 == 0 else bob).id)
        clips.append(clip)
        if index:
            parts.append((None, " "))
        parts.append((clip, _line(chr(97 + index))))
    if jitter is not None:
        settings["gap_jitter_s"] = jitter
    return _document(parts, characters=[alice, bob], **settings), clips


def test_gap_jitter_range_reads_two_non_negative_numbers():
    doc = _document([(Clip(), "x")])
    assert gap_jitter_range(doc) is None
    for value, expected in [([0.1, 0.4], (0.1, 0.4)), ((0.4, 0.1), (0.1, 0.4)), (["0.2", 0.3], (0.2, 0.3)),
                            ([0.1], None), ([0.1, 0.2, 0.3], None), ("0.1", None), ([-0.1, 0.2], None),
                            (["a", 1], None), (None, None)]:
        doc.settings["gap_jitter_s"] = value
        assert gap_jitter_range(doc) == expected


def test_jitter_is_stable_across_runs_and_calls():
    doc, _clips = _dialogue(6, jitter=[0.2, 0.9])
    assert _starts(doc) == _starts(doc)
    # The draw comes from the clip id through hashlib, never `hash()`.
    assert jitter_gap_s("clip-a", 0.2, 0.9) == jitter_gap_s("clip-a", 0.2, 0.9)
    assert jitter_gap_s("clip-a", 0.2, 0.9) != jitter_gap_s("clip-b", 0.2, 0.9)


def test_jitter_draw_is_pinned():
    """The seed recipe is sha256 of the id, its first 8 bytes, `random.Random`.
    Changing it moves every jittered clip in every saved project."""
    assert jitter_gap_s("pin", 0.0, 1.0) == pytest.approx(0.6172996622363893)


def test_jitter_gaps_stay_within_bounds_at_every_speaker_change():
    doc, clips = _dialogue(8, jitter=[0.2, 0.9])
    starts = _starts(doc)
    gaps = [starts[i + 1] - (starts[i] + 1.0) for i in range(len(clips) - 1)]
    assert all(0.2 - 1e-9 <= gap <= 0.9 + 1e-9 for gap in gaps)
    assert len({round(gap, 6) for gap in gaps}) > 1  # they differ from one another


def test_jitter_only_applies_at_speaker_changes_and_not_across_paragraphs():
    alice = Character.from_preset_dict("Alice", {})
    bob = Character.from_preset_dict("Bob", {})
    a1, a2, b1, a3 = (Clip(character_id=alice.id), Clip(character_id=alice.id), Clip(character_id=bob.id),
                      Clip(character_id=alice.id))
    doc = _document([(a1, _line("a")), (None, " "), (a2, _line("b")), (None, " "), (b1, _line("c")),
                     (None, "\n\n"), (a3, _line("d"))],
                    characters=[alice, bob], gap_jitter_s=[0.1, 0.2], gap_s=0.5, paragraph_gap_s=2.0)
    starts = _starts(doc)
    assert starts[1] - 1.0 == pytest.approx(0.5)  # same speaker: the normal gap
    assert 0.1 <= starts[2] - (starts[1] + 1.0) <= 0.2  # speaker change: drawn
    assert starts[3] - (starts[2] + 1.0) == pytest.approx(2.0)  # paragraph break wins


def test_a_clip_gap_override_beats_jitter_and_jitter_off_is_the_plain_gap():
    doc, clips = _dialogue(3, jitter=[0.2, 0.9])
    clips[1].gap_before_s = 0.05
    starts = _starts(doc)
    assert starts[1] == pytest.approx(1.05)
    doc, _clips = _dialogue(3)
    assert _starts(doc) == pytest.approx([0.0, 1.5, 3.0])


# -- overlap -------------------------------------------------------------------


@pytest.mark.parametrize("value, expected", [(0.3, 0.3), ("0.3", 0.3), (0, 0.0), (9, 5.0), (-1, 0.0),
                                             (None, None), ("x", None), (True, None)])
def test_overlap_value(value, expected):
    clip = Clip(overrides={} if value is None else {"overlap_s": value})
    assert overlap_s(clip) == expected


def _backchannel_document(overlap, text="right."):
    alice = Character.from_preset_dict("Alice", {})
    bob = Character.from_preset_dict("Bob", {})
    long = Clip(character_id=alice.id)
    back = Clip(character_id=bob.id, overrides={"overlap_s": overlap})
    after = Clip(character_id=alice.id)
    doc = _document([(long, "x" * 29 + "."), (None, " "), (back, text), (None, " "), (after, _line("c"))],
                    characters=[alice, bob])
    return doc, long, back, after


def test_a_backchannel_starts_before_the_end_of_the_line_it_answers():
    doc, long, back, after = _backchannel_document(0.3)
    placed = compute_arrangement(doc, chars_per_second=10.0).by_clip_id()
    assert placed[long.id].duration_s == pytest.approx(3.0)
    assert placed[back.id].start_s == pytest.approx(3.0 - 0.3)
    # It runs 0.6 s, so it ends after the long line and the next line follows it.
    assert placed[after.id].start_s == pytest.approx(3.3 + 0.5)


def test_a_backchannel_inside_the_long_line_leaves_the_next_line_after_the_long_one():
    doc, long, back, after = _backchannel_document(1.0)
    placed = compute_arrangement(doc, chars_per_second=10.0).by_clip_id()
    assert placed[back.id].start_s == pytest.approx(2.0)
    assert placed[back.id].end_s < placed[long.id].end_s
    # The cursor stays at the later end: the next line starts after the long one.
    assert placed[after.id].start_s == pytest.approx(3.0 + 0.5)


def test_an_overlap_longer_than_the_previous_clip_starts_with_it():
    first = Clip()
    second = Clip(overrides={"overlap_s": 5.0})
    doc = _document([(first, _line("a")), (None, " "), (second, _line("b"))])
    assert _starts(doc) == pytest.approx([0.0, 0.0])


def test_overlap_zero_removes_the_gap_and_the_first_clip_ignores_overlap():
    first = Clip(overrides={"overlap_s": 0.5})
    second = Clip(overrides={"overlap_s": 0.0})
    doc = _document([(first, _line("a")), (None, " "), (second, _line("b"))], gap_s=0.9)
    assert _starts(doc) == pytest.approx([0.0, 1.0])


def test_overlap_beats_a_gap_override_and_a_pinned_clip_ignores_it():
    first = Clip()
    second = Clip(overrides={"overlap_s": 0.4}, gap_before_s=2.0)
    third = Clip(overrides={"overlap_s": 0.4}, timeline_timestamp=9.0)
    doc = _document([(first, _line("a")), (None, " "), (second, _line("b")), (None, " "), (third, _line("c"))])
    starts = _starts(doc)
    assert starts[1] == pytest.approx(0.6)
    assert starts[2] == pytest.approx(9.0)


def test_the_clip_after_a_backchannel_follows_the_longer_clip_with_no_gap():
    first, back, after = Clip(), Clip(overrides={"overlap_s": 1.0}), Clip()
    doc = _document([(first, _line("a", 30)), (None, " "), (back, _line("b", 5)), (None, " "),
                     (after, _line("c"))], gap_s=0.0)
    placed = compute_arrangement(doc, chars_per_second=10.0).by_clip_id()
    assert placed[back.id].start_s == pytest.approx(2.0)
    assert placed[after.id].start_s == pytest.approx(placed[first.id].end_s)
