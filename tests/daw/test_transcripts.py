"""Tests for kokoro_gui/daw/transcripts.py: the transcript and chapter files
`mixdown(extras=...)` writes. Qt-free, no audio: the arrangement is laid out
from fixed clip lengths."""
import json

import pytest

from kokoro_gui.daw import markers as marker_ops
from kokoro_gui.daw.arrangement import compute_arrangement
from kokoro_gui.daw.models import PLACEHOLDER, Character, Clip, Document, Run
from kokoro_gui.daw.transcripts import (
    EXTRA_SUFFIXES, TEXT_EXTRAS, chapter_rows, clean_extras, cue_rows, extra_path, format_timestamp,
    write_chapters_json, write_extras, write_plain_text, write_podcast_transcript_json, write_show_notes,
    write_srt_speakers, write_vtt,
)

LINES = [
    ("Alice", "[Alice:Radio]: Hello   there.", 1.0),
    ("Bob", "[Bob]: General [pause:1.5] Kenobi.", 2.5),
    ("Alice", "[Alice]: You are a\nbold one.", 1.5),
]


def _read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def _dialogue(lines=LINES):
    """`(doc, arrangement)` for tagged lines, each its own clip placed end to end."""
    people = {name: Character.from_preset_dict(name, {}) for name in {n for n, _t, _s in lines}}
    text = "\n".join(t for _n, t, _s in lines)
    doc = Document.from_plain_text(text, characters=list(people.values()))
    doc.settings.update({"gap_s": 0.0, "paragraph_gap_s": 0.0})
    durations, cursor = {}, 0
    for name, line, seconds in lines:
        clip = doc.assign_character_to_range(cursor, cursor + len(line), people[name].id)
        durations[clip.id] = seconds
        cursor += len(line) + 1
    return doc, compute_arrangement(doc, clip_duration=lambda c: durations.get(c.id), chars_per_second=15.0)


def _markers(doc, *named):
    for seconds, name, note in named:
        doc.settings["markers"], _m = marker_ops.add_marker(doc.settings, seconds, name, note)


# -- cues ----------------------------------------------------------------------------------------


def test_cue_rows_are_in_timeline_order_with_tags_and_whitespace_gone():
    doc, arrangement = _dialogue()

    assert cue_rows(doc, arrangement) == [
        (0.0, 1.0, "Alice", "Hello there."),
        (1.0, 3.5, "Bob", "General Kenobi."),
        (3.5, 5.0, "Alice", "You are a bold one."),
    ]


def test_beds_subprojects_estimated_and_empty_clips_get_no_cue():
    doc, _arrangement = _dialogue(LINES + [("Bob", "[Bob]: Not generated yet.", None)])
    nested = doc.insert_nested_clip(len(doc.text), {"kind": "embedded", "id": "c1"}, "Chapter 1")
    bed = Clip(source="imported", original_audio_path="theme.wav")
    doc.runs.append(Run(text="theme.wav", clip_id=bed.id, kind=PLACEHOLDER))
    doc.clips.append(bed)
    seconds = {c.id: s for c, s in zip(doc.clips, (1.0, 2.5, 1.5, None, 4.0, 6.0))}
    arrangement = compute_arrangement(doc, clip_duration=lambda c: seconds.get(c.id), chars_per_second=15.0)

    assert {p.clip.id for p in arrangement.placed} >= {nested.id, bed.id}
    assert [row[3] for row in cue_rows(doc, arrangement)] == ["Hello there.", "General Kenobi.", "You are a bold one."]


def test_a_clip_with_no_character_has_no_speaker():
    doc = Document.from_plain_text("Just narration.")
    doc.assign_character_to_range(0, 15, None)
    arrangement = compute_arrangement(doc, clip_duration=lambda c: 2.0, chars_per_second=15.0)

    assert cue_rows(doc, arrangement) == [(0.0, 2.0, "", "Just narration.")]


# -- the four transcript formats -----------------------------------------------------------------


def test_vtt_has_voice_spans_and_milliseconds_with_a_dot(tmp_path):
    doc, arrangement = _dialogue()

    write_vtt(doc, arrangement, str(tmp_path / "t.vtt"))

    assert _read(tmp_path / "t.vtt") == (
        "WEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:01.000\n"
        "<v Alice>Hello there.\n"
        "\n"
        "00:00:01.000 --> 00:00:03.500\n"
        "<v Bob>General Kenobi.\n"
        "\n"
        "00:00:03.500 --> 00:00:05.000\n"
        "<v Alice>You are a bold one.\n"
    )


def test_vtt_without_speakers_has_plain_cue_text(tmp_path):
    doc, arrangement = _dialogue()

    write_vtt(doc, arrangement, str(tmp_path / "t.vtt"), speakers=False)

    text = _read(tmp_path / "t.vtt")
    assert "<v " not in text and "Alice" not in text
    assert "00:00:01.000 --> 00:00:03.500\nGeneral Kenobi.\n" in text


def test_vtt_escapes_the_characters_a_cue_cannot_hold(tmp_path):
    doc, arrangement = _dialogue([("A&B <x>", "[A&B <x>]: Fish & chips --> <b>now</b>.", 2.0)])

    write_vtt(doc, arrangement, str(tmp_path / "t.vtt"))

    assert "<v A&amp;B &lt;x&gt;>Fish &amp; chips --&gt; &lt;b&gt;now&lt;/b&gt;.\n" in _read(tmp_path / "t.vtt")


def test_srt_with_speakers_prefixes_the_name(tmp_path):
    doc, arrangement = _dialogue()

    write_srt_speakers(doc, arrangement, str(tmp_path / "t.srt"))

    assert _read(tmp_path / "t.srt") == (
        "1\n00:00:00,000 --> 00:00:01,000\nAlice: Hello there.\n"
        "\n"
        "2\n00:00:01,000 --> 00:00:03,500\nBob: General Kenobi.\n"
        "\n"
        "3\n00:00:03,500 --> 00:00:05,000\nAlice: You are a bold one.\n"
    )


def test_srt_without_speakers_is_the_plain_text(tmp_path):
    doc, arrangement = _dialogue()

    write_srt_speakers(doc, arrangement, str(tmp_path / "t.srt"), speakers=False)

    text = _read(tmp_path / "t.srt")
    assert "Alice" not in text and "2\n00:00:01,000 --> 00:00:03,500\nGeneral Kenobi.\n" in text


def test_podcast_transcript_json_has_the_spec_fields(tmp_path):
    doc, arrangement = _dialogue()

    write_podcast_transcript_json(doc, arrangement, str(tmp_path / "t.json"))

    data = json.loads(_read(tmp_path / "t.json"))
    assert data["version"] == "1.0.0"
    assert data["segments"] == [
        {"speaker": "Alice", "startTime": 0, "endTime": 1, "body": "Hello there."},
        {"speaker": "Bob", "startTime": 1, "endTime": 3.5, "body": "General Kenobi."},
        {"speaker": "Alice", "startTime": 3.5, "endTime": 5, "body": "You are a bold one."},
    ]


def test_podcast_transcript_json_without_speakers_drops_the_field(tmp_path):
    doc, arrangement = _dialogue()

    write_podcast_transcript_json(doc, arrangement, str(tmp_path / "t.json"), speakers=False)

    segments = json.loads(_read(tmp_path / "t.json"))["segments"]
    assert all(set(s) == {"startTime", "endTime", "body"} for s in segments)


def test_plain_text_breaks_the_paragraph_when_the_speaker_changes(tmp_path):
    doc, arrangement = _dialogue([("Alice", "[Alice]: One.", 1.0), ("Alice", "[Alice]: Two.", 1.0),
                                  ("Bob", "[Bob]: Three.", 1.0), ("Alice", "[Alice]: Four.", 1.0)])

    write_plain_text(doc, arrangement, str(tmp_path / "t.txt"))
    with_names = _read(tmp_path / "t.txt")
    write_plain_text(doc, arrangement, str(tmp_path / "t.txt"), speakers=False)

    assert with_names == "Alice: One.\nAlice: Two.\n\nBob: Three.\n\nAlice: Four.\n"
    assert _read(tmp_path / "t.txt") == "One.\nTwo.\n\nThree.\n\nFour.\n"


def test_no_transcript_ever_holds_a_tag_or_a_pause_marker(tmp_path):
    doc, arrangement = _dialogue()
    for name, writer in (("vtt", write_vtt), ("srt", write_srt_speakers), ("json", write_podcast_transcript_json),
                         ("txt", write_plain_text)):
        for speakers in (True, False):
            writer(doc, arrangement, str(tmp_path / f"t.{name}"), speakers=speakers)
            text = _read(tmp_path / f"t.{name}")
            assert "]:" not in text and "Radio" not in text and "pause" not in text


def test_an_empty_document_writes_valid_empty_files(tmp_path):
    doc = Document.from_plain_text("")
    arrangement = compute_arrangement(doc)

    write_vtt(doc, arrangement, str(tmp_path / "t.vtt"))
    write_podcast_transcript_json(doc, arrangement, str(tmp_path / "t.json"))
    write_plain_text(doc, arrangement, str(tmp_path / "t.txt"))

    assert _read(tmp_path / "t.vtt").startswith("WEBVTT")
    assert json.loads(_read(tmp_path / "t.json"))["segments"] == []
    assert _read(tmp_path / "t.txt") == ""


def test_format_timestamp_rounds_to_milliseconds_and_carries():
    assert format_timestamp(3723.004) == "01:02:03,004"
    assert format_timestamp(0.9996, ".") == "00:00:01.000"
    assert format_timestamp(-1.0) == "00:00:00,000"


# -- chapters ------------------------------------------------------------------------------------


def _book():
    doc, arrangement = _dialogue()
    return doc, arrangement


def test_chapters_come_from_markers_first():
    doc, arrangement = _book()
    _markers(doc, (0.0, "Opening", ""), (1.0, "The pause", "Cut the cough."), (3.5, "", ""))

    assert chapter_rows(doc, arrangement) == [
        (0.0, "Opening", ""), (1.0, "The pause", "Cut the cough."), (3.5, "M3", ""),
    ]


def test_chapters_fall_back_to_subprojects_titled_by_their_placeholder():
    doc, arrangement = _dialogue()
    first = doc.insert_nested_clip(len(doc.text), {"kind": "embedded", "id": "c1"}, "Part  one")
    second = doc.insert_nested_clip(len(doc.text), {"kind": "embedded", "id": "c2"}, "Part two")
    lengths = {c.id: s for c, s in zip(doc.clips, (1.0, 2.5, 1.5, 4.0, 3.0))}
    arrangement = compute_arrangement(doc, clip_duration=lambda c: lengths.get(c.id), chars_per_second=15.0)

    assert chapter_rows(doc, arrangement) == [(5.0, "Part one", ""), (9.0, "Part two", "")]
    assert first.id != second.id


def test_a_project_with_neither_has_no_chapters(tmp_path):
    doc, arrangement = _book()

    write_chapters_json(doc, arrangement, str(tmp_path / "c.json"))
    write_show_notes(doc, arrangement, str(tmp_path / "n.md"))

    assert json.loads(_read(tmp_path / "c.json")) == {"version": "1.2.0", "chapters": []}
    assert _read(tmp_path / "n.md") == ""


def test_chapters_json_has_the_spec_fields(tmp_path):
    doc, arrangement = _book()
    _markers(doc, (0.0, "Opening", ""), (1.25, "Second", "A note."))

    write_chapters_json(doc, arrangement, str(tmp_path / "c.json"))

    assert json.loads(_read(tmp_path / "c.json")) == {
        "version": "1.2.0",
        "chapters": [{"startTime": 0, "title": "Opening"}, {"startTime": 1.25, "title": "Second"}],
    }


def test_show_notes_list_each_chapter_with_its_note_under_it(tmp_path):
    doc, arrangement = _book()
    _markers(doc, (0.0, "Opening", ""), (65.0, "Second", "Line one.\nLine two."), (3725.0, "Late", ""))

    write_show_notes(doc, arrangement, str(tmp_path / "n.md"))

    assert _read(tmp_path / "n.md") == (
        "- (00:00) Opening\n"
        "- (01:05) Second\n"
        "  Line one.\n"
        "  Line two.\n"
        "- (1:02:05) Late\n"
    )


def test_a_range_drops_earlier_markers_and_shifts_the_rest(tmp_path):
    doc, arrangement = _book()
    _markers(doc, (0.0, "Before", ""), (1.0, "Start", ""), (2.0, "Middle", ""), (5.0, "After", ""))

    assert chapter_rows(doc, arrangement, range_s=(1.0, 4.0)) == [(0.0, "Start", ""), (1.0, "Middle", "")]
    assert chapter_rows(doc, arrangement, range_s=(4.0, 1.0)) == [(0.0, "Start", ""), (1.0, "Middle", "")]


def test_head_padding_moves_every_chapter_later():
    doc, arrangement = _book()
    _markers(doc, (0.0, "Start", ""), (2.0, "Middle", ""))

    assert chapter_rows(doc, arrangement, head_s=1.5) == [(1.5, "Start", ""), (3.5, "Middle", "")]
    assert chapter_rows(doc, arrangement, range_s=(2.0, 5.0), head_s=0.5) == [(0.5, "Middle", "")]


# -- write_extras --------------------------------------------------------------------------------


def test_write_extras_writes_each_file_next_to_the_audio(tmp_path):
    doc, arrangement = _book()
    _markers(doc, (0.0, "Opening", ""))

    paths, warnings = write_extras(doc, arrangement, str(tmp_path), "story", list(TEXT_EXTRAS))

    names = sorted(p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in paths)
    assert names == sorted(f"story{suffix}" for suffix in EXTRA_SUFFIXES.values())
    assert all((tmp_path / n).exists() for n in names)
    assert warnings == []
    assert "Alice: Hello there." in _read(tmp_path / "story.speakers.srt")


def test_write_extras_says_when_the_chapter_files_are_empty(tmp_path):
    doc, arrangement = _book()

    paths, warnings = write_extras(doc, arrangement, str(tmp_path), "story", ["chapters_json", "show_notes"])

    assert len(paths) == 2 and len(warnings) == 1 and "no markers or subprojects" in warnings[0]


def test_write_extras_without_speakers_leaves_the_names_out(tmp_path):
    doc, arrangement = _book()

    write_extras(doc, arrangement, str(tmp_path), "story", ["vtt", "txt"], speakers=False)

    assert "Alice" not in _read(tmp_path / "story.vtt") and "Alice" not in _read(tmp_path / "story.txt")


def test_clean_extras_keeps_known_keys_in_order_and_drops_the_rest():
    assert clean_extras(["txt", "bogus", "vtt", "vtt", 3]) == ["vtt", "txt"]
    assert clean_extras("vtt") == [] and clean_extras(None) == []


def test_extra_path_uses_the_suffix_table(tmp_path):
    assert extra_path(str(tmp_path), "a", "vtt") == str(tmp_path / "a.vtt")
    assert set(EXTRA_SUFFIXES) == set(TEXT_EXTRAS)
    # no extra may land on the plain .srt or the cue sheet's .csv
    assert not {".srt", ".csv"} & set(EXTRA_SUFFIXES.values())
