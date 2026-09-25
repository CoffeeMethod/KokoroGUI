"""Tests for kokoro_gui/daw/subtitles.py: SRT, WebVTT and ASS cues (phase 5
D2). The files under fixtures/subtitles/ are byte-exact (see the
.gitattributes there): dialogue.srt has CRLF line endings and no trailing
blank line, bom.srt starts with a UTF-8 BOM."""
import os

import pytest

from kokoro_gui.daw.models import Character, Document
from kokoro_gui.daw.subtitles import Cue, SubtitleError, decode_bytes, format_for_path, parse, parse_text
from kokoro_gui.daw.undo import ImportCuesCommand

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "subtitles")


def _fixture(name):
    return os.path.join(FIXTURES, name)


def _approx(cue, start, end):
    return abs(cue.start_s - start) < 1e-9 and abs(cue.end_s - end) < 1e-9


# -- SRT -------------------------------------------------------------------------------


def test_srt_file_with_crlf_markup_and_no_trailing_blank_line():
    with open(_fixture("dialogue.srt"), "rb") as f:
        assert b"\r\n" in f.read()  # the fixture really is CRLF

    cues = parse(_fixture("dialogue.srt"))

    assert [c.text for c in cues] == ["Where were you\nlast night?", "Out.", "Late, then."]
    assert _approx(cues[0], 1.0, 3.5)
    assert _approx(cues[1], 4.25, 6.0)
    assert _approx(cues[2], 3723.004, 3725.0)
    assert all(c.speaker is None for c in cues)


def test_srt_file_with_a_utf8_bom():
    cues = parse(_fixture("bom.srt"))

    assert cues == [Cue(0.5, 2.0, "Café au lait, s'il vous plaît.", None)]
    # parse_text drops a BOM left on a string too.
    assert parse_text(chr(0xFEFF) + "1\n00:00:00,500 --> 00:00:02,000\nX\n", "srt")[0].text == "X"


def test_overlapping_cues_are_kept_and_sorted_by_start():
    cues = parse(_fixture("overlap.srt"))

    assert [c.text for c in cues] == [
        "First, and it runs into the next.",
        "Second, but listed first.",
        "Same start as the first block.",  # equal starts keep file order
    ]
    assert [(c.start_s, c.end_s) for c in cues] == [(1.0, 6.0), (5.0, 8.0), (5.0, 7.0)]


def test_srt_index_line_is_optional_and_blank_text_cues_are_dropped():
    text = "00:00:01,000 --> 00:00:02,000\nNo index.\n\n2\n00:00:03,000 --> 00:00:04,000\n<i></i>\n\n"

    cues = parse_text(text, "srt")

    assert [c.text for c in cues] == ["No index."]


def test_srt_end_before_start_is_raised_to_the_start():
    cues = parse_text("1\n00:00:05,000 --> 00:00:04,000\nBackwards.\n", ".SRT")

    assert (cues[0].start_s, cues[0].end_s) == (5.0, 5.0)


# -- WebVTT ------------------------------------------------------------------------------


def test_vtt_file_speakers_settings_and_skipped_blocks():
    cues = parse(_fixture("dialogue.vtt"))

    assert [(c.text, c.speaker) for c in cues] == [
        ("Hello & welcome.", "Alice"),
        ("Thanks for having me.", "Bob Smith"),
        ("Nobody in particular\nsaid this.", None),
    ]
    assert _approx(cues[0], 1.0, 3.5)  # no hours, cue settings dropped
    assert _approx(cues[1], 4.0, 5.0)
    assert _approx(cues[2], 6.0, 7.25)


def test_vtt_note_block_holding_a_timing_line_is_not_a_cue():
    text = "WEBVTT\n\nNOTE\n00:00:01.000 --> 00:00:02.000\n\n00:00:03.000 --> 00:00:04.000\nReal.\n"

    assert [c.text for c in parse_text(text, "vtt")] == ["Real."]


# -- ASS / SSA -------------------------------------------------------------------------


def test_ass_file_names_overrides_and_line_breaks():
    cues = parse(_fixture("dialogue.ass"))

    assert [(c.text, c.speaker) for c in cues] == [
        ("Hi, there, Bob.", "Alice"),  # \h is a space, commas in Text survive
        ("Well, that's one way\nto put it.", "Bob"),
        ("Nobody speaks here.", None),
    ]
    assert _approx(cues[0], 1.0, 2.0)
    assert _approx(cues[1], 2.5, 4.0)


def test_ass_format_line_decides_the_column_order():
    text = (
        "[Events]\n"
        "Format: Start, End, Text, Name\n"  # Text needn't be the last column
        "Dialogue: 0:00:01.00,0:00:02.00,Hello,Carol\n"
        "[Fonts]\n"
        "Dialogue: 0:00:09.00,0:00:10.00,Outside events,Dan\n"
    )

    cues = parse_text(text, "ssa")

    assert cues == [Cue(1.0, 2.0, "Hello", "Carol")]


def test_ssa_marked_column_and_lowercase_break():
    text = (
        "[Events]\n"
        "Format: Marked, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: Marked=0,0:00:01.50,0:00:02.00,*Default,Eve,0000,0000,0000,,One\\ntwo\n"
    )

    assert parse_text(text, "ssa") == [Cue(1.5, 2.0, "One\ntwo", "Eve")]


# -- reading ---------------------------------------------------------------------------


def test_latin1_file_falls_back_from_utf8(tmp_path):
    path = tmp_path / "old.srt"
    path.write_bytes("1\n00:00:01,000 --> 00:00:02,000\nNaïve résumé\n".encode("latin-1"))

    assert parse(str(path))[0].text == "Naïve résumé"


def test_utf16_with_bom_decodes():
    data = "1\n00:00:01,000 --> 00:00:02,000\nHi\n".encode("utf-16")  # starts with a BOM

    assert data[:2] in (b"\xff\xfe", b"\xfe\xff")
    assert parse_text(decode_bytes(data), "srt")[0].text == "Hi"


def test_relative_path_is_resolved_before_reading(tmp_path, monkeypatch):
    sub = tmp_path / "a"
    sub.mkdir()
    (tmp_path / "cues.vtt").write_text("WEBVTT\n\n00:01.000 --> 00:02.000\nUp one.\n", encoding="utf-8")
    monkeypatch.chdir(sub)

    assert parse(os.path.join("..", "cues.vtt"))[0].text == "Up one."


def test_unknown_extension_and_missing_file_are_refused(tmp_path):
    with pytest.raises(SubtitleError):
        format_for_path("notes.txt")
    with pytest.raises(SubtitleError):
        parse(str(tmp_path / "missing.srt"))
    with pytest.raises(SubtitleError):
        parse_text("", "sub")


def test_cue_is_frozen():
    cue = Cue(1.0, 2.0, "x")
    with pytest.raises(Exception):
        cue.text = "y"
    assert cue.duration_s == 1.0


# -- ImportCuesCommand (kokoro_gui/daw/undo.py) -------------------------------------


def _cue_doc(text="Intro."):
    narrator = Character.from_preset_dict("Default", {})
    return Document.from_plain_text(text, characters=[narrator]), narrator


def test_import_cues_appends_a_pinned_clip_per_cue_as_its_own_paragraph():
    doc, narrator = _cue_doc()
    bob = Character.from_preset_dict("Bob", {})
    cues = [Cue(1.0, 2.5, "Two\nlines", "Alice"), Cue(3.0, 4.0, "Hi.", None)]

    command = ImportCuesCommand(cues, [bob.id, narrator.id], new_characters=[bob])
    doc.undo_stack.push(command)

    assert doc.text == "Intro.\n\nTwo lines\n\nHi."
    first, second = (doc.get_clip(i) for i in command.clip_ids)
    assert doc.clip_text(first) == "Two lines"
    assert (first.timeline_timestamp, first.pinned, first.source_text) == (1.0, True, "Two\nlines")
    assert first.overrides == {"target_duration_s": 1.5, "reference_range": [1.0, 2.5]}
    assert first.character_id == bob.id and doc.get_character(bob.id) is not None
    assert doc.get_track(first.track_id).character_id == bob.id
    assert (second.character_id, second.timeline_timestamp, second.overrides["target_duration_s"]) == \
        (narrator.id, 3.0, 1.0)


def test_import_cues_fills_each_clips_reference_range_from_its_cue():
    from kokoro_gui.daw.reference import reference_range

    doc, narrator = _cue_doc("")
    command = ImportCuesCommand([Cue(0.5, 1.25, "A"), Cue(3.0, 4.5, "B")], [narrator.id, narrator.id])
    doc.undo_stack.push(command)

    assert [reference_range(doc.get_clip(i)) for i in command.clip_ids] == [(0.5, 1.25), (3.0, 4.5)]
    doc.undo_stack.undo()
    doc.undo_stack.redo()
    assert [doc.get_clip(i).overrides["reference_range"] for i in command.clip_ids] == [[0.5, 1.25], [3.0, 4.5]]


def test_import_cues_is_one_undo_step_and_redo_recreates_the_same_clips():
    doc, narrator = _cue_doc("Intro.\n")
    bob = Character.from_preset_dict("Bob", {})
    command = ImportCuesCommand([Cue(0.0, 1.0, "A", "Bob"), Cue(1.0, 2.0, "B", None)], [bob.id, narrator.id],
                                new_characters=[bob])
    doc.undo_stack.push(command)
    # One newline is already there: one more makes the blank line.
    assert doc.text == "Intro.\n\nA\n\nB"

    doc.undo_stack.undo()
    assert doc.text == "Intro.\n"
    assert doc.clips == [] and doc.tracks == []
    assert [c.name for c in doc.characters] == ["Default"]

    doc.undo_stack.redo()
    assert [c.id for c in doc.clips] == command.clip_ids
    assert {c.name for c in doc.characters} == {"Default", "Bob"}


def test_imported_cue_fields_survive_a_save_and_load():
    from kokoro_gui.daw.serialization import document_from_dict, document_to_dict

    doc, narrator = _cue_doc("")
    doc.undo_stack.push(ImportCuesCommand([Cue(1.0, 2.5, "a\nb")], [narrator.id]))

    clip = document_from_dict(document_to_dict(doc)).clips[0]
    assert (clip.timeline_timestamp, clip.pinned, clip.source_text, clip.overrides) == \
        (1.0, True, "a\nb", {"target_duration_s": 1.5, "reference_range": [1.0, 2.5]})


def test_import_cues_into_an_empty_document_starts_without_a_separator():
    doc, narrator = _cue_doc("")
    doc.undo_stack.push(ImportCuesCommand([Cue(0.0, 1.0, "Only.")], [narrator.id]))

    assert doc.text == "Only."


def test_imported_cues_are_laned_in_the_unified_layout():
    doc, narrator = _cue_doc("")
    doc.settings["track_layout"] = {"mode": "unified", "lanes": 2}
    command = ImportCuesCommand([Cue(0.0, 1.0, "A"), Cue(0.5, 1.5, "B")], [narrator.id, narrator.id])
    doc.undo_stack.push(command)

    assert all(doc.get_clip(i).track_id is not None for i in command.clip_ids)
    doc.undo_stack.undo()
    assert doc.clips == [] and doc.tracks == []
