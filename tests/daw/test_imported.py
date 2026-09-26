"""Tests for kokoro_gui/daw/imported.py: segments derived from an imported
recording's timed words (phase 5 P3, grill Q32/Q33), the word builders for
Whisper and caption imports, the clipboard payload, and the crossfade plan
playback reads. No Qt."""
import json

import numpy as np
import pytest
import soundfile as sf

from kokoro_gui.daw import imported
from kokoro_gui.daw.imported import (
    JOIN_CROSSFADE_S,
    run_from_asr_words,
    run_words_for_text,
    segment_plays,
    segments_for,
    words_from_cue,
    words_payload,
)
from kokoro_gui.daw.models import Character, Clip, Document, Run, Segment
from kokoro_gui.daw.undo import ImportRecordingCommand

A = "aaaaaaaaaaaaaaaa"
B = "bbbbbbbbbbbbbbbb"
SOURCES = {A: {"path": "/p/audio/imported/a.wav", "sample_rate": 24000, "duration_s": 60.0},
           B: {"path": "/p/audio/imported/b.wav", "sample_rate": 24000, "duration_s": 60.0}}


def _doc(text, words, sources=None):
    clip = Clip(source="imported")
    doc = Document(runs=[Run(text=text, clip_id=clip.id, kind="imported", words=words)], clips=[clip],
                   settings={"sources": dict(sources if sources is not None else SOURCES)})
    return doc, clip


# -- segments_for -------------------------------------------------------------------


def test_contiguous_words_of_one_source_make_one_segment():
    doc, clip = _doc("one two three", [[0, 3, A, 1.0, 1.3], [4, 7, A, 1.33, 1.6], [8, 13, A, 1.6, 2.0]])
    segment, = segments_for(doc, clip)
    assert segment.range == [1.0, 2.0]
    assert segment.audio_path == "/p/audio/imported/a.wav"
    assert segment.duration == 1.0
    assert segment.raw is True
    assert segment.text == "one two three"
    assert segment.words == [["one", 0.0, 0.3], ["two", 0.33, 0.6], ["three", 0.6, 1.0]]


def test_a_200ms_gap_makes_two_segments():
    doc, clip = _doc("one two", [[0, 3, A, 1.0, 1.3], [4, 7, A, 1.5, 1.8]])
    first, second = segments_for(doc, clip)
    assert (first.range, second.range) == ([1.0, 1.3], [1.5, 1.8])
    assert (first.order_index, second.order_index) == (0, 1)
    assert (first.text, second.text) == ("one", "two")
    assert second.words == [["two", 0.0, 0.3]]


def test_a_word_played_earlier_in_the_file_starts_a_new_segment():
    doc, clip = _doc("two one", [[0, 3, A, 1.3, 1.6], [4, 7, A, 1.0, 1.3]])
    assert [s.range for s in segments_for(doc, clip)] == [[1.3, 1.6], [1.0, 1.3]]


def test_two_sources_make_two_segments():
    doc, clip = _doc("one two", [[0, 3, A, 1.0, 1.3], [4, 7, B, 1.3, 1.6]])
    first, second = segments_for(doc, clip)
    assert first.audio_path.endswith("a.wav") and second.audio_path.endswith("b.wav")
    assert (first.range, second.range) == ([1.0, 1.3], [1.3, 1.6])


def test_no_words_means_no_segments():
    doc, clip = _doc("untimed text", [])
    assert segments_for(doc, clip) == []


def test_words_spread_over_several_runs_are_walked_in_text_order():
    clip = Clip(source="imported")
    doc = Document(runs=[Run("one ", clip.id, "imported", words=[[0, 3, A, 1.0, 1.3]]),
                         Run("two", clip.id, "imported", words=[[0, 3, A, 1.3, 1.6]])],
                   clips=[clip], settings={"sources": dict(SOURCES)})
    segment, = segments_for(doc, clip)
    assert segment.range == [1.0, 1.6] and segment.text == "one two"


def test_a_source_missing_on_open_gives_a_pathless_segment(tmp_path):
    doc, clip = _doc("one", [[0, 3, A, 1.0, 1.3]], {A: {"path": None}})
    segment, = segments_for(doc, clip)
    assert segment.audio_path is None
    assert imported.missing_sources(doc) == [A]
    assert imported.clip_missing_source(doc, clip)


def test_a_source_whose_file_is_there_is_not_missing(tmp_path):
    path = tmp_path / "a.wav"
    sf.write(str(path), np.zeros(800, dtype=np.float32), 8000)
    doc, clip = _doc("one", [[0, 3, A, 0.0, 0.1]], {A: {"path": str(path)}})
    assert imported.missing_sources(doc) == []
    assert not imported.clip_missing_source(doc, clip)


def test_a_music_bed_is_not_a_recording_clip():
    assert imported.is_recording_clip(Clip(source="imported"))
    assert not imported.is_recording_clip(Clip(source="imported", original_audio_path="/x.wav"))
    assert not imported.is_recording_clip(Clip())


def test_refresh_leaves_a_bed_clips_segments_alone():
    bed = Clip(source="imported", original_audio_path="/p/audio/imported/bed.wav",
               segments=[Segment(audio_path="/p/audio/imported/bed.wav")])
    doc = Document(runs=[Run("bed", bed.id, "imported")], clips=[bed])
    doc.refresh_imported_segments()
    assert bed.segments[0].audio_path == "/p/audio/imported/bed.wav"


def test_word_spans_are_document_offsets_of_every_timed_word():
    clip = Clip(source="imported")
    doc = Document(runs=[Run("Intro. "), Run("one two", clip.id, "imported",
                                            words=[[0, 3, A, 1.0, 1.3], [4, 7, A, 1.3, 1.6]])], clips=[clip])
    assert imported.word_spans(doc) == [(7, 10), (11, 14)]


# -- building words -----------------------------------------------------------------


def test_words_from_cue_share_the_span_by_character_count():
    words = words_from_cue("Hi there, friend.", 10.0, 17.0, A)
    # "Hi" 2, "there," 6, "friend." 7 characters of 15: 7 s split 2:6:7.
    assert words == [[0, 2, A, 10.0, 10.933333], [3, 8, A, 10.933333, 13.733333],
                     [10, 16, A, 13.733333, 17.0]]


def test_words_from_cue_tile_the_cue_so_it_plays_as_one_range():
    text = "Hi there, friend."
    doc, clip = _doc(text, words_from_cue(text, 10.0, 17.0, A))
    segment, = segments_for(doc, clip)
    assert segment.range == [10.0, 17.0]


def test_words_from_cue_with_no_text_or_no_time_is_empty():
    assert words_from_cue("   ", 1.0, 2.0, A) == []
    assert words_from_cue("hi", 2.0, 2.0, A) == []


def test_run_from_asr_words_joins_the_words_and_splits_the_pauses():
    text, words = run_from_asr_words([(" Hello,", 0.2, 0.5), ("world", 0.9, 1.3), ("", 1.3, 1.4)], A)
    assert text == "Hello, world"
    assert words == [[0, 5, A, 0.2, 0.7], [7, 12, A, 0.7, 1.3]]


def test_deleting_an_asr_word_takes_its_share_of_the_pauses():
    text, words = run_from_asr_words([("one", 0.0, 0.3), ("two", 0.5, 0.8), ("three", 1.0, 1.4)], A)
    doc, clip = _doc(text, words)
    doc.refresh_imported_segments()
    assert [s.range for s in clip.segments] == [[0.0, 1.4]]
    doc.replace_text(4, 4, 0, "one three")
    assert [s.range for s in clip.segments] == [[0.0, 0.4], [0.9, 1.4]]


def test_run_words_for_text_keeps_times_for_an_edited_spelling():
    from kokoro_gui.daw.wordalign import align

    asr = [("one", 0.0, 0.3), ("tow", 0.3, 0.6), ("three", 0.6, 1.0)]
    words = run_words_for_text("one two three", align("one two three", asr), A)
    assert words == [[0, 3, A, 0.0, 0.3], [4, 7, A, 0.3, 0.6], [8, 13, A, 0.6, 1.0]]


def test_source_entry_reads_rate_and_length_from_the_header(tmp_path):
    path = tmp_path / "0011223344556677.wav"
    sf.write(str(path), np.zeros(16000, dtype=np.float32), 8000)
    source, entry = imported.source_entry(str(path))
    assert source == "0011223344556677"
    assert entry == {"path": str(path), "sample_rate": 8000, "duration_s": 2.0}


# -- clipboard ----------------------------------------------------------------------


def test_words_payload_rebases_the_words_inside_the_selection():
    clip = Clip(source="imported")
    doc = Document(runs=[Run("Say "), Run("one two three", clip.id, "imported",
                                         words=[[0, 3, A, 1.0, 1.3], [4, 7, B, 1.3, 1.6], [8, 13, A, 1.6, 2.0]])],
                   clips=[clip], settings={"sources": dict(SOURCES)})
    payload = words_payload(doc, 6, 15)  # "e two thr": "two" only, "one"/"three" cut
    assert payload == {"words": [[2, 5, B, 1.3, 1.6]], "sources": {B: SOURCES[B]}}
    assert json.loads(json.dumps(payload)) == payload
    assert words_payload(doc, 0, 3) == {"words": [], "sources": {}}


def test_a_payload_pasted_elsewhere_plays_the_same_audio():
    clip = Clip(source="imported")
    doc = Document(runs=[Run("one two three", clip.id, "imported",
                             words=[[0, 3, A, 1.0, 1.3], [4, 7, A, 1.3, 1.6], [8, 13, A, 1.6, 2.0]]),
                         Run("\n\nLater: ")], clips=[clip], settings={"sources": dict(SOURCES)})
    payload = words_payload(doc, 4, 7)
    doc.replace_text(len(doc.text), 0, 3, doc.text + "two")
    clip_id = doc.apply_words(len(doc.text) - 3, 3, payload["words"], payload["sources"])
    segment, = doc.get_clip(clip_id).segments
    assert segment.range == [1.3, 1.6] and segment.text == "two"


def test_a_paste_relinks_a_source_whose_file_was_missing():
    from kokoro_gui.daw.undo import ApplyWordsCommand

    doc, clip = _doc("one", [[0, 3, A, 1.0, 1.3]], {A: {"path": None}})
    doc.runs.append(Run("\n\nLater: "))
    assert imported.missing_sources(doc) == [A]
    payload = {"words": [[0, 3, A, 1.3, 1.6]], "sources": {A: SOURCES[A]}}
    doc.replace_text(len(doc.text), 0, 3, doc.text + "two")

    command = ApplyWordsCommand(len(doc.text) - 3, 3, payload["words"], payload["sources"])
    doc.undo_stack.push(command)

    assert command.clip_id is not None
    assert doc.source_path(A) == SOURCES[A]["path"]
    segment, = doc.get_clip(command.clip_id).segments
    assert segment.range == [1.3, 1.6] and segment.audio_path == SOURCES[A]["path"]
    # The clip that was already there plays from the relinked file too.
    assert [s.audio_path for s in clip.segments] == [SOURCES[A]["path"]]

    doc.undo_stack.undo()
    assert doc.source_path(A) is None
    assert imported.missing_sources(doc) == [A]


def test_add_sources_keeps_a_known_source_that_has_a_file():
    doc, _clip = _doc("one", [[0, 3, A, 1.0, 1.3]], {A: {"path": "/p/audio/imported/a.wav"}})
    assert doc.add_sources({A: {"path": "/elsewhere/a.wav"}, B: SOURCES[B]}) == [B]
    assert doc.source_path(A) == "/p/audio/imported/a.wav"
    # Nothing to relink with: an entry without a path.
    doc.settings["sources"][A] = {"path": None}
    assert doc.add_sources({A: {"path": None}}) == []
    assert doc.source_path(A) is None


# -- import commit ------------------------------------------------------------------


def test_import_recording_appends_one_imported_clip_per_row_with_the_pauses_kept():
    host = Character.from_preset_dict("Host", {})
    doc = Document.from_plain_text("Notes", characters=[host])
    first_text, first_words = run_from_asr_words([("Hello", 0.0, 0.4), ("there", 0.5, 0.9)], A)
    second_text, second_words = run_from_asr_words([("Next", 2.4, 2.8)], A)
    command = ImportRecordingCommand(
        [{"text": first_text, "words": first_words, "character_id": host.id},
         {"text": second_text, "words": second_words, "character_id": host.id}],
        {A: SOURCES[A]})

    doc.undo_stack.push(command)

    assert doc.text == "Notes\n\nHello there\n\nNext"
    first, second = (doc.get_clip(i) for i in command.clip_ids)
    assert first.source == second.source == "imported"
    assert first.track_id == doc.track_for_character(host.id)
    assert [s.range for s in first.segments] == [[0.0, 0.9]]
    assert [s.range for s in second.segments] == [[2.4, 2.8]]
    assert first.gap_before_s is None and second.gap_before_s == pytest.approx(1.5)
    assert doc.sources == {A: SOURCES[A]}
    assert doc.dirty_clips() == []

    doc.undo_stack.undo()
    assert doc.text == "Notes" and doc.clips == [] and "sources" not in doc.settings and doc.tracks == []
    doc.undo_stack.redo()
    assert [c.id for c in doc.clips] == command.clip_ids


# -- playback -----------------------------------------------------------------------


def test_segment_plays_crossfade_the_joins_of_an_imported_clip():
    clip = Clip(source="imported", segments=[
        Segment(order_index=0, audio_path="a.wav", range=[0.0, 0.5]),
        Segment(order_index=1, audio_path="a.wav", range=[1.0, 1.5]),
        Segment(order_index=2, audio_path="a.wav", range=[2.0, 2.5]),
    ])
    plays = segment_plays(clip, fade_in_s=0.1, fade_out_s=0.2)
    assert [p.range_s for p in plays] == [(0.0, 0.5), (1.0, 1.5), (2.0, 2.5)]
    assert [p.play_range_s for p in plays] == [(0.0, 0.5 + JOIN_CROSSFADE_S), (1.0, 1.5 + JOIN_CROSSFADE_S),
                                               (2.0, 2.5)]
    assert [(p.fade_in_s, p.fade_out_s) for p in plays] == [
        (0.1, JOIN_CROSSFADE_S), (JOIN_CROSSFADE_S, JOIN_CROSSFADE_S), (JOIN_CROSSFADE_S, 0.2)]


def test_segment_plays_give_a_looped_bed_its_passes_without_a_crossfade(tmp_path):
    path = tmp_path / "bed.wav"
    sf.write(str(path), np.zeros(8000, dtype=np.float32), 8000)
    bed = Clip(source="imported", original_audio_path=str(path),
               overrides={"loop": True, "loop_length_s": 2.5})
    plays = segment_plays(bed, fade_in_s=0.1, fade_out_s=0.2)
    assert len(plays) == 3
    assert all(p.play_range_s == p.range_s for p in plays)
    assert [(p.fade_in_s, p.fade_out_s) for p in plays] == [(0.1, 0.0), (0.0, 0.0), (0.0, 0.2)]


def test_segment_plays_leave_a_generated_clip_as_it_was():
    clip = Clip(segments=[Segment(order_index=1, audio_path="b.wav"), Segment(order_index=0, audio_path="a.wav"),
                          Segment(order_index=2)])
    plays = segment_plays(clip, fade_in_s=0.1, fade_out_s=0.2)
    assert [p.segment.audio_path for p in plays] == ["a.wav", "b.wav"]
    assert [(p.play_range_s, p.fade_in_s, p.fade_out_s) for p in plays] == [(None, 0.1, 0.0), (None, 0.0, 0.2)]


# -- import grouping and review ---------------------------------------------------


def test_group_asr_words_ends_a_clip_at_a_sentence_end_a_long_pause_or_the_length_cap():
    words = [("Hello", 0.0, 0.3), ("there.", 0.3, 0.6), ("How", 0.7, 0.9), ("are", 0.9, 1.1),
             ("you", 2.0, 2.3), ("today?\"", 2.3, 2.7), ("Fine", 3.0, 3.2), ("", 3.2, 3.3), ("bad",)]
    groups = imported.group_asr_words(words)
    assert [[w[0] for w in g] for g in groups] == [["Hello", "there."], ["How", "are"], ["you", "today?\""],
                                                  ["Fine"]]
    long = [(f"w{i}", float(i), i + 0.9) for i in range(40)]
    assert [len(g) for g in imported.group_asr_words(long, max_s=10.0)] == [10, 10, 10, 10]


def test_realign_words_keeps_the_times_of_a_corrected_line():
    text, words = run_from_asr_words([("Helo", 1.0, 1.4), ("wrld", 1.4, 1.9)], A)
    heard = imported.heard_words(text, words)
    assert heard == [("Helo", 1.0, 1.4), ("wrld", 1.4, 1.9)]
    # As many words as were heard: each keeps the time of the one in its place.
    assert imported.realign_words("Hello world", heard, A) == [[0, 5, A, 1.0, 1.4], [6, 11, A, 1.4, 1.9]]
    # A word dropped: the matched one keeps its times exactly.
    assert imported.realign_words("wrld", heard, A) == [[0, 4, A, 1.4, 1.9]]
    # A word added in front, before anything matched, gets what was heard
    # before the first match; one added past the end was never heard and
    # stays untimed.
    early = [("uh", 0.2, 0.6)] + heard
    assert imported.realign_words("Hi there Helo wrld", early, A) == [
        [0, 2, A, 0.2, 0.6], [3, 8, A, 0.6, 1.0], [9, 13, A, 1.0, 1.4], [14, 18, A, 1.4, 1.9]]
    assert imported.realign_words("So Helo wrld", heard, A) == [[3, 7, A, 1.0, 1.4], [8, 12, A, 1.4, 1.9]]
    assert imported.realign_words("Helo wrld again", heard, A) == [[0, 4, A, 1.0, 1.4], [5, 9, A, 1.4, 1.9]]


def test_untimed_gaps_are_untagged_text_inside_a_recording_paragraph():
    clip = Clip(source="imported")
    doc = Document(runs=[Run("Intro line\n"),
                         Run("Hello there, ", clip.id, "imported", words=[[0, 5, A, 0.0, 0.4], [6, 11, A, 0.4, 0.8]]),
                         Run("my friend "),
                         Run("how are you", clip.id, "imported", words=[[0, 3, A, 0.8, 1.0]]),
                         Run("\n\nOutro")],
                   clips=[clip], settings={"sources": dict(SOURCES)})
    start = len("Intro line\nHello there, ")
    assert imported.untimed_gaps(doc) == [(start, start + len("my friend "))]
    assert imported.untimed_gaps(Document.from_plain_text("plain\ntext")) == []
