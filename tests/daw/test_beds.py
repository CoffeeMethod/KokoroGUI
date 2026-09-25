"""Music beds (phase 5 P2, grill Q30): the clip shape, trim and loop as
virtual segments, placement, lanes and the import command."""
import numpy as np
import pytest
import soundfile as sf

from kokoro_gui.daw import beds
from kokoro_gui.daw.arrangement import clip_audio_duration_s, compute_arrangement
from kokoro_gui.daw.lanes import lane_numbers
from kokoro_gui.daw.models import PLACEHOLDER, Character, Clip, Document, Run, Segment, Track
from kokoro_gui.daw.undo import AssignCharacterCommand, ImportBedCommand, SetFieldCommand

RATE = 8000


@pytest.fixture
def bed_file(tmp_path):
    path = str(tmp_path / "bed.wav")
    sf.write(path, np.zeros(4 * RATE, dtype=np.float32), RATE)
    return path


def test_a_bed_is_an_imported_clip_with_a_file_and_owns_a_placeholder_run(bed_file):
    bed = Clip(source="imported", original_audio_path=bed_file)
    assert bed.is_bed and bed.has_placeholder and bed.run_kind == PLACEHOLDER
    # An imported recording (phase 5 P3) keeps its timing on its runs.
    recording = Clip(source="imported")
    assert not recording.is_bed and recording.run_kind == "imported"
    assert not Clip().is_bed and not Clip(source="nested").is_bed


def test_bed_segments_play_the_whole_file_by_default(bed_file):
    bed = Clip(source="imported", original_audio_path=bed_file)
    segments = beds.bed_segments(bed)
    assert [s.range for s in segments] == [[0.0, 4.0]]
    assert beds.audio_file_seconds(bed_file) == 4.0
    assert clip_audio_duration_s(bed) == 4.0
    assert bed.segments == []  # nothing stored


def test_trim_is_clamped_to_the_file(bed_file):
    bed = Clip(source="imported", original_audio_path=bed_file, overrides={"trim": [1.0, 9.0]})
    assert beds.bed_trim(bed, 4.0) == (1.0, 4.0)
    assert [s.range for s in beds.bed_segments(bed)] == [[1.0, 4.0]]
    bed.overrides["trim"] = [2.0, 2.01]  # shorter than MIN_BED_S
    assert beds.bed_trim(bed, 4.0) == (0.0, 4.0)
    bed.overrides["trim"] = "junk"
    assert beds.bed_trim(bed, 4.0) == (0.0, 4.0)


def test_a_loop_repeats_the_trim_range_to_its_length(bed_file):
    bed = Clip(source="imported", original_audio_path=bed_file,
               overrides={"trim": [1.0, 3.0], "loop": True, "loop_length_s": 5.0})
    segments = beds.bed_segments(bed)
    assert [s.range for s in segments] == [[1.0, 3.0], [1.0, 3.0], [1.0, 2.0]]
    assert clip_audio_duration_s(bed) == pytest.approx(5.0)
    del bed.overrides["loop_length_s"]
    assert [s.range for s in beds.bed_segments(bed)] == [[1.0, 3.0]]


def test_a_missing_file_has_no_segments(tmp_path):
    bed = Clip(source="imported", original_audio_path=str(tmp_path / "gone.wav"))
    assert beds.bed_segments(bed) == []
    assert clip_audio_duration_s(bed) is None


def test_playable_segments_of_an_ordinary_clip_are_its_own_in_order():
    a, b, c = Segment(1, audio_path="b.wav"), Segment(0, audio_path="a.wav"), Segment(2)
    clip = Clip(segments=[a, b, c])
    assert beds.playable_segments(clip) == [b, a]


def _doc_with_bed(bed_file, **clip_kwargs):
    narrator = Character.from_preset_dict("N", {})
    first = Clip(character_id=narrator.id, segments=[Segment(0, audio_path="x", duration=2.0)])
    second = Clip(character_id=narrator.id, segments=[Segment(0, audio_path="y", duration=1.0)])
    bed = Clip(source="imported", original_audio_path=bed_file, **clip_kwargs)
    doc = Document(runs=[Run("One.", first.id, "generated"), Run("Bed", bed.id, PLACEHOLDER),
                         Run("Two.", second.id, "generated")],
                   clips=[first, bed, second], characters=[narrator], settings={"gap_s": 0.0})
    return doc, first, bed, second


def test_a_bed_placed_in_time_does_not_push_the_read_through(bed_file):
    doc, first, bed, second = _doc_with_bed(bed_file, timeline_timestamp=0.0, pinned=True)
    placed = compute_arrangement(doc, chars_per_second=10.0).by_clip_id()
    assert placed[bed.id].start_s == 0.0 and placed[bed.id].duration_s == 4.0
    assert placed[second.id].start_s == 2.0  # right after the first clip, not after the bed


def test_a_bed_in_text_order_is_a_stinger_between_lines(bed_file):
    doc, first, bed, second = _doc_with_bed(bed_file)
    placed = compute_arrangement(doc, chars_per_second=10.0).by_clip_id()
    assert placed[bed.id].start_s == 2.0
    assert placed[second.id].start_s == 6.0


def test_the_lane_rule_leaves_a_bed_on_its_track(bed_file):
    doc, first, bed, second = _doc_with_bed(bed_file)
    lanes = lane_numbers(doc, 3)
    assert bed.id not in lanes
    assert lanes == {first.id: 1, second.id: 1}


def test_a_bed_line_refuses_a_character_assignment(bed_file):
    doc, _first, _bed, _second = _doc_with_bed(bed_file)
    assert doc.overlaps_nested(4, 6)
    with pytest.raises(ValueError):
        doc.assign_character_to_range(4, 6, None)
    assert doc.placeholder_extent(_bed.id) == (4, 7)


def test_typing_after_a_bed_line_is_not_part_of_it(bed_file):
    doc, _first, bed, _second = _doc_with_bed(bed_file)
    new_text = doc.text[:7] + "x" + doc.text[7:]
    doc.replace_text(7, 0, 1, new_text)
    assert doc.clip_text(bed) == "Bed"


def test_import_bed_command_appends_a_paragraph_on_a_music_track_and_undoes(bed_file):
    doc = Document.from_plain_text("Hello there.")
    command = ImportBedCommand(bed_file, "theme", at_s=2.5)
    doc.undo_stack.push(command)

    assert doc.text == "Hello there.\n\ntheme"
    bed = doc.get_clip(command.clip_id)
    assert bed.is_bed and bed.pinned and bed.timeline_timestamp == 2.5
    assert bed.original_audio_path == bed_file and bed.segments == []
    track = doc.get_track(bed.track_id)
    assert (track.name, track.role) == ("Music", "music")
    assert [r.kind for r in doc.runs if r.clip_id == bed.id] == [PLACEHOLDER]
    assert doc.dirty_clips() == []

    doc.undo_stack.undo()
    assert doc.text == "Hello there." and doc.clips == [] and doc.tracks == []
    doc.undo_stack.redo()
    assert doc.get_clip(command.clip_id).is_bed

    # A second bed reuses the Music track.
    second = ImportBedCommand(bed_file, "sting")
    doc.undo_stack.push(second)
    assert doc.text == "Hello there.\n\ntheme\n\nsting"
    assert doc.get_clip(second.clip_id).track_id == bed.track_id
    assert len(doc.tracks) == 1
    doc.undo_stack.undo()
    assert len(doc.tracks) == 1


def test_import_bed_into_an_empty_document(bed_file):
    doc = Document()
    doc.undo_stack.push(ImportBedCommand(bed_file, "theme"))
    assert doc.text == "theme"


def test_track_duck_is_an_undoable_field():
    track = Track(name="Music")
    doc = Document(tracks=[track])
    assert track.duck is False
    doc.undo_stack.push(SetFieldCommand("track", track.id, "duck", True))
    assert track.duck is True
    doc.undo_stack.undo()
    assert track.duck is False


def test_assign_character_command_never_splits_a_bed(bed_file):
    doc, _first, bed, _second = _doc_with_bed(bed_file)
    with pytest.raises(ValueError):
        doc.undo_stack.push(AssignCharacterCommand(0, 7, None))
    assert doc.get_clip(bed.id) is not None


def test_a_bed_gets_no_subtitle(bed_file, tmp_path):
    from kokoro_gui.daw.mixdown import write_srt

    doc, _first, _bed, _second = _doc_with_bed(bed_file, timeline_timestamp=0.0, pinned=True)
    path = write_srt(doc, compute_arrangement(doc, chars_per_second=10.0), str(tmp_path / "out.srt"))
    text = open(path, encoding="utf-8").read()
    assert "One." in text and "Two." in text and "Bed" not in text
