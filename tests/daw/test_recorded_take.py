"""A recorded guide take (plan 31, grill PG8): the dirty rule, AddTakeCommand
and keep_recorded_take, which stops a generation landing on its index from
writing over it."""
import os

from kokoro_gui.daw.dirty import is_clip_dirty, is_recorded_segment, is_recorded_take
from kokoro_gui.daw.models import Clip, Document, Segment
from kokoro_gui.daw.undo import AddTakeCommand, keep_recorded_take, next_take_index


def _config():
    return {"voice": "af_bella", "speed": 1.0, "lang_code": "a", "engine_id": "kokoro"}


def _recorded(tmp_path, name="abc.wav"):
    folder = tmp_path / "audio" / "imported"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"RIFF")
    return Segment(order_index=0, text="Hello there.", cache_key=None, audio_path=str(path), duration=1.5,
                   range=[3.0, 4.5])


def _generated(text="Hello there.", key="k"):
    return Segment(order_index=0, text=text, cache_key=key, audio_path=f"/p/audio/generated/{key}_0.wav",
                   duration=1.0)


def test_a_recorded_segment_has_no_key_and_lives_under_audio_imported(tmp_path):
    assert is_recorded_segment(_recorded(tmp_path))
    assert is_recorded_segment(Segment(audio_path="C:\\p\\audio\\imported\\a.wav"))
    assert not is_recorded_segment(Segment(cache_key="k", audio_path="/p/audio/imported/a.wav"))
    assert not is_recorded_segment(Segment(audio_path="/p/audio/generated/a.wav"))
    assert not is_recorded_segment(Segment())
    assert not is_recorded_take([])
    assert not is_recorded_take([_recorded(tmp_path), _generated()])


def test_the_audio_dir_matches_the_one_import_audio_file_writes_to():
    from kokoro_gui.daw import dirty
    from kokoro_gui.qt import project

    assert dirty.IMPORTED_AUDIO_DIR == project.AUDIO_IMPORTED


def test_a_clip_whose_active_take_is_recorded_is_never_stale(tmp_path):
    clip = Clip(segments=[_recorded(tmp_path)])
    assert not is_clip_dirty(clip, "Hello there.", _config())
    # Not for a text change, a voice change or any other generation input.
    assert not is_clip_dirty(clip, "Something else entirely.", {**_config(), "voice": "am_adam"})


def test_a_recorded_take_whose_file_is_gone_is_stale(tmp_path):
    segment = _recorded(tmp_path)
    os.remove(segment.audio_path)
    assert is_clip_dirty(Clip(segments=[segment]), "Hello there.", _config())


def test_a_parked_recorded_take_does_not_change_the_active_takes_check(tmp_path):
    clip = Clip(segments=[], takes={1: [_recorded(tmp_path)]})
    assert is_clip_dirty(clip, "Hello there.", _config())


def test_next_take_index_counts_parked_takes_and_an_active_take_with_audio(tmp_path):
    assert next_take_index(Clip()) == 0
    assert next_take_index(Clip(segments=[_generated()])) == 1
    assert next_take_index(Clip(segments=[_generated()], overrides={"take": 2}, takes={0: [_generated()]})) == 3
    assert next_take_index(Clip(takes={4: [_generated()]})) == 5


def _doc_with(clip):
    doc = Document()
    doc.clips.append(clip)
    return doc


def test_add_take_parks_the_active_take_makes_the_new_one_active_and_undoes_in_one_step(tmp_path):
    generated = [_generated()]
    clip = Clip(segments=generated)
    doc = _doc_with(clip)
    recorded = [_recorded(tmp_path)]

    doc.undo_stack.push(AddTakeCommand(clip.id, recorded))
    assert clip.segments == recorded and clip.overrides["take"] == 1
    assert clip.takes == {0: generated}

    doc.undo_stack.undo()
    assert clip.segments == generated and clip.takes == {} and "take" not in clip.overrides

    doc.undo_stack.redo()
    assert clip.segments == recorded and clip.overrides["take"] == 1 and clip.takes == {0: generated}


def test_add_take_on_a_clip_with_no_audio_is_take_one(tmp_path):
    clip = Clip()
    doc = _doc_with(clip)
    doc.undo_stack.push(AddTakeCommand(clip.id, [_recorded(tmp_path)]))
    assert "take" not in clip.overrides and clip.takes == {} and len(clip.segments) == 1
    doc.undo_stack.undo()
    assert clip.segments == [] and clip.takes == {}


def test_add_take_after_a_later_parked_take_restores_the_right_index(tmp_path):
    first, second = [_generated(key="a")], [_generated(key="b")]
    clip = Clip(segments=second, overrides={"take": 1}, takes={0: first})
    doc = _doc_with(clip)
    doc.undo_stack.push(AddTakeCommand(clip.id, [_recorded(tmp_path)]))
    assert clip.overrides["take"] == 2 and clip.takes == {0: first, 1: second}
    doc.undo_stack.undo()
    assert clip.segments == second and clip.overrides["take"] == 1 and clip.takes == {0: first}


def test_a_generation_landing_on_the_active_recorded_take_parks_it_first(tmp_path):
    recorded = [_recorded(tmp_path)]
    clip = Clip(segments=recorded, overrides={"take": 1}, takes={0: [_generated()]})
    keep_recorded_take(clip, 1)
    assert clip.segments == [] and clip.takes[2] == recorded and 0 in clip.takes


def test_a_generation_landing_on_a_parked_recorded_take_moves_it_aside(tmp_path):
    recorded = [_recorded(tmp_path)]
    generated = [_generated()]
    clip = Clip(segments=generated, takes={1: recorded})
    keep_recorded_take(clip, 1)
    assert clip.segments == generated and 1 not in clip.takes and clip.takes[2] == recorded


def test_keep_recorded_take_leaves_generated_takes_alone():
    generated, parked = [_generated(key="a")], [_generated(key="b")]
    clip = Clip(segments=generated, takes={1: parked})
    keep_recorded_take(clip, 0)
    keep_recorded_take(clip, 1)
    assert clip.segments == generated and clip.takes == {1: parked}
