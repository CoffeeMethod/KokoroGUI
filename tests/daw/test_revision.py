"""The change counters the derived caches key on (kokoro_gui/daw/revision.py,
Claude/PLAN_performance.md)."""
import copy

from kokoro_gui.daw import revision
from kokoro_gui.daw.models import Character, Clip, Document, Run, Segment
from kokoro_gui.daw.serialization import document_from_dict, document_to_dict
from kokoro_gui.daw.undo import SetFieldCommand


def _moved(fn) -> tuple:
    text, model = revision.TEXT, revision.MODEL
    fn()
    return revision.TEXT != text, revision.MODEL != model


def test_a_runs_text_moves_text_and_its_words_move_model():
    run = Run(text="hi")
    assert _moved(lambda: setattr(run, "text", "hello")) == (True, False)
    assert _moved(lambda: setattr(run, "clip_id", "c")) == (True, False)
    assert _moved(lambda: setattr(run, "words", [])) == (False, True)


def test_a_clip_or_segment_field_moves_model_and_an_id_moves_text():
    clip = Clip()
    segment = Segment()
    assert _moved(lambda: setattr(clip, "status", "approved")) == (False, True)
    assert _moved(lambda: setattr(segment, "audio_path", "x.wav")) == (False, True)
    assert _moved(lambda: setattr(clip, "id", "new-id"))[0] is True


def test_reassigning_the_run_or_clip_list_moves_text():
    doc = Document.from_plain_text("hello")
    assert _moved(lambda: setattr(doc, "runs", [Run(text="bye")]))[0] is True
    assert _moved(lambda: setattr(doc, "clips", []))[0] is True
    assert _moved(lambda: setattr(doc, "settings", {"gap_s": 1.0})) == (False, True)


def test_every_undo_stack_step_moves_both_counters():
    """A command can edit a dict field in place, which no attribute set
    reports."""
    doc = Document.from_plain_text("hello")
    assert _moved(lambda: doc.undo_stack.push(SetFieldCommand("document", None, "settings", 2.0, key="gap_s"))) \
        == (True, True)
    assert _moved(doc.undo_stack.undo) == (True, True)
    assert _moved(doc.undo_stack.redo) == (True, True)
    assert _moved(doc.touch) == (True, True)


def test_a_present_file_is_remembered_until_files_moves(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"x")
    assert revision.file_exists(str(path))
    path.unlink()
    assert revision.file_exists(str(path))  # remembered
    revision.bump_files()
    assert not revision.file_exists(str(path))


def test_a_missing_file_is_asked_again_every_time(tmp_path):
    path = tmp_path / "b.wav"
    assert not revision.file_exists(str(path))
    path.write_bytes(b"x")
    assert revision.file_exists(str(path))  # a Generate's new file shows at once


def test_tracking_leaves_deepcopy_equality_and_the_saved_shape_alone():
    alice = Character.from_preset_dict("Alice", {"voice": "af_heart"})
    doc = Document.from_plain_text("one two three", characters=[alice])
    doc.assign_character_to_range(0, 3, alice.id)
    doc.clips[0].segments = [Segment(text="one", audio_path="x.wav", duration=1.0)]

    copied = copy.deepcopy(doc.runs)
    assert copied == doc.runs and copied[0] is not doc.runs[0]
    reloaded = document_from_dict(document_to_dict(doc))
    assert document_to_dict(reloaded) == document_to_dict(doc)
    assert reloaded.text == doc.text
    assert reloaded.clip_text(reloaded.clips[0]) == "one"
