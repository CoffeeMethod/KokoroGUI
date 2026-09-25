"""Tests for the seconds-axis undo commands (UI9) and the named FX
override (UI shell pass) in kokoro_gui/daw/undo.py."""
from kokoro_gui.daw.models import Character, Clip, Document, Run, Track
from kokoro_gui.daw.undo import MoveClipBeforeCommand, SetClipFxCommand, SetClipTimestampCommand


def _doc():
    alice = Character.from_preset_dict("Alice", {})
    track = Track(name="A", character_id=alice.id)
    a = Clip(character_id=alice.id, track_id=track.id)
    b = Clip(character_id=alice.id, track_id=track.id)
    c = Clip(character_id=alice.id, track_id=track.id)
    runs = [
        Run(text="AAA ", clip_id=a.id, kind="generated"),
        Run(text="plain "),
        Run(text="BBB ", clip_id=b.id, kind="generated"),
        Run(text="CCC", clip_id=c.id, kind="generated"),
    ]
    doc = Document(runs=runs, clips=[a, b, c], tracks=[track], characters=[alice])
    return doc, a, b, c


def test_set_clip_timestamp_is_undoable():
    doc, a, _b, _c = _doc()
    doc.undo_stack.push(SetClipTimestampCommand(a.id, 3.5))
    assert a.timeline_timestamp == 3.5
    doc.undo_stack.undo()
    assert a.timeline_timestamp is None
    doc.undo_stack.redo()
    assert a.timeline_timestamp == 3.5
    doc.undo_stack.push(SetClipTimestampCommand(a.id, None))
    assert a.timeline_timestamp is None


def test_move_clip_before_reorders_runs_and_pins_timestamp():
    doc, a, b, c = _doc()

    doc.undo_stack.push(MoveClipBeforeCommand(c.id, a.id, timestamp=0.0))

    assert doc.text == "CCCAAA plain BBB "
    assert doc.clip_extent(c.id) == (0, 3)
    assert doc.clip_extent(a.id) == (3, 7)
    assert c.timeline_timestamp == 0.0
    doc.undo_stack.undo()
    assert doc.text == "AAA plain BBB CCC"
    assert doc.get_clip(c.id).timeline_timestamp is None


def test_move_clip_before_leaves_untagged_text_in_place():
    doc, a, b, c = _doc()
    doc.undo_stack.push(MoveClipBeforeCommand(b.id, a.id))
    assert doc.text == "BBB AAA plain CCC"


def test_move_clip_before_itself_or_unknown_target_is_a_noop():
    doc, a, _b, _c = _doc()
    before = doc.text
    doc.undo_stack.push(MoveClipBeforeCommand(a.id, a.id))
    doc.undo_stack.push(MoveClipBeforeCommand(a.id, "nope"))
    assert doc.text == before


def test_set_clip_fx_records_and_clears_the_preset_name():
    doc, a, _b, _c = _doc()
    doc.undo_stack.push(SetClipFxCommand(a.id, {"reverb_enabled": True}, preset_name="Hall"))
    assert a.fx_override == {"reverb_enabled": True}
    assert a.overrides["fx_preset"] == "Hall"

    doc.undo_stack.push(SetClipFxCommand(a.id, {"reverb_enabled": False}, preset_name=None))
    assert a.overrides["fx_preset"] == "Hall"  # an unnamed edit keeps the last name

    doc.undo_stack.push(SetClipFxCommand(a.id, None))
    assert a.fx_override is None
    assert "fx_preset" not in a.overrides

    doc.undo_stack.undo()
    assert a.overrides["fx_preset"] == "Hall"
    doc.undo_stack.undo()
    doc.undo_stack.undo()
    assert a.fx_override is None
    assert "fx_preset" not in a.overrides


def test_ripple_command_shifts_timestamps_and_undo_restores_them():
    from kokoro_gui.daw.undo import RippleCommand

    doc, a, b, c = _doc()
    b.timeline_timestamp = 3.0
    c.timeline_timestamp = 0.2
    doc.undo_stack.push(RippleCommand({b.id: 0.5, c.id: -1.0, a.id: 2.0}))

    assert b.timeline_timestamp == 3.5
    assert c.timeline_timestamp == 0.0  # never below zero
    assert a.timeline_timestamp is None  # text-ordered clips aren't touched

    doc.undo_stack.undo()
    assert (b.timeline_timestamp, c.timeline_timestamp) == (3.0, 0.2)


def test_clip_pinned_round_trips():
    from kokoro_gui.daw.serialization import document_from_dict, document_to_dict

    doc, a, _b, _c = _doc()
    a.pinned = True
    restored = document_from_dict(document_to_dict(doc))
    assert restored.get_clip(a.id).pinned is True
    assert all(not c.pinned for c in restored.clips if c.id != a.id)
