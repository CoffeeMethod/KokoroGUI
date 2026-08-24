"""Tests for kokoro_gui/daw/undo.py - the plain-Python Command/UndoStack
pair behind item 4 ("Undo/redo") of the DAW-for-text redesign's
remaining-work roadmap. Mirrors tests/daw/test_assign_character.py's
fixtures/conventions - no Qt, no QT_QPA_PLATFORM needed."""
import json

from kokoro_gui.daw.models import Character, Clip, Document, Track
from kokoro_gui.daw.serialization import document_from_dict, document_to_dict, load_document, save_document
from kokoro_gui.daw.undo import AssignCharacterCommand, SetClipFxCommand, TextEditCommand, UndoStack


def _document_with_characters(text="0123456789ABCDEFGHIJ"):
    alice = Character.from_preset_dict("Alice", {"voice": "af_bella"})
    bob = Character.from_preset_dict("Bob", {"voice": "am_michael"})
    tracks = [Track(name="Alice", character_id=alice.id), Track(name="Bob", character_id=bob.id)]
    doc = Document(text=text, characters=[alice, bob], tracks=tracks)
    return doc, alice, bob


# ---------------------------------------------------------------------------
# AssignCharacterCommand round trip
# ---------------------------------------------------------------------------

def test_assign_character_command_push_creates_clip():
    doc, alice, _ = _document_with_characters()
    stack = UndoStack(doc)

    stack.push(AssignCharacterCommand(2, 6, alice.id))

    assert len(doc.clips) == 1
    clip = doc.clips[0]
    assert (clip.start_offset, clip.end_offset) == (2, 6)
    assert clip.character_id == alice.id


def test_assign_character_command_undo_removes_clip():
    doc, alice, _ = _document_with_characters()
    stack = UndoStack(doc)
    stack.push(AssignCharacterCommand(2, 6, alice.id))

    stack.undo()

    assert doc.clips == []


def test_assign_character_command_redo_restores_clip_with_same_properties():
    doc, alice, _ = _document_with_characters()
    stack = UndoStack(doc)
    stack.push(AssignCharacterCommand(2, 6, alice.id))
    stack.undo()

    stack.redo()

    assert len(doc.clips) == 1
    clip = doc.clips[0]
    assert (clip.start_offset, clip.end_offset) == (2, 6)
    assert clip.character_id == alice.id


def test_assign_character_command_undo_restores_split_leftovers_verbatim():
    doc, alice, bob = _document_with_characters()
    stack = UndoStack(doc)
    original = doc.assign_character_to_range(0, 10, alice.id)
    original.segments = ["pretend-generated"]  # simulate previously-generated audio
    original_id = original.id

    stack.push(AssignCharacterCommand(5, 10, bob.id))
    assert len(doc.clips) == 2  # alice leftover [0,5) + bob [5,10)

    stack.undo()

    assert len(doc.clips) == 1
    restored = doc.clips[0]
    assert restored.id == original_id
    assert restored.start_offset == 0
    assert restored.end_offset == 10
    assert restored.character_id == alice.id
    assert restored.segments == ["pretend-generated"]  # cache-hit-preserving restore


# ---------------------------------------------------------------------------
# TextEditCommand round trip
# ---------------------------------------------------------------------------

def test_text_edit_command_push_changes_document_text():
    doc, _, _ = _document_with_characters(text="hello world")
    stack = UndoStack(doc)

    command = TextEditCommand(5, 0, 6, old_text="hello world", new_text="hello, world world")
    stack.push(command)

    assert doc.text == "hello, world world"


def test_text_edit_command_undo_restores_original_text():
    doc, _, _ = _document_with_characters(text="hello world")
    stack = UndoStack(doc)
    command = TextEditCommand(5, 0, 6, old_text="hello world", new_text="hello, world world")
    stack.push(command)

    stack.undo()

    assert doc.text == "hello world"


def test_text_edit_command_redo_reapplies_insertion():
    doc, _, _ = _document_with_characters(text="hello world")
    stack = UndoStack(doc)
    command = TextEditCommand(5, 0, 6, old_text="hello world", new_text="hello, world world")
    stack.push(command)
    stack.undo()

    stack.redo()

    assert doc.text == "hello, world world"


def test_text_edit_command_lossy_offset_edge_case_restores_exact_clip_offsets():
    """apply_text_change's docstring documents that an offset falling
    strictly inside a replaced range collapses to the edit's start - a
    naive reverse replay would NOT restore a surviving clip's boundary that
    sat inside the original edited range. This is the scenario:
    Clip covers [5, 15). Edit replaces [10, 20) - it starts inside the clip
    and ends past it, so the clip survives (not fully consumed) but its end
    offset collapses to the edit's start (10) on the forward pass."""
    clip = Clip(start_offset=5, end_offset=15)
    text = "0123456789ABCDEFGHIJKLMNOPQRST"  # len 30
    doc = Document(text=text, clips=[clip])
    stack = UndoStack(doc)

    old_text = text
    new_text = text[:10] + "XYZ" + text[20:]  # replace [10,20) (10 chars) with "XYZ" (3 chars)
    command = TextEditCommand(10, 10, 3, old_text=old_text, new_text=new_text)
    stack.push(command)

    # Forward edit: clip survives, start stays 5, end collapses to 10.
    assert clip.start_offset == 5
    assert clip.end_offset == 10

    stack.undo()

    assert doc.text == old_text
    restored = doc.get_clip(clip.id)
    assert restored is not None
    assert restored.start_offset == 5
    assert restored.end_offset == 15  # exact pre-edit value, not whatever naive replay would give


def test_text_edit_command_undo_restores_fully_consumed_clip():
    clip = Clip(start_offset=6, end_offset=11)
    clip.segments = ["pretend-generated"]
    text = "hello world"
    doc = Document(text=text, clips=[clip])
    stack = UndoStack(doc)

    command = TextEditCommand(6, 5, 0, old_text=text, new_text="hello ")
    stack.push(command)
    assert doc.clips == []

    stack.undo()

    assert len(doc.clips) == 1
    restored = doc.clips[0]
    assert restored.id == clip.id
    assert (restored.start_offset, restored.end_offset) == (6, 11)
    assert restored.segments == ["pretend-generated"]


# ---------------------------------------------------------------------------
# SetClipFxCommand round trip (item 5, "Per-clip FX button")
# ---------------------------------------------------------------------------

def test_set_clip_fx_command_push_sets_fx_override():
    clip = Clip(start_offset=0, end_offset=5)
    doc = Document(text="hello", clips=[clip])
    stack = UndoStack(doc)

    stack.push(SetClipFxCommand(clip.id, {"reverb_enabled": True, "comp_threshold": -10}))

    assert clip.fx_override == {"reverb_enabled": True, "comp_threshold": -10}


def test_set_clip_fx_command_undo_restores_none_when_previously_unset():
    clip = Clip(start_offset=0, end_offset=5)
    doc = Document(text="hello", clips=[clip])
    stack = UndoStack(doc)
    stack.push(SetClipFxCommand(clip.id, {"reverb_enabled": True}))

    stack.undo()

    assert clip.fx_override is None


def test_set_clip_fx_command_redo_reapplies_fx_override():
    clip = Clip(start_offset=0, end_offset=5)
    doc = Document(text="hello", clips=[clip])
    stack = UndoStack(doc)
    stack.push(SetClipFxCommand(clip.id, {"reverb_enabled": True}))
    stack.undo()

    stack.redo()

    assert clip.fx_override == {"reverb_enabled": True}


def test_set_clip_fx_command_clears_a_previously_set_override():
    clip = Clip(start_offset=0, end_offset=5, fx_override={"reverb_enabled": True})
    doc = Document(text="hello", clips=[clip])
    stack = UndoStack(doc)

    stack.push(SetClipFxCommand(clip.id, None))

    assert clip.fx_override is None


def test_set_clip_fx_command_undo_restores_previous_override_after_clear():
    clip = Clip(start_offset=0, end_offset=5, fx_override={"reverb_enabled": True, "comp_ratio": 4})
    doc = Document(text="hello", clips=[clip])
    stack = UndoStack(doc)
    stack.push(SetClipFxCommand(clip.id, None))
    assert clip.fx_override is None

    stack.undo()

    assert clip.fx_override == {"reverb_enabled": True, "comp_ratio": 4}


def test_set_clip_fx_command_does_not_alias_caller_dict():
    clip = Clip(start_offset=0, end_offset=5)
    doc = Document(text="hello", clips=[clip])
    stack = UndoStack(doc)
    fx_values = {"reverb_enabled": True}

    stack.push(SetClipFxCommand(clip.id, fx_values))
    fx_values["reverb_enabled"] = False  # mutate the caller's own dict afterward

    assert clip.fx_override == {"reverb_enabled": True}  # unaffected


# ---------------------------------------------------------------------------
# Mixed sequence
# ---------------------------------------------------------------------------

def test_mixed_sequence_type_then_assign_undo_twice_redo_once():
    doc, alice, _ = _document_with_characters(text="hello world")
    stack = UndoStack(doc)

    # 1. Type text: insert ", there" after "hello" (position 5).
    text_cmd = TextEditCommand(5, 0, 7, old_text="hello world", new_text="hello, there world")
    stack.push(text_cmd)
    assert doc.text == "hello, there world"

    # 2. Assign a character to a range of the new text.
    assign_cmd = AssignCharacterCommand(0, 5, alice.id)
    stack.push(assign_cmd)
    assert len(doc.clips) == 1

    # Undo twice: first reverts the assignment, then the text edit.
    stack.undo()
    assert doc.clips == []
    assert doc.text == "hello, there world"

    stack.undo()
    assert doc.text == "hello world"

    # Redo once: only the text edit comes back, not the assignment.
    stack.redo()
    assert doc.text == "hello, there world"
    assert doc.clips == []


# ---------------------------------------------------------------------------
# can_undo/can_redo and push()-clears-redo
# ---------------------------------------------------------------------------

def test_can_undo_can_redo_track_state_through_push_undo_redo():
    doc, alice, _ = _document_with_characters()
    stack = UndoStack(doc)
    assert stack.can_undo() is False
    assert stack.can_redo() is False

    stack.push(AssignCharacterCommand(0, 5, alice.id))
    assert stack.can_undo() is True
    assert stack.can_redo() is False

    stack.undo()
    assert stack.can_undo() is False
    assert stack.can_redo() is True

    stack.redo()
    assert stack.can_undo() is True
    assert stack.can_redo() is False


def test_push_clears_redo_stack():
    doc, alice, bob = _document_with_characters()
    stack = UndoStack(doc)
    stack.push(AssignCharacterCommand(0, 5, alice.id))
    stack.undo()
    assert stack.can_redo() is True

    stack.push(AssignCharacterCommand(10, 15, bob.id))

    assert stack.can_redo() is False
    stack.redo()  # no-op - nothing to redo
    assert len(doc.clips) == 1
    assert doc.clips[0].character_id == bob.id


def test_undo_on_empty_stack_is_a_noop():
    doc, _, _ = _document_with_characters()
    stack = UndoStack(doc)
    stack.undo()  # must not raise
    assert doc.clips == []


def test_redo_on_empty_stack_is_a_noop():
    doc, _, _ = _document_with_characters()
    stack = UndoStack(doc)
    stack.redo()  # must not raise
    assert doc.clips == []


# ---------------------------------------------------------------------------
# undo_stack is never serialized
# ---------------------------------------------------------------------------

def test_document_to_dict_never_includes_undo_stack():
    doc, alice, _ = _document_with_characters()
    doc.undo_stack.push(AssignCharacterCommand(0, 5, alice.id))

    data = document_to_dict(doc)

    assert "undo_stack" not in data
    assert "undo_stack" not in json.dumps(data)


def test_document_from_dict_constructs_a_fresh_undo_stack():
    doc, alice, _ = _document_with_characters()
    doc.undo_stack.push(AssignCharacterCommand(0, 5, alice.id))

    data = document_to_dict(doc)
    reloaded = document_from_dict(data)

    assert reloaded.undo_stack is not None
    assert reloaded.undo_stack.can_undo() is False
    assert reloaded.undo_stack.can_redo() is False


def test_save_and_load_document_round_trip_excludes_undo_stack(tmp_path):
    doc, alice, _ = _document_with_characters()
    doc.undo_stack.push(AssignCharacterCommand(0, 5, alice.id))

    path = str(tmp_path / "document.json")
    save_document(doc, path)

    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    assert "undo_stack" not in raw

    reloaded = load_document(path)
    assert reloaded is not None
    assert len(reloaded.clips) == 1
    assert reloaded.undo_stack.can_undo() is False
