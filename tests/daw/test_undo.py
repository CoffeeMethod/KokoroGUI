"""Tests for kokoro_gui/daw/undo.py - the plain-Python Command/UndoStack
pair behind item 4 ("Undo/redo") of the DAW-for-text redesign's
remaining-work roadmap. Mirrors tests/daw/test_assign_character.py's
fixtures/conventions - no Qt, no QT_QPA_PLATFORM needed.

Per Claude/PLAN_text_editor_redesign.md, `TextEditCommand` here models the
custom-stack's one remaining text-mutation use (the sub-range TTS replace
button - a programmatic, non-typing text replacement), not interactive
typing, which now rides the real GUI's native QTextDocument undo instead
(see kokoro_gui/qt/transcript_editor.py)."""
import json

from kokoro_gui.daw.models import Character, Clip, Document, Run, Track
from kokoro_gui.daw.serialization import document_from_dict, document_to_dict, load_document, save_document
from kokoro_gui.daw.undo import AssignCharacterCommand, SetClipFxCommand, TextEditCommand, UndoStack


def _document_with_characters(text="0123456789ABCDEFGHIJ"):
    alice = Character.from_preset_dict("Alice", {"voice": "af_bella"})
    bob = Character.from_preset_dict("Bob", {"voice": "am_michael"})
    tracks = [Track(name="Alice", character_id=alice.id), Track(name="Bob", character_id=bob.id)]
    doc = Document.from_plain_text(text, characters=[alice, bob], tracks=tracks)
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
    assert doc.clip_extent(clip.id) == (2, 6)
    assert clip.character_id == alice.id


def test_assign_character_command_undo_removes_clip():
    doc, alice, _ = _document_with_characters()
    stack = UndoStack(doc)
    stack.push(AssignCharacterCommand(2, 6, alice.id))

    stack.undo()

    assert doc.clips == []
    assert doc.clip_covering(3) is None


def test_assign_character_command_redo_restores_clip_with_same_properties():
    doc, alice, _ = _document_with_characters()
    stack = UndoStack(doc)
    stack.push(AssignCharacterCommand(2, 6, alice.id))
    stack.undo()

    stack.redo()

    assert len(doc.clips) == 1
    clip = doc.clips[0]
    assert doc.clip_extent(clip.id) == (2, 6)
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
    assert doc.clip_extent(restored.id) == (0, 10)
    assert restored.character_id == alice.id
    assert restored.segments == ["pretend-generated"]  # cache-hit-preserving restore


# ---------------------------------------------------------------------------
# TextEditCommand round trip
# ---------------------------------------------------------------------------

_INSERTED_AFTER_HELLO = ", there"


def _text_edit_insert_after_hello(old_text="hello world"):
    position = 5  # right after "hello"
    new_text = old_text[:position] + _INSERTED_AFTER_HELLO + old_text[position:]
    return TextEditCommand(position, 0, len(_INSERTED_AFTER_HELLO), new_text=new_text), new_text


def test_text_edit_command_push_changes_document_text():
    doc, _, _ = _document_with_characters(text="hello world")
    stack = UndoStack(doc)

    command, new_text = _text_edit_insert_after_hello()
    stack.push(command)

    assert doc.text == new_text


def test_text_edit_command_undo_restores_original_text():
    doc, _, _ = _document_with_characters(text="hello world")
    stack = UndoStack(doc)
    command, _new_text = _text_edit_insert_after_hello()
    stack.push(command)

    stack.undo()

    assert doc.text == "hello world"


def test_text_edit_command_redo_reapplies_insertion():
    doc, _, _ = _document_with_characters(text="hello world")
    stack = UndoStack(doc)
    command, new_text = _text_edit_insert_after_hello()
    stack.push(command)
    stack.undo()

    stack.redo()

    assert doc.text == new_text


def test_text_edit_command_undo_restores_exact_clip_tagging_across_a_split():
    """A clip covers [5, 15). The edit replaces [10, 20) - it starts inside
    the clip (position 10 sits strictly within it) and ends past it. The
    replacement text inherits the clip's tag (ordinary "typing inside a
    run extends it" behavior), so the clip survives, now covering its
    original [5, 10) portion plus the 3-character replacement. Undo must
    restore the clip's exact original extent, not whatever a naive
    reverse-replay of the edit would reconstruct."""
    text = "0123456789ABCDEFGHIJKLMNOPQRST"  # len 30
    doc, alice, _ = _document_with_characters(text=text)
    stack = UndoStack(doc)
    stack.push(AssignCharacterCommand(5, 15, alice.id))
    clip = doc.clips[0]

    new_text = text[:10] + "XYZ" + text[20:]  # replace [10,20) (10 chars) with "XYZ" (3 chars)
    command = TextEditCommand(10, 10, 3, new_text=new_text)
    stack.push(command)

    # Forward edit: clip survives, now [5, 10) plus the inherited "XYZ".
    assert doc.clip_extent(clip.id) == (5, 13)
    assert doc.clip_text(clip) == "56789XYZ"

    stack.undo()

    assert doc.text == text
    restored = doc.get_clip(clip.id)
    assert restored is not None
    assert doc.clip_extent(restored.id) == (5, 15)  # exact pre-edit extent, not a naive replay's guess


def test_text_edit_command_undo_restores_fully_consumed_clip():
    doc, alice, _ = _document_with_characters(text="hello world")
    stack = UndoStack(doc)
    clip = doc.assign_character_to_range(6, 11, alice.id)
    clip.segments = ["pretend-generated"]

    command = TextEditCommand(6, 5, 0, new_text="hello ")
    stack.push(command)
    assert doc.clips == []

    stack.undo()

    assert len(doc.clips) == 1
    restored = doc.clips[0]
    assert restored.id == clip.id
    assert doc.clip_extent(restored.id) == (6, 11)
    assert restored.segments == ["pretend-generated"]


# ---------------------------------------------------------------------------
# SetClipFxCommand round trip (item 5, "Per-clip FX button")
# ---------------------------------------------------------------------------

def test_set_clip_fx_command_push_sets_fx_override():
    doc = Document(runs=[Run(text="hello")])
    clip = Clip()
    doc.clips.append(clip)
    stack = UndoStack(doc)

    stack.push(SetClipFxCommand(clip.id, {"reverb_enabled": True, "comp_threshold": -10}))

    assert clip.fx_override == {"reverb_enabled": True, "comp_threshold": -10}


def test_set_clip_fx_command_undo_restores_none_when_previously_unset():
    doc = Document(runs=[Run(text="hello")])
    clip = Clip()
    doc.clips.append(clip)
    stack = UndoStack(doc)
    stack.push(SetClipFxCommand(clip.id, {"reverb_enabled": True}))

    stack.undo()

    assert clip.fx_override is None


def test_set_clip_fx_command_redo_reapplies_fx_override():
    doc = Document(runs=[Run(text="hello")])
    clip = Clip()
    doc.clips.append(clip)
    stack = UndoStack(doc)
    stack.push(SetClipFxCommand(clip.id, {"reverb_enabled": True}))
    stack.undo()

    stack.redo()

    assert clip.fx_override == {"reverb_enabled": True}


def test_set_clip_fx_command_clears_a_previously_set_override():
    doc = Document(runs=[Run(text="hello")])
    clip = Clip(fx_override={"reverb_enabled": True})
    doc.clips.append(clip)
    stack = UndoStack(doc)

    stack.push(SetClipFxCommand(clip.id, None))

    assert clip.fx_override is None


def test_set_clip_fx_command_undo_restores_previous_override_after_clear():
    doc = Document(runs=[Run(text="hello")])
    clip = Clip(fx_override={"reverb_enabled": True, "comp_ratio": 4})
    doc.clips.append(clip)
    stack = UndoStack(doc)
    stack.push(SetClipFxCommand(clip.id, None))
    assert clip.fx_override is None

    stack.undo()

    assert clip.fx_override == {"reverb_enabled": True, "comp_ratio": 4}


def test_set_clip_fx_command_does_not_alias_caller_dict():
    doc = Document(runs=[Run(text="hello")])
    clip = Clip()
    doc.clips.append(clip)
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

    # 1. Programmatically replace text: insert ", there" after "hello" (position 5).
    text_cmd = TextEditCommand(5, 0, 7, new_text="hello, there world")
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
