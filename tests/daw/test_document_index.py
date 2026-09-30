"""`DocumentIndex` against the run walks it replaced
(kokoro_gui/daw/derived.py, Claude/PLAN_performance.md).

A random sequence of edits, character assignments, undo commands, undo and
redo, with every lookup compared to a linear walk over the runs after each
step. Verify mode (on for the whole suite) also checks the cached index
against a fresh one on every read.
"""
import random

import pytest

from kokoro_gui.daw.models import Character, Document
from kokoro_gui.daw.undo import AssignCharacterCommand, TextEditCommand


def _walk_covering(doc, position):
    pos = 0
    for run in doc.runs:
        end = pos + len(run.text)
        if pos <= position < end:
            return run
        pos = end
    return None


def _walk_extent(doc, clip_id):
    start = end = None
    pos = 0
    for run in doc.runs:
        r_end = pos + len(run.text)
        if run.clip_id == clip_id:
            if start is None:
                start = pos
            end = r_end
        pos = r_end
    return None if start is None else (start, end)


def _assert_index_matches_walk(doc):
    text = "".join(run.text for run in doc.runs)
    assert doc.index().text == text
    for position in range(-1, len(text) + 2):
        walked = _walk_covering(doc, position)
        assert doc._run_covering(position) is walked, position
        expected_clip = next((c for c in doc.clips if walked is not None and c.id == walked.clip_id), None)
        assert doc.clip_covering(position) is expected_clip, position
    for clip in doc.clips:
        assert doc.clip_extent(clip.id) == _walk_extent(doc, clip.id)
        assert doc.clip_text(clip) == doc._clip_text_walk(clip)
        assert doc.get_clip(clip.id) is next(c for c in doc.clips if c.id == clip.id)
    for character in doc.characters:
        assert doc.get_character(character.id) is next(c for c in doc.characters if c.id == character.id)


@pytest.mark.parametrize("seed", range(12))
def test_index_matches_the_run_walk_through_random_edits(seed):
    rng = random.Random(seed)
    alice = Character.from_preset_dict("Alice", {})
    bob = Character.from_preset_dict("Bob", {})
    doc = Document.from_plain_text("The quick brown fox.\nJumps over\n\nthe lazy dog.", characters=[alice, bob])
    _assert_index_matches_walk(doc)
    for _ in range(40):
        text = doc.text
        op = rng.choice(["assign", "unassign", "type", "delete", "command_edit", "undo", "redo"])
        if op in ("assign", "unassign") and text:
            start = rng.randrange(len(text))
            end = rng.randrange(start + 1, len(text) + 1)
            character = rng.choice([alice, bob]).id if op == "assign" else None
            doc.undo_stack.push(AssignCharacterCommand(start, end, character))
        elif op == "type":
            position = rng.randrange(len(text) + 1)
            insert = rng.choice(["x", " yz", "\n", "word "])
            new_text = text[:position] + insert + text[position:]
            doc.replace_text(position, 0, len(insert), new_text)
        elif op == "delete" and text:
            position = rng.randrange(len(text))
            removed = rng.randrange(1, min(6, len(text) - position) + 1)
            doc.replace_text(position, removed, 0, text[:position] + text[position + removed:])
        elif op == "command_edit" and text:
            position = rng.randrange(len(text))
            removed = rng.randrange(0, min(4, len(text) - position) + 1)
            new_text = text[:position] + "ABC" + text[position + removed:]
            doc.undo_stack.push(TextEditCommand(position, removed, 3, new_text))
        elif op == "undo":
            doc.undo_stack.undo()
        elif op == "redo":
            doc.undo_stack.redo()
        _assert_index_matches_walk(doc)


def test_an_append_to_the_clip_list_without_an_attribute_set_still_reaches_the_index():
    """The index keys on the lists' identity and length too, so an
    in-place append nobody reported is still seen."""
    from kokoro_gui.daw.models import Clip

    doc = Document.from_plain_text("hello")
    doc.index()
    clip = Clip()
    doc.clips.append(clip)
    assert doc.get_clip(clip.id) is clip
