"""`AssignCharacterCommand` and `TextEditCommand` snapshot the runs around
their range and the clips those runs name, not the whole document
(kokoro_gui/daw/undo.py `_EditSnapshot`, Claude/old/PLAN_performance.md). A
random sequence of edits on documents of several sizes, each undone and
compared with a deep copy taken before it."""
import copy
import dataclasses
import random

import pytest

from kokoro_gui.daw import undo as undo_module
from kokoro_gui.daw.models import IMPORTED, Character, Clip, Document, Run, Segment
from kokoro_gui.daw.undo import AssignCharacterCommand, TextEditCommand

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]


def _state(doc):
    return ([(r.text, r.clip_id, r.kind, r.words) for r in doc.runs],
            [dataclasses.asdict(c) for c in doc.clips],
            [dataclasses.asdict(t) for t in doc.tracks])


def _document(rng, lines: int):
    alice = Character.from_preset_dict("Alice", {})
    bob = Character.from_preset_dict("Bob", {})
    text = "\n".join(" ".join(rng.choice(WORDS) for _ in range(rng.randint(2, 6))) for _ in range(lines))
    doc = Document.from_plain_text(text, characters=[alice, bob])
    pos = 0
    for line in text.split("\n"):
        if rng.random() < 0.7 and line:
            clip = doc.assign_character_to_range(pos, pos + len(line), rng.choice([alice, bob]).id)
            if rng.random() < 0.5:
                clip.segments = [Segment(order_index=0, text=line, cache_key="k" + clip.id,
                                         audio_path=f"/tmp/{clip.id}.wav", duration=1.0)]
                clip.overrides["pitch"] = rng.random()
        pos += len(line) + 1
    return doc, [alice, bob]


def _random_command(rng, doc, characters, wide: bool):
    text = doc.text
    if rng.random() < 0.5 and text:
        start = rng.randrange(len(text))
        span = rng.randint(1, len(text) - start) if wide else rng.randint(1, min(40, len(text) - start))
        character = rng.choice([*characters, None])
        return AssignCharacterCommand(start, start + span, character.id if character else None)
    position = rng.randrange(len(text) + 1)
    removed = rng.randint(0, min(12 if not wide else len(text), len(text) - position))
    inserted = rng.choice(["", "x", " yz ", "\n", "new words\nhere"])
    new_text = text[:position] + inserted + text[position + removed:]
    return TextEditCommand(position, removed, len(inserted), new_text)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("lines,wide", [(6, True), (80, False), (80, True)])
def test_undo_restores_the_document_a_command_started_from(seed, lines, wide):
    rng = random.Random(seed * 1000 + lines)
    doc, characters = _document(rng, lines)
    history = []
    for _ in range(12):
        before = _state(doc)
        command = _random_command(rng, doc, characters, wide)
        try:
            doc.undo_stack.push(command)
        except ValueError:
            continue  # a range that can't be retagged (a placeholder); nothing was recorded
        assert command._snapshot.fallback is None  # the edit stayed inside what was snapshotted
        history.append(before)
        after = _state(doc)
        doc.undo_stack.undo()
        assert _state(doc) == before
        doc.undo_stack.redo()
        assert _state(doc)[0] and doc.text == "".join(r[0] for r in after[0])
    for before in reversed(history):
        doc.undo_stack.undo()
        assert _state(doc) == before


def test_a_one_clip_edit_in_a_long_document_copies_a_handful_of_objects(monkeypatch):
    rng = random.Random(3)
    doc, characters = _document(rng, 300)
    target = doc.clips[150]
    start, end = doc.clip_extent(target.id)
    copied = []
    real = copy.deepcopy
    monkeypatch.setattr(undo_module.copy, "deepcopy",
                        lambda obj, *a: copied.append(len(obj) if isinstance(obj, list) else 1) or real(obj, *a))

    command = AssignCharacterCommand(start + 1, end - 1, characters[0].id)
    doc.undo_stack.push(command)
    doc.undo_stack.undo()

    assert 0 < sum(copied) < 30  # not 300 clips and 600 runs, twice


def test_an_edit_over_many_clips_snapshots_the_whole_lists(monkeypatch):
    rng = random.Random(5)
    doc, characters = _document(rng, 200)
    before = _state(doc)
    command = AssignCharacterCommand(0, len(doc.text), characters[0].id)

    doc.undo_stack.push(command)

    assert command._snapshot.whole is not None
    doc.undo_stack.undo()
    assert _state(doc) == before


def test_a_document_with_imported_clips_snapshots_the_whole_lists():
    alice = Character.from_preset_dict("Alice", {})
    recording = Clip(source=IMPORTED)
    doc = Document(runs=[Run("one two", recording.id, IMPORTED, words=[[0, 3, "rec", 0.0, 0.5], [4, 7, "rec", 1.0, 1.5]]),
                         Run("\n\nplain text")],
                   clips=[recording], characters=[alice])
    before = _state(doc)
    command = AssignCharacterCommand(9, 19, alice.id)

    doc.undo_stack.push(command)

    assert command._snapshot.whole is not None
    doc.undo_stack.undo()
    assert _state(doc) == before


def test_an_edit_that_strays_outside_the_slice_still_undoes_to_the_objects_it_started_with(monkeypatch):
    """`seal` notices a run changed outside the slice and falls back to the
    run and clip objects the document held."""
    rng = random.Random(7)
    doc, characters = _document(rng, 30)
    runs_before = list(doc.runs)
    clips_before = list(doc.clips)
    command = TextEditCommand(0, 0, 1, "x" + doc.text)
    real = doc.replace_text

    def stray(*args):
        removed = real(*args)
        doc.runs = [*doc.runs[:-1], Run(text=doc.runs[-1].text, clip_id=doc.runs[-1].clip_id, kind=doc.runs[-1].kind)]
        return removed

    monkeypatch.setattr(doc, "replace_text", stray)
    doc.undo_stack.push(command)
    assert command._snapshot.slice is None and command._snapshot.fallback is not None

    doc.undo_stack.undo()
    assert doc.runs == runs_before and all(a is b for a, b in zip(doc.runs, runs_before))
    assert doc.clips == clips_before
