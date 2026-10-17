"""Tests for kokoro_gui/daw/textsearch.py: the literal search, the hits it
drops, the plan a replacement states beforehand, and the command that applies
it (plan 34). No Qt."""
import copy

import pytest

from kokoro_gui.daw import imported, textsearch
from kokoro_gui.daw.imported import run_from_asr_words
from kokoro_gui.daw.models import Character, Clip, Document, Run

A = "aaaaaaaaaaaaaaaa"


# -- find_all ---------------------------------------------------------------------------

def test_finds_every_hit_left_to_right():
    assert textsearch.find_all("one cat, two cats", "cat") == [(4, 7), (13, 16)]


def test_an_empty_needle_or_text_has_no_hits():
    assert textsearch.find_all("abc", "") == []
    assert textsearch.find_all("", "a") == []


def test_case_is_ignored_unless_match_case_is_on():
    text = "Cat cat CAT"
    assert textsearch.find_all(text, "cat") == [(0, 3), (4, 7), (8, 11)]
    assert textsearch.find_all(text, "cat", match_case=True) == [(4, 7)]


def test_whole_word_skips_hits_inside_a_longer_word():
    text = "cat concatenate cats cat."
    assert textsearch.find_all(text, "cat") == [(0, 3), (7, 10), (16, 19), (21, 24)]
    assert textsearch.find_all(text, "cat", whole_word=True) == [(0, 3), (21, 24)]


def test_whole_word_counts_an_apostrophe_and_a_hyphen_as_a_boundary():
    assert textsearch.find_all("the dog's bone, a dog-eared page", "dog", whole_word=True) == [(4, 7), (18, 21)]


def test_whole_word_checks_only_the_edges_that_are_word_characters():
    # "-x-" starts and ends on a symbol, so only the text around it is free to be anything.
    assert textsearch.find_all("a -x- b", "-x-", whole_word=True) == [(2, 5)]
    assert textsearch.find_all("a-x-b", "-x-", whole_word=True) == [(1, 4)]
    # A needle that ends on a word character still needs a boundary after it.
    assert textsearch.find_all("a.b a.bc", "a.b", whole_word=True) == [(0, 3)]


def test_hits_do_not_overlap():
    assert textsearch.find_all("aaaa", "aa") == [(0, 2), (2, 4)]
    assert textsearch.find_all("aaa", "aa") == [(0, 2)]


def test_regex_characters_in_the_needle_are_literal():
    text = "cost $5.00 (approx) [a+b] a|b"
    assert textsearch.find_all(text, "$5.00") == [(5, 10)]
    assert textsearch.find_all(text, "(approx)") == [(11, 19)]
    assert textsearch.find_all(text, "[a+b]") == [(20, 25)]
    assert textsearch.find_all(text, "a|b") == [(26, 29)]
    assert textsearch.find_all("a.b axb", ".") == [(1, 2)]


def test_case_folding_keeps_the_offsets_of_the_original_text():
    # "ß".lower() is one character and "İ".lower() is two: offsets must still be the text's own.
    text = "Straße İstanbul STRASSE"
    assert textsearch.find_all(text, "stan") == [(8, 12)]
    assert textsearch.find_all(text, "straße") == [(0, 6)]


# -- find_in_document -------------------------------------------------------------------

def _doc_with_placeholder():
    nested = Clip(source="nested", child={"id": "c1", "kind": "embedded"})
    runs = [Run(text="the cat sat\n"), Run(text="cat tales", clip_id=nested.id, kind="placeholder"),
            Run(text="\nanother cat")]
    return Document(runs=runs, clips=[nested])


def test_a_placeholder_line_is_never_a_hit():
    document = _doc_with_placeholder()
    hits = textsearch.find_in_document(document, "cat")
    assert hits == [(4, 7), (document.text.rindex("cat"), document.text.rindex("cat") + 3)]
    assert textsearch.find_all(document.text, "cat")[1] == (12, 15)  # the placeholder's own "cat"


def test_a_hit_that_reaches_into_a_placeholder_is_dropped():
    document = _doc_with_placeholder()
    # "sat\ncat" runs from the first line into the placeholder.
    assert textsearch.find_in_document(document, "sat\ncat") == []
    assert textsearch.find_in_document(document, "tales\nanother") == []


# -- plan_replace -----------------------------------------------------------------------

def _generated_document():
    """Three one-line generated clips and a bit of untagged text."""
    character = Character(name="Ann")
    document = Document.from_plain_text("The cat sat.\nA cat and a cat.\nNo felines here.\nUntagged cat.")
    document.characters.append(character)
    text = document.text
    for line in ("The cat sat.", "A cat and a cat.", "No felines here."):
        start = text.index(line)
        document.assign_character_to_range(start, start + len(line), character.id)
    return document


def test_the_plan_counts_matches_clips_and_the_clips_that_turn_stale(monkeypatch):
    document = _generated_document()
    hits = textsearch.find_in_document(document, "cat")
    assert len(hits) == 4
    first = document.clips[0].id
    # The first clip is stale already; the second is up to date.
    monkeypatch.setattr(document, "dirty_ids", lambda: frozenset({first, document.clips[2].id}))

    plan = textsearch.plan_replace(document, hits, "dog")

    assert plan.matches == 4
    assert plan.clips == 2  # the untagged line belongs to no clip
    assert plan.stale == 1
    assert plan.recordings == 0


def test_a_hit_that_already_reads_as_the_replacement_changes_nothing():
    document = _generated_document()
    plan = textsearch.plan_replace(document, textsearch.find_in_document(document, "cat", match_case=True), "cat")
    assert plan.matches == 0 and plan.clips == 0 and plan.stale == 0
    # Only the capital is different.
    plan = textsearch.plan_replace(document, textsearch.find_in_document(document, "the"), "The")
    assert plan.matches == 0


def _recording_document():
    text, words = run_from_asr_words([("So", 0.0, 0.4), ("the", 0.5, 0.9), ("cat", 1.0, 1.4), ("sat", 1.5, 1.9)], A)
    clip = Clip(source="imported")
    run = Run(text=text, clip_id=clip.id, kind="imported", words=words)
    return Document(runs=[run, Run(text="\ncat")], clips=[clip], settings={"sources": {A: {
        "path": "/p/audio/imported/a.wav", "sample_rate": 24000, "duration_s": 5.0}}})


def test_a_recording_is_counted_apart_from_the_generated_clips():
    document = _recording_document()
    plan = textsearch.plan_replace(document, textsearch.find_in_document(document, "cat"), "dog")
    assert plan.matches == 2
    assert plan.recordings == 1
    assert plan.clips == 1
    assert plan.stale == 0


# -- replace_command --------------------------------------------------------------------

def test_the_command_replaces_every_edit_and_one_undo_takes_them_all_back():
    document = _generated_document()
    before_text, before_runs = document.text, copy.deepcopy(document.runs)
    edits = textsearch.edits_for(document, textsearch.find_in_document(document, "cat"), "wolf")

    document.undo_stack.push(textsearch.replace_command(document, edits))

    assert document.text == before_text.replace("cat", "wolf")
    document.undo_stack.undo()
    assert document.text == before_text
    assert document.runs == before_runs
    document.undo_stack.redo()
    assert document.text == before_text.replace("cat", "wolf")


def test_a_clip_that_is_only_the_match_keeps_its_character():
    character = Character(name="Ann")
    document = Document.from_plain_text("Hello\nWorld")
    document.characters.append(character)
    document.assign_character_to_range(0, 5, character.id)
    clip_id = document.clips[0].id
    edits = textsearch.edits_for(document, textsearch.find_in_document(document, "Hello"), "Hi")

    document.undo_stack.push(textsearch.replace_command(document, edits))

    assert document.text == "Hi\nWorld"
    assert [c.id for c in document.clips] == [clip_id]
    assert document.clip_text(document.clips[0]) == "Hi"
    document.undo_stack.undo()
    assert document.clip_text(document.get_clip(clip_id)) == "Hello"


def test_deleting_a_match_takes_it_out_of_its_clip():
    document = _generated_document()
    edits = textsearch.edits_for(document, textsearch.find_in_document(document, " cat", match_case=True), "")
    document.undo_stack.push(textsearch.replace_command(document, edits))
    assert document.text == "The sat.\nA and a.\nNo felines here.\nUntagged."


def test_a_replacement_inside_a_recording_splits_it_the_way_typing_does():
    typed = _recording_document()
    start = typed.text.index("cat")
    typed.replace_text(start, 3, 3, typed.text[:start] + "dog" + typed.text[start + 3:])

    replaced = _recording_document()
    edits = textsearch.edits_for(replaced, [(start, start + 3)], "dog")
    replaced.undo_stack.push(textsearch.replace_command(replaced, edits))

    assert replaced.text == typed.text == "So the dog sat\ncat"
    assert len(replaced.clips) == len(typed.clips) == 2
    assert [(r.text, r.kind, r.words) for r in replaced.runs] == [(r.text, r.kind, r.words) for r in typed.runs]
    # The replaced word has no audio of its own: no timed word spells it.
    assert not any(replaced.text[a:b] == "dog" for clip in replaced.clips for a, b, *_ in imported.clip_words(replaced, clip))


@pytest.mark.parametrize("text, needle, expected", [
    ("a b a", "a", 2),
    ("ab ba", "ab", 1),
])
def test_edits_for_skips_nothing_when_text_differs(text, needle, expected):
    document = Document.from_plain_text(text)
    assert len(textsearch.edits_for(document, textsearch.find_all(text, needle), "z")) == expected
