"""Tests for kokoro_gui/daw/auto_split.py's `plan_auto_split_clips` and
kokoro_gui/daw/models.py's `Document.get_character_by_name` (item 7,
"Auto-split on generation + combined-vs-separate clip generation", of the
DAW-for-text remaining-work roadmap). Mirrors tests/daw/test_assign_character.py's
conventions - plain Python, no Qt."""
from kokoro_gui.daw.auto_split import plan_auto_split_clips
from kokoro_gui.daw.models import Character, Document
from kokoro_gui.engine.text_extraction import find_character_fx_spans

# ---------------------------------------------------------------------------
# Document.get_character_by_name
# ---------------------------------------------------------------------------


def test_get_character_by_name_exact_match():
    alice = Character.from_preset_dict("Alice", {})
    doc = Document(text="", characters=[alice])
    assert doc.get_character_by_name("Alice") is alice


def test_get_character_by_name_case_insensitive():
    alice = Character.from_preset_dict("Alice", {})
    doc = Document(text="", characters=[alice])
    assert doc.get_character_by_name("alice") is alice
    assert doc.get_character_by_name("ALICE") is alice


def test_get_character_by_name_whitespace_tolerant():
    alice = Character.from_preset_dict("Alice", {})
    doc = Document(text="", characters=[alice])
    assert doc.get_character_by_name("  Alice  ") is alice


def test_get_character_by_name_no_match_returns_none():
    alice = Character.from_preset_dict("Alice", {})
    doc = Document(text="", characters=[alice])
    assert doc.get_character_by_name("Carol") is None
    assert doc.get_character_by_name(None) is None


# ---------------------------------------------------------------------------
# plan_auto_split_clips - combined mode (split_by_paragraph=False)
# ---------------------------------------------------------------------------


def test_combined_mode_one_triple_per_tagged_span():
    alice = Character.from_preset_dict("Alice", {})
    bob = Character.from_preset_dict("Bob", {})
    text = "[Alice]: Hello there.\n\n[Bob]: Hi Alice, how are you?"
    doc = Document(text=text, characters=[alice, bob])

    triples, unmatched = plan_auto_split_clips(doc, split_by_paragraph=False)

    spans = find_character_fx_spans(text)
    assert len(spans) == 2
    assert unmatched == []
    assert triples == [
        (spans[0].start, spans[0].end, alice.id),
        (spans[1].start, spans[1].end, bob.id),
    ]


# ---------------------------------------------------------------------------
# plan_auto_split_clips - auto-split mode (split_by_paragraph=True)
# ---------------------------------------------------------------------------


def test_auto_split_mode_splits_a_paragraph_break_into_multiple_triples():
    alice = Character.from_preset_dict("Alice", {})
    bob = Character.from_preset_dict("Bob", {})
    text = (
        "[Alice]: First paragraph.\n\n"
        "Second paragraph for Alice.\n\n"
        "[Bob]: Single paragraph for Bob."
    )
    doc = Document(text=text, characters=[alice, bob])

    triples, unmatched = plan_auto_split_clips(doc, split_by_paragraph=True)

    assert unmatched == []
    alice_triples = [t for t in triples if t[2] == alice.id]
    bob_triples = [t for t in triples if t[2] == bob.id]
    assert len(alice_triples) == 2
    assert len(bob_triples) == 1

    # Every triple's offsets round-trip back to non-empty, correctly-placed text.
    for start, end, _character_id in triples:
        assert text[start:end].strip()
    assert triples == sorted(triples, key=lambda t: t[0])


def test_auto_split_mode_skips_empty_paragraphs_within_a_span():
    alice = Character.from_preset_dict("Alice", {})
    bob = Character.from_preset_dict("Bob", {})
    # Four blank-line-separated newlines between the two sentences produce an
    # empty middle "paragraph" per str.split('\n\n') - it must be skipped,
    # not emitted as a zero-width or garbage triple.
    text = "[Alice]: First paragraph.\n\n\n\nSecond paragraph."
    doc = Document(text=text, characters=[alice, bob])

    triples, unmatched = plan_auto_split_clips(doc, split_by_paragraph=True)

    assert unmatched == []
    assert len(triples) == 2
    for start, end, character_id in triples:
        assert character_id == alice.id
        assert text[start:end].strip()
        assert start < end


# ---------------------------------------------------------------------------
# unmatched tag names
# ---------------------------------------------------------------------------


def test_unmatched_tag_name_contributes_no_triples_and_is_reported():
    alice = Character.from_preset_dict("Alice", {})
    text = "[Carol]: I have no matching character."
    doc = Document(text=text, characters=[alice])

    triples, unmatched = plan_auto_split_clips(doc, split_by_paragraph=False)

    assert triples == []
    assert unmatched == ["Carol"]


def test_unmatched_span_does_not_block_matched_spans():
    alice = Character.from_preset_dict("Alice", {})
    text = "[Carol]: Unknown speaker.\n\n[Alice]: Known speaker."
    doc = Document(text=text, characters=[alice])

    triples, unmatched = plan_auto_split_clips(doc, split_by_paragraph=False)

    assert unmatched == ["Carol"]
    assert len(triples) == 1
    assert triples[0][2] == alice.id


# ---------------------------------------------------------------------------
# untagged narration (Decision 2: only default to a single character)
# ---------------------------------------------------------------------------


def test_untagged_text_assigned_to_the_sole_character():
    alice = Character.from_preset_dict("Alice", {})
    text = "Untagged narration with no tags at all."
    doc = Document(text=text, characters=[alice])

    triples, unmatched = plan_auto_split_clips(doc, split_by_paragraph=False)

    assert unmatched == []
    assert triples == [(0, len(text), alice.id)]


def test_untagged_text_produces_no_triples_with_zero_characters():
    text = "Untagged narration with no tags at all."
    doc = Document(text=text, characters=[])

    triples, unmatched = plan_auto_split_clips(doc, split_by_paragraph=False)

    assert triples == []
    assert unmatched == []


def test_untagged_text_produces_no_triples_with_two_or_more_characters():
    alice = Character.from_preset_dict("Alice", {})
    bob = Character.from_preset_dict("Bob", {})
    text = "Untagged narration with no tags at all."
    doc = Document(text=text, characters=[alice, bob])

    triples, unmatched = plan_auto_split_clips(doc, split_by_paragraph=False)

    assert triples == []
    assert unmatched == []


def test_untagged_gap_before_a_tagged_span_with_sole_character():
    alice = Character.from_preset_dict("Alice", {})
    text = "Untagged intro.\n\n[Alice]: Tagged block."
    doc = Document(text=text, characters=[alice])

    triples, unmatched = plan_auto_split_clips(doc, split_by_paragraph=False)

    assert unmatched == []
    spans = find_character_fx_spans(text)
    assert len(spans) == 1
    gap_text = text[: spans[0].start]
    assert triples == [
        (0, len(gap_text), alice.id),
        (spans[0].start, spans[0].end, alice.id),
    ]


def test_whitespace_only_gap_is_skipped():
    alice = Character.from_preset_dict("Alice", {})
    text = "   \n\n[Alice]: Tagged block."
    doc = Document(text=text, characters=[alice])

    triples, unmatched = plan_auto_split_clips(doc, split_by_paragraph=False)

    assert unmatched == []
    spans = find_character_fx_spans(text)
    assert triples == [(spans[0].start, spans[0].end, alice.id)]
