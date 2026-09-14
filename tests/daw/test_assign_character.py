"""Tests for kokoro_gui/daw/models.py's Document.assign_character_to_range
and clip_covering - the shared split-or-create primitive behind the
transcript panel's Characters menu, gutter dropdowns, and paste-splitting."""
import pytest

from kokoro_gui.daw.dirty import is_clip_dirty
from kokoro_gui.daw.models import Character, Document, Track


def _document_with_characters():
    alice = Character.from_preset_dict("Alice", {"voice": "af_bella"})
    bob = Character.from_preset_dict("Bob", {"voice": "am_michael"})
    tracks = [Track(name="Alice", character_id=alice.id), Track(name="Bob", character_id=bob.id)]
    doc = Document.from_plain_text("0123456789ABCDEFGHIJ", characters=[alice, bob], tracks=tracks)
    return doc, alice, bob


# ---------------------------------------------------------------------------
# clip_covering
# ---------------------------------------------------------------------------

def test_clip_covering_returns_none_when_no_clip_present():
    doc, _, _ = _document_with_characters()
    assert doc.clip_covering(5) is None


def test_clip_covering_inclusive_start_exclusive_end():
    doc, alice, _ = _document_with_characters()
    clip = doc.assign_character_to_range(5, 10, alice.id)
    assert doc.clip_covering(4) is not clip
    assert doc.clip_covering(5) is clip
    assert doc.clip_covering(9) is clip
    assert doc.clip_covering(10) is not clip


# ---------------------------------------------------------------------------
# assign_character_to_range
# ---------------------------------------------------------------------------

def test_raises_on_empty_or_inverted_range():
    doc, alice, _ = _document_with_characters()
    with pytest.raises(ValueError):
        doc.assign_character_to_range(5, 5, alice.id)
    with pytest.raises(ValueError):
        doc.assign_character_to_range(5, 3, alice.id)


def test_creates_clip_when_none_exists():
    doc, alice, _ = _document_with_characters()
    clip = doc.assign_character_to_range(2, 6, alice.id)
    assert clip in doc.clips
    assert doc.clip_extent(clip.id) == (2, 6)
    assert clip.character_id == alice.id
    assert clip.track_id == next(t.id for t in doc.tracks if t.character_id == alice.id)
    assert clip.segments == []


def test_track_id_falls_back_to_none_without_a_matching_track():
    doc = Document.from_plain_text("hello")
    clip = doc.assign_character_to_range(0, 5, "nonexistent-character")
    assert clip.track_id is None


def test_exact_match_reassignment_creates_new_id():
    doc, alice, bob = _document_with_characters()
    old_clip = doc.assign_character_to_range(0, 5, alice.id)
    old_id = old_clip.id

    new_clip = doc.assign_character_to_range(0, 5, bob.id)

    assert new_clip.id != old_id
    assert doc.get_clip(old_id) is None
    assert new_clip.character_id == bob.id
    assert len(doc.clips) == 1


def test_left_only_split():
    doc, alice, bob = _document_with_characters()
    existing = doc.assign_character_to_range(0, 10, alice.id)

    doc.assign_character_to_range(5, 10, bob.id)

    assert existing.id not in {c.id for c in doc.clips}
    leftover = next(c for c in doc.clips if c.character_id == alice.id)
    assert doc.clip_extent(leftover.id) == (0, 5)
    assigned = next(c for c in doc.clips if c.character_id == bob.id)
    assert doc.clip_extent(assigned.id) == (5, 10)
    assert len(doc.clips) == 2


def test_right_only_split():
    doc, alice, bob = _document_with_characters()
    doc.assign_character_to_range(0, 10, alice.id)

    doc.assign_character_to_range(0, 5, bob.id)

    leftover = next(c for c in doc.clips if c.character_id == alice.id)
    assert doc.clip_extent(leftover.id) == (5, 10)
    assigned = next(c for c in doc.clips if c.character_id == bob.id)
    assert doc.clip_extent(assigned.id) == (0, 5)
    assert len(doc.clips) == 2


def test_both_sided_split():
    doc, alice, bob = _document_with_characters()
    doc.assign_character_to_range(0, 20, alice.id)

    doc.assign_character_to_range(5, 10, bob.id)

    alice_leftovers = sorted(
        (c for c in doc.clips if c.character_id == alice.id),
        key=lambda c: doc.clip_extent(c.id),
    )
    assert len(alice_leftovers) == 2
    assert doc.clip_extent(alice_leftovers[0].id) == (0, 5)
    assert doc.clip_extent(alice_leftovers[1].id) == (10, 20)
    assigned = next(c for c in doc.clips if c.character_id == bob.id)
    assert doc.clip_extent(assigned.id) == (5, 10)
    assert len(doc.clips) == 3


def test_selection_spanning_three_clips_worked_example():
    # Clip A: [0, 10) Alice, Clip B: [10, 15) Bob, Clip C: [15, 20) Alice.
    doc, alice, bob = _document_with_characters()
    clip_a = doc.assign_character_to_range(0, 10, alice.id)
    clip_b = doc.assign_character_to_range(10, 15, bob.id)
    clip_c = doc.assign_character_to_range(15, 20, alice.id)

    # Assign Bob to [5, 18) - overlaps all three.
    new_clip = doc.assign_character_to_range(5, 18, bob.id)

    assert len(doc.clips) == 3  # leftover of A, leftover of C, and the new clip - B fully consumed
    a_leftover = next(c for c in doc.clips if doc.clip_extent(c.id) is not None and doc.clip_extent(c.id)[0] == 0)
    assert doc.clip_extent(a_leftover.id) == (0, 5)
    assert a_leftover.character_id == alice.id
    c_leftover = next(c for c in doc.clips if doc.clip_extent(c.id) is not None and doc.clip_extent(c.id)[1] == 20)
    assert doc.clip_extent(c_leftover.id) == (18, 20)
    assert c_leftover.character_id == alice.id
    assert doc.clip_extent(new_clip.id) == (5, 18)
    assert new_clip.character_id == bob.id
    assert clip_a.id not in {c.id for c in doc.clips}
    assert clip_b.id not in {c.id for c in doc.clips}
    assert clip_c.id not in {c.id for c in doc.clips}


def test_leftover_clips_are_dirty_and_have_fresh_ids():
    doc, alice, bob = _document_with_characters()
    original = doc.assign_character_to_range(0, 10, alice.id)
    original.segments = ["pretend-generated"]  # simulate a previously-generated clip

    doc.assign_character_to_range(5, 10, bob.id)

    leftover = next(c for c in doc.clips if c.character_id == alice.id)
    assert leftover.id != original.id
    assert leftover.segments == []


def test_new_and_split_clips_are_reported_dirty():
    doc, alice, _ = _document_with_characters()
    clip = doc.assign_character_to_range(0, 5, alice.id)
    assert is_clip_dirty(clip, doc.clip_text(clip), doc.effective_config_for_clip(clip)) is True
