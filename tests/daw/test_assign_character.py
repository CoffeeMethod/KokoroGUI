"""Tests for kokoro_gui/daw/models.py's Document.assign_character_to_range
and clip_covering - the shared split-or-create primitive behind the
transcript panel's Characters menu and paste-splitting."""
import pytest

from kokoro_gui.daw.dirty import is_clip_dirty
from kokoro_gui.daw.models import Character, Clip, Document, Track


def _document_with_characters():
    alice = Character.from_preset_dict("Alice", {"voice": "af_bella"})
    bob = Character.from_preset_dict("Bob", {"voice": "am_michael"})
    tracks = [Track(name="Alice", character_id=alice.id), Track(name="Bob", character_id=bob.id)]
    doc = Document(text="0123456789ABCDEFGHIJ", characters=[alice, bob], tracks=tracks)
    return doc, alice, bob


# ---------------------------------------------------------------------------
# clip_covering
# ---------------------------------------------------------------------------

def test_clip_covering_returns_none_when_no_clip_present():
    doc, _, _ = _document_with_characters()
    assert doc.clip_covering(5) is None


def test_clip_covering_inclusive_start_exclusive_end():
    clip = Clip(start_offset=5, end_offset=10)
    doc, _, _ = _document_with_characters()
    doc.clips.append(clip)
    assert doc.clip_covering(4) is None
    assert doc.clip_covering(5) is clip
    assert doc.clip_covering(9) is clip
    assert doc.clip_covering(10) is None


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
    assert clip.start_offset == 2
    assert clip.end_offset == 6
    assert clip.character_id == alice.id
    assert clip.track_id == next(t.id for t in doc.tracks if t.character_id == alice.id)
    assert clip.segments == []


def test_track_id_falls_back_to_none_without_a_matching_track():
    doc = Document(text="hello")
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

    remaining = [c for c in doc.clips if c.id != existing.id]
    leftover = next(c for c in doc.clips if c.character_id == alice.id)
    assert leftover.start_offset == 0
    assert leftover.end_offset == 5
    assigned = next(c for c in doc.clips if c.character_id == bob.id)
    assert assigned.start_offset == 5
    assert assigned.end_offset == 10
    assert len(doc.clips) == 2


def test_right_only_split():
    doc, alice, bob = _document_with_characters()
    doc.assign_character_to_range(0, 10, alice.id)

    doc.assign_character_to_range(0, 5, bob.id)

    leftover = next(c for c in doc.clips if c.character_id == alice.id)
    assert leftover.start_offset == 5
    assert leftover.end_offset == 10
    assigned = next(c for c in doc.clips if c.character_id == bob.id)
    assert assigned.start_offset == 0
    assert assigned.end_offset == 5
    assert len(doc.clips) == 2


def test_both_sided_split():
    doc, alice, bob = _document_with_characters()
    doc.assign_character_to_range(0, 20, alice.id)

    doc.assign_character_to_range(5, 10, bob.id)

    alice_leftovers = sorted(
        (c for c in doc.clips if c.character_id == alice.id),
        key=lambda c: c.start_offset,
    )
    assert len(alice_leftovers) == 2
    assert (alice_leftovers[0].start_offset, alice_leftovers[0].end_offset) == (0, 5)
    assert (alice_leftovers[1].start_offset, alice_leftovers[1].end_offset) == (10, 20)
    assigned = next(c for c in doc.clips if c.character_id == bob.id)
    assert (assigned.start_offset, assigned.end_offset) == (5, 10)
    assert len(doc.clips) == 3


def test_selection_spanning_three_clips_worked_example():
    # Clip A: [0, 10) Alice, Clip B: [10, 15) Bob, Clip C: [15, 20) Alice.
    doc, alice, bob = _document_with_characters()
    clip_a = Clip(start_offset=0, end_offset=10, character_id=alice.id)
    clip_b = Clip(start_offset=10, end_offset=15, character_id=bob.id)
    clip_c = Clip(start_offset=15, end_offset=20, character_id=alice.id)
    doc.clips.extend([clip_a, clip_b, clip_c])

    # Assign Bob to [5, 18) - overlaps all three.
    new_clip = doc.assign_character_to_range(5, 18, bob.id)

    assert len(doc.clips) == 3  # leftover of A, leftover of C, and the new clip - B fully consumed
    a_leftover = next(c for c in doc.clips if c.start_offset == 0)
    assert a_leftover.end_offset == 5
    assert a_leftover.character_id == alice.id
    c_leftover = next(c for c in doc.clips if c.end_offset == 20)
    assert c_leftover.start_offset == 18
    assert c_leftover.character_id == alice.id
    assert new_clip.start_offset == 5
    assert new_clip.end_offset == 18
    assert new_clip.character_id == bob.id
    assert clip_b not in doc.clips


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
