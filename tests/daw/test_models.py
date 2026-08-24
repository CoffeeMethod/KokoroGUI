"""Plain-Python tests for kokoro_gui/daw/models.py - no Qt, no QT_QPA_PLATFORM
needed, mirroring how tests/test_caching.py tests kokoro_gui/engine/caching.py
with zero GUI dependency."""
import pytest

from kokoro_gui.daw.models import Character, Clip, Document, Segment, Track


# ---------------------------------------------------------------------------
# Character
# ---------------------------------------------------------------------------

def test_character_from_preset_dict_filters_disallowed_keys():
    # out_dir/filename are never allowed into a preset-derived trust
    # boundary (see kokoro_gui/engine/presets.py's ALLOWED_PRESET_KEYS docs).
    character = Character.from_preset_dict(
        "Alice", {"voice": "af_bella", "speed": 1.2, "out_dir": "/etc", "filename": "pwn"}
    )
    assert character.preset_data == {"voice": "af_bella", "speed": 1.2}
    assert character.name == "Alice"
    assert character.id  # auto-assigned


def test_character_to_preset_dict_round_trips():
    original = {"voice": "af_bella", "speed": 1.0}
    character = Character.from_preset_dict("Alice", original)
    assert character.to_preset_dict() == original


def test_character_ids_are_unique():
    a = Character.from_preset_dict("A", {})
    b = Character.from_preset_dict("B", {})
    assert a.id != b.id


# ---------------------------------------------------------------------------
# Clip
# ---------------------------------------------------------------------------

def test_clip_rejects_invalid_source():
    with pytest.raises(ValueError):
        Clip(source="recorded")


def test_clip_default_source_is_generated():
    assert Clip().source == "generated"


# ---------------------------------------------------------------------------
# Document lookups
# ---------------------------------------------------------------------------

def _make_document_with_one_character():
    character = Character.from_preset_dict("Alice", {"voice": "af_bella", "speed": 1.0})
    track = Track(name="Alice", character_id=character.id)
    clip = Clip(start_offset=0, end_offset=5, character_id=character.id, track_id=track.id)
    doc = Document(text="hello world", clips=[clip], tracks=[track], characters=[character])
    return doc, character, track, clip


def test_get_character_get_track_get_clip():
    doc, character, track, clip = _make_document_with_one_character()
    assert doc.get_character(character.id) is character
    assert doc.get_track(track.id) is track
    assert doc.get_clip(clip.id) is clip
    assert doc.get_character(None) is None
    assert doc.get_character("nonexistent") is None


def test_clip_text_slices_document_text():
    doc, _, _, clip = _make_document_with_one_character()
    assert doc.clip_text(clip) == "hello"


def test_effective_config_merges_character_preset_and_overrides():
    doc, character, _, clip = _make_document_with_one_character()
    clip.overrides = {"speed": 1.5}  # override wins over the character's speed=1.0
    config = doc.effective_config_for_clip(clip)
    assert config == {"voice": "af_bella", "speed": 1.5}


def test_effective_config_filters_disallowed_override_keys():
    doc, _, _, clip = _make_document_with_one_character()
    clip.overrides = {"out_dir": "/etc"}
    config = doc.effective_config_for_clip(clip)
    assert "out_dir" not in config


def test_effective_config_with_no_character_uses_only_overrides():
    clip = Clip(start_offset=0, end_offset=5, character_id=None, overrides={"voice": "af_bella"})
    doc = Document(text="hello", clips=[clip])
    assert doc.effective_config_for_clip(clip) == {"voice": "af_bella"}


# ---------------------------------------------------------------------------
# Document.apply_text_change
# ---------------------------------------------------------------------------

def test_apply_text_change_shifts_clip_entirely_after_insertion():
    clip = Clip(start_offset=10, end_offset=15)
    doc = Document(text="0123456789ABCDE", clips=[clip])
    # Insert 3 chars at position 2.
    new_text = "01" + "XYZ" + "23456789ABCDE"
    removed = doc.apply_text_change(position=2, chars_removed=0, chars_added=3, new_text=new_text)
    assert removed == []
    assert clip.start_offset == 13
    assert clip.end_offset == 18
    assert doc.text == new_text


def test_apply_text_change_leaves_clip_entirely_before_edit_untouched():
    clip = Clip(start_offset=0, end_offset=3)
    doc = Document(text="ABCDEFGHIJ", clips=[clip])
    new_text = "ABCXXFGHIJ"  # edit happens at position 3..5
    doc.apply_text_change(position=3, chars_removed=2, chars_added=2, new_text=new_text)
    assert clip.start_offset == 0
    assert clip.end_offset == 3


def test_apply_text_change_extends_clip_edited_in_place():
    # Clip spans "world" in "hello world" (offsets 6..11).
    clip = Clip(start_offset=6, end_offset=11)
    doc = Document(text="hello world", clips=[clip])
    new_text = "hello wonderful world"  # inserted "onderful " at position 8
    doc.apply_text_change(position=8, chars_removed=0, chars_added=11, new_text=new_text)
    assert clip.start_offset == 6
    # end_offset should have grown to cover the insertion.
    assert clip.end_offset == 11 + 11


def test_apply_text_change_removes_clip_fully_consumed_by_deletion():
    clip = Clip(start_offset=6, end_offset=11)
    doc = Document(text="hello world", clips=[clip])
    new_text = "hello "
    removed = doc.apply_text_change(position=6, chars_removed=5, chars_added=0, new_text=new_text)
    assert removed == [clip]
    assert clip not in doc.clips
    assert doc.text == new_text


def test_apply_text_change_removes_clip_when_replacement_spans_it():
    clip = Clip(start_offset=6, end_offset=11)
    doc = Document(text="hello world", clips=[clip])
    # Replace a range that starts before and ends after the clip.
    new_text = "hello EVERYONE!"
    removed = doc.apply_text_change(position=3, chars_removed=8, chars_added=10, new_text=new_text)
    assert removed == [clip]


def test_dirty_clips_delegates_to_dirty_module(monkeypatch):
    doc, _, _, clip = _make_document_with_one_character()
    other = Clip(start_offset=0, end_offset=0)
    doc.clips.append(other)

    calls = []

    def fake_is_dirty(c, text, config):
        calls.append(c)
        return c is clip

    monkeypatch.setattr("kokoro_gui.daw.dirty.is_clip_dirty", fake_is_dirty)
    assert doc.dirty_clips() == [clip]
    assert calls == [clip, other]
