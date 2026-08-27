"""Plain-Python tests for kokoro_gui/daw/models.py - no Qt, no QT_QPA_PLATFORM
needed, mirroring how tests/test_caching.py tests kokoro_gui/engine/caching.py
with zero GUI dependency."""
import pytest

from kokoro_gui.daw.models import Character, Clip, Document, Run, Segment, Track


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


def test_clip_has_no_offset_fields():
    # The whole point of the run-based rework: a Clip's extent lives in
    # Document.runs, not on the Clip object itself.
    clip = Clip()
    assert not hasattr(clip, "start_offset")
    assert not hasattr(clip, "end_offset")


# ---------------------------------------------------------------------------
# Document construction / text (derived property)
# ---------------------------------------------------------------------------

def test_from_plain_text_creates_one_untagged_run():
    doc = Document.from_plain_text("hello world")
    assert doc.text == "hello world"
    assert len(doc.runs) == 1
    assert doc.runs[0].clip_id is None


def test_from_plain_text_empty_string_creates_no_runs():
    doc = Document.from_plain_text("")
    assert doc.text == ""
    assert doc.runs == []


def test_text_is_the_join_of_every_run():
    doc = Document(runs=[Run(text="hello "), Run(text="world", clip_id="c1")])
    assert doc.text == "hello world"


def test_set_plain_text_discards_existing_tags():
    doc = Document(runs=[Run(text="hello", clip_id="c1")])
    doc.set_plain_text("goodbye")
    assert doc.text == "goodbye"
    assert doc.runs == [Run(text="goodbye")]


# ---------------------------------------------------------------------------
# Document lookups
# ---------------------------------------------------------------------------

def _make_document_with_one_character():
    character = Character.from_preset_dict("Alice", {"voice": "af_bella", "speed": 1.0})
    track = Track(name="Alice", character_id=character.id)
    clip = Clip(character_id=character.id, track_id=track.id)
    doc = Document(
        runs=[Run(text="hello", clip_id=clip.id, kind=clip.source), Run(text=" world")],
        clips=[clip], tracks=[track], characters=[character],
    )
    return doc, character, track, clip


def test_get_character_get_track_get_clip():
    doc, character, track, clip = _make_document_with_one_character()
    assert doc.get_character(character.id) is character
    assert doc.get_track(track.id) is track
    assert doc.get_clip(clip.id) is clip
    assert doc.get_character(None) is None
    assert doc.get_character("nonexistent") is None


def test_clip_covering_inclusive_start_exclusive_end():
    doc, _, _, clip = _make_document_with_one_character()
    assert doc.clip_covering(0) is clip
    assert doc.clip_covering(4) is clip
    assert doc.clip_covering(5) is None  # " world" is untagged
    assert doc.clip_covering(100) is None


def test_clip_extent_walks_runs():
    doc, _, _, clip = _make_document_with_one_character()
    assert doc.clip_extent(clip.id) == (0, 5)
    assert doc.clip_extent("nonexistent") is None


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
    clip = Clip(character_id=None, overrides={"voice": "af_bella"})
    doc = Document(runs=[Run(text="hello", clip_id=clip.id)], clips=[clip])
    assert doc.effective_config_for_clip(clip) == {"voice": "af_bella"}


# ---------------------------------------------------------------------------
# Document.replace_text
# ---------------------------------------------------------------------------

def test_replace_text_pure_insertion_outside_any_clip():
    doc = Document.from_plain_text("0123456789")
    removed = doc.replace_text(position=2, chars_removed=0, chars_added=3, new_text="01XYZ23456789")
    assert removed == []
    assert doc.text == "01XYZ23456789"


def test_replace_text_leaves_clip_entirely_before_edit_untouched():
    clip = Clip()
    doc = Document(runs=[Run(text="ABC", clip_id=clip.id), Run(text="DEFGHIJ")], clips=[clip])
    # Edit at position 4..6 - a full untagged character (index 3, "D") sits
    # between the clip's run and the edit, so this can't be mistaken for
    # "typing right at the clip's boundary" (see the inherited-tag rule).
    new_text = "ABCD" + "XX" + "GHIJ"
    doc.replace_text(position=4, chars_removed=2, chars_added=2, new_text=new_text)
    assert doc.clip_extent(clip.id) == (0, 3)
    assert doc.text == new_text


def test_replace_text_extends_clip_edited_in_place():
    # Clip spans "world" in "hello world".
    clip = Clip()
    doc = Document(runs=[Run(text="hello "), Run(text="world", clip_id=clip.id)], clips=[clip])
    old_text = doc.text
    position = 8  # inside the clip's run ("world" spans offsets 6..11)
    inserted = "onderful "
    new_text = old_text[:position] + inserted + old_text[position:]

    doc.replace_text(position=position, chars_removed=0, chars_added=len(inserted), new_text=new_text)

    assert doc.text == new_text
    # The insertion sat inside the clip's run, so it extends that same clip
    # rather than leaving a gap or spilling into the surrounding untagged text.
    assert doc.clip_text(clip) == "world"[:2] + inserted + "world"[2:]
    start, end = doc.clip_extent(clip.id)
    assert new_text[start:end] == doc.clip_text(clip)


def test_replace_text_removes_clip_fully_consumed_by_deletion():
    clip = Clip()
    doc = Document(runs=[Run(text="hello "), Run(text="world", clip_id=clip.id)], clips=[clip])
    new_text = "hello "
    removed = doc.replace_text(position=6, chars_removed=5, chars_added=0, new_text=new_text)
    assert removed == [clip]
    assert clip not in doc.clips
    assert doc.text == new_text


def test_replace_text_removes_clip_when_replacement_spans_it():
    clip = Clip()
    doc = Document(runs=[Run(text="hello "), Run(text="world", clip_id=clip.id)], clips=[clip])
    # Replace a range that starts before and ends after the clip's run.
    new_text = "hel" + "EVERYONE!"
    removed = doc.replace_text(position=3, chars_removed=8, chars_added=9, new_text=new_text)
    assert removed == [clip]
    assert doc.text == new_text
    # The inserted text was untagged (position 2, right before the edit, is
    # part of the untagged "hello " run) - it does not inherit the consumed
    # clip's id.
    assert doc.clip_covering(5) is None


def test_replace_text_at_document_start_is_untagged_by_default():
    clip = Clip()
    doc = Document(runs=[Run(text="hello", clip_id=clip.id)], clips=[clip])
    doc.replace_text(position=0, chars_removed=0, chars_added=3, new_text="Hi!hello")
    assert doc.clip_covering(0) is None
    assert doc.clip_extent(clip.id) == (3, 8)


def test_dirty_clips_delegates_to_dirty_module(monkeypatch):
    doc, _, _, clip = _make_document_with_one_character()
    other = Clip()
    doc.clips.append(other)  # untagged - no run points at it

    calls = []

    def fake_is_dirty(c, text, config):
        calls.append(c)
        return c is clip

    monkeypatch.setattr("kokoro_gui.daw.dirty.is_clip_dirty", fake_is_dirty)
    assert doc.dirty_clips() == [clip]
    assert calls == [clip, other]
