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

    def fake_is_dirty(c, text, config, key_fn=None):
        calls.append(c)
        return c is clip

    monkeypatch.setattr("kokoro_gui.daw.dirty.is_clip_dirty", fake_is_dirty)
    assert doc.dirty_clips() == [clip]
    assert calls == [clip, other]


# -- tracks made on first use (grill PR4) ---------------------------------------


def test_first_assignment_makes_the_characters_track_named_after_it():
    from kokoro_gui.daw.undo import AssignCharacterCommand

    alice = Character.from_preset_dict("Alice", {})
    bob = Character.from_preset_dict("Bob", {})
    doc = Document.from_plain_text("hello world", characters=[alice, bob])
    assert doc.tracks == []

    doc.undo_stack.push(AssignCharacterCommand(0, 5, alice.id))
    assert [(t.name, t.character_id) for t in doc.tracks] == [("Alice", alice.id)]
    first = doc.clip_covering(0)
    assert first.track_id == doc.tracks[0].id

    doc.undo_stack.push(AssignCharacterCommand(6, 11, bob.id))
    assert [t.name for t in doc.tracks] == ["Alice", "Bob"]
    assert doc.tracks[1].order_index > doc.tracks[0].order_index

    # A second use reuses the track.
    doc.undo_stack.push(AssignCharacterCommand(0, 5, alice.id))
    assert len(doc.tracks) == 2

    # Undoing a first use removes the track it made.
    doc.undo_stack.undo()
    doc.undo_stack.undo()
    assert [t.name for t in doc.tracks] == ["Alice"]
    doc.undo_stack.undo()
    assert doc.tracks == []


def test_an_unused_track_stays_in_the_model_but_is_not_used():
    alice = Character.from_preset_dict("Alice", {})
    doc = Document.from_plain_text("hello", characters=[alice])
    clip = doc.assign_character_to_range(0, 5, alice.id)
    track = doc.get_track(clip.track_id)
    track.gain = 0.5
    assert doc.used_tracks() == [track]

    doc.replace_text(0, 5, 0, "")  # the only clip is gone
    assert doc.used_tracks() == []
    assert track in doc.tracks

    doc.replace_text(0, 0, 3, "new")
    again = doc.assign_character_to_range(0, 3, alice.id)
    assert again.track_id == track.id and track.gain == 0.5


def test_unknown_or_no_character_makes_no_track():
    doc = Document.from_plain_text("hello")
    clip = doc.assign_character_to_range(0, 5, None)
    assert clip.track_id is None and doc.tracks == []
    doc.assign_character_to_range(0, 5, "nobody")
    assert doc.tracks == []


# ---------------------------------------------------------------------------
# Imported text (phase 5 P3, grill Q32): timing belongs to the text
# ---------------------------------------------------------------------------

SOURCE = "0123456789abcdef"
SOURCE_PATH = "/p/audio/imported/%s.wav" % SOURCE


def _word_rows(text, times, source=SOURCE):
    """`Run.words` for the space-separated words of `text`, one `(start_s,
    end_s)` pair each."""
    rows, pos = [], 0
    for token, (start_s, end_s) in zip(text.split(" "), times):
        at = text.index(token, pos)
        rows.append([at, at + len(token), source, start_s, end_s])
        pos = at + len(token)
    return rows


def _recording(*paragraphs, character=None):
    """A document of imported clips, one per `(text, times)` paragraph,
    joined by blank lines, over one source file."""
    runs, clips, tracks = [], [], []
    if character is not None:
        tracks.append(Track(name=character.name, character_id=character.id))
    for index, (text, times) in enumerate(paragraphs):
        if index:
            runs.append(Run(text="\n\n"))
        clip = Clip(source="imported", character_id=character.id if character else None,
                    track_id=tracks[0].id if tracks else None)
        clips.append(clip)
        runs.append(Run(text=text, clip_id=clip.id, kind="imported", words=_word_rows(text, times)))
    doc = Document(runs=runs, clips=clips, tracks=tracks, characters=[character] if character else [],
                   settings={"sources": {SOURCE: {"path": SOURCE_PATH, "sample_rate": 24000,
                                                  "duration_s": 10.0}}})
    doc.refresh_imported_segments()
    return doc, clips


HELLO = ("Hello there world", ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5)))


def _ranges(clip):
    return [list(s.range) for s in clip.segments]


def _words_text(doc, clip):
    """The text each of the clip's words covers, in order."""
    out = []
    for run in doc.runs:
        if run.clip_id == clip.id:
            out.extend(run.text[w[0]:w[1]] for w in run.words)
    return out


def test_an_imported_clip_derives_one_segment_from_contiguous_words():
    doc, (clip,) = _recording(HELLO)
    assert _ranges(clip) == [[0.0, 1.5]]
    segment = clip.segments[0]
    assert segment.audio_path == SOURCE_PATH
    assert segment.words == [["Hello", 0.0, 0.5], ["there", 0.5, 1.0], ["world", 1.0, 1.5]]
    assert doc.dirty_clips() == []


def test_deleting_a_middle_word_drops_its_timing_and_closes_up_the_audio():
    doc, (clip,) = _recording(HELLO)
    doc.replace_text(6, 6, 0, "Hello world")  # "there "
    assert doc.text == "Hello world"
    assert _words_text(doc, clip) == ["Hello", "world"]
    assert doc.runs[0].words == [[0, 5, SOURCE, 0.0, 0.5], [6, 11, SOURCE, 1.0, 1.5]]
    assert _ranges(clip) == [[0.0, 0.5], [1.0, 1.5]]


def test_deleting_the_first_word_leaves_one_contiguous_range():
    doc, (clip,) = _recording(HELLO)
    doc.replace_text(0, 6, 0, "there world")
    assert _words_text(doc, clip) == ["there", "world"]
    assert _ranges(clip) == [[0.5, 1.5]]


def test_deleting_half_a_word_deletes_its_audio_and_keeps_the_rest_as_untimed_text():
    doc, (clip,) = _recording(HELLO)
    doc.replace_text(8, 3, 0, "Hello th world")  # "ere" of "there"
    assert doc.text == "Hello th world"
    assert [c.id for c in doc.clips] == [clip.id]
    assert doc.clip_text(clip) == "Hello th world"
    assert _words_text(doc, clip) == ["Hello", "world"]
    assert _ranges(clip) == [[0.0, 0.5], [1.0, 1.5]]


def test_typing_inside_an_imported_run_splits_the_clip_around_an_untagged_run():
    host = Character.from_preset_dict("Host", {})
    doc, (clip,) = _recording(HELLO, character=host)
    doc.replace_text(12, 0, 3, "Hello there my world")  # "my " before "world"
    assert doc.text == "Hello there my world"
    assert [(r.text, r.kind) for r in doc.runs] == [
        ("Hello there ", "imported"), ("my ", None), ("world", "imported")]
    assert doc.runs[1].clip_id is None
    first, second = doc.clips
    assert first is clip and second.id != clip.id
    assert second.source == "imported"
    assert second.character_id == host.id and second.track_id == clip.track_id
    assert doc.runs[2].clip_id == second.id
    assert _words_text(doc, first) == ["Hello", "there"]
    assert _words_text(doc, second) == ["world"]
    assert _ranges(first) == [[0.0, 1.0]]
    assert _ranges(second) == [[1.0, 1.5]]
    assert doc.dirty_clips() == []


def test_typing_inside_a_word_drops_that_words_timing():
    doc, (clip,) = _recording(HELLO)
    doc.replace_text(8, 0, 1, "Hello thXere world")
    first, second = doc.clips
    assert _words_text(doc, first) == ["Hello"]
    assert doc.clip_text(first) == "Hello th"
    assert _words_text(doc, second) == ["world"]
    assert doc.clip_text(second) == "ere world"
    assert _ranges(first) == [[0.0, 0.5]] and _ranges(second) == [[1.0, 1.5]]


def test_typing_at_the_start_or_end_of_an_imported_run_does_not_split_it():
    doc, (clip,) = _recording(HELLO)
    doc.replace_text(0, 0, 3, "Oh Hello there world")
    doc.replace_text(20, 0, 1, "Oh Hello there world!")
    assert [(r.text, r.clip_id) for r in doc.runs] == [
        ("Oh ", None), ("Hello there world", clip.id), ("!", None)]
    assert doc.clips == [clip]
    assert _ranges(clip) == [[0.0, 1.5]]


def test_replacing_a_word_by_typing_over_it_splits_around_the_new_text():
    doc, (clip,) = _recording(HELLO)
    doc.replace_text(6, 5, 4, "Hello here world")  # "there" -> "here"
    assert [(r.text, r.kind) for r in doc.runs] == [
        ("Hello ", "imported"), ("here", None), (" world", "imported")]
    first, second = doc.clips
    assert _ranges(first) == [[0.0, 0.5]] and _ranges(second) == [[1.0, 1.5]]


def test_deleting_the_typed_text_between_two_halves_joins_them_again():
    # What Qt's native undo of the typing sends.
    doc, (clip,) = _recording(HELLO)
    doc.replace_text(12, 0, 3, "Hello there my world")
    doc.replace_text(12, 3, 0, "Hello there world")
    assert doc.clips == [clip]
    assert [(r.text, r.clip_id) for r in doc.runs] == [("Hello there world", clip.id)]
    assert _ranges(clip) == [[0.0, 1.5]]


def test_deleting_a_paragraph_break_does_not_join_two_imported_clips():
    doc, (a, b) = _recording(("Hello there", ((0.0, 0.5), (0.5, 1.0))),
                             ("Good morning", ((2.0, 2.4), (2.4, 3.0))))
    doc.replace_text(11, 2, 0, "Hello thereGood morning")
    assert [c.id for c in doc.clips] == [a.id, b.id]


def test_edit_touches_imported_says_which_edits_change_timed_text():
    doc, (clip,) = _recording(HELLO)
    doc.replace_text(17, 0, 7, "Hello there world, again")
    assert doc.edit_touches_imported(5, 1)  # the space between two words
    assert doc.edit_touches_imported(8, 0)  # typing inside
    assert not doc.edit_touches_imported(0, 0)  # typing at the start
    assert not doc.edit_touches_imported(17, 0)  # or the end
    assert not doc.edit_touches_imported(18, 3)  # untagged text after it


def test_deleting_the_text_between_two_halves_of_a_clip_touches_imported():
    doc, (clip,) = _recording(HELLO)
    doc.replace_text(11, 0, 2, "Hello there x world")  # " x" splits the clip
    assert doc.text == "Hello there x world"
    assert not doc.edit_touches_imported(12, 1)  # the "x" alone: the halves stay apart
    doc.replace_text(12, 1, 0, "Hello there  world")
    assert len(doc.clips) == 2
    # Deleting the last typed character joins the halves: a joined step.
    assert doc.edit_touches_imported(11, 1)
    doc.replace_text(11, 1, 0, "Hello there world")
    assert doc.clips == [clip]


def test_deleting_across_the_boundary_of_two_imported_clips_keeps_both():
    doc, (a, b) = _recording(("Hello there", ((0.0, 0.5), (0.5, 1.0))),
                             ("Good morning", ((2.0, 2.4), (2.4, 3.0))))
    assert doc.text == "Hello there\n\nGood morning"
    doc.replace_text(6, 12, 0, "Hello morning")  # "there\n\nGood "
    assert doc.text == "Hello morning"
    assert [c.id for c in doc.clips] == [a.id, b.id]
    assert doc.clip_text(a) == "Hello " and doc.clip_text(b) == "morning"
    assert _ranges(a) == [[0.0, 0.5]] and _ranges(b) == [[2.4, 3.0]]
    assert b.segments[0].words == [["morning", 0.0, 0.6]]


def test_deleting_a_whole_imported_clip_removes_it():
    doc, (a, b) = _recording(("Hello there", ((0.0, 0.5), (0.5, 1.0))),
                             ("Good morning", ((2.0, 2.4), (2.4, 3.0))))
    removed = doc.replace_text(0, 13, 0, "Good morning")
    assert removed == [a]
    assert doc.clips == [b] and _ranges(b) == [[2.0, 3.0]]


def test_undoing_a_text_edit_restores_the_words_and_the_one_clip():
    from kokoro_gui.daw.undo import TextEditCommand

    doc, (clip,) = _recording(HELLO)
    doc.undo_stack.push(TextEditCommand(12, 0, 3, "Hello there my world"))
    doc.undo_stack.push(TextEditCommand(0, 6, 0, "there my world"))
    assert len(doc.clips) == 2
    doc.undo_stack.undo()
    doc.undo_stack.undo()
    assert doc.text == "Hello there world"
    assert [c.id for c in doc.clips] == [clip.id]
    restored = doc.clips[0]
    assert _words_text(doc, restored) == ["Hello", "there", "world"]
    assert _ranges(restored) == [[0.0, 1.5]]


def test_assigning_a_character_to_the_typed_run_makes_an_ordinary_generated_clip():
    host = Character.from_preset_dict("Host", {})
    doc, (clip,) = _recording(HELLO, character=host)
    doc.replace_text(12, 0, 3, "Hello there my world")
    first, second = list(doc.clips)
    new = doc.assign_character_to_range(12, 15, host.id)
    assert new.source == "generated" and new.segments == []
    assert new in doc.dirty_clips()
    assert doc.get_clip(first.id) is first and doc.get_clip(second.id) is second
    assert _ranges(first) == [[0.0, 1.0]] and _ranges(second) == [[1.0, 1.5]]
    assert [(r.text, r.kind) for r in doc.runs] == [
        ("Hello there ", "imported"), ("my ", "generated"), ("world", "imported")]


def test_assigning_another_character_to_imported_text_only_changes_its_label():
    host = Character.from_preset_dict("Host", {})
    guest = Character.from_preset_dict("Guest", {})
    doc, (clip,) = _recording(HELLO, character=host)
    doc.characters.append(guest)
    segments = list(clip.segments)

    result = doc.assign_character_to_range(0, len(doc.text), guest.id)

    assert result is clip and doc.clips == [clip]
    assert clip.source == "imported" and clip.character_id == guest.id
    assert doc.get_track(clip.track_id).character_id == guest.id
    assert _words_text(doc, clip) == ["Hello", "there", "world"]
    assert clip.segments == segments
    assert doc.dirty_clips() == []


def test_assigning_a_character_to_part_of_an_imported_clip_splits_off_a_relabeled_clip():
    host = Character.from_preset_dict("Host", {})
    guest = Character.from_preset_dict("Guest", {})
    doc, (clip,) = _recording(HELLO, character=host)
    doc.characters.append(guest)

    middle = doc.assign_character_to_range(6, 11, guest.id)  # "there"

    assert [c.source for c in doc.clips] == ["imported"] * 3
    before, after = clip, next(c for c in doc.clips if c not in (clip, middle))
    assert middle.character_id == guest.id
    assert before.character_id == host.id and after.character_id == host.id
    assert doc.clip_text(before) == "Hello " and doc.clip_text(middle) == "there"
    assert doc.clip_text(after) == " world"
    assert _ranges(before) == [[0.0, 0.5]]
    assert _ranges(middle) == [[0.5, 1.0]]
    assert _ranges(after) == [[1.0, 1.5]]


def test_assigning_over_imported_and_typed_text_relabels_one_and_generates_the_other():
    host = Character.from_preset_dict("Host", {})
    guest = Character.from_preset_dict("Guest", {})
    doc, (clip,) = _recording(HELLO, character=host)
    doc.characters.append(guest)
    doc.replace_text(17, 0, 7, "Hello there world, again")  # untagged ", again"

    new = doc.assign_character_to_range(0, len(doc.text), guest.id)

    assert new.source == "generated" and doc.clip_text(new) == ", again"
    assert clip.character_id == guest.id and clip.source == "imported"
    assert _ranges(clip) == [[0.0, 1.5]]


def test_apply_words_tags_a_pasted_span_as_a_new_imported_clip():
    host = Character.from_preset_dict("Host", {})
    doc = Document.from_plain_text("Intro. ", characters=[host])
    sources = {SOURCE: {"path": SOURCE_PATH, "sample_rate": 24000, "duration_s": 10.0}}

    doc.replace_text(7, 0, 5, "Intro. there")
    clip_id = doc.apply_words(7, 5, [[0, 5, SOURCE, 0.5, 1.0]], sources, character_id=host.id)

    clip = doc.get_clip(clip_id)
    assert clip.source == "imported" and clip.character_id == host.id
    assert clip.track_id == doc.track_for_character(host.id)
    assert doc.clip_extent(clip_id) == (7, 12)
    assert _ranges(clip) == [[0.5, 1.0]]
    assert doc.sources == sources


def test_apply_words_on_a_generated_clip_carves_the_span_out_of_it():
    host = Character.from_preset_dict("Host", {})
    doc = Document.from_plain_text("Say hi now", characters=[host])
    generated = doc.assign_character_to_range(0, 10, host.id)
    sources = {SOURCE: {"path": SOURCE_PATH, "sample_rate": 24000, "duration_s": 10.0}}
    doc.replace_text(4, 0, 6, "Say there hi now")  # the paste joined the generated clip

    clip_id = doc.apply_words(4, 6, [[0, 5, SOURCE, 0.5, 1.0]], sources)

    assert doc.clip_extent(clip_id) == (4, 10)
    assert generated not in doc.clips
    kinds = [(r.text, r.kind) for r in doc.runs]
    assert kinds == [("Say ", "generated"), ("there ", "imported"), ("hi now", "generated")]


def test_pasting_a_cut_word_back_rejoins_the_two_halves():
    doc, (clip,) = _recording(HELLO)
    doc.replace_text(6, 6, 0, "Hello world")  # cut "there "
    assert _ranges(clip) == [[0.0, 0.5], [1.0, 1.5]]
    doc.replace_text(6, 0, 6, "Hello there world")  # the plain paste splits the clip
    assert len(doc.clips) == 2

    clip_id = doc.apply_words(6, 6, [[0, 5, SOURCE, 0.5, 1.0]], {})

    assert clip_id == clip.id and doc.clips == [clip]
    assert _words_text(doc, clip) == ["Hello", "there", "world"]
    assert _ranges(clip) == [[0.0, 1.5]]


def test_apply_words_extends_the_imported_clip_it_lands_at_the_edge_of():
    doc, (clip,) = _recording(("Hello there", ((0.0, 0.5), (0.5, 1.0))))
    doc.replace_text(11, 0, 6, "Hello there world")
    clip_id = doc.apply_words(11, 6, [[1, 6, SOURCE, 1.0, 1.5]], {})
    assert clip_id == clip.id
    assert doc.clip_extent(clip.id) == (0, 17)
    assert _ranges(clip) == [[0.0, 1.5]]


def test_apply_words_with_an_unknown_source_leaves_the_span_untagged():
    doc = Document.from_plain_text("there")
    assert doc.apply_words(0, 5, [[0, 5, "missing", 0.5, 1.0]], {}) is None
    assert doc.apply_words(0, 5, [[0, 5, "gone", 0.5, 1.0]], {"gone": {"path": None}}) is None
    assert doc.clips == [] and doc.runs[0].clip_id is None


def test_apply_words_command_undo_restores_runs_clips_and_sources():
    from kokoro_gui.daw.undo import ApplyWordsCommand

    doc = Document.from_plain_text("there")
    sources = {SOURCE: {"path": SOURCE_PATH, "sample_rate": 24000, "duration_s": 10.0}}
    command = ApplyWordsCommand(0, 5, [[0, 5, SOURCE, 0.5, 1.0]], sources)
    doc.undo_stack.push(command)
    assert command.clip_id is not None and doc.get_clip(command.clip_id).source == "imported"
    doc.undo_stack.undo()
    assert doc.clips == [] and doc.runs[0].words == [] and "sources" not in doc.settings
    doc.undo_stack.redo()
    assert doc.sources == sources and len(doc.clips) == 1
