"""Tests for kokoro_gui/daw/lanes.py: the unified track layout's lane rule
and relaning (grill PR4)."""
from kokoro_gui.daw.lanes import assign_lanes, lane_numbers, lane_tracks, plan_relane
from kokoro_gui.daw.models import Character, Clip, Document, Run, Track
from kokoro_gui.daw.undo import AssignCharacterCommand, RelaneCommand, SetFieldCommand


def _conversation(speakers, **settings):
    """A document with one clip per letter in `speakers` ("ABAB"), in text
    order, each on its character's track."""
    characters = {name: Character.from_preset_dict(name, {}) for name in sorted(set(speakers))}
    tracks = {name: Track(name=name, character_id=c.id, order_index=i)
              for i, (name, c) in enumerate(characters.items())}
    clips, runs = [], []
    for i, name in enumerate(speakers):
        clip = Clip(character_id=characters[name].id, track_id=tracks[name].id)
        clips.append(clip)
        runs.append(Run(text=f"{name}{i} ", clip_id=clip.id, kind="generated"))
    doc = Document(runs=runs, clips=clips, tracks=list(tracks.values()),
                   characters=list(characters.values()), settings=dict(settings))
    return doc, clips


def _lanes(doc, clips, n):
    numbers = lane_numbers(doc, n)
    return [numbers[c.id] for c in clips]


def test_alternating_speakers_walk_the_lanes_and_wrap():
    doc, clips = _conversation("ABAB")
    assert _lanes(doc, clips, 3) == [1, 2, 3, 1]


def test_same_speaker_stays_on_its_lane():
    doc, clips = _conversation("AAB")
    assert _lanes(doc, clips, 3) == [1, 1, 2]


def test_one_lane_puts_everything_on_lane_one():
    doc, clips = _conversation("ABCAB")
    assert _lanes(doc, clips, 1) == [1, 1, 1, 1, 1]


def test_lane_order_follows_the_text_not_the_clip_list():
    doc, clips = _conversation("AB")
    doc.clips = list(reversed(doc.clips))
    assert _lanes(doc, clips, 3) == [1, 2]


def test_relane_creates_only_the_lanes_used_and_undo_removes_them():
    doc, clips = _conversation("AB", track_layout={"mode": "unified", "lanes": 3})
    character_tracks = list(doc.tracks)

    doc.undo_stack.push(RelaneCommand())

    lanes = lane_tracks(doc)
    assert sorted(lanes) == [1, 2]
    assert [lanes[1].name, lanes[2].name] == ["Lane 1", "Lane 2"]
    assert all(t.character_id is None for t in lanes.values())
    assert [c.track_id for c in clips] == [lanes[1].id, lanes[2].id]
    assert assign_lanes(doc, 3) == {clips[0].id: lanes[1].id, clips[1].id: lanes[2].id}
    # The character tracks stay (with their mixer settings), just unused.
    assert all(t in doc.tracks for t in character_tracks)
    assert [t.name for t in doc.used_tracks()] == ["Lane 1", "Lane 2"]

    doc.undo_stack.undo()
    assert lane_tracks(doc) == {}
    assert [c.track_id for c in clips] == [character_tracks[0].id, character_tracks[1].id]


def test_switching_the_layout_relanes_in_the_same_undo_step():
    doc, clips = _conversation("ABA")
    before = [c.track_id for c in clips]

    doc.undo_stack.push(SetFieldCommand("document", None, "settings",
                                        {"mode": "unified", "lanes": 2}, key="track_layout"))

    lanes = lane_tracks(doc)
    assert [c.track_id for c in clips] == [lanes[1].id, lanes[2].id, lanes[1].id]

    doc.undo_stack.undo()  # one step: the setting and the lanes
    assert "track_layout" not in doc.settings
    assert [c.track_id for c in clips] == before
    assert lane_tracks(doc) == {}

    doc.undo_stack.redo()
    assert [doc.get_track(c.track_id).lane for c in clips] == [1, 2, 1]


def test_switching_back_puts_clips_on_their_character_tracks_creating_missing_ones():
    doc, clips = _conversation("AB", track_layout={"mode": "unified", "lanes": 3})
    doc.undo_stack.push(RelaneCommand())
    # B's character track was deleted while in unified mode.
    b_character = clips[1].character_id
    doc.tracks = [t for t in doc.tracks if t.character_id != b_character]

    doc.undo_stack.push(SetFieldCommand("document", None, "settings", {"mode": "character"}, key="track_layout"))

    a_track = doc.get_track(clips[0].track_id)
    b_track = doc.get_track(clips[1].track_id)
    assert a_track.character_id == clips[0].character_id
    assert b_track.character_id == b_character
    assert b_track.name == doc.get_character(b_character).name


def test_an_assignment_in_unified_mode_relanes_in_one_undo_step():
    doc, clips = _conversation("AB", track_layout={"mode": "unified", "lanes": 3})
    doc.undo_stack.push(RelaneCommand())
    doc.runs.append(Run(text="new words"))
    start = len(doc.text) - len("new words")
    a = clips[0].character_id

    doc.undo_stack.push(AssignCharacterCommand(start, len(doc.text), a))

    new_clip = doc.clip_covering(start)
    assert doc.get_track(new_clip.track_id).lane == 3  # A, B, A: a speaker change
    assert not any(t.character_id == a and t.id == new_clip.track_id for t in doc.tracks)

    doc.undo_stack.undo()
    assert doc.clip_covering(start) is None
    assert sorted(lane_tracks(doc)) == [1, 2]


def test_per_character_layout_does_not_relane_after_assignments():
    doc, clips = _conversation("AB")
    doc.runs.append(Run(text="more"))
    start = len(doc.text) - 4
    doc.undo_stack.push(AssignCharacterCommand(start, len(doc.text), clips[0].character_id))
    assert lane_tracks(doc) == {}
    assert doc.clip_covering(start).track_id == clips[0].track_id


def test_plan_is_empty_when_everything_is_in_place():
    doc, _clips = _conversation("AB", track_layout={"mode": "unified", "lanes": 3})
    doc.undo_stack.push(RelaneCommand())
    assert not plan_relane(doc)


def test_track_layout_is_normalised():
    doc = Document(settings={"track_layout": {"mode": "unified", "lanes": "0"}})
    assert doc.track_layout() == {"mode": "unified", "lanes": 1}
    assert Document(settings={"track_layout": "junk"}).track_layout() == {"mode": "character"}
    assert Document().track_layout() == {"mode": "character"}


def test_an_imported_recording_and_a_pasted_timed_span_get_lanes_in_unified_mode():
    from kokoro_gui.daw.undo import ApplyWordsCommand, ImportRecordingCommand

    source = "a" * 16
    sources = {source: {"path": "/p/audio/imported/a.wav", "sample_rate": 24000, "duration_s": 60.0}}
    doc, clips = _conversation("AB", track_layout={"mode": "unified", "lanes": 3})
    doc.undo_stack.push(RelaneCommand())
    a, b = clips[0].character_id, clips[1].character_id

    command = ImportRecordingCommand([{"text": "hello there", "character_id": b,
                                       "words": [[0, 5, source, 0.0, 0.5], [6, 11, source, 0.5, 1.0]]}], sources)
    doc.undo_stack.push(command)

    recording = doc.get_clip(command.clip_ids[0])
    assert recording.track_id is not None
    assert doc.get_track(recording.track_id).lane == 2  # A, B, B: no speaker change

    start = len(doc.text)
    doc.replace_text(start, 0, 4, doc.text + " two")
    paste = ApplyWordsCommand(start + 1, 3, [[0, 3, source, 1.3, 1.6]], sources, character_id=a)
    doc.undo_stack.push(paste)

    pasted = doc.get_clip(paste.clip_id)
    assert pasted is not None and pasted.track_id is not None
    assert doc.get_track(pasted.track_id).lane == 3  # A, B, B, A
    assert all(doc.get_track(c.track_id).lane is not None for c in doc.clips)

    doc.undo_stack.undo()
    doc.undo_stack.undo()
    assert sorted(lane_tracks(doc)) == [1, 2]
