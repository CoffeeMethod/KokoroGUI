"""Tests for kokoro_gui/daw/serialization.py's document.json round trip."""
from kokoro_gui.daw.models import Character, Clip, Document, Run, Segment, Track
from kokoro_gui.daw.serialization import (
    document_from_dict,
    document_to_dict,
    load_document,
    save_document,
)


def _sample_document():
    character = Character.from_preset_dict("Alice", {"voice": "af_bella", "speed": 1.1})
    track = Track(name="Alice", character_id=character.id)
    segment = Segment(order_index=0, text="hello", cache_key="deadbeef", duration=1.5)
    clip = Clip(
        character_id=character.id, track_id=track.id,
        overrides={"speed": 1.5}, segments=[segment],
    )
    doc = Document(
        runs=[Run(text="hello", clip_id=clip.id, kind=clip.source), Run(text=" world")],
        clips=[clip], tracks=[track], characters=[character], settings={"x": 1},
    )
    return doc


def test_document_to_dict_is_json_plain_shapes():
    doc = _sample_document()
    data = document_to_dict(doc)
    assert data["settings"] == {"x": 1}
    assert data["runs"][0]["text"] == "hello"
    assert data["runs"][0]["clip_id"] == doc.clips[0].id
    assert data["clips"][0]["segments"][0]["cache_key"] == "deadbeef"
    assert data["characters"][0]["preset_data"] == {"voice": "af_bella", "speed": 1.1}


def test_document_from_dict_round_trips_to_dict():
    doc = _sample_document()
    restored = document_from_dict(document_to_dict(doc))

    assert restored.text == doc.text
    assert restored.settings == doc.settings
    assert len(restored.clips) == 1
    assert restored.clips[0].id == doc.clips[0].id
    assert restored.clips[0].overrides == {"speed": 1.5}
    assert len(restored.clips[0].segments) == 1
    assert restored.clips[0].segments[0].cache_key == "deadbeef"
    assert restored.tracks[0].id == doc.tracks[0].id
    assert restored.characters[0].preset_data == {"voice": "af_bella", "speed": 1.1}
    assert restored.clip_extent(restored.clips[0].id) == doc.clip_extent(doc.clips[0].id)


def test_document_from_dict_tolerates_missing_keys():
    restored = document_from_dict({})
    assert restored.text == ""
    assert restored.runs == []
    assert restored.clips == []
    assert restored.tracks == []
    assert restored.characters == []
    assert restored.settings == {}


def test_save_and_load_document_round_trip(tmp_path):
    doc = _sample_document()
    path = tmp_path / "sub" / "document.json"
    save_document(doc, str(path))
    assert path.exists()

    loaded = load_document(str(path))
    assert loaded.text == doc.text
    assert loaded.clips[0].id == doc.clips[0].id


def test_load_document_missing_file_returns_none(tmp_path):
    assert load_document(str(tmp_path / "nope.json")) is None


def test_load_document_corrupt_json_returns_none(tmp_path):
    path = tmp_path / "document.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert load_document(str(path)) is None


def test_generated_clip_round_trips_through_save_and_load(tmp_path, engine, fake_pipeline, make_config):
    """Segment round-tripping was previously only exercised with hand-built
    data (see _sample_document above) - this drives it through the actual
    per-clip Generate path (kokoro_gui/engine/conversion.py's
    generate_clip_audio) so real generation-shaped Segments are covered too."""
    import asyncio

    from kokoro_gui.daw.dirty import compute_expected_cache_hash

    character = Character.from_preset_dict("Alice", {"voice": "af_heart", "speed": 1.0})
    track = Track(name="Alice", character_id=character.id)
    clip = Clip(character_id=character.id, track_id=track.id)
    doc = Document(
        runs=[Run(text="hello world", clip_id=clip.id, kind=clip.source)],
        clips=[clip], tracks=[track], characters=[character],
    )

    config = make_config(voice="af_heart", speed=1.0)
    text = doc.clip_text(clip)
    results = asyncio.run(engine.generate_clip_audio((0, text, config)))
    expected_hash = compute_expected_cache_hash(text, config)
    clip.segments = [
        Segment(order_index=i, text=r["text"], cache_key=expected_hash, audio_path=r["path"], duration=r["duration"])
        for i, r in enumerate(results)
    ]

    path = tmp_path / "document.json"
    save_document(doc, str(path))
    loaded = load_document(str(path))

    assert len(loaded.clips[0].segments) == len(clip.segments)
    for original, restored in zip(clip.segments, loaded.clips[0].segments):
        assert restored.order_index == original.order_index
        assert restored.cache_key == original.cache_key
        assert restored.audio_path == original.audio_path
        assert restored.duration == original.duration


# ---------------------------------------------------------------------------
# Legacy (pre-run-list) document.json migration
# ---------------------------------------------------------------------------

def test_document_from_dict_migrates_legacy_offset_shape():
    """A document.json written before the tagged-run rework has no "runs"
    key at all - just a flat "text" string plus offset-ranged clips. Loading
    one should synthesize an equivalent run list on the fly, per
    Claude/PLAN_text_editor_redesign.md's "Migration path" section."""
    character = Character.from_preset_dict("Alice", {"voice": "af_bella"})
    legacy_data = {
        "text": "hello world",
        "clips": [
            {
                "start_offset": 6, "end_offset": 11,
                "character_id": character.id, "track_id": None,
                "overrides": {}, "fx_override": None, "timeline_timestamp": None,
                "segments": [], "source": "generated", "original_audio_path": None,
                "id": "legacy-clip-id",
            },
        ],
        "tracks": [],
        "characters": [{"name": "Alice", "preset_data": {"voice": "af_bella"},
                        "highlight_color": "#f4b400", "backend_id": "kokoro", "id": character.id}],
        "settings": {},
    }

    doc = document_from_dict(legacy_data)

    assert doc.text == "hello world"
    clip = doc.get_clip("legacy-clip-id")
    assert clip is not None
    assert doc.clip_extent(clip.id) == (6, 11)
    assert doc.clip_text(clip) == "world"
    assert doc.clip_covering(0) is None  # "hello " stays untagged


def test_document_from_dict_migrates_legacy_shape_with_gap_at_start():
    legacy_data = {
        "text": "0123456789",
        "clips": [
            {
                "start_offset": 5, "end_offset": 10,
                "character_id": None, "track_id": None,
                "overrides": {}, "fx_override": None, "timeline_timestamp": None,
                "segments": [], "source": "generated", "original_audio_path": None,
                "id": "clip-a",
            },
        ],
        "tracks": [], "characters": [], "settings": {},
    }

    doc = document_from_dict(legacy_data)

    assert doc.text == "0123456789"
    assert doc.clip_covering(0) is None
    assert doc.clip_covering(5).id == "clip-a"
    assert doc.clip_extent("clip-a") == (5, 10)


# ---------------------------------------------------------------------------
# Unknown fields round-trip (Claude/old/PLAN_tbaw_bundle.md section 2.2)
# ---------------------------------------------------------------------------

def test_unknown_keys_on_every_object_survive_a_round_trip():
    data = {
        "runs": [{"text": "hello", "clip_id": "c1", "kind": "generated", "future_run_key": 1}],
        "clips": [{
            "id": "c1", "character_id": "ch1", "future_clip_key": {"nested": True},
            "segments": [{"order_index": 0, "text": "hello", "cache_key": "k", "raw": True,
                          "word_timings": [[0, 0.5]]}],
        }],
        "tracks": [{"name": "T", "id": "t1", "future_track_key": "x"}],
        "characters": [{
            "name": "Alice", "id": "ch1", "future_character_key": "lib-1",
            "preset_data": {"voice": "af_bella", "unknown_preset_key": 7},
        }],
        "settings": {},
    }
    doc = document_from_dict(data)

    # The whitelist still guards what reaches a config dict.
    assert doc.characters[0].preset_data == {"voice": "af_bella"}
    assert doc.characters[0].extra == {"future_character_key": "lib-1", "preset_data": {"unknown_preset_key": 7}}
    assert doc.clips[0].extra == {"future_clip_key": {"nested": True}}
    assert doc.clips[0].segments[0].extra == {"word_timings": [[0, 0.5]]}
    assert doc.runs[0].extra == {"future_run_key": 1}
    assert doc.tracks[0].extra == {"future_track_key": "x"}

    out = document_to_dict(doc)
    assert out["runs"][0]["future_run_key"] == 1
    assert out["clips"][0]["future_clip_key"] == {"nested": True}
    assert out["clips"][0]["segments"][0]["word_timings"] == [[0, 0.5]]
    assert out["tracks"][0]["future_track_key"] == "x"
    assert out["characters"][0]["future_character_key"] == "lib-1"
    assert out["characters"][0]["preset_data"] == {"voice": "af_bella", "unknown_preset_key": 7}
    assert "extra" not in out["clips"][0] and "extra" not in out["characters"][0]

    # And it reads back the same at the dict level.
    assert document_to_dict(document_from_dict(out)) == out


def test_segment_engine_version_round_trips():
    segment = Segment(order_index=0, text="x", cache_key="k", engine_version="0.9.4")
    clip = Clip(segments=[segment])
    doc = Document(runs=[Run(text="x", clip_id=clip.id)], clips=[clip])
    restored = document_from_dict(document_to_dict(doc))
    assert restored.clips[0].segments[0].engine_version == "0.9.4"


def test_phase_two_fields_round_trip_when_set():
    import json

    doc = _sample_document()
    clip, track, character = doc.clips[0], doc.tracks[0], doc.characters[0]
    clip.gap_before_s = 1.25
    clip.fade_in_s, clip.fade_out_s = 0.1, 0.2
    clip.status, clip.note, clip.source_text = "approved", "good read", "Bonjour"
    clip.takes = {0: [Segment(order_index=0, text="hello", cache_key="old", audio_path="/p/old_0.wav",
                              duration=1.0, words=[["hello", 0.0, 0.5]], onset_s=0.01, tail_s=0.02)]}
    clip.segments[0].words = [["hello", 0.1, 0.6]]
    clip.segments[0].onset_s, clip.segments[0].tail_s = 0.05, 0.1
    track.gain, track.mute, track.solo, track.pan = 0.5, True, True, -0.5
    track.automation = [[0.0, 1.0], [2.0, 0.5]]
    character.variants = {"angry": "alice_angry"}

    back = document_from_dict(json.loads(json.dumps(document_to_dict(doc))))
    c2, t2, ch2 = back.clips[0], back.tracks[0], back.characters[0]
    assert (c2.gap_before_s, c2.fade_in_s, c2.fade_out_s) == (1.25, 0.1, 0.2)
    assert (c2.status, c2.note, c2.source_text) == ("approved", "good read", "Bonjour")
    assert list(c2.takes) == [0]
    parked = c2.takes[0][0]
    assert isinstance(parked, Segment)
    assert parked.cache_key == "old" and parked.words == [["hello", 0.0, 0.5]] and parked.tail_s == 0.02
    assert c2.segments[0].words == [["hello", 0.1, 0.6]]
    assert (c2.segments[0].onset_s, c2.segments[0].tail_s) == (0.05, 0.1)
    assert (t2.gain, t2.mute, t2.solo, t2.pan) == (0.5, True, True, -0.5)
    assert t2.automation == [[0.0, 1.0], [2.0, 0.5]]
    assert ch2.variants == {"angry": "alice_angry"}


def test_phase_two_fields_default_when_absent():
    doc = _sample_document()
    data = document_to_dict(doc)
    for key in ("gap_before_s", "takes", "fade_in_s", "fade_out_s", "status", "note", "source_text"):
        data["clips"][0].pop(key)
    for key in ("gain", "mute", "solo", "pan", "automation"):
        data["tracks"][0].pop(key)
    for key in ("words", "onset_s", "tail_s"):
        data["clips"][0]["segments"][0].pop(key)
    data["characters"][0].pop("variants")

    back = document_from_dict(data)
    clip, track = back.clips[0], back.tracks[0]
    assert clip.gap_before_s is None and clip.takes == {} and clip.status == "todo"
    assert clip.fade_in_s == 0.0 and clip.note == "" and clip.source_text is None
    assert (track.gain, track.mute, track.solo, track.pan, track.automation) == (1.0, False, False, 0.0, [])
    segment = clip.segments[0]
    assert segment.words == [] and segment.onset_s is None and segment.tail_s is None
    assert back.characters[0].variants == {}


def test_rewrite_audio_paths_walks_parked_takes():
    from kokoro_gui.daw.serialization import rewrite_audio_paths

    doc = _sample_document()
    doc.clips[0].segments[0].audio_path = "a.wav"
    doc.clips[0].takes = {2: [Segment(audio_path="b.wav")]}
    data = rewrite_audio_paths(document_to_dict(doc), lambda p: "X/" + p)
    assert data["clips"][0]["segments"][0]["audio_path"] == "X/a.wav"
    assert data["clips"][0]["takes"]["2"][0]["audio_path"] == "X/b.wav"


# ---------------------------------------------------------------------------
# Character.library_id (phase 3, grill WF12)
# ---------------------------------------------------------------------------

def test_character_library_id_round_trips():
    linked = Character.from_preset_dict("Narrator", {"voice": "af_bella"}, library_id="lib-7")
    local = Character.from_preset_dict("Guest", {"voice": "am_adam"})
    doc = Document(characters=[linked, local])

    data = document_to_dict(doc)
    assert data["characters"][0]["library_id"] == "lib-7"
    assert data["characters"][1]["library_id"] is None

    restored = document_from_dict(data)
    assert restored.characters[0].library_id == "lib-7"
    assert restored.characters[0].id == linked.id
    assert restored.characters[1].library_id is None
    assert restored.characters[0].extra == {}


def test_character_without_library_id_loads_local():
    doc = document_from_dict({"characters": [{"name": "Old", "id": "c1", "preset_data": {}}]})
    assert doc.characters[0].library_id is None
