"""Tests for kokoro_gui/daw/serialization.py's document.json round trip."""
from kokoro_gui.daw.models import Character, Clip, Document, Segment, Track
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
        start_offset=0,
        end_offset=5,
        character_id=character.id,
        track_id=track.id,
        overrides={"speed": 1.5},
        segments=[segment],
    )
    return Document(text="hello world", clips=[clip], tracks=[track], characters=[character], settings={"x": 1})


def test_document_to_dict_is_json_plain_shapes():
    doc = _sample_document()
    data = document_to_dict(doc)
    assert data["text"] == "hello world"
    assert data["settings"] == {"x": 1}
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


def test_document_from_dict_tolerates_missing_keys():
    restored = document_from_dict({})
    assert restored.text == ""
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
    clip = Clip(start_offset=0, end_offset=11, character_id=character.id, track_id=track.id)
    doc = Document(text="hello world", clips=[clip], tracks=[track], characters=[character])

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
