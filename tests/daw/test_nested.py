"""Nested clips (phase 4, grill NP1-NP8): a subproject placed on a parent's
timeline as a clip with `source == "nested"`, a `child` reference and one
read-only placeholder run holding the child's title."""
import pytest

from kokoro_gui.daw.arrangement import compute_arrangement
from kokoro_gui.daw.auto_split import plan_auto_split_clips
from kokoro_gui.daw.models import PLACEHOLDER, Character, Clip, Document
from kokoro_gui.daw.serialization import document_from_dict, document_to_dict


def _book():
    narrator = Character.from_preset_dict("Narrator", {})
    doc = Document.from_plain_text("Intro text.\n\nOutro text.", characters=[narrator])
    chapter = doc.insert_nested_clip(len("Intro text.\n\n"), {"kind": "embedded", "id": "abc123"}, "Chapter 1")
    return doc, chapter, narrator


def test_insert_puts_one_placeholder_run_with_the_title():
    doc, chapter, _ = _book()
    assert chapter.is_nested and chapter.source == "nested"
    assert chapter.child == {"kind": "embedded", "id": "abc123"}
    assert chapter.segments == [] and chapter.takes == {}
    assert doc.clip_text(chapter) == "Chapter 1"
    assert doc.text == "Intro text.\n\nChapter 1Outro text."
    runs = [r for r in doc.runs if r.clip_id == chapter.id]
    assert len(runs) == 1 and runs[0].kind == PLACEHOLDER
    assert doc.clip_extent(chapter.id) == (13, 22)
    assert doc.placeholder_extent(chapter.id) == (13, 22)
    assert chapter.run_kind == PLACEHOLDER


def test_insert_inside_a_clip_lands_after_it():
    narrator = Character.from_preset_dict("Narrator", {})
    doc = Document.from_plain_text("hello world", characters=[narrator])
    clip = doc.assign_character_to_range(0, 11, narrator.id)
    nested = doc.insert_nested_clip(5, {"kind": "embedded", "id": "x"}, "Part")
    assert doc.clip_extent(clip.id) == (0, 11)
    assert doc.clip_extent(nested.id) == (11, 15)


def test_round_trip_keeps_child_source_and_placeholder():
    doc, chapter, _ = _book()
    doc.settings["x"] = 1
    restored = document_from_dict(document_to_dict(doc))
    again = restored.get_clip(chapter.id)
    assert again.source == "nested"
    assert again.child == {"kind": "embedded", "id": "abc123"}
    assert [r.kind for r in restored.runs if r.clip_id == chapter.id] == [PLACEHOLDER]
    assert restored.text == doc.text


def test_linked_child_round_trips_its_path():
    doc = Document.from_plain_text("")
    clip = doc.insert_nested_clip(0, {"kind": "linked", "id": "p1", "path": "chapters/one.tbaw"}, "One")
    restored = document_from_dict(document_to_dict(doc))
    assert restored.get_clip(clip.id).child["path"] == "chapters/one.tbaw"


def test_unknown_source_is_still_refused():
    with pytest.raises(ValueError):
        Clip(source="video")


def test_staleness_comes_from_the_nested_state_fn():
    doc, chapter, _ = _book()
    assert chapter in doc.dirty_clips()  # headless: stale
    doc.nested_state_fn = lambda clip: False
    assert chapter not in doc.dirty_clips()
    doc.nested_state_fn = lambda clip: clip.id == chapter.id
    assert chapter in doc.dirty_clips()


def test_assigning_a_character_over_the_placeholder_is_refused():
    doc, chapter, narrator = _book()
    start, end = doc.clip_extent(chapter.id)
    with pytest.raises(ValueError):
        doc.assign_character_to_range(start - 2, end + 2, narrator.id)
    assert doc.overlaps_nested(start, start + 1)
    assert not doc.overlaps_nested(0, 5)
    doc.assign_character_to_range(0, 5, narrator.id)  # elsewhere is fine


def test_auto_split_never_retags_the_placeholder():
    doc, chapter, narrator = _book()
    triples, _unmatched = plan_auto_split_clips(doc, split_by_paragraph=True)
    for start, end, _cid in triples:
        assert not doc.overlaps_nested(start, end)
    assert triples  # the narration around it still splits


def test_arrangement_places_a_nested_clip_at_an_estimate_without_the_app():
    doc, chapter, _ = _book()
    placed = compute_arrangement(doc, chars_per_second=10.0).by_clip_id()[chapter.id]
    assert placed.estimated
    assert placed.duration_s == pytest.approx(len("Chapter 1") / 10.0)


def test_renaming_rewrites_the_placeholder_run():
    doc, chapter, _ = _book()
    doc.set_placeholder_text(chapter.id, "The Beginning")
    assert doc.clip_text(chapter) == "The Beginning"
    doc.set_placeholder_text(chapter.id, "")
    assert doc.clip_text(chapter) == "Subproject"


def test_required_features_name_nested():
    from kokoro_gui.qt.project import SUPPORTED_FEATURES, required_features

    doc, _chapter, _ = _book()
    assert "nested" in required_features(doc)
    assert "nested" in SUPPORTED_FEATURES
    assert required_features(Document.from_plain_text("plain")) == []


def test_display_title_prefers_the_title_setting():
    from kokoro_gui.qt.project import display_title

    assert display_title({"title": " Chapter 12 "}, "/x/book.tbaw") == "Chapter 12"
    assert display_title({}, "/x/book.tbaw") == "book"
    assert display_title({}, None, "Subproject") == "Subproject"
