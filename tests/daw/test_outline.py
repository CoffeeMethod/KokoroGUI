"""The Outline dock's rows (plan 20): chapter status derived from the nested
clip's state and its own `status`, totals, the estimated flag and the marker
fallback."""
import pytest

from kokoro_gui.daw import outline
from kokoro_gui.daw.arrangement import compute_arrangement
from kokoro_gui.daw.models import Character, Document


def _book(titles=("Chapter 1", "Chapter 2")):
    doc = Document.from_plain_text("Intro.\n\n", characters=[Character.from_preset_dict("Narrator", {})])
    clips = []
    for index, title in enumerate(titles):
        clips.append(doc.insert_nested_clip(len(doc.text), {"kind": "embedded", "id": f"c{index}"}, title))
    return doc, clips


def _arrange(doc, lengths):
    """Placed in text order; `lengths[clip_id]` is a rendered length, a clip
    without one is estimated at 2 s."""
    return compute_arrangement(
        doc, chars_per_second=15.0,
        clip_duration=lambda clip: lengths.get(clip.id),
        clip_estimate=lambda clip: 2.0 if clip.is_nested else None)


def _states(mapping):
    return lambda clip: mapping[clip.id]


def test_a_chapter_row_per_nested_clip_in_text_order():
    doc, (one, two) = _book()
    result = outline.build(doc, _arrange(doc, {one.id: 60.0, two.id: 90.0}), _states({one.id: "ok", two.id: "ok"}))
    assert result.kind == outline.CHAPTERS
    assert [r.title for r in result.rows] == ["Chapter 1", "Chapter 2"]
    assert [r.clip_id for r in result.rows] == [one.id, two.id]
    assert [r.duration_s for r in result.rows] == [60.0, 90.0]
    assert result.rows[0].start_s < result.rows[1].start_s


@pytest.mark.parametrize("state, clip_status, started, expected", [
    ("missing", "approved", True, outline.MISSING),
    ("ok", "todo", None, outline.DONE),
    ("ok", "generated", None, outline.DONE),
    ("ok", "approved", None, outline.PROOFED),
    ("stale", "approved", True, outline.IN_PROGRESS),
    ("stale", "todo", True, outline.IN_PROGRESS),
    ("stale", "todo", False, outline.NOT_STARTED),
    # A closed child can't say: stale reads "in progress".
    ("stale", "todo", None, outline.IN_PROGRESS),
])
def test_derived_status(state, clip_status, started, expected):
    doc, (one, _two) = _book()
    one.status = clip_status
    result = outline.build(doc, _arrange(doc, {}), _states({one.id: state, _two.id: "ok"}),
                           started_of=lambda clip: started)
    assert result.rows[0].status == expected
    assert result.rows[0].state == state


def test_started_of_may_be_left_out():
    doc, (one, two) = _book()
    result = outline.build(doc, _arrange(doc, {}), _states({one.id: "stale", two.id: "stale"}))
    assert [r.status for r in result.rows] == [outline.IN_PROGRESS, outline.IN_PROGRESS]


def test_total_sums_the_chapters_and_flags_an_estimate():
    doc, (one, two) = _book()
    states = _states({one.id: "ok", two.id: "stale"})
    result = outline.build(doc, _arrange(doc, {one.id: 60.0}), states)
    assert [r.estimated for r in result.rows] == [False, True]
    assert result.rows[1].duration_s == 2.0
    assert result.total_s == 62.0
    assert result.estimated is True

    exact = outline.build(doc, _arrange(doc, {one.id: 60.0, two.id: 30.0}), states)
    assert exact.total_s == 90.0 and exact.estimated is False


def test_a_plain_clip_between_chapters_is_not_a_row_or_in_the_total():
    doc, (one, two) = _book()
    narrator = doc.characters[0]
    plain = doc.assign_character_to_range(0, 6, narrator.id)
    result = outline.build(doc, _arrange(doc, {one.id: 10.0, two.id: 20.0, plain.id: 5.0}),
                           _states({one.id: "ok", two.id: "ok"}))
    assert len(result.rows) == 2
    assert result.total_s == 30.0


def test_markers_stand_in_when_there_are_no_subprojects():
    doc = Document.from_plain_text("One. Two. Three.", characters=[Character.from_preset_dict("N", {})])
    clip = doc.assign_character_to_range(0, 16, doc.characters[0].id)
    doc.settings["markers"] = [
        {"id": "b", "seconds": 30.0, "name": "Second half", "note": ""},
        {"id": "a", "seconds": 0.0, "name": "Cold open", "note": ""},
    ]
    result = outline.build(doc, _arrange(doc, {clip.id: 100.0}), lambda clip: "ok")
    assert result.kind == outline.MARKERS
    assert [r.title for r in result.rows] == ["Cold open", "Second half"]
    assert [r.marker_id for r in result.rows] == ["a", "b"]
    assert [r.start_s for r in result.rows] == [0.0, 30.0]
    # To the next marker, and the last one to the end of the arrangement.
    assert [r.duration_s for r in result.rows] == [30.0, 70.0]
    assert all(r.status == "" and r.clip_id is None for r in result.rows)
    assert result.total_s == 100.0 and result.estimated is False


def test_a_marker_over_an_estimated_clip_is_estimated():
    doc = Document.from_plain_text("Some text here.", characters=[Character.from_preset_dict("N", {})])
    doc.assign_character_to_range(0, 15, doc.characters[0].id)
    doc.settings["markers"] = [{"id": "a", "seconds": 0.0, "name": "Start", "note": ""}]
    result = outline.build(doc, _arrange(doc, {}), lambda clip: "ok")
    assert result.rows[0].estimated is True and result.estimated is True


def test_chapters_win_over_markers():
    doc, (one, two) = _book()
    doc.settings["markers"] = [{"id": "a", "seconds": 0.0, "name": "M", "note": ""}]
    result = outline.build(doc, _arrange(doc, {}), _states({one.id: "ok", two.id: "ok"}))
    assert result.kind == outline.CHAPTERS


def test_nothing_to_list_is_an_empty_outline():
    result = outline.build(Document(), compute_arrangement(Document(), chars_per_second=15.0), lambda clip: "ok")
    assert result.kind == outline.EMPTY and result.rows == [] and result.total_s == 0.0


def test_format_hms():
    assert outline.format_hms(0) == "0:00:00"
    assert outline.format_hms(3725.4) == "1:02:05"
    assert outline.format_hms(59.6) == "0:01:00"
    assert outline.format_hms(125, estimated=True) == "~0:02:05"
    assert outline.format_hms(None) == "0:00:00"
