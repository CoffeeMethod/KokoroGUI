"""Tests for kokoro_gui/daw/markers.py (phase 2, A3)."""
from kokoro_gui.daw import markers
from kokoro_gui.daw.models import Document
from kokoro_gui.daw.undo import SetFieldCommand


def test_add_keeps_the_list_sorted_and_names_defaults():
    settings = {}
    first, a = markers.add_marker(settings, 5.0)
    settings["markers"] = first
    second, b = markers.add_marker(settings, 1.0, name="Intro")
    assert [m["seconds"] for m in second] == [1.0, 5.0]
    assert a["name"] == "M1" and b["name"] == "Intro"
    assert settings["markers"] == first  # the input list isn't touched


def test_move_rename_delete_and_range():
    settings = {}
    settings["markers"], a = markers.add_marker(settings, 1.0)
    settings["markers"], b = markers.add_marker(settings, 3.0)
    settings["markers"] = markers.move_marker(settings, a["id"], 4.0)
    assert [m["id"] for m in markers.list_markers(settings)] == [b["id"], a["id"]]
    settings["markers"] = markers.rename_marker(settings, a["id"], "Chorus", note="check the breath")
    assert markers.get_marker(settings, a["id"])["note"] == "check the breath"
    assert markers.range_between(settings, a["id"], b["id"]) == (3.0, 4.0)
    settings["markers"] = markers.delete_marker(settings, b["id"])
    assert markers.range_between(settings, a["id"], b["id"]) is None


def test_garbage_entries_are_dropped():
    assert markers.list_markers({"markers": [{"seconds": "x"}, 3, {"seconds": 2}]})[0]["seconds"] == 2.0
    assert markers.list_markers({"markers": "nope"}) == []


def test_marker_edit_is_undoable_through_set_field_command():
    doc = Document()
    new_list, _m = markers.add_marker(doc.settings, 2.0)
    doc.undo_stack.push(SetFieldCommand("document", None, "settings", new_list, key="markers"))
    assert len(markers.list_markers(doc.settings)) == 1
    doc.undo_stack.undo()
    assert markers.list_markers(doc.settings) == []
