"""Tests for kokoro_gui/qt/selection.py's SelectionModel - pure state-machine
tests, no qt_app fixture needed (lives under tests/gui_qt/ to inherit that
directory's conftest.py's PySide6 import-guard/offscreen-platform setup)."""
import pytest

from kokoro_gui.qt.selection import SelectionModel


def _count_calls(model):
    calls = []
    model.changed.connect(lambda: calls.append(True))
    return calls


def test_initial_state_is_none():
    model = SelectionModel()
    assert model.kind == "none"
    assert model.selected_clip_id is None
    assert model.selected_character_id is None
    assert model.selected_range is None


def test_select_clip_sets_kind_and_clears_others():
    model = SelectionModel()
    model.select_character("char-1")
    model.select_clip("clip-1")
    assert model.kind == "clip"
    assert model.selected_clip_id == "clip-1"
    assert model.selected_character_id is None
    assert model.selected_range is None


def test_select_character_sets_kind_and_clears_others():
    model = SelectionModel()
    model.select_clip("clip-1")
    model.select_character("char-1")
    assert model.kind == "character"
    assert model.selected_character_id == "char-1"
    assert model.selected_clip_id is None
    assert model.selected_range is None


def test_select_range_sets_kind_and_clears_others():
    model = SelectionModel()
    model.select_clip("clip-1")
    model.select_range(2, 8)
    assert model.kind == "range"
    assert model.selected_range == (2, 8)
    assert model.selected_clip_id is None
    assert model.selected_character_id is None


def test_clear_resets_to_none():
    model = SelectionModel()
    model.select_clip("clip-1")
    model.clear()
    assert model.kind == "none"
    assert model.selected_clip_id is None


def test_changed_emits_once_per_real_state_change():
    model = SelectionModel()
    calls = _count_calls(model)

    model.select_clip("clip-1")
    assert len(calls) == 1

    model.select_character("char-1")
    assert len(calls) == 2

    model.select_range(0, 3)
    assert len(calls) == 3

    model.clear()
    assert len(calls) == 4


def test_repeated_mutator_call_with_same_value_does_not_reemit():
    model = SelectionModel()
    model.select_clip("clip-1")
    calls = _count_calls(model)

    model.select_clip("clip-1")

    assert calls == []
    assert model.selected_clip_id == "clip-1"


def test_repeated_clear_does_not_reemit():
    model = SelectionModel()
    calls = _count_calls(model)

    model.clear()

    assert calls == []


def test_repeated_select_range_with_same_bounds_does_not_reemit():
    model = SelectionModel()
    model.select_range(5, 10)
    calls = _count_calls(model)

    model.select_range(5, 10)

    assert calls == []


def test_select_range_with_equal_bounds_raises_value_error():
    model = SelectionModel()
    with pytest.raises(ValueError):
        model.select_range(5, 5)


def test_select_range_with_end_before_start_raises_value_error():
    model = SelectionModel()
    with pytest.raises(ValueError):
        model.select_range(5, 2)
