"""Tests for kokoro_gui/qt/playhead_calc.py's wall-clock playhead position
calculator - pure Python, no Qt, no real-time waiting."""
import pytest

from kokoro_gui.qt.playhead_calc import playhead_x


def test_playhead_at_start():
    assert playhead_x(0.0, duration_seconds=10.0, view_width=100.0) == 0.0


def test_playhead_at_midpoint():
    assert playhead_x(5.0, duration_seconds=10.0, view_width=100.0) == pytest.approx(50.0)


def test_playhead_clamped_at_end_when_elapsed_exceeds_duration():
    assert playhead_x(15.0, duration_seconds=10.0, view_width=100.0) == 100.0


def test_playhead_at_exactly_duration_is_clamped_at_end():
    assert playhead_x(10.0, duration_seconds=10.0, view_width=100.0) == 100.0


def test_playhead_none_for_zero_duration():
    assert playhead_x(1.0, duration_seconds=0.0, view_width=100.0) is None


def test_playhead_none_for_negative_duration():
    assert playhead_x(1.0, duration_seconds=-5.0, view_width=100.0) is None


def test_playhead_clamped_at_zero_for_negative_elapsed():
    assert playhead_x(-5.0, duration_seconds=10.0, view_width=100.0) == 0.0


def test_playhead_scales_with_view_width():
    a = playhead_x(2.5, duration_seconds=10.0, view_width=100.0)
    b = playhead_x(2.5, duration_seconds=10.0, view_width=200.0)
    assert b == pytest.approx(a * 2)
