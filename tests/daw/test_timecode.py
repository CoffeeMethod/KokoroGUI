"""Tests for kokoro_gui/daw/timecode.py (phase 2, A4)."""
import pytest

from kokoro_gui.daw.timecode import (
    format_position, frames_to_tc, seconds_to_tc, tc_to_frames, tc_to_seconds, timecode_settings,
)


def test_non_drop_frame_round_trip_at_25():
    assert seconds_to_tc(0.0, 25.0) == "00:00:00:00"
    assert seconds_to_tc(61.52, 25.0) == "00:01:01:13"
    assert tc_to_seconds("00:01:01:13", 25.0) == pytest.approx(61.52)


def test_start_offset_shifts_both_ways():
    assert seconds_to_tc(1.0, 25.0, start="01:00:00:00") == "01:00:01:00"
    assert tc_to_seconds("01:00:01:00", 25.0, start="01:00:00:00") == pytest.approx(1.0)


def test_drop_frame_skips_two_numbers_each_minute_but_not_the_tenth():
    assert frames_to_tc(1799, 29.97, drop=True) == "00:00:59;29"
    assert frames_to_tc(1800, 29.97, drop=True) == "00:01:00;02"
    assert frames_to_tc(17982, 29.97, drop=True) == "00:10:00;00"
    for frames in (0, 1799, 1800, 5000, 17981, 17982, 107892):
        assert tc_to_frames(frames_to_tc(frames, 29.97, drop=True), 29.97, drop=True) == frames


def test_drop_frame_hour_is_real_time():
    # 29.97 drop-frame stays within a frame of the wall clock over an hour.
    assert seconds_to_tc(3600.0, 29.97, drop=True) == "01:00:00;00"


def test_bad_timecode_raises():
    with pytest.raises(ValueError):
        tc_to_frames("1:2", 25.0)


def test_format_position_only_when_enabled():
    assert format_position({}, 1.0) is None
    settings = {"timecode": {"enabled": True, "frame_rate": 30.0}}
    assert format_position(settings, 1.5) == "00:00:01:15"
    assert timecode_settings(settings)["start"] == "00:00:00:00"
