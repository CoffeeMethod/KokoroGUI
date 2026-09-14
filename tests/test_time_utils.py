"""format_duration (kokoro_gui/engine/time_utils.py) - the fix for the
elapsed/ETA display "resetting" past the one-hour mark. The old
`time.strftime('%M:%S', time.gmtime(seconds))` never carried minutes into an
hours field, so e.g. 3661s ("1:01:01") printed as "01:01", indistinguishable
from 61s."""
import pytest

from kokoro_gui.engine.time_utils import format_duration


@pytest.mark.parametrize("seconds,expected", [
    (0, "00:00"),
    (5, "00:05"),
    (59, "00:59"),
    (65, "01:05"),
    (3599, "59:59"),
    (3600, "1:00:00"),
    (3661, "1:01:01"),
    (7325, "2:02:05"),
    (86400, "24:00:00"),
])
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


def test_format_duration_does_not_reset_past_one_hour():
    # This is the actual reported bug: minutes:seconds must keep climbing
    # (via a growing hours field) rather than wrapping back toward 00:00.
    just_under_hour = format_duration(3599)
    just_over_hour = format_duration(3601)
    assert just_under_hour == "59:59"
    assert just_over_hour == "1:00:01"


def test_format_duration_clamps_negative_to_zero():
    assert format_duration(-42) == "00:00"


def test_format_duration_accepts_float_seconds():
    assert format_duration(90.9) == "01:30"
