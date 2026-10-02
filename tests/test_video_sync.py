"""kokoro_gui/qt/video_sync.py: how the reference video follows the
transport (phase 5, TB16). Pure functions, no Qt and no QtMultimedia, so
this runs in the non-GUI suite."""
from kokoro_gui.qt import video_sync


def test_target_is_transport_time_plus_offset_in_ms():
    assert video_sync.target_ms(2.0, 0.0) == 2000
    assert video_sync.target_ms(2.0, 1.25) == 3250
    assert video_sync.target_ms(2.0, -0.5) == 1500


def test_target_never_goes_before_the_start_of_the_video():
    assert video_sync.target_ms(0.2, -1.0) == 0


def test_a_player_within_the_drift_threshold_is_left_alone():
    assert video_sync.seek_target_ms(10.0, 0.0, 10_000) is None
    assert video_sync.seek_target_ms(10.0, 0.0, 10_040) is None
    assert video_sync.seek_target_ms(10.0, 0.0, 9_960) is None


def test_a_player_past_the_drift_threshold_is_seeked_to_the_target():
    assert video_sync.seek_target_ms(10.0, 0.0, 10_041) == 10_000
    assert video_sync.seek_target_ms(10.0, 0.0, 9_959) == 10_000
    assert video_sync.seek_target_ms(10.0, 2.0, 0) == 12_000
    assert video_sync.DRIFT_MS == 40


def test_a_custom_threshold_applies():
    assert video_sync.seek_target_ms(1.0, 0.0, 1_100, threshold_ms=200) is None
    assert video_sync.seek_target_ms(1.0, 0.0, 1_300, threshold_ms=200) == 1_000


def test_the_player_plays_only_while_the_transport_plays():
    assert video_sync.player_command("playing") == "play"
    assert video_sync.player_command("paused") == "pause"
    assert video_sync.player_command("stopped") == "pause"
