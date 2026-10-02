"""Tests for kokoro_gui/daw/reference.py: the source track setting, a clip's
reference range and the per-clip slices the transport's alt schedule plays
(phase 5 D5)."""
import pytest

from kokoro_gui.daw.arrangement import Arrangement, PlacedClip
from kokoro_gui.daw.models import Clip
from kokoro_gui.daw.reference import reference_range, reference_slices, source_track_settings


def test_source_track_settings_normalises_or_rejects():
    assert source_track_settings({}) is None
    assert source_track_settings({"source_track": {"path": ""}}) is None
    assert source_track_settings({"source_track": "audio/x.wav"}) is None
    assert source_track_settings({"source_track": {"path": "audio\\imported\\a.wav", "offset_s": "bad"}}) == \
        {"path": "audio/imported/a.wav", "offset_s": 0.0}
    assert source_track_settings({"source_track": {"path": "a.wav", "offset_s": 1.5}})["offset_s"] == 1.5


def test_reference_range_reads_the_override_and_skips_bad_values():
    assert reference_range(Clip(overrides={"reference_range": [1.0, 2.5]})) == (1.0, 2.5)
    for bad in (None, [2.0, 1.0], [1.0], ["a", 2.0], [-1.0, 2.0], [1.0, 1.0]):
        assert reference_range(Clip(overrides={"reference_range": bad})) is None
    # A pinned clip with a target duration but no range has none: never derived.
    pinned = Clip(timeline_timestamp=3.0, pinned=True, overrides={"target_duration_s": 1.0})
    assert reference_range(pinned) is None


def test_reference_slices_play_under_the_clips_anchor_shifted_by_the_offset():
    a = Clip(overrides={"reference_range": [1.0, 2.0]})
    b = Clip()
    c = Clip(overrides={"reference_range": [0.2, 0.8]})
    arrangement = Arrangement(placed=[
        PlacedClip(a, start_s=0.9, duration_s=1.0, estimated=False, aligned_onset_s=0.1),
        PlacedClip(b, start_s=3.0, duration_s=1.0, estimated=False),
        PlacedClip(c, start_s=5.0, duration_s=1.0, estimated=True),
    ], total_duration_s=6.0)

    assert reference_slices(arrangement, 0.5) == [(a.id, 1.0, (1.5, 2.5)), (c.id, 5.0, pytest.approx((0.7, 1.3)))]
    # A negative offset cuts what falls before the file and starts later.
    slices = reference_slices(arrangement, -0.5)
    assert slices[0][0] == a.id and slices[0][2] == (0.5, 1.5)
    assert slices[1][0] == c.id and slices[1][1] == pytest.approx(5.3) and slices[1][2] == pytest.approx((0.0, 0.3))
    assert [s[0] for s in reference_slices(arrangement, -1.0)] == [a.id]
