"""kokoro_gui/daw/export_presets.py: each preset's checks pass on a report at
its targets and fail with a readable message when the mix is 3 dB off."""
import math

import pytest

from kokoro_gui.audio.loudness import LoudnessReport
from kokoro_gui.daw.export_presets import CUSTOM_ID, PRESETS, Check, get_preset, preset_ids


def _report(lufs=-16.0, peak_dbtp=-2.0, rms=-20.0, floor=-70.0, sample_peak=-4.0, minutes=30.0):
    return LoudnessReport(lufs, peak_dbtp, sample_peak, rms, floor, minutes * 60.0)


def _at_targets(preset):
    v = preset.values
    if v["normalize_mode"] == "rms":
        return _report(rms=v["target_rms_dbfs"], sample_peak=v["limiter_dbfs"])
    return _report(lufs=v["target_lufs"], peak_dbtp=v["ceiling_dbtp"])


def test_the_ids_are_unique_and_custom_comes_first():
    ids = preset_ids()
    assert ids[0] == CUSTOM_ID and len(set(ids)) == len(ids)
    assert get_preset("acx") is not None and get_preset(CUSTOM_ID) is None and get_preset("nope") is None


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.id)
def test_a_file_at_the_targets_passes_every_check(preset):
    assert preset.failed(_at_targets(preset)) == []


@pytest.mark.parametrize("preset", [p for p in PRESETS if p.id != "acx"], ids=lambda p: p.id)
def test_a_podcast_file_3_db_off_fails_by_name(preset):
    target = preset.values["target_lufs"]
    for off in (-3.0, 3.0):
        failed = preset.failed(_report(lufs=target + off, peak_dbtp=preset.values["ceiling_dbtp"]))
        assert len(failed) == 1 and failed[0].startswith("Loudness ") and "needs" in failed[0]
    assert preset.failed(_report(lufs=target, peak_dbtp=-0.5))[0].startswith("True peak ")


def test_acx_names_each_failed_requirement():
    acx = get_preset("acx")
    failed = acx.failed(_report(rms=-17.0, sample_peak=-0.5, floor=-52.0, minutes=130.0))
    assert [f.split(" ")[0] for f in failed] == ["RMS", "Peak", "Noise", "Length"]
    assert acx.failed(_report(rms=-23.5))[0] == "RMS -23.5 dBFS, needs -23 to -18 dBFS"
    assert acx.failed(_report(floor=-52.0)) == ["Noise floor -52.0 dBFS, needs at most -60 dBFS"]
    assert acx.failed(_report(minutes=121.0)) == ["Length 121.0 min, needs at most 120 min"]


def test_the_acx_limits_are_inclusive():
    acx = get_preset("acx")
    assert acx.failed(_report(rms=-23.0, sample_peak=-3.0, floor=-60.0, minutes=120.0)) == []


def test_silence_fails_a_lower_bound_and_passes_an_upper_one():
    silent = _report(lufs=-math.inf, peak_dbtp=-math.inf, rms=-math.inf, floor=-math.inf, sample_peak=-math.inf)
    failed = get_preset("acx").failed(silent)
    assert failed == ["RMS silent, needs -23 to -18 dBFS"]
    assert Check("x", "rms_dbfs", high=-3.0).passed(silent)


def test_apple_mono_sits_3_lu_under_stereo():
    assert get_preset("apple").values["target_lufs"] - get_preset("apple_mono").values["target_lufs"] == 3.0
    assert get_preset("apple_mono").values["channels"] == 1


def test_presets_set_every_field_so_switching_leaves_nothing_behind():
    keys = set(PRESETS[0].values)
    assert all(set(p.values) == keys for p in PRESETS)
