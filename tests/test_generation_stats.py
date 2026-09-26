"""kokoro_gui/engine/stats.py - the per-engine generation-history store that
seeds/refines the batch conversion ETA (kokoro_gui/engine/conversion.py).

Uses the `isolated_dirs` fixture (tests/conftest.py) purely for its
monkeypatched `runtime.STATS_FILE`, so these tests never touch a real
generation_stats.json in the repo working directory.
"""
import json

from kokoro_gui.engine import runtime
from kokoro_gui.engine import stats as generation_stats


def test_estimate_chars_per_sec_with_no_history_returns_none(isolated_dirs):
    assert generation_stats.estimate_chars_per_sec("kokoro") is None


def test_record_and_estimate_round_trip(isolated_dirs):
    generation_stats.record_generation("kokoro", chars=1000, words=180, duration=10.0)
    assert generation_stats.estimate_chars_per_sec("kokoro") == 100.0


def test_estimate_sums_across_history_rather_than_averaging_per_run_rates(isolated_dirs):
    # One long, slow run and one short, fast run: summing chars/summing
    # duration should weight the long run more heavily than a naive average
    # of each run's own rate would.
    generation_stats.record_generation("kokoro", chars=9000, words=1500, duration=90.0)  # 100 chars/s
    generation_stats.record_generation("kokoro", chars=100, words=20, duration=0.5)       # 200 chars/s
    rate = generation_stats.estimate_chars_per_sec("kokoro")
    total_chars, total_duration = 9100, 90.5
    assert rate == total_chars / total_duration
    assert rate < 150.0  # nowhere near a plain average of the two per-run rates


def test_stats_are_isolated_per_engine(isolated_dirs):
    generation_stats.record_generation("kokoro", chars=1000, words=200, duration=10.0)
    assert generation_stats.estimate_chars_per_sec("dummy") is None
    assert generation_stats.estimate_chars_per_sec("audio8") is None

    generation_stats.record_generation("audio8", chars=100, words=20, duration=50.0)
    assert generation_stats.estimate_chars_per_sec("kokoro") == 100.0
    assert generation_stats.estimate_chars_per_sec("audio8") == 2.0


def test_record_generation_ignores_zero_chars(isolated_dirs):
    generation_stats.record_generation("kokoro", chars=0, words=0, duration=5.0)
    assert generation_stats.estimate_chars_per_sec("kokoro") is None


def test_record_generation_ignores_non_positive_duration(isolated_dirs):
    generation_stats.record_generation("kokoro", chars=500, words=90, duration=0.0)
    generation_stats.record_generation("kokoro", chars=500, words=90, duration=-1.0)
    assert generation_stats.estimate_chars_per_sec("kokoro") is None


def test_record_generation_trims_to_history_limit(isolated_dirs):
    for i in range(generation_stats.HISTORY_LIMIT + 5):
        generation_stats.record_generation("kokoro", chars=100, words=20, duration=1.0)

    with open(runtime.STATS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["kokoro"]) == generation_stats.HISTORY_LIMIT


def test_record_generation_missing_engine_id_falls_back_to_unknown_bucket(isolated_dirs):
    generation_stats.record_generation(None, chars=100, words=20, duration=2.0)
    assert generation_stats.estimate_chars_per_sec(None) == 50.0
    assert generation_stats.estimate_chars_per_sec("unknown") == 50.0
