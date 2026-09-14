"""Per-engine generation-history tracking that seeds and refines the batch
conversion ETA in `conversion.py`.

Persisted to `kokoro_engine.STATS_FILE`, read/written qualified through the
`kokoro_engine` module rather than imported as a bare constant - the same
convention `caching.py` uses for `kokoro_engine.CACHE_DIR` - so
`tests/conftest.py`'s `isolated_dirs` fixture can monkeypatch it into a
tmp_path and keep tests from writing a real `generation_stats.json` into the
repo working directory.

Keyed by `engine_id` (Kokoro/Dummy/Audio8 have wildly different chars/sec
throughput - Audio8 in particular serializes every chunk through one shared
model lock, per `kokoro_gui/engines/audio8_tts.py`) so switching backends
never blends one engine's speed into another's estimate. Each engine keeps a
bounded rolling window (`HISTORY_LIMIT`) of its most recent generations
rather than a lifetime average, so the estimate tracks changes in
hardware/settings/thread count instead of being anchored by a stale run.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Optional

import kokoro_engine

HISTORY_LIMIT = 20  # most-recent completed generations kept, per engine

_lock = threading.Lock()


def _load_all() -> dict:
    path = kokoro_engine.STATS_FILE
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_all(data: dict) -> None:
    try:
        with open(kokoro_engine.STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Failed to save generation stats: {e}")


def record_generation(engine_id: str, chars: int, words: int, duration: float) -> None:
    """Append one generation's (chars, words, duration) under `engine_id`,
    trimming to the most recent `HISTORY_LIMIT` entries. No-ops on
    degenerate input (nothing processed, or non-positive duration) so an
    instant cancel/failure can't poison the rate estimate."""
    if chars <= 0 or duration <= 0:
        return
    engine_id = engine_id or "unknown"
    with _lock:
        data = _load_all()
        entries = data.get(engine_id, [])
        entries.append({"chars": chars, "words": words, "duration": duration})
        data[engine_id] = entries[-HISTORY_LIMIT:]
        _save_all(data)


def estimate_chars_per_sec(engine_id: str) -> Optional[float]:
    """Recent chars/sec throughput for `engine_id`, or None with no history
    yet. Sums chars and duration across the retained window and divides once
    (rather than averaging each entry's own rate), so a handful of short
    generations can't outvote one long, more representative one."""
    engine_id = engine_id or "unknown"
    with _lock:
        entries = _load_all().get(engine_id, [])
    total_chars = sum(e.get("chars", 0) for e in entries)
    total_duration = sum(e.get("duration", 0) for e in entries)
    if total_chars <= 0 or total_duration <= 0:
        return None
    return total_chars / total_duration
