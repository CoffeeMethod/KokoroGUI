"""Duration formatting shared by the engine's ETA calculation
(`conversion.py`'s `on_chunk_progress`) and the Qt status bar
(`kokoro_gui/qt/app.py`'s `on_engine_progress`).

Both call sites used to format with `time.strftime('%M:%S', time.gmtime(seconds))`.
`time.gmtime` turns a seconds count into a full `struct_time` (days/hours/
minutes/seconds), but `%M` only ever prints `minutes % 60` - once `seconds`
passed 3600 the hours field kept climbing invisibly while the displayed
minutes wrapped back through 00, which read as the elapsed/ETA clock
"resetting" every hour. `format_duration` grows into `H:MM:SS` instead of
wrapping.
"""
from __future__ import annotations


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as `MM:SS`, or `H:MM:SS` once it reaches
    an hour. Negative input is clamped to 0."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
