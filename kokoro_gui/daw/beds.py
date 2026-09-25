"""Music beds (phase 5 P2, grill Q30): an imported audio file placed as a
clip.

A bed is a `Clip` with `source == "imported"` and `original_audio_path`
under the project's `audio/imported/` (`Clip.is_bed`). Its one run is a
read-only placeholder holding the file name, and it stores no segments.
What it plays is derived here, never stored:

- `overrides["trim"] = [start_s, end_s]`: the part of the file it plays.
  Unset plays the whole file.
- `overrides["loop"] = True`: the trim range repeats to
  `overrides["loop_length_s"]` (the length the block's right edge was
  dragged to; unset, one pass).

`bed_segments(clip)` turns that into virtual `Segment`s, one per pass,
each a `range` into the file, so every reader of a clip's segments (the
transport schedule, the exporter, the duration and the waveform) plays a
bed the way it plays a sliced recording. `playable_segments(clip)` is the
one call those readers make: a bed's virtual segments, else the clip's own
segments with audio, in order.

The file's length comes from `soundfile.info` (a header read, no decode),
memoized by path and mtime. Qt-free.
"""
from __future__ import annotations

import os
from typing import Optional

from kokoro_gui.daw.models import Segment

# The shortest trim or loop a bed keeps, so an edge drag can't make it
# vanish.
MIN_BED_S = 0.05
# A loop stops after this many passes whatever its length says.
MAX_LOOP_PASSES = 10000

_FILE_SECONDS: dict = {}


def audio_file_seconds(path: Optional[str]) -> Optional[float]:
    """The length of the audio file at `path` in seconds, or None when it
    is missing or unreadable."""
    if not path:
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    key = (os.path.abspath(path), mtime)
    if key in _FILE_SECONDS:
        return _FILE_SECONDS[key]
    try:
        import soundfile as sf

        info = sf.info(path)
        seconds = float(info.frames) / float(info.samplerate) if info.samplerate else None
    except Exception:
        seconds = None
    _FILE_SECONDS[key] = seconds
    return seconds


def _float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bed_trim(clip, file_s: float) -> tuple:
    """`(start_s, end_s)` into the file: `overrides["trim"]` clamped to
    the file and at least `MIN_BED_S` long, else the whole file."""
    file_s = max(0.0, float(file_s))
    raw = (clip.overrides or {}).get("trim")
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        start, end = _float(raw[0]), _float(raw[1])
        if start is not None and end is not None:
            start = max(0.0, min(start, file_s))
            end = max(0.0, min(end, file_s))
            if end - start >= MIN_BED_S or file_s < MIN_BED_S:
                return start, max(start, end)
    return 0.0, file_s


def bed_loops(clip) -> bool:
    return bool((clip.overrides or {}).get("loop", False))


def bed_length_s(clip, file_s: float) -> float:
    """The seconds the bed plays: its loop length when it loops (one pass
    when none is set), else its trim range."""
    start, end = bed_trim(clip, file_s)
    one_pass = end - start
    if bed_loops(clip):
        length = _float((clip.overrides or {}).get("loop_length_s"))
        if length is not None and length > 0:
            return max(MIN_BED_S, length)
    return one_pass


def bed_segments(clip) -> list:
    """The virtual segments a bed plays, one per pass of its trim range;
    [] when its file can't be read."""
    path = clip.original_audio_path
    file_s = audio_file_seconds(path)
    if file_s is None or file_s <= 0:
        return []
    start, end = bed_trim(clip, file_s)
    one_pass = end - start
    if one_pass <= 0:
        return []
    remaining = bed_length_s(clip, file_s)
    segments = []
    while remaining > 1e-6 and len(segments) < MAX_LOOP_PASSES:
        take = min(one_pass, remaining)
        segments.append(Segment(order_index=len(segments), audio_path=path, range=[start, start + take],
                                duration=take, onset_s=0.0, tail_s=0.0))
        remaining -= take
    return segments


def playable_segments(clip) -> list:
    """The segments a clip's audio is read from, in order: a bed's virtual
    ones, else its own segments that have a file."""
    if getattr(clip, "is_bed", False):
        return bed_segments(clip)
    return sorted((s for s in clip.segments if s.audio_path), key=lambda s: s.order_index)
