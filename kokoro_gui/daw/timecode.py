"""SMPTE timecode for the ruler, the transport label and the cue sheet.

`Document.settings["timecode"]` is `{"frame_rate", "start", "drop_frame",
"enabled"}` (`DEFAULT_TIMECODE`). Seconds stay the unit everywhere
internally; this module only formats and parses. Drop-frame (29.97 and
59.94) skips frame numbers 0 and 1 (0-3 at 59.94) at the start of every
minute except each tenth, and separates frames with ";".
"""
from __future__ import annotations

import math
import re

DEFAULT_TIMECODE = {"frame_rate": 25.0, "start": "00:00:00:00", "drop_frame": False, "enabled": False}
FRAME_RATES = (23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0)

_TC = re.compile(r"^\s*(\d{1,2})[:;.](\d{1,2})[:;.](\d{1,2})[:;.](\d{1,3})\s*$")


def timecode_settings(settings: dict | None) -> dict:
    """`DEFAULT_TIMECODE` with whatever the document stores laid over it."""
    block = (settings or {}).get("timecode")
    merged = dict(DEFAULT_TIMECODE)
    if isinstance(block, dict):
        merged.update({k: block[k] for k in DEFAULT_TIMECODE if k in block})
    return merged


def _nominal(fps: float) -> int:
    return int(round(float(fps)))


def _drop_count(fps: float) -> int:
    """Frame numbers dropped per minute: 2 at 29.97, 4 at 59.94."""
    return int(round(float(fps) * 0.066666))


def frames_to_tc(frames: int, fps: float, drop: bool = False) -> str:
    nominal = _nominal(fps)
    frames = max(0, int(frames))
    if drop:
        dropped = _drop_count(fps)
        per_10min = int(round(float(fps) * 600))
        per_min = nominal * 60 - dropped
        tens, rest = divmod(frames, per_10min)
        frames += dropped * 9 * tens
        if rest > dropped:
            frames += dropped * ((rest - dropped) // per_min)
    ff = frames % nominal
    ss = (frames // nominal) % 60
    mm = (frames // (nominal * 60)) % 60
    hh = (frames // (nominal * 3600)) % 24
    sep = ";" if drop else ":"
    return f"{hh:02d}:{mm:02d}:{ss:02d}{sep}{ff:02d}"


def tc_to_frames(tc: str, fps: float, drop: bool = False) -> int:
    """Raises ValueError for a string that isn't `HH:MM:SS:FF`."""
    match = _TC.match(tc or "")
    if not match:
        raise ValueError(f"not a timecode: {tc!r}")
    hh, mm, ss, ff = (int(g) for g in match.groups())
    nominal = _nominal(fps)
    frames = (hh * 3600 + mm * 60 + ss) * nominal + ff
    if drop:
        minutes = hh * 60 + mm
        frames -= _drop_count(fps) * (minutes - minutes // 10)
    return frames


def seconds_to_tc(seconds: float, fps: float, start: str = "00:00:00:00", drop: bool = False) -> str:
    offset = tc_to_frames(start, fps, drop) if start else 0
    # The frame that contains `seconds` (floor, with float slack).
    frame = int(math.floor(max(0.0, seconds) * float(fps) + 1e-6))
    return frames_to_tc(frame + offset, fps, drop)


def tc_to_seconds(tc: str, fps: float, start: str = "00:00:00:00", drop: bool = False) -> float:
    offset = tc_to_frames(start, fps, drop) if start else 0
    return max(0, tc_to_frames(tc, fps, drop) - offset) / float(fps)


def format_position(settings: dict | None, seconds: float):
    """The timecode for `seconds` when the document has timecode enabled,
    else None (the caller shows its seconds format)."""
    tc = timecode_settings(settings)
    if not tc.get("enabled"):
        return None
    try:
        return seconds_to_tc(seconds, float(tc["frame_rate"]), str(tc["start"]), bool(tc["drop_frame"]))
    except (TypeError, ValueError):
        return seconds_to_tc(seconds, float(tc["frame_rate"]), "00:00:00:00", bool(tc["drop_frame"]))
