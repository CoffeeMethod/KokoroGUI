"""Where every clip sits on the seconds axis (UI9 of
Claude/PLAN_ui_shell_redesign.md).

`compute_arrangement(document)` is the single source of truth the timeline
ruler, the clip renderer, the transport's playback schedule and the exporter
all read from. Qt-free, like the rest of `kokoro_gui/daw/`.

Placement rule: walk clips in text order (`Document.clip_extent` start).
A clip with generated audio is as long as its segments say; one without is
estimated from its text length at the engine's recorded chars/sec (UI11:
learned from `generation_stats.json`, 15 chars/s when there's no history),
divided by the clip's effective speed. A clip starts at its own
`timeline_timestamp` when the user has dragged it, else right where the
previous clip in text order ended - across every track, so the default is
one continuous read-through no matter how many lanes there are.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

FALLBACK_CHARS_PER_SECOND = 15.0


@dataclass(frozen=True)
class PlacedClip:
    clip: object
    start_s: float
    duration_s: float
    estimated: bool

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


@dataclass(frozen=True)
class Arrangement:
    placed: list
    total_duration_s: float

    def by_clip_id(self) -> dict:
        return {p.clip.id: p for p in self.placed}

    def at_time(self, seconds: float) -> list:
        """Every placed clip whose span covers `seconds`."""
        return [p for p in self.placed if p.start_s <= seconds < p.end_s]


def recorded_chars_per_second(engine_id: Optional[str]) -> Optional[float]:
    """The engine's recorded throughput from `generation_stats.json`, or
    None. Imported lazily: stats.py imports `kokoro_engine`, which pulls in
    the kokoro package - too heavy to make a hard dependency of this
    otherwise-pure module (and its tests)."""
    try:
        from kokoro_gui.engine.stats import estimate_chars_per_sec
    except Exception:
        return None
    try:
        return estimate_chars_per_sec(engine_id or "kokoro")
    except Exception:
        return None


def estimate_duration_s(text: str, speed: float, chars_per_second: Optional[float]) -> float:
    rate = chars_per_second if chars_per_second and chars_per_second > 0 else FALLBACK_CHARS_PER_SECOND
    speed = speed if speed and speed > 0 else 1.0
    chars = len(text.strip())
    if chars == 0:
        return 0.0
    return chars / (rate * speed)


def clip_audio_duration_s(clip) -> Optional[float]:
    """Sum of the clip's segment durations, or None when no segment carries
    audio yet."""
    if not any(getattr(s, "audio_path", None) for s in clip.segments):
        return None
    return float(sum(s.duration or 0.0 for s in clip.segments))


def compute_arrangement(document, engine_id: Optional[str] = None,
                        chars_per_second: Optional[float] = None) -> Arrangement:
    """Pass `chars_per_second` to bypass the stats lookup (tests, or a
    caller that already has the number)."""
    if chars_per_second is None:
        chars_per_second = recorded_chars_per_second(engine_id)

    with_extent = []
    for clip in document.clips:
        extent = document.clip_extent(clip.id)
        if extent is None:
            continue
        with_extent.append((extent[0], clip))
    with_extent.sort(key=lambda pair: pair[0])

    placed = []
    cursor = 0.0
    for _start_offset, clip in with_extent:
        audio_duration = clip_audio_duration_s(clip)
        if audio_duration is not None:
            duration = audio_duration
            estimated = False
        else:
            config = document.effective_config_for_clip(clip)
            duration = estimate_duration_s(document.clip_text(clip), config.get("speed", 1.0), chars_per_second)
            estimated = True
        start = clip.timeline_timestamp if clip.timeline_timestamp is not None else cursor
        start = max(0.0, float(start))
        placed.append(PlacedClip(clip=clip, start_s=start, duration_s=duration, estimated=estimated))
        cursor = start + duration

    total = max((p.end_s for p in placed), default=0.0)
    return Arrangement(placed=placed, total_duration_s=total)


def text_order_predecessor(arrangement: Arrangement, clip_id: str):
    """The placed clip immediately before `clip_id` in text order, or None."""
    for index, placed in enumerate(arrangement.placed):
        if placed.clip.id == clip_id:
            return arrangement.placed[index - 1] if index > 0 else None
    return None
