"""Where every clip sits on the seconds axis (UI9 of
Claude/PLAN_ui_shell_redesign.md).

`compute_arrangement(document)` is the single source of truth the timeline
ruler, the clip renderer, the transport's playback schedule and the exporter
all read from. Qt-free, like the rest of `kokoro_gui/daw/`.

Placement rule: walk clips in text order (`Document.clip_extent` start).
A clip with generated audio is as long as its segments say (or as long as
the caller's `clip_duration` measures them - the app passes one that
renders each segment through `kokoro_gui.audio.post`, since trim and pitch
change the length the raw file has); one without is
estimated from its text length at the engine's recorded chars/sec (UI11:
learned from `generation_stats.json`, 15 chars/s when there's no history),
divided by the clip's effective speed. A clip starts at its own
`timeline_timestamp` when the user has dragged it, else a gap after the
previous clip in text order ended - across every track, so the default is
one continuous read-through no matter how many lanes there are.

The gap is the clip's own `gap_before_s` when set (a `[pause:x]` marker
sets it), else `document.settings["paragraph_gap_s"]` when the text between
the two clips holds a blank line, else `document.settings["gap_s"]`. The
first clip gets only its own override. A pinned clip ignores gaps.

`overlaps` lists clips on the same track whose spans intersect (the
timeline paints them red). `plan_ripple` is ripple on regenerate: when a
regenerated clip comes back longer or shorter, every clip placed by
timestamp after its old end moves by the difference, except a clip with
`Clip.pinned` set. Clips placed in text order follow on their own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

FALLBACK_CHARS_PER_SECOND = 15.0
# Two clips closer than this aren't reported as overlapping, and a duration
# change smaller than this doesn't ripple.
OVERLAP_EPSILON_S = 1e-3
DEFAULT_GAP_S = 0.35
DEFAULT_PARAGRAPH_GAP_S = 0.9
# A blank line, whitespace-only lines included.
_PARAGRAPH_BREAK = re.compile(r"\n[ \t\r\f\v]*\n")


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


def _setting_s(document, key: str, default: float) -> float:
    try:
        return max(0.0, float((document.settings or {}).get(key, default)))
    except (TypeError, ValueError):
        return default


def boundary_gap_s(document, text: str, previous_end: Optional[int], clip, extent_start: int) -> float:
    """The silence placed before `clip`: its `gap_before_s` override, else
    the paragraph or clip gap depending on the text between the previous
    clip's extent end and this one's start. `previous_end=None` means this
    is the first clip, which has no default gap."""
    if clip.gap_before_s is not None:
        try:
            return max(0.0, float(clip.gap_before_s))
        except (TypeError, ValueError):
            pass
    if previous_end is None:
        return 0.0
    between = text[previous_end:extent_start]
    if _PARAGRAPH_BREAK.search(between):
        return _setting_s(document, "paragraph_gap_s", DEFAULT_PARAGRAPH_GAP_S)
    return _setting_s(document, "gap_s", DEFAULT_GAP_S)


def compute_arrangement(document, engine_id: Optional[str] = None,
                        chars_per_second: Optional[float] = None,
                        clip_duration: Optional[Callable] = None,
                        clip_estimate: Optional[Callable] = None) -> Arrangement:
    """Pass `chars_per_second` to bypass the stats lookup (tests, or a
    caller that already has the number). `clip_duration(clip)` replaces
    `clip_audio_duration_s` when given: it returns the clip's audible
    length in seconds, or None for a clip with no audio yet.
    `clip_estimate(clip)` may give a better estimate than the clip's own
    text for a clip with no audio (a subproject: its content, not its
    title), or None to use the text."""
    if chars_per_second is None:
        chars_per_second = recorded_chars_per_second(engine_id)
    if clip_duration is None:
        clip_duration = clip_audio_duration_s

    with_extent = []
    for clip in document.clips:
        extent = document.clip_extent(clip.id)
        if extent is None:
            continue
        with_extent.append((extent[0], extent[1], clip))
    with_extent.sort(key=lambda item: item[0])

    text = document.text
    placed = []
    cursor = 0.0
    previous_end = None
    for extent_start, extent_end, clip in with_extent:
        audio_duration = clip_duration(clip)
        if audio_duration is not None:
            duration = audio_duration
            estimated = False
        else:
            better = clip_estimate(clip) if clip_estimate is not None else None
            if better is not None:
                duration = float(better)
            else:
                config = document.effective_config_for_clip(clip)
                duration = estimate_duration_s(document.clip_text(clip), config.get("speed", 1.0), chars_per_second)
            estimated = True
        if clip.timeline_timestamp is not None:
            start = clip.timeline_timestamp
        else:
            start = cursor + boundary_gap_s(document, text, previous_end, clip, extent_start)
        start = max(0.0, float(start))
        placed.append(PlacedClip(clip=clip, start_s=start, duration_s=duration, estimated=estimated))
        cursor = start + duration
        previous_end = extent_end

    total = max((p.end_s for p in placed), default=0.0)
    return Arrangement(placed=placed, total_duration_s=total)


def segment_timeline(placed: PlacedClip) -> list:
    """`[(segment, start_s, scale)]` for a placed clip's audio segments in
    order: where each segment starts on the timeline and the factor from its
    raw seconds to placed seconds. The placed length is the rendered one
    (trim and pitch change it), so raw times map linearly onto it. Word
    times go through this: `start_s + word_start * scale`."""
    segments = sorted((s for s in placed.clip.segments if s.audio_path), key=lambda s: s.order_index)
    raw_total = sum(float(s.duration or 0.0) for s in segments)
    scale = placed.duration_s / raw_total if raw_total > 0 else 1.0
    out = []
    cursor = placed.start_s
    for segment in segments:
        out.append((segment, cursor, scale))
        cursor += float(segment.duration or 0.0) * scale
    return out


def text_order_predecessor(arrangement: Arrangement, clip_id: str):
    """The placed clip immediately before `clip_id` in text order, or None."""
    for index, placed in enumerate(arrangement.placed):
        if placed.clip.id == clip_id:
            return arrangement.placed[index - 1] if index > 0 else None
    return None


def overlaps(arrangement: Arrangement) -> list:
    """`[(clip_id, clip_id)]` for every pair of placed clips on the same
    track whose spans intersect by more than `OVERLAP_EPSILON_S`, earlier
    clip first. Clips with no track are left out."""
    by_track: dict = {}
    for placed in arrangement.placed:
        track_id = getattr(placed.clip, "track_id", None)
        if track_id is not None:
            by_track.setdefault(track_id, []).append(placed)
    pairs = []
    for placed_list in by_track.values():
        placed_list.sort(key=lambda p: (p.start_s, p.end_s))
        for i, a in enumerate(placed_list):
            for b in placed_list[i + 1:]:
                if b.start_s >= a.end_s - OVERLAP_EPSILON_S:
                    break
                if b.end_s > a.start_s + OVERLAP_EPSILON_S and b.duration_s > OVERLAP_EPSILON_S:
                    pairs.append((a.clip.id, b.clip.id))
    return pairs


def plan_ripple(arrangement: Arrangement, deltas: dict) -> dict:
    """Ripple on regenerate. `deltas` is `{clip_id: new duration - old
    duration}` for regenerated clips, measured against `arrangement` (the
    placement before the new audio). Returns `{clip_id: shift_s}` for every
    clip placed by `timeline_timestamp`, not `pinned`, that starts at or
    after a regenerated clip's old end: the sum of those clips' deltas.
    Several regenerated clips compose, since each shift is measured in the
    same before-coordinates."""
    old_end = {}
    for placed in arrangement.placed:
        if placed.clip.id in deltas:
            old_end[placed.clip.id] = placed.end_s
    shifts = {}
    for placed in arrangement.placed:
        clip = placed.clip
        if clip.timeline_timestamp is None or getattr(clip, "pinned", False):
            continue
        shift = sum(delta for clip_id, delta in deltas.items()
                    if clip_id != clip.id and clip_id in old_end
                    and placed.start_s >= old_end[clip_id] - OVERLAP_EPSILON_S)
        if abs(shift) > OVERLAP_EPSILON_S:
            shifts[clip.id] = shift
    return shifts
