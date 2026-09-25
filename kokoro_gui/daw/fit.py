"""Duration targets and fit to slot (phase 5, D4). Qt-free.

A clip's slot is `Clip.overrides["target_duration_s"]` seconds from its
timeline timestamp (a subtitle cue's in-time; D2's import sets both). Its
fit ratio is the rendered duration over the target: 1.0 fills the slot
exactly, above 1.0 runs past it.

Fit to slot (`TimelineDock`) changes the clip until the ratio is within
`FIT_TOLERANCE` of 1.0. An engine with a speed control (Kokoro) regenerates
at `next_speed`, at most `MAX_FIT_PASSES` times; one without (Audio8) gets a
read-time `overrides["time_stretch"]` from `stretch_for`, which stops at
`STRETCH_MAX` and asks for a rewrite past it.

`reading_rate_ratio` is the same comparison made from the text alone, for
the transcript's warning while a line is typed.
"""
from __future__ import annotations

import math
from typing import Optional

from kokoro_gui.daw.arrangement import FALLBACK_CHARS_PER_SECOND, clip_audio_duration_s, estimate_duration_s

TARGET_KEY = "target_duration_s"
# A fit stops once the clip is within this fraction of its target, and the
# timeline counts a clip that close as fitting (no tint).
FIT_TOLERANCE = 0.03
# Above this ratio the timeline tints a clip red instead of amber.
FIT_OVER_RATIO = 1.15
SPEED_MIN = 0.7
SPEED_MAX = 1.6
MAX_FIT_PASSES = 3
# Time stretch past these starts to sound processed; a line that needs more
# is a line to rewrite.
STRETCH_MAX = 1.15
STRETCH_MIN = 0.87


def target_duration_s(clip) -> Optional[float]:
    """The clip's target in seconds, or None when it has none (or a value
    that isn't a positive number, as a hand-edited document.json might
    hold)."""
    value = (getattr(clip, "overrides", None) or {}).get(TARGET_KEY)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def fit_ratio(duration_s, target_s) -> Optional[float]:
    """`duration_s / target_s`, or None when either is missing."""
    if duration_s is None or not target_s or target_s <= 0:
        return None
    return float(duration_s) / float(target_s)


def fit_level(ratio) -> Optional[str]:
    """How the timeline shows a ratio: None (no target), "fit" (within the
    tolerance or short), "over" (amber) or "far_over" (red, past
    `FIT_OVER_RATIO`)."""
    if ratio is None:
        return None
    if ratio > FIT_OVER_RATIO:
        return "far_over"
    if ratio > 1.0 + FIT_TOLERANCE:
        return "over"
    return "fit"


def is_fitted(ratio) -> bool:
    return ratio is not None and abs(ratio - 1.0) <= FIT_TOLERANCE


def clamp_speed(speed: float) -> float:
    return max(SPEED_MIN, min(SPEED_MAX, float(speed)))


def next_speed(speed: float, duration_s: float, target_s: float) -> float:
    """The speed the next fit pass generates at: the current one scaled by
    how far over (or under) the target the clip ran, clamped to
    `SPEED_MIN..SPEED_MAX` and rounded so the segment key is stable."""
    return round(clamp_speed(float(speed) * float(duration_s) / float(target_s)), 3)


def stretch_for(natural_s: float, target_s: float) -> tuple:
    """`(factor, needs_rewrite)` for a clip that renders `natural_s` long
    unstretched. The factor is None when the clip already fits; it is
    capped at `STRETCH_MAX` (then `needs_rewrite` is True) and floored at
    `STRETCH_MIN`."""
    factor = float(natural_s) / float(target_s)
    if abs(factor - 1.0) <= FIT_TOLERANCE:
        return None, False
    if factor > STRETCH_MAX:
        return STRETCH_MAX, True
    return round(max(STRETCH_MIN, factor), 4), False


def _clip_speed(document, clip) -> float:
    speed = document.effective_config_for_clip(clip).get("speed", 1.0)
    try:
        speed = float(speed)
    except (TypeError, ValueError):
        return 1.0
    return speed if math.isfinite(speed) and speed > 0 else 1.0


def speaking_rates(document) -> dict:
    """Characters per second at speed 1.0, learned from the document's
    generated clips: `{character_id: rate}` plus the whole document under
    None. Generation stats (`arrangement.recorded_chars_per_second`) time
    the model's work, not the speech, so they can't answer this."""
    totals: dict = {}
    for clip in document.clips:
        if getattr(clip, "is_nested", False):
            continue
        seconds = clip_audio_duration_s(clip)
        chars = len(document.clip_text(clip).strip())
        if not seconds or not chars:
            continue
        speed = _clip_speed(document, clip)
        for key in (clip.character_id, None):
            entry = totals.setdefault(key, [0, 0.0])
            entry[0] += chars
            entry[1] += seconds * speed
    return {key: chars / seconds for key, (chars, seconds) in totals.items() if seconds > 0}


def reading_rate_ratio(document, clip, rates: Optional[dict] = None) -> Optional[float]:
    """The clip's text read at its speed and its character's learned rate
    (`speaking_rates`, else the document's, else
    `FALLBACK_CHARS_PER_SECOND`) over its target, or None without one."""
    target = target_duration_s(clip)
    if target is None:
        return None
    rates = speaking_rates(document) if rates is None else rates
    rate = rates.get(clip.character_id) or rates.get(None) or FALLBACK_CHARS_PER_SECOND
    return estimate_duration_s(document.clip_text(clip), _clip_speed(document, clip), rate) / target
