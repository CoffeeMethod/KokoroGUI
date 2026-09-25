"""What the mixer applies to each placed clip, from the track controls and
the clip's fades. The transport's schedule (`QtTTSApp._rebuild_transport_schedule`)
and the exporter (`mixdown`) both read `clip_mixes`, so export plays what
the transport plays.

- Gain is the track fader. A muted track's clips, and every clip on a
  non-solo track while any track is soloed, get no entry at all: the caller
  leaves them out of the schedule instead of loading them at gain 0.
- Pan and the automation lane come from the clip's track.
- `duck` is the track's `Track.duck`: the mixer turns the clip down while
  speech plays. `sidechain` marks speech the ducking listens to: every
  clip except a music bed (`Clip.is_bed`) and a ducked clip.
- Fades are the clip's own `fade_in_s`/`fade_out_s`. With
  `document.settings["auto_crossfade"]` on, an edge that lies inside
  another placed clip (either track) gets at least `AUTO_CROSSFADE_S`: a
  schedule-time default, never written back to the clip.

Qt-free, like the rest of `kokoro_gui/daw/`.
"""
from __future__ import annotations

from dataclasses import dataclass

AUTO_CROSSFADE_S = 0.01


@dataclass(frozen=True)
class ClipMix:
    gain: float = 1.0
    pan: float = 0.0
    fade_in_s: float = 0.0
    fade_out_s: float = 0.0
    automation: tuple = ()  # the track's `[seconds, gain]` pairs
    duck: bool = False
    sidechain: bool = True


def _float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _inside_another(placed, seconds: float) -> bool:
    return any(other.start_s < seconds < other.end_s for other in placed)


def clip_mixes(document, arrangement) -> dict:
    """`{clip_id: ClipMix}` for every audible placed clip."""
    tracks = {t.id: t for t in document.tracks}
    # Only a track with clips counts: an unused one isn't drawn (grill PR4),
    # so its solo button can't be reached to switch it off.
    any_solo = any(t.solo for t in document.used_tracks())
    auto_crossfade = bool((document.settings or {}).get("auto_crossfade", False))
    placed = list(arrangement.placed)
    mixes = {}
    for index, item in enumerate(placed):
        clip = item.clip
        track = tracks.get(clip.track_id)
        if track is not None:
            if track.mute or (any_solo and not track.solo):
                continue
            gain = max(0.0, _float(track.gain, 1.0))
            pan = max(-1.0, min(1.0, _float(track.pan, 0.0)))
            automation = tuple(tuple(p) for p in (track.automation or []))
            duck = bool(getattr(track, "duck", False))
        else:
            gain, pan, automation, duck = 1.0, 0.0, (), False
        fade_in = max(0.0, _float(clip.fade_in_s, 0.0))
        fade_out = max(0.0, _float(clip.fade_out_s, 0.0))
        if auto_crossfade:
            others = placed[:index] + placed[index + 1:]
            if _inside_another(others, item.start_s):
                fade_in = max(fade_in, AUTO_CROSSFADE_S)
            if _inside_another(others, item.end_s):
                fade_out = max(fade_out, AUTO_CROSSFADE_S)
        sidechain = not duck and not getattr(clip, "is_bed", False)
        mixes[clip.id] = ClipMix(gain=gain, pan=pan, fade_in_s=fade_in, fade_out_s=fade_out,
                                 automation=automation, duck=duck, sidechain=sidechain)
    return mixes
