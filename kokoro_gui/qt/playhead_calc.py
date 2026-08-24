"""Wall-clock playhead position calculator for the waveform-view spike
(Workstream 3 of Claude/PLAN_daw_ui_ux_redesign.md). Pure Python, no Qt
import, no real-time waiting - lives under `kokoro_gui/qt/` for the same
reason waveform_data.py does: it's presentation-layer math for one specific
widget, colocated with its sole consumer (waveform_view.py).
"""
from typing import Optional


def playhead_x(elapsed_seconds: float, duration_seconds: float, view_width: float) -> Optional[float]:
    """The playhead's x-coordinate after `elapsed_seconds` of (assumed
    real-time-speed) playback into an audio clip `duration_seconds` long,
    rendered across a view `view_width` pixels wide.

    This is a **wall-clock approximation**, not sample-accurate:
    `playback.py`'s `play()` is fire-and-forget with no stream/position
    handle, so there's no way to ask "how many frames has the sound card
    actually consumed" - this function instead assumes playback started at
    `elapsed_seconds == 0` and proceeds at exactly real-time speed. It will
    drift from the true position under any scheduling jitter. The concrete
    fix - extending `playback.py`'s `play()` into a
    `sounddevice.OutputStream` with a frame-count callback for real
    position tracking - is deliberately out of scope for this spike.

    Returns `None` only when `duration_seconds` isn't a usable audio
    duration (`<= 0`) - not once playback has finished. A finished/
    over-elapsed playhead clamps to `view_width` instead of vanishing, since
    a playhead should visibly rest at the end of playback. A negative
    `elapsed_seconds` clamps to the start (`0.0`).
    """
    if duration_seconds <= 0:
        return None
    if elapsed_seconds < 0:
        elapsed_seconds = 0.0
    if elapsed_seconds >= duration_seconds:
        return float(view_width)
    return (elapsed_seconds / duration_seconds) * view_width
