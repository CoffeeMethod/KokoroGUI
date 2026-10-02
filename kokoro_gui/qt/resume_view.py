"""The "resume where you left off" record: what `session.json["view"]` holds
and how a value read back from it is checked.

`session.json` is runtime state in the project dir. It is never bundled, so
a `.tbaw` carries none of this. A crafted or damaged file must not break
Open, so `clean_view` keeps only the keys whose value has the right type and
range, and `QtTTSApp._restore_view` applies what is left.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

from PySide6.QtCore import QTimer

from kokoro_gui.qt import project as project_io
from kokoro_gui.qt.timeline_view import MAX_PIXELS_PER_SECOND, MIN_PIXELS_PER_SECOND

VIEW_KEY = "view"
# A value past these is damage, not a view: eleven days of audio, a
# ten-million-pixel scroll.
MAX_PLAYHEAD_S = 1_000_000.0
MAX_SCROLL = 10_000_000
MAX_CLIP_ID_CHARS = 200
# A scroll restore that the layout clamped tries once more after this long.
SCROLL_RETRY_MS = 200


def _number(value) -> Optional[float]:
    """A finite float, or None (a bool is not a number here)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _scroll(value) -> Optional[int]:
    number = _number(value)
    if number is None or not 0 <= number <= MAX_SCROLL:
        return None
    return int(number)


def clean_view(raw, has_clip: Callable[[str], bool]) -> dict:
    """The valid part of a `view` read from `session.json`: any of
    `playhead_s`, `zoom`, `timeline_scroll_x`, `transcript_scroll_y` and
    `selected_clip_id`. `has_clip(id)` says whether a clip still exists; one
    that doesn't is dropped. Anything that isn't a dict gives {}."""
    if not isinstance(raw, dict):
        return {}
    view: dict = {}
    playhead = _number(raw.get("playhead_s"))
    if playhead is not None and 0.0 <= playhead <= MAX_PLAYHEAD_S:
        view["playhead_s"] = playhead
    zoom = _number(raw.get("zoom"))
    if zoom is not None and MIN_PIXELS_PER_SECOND <= zoom <= MAX_PIXELS_PER_SECOND:
        view["zoom"] = zoom
    for key in ("timeline_scroll_x", "transcript_scroll_y"):
        value = _scroll(raw.get(key))
        if value is not None:
            view[key] = value
    clip_id = raw.get("selected_clip_id")
    if isinstance(clip_id, str) and 0 < len(clip_id) <= MAX_CLIP_ID_CHARS and has_clip(clip_id):
        view["selected_clip_id"] = clip_id
    return view


def remember(app, keep_playhead: bool = False) -> None:
    """Writes where the user is (playhead, timeline zoom and scroll,
    transcript scroll, selected clip) to `session.json["view"]`, like the
    loop region: runtime state, never in the document or the bundle. Only the
    root project is remembered; with a subproject open on the timeline or in
    the transcript nothing is written. `keep_playhead` keeps the stored
    playhead (a closing window has already stopped the transport, which
    rewinds it)."""
    root = app.root
    if not root.project_dir or app.level is not root or app.focus is not root:
        return
    if app.timeline_dock is None or app.editor is None:
        return
    timeline_view = app.timeline_dock.timeline_view
    session = project_io.read_session(root.project_dir) or {}
    previous = session.get(VIEW_KEY)
    playhead = app.transport.position()
    if keep_playhead and isinstance(previous, dict) and "playhead_s" in previous:
        playhead = previous["playhead_s"]
    session[VIEW_KEY] = {
        "playhead_s": playhead,
        "zoom": timeline_view.zoom,
        "timeline_scroll_x": timeline_view.horizontalScrollBar().value(),
        "transcript_scroll_y": app.editor.verticalScrollBar().value(),
        "selected_clip_id": app.selection.selected_clip_id,
    }
    try:
        project_io.write_session(root.project_dir, session)
    except OSError:
        pass


def restore(app, session) -> None:
    """Applies `session["view"]` after Open: zoom, the selected clip, the
    playhead (held in `app._resume_playhead_s` until the transport has its
    schedule, so an empty one doesn't clamp it to 0), then both scroll
    positions on the next event loop pass, after the layout has sizes. Each
    value is checked first (`clean_view`); a clip that no longer exists is
    ignored."""
    document = app.root.document
    view = clean_view((session or {}).get(VIEW_KEY), lambda clip_id: document.get_clip(clip_id) is not None)
    app._resume_playhead_s = view.get("playhead_s")
    if app.timeline_dock is None or not view:
        return
    timeline_view = app.timeline_dock.timeline_view
    if "zoom" in view:
        timeline_view.set_zoom(view["zoom"])
    if "selected_clip_id" in view:
        app.selection.select_clip(view["selected_clip_id"])
    scroll_x, scroll_y = view.get("timeline_scroll_x"), view.get("transcript_scroll_y")

    def _scroll(retry: bool = True) -> None:
        if app._closed or app.root.document is not document:
            return
        bars = []
        if scroll_x is not None:
            bars.append((timeline_view.horizontalScrollBar(), scroll_x))
        if scroll_y is not None and app.editor is not None:
            bars.append((app.editor.verticalScrollBar(), scroll_y))
        for bar, value in bars:
            bar.setValue(value)
        # A window that isn't laid out yet clamps to a range of 0; once more
        # after it is.
        if retry and any(bar.value() != value for bar, value in bars):
            QTimer.singleShot(SCROLL_RETRY_MS, app, lambda: _scroll(False))

    QTimer.singleShot(0, app, _scroll)
