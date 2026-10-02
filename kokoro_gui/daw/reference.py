"""Original dialogue as a per-clip reference (phase 5 D5).

`Document.settings["source_track"]` is `{"path": str, "offset_s": float}`:
the original dialogue, copied once into the project dir by
`kokoro_gui.qt.project.import_audio_file`. The path is always
project-relative (`audio/imported/<hash>.<ext>`), so the document names the
same file wherever the project dir lives; `kokoro_gui.qt.project.
source_track_path` resolves it at use time. `offset_s` is where the
reference times start in the file: source-file seconds are reference
seconds plus `offset_s` (a track with a two-second leader has 2.0).

`Clip.overrides["reference_range"]` is `[start_s, end_s]` in reference
seconds: the part of the source track that the clip dubs. A subtitle import
fills it from each cue (`kokoro_gui.daw.undo.ImportCuesCommand`), or it is
set by hand in the Settings tab. It is never derived from a pinned clip's
timestamp and target duration: a drag moves the timestamp, and the original
has to stay the line it was.

Playback only: the transport's Original and Both monitor modes play these
slices under each clip (`reference_slices`); export is the dub alone. Pure
module, no Qt.
"""
from __future__ import annotations

from typing import Optional

SOURCE_TRACK_KEY = "source_track"
REFERENCE_RANGE_KEY = "reference_range"


def source_track_settings(settings) -> Optional[dict]:
    """`settings["source_track"]` as `{"path": str, "offset_s": float}` with
    the path's separators as `/`, or None when unset or malformed."""
    block = (settings or {}).get(SOURCE_TRACK_KEY) if isinstance(settings, dict) else None
    if not isinstance(block, dict):
        return None
    path = block.get("path")
    if not isinstance(path, str) or not path.strip():
        return None
    try:
        offset = float(block.get("offset_s", 0.0) or 0.0)
    except (TypeError, ValueError):
        offset = 0.0
    return {"path": path.replace("\\", "/"), "offset_s": offset}


def reference_range(clip) -> Optional[tuple]:
    """`(start_s, end_s)` from `clip.overrides["reference_range"]`, or None
    when unset, malformed or empty."""
    raw = (getattr(clip, "overrides", None) or {}).get(REFERENCE_RANGE_KEY)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        start, end = float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None
    if start < 0.0 or end <= start:
        return None
    return start, end


def reference_slices(arrangement, offset_s: float = 0.0) -> list:
    """`[(clip_id, start_s, (file_start_s, file_end_s))]` for every placed
    clip with a reference range: where its original plays on the timeline
    and which seconds of the source track. The original starts at the
    clip's anchor (its timestamp for a locked clip, `start_s +
    aligned_onset_s`), so it lines up with the dub's first word the way the
    cue did. A clip still estimated plays its original too. A slice that
    starts before the file (a negative offset) is cut at 0."""
    slices = []
    for placed in arrangement.placed:
        rng = reference_range(placed.clip)
        if rng is None:
            continue
        start, end = rng[0] + offset_s, rng[1] + offset_s
        shift = max(0.0, -start)
        start = max(0.0, start)
        if end <= start:
            continue
        anchor = placed.start_s + float(getattr(placed, "aligned_onset_s", 0.0) or 0.0)
        slices.append((placed.clip.id, anchor + shift, (start, end)))
    return slices
