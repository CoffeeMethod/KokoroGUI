"""Named points on the timeline: `Document.settings["markers"]`, a list of
`{"id", "seconds", "name", "note"}` dicts kept sorted by time. A marker
lives in `settings` rather than a `Document` field because 4.0 drops
unknown top-level `document.json` keys; `settings` round-trips whole.

Every function here returns a new list and leaves the document alone, so
the caller applies it with one `SetFieldCommand("document", None,
"settings", new_list, key="markers")` and the edit is undoable. A
listen-through flag is a marker with a note.
"""
from __future__ import annotations

import uuid

MARKERS_KEY = "markers"


def _clean(marker) -> dict | None:
    if not isinstance(marker, dict):
        return None
    try:
        seconds = max(0.0, float(marker.get("seconds", 0.0)))
    except (TypeError, ValueError):
        return None
    return {
        "id": str(marker.get("id") or uuid.uuid4().hex),
        "seconds": seconds,
        "name": str(marker.get("name") or ""),
        "note": str(marker.get("note") or ""),
    }


def list_markers(settings: dict | None) -> list:
    """The document's markers, cleaned and sorted by time (copies)."""
    raw = (settings or {}).get(MARKERS_KEY)
    cleaned = [m for m in (_clean(m) for m in (raw if isinstance(raw, list) else [])) if m is not None]
    return sorted(cleaned, key=lambda m: m["seconds"])


def get_marker(settings: dict | None, marker_id: str) -> dict | None:
    return next((m for m in list_markers(settings) if m["id"] == marker_id), None)


def add_marker(settings: dict | None, seconds: float, name: str = "", note: str = "") -> tuple:
    """`(new_list, marker)`. An empty name becomes "M<n>"."""
    markers = list_markers(settings)
    marker = _clean({"seconds": seconds, "name": name or f"M{len(markers) + 1}", "note": note})
    return sorted(markers + [marker], key=lambda m: m["seconds"]), marker


def move_marker(settings: dict | None, marker_id: str, seconds: float) -> list:
    return sorted(
        ({**m, "seconds": max(0.0, float(seconds))} if m["id"] == marker_id else m for m in list_markers(settings)),
        key=lambda m: m["seconds"],
    )


def rename_marker(settings: dict | None, marker_id: str, name: str, note: str | None = None) -> list:
    out = []
    for m in list_markers(settings):
        if m["id"] == marker_id:
            m = {**m, "name": name}
            if note is not None:
                m["note"] = note
        out.append(m)
    return out


def delete_marker(settings: dict | None, marker_id: str) -> list:
    return [m for m in list_markers(settings) if m["id"] != marker_id]


def range_between(settings: dict | None, first_id: str, second_id: str) -> tuple | None:
    """`(start_s, end_s)` between two markers, or None when either is gone
    or they sit at the same time."""
    a, b = get_marker(settings, first_id), get_marker(settings, second_id)
    if a is None or b is None or a["seconds"] == b["seconds"]:
        return None
    return tuple(sorted((a["seconds"], b["seconds"])))
