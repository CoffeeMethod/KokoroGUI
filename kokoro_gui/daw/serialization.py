"""Save/load a `Document` as `document.json` - a new project file, kept
separate from `config_qt.json`'s flat app-settings shape rather than folded
into it, but following the same zero-ceremony "one implicit session,
autoloaded/autosaved" model (no File>Open/Save-As UX added here; that's an
open product question for a later pass, per
Claude/PLAN_daw_ui_ux_redesign.md).
"""
import dataclasses
import json
import os

from kokoro_gui.daw.models import Character, Clip, Document, Segment, Track


def document_to_dict(doc: Document) -> dict:
    """Plain-JSON-serializable shape for `doc` - dataclasses become dicts via
    `dataclasses.asdict`, which already handles the nested `Segment` objects
    inside each `Clip`."""
    return {
        "text": doc.text,
        "clips": [dataclasses.asdict(c) for c in doc.clips],
        "tracks": [dataclasses.asdict(t) for t in doc.tracks],
        "characters": [dataclasses.asdict(c) for c in doc.characters],
        "settings": dict(doc.settings),
    }


def document_from_dict(data: dict) -> Document:
    """Inverse of `document_to_dict`. Tolerant of missing keys (an older or
    hand-edited `document.json`) the same way the rest of this codebase reads
    config/preset dicts with `.get(...)` defaults rather than requiring every
    key."""
    clips = []
    for clip_data in data.get("clips", []):
        clip_data = dict(clip_data)
        segments = [Segment(**seg) for seg in clip_data.pop("segments", [])]
        clips.append(Clip(segments=segments, **clip_data))

    tracks = [Track(**t) for t in data.get("tracks", [])]
    characters = [Character(**c) for c in data.get("characters", [])]

    return Document(
        text=data.get("text", ""),
        clips=clips,
        tracks=tracks,
        characters=characters,
        settings=dict(data.get("settings", {})),
    )


def save_document(doc: Document, path: str) -> None:
    """Writes `doc` to `path` as JSON. Creates the parent directory if
    needed, matching the tolerant-write style the rest of the app's
    settings/preset save paths use."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(document_to_dict(doc), f, indent=2)


def load_document(path: str):
    """Reads `path` back into a `Document`, or returns `None` if the file
    doesn't exist or fails to parse - callers should treat `None` as "start a
    fresh Document" the same way a missing/corrupt `config_qt.json` falls
    back to defaults elsewhere in this app."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return document_from_dict(data)
