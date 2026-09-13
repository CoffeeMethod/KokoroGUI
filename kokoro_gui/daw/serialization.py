"""Save/load a `Document` as `document.json` - a new project file, kept
separate from `config_qt.json`'s flat app-settings shape rather than folded
into it, but following the same zero-ceremony "one implicit session,
autoloaded/autosaved" model (no File>Open/Save-As UX added here; that's an
open product question for a later pass, per
Claude/PLAN_daw_ui_ux_redesign.md).

Since Claude/PLAN_text_editor_redesign.md, the on-disk shape is a run list
(`{"runs": [{"text": ..., "clip_id": ..., "kind": ...}, ...], "clips": [...],
...}`) rather than a flat `"text"` string plus offset-ranged clips - this
*is* the "JSON tagging" the redesign asked for, produced by walking
`Document.runs` directly rather than hand-serializing a parallel
offset-tracked object. `document_from_dict` still reads the old
`{"text": ..., "clips": [{"start_offset": ..., "end_offset": ..., ...}]}`
shape for any `document.json` written before this rework - see
`_runs_from_legacy_offsets` below.
"""
import dataclasses
import json
import os

from kokoro_gui.daw.models import Character, Clip, Document, Run, Segment, Track


def document_to_dict(doc: Document) -> dict:
    """Plain-JSON-serializable shape for `doc` - dataclasses become dicts via
    `dataclasses.asdict`, which already handles the nested `Segment` objects
    inside each `Clip`.

    Deliberately built field-by-field rather than via a blanket
    `dataclasses.asdict(doc)` - this is what keeps `doc.undo_stack` (item 4,
    "Undo/redo") out of the saved file for free: it's runtime/session-only
    editing history, not part of the persisted project, and isn't even
    JSON-serializable (it holds `Command` objects, not plain data)."""
    return {
        "runs": [dataclasses.asdict(r) for r in doc.runs],
        "clips": [dataclasses.asdict(c) for c in doc.clips],
        "tracks": [dataclasses.asdict(t) for t in doc.tracks],
        "characters": [dataclasses.asdict(c) for c in doc.characters],
        "settings": dict(doc.settings),
    }


def _runs_from_legacy_offsets(text: str, clips: list, legacy_offsets: dict) -> list:
    """Migration path for a `document.json` written before the tagged-run
    rework: walks `clips` sorted by their old `start_offset`, emitting one
    run per clip plus untagged runs for whatever text fell outside every
    clip's old range - the one-time offsets-to-runs conversion
    Claude/PLAN_text_editor_redesign.md's "Migration path" section calls
    for, same spirit as `migration.py`'s existing presets-to-Character
    bootstrap."""
    ranges = sorted(
        (
            (start, end, clip.id, clip.source)
            for clip in clips
            if clip.id in legacy_offsets
            for start, end in [legacy_offsets[clip.id]]
        ),
        key=lambda r: r[0],
    )
    runs = []
    cursor = 0
    for start, end, clip_id, kind in ranges:
        if start > cursor:
            runs.append(Run(text=text[cursor:start]))
        runs.append(Run(text=text[start:end], clip_id=clip_id, kind=kind))
        cursor = max(cursor, end)
    if cursor < len(text):
        runs.append(Run(text=text[cursor:]))
    return runs


def document_from_dict(data: dict) -> Document:
    """Inverse of `document_to_dict`. Tolerant of missing keys (an older or
    hand-edited `document.json`) the same way the rest of this codebase reads
    config/preset dicts with `.get(...)` defaults rather than requiring every
    key. Reads a pre-rework, offset-based `document.json` transparently via
    `_runs_from_legacy_offsets` when the file has no `"runs"` key at all."""
    clips = []
    legacy_offsets = {}
    for clip_data in data.get("clips", []):
        clip_data = dict(clip_data)
        # A saved segment without "raw" predates read-time FX: its file has
        # FX baked in, so it must not be post-processed again (see
        # Segment's docstring; dirty.is_clip_dirty regenerates it).
        segments = [Segment(**{"raw": False, **seg}) for seg in clip_data.pop("segments", [])]
        start_offset = clip_data.pop("start_offset", None)
        end_offset = clip_data.pop("end_offset", None)
        clip = Clip(segments=segments, **clip_data)
        clips.append(clip)
        if start_offset is not None and end_offset is not None:
            legacy_offsets[clip.id] = (start_offset, end_offset)

    tracks = [Track(**t) for t in data.get("tracks", [])]
    characters = [Character(**c) for c in data.get("characters", [])]

    if "runs" in data:
        runs = [Run(**r) for r in data["runs"]]
    else:
        runs = _runs_from_legacy_offsets(data.get("text", ""), clips, legacy_offsets)

    return Document(
        runs=runs,
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
