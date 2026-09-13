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

Unknown fields round-trip. Every model object has an `extra` dict:
`document_from_dict` splits each object's dict into the fields the running
version knows and the rest, and `document_to_dict` merges the rest back at
the same level, so a `document.json` written by a newer KokoroGUI survives a
load and save through an older one with its extra fields intact (the `.tbaw`
plan's version policy depends on this). `Character.preset_data` is filtered
through `ALLOWED_PRESET_KEYS` on the way in, and the stripped keys ride in
`extra["preset_data"]`.
"""
import dataclasses
import json
import os

from kokoro_gui.daw.models import Character, Clip, Document, Run, Segment, Track
from kokoro_gui.engine.presets import ALLOWED_PRESET_KEYS, filter_allowed_keys


def _known_fields(cls) -> set:
    return {f.name for f in dataclasses.fields(cls) if f.init}


def _split_unknown(cls, data: dict) -> tuple:
    """`(known, extra)`: `data`'s keys the dataclass `cls` accepts, and the
    rest. An `extra` key already in `data` (a file written by this version)
    is folded into the returned extra rather than nested twice."""
    known_names = _known_fields(cls) - {"extra"}
    known = {}
    extra = {}
    for key, value in data.items():
        if key == "extra" and isinstance(value, dict):
            extra.update(value)
        elif key in known_names:
            known[key] = value
        else:
            extra[key] = value
    return known, extra


def _to_dict(obj) -> dict:
    """`dataclasses.asdict` minus `extra`, whose contents are merged back at
    the same level. A known field always wins over a stale `extra` entry of
    the same name."""
    data = dataclasses.asdict(obj)
    extra = data.pop("extra", {}) or {}
    return {**extra, **data}


def _character_to_dict(character: Character) -> dict:
    data = _to_dict(character)
    stripped = (character.extra or {}).get("preset_data")
    if isinstance(stripped, dict):
        data.pop("preset_data", None)
        data["preset_data"] = {**stripped, **character.preset_data}
    return data


def document_to_dict(doc: Document) -> dict:
    """Plain-JSON-serializable shape for `doc`.

    Deliberately built field-by-field rather than via a blanket
    `dataclasses.asdict(doc)` - this is what keeps `doc.undo_stack` (item 4,
    "Undo/redo") and `doc.segment_key_fn` out of the saved file for free:
    both are runtime/session-only and neither is JSON-serializable."""
    clips = []
    for clip in doc.clips:
        data = _to_dict(clip)
        data["segments"] = [_to_dict(s) for s in clip.segments]
        clips.append(data)
    return {
        "runs": [_to_dict(r) for r in doc.runs],
        "clips": clips,
        "tracks": [_to_dict(t) for t in doc.tracks],
        "characters": [_character_to_dict(c) for c in doc.characters],
        "settings": dict(doc.settings),
    }


def rewrite_audio_paths(data: dict, fn) -> dict:
    """Applies `fn(path) -> path` to every `Segment.audio_path` and
    `Clip.original_audio_path` in a `document_to_dict`-shaped dict, in
    place, skipping `None`. Used in both directions by the `.tbaw` bundle
    (absolute inside the project dir <-> bundle-relative)."""
    for clip in data.get("clips", []):
        if clip.get("original_audio_path"):
            clip["original_audio_path"] = fn(clip["original_audio_path"])
        for segment in clip.get("segments", []):
            if segment.get("audio_path"):
                segment["audio_path"] = fn(segment["audio_path"])
    return data


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
        segments = []
        for seg in clip_data.pop("segments", []):
            known, extra = _split_unknown(Segment, {"raw": False, **seg})
            segments.append(Segment(extra=extra, **known))
        start_offset = clip_data.pop("start_offset", None)
        end_offset = clip_data.pop("end_offset", None)
        known, extra = _split_unknown(Clip, clip_data)
        clip = Clip(segments=segments, extra=extra, **known)
        clips.append(clip)
        if start_offset is not None and end_offset is not None:
            legacy_offsets[clip.id] = (start_offset, end_offset)

    tracks = []
    for t in data.get("tracks", []):
        known, extra = _split_unknown(Track, t)
        tracks.append(Track(extra=extra, **known))

    characters = []
    for c in data.get("characters", []):
        known, extra = _split_unknown(Character, c)
        preset_data = known.get("preset_data") or {}
        if isinstance(preset_data, dict):
            stripped = {k: v for k, v in preset_data.items() if k not in ALLOWED_PRESET_KEYS}
            known["preset_data"] = filter_allowed_keys(preset_data, ALLOWED_PRESET_KEYS)
            if stripped:
                extra["preset_data"] = stripped
        characters.append(Character(extra=extra, **known))

    if "runs" in data:
        runs = []
        for r in data["runs"]:
            known, extra = _split_unknown(Run, r)
            runs.append(Run(extra=extra, **known))
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
