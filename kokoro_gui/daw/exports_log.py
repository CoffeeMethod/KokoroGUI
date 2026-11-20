"""The exports log (plan 38): what the user exported, newest first.

A log and not a scan of `audio_output/`, because exports go to any folder and a
scan would list files nothing here wrote. `exports.json` sits beside
`config_qt.json` and keeps the last `MAX_ENTRIES` records. It is local state like
the config: never part of a project, and untrusted input when read, so every
field is type-checked on load.

No Qt, no app import. The Exports dock (`qt/docks/output_dock.py`) shows it.
"""
from __future__ import annotations

import json
import os
import time

LOG_NAME = "exports.json"
MAX_ENTRIES = 200


def log_path(config_file: str) -> str:
    """`exports.json` in the folder that holds `config_file`."""
    return os.path.join(os.path.dirname(os.path.abspath(config_file)), LOG_NAME)


def _clean(raw) -> dict | None:
    """One stored entry with every field the right type, or None."""
    if not isinstance(raw, dict):
        return None
    path = raw.get("path")
    if not isinstance(path, str) or not path:
        return None
    extras = raw.get("extras")
    when = raw.get("when")
    duration = raw.get("duration_s")
    return {
        "path": path,
        "when": float(when) if isinstance(when, (int, float)) and not isinstance(when, bool) else 0.0,
        "project_title": raw.get("project_title") if isinstance(raw.get("project_title"), str) else "",
        "project_path": raw.get("project_path") if isinstance(raw.get("project_path"), str) else "",
        "duration_s": float(duration) if isinstance(duration, (int, float)) and not isinstance(duration, bool)
        else 0.0,
        "format": raw.get("format") if isinstance(raw.get("format"), str) else "",
        "extras": [e for e in extras if isinstance(e, str) and e] if isinstance(extras, list) else [],
    }


def _load(file: str) -> list:
    """Stored entries, oldest first. A missing, unreadable or malformed file is an empty log."""
    try:
        with open(file, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [entry for entry in map(_clean, data) if entry is not None]


def _write(file: str, entries: list) -> None:
    tmp = file + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
        os.replace(tmp, file)
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass


def record(file: str, path: str, *, project_title: str = "", project_path: str = "", duration_s: float = 0.0,
           fmt: str = "", extras=(), when: float | None = None) -> None:
    """Appends one export, keeping the last `MAX_ENTRIES`. A write that fails
    is dropped; losing a log line must not fail an export."""
    entry = _clean({
        "path": os.path.abspath(path), "when": time.time() if when is None else when,
        "project_title": project_title, "project_path": project_path or "", "duration_s": duration_s,
        "format": fmt, "extras": [os.path.abspath(e) for e in extras],
    })
    if entry is None:
        return
    _write(file, (_load(file) + [entry])[-MAX_ENTRIES:])


def entries(file: str) -> list:
    """Newest first. Each entry gains `missing`, true when its audio file no longer exists."""
    rows = _load(file)[::-1]
    for row in rows:
        row["missing"] = not os.path.isfile(row["path"])
    return rows


def delete_targets(entry: dict) -> list:
    """The files Delete removes for `entry`: its audio and the extras that exist and sit in the
    audio's own folder. The log file is untrusted, so an extra pointing anywhere else is ignored, and
    so is a folder or a link."""
    audio = os.path.abspath(entry["path"])
    folder = os.path.normcase(os.path.dirname(audio))
    targets = []
    for path in [audio, *entry.get("extras", [])]:
        full = os.path.abspath(path)
        if os.path.normcase(os.path.dirname(full)) != folder or full in targets:
            continue
        if os.path.isfile(full) and not os.path.islink(full):
            targets.append(full)
    return targets


def forget(file: str, path: str) -> None:
    """Drops the entries for `path` from the log (after Delete)."""
    full = os.path.normcase(os.path.abspath(path))
    rows = _load(file)
    kept = [r for r in rows if os.path.normcase(os.path.abspath(r["path"])) != full]
    if len(kept) != len(rows):
        _write(file, kept)
