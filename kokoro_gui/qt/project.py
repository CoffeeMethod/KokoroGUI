"""Project files for the File menu (Claude/PLAN_ui_shell_redesign.md
section 7). Pure functions, no Qt.

A project is a `.json` file in `document.json`'s shape
(`kokoro_gui.daw.serialization`) plus a top-level `"project_settings"`
block for per-project things that don't belong in `config_qt.json` (last
export settings, workspace override). `settings["last_project"]` is what
launch reopens (WF2, revised: the welcome dialog offers it as Resume);
`settings["recent_projects"]` is the File > Recent list and the welcome
dialog's rows, most recent first, at most `MAX_RECENT` entries.

`format_for_path` exists so the `.tbaw` zip bundle (section 8, spec only
for now) can plug in later without touching the menu code: everything
routes through `load_project`/`save_project`, which dispatch on the
extension.
"""
from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass, field

from kokoro_gui.daw import serialization
from kokoro_gui.daw.models import Document

MAX_RECENT = 10
PROJECT_FILTER = "KokoroGUI project (*.json)"
DEFAULT_EXTENSION = ".json"


@dataclass
class LoadedProject:
    document: Document
    project_settings: dict = field(default_factory=dict)


def format_for_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".tbaw":
        return "tbaw"
    return "json"


def load_project(path: str) -> LoadedProject | None:
    """`None` when the file is missing or unreadable, same tolerance
    `serialization.load_document` has."""
    if not path or not os.path.exists(path):
        return None
    if format_for_path(path) == "tbaw":
        raise NotImplementedError(".tbaw bundles are specified but not implemented yet (see the plan's section 8).")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    document = serialization.document_from_dict(data)
    project_settings = data.get("project_settings", {})
    return LoadedProject(document=document, project_settings=dict(project_settings) if isinstance(project_settings, dict) else {})


def save_project(document: Document, path: str, project_settings: dict | None = None) -> None:
    if format_for_path(path) == "tbaw":
        raise NotImplementedError(".tbaw bundles are specified but not implemented yet (see the plan's section 8).")
    data = serialization.document_to_dict(document)
    data["project_settings"] = dict(project_settings or {})
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def project_title(path: str | None) -> str:
    if not path:
        return "Untitled"
    return os.path.splitext(os.path.basename(path))[0] or "Untitled"


def remember_recent(settings: dict, path: str) -> list:
    """Moves `path` to the front of `settings["recent_projects"]`, dropping
    duplicates and trimming to `MAX_RECENT`. Returns the new list."""
    path = os.path.abspath(path)
    recent = [p for p in settings.get("recent_projects", []) if isinstance(p, str) and os.path.abspath(p) != path]
    recent.insert(0, path)
    settings["recent_projects"] = recent[:MAX_RECENT]
    settings["last_project"] = path
    return settings["recent_projects"]


def forget_recent(settings: dict, path: str) -> None:
    path = os.path.abspath(path)
    settings["recent_projects"] = [p for p in settings.get("recent_projects", []) if os.path.abspath(p) != path]


def clear_recent(settings: dict) -> None:
    """Empties the list. `last_project` stays: the open project is still the
    one launch resumes, it just isn't listed any more."""
    settings["recent_projects"] = []


def project_summary(path: str) -> dict | None:
    """What the welcome dialog's details pane shows for a row. Reads the raw
    JSON rather than building a `Document` - counts are all it needs, and
    it runs on every selection change. `None` when missing or unreadable."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        modified = datetime.datetime.fromtimestamp(os.path.getmtime(path))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    characters = data.get("characters", [])
    clips = data.get("clips", [])
    return {
        "path": os.path.abspath(path),
        "modified": modified,
        "characters": len(characters) if isinstance(characters, list) else 0,
        "clips": len(clips) if isinstance(clips, list) else 0,
    }


def new_document_from(previous: Document | None) -> Document:
    """WF3: a new project inherits the previous project's characters (a
    copy for now - the global library from WF4-WF7 is future work) and one
    track per character."""
    import copy

    from kokoro_gui.daw.models import Track

    if previous is None or not previous.characters:
        return Document(runs=[], clips=[], tracks=[], characters=[], settings={})
    characters = copy.deepcopy(previous.characters)
    tracks = [Track(name=c.name, character_id=c.id, order_index=i) for i, c in enumerate(characters)]
    return Document(runs=[], clips=[], tracks=tracks, characters=characters, settings={})
