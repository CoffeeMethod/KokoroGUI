"""The projects a window has open (phase 4, subprojects).

A window always has a root project, the `.tbaw` the File menu opened, and
may have subprojects of it open too: one `OpenProject` each, with its own
document, project dir, lock and session. `QtTTSApp` keeps them in
`root` plus `children` (keyed by project id) and points three roles at them:

- `focus`: the project the transcript, Settings and Audio FX docks show
  (the selection decides: a nested block's child, else `level`).
- `level`: the project the timeline and transport show (double-clicking a
  nested block enters its child; the breadcrumb goes back up).
- `root`: what Save, Save As, Close and the title bar act on.

A tree the queue still works in when another project opens is parked
(`ParkedTree`) instead of closed, and comes back as it was when its root
is opened again (plan 30).

Qt-free on purpose, like `project.py`'s helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from kokoro_gui.daw.models import Document


@dataclass
class OpenProject:
    document: Document
    project_dir: Optional[str] = None
    project_id: Optional[str] = None
    lock: object = None  # project.ProjectLock
    project_settings: dict = field(default_factory=dict)
    # The `.tbaw` on disk: the root's file, or a linked child's. None for
    # an Untitled root and for an embedded child (it lives in its parent).
    path: Optional[str] = None
    # None for the root; else the parent's project id and the nested clip
    # in the parent's document that stands for this project.
    parent_id: Optional[str] = None
    clip_id: Optional[str] = None
    # A child's `Clip.child["kind"]`: "embedded" or "linked". None for the root.
    kind: Optional[str] = None
    manifest: dict = field(default_factory=dict)
    # The project dir is ahead of the file (autosave's digest differs from
    # the last Save or Open).
    dirty: bool = False
    # `<project_dir>/mixdown.<fmt>` and the document digest it was rendered
    # from (`mixdown.json`), for a child; see app.py's subproject mixdown.
    mixdown_path: Optional[str] = None
    mixdown_digest: Optional[str] = None
    # The digest of the document as autosave last wrote it; what a mixdown
    # is compared against.
    digest: Optional[str] = None
    # "ok" / "stale" for a child, computed on demand and dropped on any edit
    # (app.py's `child_state`).
    state_cache: Optional[str] = None

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    def title(self) -> str:
        from kokoro_gui.qt.project import display_title

        return display_title(self.project_settings, self.path, "Untitled" if self.is_root else "Subproject")


@dataclass
class ParkedTree:
    """A root and its open subprojects that the window moved away from while
    the generation queue still had work for them (plan 30). Nothing is torn
    down: the documents, dirs, locks and sessions stay as they were, results
    land in them and autosave writes them. `QtTTSApp.parked_projects` holds
    one per root project id until the last item for the tree ends, or the
    root is opened again."""

    root: OpenProject
    children: dict = field(default_factory=dict)

    def projects(self) -> list:
        return [self.root, *self.children.values()]

    def ids(self) -> set:
        return {project.project_id for project in self.projects()}

    def holds(self, project) -> bool:
        return any(candidate is project for candidate in self.projects())
