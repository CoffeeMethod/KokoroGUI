"""Named dock layouts ("workspaces") for the Qt shell: UI1/UI8 of
Claude/PLAN_ui_shell_redesign.md, custom workspaces from plan 47 pass 1.

`config_qt.json` holds:

- `"workspaces": {name: {"state": b64, "geometry": b64}}` for the two
  built-ins. Advanced is the drawing's 2x2 grid (transcript | settings tabs
  over timeline | transport), built by `QtTTSApp.arrange_docks_default()`;
  Simple is the same grid with the timeline and video docks hidden (UI8,
  layout only). A built-in remembers what you do to it: the debounced layout
  save and every settings save call `capture()`, which writes its entry.
- `"custom_workspaces": {name: {"state": b64, "geometry": b64}}`, the named
  snapshots, in menu order. Grill UI17: only Save workspace as and Update
  workspace write one. Capturing, switching and closing never do.
- `"workspace_live": {"workspace", "state", "geometry", "modified"}`, the
  live arrangement of the active custom workspace. Grill PG31: a restart
  brings it back, with its "modified" mark, instead of the snapshot.
  Choosing the workspace from the menu loads the snapshot.
- `"active_workspace"`, a built-in or custom name.

Every entry is checked on load (`_clean`): a damaged one is dropped with a
notice in `notices`, and an active name that no longer exists falls back to
Advanced. A saved blob is checked again (size, and the leading bytes
`saveState()` writes on this Qt) before it reaches `restoreState`; one that
fails gets the default layout and a notice.

"Modified" compares a fingerprint of the live docks (`layout_fingerprint`:
which docks are closed, floating, in which area and tab group, and their
sizes) with the one taken when the layout was loaded, saved or updated.
Which tab of a group is in front is left out, so raising a tab is no edit.

The pre-workspace `dock_state`/`geometry` keys migrate into
`workspaces.Advanced` the first time this loads them, then get dropped.
"""
from __future__ import annotations

import unicodedata

from kokoro_gui.qt import settings as qt_settings

# A saved blob outside these bounds is not something `saveState()` or
# `saveGeometry()` wrote. Qt's restore functions parse it natively, and a
# crafted blob in `config_qt.json` can crash them where `try/except` can't help.
MIN_BLOB_BYTES = 8
MAX_BLOB_BYTES = 1024 * 1024
PREFIX_BYTES = 4

ADVANCED = "Advanced"
SIMPLE = "Simple"
WORKSPACE_NAMES = (ADVANCED, SIMPLE)  # the built-ins

MAX_NAME_CHARS = 64
MAX_CUSTOM_WORKSPACES = 50

# How long after a load (or a window resize) the dock sizes count as settled.
# Until then "modified" compares the dock structure only.
SETTLE_MS = 400


class WorkspaceNameError(ValueError):
    """A workspace name that `validate_name` refuses; the message says why."""


class WorkspaceExistsError(WorkspaceNameError):
    """Save workspace as was given the name of an existing custom workspace.
    `name` is that workspace's name; the caller asks before replacing it."""

    def __init__(self, name: str):
        super().__init__(f"A workspace called {name} already exists.")
        self.name = name


def validate_name(name, existing=(), current: str | None = None) -> str:
    """The cleaned name (NFC, trimmed), or `WorkspaceNameError`. A name is
    a menu label and a JSON key, never a path. `existing` are the custom
    names, compared ignoring case; `current` is the one being renamed, which
    may keep its name or change its case."""
    if not isinstance(name, str):
        raise WorkspaceNameError("Type a name for the workspace.")
    cleaned = unicodedata.normalize("NFC", name).strip()
    if not cleaned:
        raise WorkspaceNameError("Type a name for the workspace.")
    if len(cleaned) > MAX_NAME_CHARS:
        raise WorkspaceNameError(f"Use {MAX_NAME_CHARS} characters or fewer.")
    if any(unicodedata.category(ch) in ("Cc", "Cs", "Zl", "Zp") for ch in cleaned):
        raise WorkspaceNameError("Use a name on one line, without control characters.")
    folded = cleaned.casefold()
    if folded in {builtin.casefold() for builtin in WORKSPACE_NAMES}:
        raise WorkspaceNameError(f"{cleaned} is a built-in workspace. Choose another name.")
    for other in existing:
        if other != current and other.casefold() == folded:
            raise WorkspaceExistsError(other)
    return cleaned


def menu_label(name: str) -> str:
    """`name` as a QAction text: a literal "&" is doubled, so it shows
    instead of making the next letter a mnemonic."""
    return name.replace("&", "&&")


def _clean_entry(entry, need_state: bool) -> dict | None:
    """`{"state", "geometry"}` with str-or-None values, else None."""
    if not isinstance(entry, dict):
        return None
    state, geometry = entry.get("state"), entry.get("geometry")
    if not isinstance(state, (str, type(None))) or not isinstance(geometry, (str, type(None))):
        return None
    if need_state and not state:
        return None
    return {"state": state, "geometry": geometry}


def layout_fingerprint(window, docks) -> tuple:
    """(structure, sizes) of the live docks. Structure: for each dock its
    name, closed, floating, area and tab group. Sizes: a floating dock's
    geometry, and the size of the shown member of each docked group (an
    empty `visibleRegion` is a tab behind another, or a window not shown
    yet). Which tab of a group is in front is not part of it."""
    structure = []
    sizes = set()
    for dock in docks:
        name = dock.objectName()
        hidden = dock.isHidden()
        floating = dock.isFloating()
        group = tuple(sorted([name, *(other.objectName() for other in window.tabifiedDockWidgets(dock))]))
        structure.append((name, hidden, floating, str(window.dockWidgetArea(dock)), group))
        if hidden:
            continue
        if floating:
            sizes.add(((name,), "float", dock.geometry().getRect()))
        elif not dock.visibleRegion().isEmpty():
            sizes.add((group, "docked", (dock.width(), dock.height())))
    return tuple(sorted(structure)), tuple(sorted(sizes))


class WorkspaceManager:
    def __init__(self, window, settings: dict):
        self._window = window
        self._settings = settings
        self._prefixes: dict[str, bytes] = {}
        self.notices: list[str] = []
        self._loading = False
        self._baseline = None
        self._settling = False
        self._sticky_modified = False
        self._settle_timer = None
        self._migrate_legacy_keys()
        self._clean()

    # -- persistence shape -------------------------------------------------

    def _migrate_legacy_keys(self) -> None:
        legacy_state = self._settings.pop("dock_state", None)
        legacy_geometry = self._settings.pop("geometry", None)
        if not legacy_state and not legacy_geometry:
            return
        workspaces = self._settings.get("workspaces")
        if not isinstance(workspaces, dict):
            workspaces = self._settings["workspaces"] = {}
        if ADVANCED not in workspaces:
            workspaces[ADVANCED] = {"state": legacy_state, "geometry": legacy_geometry}

    def _clean(self) -> None:
        """Type-checks the workspace keys of an untrusted `config_qt.json`.
        `clean_loaded_settings` checked the top-level types; this checks
        each entry and name. Keys of `workspaces` other than the built-ins
        are left alone."""
        workspaces = self._settings.get("workspaces")
        if not isinstance(workspaces, dict):
            workspaces = self._settings["workspaces"] = {}
        for name in WORKSPACE_NAMES:
            if name in workspaces and _clean_entry(workspaces[name], need_state=False) is None:
                del workspaces[name]
                self.notices.append(f"The saved {name} layout was damaged, so {name} starts from its default.")

        customs = self._settings.get("custom_workspaces")
        cleaned: dict = {}
        dropped = []
        if isinstance(customs, dict):
            for name, entry in customs.items():
                try:
                    good = validate_name(name, cleaned)
                except WorkspaceNameError:
                    good = None
                checked = _clean_entry(entry, need_state=True)
                if good is None or checked is None or len(cleaned) >= MAX_CUSTOM_WORKSPACES:
                    dropped.append(str(name)[:MAX_NAME_CHARS])
                    continue
                cleaned[good] = checked
                if good != name and self._settings.get("active_workspace") == name:
                    self._settings["active_workspace"] = good
        elif customs is not None:
            dropped.append("all of them")
        self._settings["custom_workspaces"] = cleaned
        if dropped:
            self.notices.append("Some saved workspaces were damaged and were removed: "
                                f"{', '.join(dropped[:5])}.")

        live = self._settings.get("workspace_live")
        if not isinstance(live, dict) or not isinstance(live.get("workspace"), str) \
                or not isinstance(live.get("modified", False), bool) or _clean_entry(live, need_state=True) is None:
            self._settings["workspace_live"] = {}

        if self._settings.get("active_workspace") not in (*WORKSPACE_NAMES, *cleaned):
            self._settings["active_workspace"] = ADVANCED

    @property
    def active(self) -> str:
        return self._settings.get("active_workspace", ADVANCED)

    def custom_names(self) -> list[str]:
        return list(self._settings.get("custom_workspaces", {}))

    def is_custom(self, name: str | None = None) -> bool:
        return (self.active if name is None else name) in self._settings.get("custom_workspaces", {})

    def saved(self, name: str) -> dict | None:
        """The saved entry of a built-in (its remembered layout) or a custom
        workspace (its snapshot), or None."""
        source = self._settings.get("custom_workspaces" if self.is_custom(name) else "workspaces", {})
        entry = source.get(name) if isinstance(source, dict) else None
        return entry if isinstance(entry, dict) else None

    # -- blobs -------------------------------------------------------------

    def _checked_blob(self, kind: str, encoded) -> object | None:
        """The decoded `state` or `geometry` blob when it plausibly came from
        this Qt's `saveState()`/`saveGeometry()`, else None. The expected
        leading bytes are read from the live window once, not hard-coded."""
        try:
            blob = qt_settings.decode_bytes(encoded)
            size = blob.size()
            if not MIN_BLOB_BYTES <= size <= MAX_BLOB_BYTES:
                return None
            if kind not in self._prefixes:
                made = self._window.saveState() if kind == "state" else self._window.saveGeometry()
                self._prefixes[kind] = bytes(made[:PREFIX_BYTES])
            if bytes(blob[:PREFIX_BYTES]) != self._prefixes[kind]:
                return None
            return blob
        except Exception:
            return None

    def _restore_geometry(self, entry) -> None:
        if entry and entry.get("geometry"):
            blob = self._checked_blob("geometry", entry["geometry"])
            if blob is not None:
                try:
                    self._window.restoreGeometry(blob)
                except Exception:
                    pass

    def _restore_state(self, entry) -> bool:
        blob = self._checked_blob("state", entry["state"]) if entry and entry.get("state") else None
        if blob is None:
            return False
        self._loading = True
        try:
            return bool(self._window.restoreState(blob))
        except Exception:
            return False
        finally:
            self._loading = False

    def _snapshot(self) -> dict:
        return {"state": qt_settings.encode_bytes(self._window.saveState()),
                "geometry": qt_settings.encode_bytes(self._window.saveGeometry())}

    # -- apply / capture ---------------------------------------------------

    def restore_on_launch(self) -> None:
        """Called once after the docks exist: geometry, then the layout. A
        built-in gets its remembered entry (or the default). A custom
        workspace gets the live arrangement it was left in (PG31), else its
        snapshot."""
        name = self.active
        if not self.is_custom(name):
            self._restore_geometry(self.saved(name))
            self.activate(name, save_outgoing=False)
            return
        live = self._settings.get("workspace_live") or {}
        if live.get("workspace") == name:
            self._restore_geometry(live)
            if self._restore_state(live):
                self.recover_offscreen_docks()
                self._loaded(sticky=bool(live.get("modified")))
                return
            self.notices.append(f"The last arrangement of {name} couldn't be restored, so its saved layout "
                                "is shown.")
        else:
            self._restore_geometry(self.saved(name))
        self.activate(name, save_outgoing=False)

    def activate(self, name: str, save_outgoing: bool = True) -> None:
        """Loads `name`: a built-in's remembered layout (or its default), a
        custom workspace's snapshot. The window keeps its size. A custom
        snapshot that can't be restored leaves the workspace active on the
        Advanced default, marked modified, so Update can replace it."""
        if name not in WORKSPACE_NAMES and not self.is_custom(name):
            name = ADVANCED
        if save_outgoing:
            self.capture()
        self._settings["active_workspace"] = name
        self._settings["workspace_live"] = {}
        custom = self.is_custom(name)
        entry = self.saved(name)
        restored = self._restore_state(entry)
        if not restored:
            self._loading = True
            try:
                self.apply_default(SIMPLE if name == SIMPLE else ADVANCED)
            finally:
                self._loading = False
            if entry is not None and entry.get("state"):
                self.notices.append(f"The {name} layout couldn't be restored, so the default layout is shown"
                                    + (". Use Update workspace to save a working one." if custom else "."))
        else:
            self.recover_offscreen_docks()
        self._loaded(sticky=custom and not restored)

    def apply_default(self, name: str) -> None:
        self._window.arrange_docks_default()
        for attr in ("timeline_dock", "video_dock"):
            dock = getattr(self._window, attr, None)
            if dock is not None:
                dock.setVisible(name != SIMPLE)
        if name == SIMPLE and hasattr(self._window, "apply_simple_proportions"):
            self._window.apply_simple_proportions()

    def reset(self) -> None:
        """Restore default layout. A built-in forgets its remembered edits
        and rebuilds its default. A custom workspace gets the Advanced
        default as its live arrangement; the snapshot stays until Update."""
        if self.is_custom():
            self.apply_default(ADVANCED)
            return
        self._settings.setdefault("workspaces", {}).pop(self.active, None)
        self.apply_default(self.active)
        self._loaded()

    def capture(self) -> None:
        """Remembers the live layout: a built-in's own entry, or a custom
        workspace's `workspace_live` (never its snapshot, UI17). Nothing
        while a layout is being applied."""
        if self._loading:
            return
        if self.is_custom():
            self._settings["workspace_live"] = {"workspace": self.active, **self._snapshot(),
                                                "modified": self.is_modified()}
            return
        self._settings.setdefault("workspaces", {})[self.active] = self._snapshot()

    # -- custom workspaces -------------------------------------------------

    def save_as(self, name: str, replace: bool = False) -> str:
        """A new snapshot of the live layout under `name`, which becomes the
        active, unmodified workspace. An existing custom name raises
        `WorkspaceExistsError` unless `replace` (the user confirmed), which
        replaces that snapshot and keeps its place in the menu."""
        existing = self.custom_names()
        try:
            name = validate_name(name, existing)
        except WorkspaceExistsError as clash:
            if not replace:
                raise
            name = clash.name
        customs = self._settings["custom_workspaces"]
        if name not in customs and len(customs) >= MAX_CUSTOM_WORKSPACES:
            raise WorkspaceNameError(f"There are already {MAX_CUSTOM_WORKSPACES} workspaces. Delete one first.")
        if not self.is_custom():
            self.capture()  # the built-in remembers the layout it leaves
        customs[name] = self._snapshot()
        self._settings["active_workspace"] = name
        self._loaded()
        self.capture()
        return name

    def update(self) -> bool:
        """Update workspace: the active custom snapshot becomes the live layout."""
        if not self.is_custom():
            return False
        self._settings["custom_workspaces"][self.active] = self._snapshot()
        self._loaded()
        self.capture()
        return True

    def revert(self) -> bool:
        """Revert to saved layout: reloads the active custom snapshot."""
        if not self.is_custom():
            return False
        self.activate(self.active, save_outgoing=False)
        return True

    def rename(self, old: str, new: str) -> str:
        if not self.is_custom(old):
            raise WorkspaceNameError("Built-in workspaces can't be renamed.")
        new = validate_name(new, self.custom_names(), current=old)
        customs = self._settings["custom_workspaces"]
        self._settings["custom_workspaces"] = {(new if key == old else key): value for key, value in customs.items()}
        if self.active == old:
            self._settings["active_workspace"] = new
            live = self._settings.get("workspace_live") or {}
            if live.get("workspace") == old:
                live["workspace"] = new
        return new

    def delete(self, name: str) -> bool:
        """Removes a custom snapshot. Deleting the active one loads Advanced,
        so no other workspace's remembered layout is overwritten."""
        if not self.is_custom(name):
            return False
        was_active = self.active == name
        del self._settings["custom_workspaces"][name]
        if was_active:
            self.activate(ADVANCED, save_outgoing=False)
        return True

    # -- modified ----------------------------------------------------------

    def _fingerprint(self) -> tuple:
        docks = self._window._all_docks() if hasattr(self._window, "_all_docks") else []
        return layout_fingerprint(self._window, docks)

    def _loaded(self, sticky: bool = False) -> None:
        """A layout was just applied or saved: its fingerprint is the
        baseline, and the dock sizes count once they settle."""
        self._sticky_modified = sticky
        self._baseline = self._fingerprint()
        self._settling = False
        self.settle_later()

    def settle_later(self) -> None:
        """The dock sizes are about to move without an edit (a load, a
        window resize): compare structure only until `SETTLE_MS` passes,
        then take the sizes as they are. An arrangement that is already
        modified keeps its baseline."""
        if self.is_modified():
            return
        self._settling = True
        try:
            from PySide6.QtCore import QTimer

            if self._settle_timer is None:
                self._settle_timer = QTimer(self._window)
                self._settle_timer.setSingleShot(True)
                self._settle_timer.setInterval(SETTLE_MS)
                self._settle_timer.timeout.connect(self.settle)
            self._settle_timer.start()
        except Exception:
            self.settle()

    def settle(self) -> None:
        """Ends a settle: the baseline takes the current sizes, unless the
        structure changed meanwhile."""
        if not self._settling:
            return
        self._settling = False
        structure, sizes = self._fingerprint()
        if self._baseline is None or self._baseline[0] == structure:
            self._baseline = (structure, sizes)

    def is_modified(self) -> bool:
        """The active custom workspace's live layout differs from its
        snapshot. Always False for a built-in."""
        if not self.is_custom():
            return False
        if self._sticky_modified:
            return True
        if self._baseline is None:
            return False
        structure, sizes = self._fingerprint()
        if structure != self._baseline[0]:
            return True
        if self._settling or not sizes or not self._baseline[1]:
            return False
        return sizes != self._baseline[1]

    # -- screens -----------------------------------------------------------

    def recover_offscreen_docks(self) -> None:
        """A floating dock whose frame is on no screen (a monitor was
        removed) moves onto the window's screen, keeping its size."""
        try:
            from PySide6.QtGui import QGuiApplication
        except Exception:
            return
        screens = QGuiApplication.screens()
        if not screens:
            return
        docks = self._window._all_docks() if hasattr(self._window, "_all_docks") else []
        for dock in docks:
            if not dock.isFloating() or dock.isHidden():
                continue
            frame = dock.frameGeometry()
            if any(screen.availableGeometry().intersects(frame) for screen in screens):
                continue
            screen = self._window.screen() or QGuiApplication.primaryScreen() or screens[0]
            area = screen.availableGeometry()
            width = min(frame.width(), area.width())
            height = min(frame.height(), area.height())
            dock.setGeometry(area.x() + (area.width() - width) // 2, area.y() + (area.height() - height) // 2,
                             width, height)
