"""Named dock layouts ("workspaces") for the Qt shell - UI1/UI8 of
Claude/PLAN_ui_shell_redesign.md.

`config_qt.json` holds `"workspaces": {name: {"state": b64, "geometry":
b64}}` plus `"active_workspace"`. Two names have programmatic defaults:

- Advanced: the drawing's 2x2 grid (transcript | settings tabs over
  timeline | transport). Built by `QtTTSApp.arrange_docks_default()`.
- Simple: the same grid with the timeline dock hidden, so the transcript
  takes the whole left column. Layout only - same document, same Generate
  behavior (UI8).

Choosing a workspace restores its saved state if the user has ever dragged
something while it was active, else the programmatic default. Reset
rebuilds the default for the active one and forgets the saved edits. The
app's existing debounced autosave calls `capture()` so drag edits land in
the active entry.

The pre-workspace `dock_state`/`geometry` keys migrate into
`workspaces.Advanced` the first time this loads them, then get dropped.
"""
from __future__ import annotations

from kokoro_gui.qt import settings as qt_settings

ADVANCED = "Advanced"
SIMPLE = "Simple"
WORKSPACE_NAMES = (ADVANCED, SIMPLE)


class WorkspaceManager:
    def __init__(self, window, settings: dict):
        self._window = window
        self._settings = settings
        self._migrate_legacy_keys()
        self._settings.setdefault("workspaces", {})
        if self._settings.get("active_workspace") not in WORKSPACE_NAMES:
            self._settings["active_workspace"] = ADVANCED

    # -- persistence shape -------------------------------------------------

    def _migrate_legacy_keys(self) -> None:
        legacy_state = self._settings.pop("dock_state", None)
        legacy_geometry = self._settings.pop("geometry", None)
        if not legacy_state and not legacy_geometry:
            return
        workspaces = self._settings.setdefault("workspaces", {})
        if ADVANCED not in workspaces:
            workspaces[ADVANCED] = {"state": legacy_state, "geometry": legacy_geometry}

    @property
    def active(self) -> str:
        return self._settings.get("active_workspace", ADVANCED)

    def saved(self, name: str) -> dict | None:
        entry = self._settings.get("workspaces", {}).get(name)
        return entry if isinstance(entry, dict) else None

    # -- apply / capture ---------------------------------------------------

    def restore_on_launch(self) -> None:
        """Called once after the docks exist: geometry from the active
        entry, then the layout (saved or default)."""
        entry = self.saved(self.active)
        if entry and entry.get("geometry"):
            try:
                self._window.restoreGeometry(qt_settings.decode_bytes(entry["geometry"]))
            except Exception:
                pass
        self.activate(self.active, save_outgoing=False)

    def activate(self, name: str, save_outgoing: bool = True) -> None:
        if name not in WORKSPACE_NAMES:
            name = ADVANCED
        if save_outgoing:
            self.capture()
        self._settings["active_workspace"] = name
        entry = self.saved(name)
        restored = False
        if entry and entry.get("state"):
            try:
                restored = bool(self._window.restoreState(qt_settings.decode_bytes(entry["state"])))
            except Exception:
                restored = False
        if not restored:
            self.apply_default(name)

    def apply_default(self, name: str) -> None:
        self._window.arrange_docks_default()
        timeline = getattr(self._window, "timeline_dock", None)
        if timeline is not None:
            timeline.setVisible(name != SIMPLE)
        if name == SIMPLE and hasattr(self._window, "apply_simple_proportions"):
            self._window.apply_simple_proportions()

    def reset(self) -> None:
        """Forget the active workspace's saved edits and rebuild its
        programmatic default."""
        self._settings.setdefault("workspaces", {}).pop(self.active, None)
        self.apply_default(self.active)

    def capture(self) -> None:
        """Snapshot the live layout into the active entry."""
        workspaces = self._settings.setdefault("workspaces", {})
        workspaces[self.active] = {
            "state": qt_settings.encode_bytes(self._window.saveState()),
            "geometry": qt_settings.encode_bytes(self._window.saveGeometry()),
        }
