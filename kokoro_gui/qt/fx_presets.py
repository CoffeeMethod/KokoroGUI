"""Shared FX-preset-name listing (`presets/fx/*.json`), factored out so both
`kokoro_gui.qt.docks.fx_dock.FXDock` (its preset combo) and
`kokoro_gui.qt.timeline_view.TimelineView` (item 5's per-clip FX button menu)
glob the same directory the same way, instead of duplicating the logic.

Deliberately its own sibling module rather than living inside
`kokoro_gui/qt/docks/fx_dock.py` (as a first-pass reading of item 5's plan
suggests) or being imported by `timeline_view.py` from anywhere under
`kokoro_gui.qt.docks`: that package's `__init__.py` imports
`generation_dock.py` first, which does `import kokoro_gui.qt.app as
qt_app_module` - and `app.py` in turn does `from kokoro_gui.qt.docks import
(...)`, needing every dock class already bound. That round-trip only
resolves today because `app.py` is always the *first* module touched in
every real import chain (every dock's `import kokoro_gui.qt.app as
qt_app_module` is a safe, alias-only reference to a still-initializing
module). `timeline_view.py` is imported standalone by its own test file
(no `app.py` involved at all) - if it reached into `kokoro_gui.qt.docks.
fx_dock` directly, that would force `kokoro_gui/qt/docks/__init__.py` to run
before `app.py` has ever been touched, hitting exactly that unresolved
circular import. A plain sibling module (no dependency on the `docks`
package, and no *module-level* dependency on `app.py` either - see below)
sidesteps the whole problem.
"""
from __future__ import annotations

import os


def list_fx_preset_names(project_dir: str | None = None) -> list[str]:
    """Every FX preset's name (no extension), sorted: the open project's
    `fx/` (grill TB3, project-local first) plus `presets/fx/*.json`.

    Reads `kokoro_gui.qt.app.FX_PRESETS_DIR` via a *local* import inside this
    function (not a module-level one) purely to avoid this module ever
    triggering `app.py`'s import at *this* module's own import time - by the
    time any caller actually invokes this function, `app.py` is always
    already fully loaded (both `FXDock` and `TimelineView` are only ever
    constructed after it is). A test's
    `monkeypatch.setattr(qt_app_module, "FX_PRESETS_DIR", ...)` is still
    honored either way, since the attribute is read fresh on every call.
    """
    import kokoro_gui.qt.app as qt_app_module

    names = set()
    dirs = [qt_app_module.FX_PRESETS_DIR]
    if project_dir:
        dirs.insert(0, os.path.join(project_dir, "fx"))
    for directory in dirs:
        if os.path.isdir(directory):
            names.update(f[:-5] for f in os.listdir(directory) if f.endswith(".json"))
    return sorted(names)
