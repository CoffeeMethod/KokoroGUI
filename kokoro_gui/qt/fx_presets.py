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

The impulse-response store for the convolution reverb (grill Q31) lives
here too: `list_ir_names` for the dock's combo, `resolve_ir` for the file a
name plays from, `import_ir_file` to add a wav to `presets/fx/ir/`. The
resolver itself is `kokoro_gui.engine.presets.resolve_ir`, since
`engine/audio_fx.py` can't import from `kokoro_gui.qt`.
"""
from __future__ import annotations

import filecmp
import os
import re
import shutil

from kokoro_gui.engine import presets as engine_presets


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


def _global_ir_dir() -> str:
    import kokoro_gui.qt.app as qt_app_module

    return os.path.join(qt_app_module.FX_PRESETS_DIR, "ir")


def resolve_ir(name, project_dir: str | None = None) -> str | None:
    """The file impulse response `name` plays from (grill Q31):
    `<project_dir>/fx/ir/<name>.wav` first, then `presets/fx/ir/<name>.wav`,
    else None. `kokoro_gui.engine.presets.resolve_ir` with the global store
    read from `FX_PRESETS_DIR` the way `list_fx_preset_names` reads it."""
    return engine_presets.resolve_ir(name, project_dir, _global_ir_dir())


def list_ir_names(project_dir: str | None = None) -> list[str]:
    """Every impulse response's name, sorted: the project's `fx/ir/*.wav`
    plus `presets/fx/ir/*.wav`, the union."""
    return engine_presets.list_ir_names(project_dir, _global_ir_dir())


def import_ir_file(src_path: str) -> str:
    """Copies the wav at `src_path` into the global IR store
    (`presets/fx/ir/`) and returns the name it's stored under: the file's
    stem with path and reserved characters removed. A different file already
    stored under that name keeps it, and the new one gets a numbered name
    (`Hall 2`). Raises ValueError for a file soundfile can't read as audio
    or one longer than the chain would load (`audio_fx.MAX_IR_SECONDS`),
    OSError for a failed copy."""
    import soundfile as sf

    from kokoro_gui.engine.audio_fx import MAX_IR_SECONDS

    src = os.path.abspath(src_path)
    try:
        info = sf.info(src)
    except Exception as e:  # noqa: BLE001 - soundfile raises several types for a bad file
        raise ValueError(f"not a readable audio file: {e}") from e
    if info.samplerate <= 0 or info.frames / info.samplerate > MAX_IR_SECONDS:
        raise ValueError(f"it is longer than {MAX_IR_SECONDS:g} seconds")
    stem = os.path.splitext(os.path.basename(src))[0]
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]', "", stem).strip() or "impulse"
    directory = _global_ir_dir()
    os.makedirs(directory, exist_ok=True)
    root = os.path.realpath(directory)
    name = base
    counter = 2
    while True:
        target = os.path.realpath(os.path.join(root, f"{os.path.basename(name)}.wav"))
        if not target.startswith(root + os.sep):
            raise ValueError(f"invalid impulse response name {name!r}")
        if not os.path.exists(target):
            break
        if filecmp.cmp(src, target, shallow=False):
            return name
        name = f"{base} {counter}"
        counter += 1
    shutil.copyfile(src, target)
    return name
