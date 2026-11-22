"""The Qt side of FX resolution: reads the docks and the app's settings
and calls `kokoro_gui.core.fx`, where the layers and the `apply_fx` rule
live (see its docstring).

`QtTTSApp._assemble_clip_config` and `FXDock._resolved_values` both call
`resolve_fx` here, so the dock can never show something the transport
doesn't play. The global preset folder is `qt_app_module.FX_PRESETS_DIR`,
read per call (the GUI test fixture patches it). The moved names are
re-exported so the docks' imports stay as they are.
"""
from __future__ import annotations

from typing import Optional

from kokoro_gui.core import fx as core_fx
from kokoro_gui.core.fx import (  # noqa: F401 - re-exported for the docks
    PLACEHOLDER, PLACEHOLDER_LABEL, FxResolution, real_preset_name,
)


def _presets_dir() -> str:
    import kokoro_gui.qt.app as qt_app_module

    return qt_app_module.FX_PRESETS_DIR


def _project_dir(app, project):
    return project.project_dir if project is not None else getattr(app, "project_dir", None)


def load_fx_preset_values(app, name, project=None) -> Optional[dict]:
    """The whitelisted values of the FX preset `name`, project-local first,
    in `project`'s dir (default: the app's focus)."""
    if not real_preset_name(name):
        return None
    return core_fx.load_fx_preset_values(name, _project_dir(app, project), _presets_dir())


def resolve_fx(app, clip=None, character=None, project=None) -> FxResolution:
    """Resolve for a clip (its character is looked up in `project`'s
    document, default the app's focus), for a character alone, or for the
    project when both are None. Reads the Audio FX tab's project values,
    the Settings tab's Apply box and `settings["fx_preset"]`."""
    fx_dock = getattr(app, "fx_dock", None)
    settings_dock = getattr(app, "settings_dock", None)
    project_fx = fx_dock.project_fx_state() if fx_dock is not None else {}
    master_apply = bool(settings_dock.apply_fx_enabled()) if settings_dock is not None else True
    document = project.document if project is not None else app.document
    return core_fx.resolve_fx(
        document, clip, character, project_fx=project_fx, master_apply=master_apply,
        project_preset=app.settings.get("fx_preset"), project_dir=_project_dir(app, project),
        presets_dir=_presets_dir())
