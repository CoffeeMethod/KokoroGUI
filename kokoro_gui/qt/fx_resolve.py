"""The one place a clip's (or character's) FX stack is resolved.

Layers, lowest first: the Audio FX tab's project values, the character's
attached `fx_preset` file, the clip's own `overrides["fx_preset"]`, then
`clip.fx_override` (resolved values, set by the tab in clip scope or by the
timeline's FX menu). `QtTTSApp._assemble_clip_config` generates and
post-processes with the result and `FXDock._resolved_values` renders it, so
the dock can never show something the transport doesn't play.

`apply_fx` is the project master switch (the Settings tab's "Apply" box)
ANDed with the scope's own value: a character's `preset_data["apply_fx"]`,
overridden by an explicit `clip.overrides["apply_fx"]`. A clip that carries
an `fx_override` counts as FX-on for that clip unless it also carries an
explicit `overrides["apply_fx"]` of False; applying FX to a clip and then
hearing nothing because its character's preset says off would be a puzzle,
not a feature.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from kokoro_gui.engine.presets import filter_fx_preset_values

PLACEHOLDER = "Select FX Preset..."


@dataclass
class FxResolution:
    values: dict = field(default_factory=dict)
    preset_name: Optional[str] = None  # the name the scope resolves to; "custom" for an fx_override
    apply_fx: bool = True


def real_preset_name(name) -> Optional[str]:
    """`name` unless it's empty or the combo placeholder that older
    `preset_data`/`overrides` dicts still carry as `fx_preset`."""
    return name if name and name != PLACEHOLDER else None


def load_fx_preset_values(app, name, project=None) -> Optional[dict]:
    """The whitelisted values of `presets/fx/<name>.json`, via the engine
    first, then the file directly (tests stub `load_fx_preset` to None).
    Project-local first, in `project`'s dir (default: the app's focus)."""
    name = real_preset_name(name)
    if not name:
        return None
    project_dir = project.project_dir if project is not None else getattr(app, "project_dir", None)
    preset = app.engine.load_fx_preset(name, project_dir)
    if not preset:
        import kokoro_gui.qt.app as qt_app_module

        safe = os.path.basename(name)
        candidates = [os.path.join(qt_app_module.FX_PRESETS_DIR, f"{safe}.json")]
        if project_dir:
            candidates.insert(0, os.path.join(project_dir, "fx", f"{safe}.json"))
        fpath = next((c for c in candidates if os.path.exists(c)), candidates[-1])
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    preset = json.load(fh)
            except Exception:
                preset = None
    return filter_fx_preset_values(preset) if preset else None


def resolve_fx(app, clip=None, character=None, project=None) -> FxResolution:
    """Resolve for a clip (its character is looked up in `project`'s
    document, default the app's focus), for a character alone, or for the
    project when both are None."""
    fx_dock = getattr(app, "fx_dock", None)
    settings_dock = getattr(app, "settings_dock", None)
    values = dict(fx_dock.project_fx_state()) if fx_dock is not None else {}
    apply_fx = bool(settings_dock.apply_fx_enabled()) if settings_dock is not None else True
    # The project preset's name is only the answer at project scope; a
    # character or clip without a preset of its own shows none (placeholder).
    preset_name = real_preset_name(app.settings.get("fx_preset")) if clip is None and character is None else None

    document = project.document if project is not None else app.document
    if clip is not None and character is None:
        character = document.get_character(clip.character_id)

    scope_apply = True
    if character is not None:
        name = real_preset_name(character.preset_data.get("fx_preset"))
        preset = load_fx_preset_values(app, name, project)
        if preset:
            values.update(preset)
        if name:
            preset_name = name
        scope_apply = bool(character.preset_data.get("apply_fx", True))

    if clip is not None:
        own_name = real_preset_name(clip.overrides.get("fx_preset"))
        own = load_fx_preset_values(app, own_name, project)
        if own:
            values.update(own)
        if own_name:
            preset_name = own_name
        if clip.fx_override:
            values.update(filter_fx_preset_values(clip.fx_override))
            scope_apply = True
            if not own_name:
                preset_name = "custom"
        if "apply_fx" in clip.overrides:
            scope_apply = bool(clip.overrides["apply_fx"])

    return FxResolution(values=values, preset_name=preset_name, apply_fx=apply_fx and scope_apply)
