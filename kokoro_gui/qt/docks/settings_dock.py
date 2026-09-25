"""Settings dock: item 2 ("Settings panel rescoping") of the DAW-for-text
redesign's remaining-work roadmap. Renders the schema-driven config fields
(voice/speed/lang_code/segmentation/format/num_threads/caching, plus any
backend-specific groups) and the hand-built Audio Control widgets
(volume/pitch/FX-preset-combo/apply_fx/normalize/trim) that used to live in
`GenerationDock` - now scoped to whatever `self.app.selection` currently
points at, instead of always editing the whole document's defaults.

The Output / Processing Options groups that used to sit here moved to the
Export dialog (`kokoro_gui/qt/docks/export_dialog.py`, section 6 of
Claude/PLAN_ui_shell_redesign.md): they describe the export, not the
selection.

Three states, keyed off `SelectionModel.kind`:

- "none": values come from `self.app.settings` (today's whole-document
  defaults) - the literal migration of what `GenerationDock._build_schema_form`
  used to do. `on_change` just schedules an autosave, same as before.
- "clip": values come from `Document.effective_config_for_clip(clip)`; edits
  write into `clip.overrides` (never `app.settings`) - dirtying falls out for
  free since `dirty.is_clip_dirty` recomputes from `effective_config_for_clip`
  fresh every time, no explicit "mark dirty" call needed anywhere here.
- "character": values come from `character.preset_data`; edits write there
  directly, which every non-overridden clip using that character picks up
  live the next time `effective_config_for_clip` is read.

Only `ALLOWED_PRESET_KEYS` (kokoro_gui/engine/presets.py) can vary per clip/
character - that's exactly voice/speed/volume/pitch/normalize/
trim/format/apply_fx/fx_preset. Every other schema field (lang_code,
num_threads, caching, a backend's own non-preset fields) is rendered
disabled (not hidden) in clip/character mode, via `SchemaFormWidget.widget_for`
- its value there is still sourced from `app.settings`, since that's what
  `_assemble_clip_config` actually uses for those keys regardless of which
  clip is selected.

Below Audio Control, `ScopeFields` (kokoro_gui/qt/docks/scope_fields.py)
shows the project's pacing, crossfade and timecode fields in "none" mode
and the clip's gap, take, status, note and source text in "clip" mode;
it's hidden in "character" mode.

`get_state()` deliberately does NOT reflect whatever mode is currently
rendered: `app.py`'s `_assemble_config`/`_assemble_clip_config` need the
project-wide ("none") defaults unconditionally, no matter what's selected in
this dock's UI at the moment they're called. While "none" is rendered, that's
just this dock's live widgets; while a clip/character is rendered instead,
there's no live "none" widget to read, so the last known "none" values are
cached in `self._none_values` at the moment the dock switches away from
"none" (see `_build_for_selection`).
"""
from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget,
)

import kokoro_gui.qt.app as qt_app_module
from kokoro_gui.engine.presets import ALLOWED_PRESET_KEYS
from kokoro_gui.qt import spec
from kokoro_gui.qt.docks.scope_fields import ScopeFields
from kokoro_gui.qt.schema_form import SchemaFormWidget

# Keys tracked in the internal "none"-state cache/live-widget snapshot that
# are NOT part of get_state()'s public contract (GenerationDock.get_state()'s
# old shape never included these - "apply_fx" has its own apply_fx_enabled()
# accessor, "fx_preset" is only ever used as a display string / preset name).
_INTERNAL_ONLY_KEYS = ("apply_fx", "fx_preset")


class SettingsDock(QDockWidget):
    def __init__(self, app, parent=None):
        super().__init__("Settings", parent)
        self.setObjectName("dock_settings")
        self.app = app

        self.schema_form: SchemaFormWidget | None = None
        self._mode = "none"
        self._target = None
        # Seeded from app.settings up front (not left as {}) - the
        # TranscriptEditor built by GenerationDock (constructed just before
        # this dock) can already have selected a clip by the time this dock
        # is built, e.g. its initial cursor position lands inside a clip
        # loaded from a persisted document - so this dock's very first
        # render may start in "clip"/"character" mode, never having passed
        # through a live "none" render to snapshot from.
        self._none_values: dict = self._project_default_snapshot()
        self._constructing_schema_form = False

        content = QWidget()
        outer = QVBoxLayout(content)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        # --- Schema-driven config (moved from GenerationDock) ---
        self.schema_group = QGroupBox("Configuration")
        self.schema_layout = QVBoxLayout(self.schema_group)
        layout.addWidget(self.schema_group)

        # --- Audio control (volume/pitch/FX preset - hand-built, moved
        # from GenerationDock) ---
        audio_group = QGroupBox("Audio Control")
        audio_form = QFormLayout(audio_group)
        self.volume_spin = QDoubleSpinBox()
        self.volume_spin.setRange(0.1, 2.0)
        self.volume_spin.setSingleStep(0.1)
        audio_form.addRow("Volume:", self.volume_spin)

        self.pitch_spin = QDoubleSpinBox()
        self.pitch_spin.setRange(-12, 12)
        self.pitch_spin.setSingleStep(1)
        audio_form.addRow("Pitch (st):", self.pitch_spin)

        fx_row = QWidget()
        fx_row_layout = QHBoxLayout(fx_row)
        fx_row_layout.setContentsMargins(0, 0, 0, 0)
        self.fx_preset_combo = QComboBox()
        self.apply_fx_check = QCheckBox("Apply")
        fx_row_layout.addWidget(self.fx_preset_combo, 1)
        fx_row_layout.addWidget(self.apply_fx_check)
        audio_form.addRow("FX Preset:", fx_row)

        self.normalize_check = QCheckBox("Normalize")
        self.trim_check = QCheckBox("Trim Silence")
        toggles_row = QWidget()
        toggles_layout = QHBoxLayout(toggles_row)
        toggles_layout.setContentsMargins(0, 0, 0, 0)
        toggles_layout.addWidget(self.normalize_check)
        toggles_layout.addWidget(self.trim_check)
        audio_form.addRow("", toggles_row)
        layout.addWidget(audio_group)

        # Project or clip fields for the current scope (phase 2): pacing,
        # crossfade and timecode; or the clip's gap, take, status, note and
        # source text. Hidden in character scope.
        self.scope_group = QGroupBox("Project")
        scope_layout = QVBoxLayout(self.scope_group)
        self.scope_fields = ScopeFields(self.app)
        scope_layout.addWidget(self.scope_fields)
        layout.addWidget(self.scope_group)

        layout.addStretch(1)
        self.setWidget(content)

        # Hand-built widgets are constructed once and never torn down - only
        # their displayed values change on selection change (see
        # `_refresh_hand_built_display`); their signals are wired once, here.
        self.volume_spin.valueChanged.connect(lambda v: self._on_hand_built_changed("volume", v))
        self.pitch_spin.valueChanged.connect(lambda v: self._on_hand_built_changed("pitch", v))
        self.normalize_check.toggled.connect(lambda v: self._on_hand_built_changed("normalize", v))
        self.trim_check.toggled.connect(lambda v: self._on_hand_built_changed("trim", v))
        self.apply_fx_check.toggled.connect(lambda v: self._on_hand_built_changed("apply_fx", v))
        self.fx_preset_combo.currentTextChanged.connect(self._on_fx_preset_selected)

        self.refresh_fx_presets()
        self._build_for_selection()
        self.app.selection.changed.connect(self._on_selection_changed)

    # --- selection-driven three-state rendering ---------------------------

    def _resolve_mode(self):
        """`(mode, target)` for the current `self.app.selection` - falls
        back to `("none", None)` for a stale clip/character id (already
        removed) or for the "range"/"none" selection kinds, neither of
        which has a clip/character to scope against."""
        kind = self.app.selection.kind
        if kind == "clip":
            clip = self.app.document.get_clip(self.app.selection.selected_clip_id)
            if clip is not None:
                return "clip", clip
        elif kind == "character":
            character = self.app.document.get_character(self.app.selection.selected_character_id)
            if character is not None:
                return "character", character
        return "none", None

    def _on_selection_changed(self) -> None:
        self._build_for_selection()

    def _build_for_selection(self) -> None:
        # Snapshot the outgoing "none" state's live values before switching
        # away from it - there's no live "none" widget to read back from
        # once a clip/character is being rendered instead, but get_state()
        # must keep returning them regardless.
        if self._mode == "none" and self.schema_form is not None:
            self._none_values = self._snapshot_none_values()

        self._mode, self._target = self._resolve_mode()
        self._build_schema_form()
        self._refresh_hand_built_display()
        self.refresh_scope_fields()

    def refresh_scope_fields(self) -> None:
        """Rebuilds the scope group for the current mode. Also called after
        an undo or a generate, which change what the clip fields show."""
        if self._mode == "clip":
            self.scope_group.setTitle("Clip")
            self.scope_fields.build_clip(self._target)
            self.scope_group.show()
        elif self._mode == "none":
            self.scope_group.setTitle("Project")
            self.scope_fields.build_project()
            self.scope_group.show()
        else:
            self.scope_fields.clear()
            self.scope_group.hide()

    # --- schema form (rebuilt on selection change AND on a character's engine change) ---

    def _project_default_snapshot(self) -> dict:
        """A valid `_none_values`-shaped dict sourced purely from
        `app.settings` - used to seed `self._none_values` at construction,
        before this dock has ever necessarily rendered "none" mode live
        (see the comment where it's assigned in `__init__`)."""
        values = self._base_values_from_settings()
        values.update({
            "volume": self.app.settings.get("volume", 1.0),
            "pitch": self.app.settings.get("pitch", 0.0),
            "normalize": self.app.settings.get("normalize", False),
            "trim_silence": self.app.settings.get("trim", False),
            "apply_fx": self.app.settings.get("apply_fx", True),
            "fx_preset": self.app.settings.get("fx_preset", "Select FX Preset..."),
        })
        return values

    def _base_values_from_settings(self) -> dict:
        """Project-wide defaults sourced from `app.settings` - exactly what
        `GenerationDock._build_schema_form` used to build its `values` dict
        from. Also what a disabled (non-ALLOWED_PRESET_KEYS) field shows in
        clip/character mode, since those keys are always sourced from
        project settings regardless of selection (see
        `app.py`'s `_assemble_clip_config`)."""
        return {
            "lang_code": self.app.settings.get("lang_code", "a"),
            "voice": self.app.settings.get("voice", "af_heart"),
            "speed": self.app.settings.get("speed", 1.0),
            **{key: self.app.settings.get(key, spec.SETTINGS_DEFAULTS[key]) for key in spec.SEGMENTATION_KEYS},
            "format": self.app.settings.get("format", "wav"),
            "num_threads": self.app.settings.get("num_threads", 1),
            "caching": self.app.settings.get("caching", True),
        }

    def _current_schema_values(self) -> dict:
        values = self._base_values_from_settings()
        if self._mode == "clip":
            values.update(self.app.document.effective_config_for_clip(self._target))
        elif self._mode == "character":
            values.update(self._target.preset_data)
        return values

    def _build_schema_form(self) -> None:
        if self.schema_form is not None:
            self.schema_layout.removeWidget(self.schema_form)
            self.schema_form.deleteLater()

        schema = self.app.backend.get_config_schema()
        lang_code = self.app.settings.get("lang_code", "a")
        voice_choices = [(v, v) for v in self.app.get_all_voices(lang_code)]
        values = self._current_schema_values()
        # See GenerationDock's former `_build_schema_form` docstring note:
        # "voice" is always GUI-resolved; "lang_code" only if the backend's
        # own schema leaves it choices=None.
        overrides = {"voice": voice_choices}
        lang_field = next((f for f in schema if f.key == "lang_code"), None)
        if lang_field is not None and lang_field.choices is None:
            overrides["lang_code"] = [(label, code) for label, code in spec.LANGUAGES.items()]

        self._constructing_schema_form = True
        try:
            self.schema_form = SchemaFormWidget(
                schema, values,
                choices_overrides=overrides,
                # "lexicon" has its own dedicated widget (LexiconDock's CRUD
                # list); "pitch" has its own dedicated widget too (this
                # dock's hand-built pitch_spin, in "Audio Control") - both
                # backends that declare a schema "pitch" field (kokoro.py,
                # dummy.py) would otherwise render a second, independent
                # pitch control that fights the hand-built one for the same
                # ALLOWED_PRESET_KEYS override slot in clip/character mode.
                skip_keys={"lexicon", "pitch"},
                on_change=self._on_schema_field_changed,
            )
        finally:
            self._constructing_schema_form = False
        self.schema_layout.addWidget(self.schema_form)

        if self._mode != "none":
            for f in schema:
                if f.key in ALLOWED_PRESET_KEYS:
                    continue
                widget = self.schema_form.widget_for(f.key)
                if widget is not None:
                    widget.setEnabled(False)

    def rebuild_schema_form(self) -> None:
        """Called by app.py's `set_character_engine` - re-renders this
        dock's schema fields for the active backend (the selected clip's or
        character's engine, else the first character's), keeping whatever
        clip/character/none mode is currently selected."""
        self._build_schema_form()

    def refresh_voice_choices(self) -> None:
        lang_code = self.schema_form.values().get("lang_code", "a")
        voices = self.app.get_all_voices(lang_code)
        current = self.schema_form.values().get("voice")
        self.schema_form.set_choices("voice", [(v, v) for v in voices], current)

    def _on_schema_field_changed(self, key: str, value) -> None:
        if self._constructing_schema_form:
            return
        if key == "lang_code":
            self.refresh_voice_choices()
        if self._mode == "none":
            self.app.schedule_save()
            if key in spec.SEGMENTATION_KEYS and self.app.editor is not None:
                # New pieces can stale clips; show it now, not on the next edit.
                self.app.editor.rehighlight()
            return
        if key not in ALLOWED_PRESET_KEYS:
            return  # defense in depth - the field is disabled, unreachable via the UI
        if self._target is not None:
            if self._mode == "clip":
                self._target.overrides[key] = value
            elif self._mode == "character":
                self._target.preset_data[key] = value
        self.app.schedule_save()
        self.app.refresh_timeline()

    # --- hand-built widgets (volume/pitch/normalize/trim/apply_fx/fx_preset) --

    def _refresh_hand_built_display(self) -> None:
        if self._mode == "clip":
            cfg = self.app.document.effective_config_for_clip(self._target)
        elif self._mode == "character":
            cfg = self._target.preset_data
        else:
            cfg = {}
        base = {
            "volume": self.app.settings.get("volume", 1.0),
            "pitch": self.app.settings.get("pitch", 0.0),
            "normalize": self.app.settings.get("normalize", False),
            "trim": self.app.settings.get("trim", False),
            "apply_fx": self.app.settings.get("apply_fx", True),
            "fx_preset": self.app.settings.get("fx_preset", "Select FX Preset..."),
        }
        base.update(cfg)

        widgets = (self.volume_spin, self.pitch_spin, self.normalize_check,
                   self.trim_check, self.apply_fx_check, self.fx_preset_combo)
        for w in widgets:
            w.blockSignals(True)
        try:
            self.volume_spin.setValue(base["volume"])
            self.pitch_spin.setValue(base["pitch"])
            self.normalize_check.setChecked(bool(base["normalize"]))
            self.trim_check.setChecked(bool(base["trim"]))
            self.apply_fx_check.setChecked(bool(base["apply_fx"]))
            fx_name = base["fx_preset"] or "Select FX Preset..."
            idx = self.fx_preset_combo.findText(fx_name)
            self.fx_preset_combo.setCurrentIndex(idx if idx >= 0 else 0)
        finally:
            for w in widgets:
                w.blockSignals(False)

    def _on_hand_built_changed(self, key: str, value) -> None:
        if self._mode == "none":
            # Volume/pitch/normalize/trim/apply_fx are read-time
            # post-processing for clips: re-render, nothing to regenerate.
            self.app.schedule_save()
            self.app.refresh_timeline()
            return
        if key not in ALLOWED_PRESET_KEYS:
            return  # defense in depth - every hand-built field is in ALLOWED_PRESET_KEYS today
        if self._target is not None:
            if self._mode == "clip":
                self._target.overrides[key] = value
            elif self._mode == "character":
                self._target.preset_data[key] = value
        self.app.schedule_save()
        self.app.refresh_timeline()

    def _on_fx_preset_selected(self, name: str) -> None:
        if not name or name == "Select FX Preset...":
            return
        if self._mode == "none":
            # "none" mode's FX preset combo actually *loads* the preset's
            # resolved values into the global FX dock - project-wide FX
            # settings are a single shared instance, not per-clip.
            self.app.fx_dock.load_preset(name)
            return
        # clip/character mode only ever stores the preset *name* - resolved
        # into actual FX values later by app.py's _assemble_clip_config.
        if self._target is not None:
            if self._mode == "clip":
                self._target.overrides["fx_preset"] = name
            elif self._mode == "character":
                self._target.preset_data["fx_preset"] = name
        self.app.schedule_save()
        self.app.refresh_timeline()

    # --- state (feeds app._assemble_config) ---

    def _snapshot_none_values(self) -> dict:
        """Everything about the live "none" state worth caching for later -
        a superset of get_state()'s public contract (also carries
        apply_fx/fx_preset, used internally by `apply_fx_enabled()` and the
        legacy generation-preset combo)."""
        state = dict(self.schema_form.values())
        state.update({
            "volume": self.volume_spin.value(),
            "pitch": self.pitch_spin.value(),
            "normalize": self.normalize_check.isChecked(),
            "trim_silence": self.trim_check.isChecked(),
            "apply_fx": self.apply_fx_check.isChecked(),
            "fx_preset": self.fx_preset_combo.currentText(),
        })
        return state

    def get_state(self) -> dict:
        """Always the project-wide ("none") state's values, regardless of
        what's currently rendered - see this module's docstring."""
        src = self._snapshot_none_values() if self._mode == "none" else self._none_values
        return {k: v for k, v in src.items() if k not in _INTERNAL_ONLY_KEYS}

    def apply_fx_enabled(self) -> bool:
        if self._mode == "none":
            return self.apply_fx_check.isChecked()
        return bool(self._none_values.get("apply_fx", True))

    def set_fx_preset_display(self, name: str) -> None:
        """Sets the FX preset combo's displayed text to `name` - used after
        `fx_dock.load_preset`/a generation preset names one - without
        re-triggering `_on_fx_preset_selected`."""
        if self._mode == "none":
            self.fx_preset_combo.blockSignals(True)
            self.fx_preset_combo.setCurrentText(name)
            self.fx_preset_combo.blockSignals(False)
        else:
            self._none_values["fx_preset"] = name

    # --- FX preset combo mirror (kept in sync with the FX dock's own combo) --

    def refresh_fx_presets(self) -> None:
        presets = ["Select FX Preset..."]
        if os.path.exists(qt_app_module.FX_PRESETS_DIR):
            files = [f for f in os.listdir(qt_app_module.FX_PRESETS_DIR) if f.endswith(".json")]
            presets.extend(f[:-5] for f in files)
        self.fx_preset_combo.blockSignals(True)
        self.fx_preset_combo.clear()
        self.fx_preset_combo.addItems(presets)
        self.fx_preset_combo.setCurrentText("Select FX Preset...")
        self.fx_preset_combo.blockSignals(False)
