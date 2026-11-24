"""The Voice & Clip dock (grill PG30; `dock_settings` stays its object name, so
saved layouts restore): item 2 ("Settings panel rescoping") of the DAW-for-text
redesign's remaining-work roadmap. Renders the schema-driven voice fields
(voice/speed/lang_code, plus a backend's "Model" group) and the hand-built
Audio Control widgets
(volume/pitch/FX-preset-combo/apply_fx/normalize/trim) that used to live in
`GenerationDock` - now scoped to whatever `self.app.selection` currently
points at, instead of always editing the whole document's defaults.

Only what shapes a voice lives here. The program fields
(`spec.is_program_field`: segmentation, output format, every "Advanced"
field) are in Options > Settings... (kokoro_gui/qt/settings_window.py), and
so are the project fields. `get_state()`
reads the shared ones among them from `app.settings`.

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
trim/format/apply_fx/fx_preset. Every other schema field shown here
(lang_code, a backend's "Model" group) is rendered
disabled (not hidden) in clip/character mode, via `SchemaFormWidget.widget_for`
- its value there is still the project's, since that's what
  `_assemble_clip_config` actually uses for those keys regardless of which
  clip is selected.

A schema field is shared by every engine (`engines.base.SHARED_CONFIG_KEYS`:
speed, pitch, segmentation, format, caching) or kept per engine (grill EN5:
the default voice, `lang_code`, `num_threads`, a backend's "Model" group). Shared
values come from `app.settings`; per-engine ones from
`app.engine_settings(<the shown engine>)`, and a "none"-mode edit to one
writes straight into that engine's bucket (`app.set_engine_setting`).
`get_state()` carries only the shared keys.

Above the form, an Engine row (grill EN1, UI16), so an engine and its
voice are picked in one place. In "character" mode it sets the character's
engine (`app.set_character_engine`, through `QTimer.singleShot(0, ...)`
since that rebuilds this dock's form); in "clip" mode it does the same for
the clip's character. In "none" mode it's "Engine for new characters",
`settings["default_engine"]`, and the form below shows that engine's
defaults (`shown_backend()`), which is what a new character gets.
Character-scope edits go through `app.commit_character_edit`, so a linked
character reaches its library entry the way Edit > Characters' edits do.

Above the scroll area a `ScopeHeader` (`qt/scope_target.py`) names what the
tab edits, and a hint under it says who else an edit reaches. Both follow
`refresh_target()`, which the app calls from `refresh_timeline`, so a rename
or a reassigned character shows without a new selection. In clip mode a value
set on the clip itself is marked "(clip)" in bold at its label
(`_refresh_override_marks`); an unmarked one is inherited, and its label's
tooltip says from where.

Below Audio, `ScopeFields` (kokoro_gui/qt/docks/scope_fields.py)
shows the clip's gap, take, status, note and source text in "clip" mode;
it's hidden in the other two.

Several clips selected (plan 35, grill PG11 (a)) is still "clip" mode, with
`self._target` the primary clip and `self._targets` all of them. Each field
shows the value the clips share, or mixed (a blank spin, a "(mixed)" combo
entry, a partly checked box), and an edit writes that one override on every
selected clip as one `CompositeCommand`. When the clips' characters use
different engines, the per-engine fields (the voice) are disabled; when
the characters differ, so is the Engine row.

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

from PySide6.QtCore import QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QComboBox, QDockWidget, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget,
)

import kokoro_gui.qt.app as qt_app_module
from kokoro_gui.daw.undo import CompositeCommand, SetFieldCommand
from kokoro_gui.engine.presets import ALLOWED_PRESET_KEYS
from kokoro_gui.engines.base import SHARED_CONFIG_KEYS
from kokoro_gui.engines.registry import DEFAULT_ENGINE_ID
from kokoro_gui.qt import fx_resolve, scope_target, spec
from kokoro_gui.qt.docks.scope_fields import (
    MIXED, MIXED_LABEL, MixedCheckBox, MixedSpinBox, ScopeFields, common_value, mark_mixed,
)
from kokoro_gui.qt.schema_form import SchemaFormWidget

# Keys tracked in the internal "none"-state cache/live-widget snapshot that
# are NOT part of get_state()'s public contract (GenerationDock.get_state()'s
# old shape never included these - "apply_fx" has its own apply_fx_enabled()
# accessor, "fx_preset" is only ever used as a display string / preset name).
_INTERNAL_ONLY_KEYS = ("apply_fx", "fx_preset")

# The hand-built Audio Control values get_state() carries next to the shared
# schema keys.
_HAND_BUILT_STATE_KEYS = ("volume", "pitch", "normalize", "trim_silence")

TITLE = "Voice & Clip"
# The schema group that is the Voice box itself, and the ones that fold.
VOICE_GROUP = "Generation"
MODEL_GROUP = "Model"
# What each hand-built field is called in the "Set on this clip" hint.
HAND_BUILT_LABELS = {"volume": "Volume", "pitch": "Pitch", "fx_preset": "FX", "apply_fx": "Enable FX",
                     "normalize": "Normalize", "trim": "Trim Silence"}
ENABLE_FX_TIPS = {
    "none": "The main switch. Off, no FX run, whatever a character or clip is set to.",
    "character": "Run the FX on this character's clips. The main switch is Enable FX with nothing selected.",
    "clip": "Run the FX on this clip. The main switch is Enable FX with nothing selected.",
}
BUSY_FIELD_TIP = "Off while a generate runs."


class SettingsDock(QDockWidget):
    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.setObjectName("dock_settings")
        self.app = app
        # Not `isFloating()`, which is true of a dock no window has taken yet.
        self._apply_title(False)
        self.topLevelChanged.connect(self._apply_title)

        self.schema_form: SchemaFormWidget | None = None
        self._mode = "none"
        self._target = None
        # Every selected clip in "clip" mode, the primary (`_target`) last.
        self._targets: list = []
        # Seeded from app.settings up front (not left as {}) - the
        # TranscriptEditor built by GenerationDock (constructed just before
        # this dock) can already have selected a clip by the time this dock
        # is built, e.g. its initial cursor position lands inside a clip
        # loaded from a persisted document - so this dock's very first
        # render may start in "clip"/"character" mode, never having passed
        # through a live "none" render to snapshot from.
        self._none_values: dict = self._project_default_snapshot()
        self._constructing_schema_form = False
        # Which schema groups are folded; the form is rebuilt per selection.
        self._collapsed_groups: dict = {MODEL_GROUP: True}

        content = QWidget()
        outer = QVBoxLayout(content)

        # What the tab edits, above the scroll area so it never scrolls away.
        # `subproject_label`, phase 4 (NP1): which subproject these settings
        # edit, when the docks show one the timeline isn't in.
        self.scope_header = scope_target.ScopeHeader()
        self.subproject_label = self.scope_header.context_label
        outer.addWidget(self.scope_header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        # Who else an edit reaches, and what is set on the clip itself.
        self.scope_hint = scope_target.muted_label()
        layout.addWidget(self.scope_hint)

        # --- Voice: the Engine row and the schema fields (moved from
        # GenerationDock) ---
        self.schema_group = QGroupBox("Voice")
        self.schema_layout = QVBoxLayout(self.schema_group)
        layout.addWidget(self.schema_group)

        # --- Engine (grill EN1) ---
        self.engine_row = engine_row = QWidget()
        engine_form = QFormLayout(engine_row)
        engine_form.setContentsMargins(0, 0, 0, 0)
        self.engine_label = QLabel("Engine:")
        self.engine_combo = QComboBox()
        for label, engine_id in self.app.engine_choices():
            self.engine_combo.addItem(label, engine_id)
        self.engine_combo.activated.connect(lambda _i: self._on_engine_picked())
        engine_form.addRow(self.engine_label, self.engine_combo)
        self.schema_layout.addWidget(engine_row)
        # The Engine row edits the character even in clip scope.
        self.engine_hint = scope_target.muted_label()
        self.engine_hint.hide()
        self.schema_layout.addWidget(self.engine_hint)

        self.reference_voice_label = QLabel()
        self.reference_voice_label.setWordWrap(True)
        self.reference_voice_label.hide()
        self.schema_layout.addWidget(self.reference_voice_label)

        # --- Audio control (volume/pitch/FX preset - hand-built, moved
        # from GenerationDock) ---
        self.audio_group = audio_group = QGroupBox("Audio")
        audio_form = QFormLayout(audio_group)
        self.volume_spin = MixedSpinBox()
        self.volume_spin.setRange(0.1, 2.0)
        self.volume_spin.setSingleStep(0.1)
        audio_form.addRow("Volume:", self.volume_spin)

        self.pitch_spin = MixedSpinBox()
        self.pitch_spin.setRange(-12, 12)
        self.pitch_spin.setSingleStep(1)
        audio_form.addRow("Pitch (st):", self.pitch_spin)

        fx_row = QWidget()
        fx_row_layout = QHBoxLayout(fx_row)
        fx_row_layout.setContentsMargins(0, 0, 0, 0)
        self.fx_preset_combo = QComboBox()
        self.apply_fx_check = MixedCheckBox(HAND_BUILT_LABELS["apply_fx"])
        fx_row_layout.addWidget(self.fx_preset_combo, 1)
        fx_row_layout.addWidget(self.apply_fx_check)
        audio_form.addRow("FX:", fx_row)

        self.normalize_check = MixedCheckBox("Normalize")
        self.trim_check = MixedCheckBox("Trim Silence")
        toggles_row = QWidget()
        toggles_layout = QHBoxLayout(toggles_row)
        toggles_layout.setContentsMargins(0, 0, 0, 0)
        toggles_layout.addWidget(self.normalize_check)
        toggles_layout.addWidget(self.trim_check)
        audio_form.addRow("", toggles_row)
        layout.addWidget(audio_group)
        # The widget whose text shows a hand-built field's name.
        self._hand_built_labels = {
            "volume": audio_form.labelForField(self.volume_spin), "pitch": audio_form.labelForField(self.pitch_spin),
            "fx_preset": audio_form.labelForField(fx_row), "apply_fx": self.apply_fx_check,
            "normalize": self.normalize_check, "trim": self.trim_check,
        }

        # The clip's gap, take, status, note and source text (phase 2).
        # Shown in clip scope only.
        self.scope_group = QGroupBox("Clip")
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
        self.fx_preset_combo.currentIndexChanged.connect(self._on_fx_preset_index)

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
        self._targets = self._selected_clips() if self._mode == "clip" else []
        self._build_schema_form()
        self._refresh_hand_built_display()
        self.refresh_scope_fields()
        self.refresh_engine_row()
        self._shown_signature = self._target_signature()
        self._refresh_scope_text()

    # --- what the tab edits, in words --------------------------------------

    def _apply_title(self, floating: bool) -> None:
        """Docked, the title is drawn with mnemonics (title bar, tab), so the
        ampersand is doubled; a floating window with the system's own title
        bar shows it as typed. The Panels entry is a menu item either way."""
        doubled = TITLE.replace("&", "&&")
        system_bar = floating and QGuiApplication.platformName() != "xcb"
        self.setWindowTitle(TITLE if system_bar else doubled)
        self.toggleViewAction().setText(doubled)

    def _target_signature(self) -> tuple:
        """What the fields were built for: the target and the characters its
        inherited values come from."""
        document = self.app.document
        mode, target = self._resolve_mode()
        if mode != "clip":
            return (id(document), mode, getattr(target, "id", None))
        ids = [cid for cid in self.app.selection.selected_clip_ids if document.get_clip(cid) is not None]
        return (id(document), mode, target.id, tuple((cid, document.get_clip(cid).character_id) for cid in ids))

    def refresh_target(self) -> None:
        """The header and hints again, after an edit elsewhere (a rename, an
        undo). When the target is gone or has another character the fields
        follow too."""
        if self.schema_form is None:
            return
        if self._target_signature() != self._shown_signature:
            self._build_for_selection()
            return
        self._refresh_scope_text()

    def _refresh_scope_text(self) -> None:
        self.scope_header.show_target(scope_target.describe(self.app, self._mode, self._target, self._targets))
        self._refresh_override_marks()
        scope_target.set_hint(self.scope_hint, self._scope_hint_text())
        scope_target.set_hint(self.engine_hint, self._engine_hint_text())
        self.apply_fx_check.setToolTip(ENABLE_FX_TIPS[self._mode])

    def _scope_hint_text(self) -> str:
        if self._mode == "character":
            name = scope_target.shorten(self._target.name)
            note = scope_target.library_note(self.app, self._target)
            return f"Edits reach every clip of {name} that doesn't set its own value. {note}".strip()
        if self.several:
            return f"An edit sets that value on all {len(self._targets)} clips, as one undo step."
        if self._mode == "clip":
            source = self._inherited_from()
            names = self._override_names()
            if not names:
                return f"Nothing is set on this clip, so every value follows {source}. An edit here changes this clip only."
            return f"Set on this clip: {scope_target.join_names(names)}. The rest follows {source}."
        shared = "shared by every project"
        if self.app.selection.kind == "range":
            return f"The selected text starts outside a clip, so edits change the defaults, {shared}."
        if self.app.selection.kind == "clip" and self.scope_header.context_label.text():
            # A subproject's block: the docks show its document.
            return f"Select a clip inside the subproject to edit it. Edits here change the defaults, {shared}."
        return "Shared by every project. A character or clip that sets its own value keeps it."

    def _engine_hint_text(self) -> str:
        character = self._scope_character() if self._mode == "clip" else None
        if character is None:
            return ""
        name = scope_target.shorten(character.name)
        note = scope_target.library_note(self.app, character)
        return f"Engine changes the character {name}, so every clip of {name} follows. {note}".strip()

    def _inherited_from(self) -> str:
        character = self.app.document.get_character(self._target.character_id)
        return "the defaults" if character is None else f"{scope_target.shorten(character.name)} and the defaults"

    def _schema_labels(self) -> dict:
        """key -> field label, for the schema fields a clip can set."""
        return {f.key: f.label for f in self.shown_backend().get_config_schema()
                if f.key in ALLOWED_PRESET_KEYS and self.schema_form.widget_for(f.key) is not None}

    def _override_names(self) -> list:
        labels = {**self._schema_labels(), **HAND_BUILT_LABELS}
        return [label for key, label in labels.items() if key in (self._target.overrides or {})]

    def _refresh_override_marks(self) -> None:
        """Marks the values set on the selected clip (or on some or all of the
        selected clips) at their labels, in words and in bold; the others'
        tooltips say where they come from."""
        rows = [(key, label, self.schema_form.label_for(key)) for key, label in self._schema_labels().items()]
        rows += [(key, HAND_BUILT_LABELS[key], widget) for key, widget in self._hand_built_labels.items()]
        for key, name, widget in rows:
            if widget is None:
                continue
            mark, tip = "", ""
            if self._mode == "clip":
                count = sum(key in (clip.overrides or {}) for clip in self._targets)
                if count and count == len(self._targets):
                    mark, tip = (" (clips)" if self.several else " (clip)"), "Set on the selected clip"
                    tip += "s." if self.several else "."
                elif count:
                    mark, tip = " (some clips)", f"Set on {count} of the {len(self._targets)} selected clips."
                else:
                    character = self._scope_character()
                    from_character = character is not None and key in character.preset_data
                    tip = f"From {scope_target.shorten(character.name)}." if from_character else "From the defaults."
                    if self.several and character is None:
                        tip = "From each clip's character, or the defaults."
            elif self._mode == "character":
                tip = "Set on this character." if key in self._target.preset_data else "From the defaults."
            unit = " (st)" if key == "pitch" else ""
            if isinstance(widget, QLabel):
                # A schema label never had the colon the hand-built rows have.
                colon = ":" if key in HAND_BUILT_LABELS else ""
                widget.setText(f"{name}{unit}{mark}{colon}")
                widget.setToolTip(tip)
            else:
                widget.setText(f"{name}{mark}")
                if key != "apply_fx":
                    widget.setToolTip(tip)
            if bool(widget.property("overridden")) != bool(mark):
                # Bold comes from the stylesheet, which beats `setFont` here.
                widget.setProperty("overridden", bool(mark))
                widget.style().unpolish(widget)
                widget.style().polish(widget)

    def set_generating(self, generating: bool) -> None:
        """Volume and pitch are fixed while a generate runs."""
        for spin in (self.volume_spin, self.pitch_spin):
            spin.setEnabled(not generating)
            spin.setToolTip(BUSY_FIELD_TIP if generating else "")

    def _selected_clips(self) -> list:
        """The selected clips that still exist, the primary (`_target`) last."""
        document = self.app.document
        clips = [c for c in (document.get_clip(cid) for cid in self.app.selection.selected_clip_ids)
                 if c is not None and c is not self._target]
        return clips + [self._target]

    @property
    def several(self) -> bool:
        """Several clips selected (plan 35)."""
        return self._mode == "clip" and len(self._targets) > 1

    def _clip_engine_id(self, clip) -> str:
        character = self.app.document.get_character(clip.character_id)
        return (character.backend_id if character is not None else None) or DEFAULT_ENGINE_ID

    def _engines_differ(self) -> bool:
        return self.several and common_value(self._targets, self._clip_engine_id) is MIXED

    # --- the Engine row (grill EN1, UI16) -------------------------------------

    def _scope_character(self):
        """The character the Engine row switches: the selected one, or the
        selected clip's (the clips' one character with several selected); None
        in "none" mode, for a clip without one, or for clips of different
        characters."""
        if self._mode == "character":
            return self._target
        if self._mode == "clip":
            if self.several and common_value(self._targets, lambda c: c.character_id) is MIXED:
                return None
            return self.app.document.get_character(self._target.character_id)
        return None

    def shown_backend(self):
        """The engine whose fields the form shows: in "none" mode the engine
        for new characters (its defaults are what a new character gets),
        else the active character's."""
        if self._mode == "none":
            backend = self.app._backend_for(self.app.default_engine_id)
            if backend is not None:
                return backend
        return self.app.backend

    def refresh_engine_row(self) -> None:
        """Shows the engine for the current scope; see the module docstring."""
        if self._mode in ("character", "clip"):
            character = self._scope_character()
            engine_id = (character.backend_id if character is not None else None) or DEFAULT_ENGINE_ID
            label, enabled = "Engine:", character is not None
            tip = ("The engine this character speaks with." if self._mode == "character"
                   else "The engine this clip's character speaks with. Every clip of the character follows.")
            if self.several and character is None:
                shared = common_value(self._targets, self._clip_engine_id)
                engine_id = None if shared is MIXED else shared
                tip = ("The selected clips have different characters. "
                       "Select one character's clips to change its engine.")
            elif character is None:
                tip = "This clip has no character. Assign one to pick its engine."
        else:
            engine_id, label, enabled, tip = (self.app.default_engine_id, "Engine for new characters:", True,
                                              "The engine a new character gets. The fields below are its defaults.")
        self.engine_label.setText(label)
        self.engine_combo.setEnabled(enabled)
        self.engine_combo.setToolTip(tip)
        self.engine_combo.blockSignals(True)
        try:
            index = self.engine_combo.findData(engine_id) if engine_id is not None else -1
            if index < 0 and engine_id is not None:
                self.engine_combo.addItem(engine_id, engine_id)
                index = self.engine_combo.count() - 1
            self.engine_combo.setCurrentIndex(index)
        finally:
            self.engine_combo.blockSignals(False)

    def _on_engine_picked(self) -> None:
        engine_id = self.engine_combo.currentData()
        if self._mode == "none":
            if engine_id != self.app.default_engine_id:
                self.app.set_default_engine(engine_id)
                self._build_schema_form()
                self._refresh_scope_text()
            return
        character = self._scope_character()
        if character is not None:
            # Switching rebuilds this dock's form, so not from inside the
            # combo's own signal.
            QTimer.singleShot(0, lambda: self.pick_character_engine(character, engine_id))

    def pick_character_engine(self, character, engine_id: str) -> bool:
        """Sets `character`'s engine; on a refusal (a job running) the row
        goes back to the engine it has."""
        ok = False
        if engine_id and engine_id != (character.backend_id or DEFAULT_ENGINE_ID):
            ok = self.app.set_character_engine(character, engine_id)
        self.refresh_engine_row()
        return ok

    def refresh_scope_fields(self) -> None:
        """Rebuilds the clip fields in clip scope. Also called after an undo
        or a generate, which change what they show."""
        if self._mode == "clip":
            self.scope_fields.build_clips(self._targets or [self._target])
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
            "fx_preset": self.app.settings.get("fx_preset", fx_resolve.PLACEHOLDER),
        })
        return values

    def _base_values_from_settings(self) -> dict:
        """Project-wide defaults sourced from `app.settings` - exactly what
        `GenerationDock._build_schema_form` used to build its `values` dict
        from. Also what a disabled (non-ALLOWED_PRESET_KEYS) field shows in
        clip/character mode, since those keys are always sourced from
        project settings regardless of selection (see
        `app.py`'s `_assemble_clip_config`)."""
        backend = self.shown_backend()
        return {
            **self.app.engine_settings(backend.id),
            "voice": self.app.default_voice(backend),
            "speed": self.app.settings.get("speed", 1.0),
            **{key: self.app.settings.get(key, spec.SETTINGS_DEFAULTS[key]) for key in spec.SEGMENTATION_KEYS},
            "format": self.app.settings.get("format", "wav"),
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
            self.schema_form.hide()  # it lives until the next event loop pass
            self.schema_form.deleteLater()

        backend = self.shown_backend()
        schema = backend.get_config_schema()
        lang_code = self.app.engine_settings(backend.id).get("lang_code")
        voice_choices = [(v, v) for v in self.app.get_all_voices(lang_code, backend=backend)]
        values = self._current_schema_values()
        is_reference = getattr(backend, "voice_kind", None) == "reference"
        self.reference_voice_label.setVisible(is_reference)
        if is_reference:
            self.reference_voice_label.setText(
                f"Voice: {values.get('voice') or 'None selected'} (choose in Voices)")
        # "voice" is resolved here: its options depend on the language and
        # the files on disk. Every other choice list is the schema's own.
        overrides = {"voice": voice_choices}

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
                # Program fields are in Options > Settings....
                skip_keys={"lexicon", "pitch", *(('voice',) if is_reference else ()),
                           *(f.key for f in schema if spec.is_program_field(f))},
                on_change=self._on_schema_field_changed,
                # The Voice box is the engine's "Generation" group.
                flat_groups={VOICE_GROUP},
                collapsible=self._collapsed_groups,
                on_collapsed=self._collapsed_groups.__setitem__,
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
                    widget.setToolTip(f"{backend.display_name}'s own setting, for every character and project. "
                                      "Edit it with nothing selected.")
        if self.several:
            self._mark_mixed_schema_fields(schema)

    def _mixed_keys(self, keys) -> set:
        """The keys whose effective value differs across the selected clips."""
        configs = [self.app.document.effective_config_for_clip(clip) for clip in self._targets]
        return {key for key in keys if common_value(configs, lambda cfg, k=key: cfg.get(k)) is MIXED}

    def _mark_mixed_schema_fields(self, schema) -> None:
        """Several clips: a field they disagree on shows mixed, and a
        per-engine field (the voice) is disabled when their engines differ."""
        engines_differ = self._engines_differ()
        self._constructing_schema_form = True
        try:
            for key in self._mixed_keys(f.key for f in schema if f.key in ALLOWED_PRESET_KEYS):
                widget = self.schema_form.widget_for(key)
                if widget is not None:
                    mark_mixed(widget, on_pick=lambda checked, k=key: self._write_scoped(k, checked))
            if engines_differ:
                for f in schema:
                    widget = self.schema_form.widget_for(f.key)
                    if widget is not None and f.key not in SHARED_CONFIG_KEYS:
                        widget.setEnabled(False)
                        widget.setToolTip("The selected clips use different engines.")
        finally:
            self._constructing_schema_form = False

    def rebuild_schema_form(self) -> None:
        """Called by app.py's `set_character_engine` - re-renders this
        dock's schema fields for the active backend (the selected clip's or
        character's engine, else the first character's), keeping whatever
        clip/character/none mode is currently selected."""
        self._build_schema_form()
        self.refresh_engine_row()
        self._shown_signature = self._target_signature()
        self._refresh_scope_text()

    def refresh_voice_choices(self) -> None:
        if getattr(self.shown_backend(), "voice_kind", None) == "reference":
            self.reference_voice_label.setText(
                f"Voice: {self._current_schema_values().get('voice') or 'None selected'} (choose in Voices)")
            return
        lang_code = self.schema_form.values().get("lang_code")
        voices = self.app.get_all_voices(lang_code, backend=self.shown_backend())
        current = self.schema_form.values().get("voice")
        self.schema_form.set_choices("voice", [(v, v) for v in voices], current)

    def _on_schema_field_changed(self, key: str, value) -> None:
        if self._constructing_schema_form:
            return
        if key == "lang_code":
            self.refresh_voice_choices()
        if self._mode == "none":
            if key not in SHARED_CONFIG_KEYS:
                self.app.set_engine_setting(self.shown_backend().id, key, value)
                if self.app.editor is not None:
                    # A language or model setting is in the key: stale clips show now.
                    self.app.editor.rehighlight()
            self.app.schedule_save()
            return
        if key not in ALLOWED_PRESET_KEYS:
            return  # defense in depth - the field is disabled, unreachable via the UI
        self._write_scoped(key, value)

    def _write_scoped(self, key: str, value) -> None:
        """A clip-scope value goes into `clip.overrides`; a character-scope
        one into `preset_data` and through `commit_character_edit` (the
        library write-through)."""
        if self._target is None:
            return
        if self.several:
            if value is None:
                return  # the "(mixed)" entry picked: nothing chosen yet
            commands = [SetFieldCommand("clip", clip.id, "overrides", value, key=key)
                        for clip in self._targets if clip.overrides.get(key) != value]
            if commands:
                self.app.document.undo_stack.push(
                    commands[0] if len(commands) == 1 else CompositeCommand(commands))
                if self.app.editor is not None:
                    self.app.editor.rehighlight()
                self.app.schedule_save()
                self.app.refresh_timeline()
        elif self._mode == "clip":
            self._target.overrides[key] = value
            self._refresh_scope_text()
            self.app.schedule_save()
            self.app.refresh_timeline()
        elif self._mode == "character":
            self._target.preset_data[key] = value
            self.app.commit_character_edit(self._target)

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
            "fx_preset": self.app.settings.get("fx_preset", fx_resolve.PLACEHOLDER),
        }
        base.update(cfg)

        widgets = {"volume": self.volume_spin, "pitch": self.pitch_spin, "normalize": self.normalize_check,
                   "trim": self.trim_check, "apply_fx": self.apply_fx_check, "fx_preset": self.fx_preset_combo}
        mixed = self._mixed_keys(widgets) if self.several else set()
        for w in widgets.values():
            w.blockSignals(True)
        try:
            mixed_entry = self.fx_preset_combo.findText(MIXED_LABEL)
            if mixed_entry >= 0:
                self.fx_preset_combo.removeItem(mixed_entry)
            self.volume_spin.setValue(base["volume"])
            self.pitch_spin.setValue(base["pitch"])
            self.normalize_check.set_mixed(False)
            self.trim_check.set_mixed(False)
            self.apply_fx_check.set_mixed(False)
            self.normalize_check.setChecked(bool(base["normalize"]))
            self.trim_check.setChecked(bool(base["trim"]))
            self.apply_fx_check.setChecked(bool(base["apply_fx"]))
            fx_name = base["fx_preset"] or fx_resolve.PLACEHOLDER
            idx = self.fx_preset_combo.findData(fx_name)
            self.fx_preset_combo.setCurrentIndex(idx if idx >= 0 else 0)
            for key, widget in widgets.items():
                if isinstance(widget, MixedSpinBox):
                    widget.set_mixed(key in mixed)
                elif key in mixed:
                    mark_mixed(widget)
        finally:
            for w in widgets.values():
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
        self._write_scoped(key, value)

    def _on_fx_preset_index(self, index: int) -> None:
        self._on_fx_preset_selected(self.fx_preset_combo.itemData(index))

    def _on_fx_preset_selected(self, name: str) -> None:
        if not name or name == fx_resolve.PLACEHOLDER:
            return
        if self._mode == "none":
            # "none" mode's FX preset combo actually *loads* the preset's
            # resolved values into the global FX dock - project-wide FX
            # settings are a single shared instance, not per-clip.
            self.app.fx_dock.load_preset(name)
            return
        # clip/character mode only ever stores the preset *name* - resolved
        # into actual FX values later by app.py's _assemble_clip_config.
        self._write_scoped("fx_preset", name)

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
            "fx_preset": self.fx_preset_combo.currentData() or "",
        })
        return state

    def get_state(self) -> dict:
        """Always the project-wide ("none") state's values, regardless of
        what's currently rendered - see this module's docstring."""
        src = self._snapshot_none_values() if self._mode == "none" else self._none_values
        state = {k: v for k, v in src.items() if k in SHARED_CONFIG_KEYS or k in _HAND_BUILT_STATE_KEYS}
        state.update(self._program_values())
        return state

    def _program_values(self) -> dict:
        """The shared keys Options > Settings... edits, from `app.settings`."""
        return {key: self.app.settings.get(key, spec.SETTINGS_DEFAULTS[key])
                for key in (*spec.PROGRAM_SCHEMA_KEYS, "caching")}

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
            index = self.fx_preset_combo.findData(name)
            if index >= 0:
                self.fx_preset_combo.setCurrentIndex(index)
            self.fx_preset_combo.blockSignals(False)
        else:
            self._none_values["fx_preset"] = name

    # --- FX preset combo mirror (kept in sync with the FX dock's own combo) --

    def refresh_fx_presets(self) -> None:
        presets = [fx_resolve.PLACEHOLDER]
        if os.path.exists(qt_app_module.FX_PRESETS_DIR):
            files = [f for f in os.listdir(qt_app_module.FX_PRESETS_DIR) if f.endswith(".json")]
            presets.extend(f[:-5] for f in files)
        self.fx_preset_combo.blockSignals(True)
        self.fx_preset_combo.clear()
        for name in presets:
            # The placeholder shows as "No FX" but stays the stored value.
            self.fx_preset_combo.addItem(fx_resolve.PLACEHOLDER_LABEL if name == fx_resolve.PLACEHOLDER else name, name)
        self.fx_preset_combo.setCurrentIndex(0)
        self.fx_preset_combo.blockSignals(False)
