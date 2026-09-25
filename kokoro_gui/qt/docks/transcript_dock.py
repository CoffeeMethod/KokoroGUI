"""Transcript dock: the panel a user stares at most (top-left of the 2x2
grid, Claude/PLAN_ui_shell_redesign.md section 2). Replaces the old
`GenerationDock` ("Generate Audio"), whose Input Source tabs, file-path row,
legacy preset row and Auto-Split row all moved out: Load File is File >
Import Text, the legacy `presets/*.json` combo is gone (Characters replaced
it), auto-split is an option on the Transport dock's Generate menu.

What's left is a header row with two combos above the editor:

- Character: reflects the run under the caret (or the selection's first
  run); changing it assigns that character to the selection, or to the
  caret's whole clip when nothing is selected, or to the caret's line for
  untagged text. "Manage characters..." at the bottom opens the Edit >
  Characters dialog.
- FX: lists `presets/fx/*.json` plus "(none)" and "Edit in FX tab...".
  Changing it sets the caret clip's `fx_override` (resolved values) and
  records the preset name in `clip.overrides["fx_preset"]` through
  `SetClipFxCommand`, so the choice is undoable and the gutter can name it.
- Variant: shown only when the caret clip's character has variants
  (`Character.variants`, a cloning backend's alternate references). Sets
  `clip.overrides["variant"]` through `SetFieldCommand`; "(default)" clears
  it.
"""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QDockWidget, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from kokoro_gui.daw.undo import SetClipFxCommand, SetFieldCommand
from kokoro_gui.engine.presets import ALLOWED_FX_PRESET_KEYS, filter_allowed_keys
from kokoro_gui.qt.fx_presets import list_fx_preset_names
from kokoro_gui.qt.transcript_editor import TranscriptEditor, clip_fx_name

FX_NONE_LABEL = "(none)"
FX_EDIT_LABEL = "Edit in FX tab..."
CHARACTER_MANAGE_LABEL = "Manage characters..."
_MIXED_LABEL = "(mixed)"
VARIANT_DEFAULT_LABEL = "(default)"


class TranscriptDock(QDockWidget):
    def __init__(self, app, parent=None):
        super().__init__("Transcript", parent)
        self.setObjectName("dock_transcript")
        self.app = app
        self._syncing = False

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Phase 4 (NP1): shown while the transcript shows a subproject the
        # timeline isn't in.
        self.scope_bar = QWidget()
        scope_layout = QHBoxLayout(self.scope_bar)
        scope_layout.setContentsMargins(0, 0, 0, 0)
        self.scope_label = QLabel()
        self.scope_back_btn = QPushButton("Back")
        self.scope_back_btn.setToolTip("Show the project the timeline is in.")
        self.scope_back_btn.clicked.connect(lambda: self.app.set_focus(self.app.level))
        scope_layout.addWidget(self.scope_label, 1)
        scope_layout.addWidget(self.scope_back_btn)
        self.scope_bar.hide()
        layout.addWidget(self.scope_bar)

        header = QHBoxLayout()
        header.addWidget(QLabel("Character:"))
        self.character_combo = QComboBox()
        self.character_combo.setMinimumWidth(120)
        header.addWidget(self.character_combo, 1)
        header.addSpacing(8)
        header.addWidget(QLabel("FX:"))
        self.fx_combo = QComboBox()
        self.fx_combo.setMinimumWidth(120)
        header.addWidget(self.fx_combo, 1)
        self.variant_label = QLabel("Variant:")
        self.variant_combo = QComboBox()
        self.variant_combo.setMinimumWidth(100)
        header.addSpacing(8)
        header.addWidget(self.variant_label)
        header.addWidget(self.variant_combo, 1)
        header.addStretch(1)
        layout.addLayout(header)

        self.editor = TranscriptEditor(self.app)
        layout.addWidget(self.editor, 1)
        self.setWidget(content)

        self.refresh_character_choices()
        self.refresh_fx_choices()
        self.character_combo.activated.connect(self._on_character_activated)
        self.fx_combo.activated.connect(self._on_fx_activated)
        self.variant_combo.activated.connect(self._on_variant_activated)
        self.editor.cursorPositionChanged.connect(self.sync_header)

    def refresh_scope(self) -> None:
        """The subproject bar: its title and a way back to the level."""
        text = self.app.scope_text() if hasattr(self.app, "scope_text") else None
        self.scope_bar.setVisible(bool(text))
        if text:
            self.scope_label.setText(text)
            self.scope_back_btn.setText(f"Back to {self.app.level.title()}")
        self.app.selection.changed.connect(self.sync_header)
        self.sync_header()

    # -- combo contents ----------------------------------------------------

    def refresh_character_choices(self) -> None:
        self._syncing = True
        try:
            self.character_combo.clear()
            self.character_combo.addItem(_MIXED_LABEL, None)
            for character in self.app.document.characters:
                self.character_combo.addItem(character.name, character.id)
            self.character_combo.insertSeparator(self.character_combo.count())
            self.character_combo.addItem(CHARACTER_MANAGE_LABEL, "__manage__")
        finally:
            self._syncing = False
        self.sync_header()

    def refresh_fx_choices(self) -> None:
        self._syncing = True
        try:
            self.fx_combo.clear()
            self.fx_combo.addItem(FX_NONE_LABEL, "")
            for name in list_fx_preset_names(self.app.project_dir):
                self.fx_combo.addItem(name, name)
            self.fx_combo.insertSeparator(self.fx_combo.count())
            self.fx_combo.addItem(FX_EDIT_LABEL, "__edit__")
        finally:
            self._syncing = False
        self.sync_header()

    # -- header <- caret ---------------------------------------------------

    def sync_header(self) -> None:
        """Reflect the caret's clip in both combos without firing their
        change handlers."""
        if self._syncing:
            return
        clip = self.editor.current_clip()
        self._syncing = True
        try:
            character_id = clip.character_id if clip is not None else None
            index = self.character_combo.findData(character_id) if character_id else 0
            self.character_combo.setCurrentIndex(index if index >= 0 else 0)

            fx_name = clip_fx_name(self.app.document, clip) if clip is not None else None
            self.fx_combo.setEnabled(clip is not None)
            if not fx_name or fx_name == "custom":
                fx_index = 0 if not fx_name else -1
                if fx_name == "custom":
                    # Resolved values with no recorded name: show as none
                    # (the gutter says "FX: custom").
                    fx_index = 0
            else:
                fx_index = self.fx_combo.findData(fx_name)
            self.fx_combo.setCurrentIndex(fx_index if fx_index >= 0 else 0)
            self._sync_variants(clip)
        finally:
            self._syncing = False

    def _sync_variants(self, clip) -> None:
        character = self.app.document.get_character(clip.character_id) if clip is not None else None
        variants = sorted((character.variants or {}).keys()) if character is not None else []
        self.variant_combo.clear()
        self.variant_combo.addItem(VARIANT_DEFAULT_LABEL, "")
        for name in variants:
            self.variant_combo.addItem(name, name)
        current = (clip.overrides or {}).get("variant", "") if clip is not None else ""
        index = self.variant_combo.findData(current or "")
        self.variant_combo.setCurrentIndex(index if index >= 0 else 0)
        self.variant_label.setVisible(bool(variants))
        self.variant_combo.setVisible(bool(variants))

    # -- header -> document ------------------------------------------------

    def _on_character_activated(self, index: int) -> None:
        if self._syncing:
            return
        data = self.character_combo.itemData(index)
        if data == "__manage__":
            self.sync_header()
            self.app.open_characters_dialog()
            return
        if not data:
            return
        target = self.editor.current_target_range()
        if target is None or target[1] <= target[0]:
            return
        start, end = target
        self.editor._push_assign_character(start, end, data)
        self.sync_header()

    def _on_fx_activated(self, index: int) -> None:
        if self._syncing:
            return
        data = self.fx_combo.itemData(index)
        if data == "__edit__":
            self.sync_header()
            self.app.raise_fx_tab()
            return
        clip = self.editor.current_clip()
        if clip is None:
            return
        self.apply_fx_preset_to_clip(clip.id, data or "")

    def _on_variant_activated(self, index: int) -> None:
        if self._syncing:
            return
        clip = self.editor.current_clip()
        if clip is None:
            return
        self.set_clip_variant(clip.id, self.variant_combo.itemData(index) or None)

    def set_clip_variant(self, clip_id: str, variant) -> None:
        """Undoable `clip.overrides["variant"]`; None clears it. A variant
        is a generation input (a different reference), so the clip goes
        stale."""
        clip = self.app.document.get_clip(clip_id)
        if clip is None or (clip.overrides or {}).get("variant") == variant:
            return
        self.app.document.undo_stack.push(SetFieldCommand("clip", clip_id, "overrides", variant, key="variant"))
        self.editor.rehighlight()
        self.app.schedule_save()
        self.app.refresh_timeline()
        self.sync_header()

    def apply_fx_preset_to_clip(self, clip_id: str, preset_name: str) -> None:
        """Shared with the timeline's FX menu (`TimelineDock.on_fx_preset_requested`
        delegates here). Empty `preset_name` clears the override."""
        clip = self.app.document.get_clip(clip_id)
        if clip is None:
            return
        if not preset_name:
            fx_values = None
        else:
            preset = self.app.engine.load_fx_preset(preset_name, self.app.project_dir)
            fx_values = filter_allowed_keys(preset, ALLOWED_FX_PRESET_KEYS) if preset else None
        self.app.document.undo_stack.push(SetClipFxCommand(clip_id, fx_values, preset_name=preset_name or None))
        self.editor.rehighlight()
        self.app.schedule_save()
        self.app.refresh_timeline()
        self.sync_header()
        if self.app.fx_dock is not None:
            self.app.fx_dock.refresh_for_selection()
