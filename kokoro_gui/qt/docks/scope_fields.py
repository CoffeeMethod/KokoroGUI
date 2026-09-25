"""The Settings tab's scope group: project fields while nothing is selected,
clip fields while a clip is. Rebuilt by `SettingsDock` on every selection
change, like the schema form.

Project: pacing (`Document.settings["gap_s"]` / `["paragraph_gap_s"]`,
kokoro_gui/daw/arrangement.py), auto-crossfade (`["auto_crossfade"]`,
kokoro_gui/daw/mixplan.py), ripple on regenerate (`["ripple"]`, on by
default, kokoro_gui/daw/arrangement.py), onset alignment of locked clips
(`["align_onset"]`, derived from the pinned clips while unset,
`arrangement.align_onset_enabled`), the track layout (`["track_layout"]`,
kokoro_gui/daw/lanes.py), and timecode (`["timecode"]`,
kokoro_gui/daw/timecode.py).

Clip: its gap override (blank inherits), take, review status, note,
source text with a syllable comparison against the clip's text, and the
duration target fit to slot aims at (`overrides["target_duration_s"]`,
kokoro_gui/daw/fit.py; blank clears it).

Every edit is one `SetFieldCommand` (or `SetActiveTakeCommand`) on the
document's undo stack, then autosave and a timeline refresh.
"""
from __future__ import annotations

import re

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
    QPushButton, QSpinBox, QWidget,
)

from kokoro_gui.daw import fit as fit_ops
from kokoro_gui.daw.arrangement import DEFAULT_GAP_S, DEFAULT_PARAGRAPH_GAP_S, align_onset_enabled
from kokoro_gui.daw.models import CLIP_STATUSES
from kokoro_gui.daw.timecode import FRAME_RATES, tc_to_frames, timecode_settings
from kokoro_gui.daw.undo import SetActiveTakeCommand, SetFieldCommand

STATUS_LABELS = {"todo": "To do", "generated": "Generated", "approved": "Approved",
                 "needs_rewrite": "Needs rewrite"}
# The clip gap spin's minimum stands for "inherit" (shown as blank text).
GAP_INHERIT = -0.05
# The target spin's minimum stands for "no target" (blank), likewise.
TARGET_NONE = 0.0
TARGET_MAX_S = 3600.0
_VOWEL_GROUPS = re.compile(r"[aeiouy]+", re.IGNORECASE)


def syllable_count(text: str) -> int:
    """Vowel groups per word, at least one per word: a rough count, fine
    for comparing a dub line's length with its source line."""
    total = 0
    for word in re.findall(r"[^\W\d_]+", text or ""):
        total += max(1, len(_VOWEL_GROUPS.findall(word)))
    return total


def _spin(lo: float, hi: float, step: float, value: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setSingleStep(step)
    spin.setDecimals(2)
    spin.setValue(value)
    return spin


class ScopeFields(QWidget):
    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.form = QFormLayout(self)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.widgets: dict = {}
        self.clip_id = None

    # -- building -------------------------------------------------------------

    def clear(self) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        self.widgets = {}
        self.clip_id = None

    def build_project(self) -> None:
        self.clear()
        settings = self.app.document.settings
        focus = getattr(self.app, "focus", None)
        title = None
        if focus is not None and focus.parent_id is not None:
            # A subproject's name, shown by its parent's placeholder line,
            # the timeline block and the breadcrumb (phase 4).
            title = QLineEdit(focus.title())
            title.setToolTip("The subproject's name in its parent.")
            title.editingFinished.connect(lambda: self.app.rename_subproject(self.app.focus, title.text()))
            self.form.addRow("Title:", title)
        gap = _spin(0.0, 10.0, 0.05, float(settings.get("gap_s", DEFAULT_GAP_S)))
        para = _spin(0.0, 10.0, 0.05, float(settings.get("paragraph_gap_s", DEFAULT_PARAGRAPH_GAP_S)))
        gap.setToolTip("Silence between clips placed one after another.")
        para.setToolTip("Silence between clips across a blank line.")
        gap.editingFinished.connect(lambda: self._set_setting("gap_s", gap.value()))
        para.editingFinished.connect(lambda: self._set_setting("paragraph_gap_s", para.value()))
        self.form.addRow("Gap (s):", gap)
        self.form.addRow("Paragraph gap (s):", para)

        crossfade = QCheckBox("Auto-crossfade overlapping clips")
        crossfade.setChecked(bool(settings.get("auto_crossfade", False)))
        crossfade.toggled.connect(lambda on: self._set_setting("auto_crossfade", bool(on)))
        self.form.addRow("", crossfade)

        ripple = QCheckBox("Ripple on regenerate")
        ripple.setChecked(bool(settings.get("ripple", True)))
        ripple.setToolTip("When a regenerated clip changes length, move the clips placed after it by the "
                          "difference. A clip locked in time stays put.")
        ripple.toggled.connect(lambda on: self._set_setting("ripple", bool(on)))
        self.form.addRow("", ripple)

        align = QCheckBox("Align locked clips to their first word")
        align.setChecked(align_onset_enabled(self.app.document))
        align.setToolTip("Start each clip locked in time a little early, by the silence before its first "
                         "word, so the word lands on the clip's time. Skipped for a clip with trim on. "
                         "On by default when the project has a locked clip.")
        align.toggled.connect(lambda on: self._set_setting("align_onset", bool(on)))
        self.form.addRow("", align)

        layout = self.app.document.track_layout()
        layout_combo = QComboBox()
        layout_combo.addItem("One per character", "character")
        layout_combo.addItem("Unified", "unified")
        layout_combo.setCurrentIndex(max(0, layout_combo.findData(layout["mode"])))
        layout_combo.setToolTip("Unified puts every clip on a few lanes and moves to the next lane "
                                "whenever the speaker changes.")
        lanes = QSpinBox()
        lanes.setRange(1, 16)
        lanes.setValue(int(layout.get("lanes", 3)))
        lanes.setSuffix(" lanes")
        lanes.setEnabled(layout["mode"] == "unified")
        layout_row = QWidget()
        layout_box = QHBoxLayout(layout_row)
        layout_box.setContentsMargins(0, 0, 0, 0)
        layout_box.addWidget(layout_combo, 1)
        layout_box.addWidget(lanes)
        layout_combo.activated.connect(lambda _i: self._commit_track_layout())
        lanes.editingFinished.connect(self._commit_track_layout)
        self.form.addRow("Track layout:", layout_row)

        tc = timecode_settings(settings)
        enabled = QCheckBox("Show timecode")
        enabled.setChecked(bool(tc["enabled"]))
        fps = QComboBox()
        for rate in FRAME_RATES:
            fps.addItem(f"{rate:g} fps", rate)
        index = fps.findData(float(tc["frame_rate"]))
        fps.setCurrentIndex(index if index >= 0 else FRAME_RATES.index(25.0))
        start = QLineEdit(str(tc["start"]))
        start.setToolTip("Timecode of the project's first frame, HH:MM:SS:FF.")
        drop = QCheckBox("Drop-frame")
        drop.setChecked(bool(tc["drop_frame"]))
        drop.setToolTip("29.97 and 59.94 only.")
        tc_row = QWidget()
        tc_layout = QHBoxLayout(tc_row)
        tc_layout.setContentsMargins(0, 0, 0, 0)
        tc_layout.addWidget(enabled)
        tc_layout.addWidget(drop)
        self.form.addRow("Timecode:", tc_row)
        self.form.addRow("Frame rate:", fps)
        self.form.addRow("Start:", start)
        for signal in (enabled.toggled, drop.toggled, fps.currentIndexChanged):
            signal.connect(lambda *_: self._commit_timecode())
        start.editingFinished.connect(self._commit_timecode)
        self.widgets = {"title": title, "gap_s": gap, "paragraph_gap_s": para, "auto_crossfade": crossfade,
                        "ripple": ripple, "align_onset": align, "track_layout": layout_combo, "track_lanes": lanes, "tc_enabled": enabled, "tc_fps": fps, "tc_start": start, "tc_drop": drop}

    def build_clip(self, clip) -> None:
        self.clear()
        self.clip_id = clip.id
        gap = _spin(GAP_INHERIT, 10.0, 0.05, GAP_INHERIT if clip.gap_before_s is None else float(clip.gap_before_s))
        gap.setSpecialValueText(" ")
        gap.setToolTip("Silence before this clip. Blank uses the project's gap.")
        gap.editingFinished.connect(lambda: self._set_clip("gap_before_s",
                                                           None if gap.value() <= GAP_INHERIT else gap.value()))
        self.form.addRow("Gap before (s):", gap)

        take = QComboBox()
        active = int(clip.overrides.get("take", 0) or 0)
        for index in sorted({active, *clip.takes}):
            segments = clip.segments if index == active else clip.takes[index]
            seconds = sum(float(s.duration or 0.0) for s in segments)
            take.addItem(f"Take {index + 1} ({seconds:.1f}s)", index)
        take.setCurrentIndex(take.findData(active))
        take.setEnabled(bool(clip.takes))
        take.activated.connect(lambda _i: self._pick_take(take.currentData()))
        self.form.addRow("Take:", take)

        status = QComboBox()
        for key in CLIP_STATUSES:
            status.addItem(STATUS_LABELS[key], key)
        status.setCurrentIndex(max(0, status.findData(clip.status)))
        status.activated.connect(lambda _i: self._set_clip("status", status.currentData()))
        self.form.addRow("Status:", status)

        note = QLineEdit(clip.note)
        note.setPlaceholderText("Review note")
        note.editingFinished.connect(lambda: self._set_clip("note", note.text()))
        self.form.addRow("Note:", note)

        source = QPlainTextEdit(clip.source_text or "")
        source.setReadOnly(True)
        source.setFixedHeight(54)
        source.setPlaceholderText("Original line, for a dub")
        edit = QPushButton("Edit")
        edit.setCheckable(True)
        edit.toggled.connect(lambda on: self._toggle_source_edit(on, source, edit))
        source_row = QWidget()
        source_layout = QHBoxLayout(source_row)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.addWidget(source, 1)
        source_layout.addWidget(edit)
        self.form.addRow("Source text:", source_row)
        syllables = QLabel(self._syllable_text(clip))
        syllables.setToolTip("Vowel groups per word: a rough lip-sync length check, not a real syllable count.")
        self.form.addRow("", syllables)

        current_target = fit_ops.target_duration_s(clip)
        target = QDoubleSpinBox()
        target.setRange(TARGET_NONE, TARGET_MAX_S)
        target.setDecimals(3)
        target.setSingleStep(0.1)
        target.setValue(TARGET_NONE if current_target is None else current_target)
        target.setSpecialValueText(" ")
        target.setToolTip("How long this clip should last, from its time on the timeline (a subtitle "
                          "cue's length). Fit to slot aims at it. Blank for none.")
        target.editingFinished.connect(
            lambda: self._set_override(fit_ops.TARGET_KEY,
                                       None if target.value() <= TARGET_NONE else round(target.value(), 3)))
        self.form.addRow("Target (s):", target)
        self.widgets = {"gap_before_s": gap, "take": take, "status": status, "note": note,
                        "source_text": source, "source_edit": edit, "syllables": syllables,
                        "target_duration_s": target}

    def _syllable_text(self, clip) -> str:
        dub = syllable_count(self.app.document.clip_text(clip))
        if not clip.source_text:
            return f"Syllables: {dub}"
        return f"Syllables: source {syllable_count(clip.source_text)} / dub {dub}"

    # -- commits ----------------------------------------------------------------

    def _after_edit(self) -> None:
        self.app.schedule_save()
        self.app.refresh_timeline()

    def _set_setting(self, key: str, value) -> None:
        if self.app.document.settings.get(key) == value:
            return
        self.app.document.undo_stack.push(SetFieldCommand("document", None, "settings", value, key=key))
        self._after_edit()

    def _commit_track_layout(self) -> None:
        """One undo step: the setting and the relane it triggers
        (`lanes.relane_follow_up`)."""
        mode = self.widgets["track_layout"].currentData()
        lanes = self.widgets["track_lanes"]
        lanes.setEnabled(mode == "unified")
        value = {"mode": "unified", "lanes": lanes.value()} if mode == "unified" else {"mode": "character"}
        if self.app.document.track_layout() == value:
            return
        self._set_setting("track_layout", value)

    def _commit_timecode(self) -> None:
        w = self.widgets
        start = w["tc_start"].text().strip() or "00:00:00:00"
        rate = float(w["tc_fps"].currentData())
        drop = w["tc_drop"].isChecked() and rate in (29.97, 59.94)
        try:
            tc_to_frames(start, rate, drop)
        except ValueError:
            start = timecode_settings(self.app.document.settings)["start"]
            w["tc_start"].setText(start)
        value = {"enabled": w["tc_enabled"].isChecked(), "frame_rate": rate, "start": start, "drop_frame": drop}
        self._set_setting("timecode", value)
        self.app.transport_dock.set_position(self.app.transport.position(), self.app.transport.duration())

    def _clip(self):
        return self.app.document.get_clip(self.clip_id) if self.clip_id else None

    def _set_clip(self, field: str, value) -> None:
        clip = self._clip()
        if clip is None or getattr(clip, field) == value:
            return
        self.app.document.undo_stack.push(SetFieldCommand("clip", clip.id, field, value))
        self._after_edit()

    def _set_override(self, key: str, value) -> None:
        """One entry of `clip.overrides`; None removes it."""
        clip = self._clip()
        if clip is None or clip.overrides.get(key) == value:
            return
        self.app.document.undo_stack.push(SetFieldCommand("clip", clip.id, "overrides", value, key=key))
        if self.app.editor is not None:
            self.app.editor.rehighlight()
        self._after_edit()

    def _pick_take(self, index) -> None:
        clip = self._clip()
        if clip is None or index is None or int(index) not in clip.takes:
            return
        self.app.document.undo_stack.push(SetActiveTakeCommand(clip.id, int(index)))
        if self.app.editor is not None:
            self.app.editor.rehighlight()
        self._after_edit()

    def _toggle_source_edit(self, editing: bool, source: QPlainTextEdit, button: QPushButton) -> None:
        source.setReadOnly(not editing)
        button.setText("Save" if editing else "Edit")
        if editing:
            source.setFocus()
            return
        text = source.toPlainText().strip()
        self._set_clip("source_text", text or None)
        clip = self._clip()
        if clip is not None and "syllables" in self.widgets:
            self.widgets["syllables"].setText(self._syllable_text(clip))
