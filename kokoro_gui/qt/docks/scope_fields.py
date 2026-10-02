"""The Settings tab's clip fields, shown while a clip is selected. Rebuilt
by `SettingsDock` on every selection change, like the schema form. The
project's fields (pacing, crossfade, ripple, onset alignment, ducking,
track layout, timecode, source track) are in Options > Settings...
(kokoro_gui/qt/settings_window.py).

Clip: its gap override (blank inherits), take, review status, note,
source text with a syllable comparison against the clip's text, and the
reference range (`overrides["reference_range"]`, typed as "start - end" in
seconds, blank for none), and the duration target fit to slot aims at
(`overrides["target_duration_s"]`, kokoro_gui/daw/fit.py; blank clears it).

Every edit is one `SetFieldCommand` (or `SetActiveTakeCommand`) on the
document's undo stack, then autosave and a timeline refresh.
"""
from __future__ import annotations

import re

from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
    QWidget,
)

from kokoro_gui.daw import fit as fit_ops
from kokoro_gui.daw.models import CLIP_STATUSES
from kokoro_gui.daw.reference import REFERENCE_RANGE_KEY, reference_range
from kokoro_gui.daw.undo import SetActiveTakeCommand, SetFieldCommand

STATUS_LABELS = {"todo": "To do", "generated": "Generated", "approved": "Approved",
                 "needs_rewrite": "Needs rewrite"}
# The clip gap spin's minimum stands for "inherit" (shown as blank text).
GAP_INHERIT = -0.05
# The target spin's minimum stands for "no target" (blank), likewise.
TARGET_NONE = 0.0
TARGET_MAX_S = 3600.0
_VOWEL_GROUPS = re.compile(r"[aeiouy]+", re.IGNORECASE)
_RANGE = re.compile(r"^\s*(\d+(?:\.\d*)?|\.\d+)\s*-\s*(\d+(?:\.\d*)?|\.\d+)\s*$")


def parse_range(text: str):
    """`"1.5 - 3"` as `[1.5, 3.0]`; `""` as None; ValueError for anything
    else or an end not after the start."""
    if not (text or "").strip():
        return None
    match = _RANGE.match(text)
    if match is None:
        raise ValueError(text)
    start, end = float(match.group(1)), float(match.group(2))
    if end <= start:
        raise ValueError(text)
    return [start, end]


def format_range(rng) -> str:
    return "" if rng is None else f"{rng[0]:.2f} - {rng[1]:.2f}"


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

        reference = QLineEdit(format_range(reference_range(clip)))
        reference.setPlaceholderText("start - end (s)")
        reference.setToolTip("The part of the source track this clip dubs, in seconds. The transport's "
                             "Original and Both play it under the clip. Blank for none.")
        reference.editingFinished.connect(lambda: self._commit_reference_range(reference))
        self.form.addRow("Original (s):", reference)
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
                        "reference_range": reference, "target_duration_s": target}

    def _syllable_text(self, clip) -> str:
        dub = syllable_count(self.app.document.clip_text(clip))
        if not clip.source_text:
            return f"Syllables: {dub}"
        return f"Syllables: source {syllable_count(clip.source_text)} / dub {dub}"

    # -- commits ----------------------------------------------------------------

    def _after_edit(self) -> None:
        self.app.schedule_save()
        self.app.refresh_timeline()

    def _clip(self):
        return self.app.document.get_clip(self.clip_id) if self.clip_id else None

    def _commit_reference_range(self, field: QLineEdit) -> None:
        """`overrides["reference_range"]` from the "start - end" field; text
        that doesn't parse puts the stored range back."""
        clip = self._clip()
        if clip is None:
            return
        try:
            value = parse_range(field.text())
        except ValueError:
            field.setText(format_range(reference_range(clip)))
            return
        if value == clip.overrides.get(REFERENCE_RANGE_KEY):
            return
        self.app.document.undo_stack.push(SetFieldCommand("clip", clip.id, "overrides", value,
                                                          key=REFERENCE_RANGE_KEY))
        field.setText(format_range(value))
        self._after_edit()

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
