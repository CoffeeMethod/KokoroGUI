"""The Voice & Clip tab's clip fields, shown while a clip is selected. Rebuilt
by `SettingsDock` on every selection change, like the schema form. The
project's fields (pacing, crossfade, ripple, onset alignment, ducking,
track layout, timecode, source track) are in Options > Settings...
(kokoro_gui/qt/settings_window.py).

Clip: its gap override (blank inherits), its overlap with the clip before
it (`overrides["overlap_s"]`, blank for none), take, review status, note,
source text with a syllable comparison against the clip's text, and the
reference range (`overrides["reference_range"]`, typed as "start - end" in
seconds, blank for none), and the duration target fit to slot aims at
(`overrides["target_duration_s"]`, kokoro_gui/daw/fit.py; blank clears it).

Status, note and take are always in view. The rest fold: Timing (gap,
overlap, target) and Source (source text, syllables, the original's range).
Both start folded, stay as the user left them across selections, and name
what is set on the clip in their title.

Every edit is one `SetFieldCommand` (or `SetActiveTakeCommand`) on the
document's undo stack, then autosave and a timeline refresh.

Several clips selected (plan 35, grill PG11 (a)): `build_clips` shows gap,
overlap, status, note and target, each with the value the clips share or
blank "mixed", and an edit sets that field on every selected clip as one
`CompositeCommand`. Take, source text and the original range belong to one
line each, so they need one clip selected. The mixed helpers here
(`MIXED`, `common_value`, `MixedSpinBox`, `MixedCheckBox`, `mark_mixed`)
serve the Voice & Clip and Audio FX docks too.
"""
from __future__ import annotations

import re

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractSpinBox, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from kokoro_gui.daw import fit as fit_ops
from kokoro_gui.daw.arrangement import OVERLAP_KEY, OVERLAP_MAX_S, overlap_s
from kokoro_gui.daw.models import CLIP_STATUSES
from kokoro_gui.daw.reference import REFERENCE_RANGE_KEY, reference_range
from kokoro_gui.daw.undo import CompositeCommand, SetActiveTakeCommand, SetFieldCommand
from kokoro_gui.engine.text_extraction import strip_markup
from kokoro_gui.i18n import N_, tr
from kokoro_gui.qt import tips
from kokoro_gui.qt.collapsible import CollapsibleSection

STATUS_LABELS = {"todo": N_("To do"), "generated": N_("Generated"), "approved": N_("Approved"),
                 "needs_rewrite": N_("Needs rewrite")}
# The clip gap spin's minimum stands for "inherit" (shown as blank text).
GAP_INHERIT = -0.05
# The overlap spin's minimum stands for "no overlap" (blank), likewise.
OVERLAP_NONE = -0.05
# The target spin's minimum stands for "no target" (blank), likewise.
TARGET_NONE = 0.0
TARGET_MAX_S = 3600.0
_VOWEL_GROUPS = re.compile(r"[aeiouy]+", re.IGNORECASE)
_RANGE = re.compile(r"^\s*(\d+(?:\.\d*)?|\.\d+)\s*-\s*(\d+(?:\.\d*)?|\.\d+)\s*$")


# What `common_value` returns when the selected clips disagree.
MIXED = object()
MIXED_LABEL = N_("(mixed)")


def common_value(items, getter):
    """`getter(item)` when every item gives the same value, else `MIXED`
    (also for no items)."""
    values = [getter(item) for item in items]
    if not values or any(v != values[0] for v in values[1:]):
        return MIXED
    return values[0]


class MixedSpinBox(QDoubleSpinBox):
    """A spin box that can show blank for a mixed value. The first step or
    typed value ends the mixed state; `is_mixed()` tells a commit handler
    that the user hasn't picked a value yet."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mixed = False
        self.valueChanged.connect(self._on_value_changed)

    def set_mixed(self, mixed: bool) -> None:
        self._mixed = bool(mixed)
        self.lineEdit().setText(self.textFromValue(self.value()))

    def is_mixed(self) -> bool:
        return self._mixed

    def _on_value_changed(self, _value) -> None:
        self._mixed = False

    def stepBy(self, steps: int) -> None:  # noqa: N802 (Qt override)
        if self._mixed:
            # Blank text doesn't parse, and Qt won't step from it.
            self._mixed = False
            self.lineEdit().setText(self.textFromValue(self.value()))
        super().stepBy(steps)

    def textFromValue(self, value: float) -> str:  # noqa: N802 (Qt override)
        return "" if self._mixed else super().textFromValue(value)


class MixedCheckBox(QCheckBox):
    """A check box that shows a mixed value partly checked. A click on it
    checks it and emits `toggled(True)`, so the owner's usual handler runs."""

    def set_mixed(self, mixed: bool) -> None:
        self.setTristate(bool(mixed))
        if mixed:
            self.setCheckState(Qt.CheckState.PartiallyChecked)

    def is_mixed(self) -> bool:
        return self.checkState() == Qt.CheckState.PartiallyChecked

    def nextCheckState(self) -> None:  # noqa: N802 (Qt override)
        if self.is_mixed():
            blocked = self.blockSignals(True)
            self.setTristate(False)
            self.setCheckState(Qt.CheckState.Unchecked)
            self.blockSignals(blocked)
            self.setChecked(True)
            return
        super().nextCheckState()


def mark_mixed(widget, on_pick=None) -> None:
    """Shows `widget` as mixed: a blank spin, a "(mixed)" combo entry (data
    None) or line edit placeholder, a partly checked box. A stock check box
    calls `on_pick(checked)` on the click that leaves the mixed state."""
    if isinstance(widget, (MixedSpinBox, MixedCheckBox)):
        widget.set_mixed(True)
    elif isinstance(widget, QComboBox):
        if widget.findText(tr(MIXED_LABEL)) < 0:
            widget.insertItem(0, tr(MIXED_LABEL), None)
        widget.setCurrentIndex(widget.findText(tr(MIXED_LABEL)))
    elif isinstance(widget, QAbstractSpinBox):
        widget.lineEdit().clear()
    elif isinstance(widget, QCheckBox):
        widget.setTristate(True)
        widget.setCheckState(Qt.CheckState.PartiallyChecked)
        if on_pick is not None:
            def picked(_checked, box=widget):
                if box.isTristate():
                    box.setTristate(False)
                    on_pick(box.isChecked())
            widget.clicked.connect(picked)
    elif isinstance(widget, QLineEdit):
        widget.clear()
        widget.setPlaceholderText(tr(MIXED_LABEL))


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
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.form = QFormLayout()
        self.form.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(self.form)
        self.timing_section = CollapsibleSection(tr("Timing"), collapsed=True)
        self.timing_form = QFormLayout()
        self.timing_section.set_body_layout(self.timing_form)
        outer.addWidget(self.timing_section)
        self.source_section = CollapsibleSection(tr("Source"), collapsed=True)
        self.source_form = QFormLayout()
        self.source_section.set_body_layout(self.source_form)
        outer.addWidget(self.source_section)
        self.widgets: dict = {}
        self.clip_id = None
        # Several clips selected (`build_clips`); empty otherwise.
        self.clip_ids: list = []

    # -- building -------------------------------------------------------------

    def clear(self) -> None:
        for form in (self.form, self.timing_form, self.source_form):
            while form.rowCount():
                form.removeRow(0)
        self.widgets = {}
        self.clip_id = None
        self.clip_ids = []

    def build_clips(self, clips) -> None:
        """The fields with several clips selected; see the module docstring."""
        if len(clips) == 1:
            self.build_clip(clips[0])
            return
        self.clear()
        self.clip_ids = [clip.id for clip in clips]

        def mixed_spin(lo, hi, step, value, none_value, decimals=2):
            spin = MixedSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            spin.setSpecialValueText(" ")
            spin.setValue(none_value if value is MIXED or value is None else float(value))
            spin.set_mixed(value is MIXED)
            return spin

        status = QComboBox()
        status.setToolTip(tr("Where the clip is in your review."))
        for key in CLIP_STATUSES:
            status.addItem(tr(STATUS_LABELS[key]), key)
        shared_status = common_value(clips, lambda c: c.status)
        if shared_status is MIXED:
            mark_mixed(status)
        else:
            status.setCurrentIndex(max(0, status.findData(shared_status)))
        status.activated.connect(lambda _i: status.currentData() is None or self._set_all(
            "status", status.currentData()))
        self.form.addRow(tr("Status:"), status)

        shared_note = common_value(clips, lambda c: c.note)
        note = QLineEdit("" if shared_note is MIXED else shared_note)
        note.setToolTip(tr("A note to yourself about the clip. It is never spoken."))
        note.setPlaceholderText(tr("Review note"))
        if shared_note is MIXED:
            mark_mixed(note)
        note.editingFinished.connect(lambda: (shared_note is MIXED and not note.text()) or self._set_all(
            "note", note.text()))
        self.form.addRow(tr("Note:"), note)

        gap = mixed_spin(GAP_INHERIT, 10.0, 0.05, common_value(clips, lambda c: c.gap_before_s), GAP_INHERIT)
        gap.setToolTip(tr("Silence before each selected clip. Blank uses the project's gap."))
        gap.editingFinished.connect(lambda: gap.is_mixed() or self._set_all(
            "gap_before_s", None if gap.value() <= GAP_INHERIT else gap.value()))
        self.timing_form.addRow(tr("Gap before (s):"), gap)

        overlap = mixed_spin(OVERLAP_NONE, OVERLAP_MAX_S, 0.05, common_value(clips, overlap_s), OVERLAP_NONE)
        overlap.editingFinished.connect(lambda: overlap.is_mixed() or self._set_all_override(
            OVERLAP_KEY, None if overlap.value() <= OVERLAP_NONE else round(overlap.value(), 3)))
        overlap.setToolTip(tr("Start each selected clip this many seconds before the clip before it ends. "
                              "Blank places it after a gap, as usual."))
        self.timing_form.addRow(tr("Overlap (s):"), overlap)

        target = mixed_spin(TARGET_NONE, TARGET_MAX_S, 0.1, common_value(clips, fit_ops.target_duration_s),
                            TARGET_NONE, decimals=3)
        target.setToolTip(tr("How long each selected clip should last. Fit to slot aims at it. Blank for none."))
        target.editingFinished.connect(lambda: target.is_mixed() or self._set_all_override(
            fit_ops.TARGET_KEY, None if target.value() <= TARGET_NONE else round(target.value(), 3)))
        self.timing_form.addRow(tr("Target (s):"), target)

        one_only = QLabel(tr("{count} clips selected. Select one for its take, source text and "
                             "original range.", count=len(clips)))
        one_only.setWordWrap(True)
        self.form.addRow("", one_only)
        self.source_section.hide()
        self.timing_section.set_state("")
        self.widgets = {"gap_before_s": gap, "overlap_s": overlap, "status": status, "note": note,
                        "target_duration_s": target, "one_only": one_only}
        tips.label_rows(self)

    def _clips(self) -> list:
        document = self.app.document
        return [c for c in (document.get_clip(cid) for cid in self.clip_ids) if c is not None]

    def _push_all(self, commands) -> None:
        """One undo step for an edit to every selected clip."""
        if not commands:
            return
        self.app.document.undo_stack.push(commands[0] if len(commands) == 1 else CompositeCommand(commands))
        if self.app.editor is not None:
            self.app.editor.rehighlight()
        self._after_edit()

    def _set_all(self, field: str, value) -> None:
        self._push_all([SetFieldCommand("clip", c.id, field, value) for c in self._clips()
                        if getattr(c, field) != value])

    def _set_all_override(self, key: str, value) -> None:
        self._push_all([SetFieldCommand("clip", c.id, "overrides", value, key=key) for c in self._clips()
                        if c.overrides.get(key) != value])

    def build_clip(self, clip) -> None:
        self.clear()
        self.clip_id = clip.id
        status = QComboBox()
        status.setToolTip(tr("Where the clip is in your review."))
        for key in CLIP_STATUSES:
            status.addItem(tr(STATUS_LABELS[key]), key)
        status.setCurrentIndex(max(0, status.findData(clip.status)))
        status.activated.connect(lambda _i: self._set_clip("status", status.currentData()))
        self.form.addRow(tr("Status:"), status)

        note = QLineEdit(clip.note)
        note.setToolTip(tr("A note to yourself about the clip. It is never spoken."))
        note.setPlaceholderText(tr("Review note"))
        note.editingFinished.connect(lambda: self._set_clip("note", note.text()))
        self.form.addRow(tr("Note:"), note)

        take = QComboBox()
        take.setToolTip(tr("Which take of the clip plays."))
        active = int(clip.overrides.get("take", 0) or 0)
        for index in sorted({active, *clip.takes}):
            segments = clip.segments if index == active else clip.takes[index]
            seconds = sum(float(s.duration or 0.0) for s in segments)
            take.addItem(tr("Take {index} ({seconds:.1f}s)", index=index + 1, seconds=seconds), index)
        take.setCurrentIndex(take.findData(active))
        take.setEnabled(bool(clip.takes))
        if not clip.takes:
            take.setToolTip(tr("Takes to switch between appear here when you regenerate the clip."))
        take.activated.connect(lambda _i: self._pick_take(take.currentData()))
        self.form.addRow(tr("Take:"), take)

        gap = _spin(GAP_INHERIT, 10.0, 0.05, GAP_INHERIT if clip.gap_before_s is None else float(clip.gap_before_s))
        gap.setSpecialValueText(" ")
        gap.setToolTip(tr("Silence before this clip. Blank uses the project's gap."))
        gap.editingFinished.connect(lambda: self._set_clip("gap_before_s",
                                                           None if gap.value() <= GAP_INHERIT else gap.value()))
        self.timing_form.addRow(tr("Gap before (s):"), gap)

        current_overlap = overlap_s(clip)
        overlap = _spin(OVERLAP_NONE, OVERLAP_MAX_S, 0.05, OVERLAP_NONE if current_overlap is None else current_overlap)
        overlap.setSpecialValueText(" ")
        overlap.setToolTip(tr("Start this many seconds before the clip before it ends, for a backchannel or an "
                              "interruption. Blank places it after a gap, as usual. A clip placed by hand ignores it."))
        overlap.editingFinished.connect(
            lambda: self._set_override(OVERLAP_KEY, None if overlap.value() <= OVERLAP_NONE
                                       else round(overlap.value(), 3)))
        self.timing_form.addRow(tr("Overlap (s):"), overlap)

        source = QPlainTextEdit(clip.source_text or "")
        source.setToolTip(tr("The original line this clip dubs."))
        source.setReadOnly(True)
        source.setFixedHeight(54)
        source.setPlaceholderText(tr("Original line, for a dub"))
        edit = QPushButton(tr("Edit"))
        edit.setToolTip(tr("Change the original line."))
        edit.setCheckable(True)
        edit.toggled.connect(lambda on: self._toggle_source_edit(on, source, edit))
        source_row = QWidget()
        source_layout = QHBoxLayout(source_row)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.addWidget(source, 1)
        source_layout.addWidget(edit)
        self.source_form.addRow(tr("Source text:"), source_row)
        syllables = QLabel(self._syllable_text(clip))
        syllables.setToolTip(tr("Vowel groups per word: a rough lip-sync length check, not a real syllable count."))
        self.source_form.addRow("", syllables)

        reference = QLineEdit(format_range(reference_range(clip)))
        reference.setPlaceholderText(tr("start - end (s)"))
        reference.setToolTip(tr("The part of the source track this clip dubs, in seconds. The transport's "
                                "Original and Both play it under the clip. Blank for none."))
        reference.editingFinished.connect(lambda: self._commit_reference_range(reference))
        self.source_form.addRow(tr("Original (s):"), reference)
        current_target = fit_ops.target_duration_s(clip)
        target = QDoubleSpinBox()
        target.setRange(TARGET_NONE, TARGET_MAX_S)
        target.setDecimals(3)
        target.setSingleStep(0.1)
        target.setValue(TARGET_NONE if current_target is None else current_target)
        target.setSpecialValueText(" ")
        target.setToolTip(tr("How long this clip should last, from its time on the timeline (a subtitle "
                             "cue's length). Fit to slot aims at it. Blank for none."))
        target.editingFinished.connect(
            lambda: self._set_override(fit_ops.TARGET_KEY,
                                       None if target.value() <= TARGET_NONE else round(target.value(), 3)))
        self.timing_form.addRow(tr("Target (s):"), target)
        self.widgets = {"gap_before_s": gap, "overlap_s": overlap, "take": take, "status": status, "note": note,
                        "source_text": source, "source_edit": edit, "syllables": syllables,
                        "reference_range": reference, "target_duration_s": target}
        tips.label_rows(self)
        self.source_section.show()
        self._sync_section_states(clip)

    def _sync_section_states(self, clip) -> None:
        """What the folded sections hold, in their titles."""
        timing = [] if clip.gap_before_s is None else [tr("gap {gap_before_s:g} s",
                                                          gap_before_s=float(clip.gap_before_s))]
        overlap, target = overlap_s(clip), fit_ops.target_duration_s(clip)
        timing += [] if overlap is None else [tr("overlap {overlap:g} s", overlap=overlap)]
        timing += [] if target is None else [tr("target {target:g} s", target=target)]
        self.timing_section.set_state(", ".join(timing))
        source = [tr("source text")] if clip.source_text else []
        source += [tr("original range")] if reference_range(clip) is not None else []
        self.source_section.set_state(", ".join(source))

    def _syllable_text(self, clip) -> str:
        dub = syllable_count(strip_markup(self.app.document.clip_text(clip)))
        if not clip.source_text:
            return tr("Syllables: {dub}", dub=dub)
        return tr("Syllables: source {source} / dub {dub}", source=syllable_count(clip.source_text), dub=dub)

    # -- commits ----------------------------------------------------------------

    def _after_edit(self) -> None:
        clip = self._clip()
        if clip is not None:
            self._sync_section_states(clip)
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
        button.setText(tr("Save") if editing else tr("Edit"))
        if editing:
            source.setFocus()
            return
        text = source.toPlainText().strip()
        self._set_clip("source_text", text or None)
        clip = self._clip()
        if clip is not None and "syllables" in self.widgets:
            self.widgets["syllables"].setText(self._syllable_text(clip))
