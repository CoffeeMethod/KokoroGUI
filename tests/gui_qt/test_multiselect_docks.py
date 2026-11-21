"""Plan 35, grill PG11 (a): with several clips selected the Settings and
Audio FX docks show each field's shared value or "mixed", and an edit sets
that field on every selected clip in one undo step."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor

import kokoro_gui.qt.app  # noqa: F401  (docks import the app module; load it first)
from kokoro_gui.qt.docks.scope_fields import MIXED, MIXED_LABEL, MixedCheckBox, MixedSpinBox, common_value


def _three_clips(qt_app):
    cursor = qt_app.editor.textCursor()
    cursor.select(QTextCursor.SelectionType.Document)
    cursor.insertText("one. two. three.")
    character = qt_app.document.characters[0]
    clips = [qt_app.document.assign_character_to_range(s, e, character.id) for s, e in ((0, 4), (5, 9), (10, 16))]
    qt_app.editor.rehighlight()
    return clips


def _select(qt_app, clips):
    qt_app.selection.select_clips([c.id for c in clips])


def test_common_value():
    assert common_value([1, 1, 1], lambda v: v) == 1
    assert common_value([1, 2], lambda v: v) is MIXED
    assert common_value([], lambda v: v) is MIXED


def test_mixed_spin_box_shows_blank_until_stepped(qtbot):
    spin = MixedSpinBox()
    qtbot.addWidget(spin)
    spin.setRange(0, 10)
    spin.setValue(2)
    spin.set_mixed(True)
    assert spin.lineEdit().text() == ""
    spin.stepBy(1)
    assert not spin.is_mixed()
    assert spin.value() == 3


def test_mixed_check_box_click_checks_and_emits_toggled(qtbot):
    box = MixedCheckBox("x")
    qtbot.addWidget(box)
    box.set_mixed(True)
    assert box.is_mixed()
    seen = []
    box.toggled.connect(seen.append)
    box.click()
    assert box.checkState() == Qt.CheckState.Checked
    assert seen == [True]


def test_scope_fields_show_mixed_status_and_set_all_in_one_undo(qt_app):
    clips = _three_clips(qt_app)
    clips[0].status = "approved"
    _select(qt_app, clips)
    fields = qt_app.settings_dock.scope_fields.widgets
    assert "take" not in fields and "source_text" not in fields
    status = fields["status"]
    assert status.currentText() == MIXED_LABEL

    stack = qt_app.document.undo_stack
    steps = len(stack._undo)
    status.setCurrentIndex(status.findData("needs_rewrite"))
    status.activated.emit(status.currentIndex())
    assert all(c.status == "needs_rewrite" for c in clips)
    assert len(stack._undo) == steps + 1
    stack.undo()
    assert clips[0].status == "approved" and clips[1].status != "needs_rewrite"


def test_scope_fields_mixed_gap_stays_untouched_until_edited(qt_app):
    clips = _three_clips(qt_app)
    clips[0].gap_before_s = 1.0
    _select(qt_app, clips)
    gap = qt_app.settings_dock.scope_fields.widgets["gap_before_s"]
    assert gap.is_mixed() and gap.lineEdit().text() == ""
    gap.editingFinished.emit()  # focus left without an edit
    assert [c.gap_before_s for c in clips] == [1.0, None, None]
    gap.setValue(0.5)
    gap.editingFinished.emit()
    assert [c.gap_before_s for c in clips] == [0.5, 0.5, 0.5]


def test_scope_fields_shared_note_shows_its_value(qt_app):
    clips = _three_clips(qt_app)
    for clip in clips:
        clip.note = "same"
    _select(qt_app, clips)
    note = qt_app.settings_dock.scope_fields.widgets["note"]
    assert note.text() == "same"


def test_settings_volume_mixed_then_edit_sets_every_clip(qt_app):
    clips = _three_clips(qt_app)
    clips[0].overrides["volume"] = 1.5
    _select(qt_app, clips)
    dock = qt_app.settings_dock
    assert dock.several
    assert dock.volume_spin.is_mixed()
    assert not dock.pitch_spin.is_mixed()

    stack = qt_app.document.undo_stack
    steps = len(stack._undo)
    dock.volume_spin.setValue(0.8)
    assert all(c.overrides["volume"] == 0.8 for c in clips)
    assert len(stack._undo) == steps + 1
    stack.undo()
    assert clips[0].overrides["volume"] == 1.5 and "volume" not in clips[1].overrides


def test_settings_normalize_mixed_is_partly_checked_and_a_click_sets_all(qt_app):
    clips = _three_clips(qt_app)
    clips[0].overrides["normalize"] = True
    clips[1].overrides["normalize"] = False
    _select(qt_app, clips)
    check = qt_app.settings_dock.normalize_check
    assert check.is_mixed()
    check.click()
    assert all(c.overrides["normalize"] is True for c in clips)


def test_settings_one_clip_again_drops_the_mixed_state(qt_app):
    clips = _three_clips(qt_app)
    clips[0].overrides["volume"] = 1.5
    _select(qt_app, clips)
    qt_app.selection.select_clip(clips[1].id)
    dock = qt_app.settings_dock
    assert not dock.several
    assert not dock.volume_spin.is_mixed()
    assert "take" in dock.scope_fields.widgets


def test_fx_dock_mixed_field_edit_sets_only_that_key_on_each_clip(qt_app):
    clips = _three_clips(qt_app)
    fx = qt_app.fx_dock
    clips[0].fx_override = {**fx.project_fx_state(), "gain_db": 3.0, "eq_bass": 2.0}
    _select(qt_app, clips)
    assert fx.several
    assert fx._value_widgets["gain_db"].is_mixed()
    assert not fx._value_widgets["eq_treble"].is_mixed()

    stack = qt_app.document.undo_stack
    steps = len(stack._undo)
    fx._value_widgets["gain_db"].setValue(-2.0)
    fx._flush_clip_edit()
    assert all(c.fx_override["gain_db"] == -2.0 for c in clips)
    assert clips[0].fx_override["eq_bass"] == 2.0  # each clip keeps its other values
    assert clips[1].fx_override["eq_bass"] != 2.0
    assert len(stack._undo) == steps + 1
    stack.undo()
    assert clips[0].fx_override["gain_db"] == 3.0 and not clips[1].fx_override
