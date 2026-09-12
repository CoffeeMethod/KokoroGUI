"""Tests for the Transcript dock (Claude/PLAN_ui_shell_redesign.md section
2): the Character/FX header combos, the two-line gutter labels with per-clip
play buttons, the dirty underline, split rules, and the playing-clip
highlight."""
import json
import os

from PySide6.QtGui import QPaintEvent, QTextCharFormat, QTextCursor

from kokoro_gui.daw.dirty import build_segments_from_results, compute_expected_cache_hash
from kokoro_gui.daw.models import Character

import kokoro_gui.qt.app  # noqa: F401 - app.py must load before any docks module (circular import)
from kokoro_gui.qt.docks.transcript_dock import FX_EDIT_LABEL, FX_NONE_LABEL  # noqa: E402


def _type(editor, text):
    cursor = editor.textCursor()
    cursor.select(QTextCursor.SelectionType.Document)
    cursor.insertText(text)


def _place_caret(editor, position):
    cursor = editor.textCursor()
    cursor.setPosition(position)
    editor.setTextCursor(cursor)


def _repaint_gutter(editor):
    gutter = editor._gutter
    gutter.paintEvent(QPaintEvent(gutter.rect()))
    return gutter


def _mark_generated(qt_app, clip, path="x.wav"):
    text = qt_app.document.clip_text(clip)
    config = qt_app.document.effective_config_for_clip(clip)
    expected = compute_expected_cache_hash(text, config)
    clip.segments = build_segments_from_results(expected, [{"text": text, "path": path, "duration": 1.0}])


def _write_fx_preset(qt_app, name):
    import kokoro_gui.qt.app as qt_app_module

    os.makedirs(qt_app_module.FX_PRESETS_DIR, exist_ok=True)
    with open(os.path.join(qt_app_module.FX_PRESETS_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump({"reverb_enabled": True, "reverb_wet_level": 0.4}, f)
    qt_app.engine.load_fx_preset.return_value = {"reverb_enabled": True, "reverb_wet_level": 0.4}
    qt_app.transcript_dock.refresh_fx_choices()


# -- header combos --------------------------------------------------------------


def test_header_combos_follow_the_caret(qt_app):
    dock = qt_app.transcript_dock
    editor = dock.editor
    bob = Character.from_preset_dict("Bob", {"voice": "am_michael", "fx_preset": "Echo"})
    qt_app.document.characters.append(bob)
    dock.refresh_character_choices()
    _write_fx_preset(qt_app, "Echo")
    _type(editor, "hello world")
    alice = qt_app.document.characters[0]
    qt_app.document.assign_character_to_range(0, 5, alice.id)
    qt_app.document.assign_character_to_range(6, 11, bob.id)
    editor.rehighlight()

    _place_caret(editor, 2)
    assert dock.character_combo.currentData() == alice.id
    assert dock.fx_combo.currentText() == FX_NONE_LABEL

    _place_caret(editor, 8)
    assert dock.character_combo.currentData() == bob.id
    assert dock.fx_combo.currentText() == "Echo"  # inherited from Bob's preset


def test_character_combo_assigns_the_selection(qt_app):
    dock = qt_app.transcript_dock
    editor = dock.editor
    bob = Character.from_preset_dict("Bob", {})
    qt_app.document.characters.append(bob)
    dock.refresh_character_choices()
    _type(editor, "hello world")
    cursor = editor.textCursor()
    cursor.setPosition(6)
    cursor.setPosition(11, QTextCursor.MoveMode.KeepAnchor)
    editor.setTextCursor(cursor)

    index = dock.character_combo.findData(bob.id)
    dock.character_combo.setCurrentIndex(index)
    dock.character_combo.activated.emit(index)

    clip = qt_app.document.clip_covering(8)
    assert clip is not None and clip.character_id == bob.id
    assert qt_app.document.clip_covering(2) is None


def test_character_combo_without_selection_retargets_the_carets_whole_clip(qt_app):
    dock = qt_app.transcript_dock
    editor = dock.editor
    bob = Character.from_preset_dict("Bob", {})
    qt_app.document.characters.append(bob)
    dock.refresh_character_choices()
    _type(editor, "hello world")
    alice = qt_app.document.characters[0]
    qt_app.document.assign_character_to_range(0, 11, alice.id)
    editor.rehighlight()
    _place_caret(editor, 3)

    index = dock.character_combo.findData(bob.id)
    dock.character_combo.setCurrentIndex(index)
    dock.character_combo.activated.emit(index)

    assert qt_app.document.clip_covering(0).character_id == bob.id
    assert qt_app.document.clip_covering(10).character_id == bob.id
    assert qt_app.document.undo_stack.can_undo()


def test_fx_combo_sets_an_undoable_named_override(qt_app):
    dock = qt_app.transcript_dock
    editor = dock.editor
    _write_fx_preset(qt_app, "Telephone")
    _type(editor, "hello world")
    alice = qt_app.document.characters[0]
    clip = qt_app.document.assign_character_to_range(0, 11, alice.id)
    editor.rehighlight()
    _place_caret(editor, 3)

    index = dock.fx_combo.findData("Telephone")
    dock.fx_combo.setCurrentIndex(index)
    dock.fx_combo.activated.emit(index)

    assert clip.fx_override == {"reverb_enabled": True, "reverb_wet_level": 0.4}
    assert clip.overrides["fx_preset"] == "Telephone"
    assert dock.fx_combo.currentText() == "Telephone"
    qt_app.undo()
    assert clip.fx_override is None
    assert "fx_preset" not in clip.overrides


def test_fx_combo_edit_entry_raises_the_fx_tab(qt_app):
    dock = qt_app.transcript_dock
    raised = []
    qt_app.raise_fx_tab = lambda: raised.append(True)
    index = dock.fx_combo.findData("__edit__")
    assert dock.fx_combo.itemText(index) == FX_EDIT_LABEL
    dock.fx_combo.activated.emit(index)
    assert raised == [True]


def test_character_combo_manage_entry_opens_the_dialog(qt_app):
    dock = qt_app.transcript_dock
    opened = []
    qt_app.open_characters_dialog = lambda: opened.append(True)
    index = dock.character_combo.findData("__manage__")
    dock.character_combo.activated.emit(index)
    assert opened == [True]


# -- gutter -----------------------------------------------------------------------


def test_gutter_draws_a_play_button_for_each_dirty_clip_only(qt_app):
    editor = qt_app.editor
    alice = qt_app.document.characters[0]
    _type(editor, "line one\nline two\nline three")
    text = qt_app.document.text
    first_end = text.index("\n")
    second_end = text.index("\n", first_end + 1)
    dirty = qt_app.document.assign_character_to_range(0, first_end, alice.id)
    clean = qt_app.document.assign_character_to_range(first_end + 1, second_end, alice.id)
    _mark_generated(qt_app, clean)
    editor.rehighlight()

    gutter = _repaint_gutter(editor)

    assert [cid for _rect, cid in gutter.button_rects()] == [dirty.id]


def test_gutter_play_button_click_runs_scoped_generate(qt_app):
    editor = qt_app.editor
    alice = qt_app.document.characters[0]
    _type(editor, "hello world")
    clip = qt_app.document.assign_character_to_range(0, 11, alice.id)
    editor.rehighlight()
    gutter = _repaint_gutter(editor)
    requested = []
    qt_app.generate_clip = lambda cid: requested.append(cid)

    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    rect, cid = gutter.button_rects()[0]
    assert cid == clip.id
    pos = QPointF(rect.center())
    event = QMouseEvent(QEvent.Type.MouseButtonPress, pos, pos, Qt.MouseButton.LeftButton,
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    gutter.mousePressEvent(event)

    assert requested == [clip.id]


def test_gutter_key_includes_fx_so_a_preset_change_relabels(qt_app):
    editor = qt_app.editor
    alice = qt_app.document.characters[0]
    _write_fx_preset(qt_app, "Echo")
    _type(editor, "line one\nline two")
    text = qt_app.document.text
    first_end = text.index("\n")
    qt_app.document.assign_character_to_range(0, first_end, alice.id)
    second = qt_app.document.assign_character_to_range(first_end + 1, len(text), alice.id)
    editor.rehighlight()
    assert len(_repaint_gutter(editor)._label_rects) == 1  # same character, same (no) FX

    second.overrides["fx_preset"] = "Echo"
    assert len(_repaint_gutter(editor)._label_rects) == 2  # FX differs -> new label


# -- dirty underline -------------------------------------------------------------------


def _underline_at(editor, position):
    block = editor.document().findBlock(position)
    offset = position - block.position()
    for fmt_range in block.layout().formats():
        if fmt_range.start <= offset < fmt_range.start + fmt_range.length:
            return fmt_range.format.underlineStyle()
    return None


def test_dirty_clip_text_is_dash_underlined_and_clean_text_is_not(qt_app):
    editor = qt_app.editor
    alice = qt_app.document.characters[0]
    _type(editor, "hello world")
    dirty = qt_app.document.assign_character_to_range(0, 5, alice.id)
    clean = qt_app.document.assign_character_to_range(6, 11, alice.id)
    _mark_generated(qt_app, clean)
    editor.rehighlight()

    assert _underline_at(editor, 2) == QTextCharFormat.UnderlineStyle.DashUnderline
    assert _underline_at(editor, 8) == QTextCharFormat.UnderlineStyle.NoUnderline
    del dirty


# -- split rules ---------------------------------------------------------------------------


def test_split_rules_mark_clip_boundaries_and_planned_splits(qt_app):
    editor = qt_app.editor
    bob = Character.from_preset_dict("Bob", {})
    qt_app.document.characters.append(bob)
    _type(editor, "[Bob]: first para.\n\nsecond para.\n\n[Default]: third.")
    qt_app.settings["auto_split_by_paragraph"] = True

    editor.refresh_split_rules()

    text = qt_app.document.text
    boundaries = editor.split_boundaries()
    assert text.index("second") in boundaries
    assert text.index("[Default]") in boundaries
    assert 0 not in boundaries and len(text) not in boundaries


def test_split_rules_refresh_after_an_edit_is_debounced(qt_app, qtbot):
    editor = qt_app.editor
    alice = qt_app.document.characters[0]
    _type(editor, "hello world")
    qt_app.document.assign_character_to_range(0, 5, alice.id)
    editor.rehighlight()
    assert editor.split_boundaries() == [5]

    cursor = editor.textCursor()
    cursor.setPosition(0)
    cursor.insertText("XX")  # untagged insert at the very start shifts the clip to [2, 7)
    qtbot.waitUntil(lambda: editor.split_boundaries() == [2, 7], timeout=2000)


# -- playing clip --------------------------------------------------------------------------


def test_playing_clip_gets_an_extra_selection_without_moving_the_caret(qt_app):
    editor = qt_app.editor
    alice = qt_app.document.characters[0]
    _type(editor, "hello world")
    clip = qt_app.document.assign_character_to_range(6, 11, alice.id)
    editor.rehighlight()
    _place_caret(editor, 1)

    qt_app.selection.set_playing_clip(clip.id)

    selections = editor.extraSelections()
    assert len(selections) == 1
    assert (selections[0].cursor.selectionStart(), selections[0].cursor.selectionEnd()) == (6, 11)
    assert editor.textCursor().position() == 1
    assert qt_app.selection.selected_clip_id is None  # playback never selects

    qt_app.selection.set_playing_clip(None)
    assert editor.extraSelections() == []
