"""The keyboard map (plan 11): plain keys only with the timeline focused,
Ctrl twins anywhere, Esc only while a generate runs."""
import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QFileDialog

from kokoro_gui.daw import markers
from kokoro_gui.qt import keymap
from kokoro_gui.qt.about_dialog import shortcuts_text

TEXT = "alpha alpha alpha. beta beta beta. gamma gamma gamma."
CTRL = Qt.KeyboardModifier.ControlModifier
CTRL_ALT = CTRL | Qt.KeyboardModifier.AltModifier
SHIFT = Qt.KeyboardModifier.ShiftModifier


def _type(editor, text):
    cursor = editor.textCursor()
    cursor.select(QTextCursor.SelectionType.Document)
    cursor.insertText(text)


def _three_clips(qt_app):
    """Three clips with a loaded transport; returns their start times."""
    _type(qt_app.editor, TEXT)
    alice = qt_app.document.characters[0]
    cuts = [0, TEXT.index("beta"), TEXT.index("gamma"), len(TEXT)]
    for start, end in zip(cuts, cuts[1:]):
        qt_app.document.assign_character_to_range(start, end, alice.id)
    qt_app.editor.rehighlight()
    qt_app.refresh_timeline()
    qt_app._schedule_timer.stop()
    qt_app._rebuild_transport_schedule()
    starts = sorted(p.start_s for p in qt_app.current_arrangement().placed)
    assert len(starts) == 3 and starts[0] == 0.0 and starts[1] > 0.5
    assert qt_app.transport.duration() > starts[2]
    return starts


def _show_active(qt_app, qtbot):
    qt_app.show()
    qtbot.waitExposed(qt_app)
    qt_app.activateWindow()
    qtbot.waitUntil(qt_app.isActiveWindow, timeout=2000)


def _focus_timeline(qt_app, qtbot):
    _show_active(qt_app, qtbot)
    view = qt_app.timeline_dock.timeline_view
    view.setFocus()
    return view


def _focus_editor(qt_app, qtbot):
    _show_active(qt_app, qtbot)
    qt_app.editor.setFocus()
    cursor = qt_app.editor.textCursor()
    cursor.setPosition(0)
    qt_app.editor.setTextCursor(cursor)
    return qt_app.editor


def test_the_timeline_takes_the_focus_when_clicked(qt_app):
    view = qt_app.timeline_dock.timeline_view
    assert view.focusPolicy() in (Qt.FocusPolicy.StrongFocus, Qt.FocusPolicy.ClickFocus)


# --- clip starts -----------------------------------------------------------------

def test_right_on_the_timeline_moves_the_playhead_to_the_next_clip_start(qt_app, qtbot):
    starts = _three_clips(qt_app)
    view = _focus_timeline(qt_app, qtbot)
    qt_app.transport.seek(0.0)

    QTest.keyClick(view, Qt.Key.Key_Right)
    assert qt_app.transport.position() == pytest.approx(starts[1], abs=1e-3)
    QTest.keyClick(view, Qt.Key.Key_Right)
    assert qt_app.transport.position() == pytest.approx(starts[2], abs=1e-3)
    QTest.keyClick(view, Qt.Key.Key_Right)  # no clip after the last: stays
    assert qt_app.transport.position() == pytest.approx(starts[2], abs=1e-3)

    QTest.keyClick(view, Qt.Key.Key_Left)
    assert qt_app.transport.position() == pytest.approx(starts[1], abs=1e-3)


def test_left_from_the_first_clip_goes_to_zero_and_from_zero_stays(qt_app, qtbot):
    starts = _three_clips(qt_app)
    view = _focus_timeline(qt_app, qtbot)
    qt_app.transport.seek(starts[0] + 0.4)
    QTest.keyClick(view, Qt.Key.Key_Left)
    assert qt_app.transport.position() == 0.0
    QTest.keyClick(view, Qt.Key.Key_Left)
    assert qt_app.transport.position() == 0.0


def test_right_with_the_transcript_focused_moves_the_caret_and_not_the_playhead(qt_app, qtbot):
    _three_clips(qt_app)
    editor = _focus_editor(qt_app, qtbot)
    qt_app.transport.seek(0.0)

    QTest.keyClick(editor, Qt.Key.Key_Right)

    assert editor.textCursor().position() == 1
    assert qt_app.transport.position() == 0.0


def test_ctrl_alt_right_works_with_the_transcript_focused(qt_app, qtbot):
    starts = _three_clips(qt_app)
    editor = _focus_editor(qt_app, qtbot)
    qt_app.transport.seek(0.0)

    QTest.keyClick(editor, Qt.Key.Key_Right, CTRL_ALT)

    assert qt_app.transport.position() == pytest.approx(starts[1], abs=1e-3)
    assert editor.textCursor().position() == 0
    QTest.keyClick(editor, Qt.Key.Key_Left, CTRL_ALT)
    assert qt_app.transport.position() == 0.0


# --- markers ---------------------------------------------------------------------

def _add_markers(qt_app, *seconds):
    for s in seconds:
        qt_app.document.settings["markers"], _m = markers.add_marker(qt_app.document.settings, s)


def test_shift_arrows_jump_between_markers(qt_app, qtbot):
    _three_clips(qt_app)
    _add_markers(qt_app, 0.7, 1.9)
    view = _focus_timeline(qt_app, qtbot)
    qt_app.transport.seek(0.0)

    QTest.keyClick(view, Qt.Key.Key_Right, SHIFT)
    assert qt_app.transport.position() == pytest.approx(0.7, abs=1e-3)
    QTest.keyClick(view, Qt.Key.Key_Right, SHIFT)
    assert qt_app.transport.position() == pytest.approx(1.9, abs=1e-3)
    QTest.keyClick(view, Qt.Key.Key_Left, SHIFT)
    assert qt_app.transport.position() == pytest.approx(0.7, abs=1e-3)


def test_a_marker_jump_with_no_marker_that_way_says_so_and_stays(qt_app, qtbot):
    _three_clips(qt_app)
    _add_markers(qt_app, 0.7)
    view = _focus_timeline(qt_app, qtbot)
    qt_app.transport.seek(0.7)

    QTest.keyClick(view, Qt.Key.Key_Right, SHIFT)

    assert qt_app.transport.position() == pytest.approx(0.7, abs=1e-3)
    assert "No marker after" in qt_app.transport_dock.status_text()


def test_ctrl_alt_shift_arrows_jump_between_markers_from_the_transcript(qt_app, qtbot):
    _three_clips(qt_app)
    _add_markers(qt_app, 0.7)
    editor = _focus_editor(qt_app, qtbot)
    qt_app.transport.seek(0.0)

    QTest.keyClick(editor, Qt.Key.Key_Right, CTRL_ALT | SHIFT)

    assert qt_app.transport.position() == pytest.approx(0.7, abs=1e-3)
    assert editor.textCursor().position() == 0 and not editor.textCursor().hasSelection()


# --- start, end, shuttle -----------------------------------------------------------

def test_home_stops_and_returns_to_the_start_and_end_goes_to_the_end(qt_app, qtbot):
    _three_clips(qt_app)
    view = _focus_timeline(qt_app, qtbot)
    qt_app.transport.seek(1.0)

    QTest.keyClick(view, Qt.Key.Key_End)
    assert qt_app.transport.position() == pytest.approx(qt_app.transport.duration(), abs=1e-3)
    QTest.keyClick(view, Qt.Key.Key_Home)
    assert qt_app.transport.position() == 0.0
    assert qt_app.transport.state == "stopped"


def test_home_and_end_in_the_transcript_move_the_caret_only(qt_app, qtbot):
    _three_clips(qt_app)
    editor = _focus_editor(qt_app, qtbot)
    qt_app.transport.seek(1.0)

    QTest.keyClick(editor, Qt.Key.Key_End)

    assert editor.textCursor().position() > 0
    assert qt_app.transport.position() == pytest.approx(1.0, abs=1e-3)


def test_ctrl_alt_home_and_end_work_from_the_transcript(qt_app, qtbot):
    _three_clips(qt_app)
    editor = _focus_editor(qt_app, qtbot)
    QTest.keyClick(editor, Qt.Key.Key_End, CTRL_ALT)
    assert qt_app.transport.position() == pytest.approx(qt_app.transport.duration(), abs=1e-3)
    QTest.keyClick(editor, Qt.Key.Key_Home, CTRL_ALT)
    assert qt_app.transport.position() == 0.0


def test_j_k_l_jump_back_pause_and_play_on_the_timeline(qt_app, qtbot):
    _three_clips(qt_app)
    view = _focus_timeline(qt_app, qtbot)
    calls = []
    qt_app.transport.play = lambda: calls.append("play")
    qt_app.transport.pause = lambda: calls.append("pause")

    QTest.keyClick(view, Qt.Key.Key_L)
    QTest.keyClick(view, Qt.Key.Key_K)
    assert calls == ["play", "pause"]

    qt_app.transport.seek(qt_app.transport.duration())
    end = qt_app.transport.position()
    QTest.keyClick(view, Qt.Key.Key_J)
    assert qt_app.transport.position() == pytest.approx(max(0.0, end - keymap.JUMP_BACK_S), abs=1e-3)


def test_j_k_l_in_the_transcript_type_letters_and_leave_the_transport_alone(qt_app, qtbot):
    _three_clips(qt_app)
    editor = _focus_editor(qt_app, qtbot)
    calls = []
    qt_app.transport.play = lambda: calls.append("play")
    qt_app.transport.pause = lambda: calls.append("pause")
    before = editor.toPlainText()

    for key in (Qt.Key.Key_J, Qt.Key.Key_K, Qt.Key.Key_L):
        QTest.keyClick(editor, key)

    assert calls == []
    assert editor.toPlainText() == "jkl" + before


# --- zoom, snap, delete ------------------------------------------------------------

def test_f_zooms_the_timeline_to_fit_and_a_letter_f_in_the_transcript_types(qt_app, qtbot, monkeypatch):
    _three_clips(qt_app)
    view = _focus_timeline(qt_app, qtbot)
    fits = []
    monkeypatch.setattr(view, "zoom_to_fit", lambda: fits.append(True))
    QTest.keyClick(view, Qt.Key.Key_F)
    assert fits == [True]

    editor = _focus_editor(qt_app, qtbot)
    QTest.keyClick(editor, Qt.Key.Key_F)
    assert fits == [True]
    assert editor.toPlainText().startswith("f")


def test_g_flips_snap_to_grid_on_the_timeline_only(qt_app, qtbot):
    view = _focus_timeline(qt_app, qtbot)
    dock = qt_app.timeline_dock
    assert view.snap_to_grid is False

    QTest.keyClick(view, Qt.Key.Key_G)
    assert view.snap_to_grid is True and dock.snap_button.isChecked()
    assert qt_app.settings["snap_to_grid"] is True
    QTest.keyClick(view, Qt.Key.Key_G)
    assert view.snap_to_grid is False

    editor = _focus_editor(qt_app, qtbot)
    QTest.keyClick(editor, Qt.Key.Key_G)
    assert view.snap_to_grid is False
    assert editor.toPlainText().startswith("g")


def _bed(qt_app, tmp_path, monkeypatch):
    path = str(tmp_path / "Theme Song.wav")
    sf.write(path, np.full(24000 * 4, 0.1, dtype=np.float32), 24000)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (path, "")))
    monkeypatch.setattr(qt_app, "_ask_audio_import", lambda path: {"kind": "bed"})
    monkeypatch.setattr(qt_app, "_ask_bed_placement", lambda playhead: 0.0)
    qt_app.document.text = "Intro."
    qt_app.editor.load_text(qt_app.document.text)
    before = {c.id for c in qt_app.document.clips}
    qt_app.import_audio_dialog()
    return next(c for c in qt_app.document.clips if c.id not in before)


def _select(qt_app, clip_id):
    qt_app.selection.select_clip(clip_id)
    assert qt_app.selection.selected_clip_id == clip_id


def test_delete_on_the_timeline_removes_the_selected_music_bed_in_one_undo_step(qt_app, qtbot, tmp_path, monkeypatch):
    bed = _bed(qt_app, tmp_path, monkeypatch)
    _select(qt_app, bed.id)
    view = _focus_timeline(qt_app, qtbot)

    QTest.keyClick(view, Qt.Key.Key_Delete)

    assert qt_app.document.get_clip(bed.id) is None
    qt_app.undo()
    assert qt_app.document.get_clip(bed.id) is not None


def test_delete_leaves_a_speech_clip_alone_and_says_why(qt_app, qtbot):
    _three_clips(qt_app)
    clip = qt_app.document.clips[0]
    _select(qt_app, clip.id)
    view = _focus_timeline(qt_app, qtbot)

    QTest.keyClick(view, Qt.Key.Key_Delete)

    assert qt_app.document.get_clip(clip.id) is not None
    assert qt_app.document.text == TEXT
    assert "music bed" in qt_app.transport_dock.status_text()


def test_delete_in_the_transcript_deletes_text_not_a_bed(qt_app, qtbot, tmp_path, monkeypatch):
    bed = _bed(qt_app, tmp_path, monkeypatch)
    _select(qt_app, bed.id)
    editor = _focus_editor(qt_app, qtbot)
    removed = []
    monkeypatch.setattr(qt_app.timeline_dock, "on_bed_action_requested", lambda *a: removed.append(a))

    QTest.keyClick(editor, Qt.Key.Key_Delete)

    assert removed == []


# --- generate ------------------------------------------------------------------------

def test_ctrl_g_generates_stale_clips_from_any_panel(qt_app, qtbot, monkeypatch):
    calls = []
    monkeypatch.setattr(qt_app, "on_generate_clicked", lambda: calls.append("generate"))
    editor = _focus_editor(qt_app, qtbot)

    QTest.keyClick(editor, Qt.Key.Key_G, CTRL)

    assert calls == ["generate"]


def test_ctrl_g_does_nothing_while_a_job_runs(qt_app, qtbot, monkeypatch):
    calls = []
    monkeypatch.setattr(qt_app, "on_generate_clicked", lambda: calls.append("generate"))
    qt_app.transport_dock.set_busy(True)
    editor = _focus_editor(qt_app, qtbot)

    QTest.keyClick(editor, Qt.Key.Key_G, CTRL)

    assert calls == []
    qt_app.transport_dock.set_busy(False)


def test_esc_cancels_a_running_generate(qt_app, qtbot, monkeypatch):
    cancelled = []
    monkeypatch.setattr(qt_app, "cancel_conversion", lambda: cancelled.append(True))
    editor = _focus_editor(qt_app, qtbot)
    qt_app.set_ui_state(True)
    assert qt_app.cancel_generate_shortcut.isEnabled()

    QTest.keyClick(editor, Qt.Key.Key_Escape)

    assert cancelled == [True]
    qt_app.set_ui_state(False)
    assert not qt_app.cancel_generate_shortcut.isEnabled()


def test_esc_while_idle_does_not_cancel_anything(qt_app, qtbot, monkeypatch):
    cancelled = []
    monkeypatch.setattr(qt_app, "cancel_conversion", lambda: cancelled.append(True))
    editor = _focus_editor(qt_app, qtbot)

    QTest.keyClick(editor, Qt.Key.Key_Escape)

    assert cancelled == []


# --- the sheet --------------------------------------------------------------------------

def test_the_shortcut_sheet_lists_every_binding(qt_app):
    text = shortcuts_text(qt_app)
    for binding in keymap.KEYS:
        assert binding.label in text, binding.id
    assert "Ctrl+Alt+Shift+Right" in text
    assert "Snap to grid" in text
