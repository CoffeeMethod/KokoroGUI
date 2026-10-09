"""Listen-through mode (plan 27): the speed combo and keys, and the flags."""
import json

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from kokoro_gui.audio import transport as transport_mod
from kokoro_gui.daw import markers
from kokoro_gui.qt import keymap, settings as qt_settings
from kokoro_gui.qt.listen_through import nearest_step, step_rate

SHIFT = Qt.KeyboardModifier.ShiftModifier


def _show_active(qt_app, qtbot):
    qt_app.show()
    qtbot.waitExposed(qt_app)
    qt_app.activateWindow()
    qtbot.waitUntil(qt_app.isActiveWindow, timeout=2000)


def _focus_timeline(qt_app, qtbot):
    _show_active(qt_app, qtbot)
    view = qt_app.timeline_dock.timeline_view
    view.setFocus()
    # The transport needs a length for a seek to land.
    qt_app.transport._total_frames = qt_app.transport.sample_rate * 10
    return view


def _focus_editor(qt_app, qtbot):
    _show_active(qt_app, qtbot)
    qt_app.editor.setFocus()
    cursor = qt_app.editor.textCursor()
    cursor.setPosition(0)
    qt_app.editor.setTextCursor(cursor)
    return qt_app.editor


def _playing(qt_app, monkeypatch):
    """The transport reports playing without opening a stream."""
    monkeypatch.setattr(type(qt_app.transport), "is_playing", property(lambda self: True))


def _flags(qt_app):
    return markers.list_markers(qt_app.document.settings)


def _drop_flag(qt_app, view):
    QTest.keyClick(view, Qt.Key.Key_M)
    return qt_app._flag_popup


# --- the pure parts -----------------------------------------------------------------

def test_step_rate_walks_a_list_up_and_down_and_stops_at_the_ends():
    steps = (1.0, 1.5, 2.0)
    assert step_rate(1.0, steps, up=True) == 1.5
    assert step_rate(1.25, steps, up=True) == 1.5
    assert step_rate(2.0, steps, up=True) is None
    assert step_rate(1.5, steps, up=False) == 1.0
    assert step_rate(1.0, steps, up=False) is None


def test_nearest_step_snaps_a_hand_edited_value():
    assert nearest_step(1.1) == 1.0
    assert nearest_step(1.4) == 1.5
    assert nearest_step(50) == 2.0
    assert nearest_step("junk") == 1.0


def test_the_flag_helpers_pick_notes_and_flag_names_and_number_the_next_one():
    settings = {}
    settings["markers"], _a = markers.add_marker(settings, 1.0, "Flag 3")
    settings["markers"], _b = markers.add_marker(settings, 2.0, "Intro")
    settings["markers"], _c = markers.add_marker(settings, 3.0, "Chapter", "fix the name")

    assert [m["seconds"] for m in markers.list_flags(settings)] == [1.0, 3.0]
    assert markers.next_flag_name(settings) == "Flag 4"
    assert markers.next_flag_name({}) == "Flag 1"


# --- the speed combo ------------------------------------------------------------------

def test_the_combo_lists_the_steps_and_starts_at_one(qt_app):
    combo = qt_app.transport_dock.rate_combo
    assert [combo.itemText(i) for i in range(combo.count())] == ["0.5x", "0.75x", "1x", "1.25x", "1.5x", "1.75x", "2x"]
    assert qt_app.transport_dock.rate_choice() == 1.0
    assert qt_app.transport.rate == 1.0


def test_choosing_a_speed_sets_the_transport_and_remembers_it_outside_the_project(qt_app):
    import kokoro_gui.qt.app as qt_app_module

    combo = qt_app.transport_dock.rate_combo
    combo.setCurrentIndex(4)
    combo.activated.emit(4)

    assert qt_app.transport.rate == 1.5
    assert qt_app.settings["playback_rate"] == 1.5
    qt_app.save_settings()
    with open(qt_app_module.CONFIG_FILE, "r", encoding="utf-8") as f:
        assert json.load(f)["playback_rate"] == 1.5
    assert "playback_rate" not in qt_app.document.settings
    assert "playback_rate" not in qt_app.project_settings


def test_a_saved_speed_comes_back_at_launch_and_junk_falls_back_to_one(qt_app, tmp_path):
    qt_app.settings["playback_rate"] = 1.75
    qt_app.apply_saved_playback_rate()
    assert qt_app.transport.rate == 1.75
    assert qt_app.transport_dock.rate_choice() == 1.75

    config = tmp_path / "config_qt.json"
    config.write_text(json.dumps({"playback_rate": "fast"}), encoding="utf-8")
    assert qt_settings.load_settings(str(config))["playback_rate"] == 1.0
    qt_app.settings["playback_rate"] = 1.1
    qt_app.apply_saved_playback_rate()
    assert qt_app.transport.rate == 1.0


def test_a_speed_set_from_elsewhere_moves_the_combo(qt_app):
    qt_app.set_playback_rate(0.75)
    assert qt_app.transport_dock.rate_choice() == 0.75
    assert qt_app.transport.rate == 0.75


# --- the speed keys ------------------------------------------------------------------------

def test_the_bracket_keys_step_the_speed_on_the_timeline(qt_app, qtbot):
    view = _focus_timeline(qt_app, qtbot)

    QTest.keyClick(view, Qt.Key.Key_BracketRight)
    assert qt_app.transport.rate == 1.25
    for _ in range(3):
        QTest.keyClick(view, Qt.Key.Key_BracketRight)
    assert qt_app.transport.rate == 2.0
    QTest.keyClick(view, Qt.Key.Key_BracketRight)  # no faster step
    assert qt_app.transport.rate == 2.0
    assert "fastest" in qt_app.transport_dock.status_text()

    QTest.keyClick(view, Qt.Key.Key_BracketLeft)
    assert qt_app.transport.rate == 1.75
    assert qt_app.transport_dock.rate_choice() == 1.75


def test_the_slowest_step_is_half_speed(qt_app, qtbot):
    view = _focus_timeline(qt_app, qtbot)
    for _ in range(5):
        QTest.keyClick(view, Qt.Key.Key_BracketLeft)

    assert qt_app.transport.rate == 0.5
    assert "slowest" in qt_app.transport_dock.status_text()


def test_the_bracket_keys_in_the_transcript_type_brackets(qt_app, qtbot):
    editor = _focus_editor(qt_app, qtbot)

    QTest.keyClick(editor, Qt.Key.Key_BracketRight)

    assert qt_app.transport.rate == 1.0
    assert editor.toPlainText().startswith("]")


def test_l_plays_then_steps_through_one_point_five_and_two(qt_app, qtbot, monkeypatch):
    view = _focus_timeline(qt_app, qtbot)
    plays = []
    monkeypatch.setattr(qt_app.transport, "play", lambda: plays.append(qt_app.transport.rate))

    QTest.keyClick(view, Qt.Key.Key_L)
    assert plays == [1.0]  # stopped: it plays, at the speed it has

    _playing(qt_app, monkeypatch)
    QTest.keyClick(view, Qt.Key.Key_L)
    assert qt_app.transport.rate == 1.5
    QTest.keyClick(view, Qt.Key.Key_L)
    assert qt_app.transport.rate == 2.0
    QTest.keyClick(view, Qt.Key.Key_L)
    assert qt_app.transport.rate == 2.0
    assert plays == [1.0]


def test_l_while_stopped_plays_at_the_remembered_speed(qt_app, qtbot, monkeypatch):
    view = _focus_timeline(qt_app, qtbot)
    qt_app.set_playback_rate(1.75)
    plays = []
    monkeypatch.setattr(qt_app.transport, "play", lambda: plays.append(qt_app.transport.rate))

    QTest.keyClick(view, Qt.Key.Key_L)

    assert plays == [1.75]


# --- flags -----------------------------------------------------------------------------------

def test_m_while_playing_drops_a_flag_at_the_playhead_in_one_undo_step(qt_app, qtbot, monkeypatch):
    view = _focus_timeline(qt_app, qtbot)
    qt_app.transport.seek(2.5)
    _playing(qt_app, monkeypatch)

    _drop_flag(qt_app, view)

    found = _flags(qt_app)
    assert [(m["name"], m["seconds"], m["note"]) for m in found] == [("Flag 1", 2.5, "")]
    qt_app.undo()
    assert _flags(qt_app) == []
    assert qt_app.transport.is_playing  # the popup never touched the transport


def test_flags_are_numbered_on(qt_app, qtbot):
    view = _focus_timeline(qt_app, qtbot)
    for seconds in (1.0, 2.0):
        qt_app.transport.seek(seconds)
        _drop_flag(qt_app, view).hide()

    assert [m["name"] for m in _flags(qt_app)] == ["Flag 1", "Flag 2"]


def test_the_popup_opens_with_the_flag_and_enter_stores_the_note(qt_app, qtbot):
    view = _focus_timeline(qt_app, qtbot)
    qt_app.transport.seek(1.0)
    popup = _drop_flag(qt_app, view)
    assert popup.isVisible() and popup.label.text() == "Flag 1"

    QTest.keyClicks(popup.edit, "say it again")
    QTest.keyClick(popup.edit, Qt.Key.Key_Return)

    assert not popup.isVisible()
    assert _flags(qt_app)[0]["note"] == "say it again"
    assert _flags(qt_app)[0]["name"] == "Flag 1"
    qt_app.undo()  # the note is one step, the flag another
    assert _flags(qt_app)[0]["note"] == ""
    qt_app.undo()
    assert _flags(qt_app) == []


def test_esc_in_the_popup_keeps_the_flag_without_a_note(qt_app, qtbot):
    view = _focus_timeline(qt_app, qtbot)
    popup = _drop_flag(qt_app, view)

    QTest.keyClicks(popup.edit, "never mind")
    QTest.keyClick(popup.edit, Qt.Key.Key_Escape)

    assert not popup.isVisible()
    assert [(m["name"], m["note"]) for m in _flags(qt_app)] == [("Flag 1", "")]


def test_closing_the_popup_another_way_keeps_what_was_typed(qt_app, qtbot):
    view = _focus_timeline(qt_app, qtbot)
    popup = _drop_flag(qt_app, view)

    QTest.keyClicks(popup.edit, "typed")
    popup.hide()  # a click outside does this

    assert _flags(qt_app)[0]["note"] == "typed"


def test_n_and_shift_n_visit_flags_and_skip_plain_markers(qt_app, qtbot):
    view = _focus_timeline(qt_app, qtbot)
    settings = qt_app.document.settings
    for seconds, name, note in ((1.0, "Flag 1", ""), (2.0, "Intro", ""), (3.0, "Chapter", "fix"), (4.0, "Flag 2", "")):
        settings["markers"], _m = markers.add_marker(settings, seconds, name, note)
    qt_app.transport.seek(0.0)

    seen = []
    for _ in range(4):
        QTest.keyClick(view, Qt.Key.Key_N)
        seen.append(qt_app.transport.position())
    assert seen == [1.0, 3.0, 4.0, 4.0]
    assert "No flag after" in qt_app.transport_dock.status_text()

    QTest.keyClick(view, Qt.Key.Key_N, SHIFT)
    assert qt_app.transport.position() == pytest.approx(3.0)
    QTest.keyClick(view, Qt.Key.Key_N, SHIFT)
    QTest.keyClick(view, Qt.Key.Key_N, SHIFT)
    assert qt_app.transport.position() == pytest.approx(1.0)
    assert "No flag before" in qt_app.transport_dock.status_text()


def test_flag_keys_in_the_transcript_type_letters(qt_app, qtbot):
    editor = _focus_editor(qt_app, qtbot)

    for key in (Qt.Key.Key_M, Qt.Key.Key_N):
        QTest.keyClick(editor, key)

    assert editor.toPlainText().startswith("mn")
    assert _flags(qt_app) == []


def test_a_flag_sits_in_the_project_as_a_marker_with_a_note(qt_app, qtbot):
    view = _focus_timeline(qt_app, qtbot)
    popup = _drop_flag(qt_app, view)
    QTest.keyClicks(popup.edit, "re-record")
    QTest.keyClick(popup.edit, Qt.Key.Key_Return)

    stored = qt_app.document.settings["markers"]
    assert set(stored[0]) == {"id", "seconds", "name", "note"}
    assert stored[0]["note"] == "re-record"


def test_the_new_keys_are_in_the_table_and_the_sheet(qt_app):
    from kokoro_gui.qt.about_dialog import shortcuts_text

    ids = {b.id: b for b in keymap.KEYS}
    for key_id in ("rate_up", "rate_down", "flag", "flag_back", "flag_forward"):
        assert ids[key_id].scope == keymap.TIMELINE
    text = shortcuts_text(qt_app)
    for label in ("Drop a flag at the playhead", "Next flag", "Previous flag", "Playback speed up a step"):
        assert label in text
    assert set(transport_mod.RATE_STEPS) >= set(keymap.LISTEN_RATES)
