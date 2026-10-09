"""The Queue dock (plan 28): one row per queue item, drag to reorder, Move to
top, Remove, Pause / Resume and Cancel all (kokoro_gui/qt/docks/queue_dock.py)."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

import kokoro_gui.qt.app as qt_app_module
from kokoro_gui.daw import genqueue
from tests.gui_qt.test_generation_queue import Batches, _many_clips, _saved


def _show(qt_app):
    """The window shown and the Queue tab in front, so the dock rebuilds
    itself on every queue change."""
    qt_app.show()
    dock = qt_app.queue_dock
    dock.raise_()
    QApplication.processEvents()
    assert dock.isVisible()
    dock.rebuild()
    return dock


def _rows(dock):
    return [[dock.tree.topLevelItem(r).text(c) for c in range(4)] for r in range(dock.tree.topLevelItemCount())]


def _paused_after_first(qt_app, tmp_path, count=20):
    clips = _many_clips(qt_app, count)
    batches = Batches(qt_app, tmp_path)
    qt_app.on_generate_clicked()
    qt_app.pause_queue()
    batches.resolve(0)
    qt_app.wait_for_queue()
    return clips, batches


def _row_id(dock, row):
    return dock.tree.topLevelItem(row).data(0, Qt.ItemDataRole.UserRole)


def test_the_dock_sits_in_the_top_right_group(qt_app):
    dock = qt_app.queue_dock
    assert dock.objectName() == "dock_queue" and dock.windowTitle() == "Queue"
    assert not dock.isFloating()
    assert qt_app.dockWidgetArea(dock) == Qt.DockWidgetArea.TopDockWidgetArea
    assert dock in qt_app.tabifiedDockWidgets(qt_app.settings_dock)
    assert dock in qt_app._all_docks()


def test_an_empty_queue_says_so_and_its_buttons_are_off(qt_app):
    dock = _show(qt_app)

    assert dock.tree.topLevelItemCount() == 0
    assert "Nothing queued" in dock.summary.text()
    assert not dock.pause_button.isEnabled() and not dock.cancel_button.isEnabled()
    assert not qt_app.transport_dock.queue_pause_btn.isEnabled()


def test_rows_show_the_items_with_their_state(qt_app, tmp_path):
    _many_clips(qt_app, 20)
    Batches(qt_app, tmp_path)
    dock = _show(qt_app)

    qt_app.on_generate_clicked()

    rows = _rows(dock)
    assert [r[1] for r in rows] == ["8", "8", "4"]
    assert [r[2] for r in rows] == ["Generating", "Queued", "Queued"]
    assert rows[0][0].endswith("clips 1-8 of 20")
    assert "0 of 20 clips" in dock.summary.text()
    assert dock.pause_button.isEnabled() and dock.pause_button.text() == "Pause"


def test_the_time_left_shows_per_item_when_the_engine_speed_is_known(qt_app, tmp_path, monkeypatch):
    from kokoro_gui.daw import arrangement

    monkeypatch.setattr(arrangement, "recorded_chars_per_second", lambda engine_id: 1.0)
    _many_clips(qt_app, 20)
    Batches(qt_app, tmp_path)
    dock = _show(qt_app)

    qt_app.on_generate_clicked()

    rows = _rows(dock)
    assert rows[0][3] == "4 min"  # 8 clips of 23 characters at 1 character a second
    assert rows[2][3] == "2 min"
    assert "about" in dock.summary.text() and "left" in dock.summary.text()


def test_the_time_left_is_empty_without_a_recorded_speed(qt_app, tmp_path, monkeypatch):
    from kokoro_gui.daw import arrangement

    monkeypatch.setattr(arrangement, "recorded_chars_per_second", lambda engine_id: None)
    _many_clips(qt_app, 10)
    Batches(qt_app, tmp_path)
    dock = _show(qt_app)

    qt_app.on_generate_clicked()

    assert [r[3] for r in _rows(dock)] == ["", ""]


def test_dropping_a_queued_row_moves_the_item(qt_app, tmp_path):
    clips, batches = _paused_after_first(qt_app, tmp_path)
    dock = _show(qt_app)
    last = _row_id(dock, 2)

    dock._on_dropped(last, 1)  # before the second row

    assert [i.clip_count for i in qt_app.generation_queue.items] == [8, 4, 8]
    assert _row_id(_show(qt_app), 1) == last
    dock._on_dropped(last, dock.tree.topLevelItemCount())  # after the last row
    assert [i.clip_count for i in qt_app.generation_queue.items] == [8, 8, 4]
    assert qt_app.generation_queue.items[2].id == last


def test_a_drop_cannot_put_an_item_above_a_finished_one_or_move_a_finished_one(qt_app, tmp_path):
    _paused_after_first(qt_app, tmp_path)
    dock = _show(qt_app)
    done, second = _row_id(dock, 0), _row_id(dock, 1)

    dock._on_dropped(second, 0)  # above the finished row: stays below it
    dock._on_dropped(done, 2)  # a finished row doesn't move

    items = qt_app.generation_queue.items
    assert items[0].id == done and items[0].state == genqueue.DONE
    assert items[1].id == second


def test_only_queued_rows_can_be_dragged(qt_app, tmp_path):
    _paused_after_first(qt_app, tmp_path)
    dock = _show(qt_app)

    flags = [dock.tree.topLevelItem(r).flags() & Qt.ItemFlag.ItemIsDragEnabled for r in range(3)]

    assert [bool(f) for f in flags] == [False, True, True]


def _menu(dock, row):
    item = dock.app.generation_queue.find(_row_id(dock, row))
    return dock.build_menu(item), item


def _choose(dock, row, label):
    menu, _item = _menu(dock, row)
    next(a for a in menu.actions() if a.text() == label).trigger()


def test_move_to_top_in_the_context_menu(qt_app, tmp_path):
    _paused_after_first(qt_app, tmp_path)
    dock = _show(qt_app)
    last = _row_id(dock, 2)

    _choose(dock, 2, "Move to top")

    assert qt_app.generation_queue.items[1].id == last
    assert _row_id(dock, 1) == last


def test_remove_in_the_context_menu_drops_the_item_and_its_clips_stay_stale(qt_app, tmp_path):
    clips, _batches = _paused_after_first(qt_app, tmp_path)
    dock = _show(qt_app)

    _choose(dock, 1, "Remove")

    assert [i.clip_count for i in qt_app.generation_queue.items] == [8, 4]
    assert dock.tree.topLevelItemCount() == 2
    assert len(qt_app.document.dirty_clips()) == 12


def test_the_context_menu_is_not_offered_for_a_finished_item(qt_app, tmp_path):
    _paused_after_first(qt_app, tmp_path)
    dock = _show(qt_app)

    assert _menu(dock, 0)[0] is None
    menu, _item = _menu(dock, 1)
    assert [a.text() for a in menu.actions()] == ["Move to top", "Remove"]


def test_pause_and_resume_buttons_follow_the_queue(qt_app, tmp_path):
    _many_clips(qt_app, 20)
    batches = Batches(qt_app, tmp_path)
    dock = _show(qt_app)
    qt_app.on_generate_clicked()

    dock.pause_button.click()
    assert qt_app.queue_paused
    assert dock.pause_button.text() == "Resume"
    assert qt_app.transport_dock.queue_pause_btn.text() == "Resume"
    batches.resolve(0)
    qt_app.wait_for_queue()
    assert dock.pause_button.text() == "Resume generating 12 clips"

    dock.pause_button.click()
    assert not qt_app.queue_paused and batches.dispatched == 2
    assert dock.pause_button.text() == "Pause"
    assert qt_app.transport_dock.queue_pause_btn.text() == "Pause"


def test_the_transport_pause_button_drives_the_same_queue(qt_app, tmp_path):
    _many_clips(qt_app, 20)
    batches = Batches(qt_app, tmp_path)
    qt_app.on_generate_clicked()

    qt_app.transport_dock.queue_pause_btn.click()
    batches.resolve(0)
    qt_app.wait_for_queue()
    assert qt_app.queue_paused and batches.dispatched == 1

    qt_app.transport_dock.queue_pause_btn.click()
    assert batches.dispatched == 2


def test_cancel_all_stops_a_running_queue_and_clears_a_paused_one(qt_app, tmp_path):
    _many_clips(qt_app, 20)
    batches = Batches(qt_app, tmp_path)
    dock = _show(qt_app)
    qt_app.on_generate_clicked()

    dock.cancel_button.click()
    assert qt_app.engine.cancel.called
    batches.resolve(0, cancelled=set(batches.ids(0)))
    qt_app.wait_for_queue()
    assert [r[2] for r in _rows(dock)] == ["Cancelled"] * 3
    assert not dock.cancel_button.isEnabled()

    qt_app.on_generate_clicked()  # a new run, then pause and cancel it while idle
    qt_app.pause_queue()
    batches.resolve(1)
    qt_app.wait_for_queue()
    assert qt_app.generation_queue.pending()
    dock.cancel_button.click()
    assert qt_app.generation_queue.pending() == []


def test_a_restored_queue_offers_resume_in_the_dock_without_a_dialog(qt_app, qtbot, tmp_path):
    clips = _saved(qt_app, tmp_path)
    batches = Batches(qt_app, tmp_path)
    qt_app.on_generate_clicked()
    qt_app.pause_queue()
    batches.resolve(0, fail={c.id for c in clips[:8]})
    qt_app.wait_for_queue()
    qt_app.close()

    second = qt_app_module.QtTTSApp()
    qtbot.addWidget(second)
    second.wait_for_project_io()

    dock = second.queue_dock
    dock.rebuild()
    assert dock.pause_button.text() == "Resume generating 12 clips"
    assert dock.pause_button.isEnabled()
    assert [r[2] for r in _rows(dock)] == ["Queued", "Queued"]
    assert second.transport_dock.queue_pause_btn.text() == "Resume"
    assert not second.is_busy()


def test_a_hidden_dock_builds_nothing_until_shown(qt_app, tmp_path):
    _many_clips(qt_app, 10)
    Batches(qt_app, tmp_path)
    dock = qt_app.queue_dock

    qt_app.on_generate_clicked()
    assert dock.tree.topLevelItemCount() == 0  # the window isn't shown

    _show(qt_app)
    assert dock.tree.topLevelItemCount() == 2
