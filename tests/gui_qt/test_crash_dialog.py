"""kokoro_gui/qt/crash_dialog.py: the dialog and the bridge that opens it
from any thread, and Help > Open Log Folder (plan 07)."""
import os
import threading

import pytest
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

import kokoro_gui.qt.app as app_module
from kokoro_gui import logging_setup
from kokoro_gui.qt import crash_dialog
from kokoro_gui.qt.crash_dialog import CrashBridge, CrashDialog

FAKE_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "somewhere.py", line 3, in boom\n'
    "ValueError: boom\n"
)


@pytest.fixture
def bridge(qtbot, tmp_path):
    bridge = CrashBridge(str(tmp_path / "logs" / "kokorogui.log"))
    yield bridge
    if bridge.dialog is not None:
        bridge.dialog.close()


def _open_dialogs():
    return [w for w in QApplication.topLevelWidgets() if isinstance(w, CrashDialog) and w.isVisible()]


def test_copy_puts_the_traceback_on_the_clipboard(qtbot, tmp_path):
    dialog = CrashDialog(FAKE_TRACEBACK, str(tmp_path / "kokorogui.log"))
    qtbot.addWidget(dialog)
    dialog.copy_button.click()
    assert QGuiApplication.clipboard().text() == FAKE_TRACEBACK.rstrip("\n") or QGuiApplication.clipboard().text() == FAKE_TRACEBACK


def test_dialog_shows_the_message_the_traceback_and_three_buttons(qtbot, tmp_path):
    dialog = CrashDialog(FAKE_TRACEBACK, str(tmp_path / "kokorogui.log"))
    qtbot.addWidget(dialog)
    assert dialog.text_edit.isReadOnly()
    assert "ValueError: boom" in dialog.text()
    assert [b.text() for b in (dialog.copy_button, dialog.open_log_button, dialog.close_button)] == [
        "Copy", "Open Log Folder", "Close"]
    assert crash_dialog.MESSAGE == "Something went wrong. The details are saved in the log."


def test_open_log_folder_reveals_the_log_file(qtbot, tmp_path, monkeypatch):
    log = tmp_path / "kokorogui.log"
    log.write_text("x")
    seen = []
    monkeypatch.setattr(crash_dialog, "reveal", lambda path: seen.append(path) or True)
    dialog = CrashDialog("t", str(log))
    qtbot.addWidget(dialog)
    dialog.open_log_button.click()
    assert seen == [str(log)]


def test_open_log_folder_falls_back_to_the_folder_when_the_file_is_gone(qtbot, tmp_path, monkeypatch):
    log = tmp_path / "kokorogui.log"
    seen = []
    monkeypatch.setattr(crash_dialog, "reveal", lambda path: seen.append(path) or path == str(tmp_path))
    dialog = CrashDialog("t", str(log))
    qtbot.addWidget(dialog)
    dialog.open_log_button.click()
    assert seen == [str(log), str(tmp_path)]


def test_the_close_button_closes_the_dialog_and_nothing_else(qtbot, bridge):
    dialog = bridge.present(FAKE_TRACEBACK)
    assert dialog.isVisible()
    dialog.close_button.click()
    assert not dialog.isVisible()
    assert bridge.dialog is None


def test_the_bridge_signal_opens_a_crash_dialog(qtbot, bridge):
    bridge.crashed.emit(FAKE_TRACEBACK)
    assert bridge.dialog is not None and bridge.dialog.isVisible()
    assert "ValueError: boom" in bridge.dialog.text()
    assert len(_open_dialogs()) == 1


def test_a_second_exception_while_the_dialog_is_open_opens_no_second_dialog(qtbot, bridge):
    bridge.crashed.emit(FAKE_TRACEBACK)
    first = bridge.dialog
    bridge.crashed.emit("RuntimeError: another\n")
    assert bridge.dialog is first
    assert len(_open_dialogs()) == 1
    assert "another" not in first.text()


def test_a_dialog_opens_again_after_the_first_is_closed(qtbot, bridge):
    bridge.crashed.emit(FAKE_TRACEBACK)
    bridge.dialog.close_button.click()
    bridge.crashed.emit("RuntimeError: later\n")
    assert bridge.dialog is not None and "later" in bridge.dialog.text()


def test_report_from_a_worker_thread_builds_the_dialog_in_the_gui_thread(qtbot, bridge):
    seen = {}
    original = CrashDialog.__init__

    def spy(self, *args, **kwargs):
        seen["thread"] = threading.current_thread()
        original(self, *args, **kwargs)

    crash_dialog.CrashDialog.__init__ = spy
    try:
        def work():
            try:
                raise ValueError("from a worker")
            except ValueError:
                import sys
                bridge.report(*sys.exc_info())

        thread = threading.Thread(target=work)
        thread.start()
        thread.join()
        qtbot.waitUntil(lambda: bridge.dialog is not None, timeout=3000)
    finally:
        crash_dialog.CrashDialog.__init__ = original
    assert seen["thread"] is threading.main_thread()
    assert "ValueError: from a worker" in bridge.dialog.text()


def test_the_dialog_is_parented_to_the_window_when_one_is_set(qtbot, bridge, qt_app):
    bridge.window = qt_app
    bridge.crashed.emit(FAKE_TRACEBACK)
    assert bridge.dialog.parent() is qt_app


# -- Help > Open Log Folder ------------------------------------------------------------


def test_open_log_folder_entry_reveals_the_log_dir(qt_app, tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(logging_setup, "resolve_log_path", lambda cache: str(logs / "kokorogui.log"))
    seen = []
    monkeypatch.setattr(app_module, "reveal", lambda path: seen.append(path) or True)
    qt_app.open_log_folder_action.trigger()
    assert seen == [str(logs)]


def test_open_log_folder_says_so_when_there_is_no_folder(qt_app, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "reveal", lambda path: False)
    qt_app.open_log_folder_action.trigger()
    assert "No log folder" in qt_app.transport_dock.status_text()
