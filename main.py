"""Entry point for the PySide6 (Qt) frontend - the sole GUI frontend since the
Tk frontend (gui.py) was retired (see PLAN_qt_and_engine_abstraction.md,
workstream 3a). PySide6 is a regular dependency in `requirements.txt`.

This is the only place the log file, the stdout/stderr tee and the exception
hooks are installed (see kokoro_gui/logging_setup.py). Tests build
`QtTTSApp` directly, so they keep pytest's own exception handling.
"""
import logging
import os
import sys

from PySide6.QtWidgets import QApplication

from kokoro_gui import logging_setup
from kokoro_gui.engine import runtime
from kokoro_gui.qt.app import QtTTSApp
from kokoro_gui.qt.crash_dialog import CrashBridge, CrashDialog


def main() -> int:
    runtime.prepare_storage()  # may move CACHE_DIR, so the log dir comes after it
    log_path = logging_setup.setup_logging(os.path.join(runtime.CACHE_DIR, logging_setup.LOG_SUBDIR))
    app = QApplication(sys.argv)
    bridge = CrashBridge(log_path)
    logging_setup.install_excepthooks(bridge.report)
    try:
        window = QtTTSApp()
    except Exception:
        # No event loop yet, so a queued dialog would never show.
        logging.getLogger("kokorogui.crash").exception("The window failed to start")
        CrashDialog(logging_setup.format_exception_text(*sys.exc_info()), log_path).exec()
        return 1
    bridge.window = window
    window.show()
    window.show_welcome_if_enabled()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
