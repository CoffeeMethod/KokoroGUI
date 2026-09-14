"""Entry point for the PySide6 (Qt) frontend - the sole GUI frontend since the
Tk frontend (gui.py) was retired (see PLAN_qt_and_engine_abstraction.md,
workstream 3a). PySide6 is a regular dependency in `requirements.txt`.
"""
import sys

from PySide6.QtWidgets import QApplication

from kokoro_gui.qt.app import QtTTSApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QtTTSApp()
    window.show()
    window.show_welcome_if_enabled()
    sys.exit(app.exec())
