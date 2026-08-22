"""Entry point for the PySide6 (Qt) frontend - ships alongside main.py's Tk
frontend during the workstream 3a transition (see
PLAN_qt_and_engine_abstraction.md). Requires the optional `requirements-qt.txt`
extras (`pip install -r requirements-qt.txt`).
"""
import sys

from PySide6.QtWidgets import QApplication

from kokoro_gui.qt.app import QtTTSApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QtTTSApp()
    window.show()
    sys.exit(app.exec())
