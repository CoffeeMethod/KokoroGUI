"""Timeline dock: renders `app.document`'s clips/tracks/characters as a
multi-track timeline (Claude/PLAN_daw_ui_ux_redesign.md). Wires
kokoro_gui/qt/timeline_view.py's `TimelineView` into the docked shell -
unconditional, not capability-gated, since it renders Document state, which
is engine-independent.
"""
from __future__ import annotations

from PySide6.QtWidgets import QDockWidget

from kokoro_gui.qt.timeline_view import TimelineView


class TimelineDock(QDockWidget):
    def __init__(self, app, parent=None):
        super().__init__("Timeline", parent)
        self.setObjectName("dock_timeline")
        self.app = app

        self.timeline_view = TimelineView()
        self.setWidget(self.timeline_view)

        self.refresh()

    def refresh(self) -> None:
        self.timeline_view.render_document(self.app.document)
