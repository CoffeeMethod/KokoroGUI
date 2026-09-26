"""A dock's minimum height is its content's, and a tab group takes the
largest minimum of its tabs, hidden ones included. A dock built from a
stack of fixed controls therefore sets how short its whole column can
get, and with it how tall the docks across the separator (the timeline)
can be dragged. `scrollable()` wraps such content so the dock can shrink
and scrolls instead."""
from __future__ import annotations

from PySide6.QtWidgets import QFrame, QScrollArea, QWidget


def scrollable(content: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidget(content)
    return scroll
