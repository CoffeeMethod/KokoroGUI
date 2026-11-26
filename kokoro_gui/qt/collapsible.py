"""A titled section that folds. The title is a button, so Tab reaches it and
Space folds it. Folding only hides the body: every widget in it keeps its
value, so a folded effect still runs.

Used for the Audio FX tab's groups, the Voice & Clip tab's Timing and Source
sections, and `SchemaFormWidget`'s `collapsible` groups.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLayout, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from kokoro_gui.i18n import tr



class CollapsibleSection(QWidget):
    collapsedChanged = Signal(bool)

    def __init__(self, title: str, collapsed: bool = False, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        self._title = title
        self._state = ""
        self.header = QToolButton()
        self.header.setProperty("section", True)
        self.header.setAutoRaise(True)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Ignored: a long state text is cut instead of widening the dock.
        self.header.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.header.clicked.connect(lambda _checked=False: self.set_collapsed(not self._collapsed))
        outer.addWidget(self.header)
        self.body = QWidget()
        outer.addWidget(self.body)
        self._collapsed = bool(collapsed)
        self._apply()

    def set_body_layout(self, layout: QLayout) -> None:
        layout.setContentsMargins(14, 0, 0, 4)
        self.body.setLayout(layout)

    @property
    def title(self) -> str:
        return self._title

    @property
    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        """Text after the title, shown folded or not ("Reverb on")."""
        if state != self._state:
            self._state = state
            self._apply()

    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool) -> None:
        collapsed = bool(collapsed)
        if collapsed == self._collapsed:
            return
        self._collapsed = collapsed
        self._apply()
        self.collapsedChanged.emit(collapsed)

    def _apply(self) -> None:
        self.body.setVisible(not self._collapsed)
        self.header.setArrowType(Qt.ArrowType.RightArrow if self._collapsed else Qt.ArrowType.DownArrow)
        text = f"{self._title}: {self._state}" if self._state else self._title
        self.header.setText(text.replace("&", "&&"))
        self.header.setAccessibleName(text)
        self.header.setToolTip(tr("{title}. Click to show the controls.", title=text) if self._collapsed
                               else tr("{title}. Click to hide the controls.", title=text))
