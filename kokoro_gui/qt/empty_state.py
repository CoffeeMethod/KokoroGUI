"""One line of guidance for an empty panel, with links that run the app's own
actions."""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel


class EmptyState(QLabel):
    """`show_message(html)` shows the line ("" hides it). An
    `<a href="name">` in it calls `links[name]()`."""

    def __init__(self, links: dict[str, Callable[[], object]], parent=None):
        super().__init__(parent)
        self._links = links
        self.setProperty("muted", True)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setWordWrap(True)
        self.setContentsMargins(4, 2, 4, 2)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse
                                     | Qt.TextInteractionFlag.LinksAccessibleByKeyboard)
        self.linkActivated.connect(self.run)
        self.hide()

    def show_message(self, html: str) -> None:
        if html != self.text():
            self.setText(html)
        self.setVisible(bool(html))

    def run(self, name: str) -> None:
        action = self._links.get(name)
        if action is not None:
            action()
