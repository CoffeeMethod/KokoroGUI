"""Queue dock (plan 28): the generation queue, item by item.

Generate plans stale clips into items of eight (`daw/genqueue.py`); this dock
lists them with their state and time left. A queued item can be dragged to a
new place, moved to the top or removed; Pause / Resume and Cancel all act on
the whole queue, the same way the Transport dock's buttons do. A queue the
last session left behind (`session.json["queue"]`) comes back paused, and the
button then reads "Resume generating N clips".

Everything shown is the app's queue; the dock keeps no state of its own. It
rebuilds while visible and notes that it is out of date otherwise.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QDockWidget, QHBoxLayout, QHeaderView, QLabel, QMenu, QPushButton, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)

from kokoro_gui.daw import genqueue

COLUMNS = ("Item", "Clips", "State", "Time left")
STATE_LABELS = {
    genqueue.QUEUED: "Queued", genqueue.RUNNING: "Generating", genqueue.DONE: "Done",
    genqueue.FAILED: "Failed", genqueue.CANCELLED: "Cancelled",
}
_ITEM_ID = Qt.ItemDataRole.UserRole


class _ItemTree(QTreeWidget):
    """Lets the user drag a row; the move itself is the queue's, so the drop
    is reported (`dropped(item id, position)`) and Qt's own reorder is
    refused. `position` is the row the item lands before, in the list as it
    was when the drag began (the row count for "after the last")."""

    dropped = Signal(str, int)

    def dropEvent(self, event) -> None:  # noqa: N802 (Qt override)
        dragged = self.currentItem()
        target = self.itemAt(event.position().toPoint())
        event.ignore()
        if dragged is None:
            return
        if target is None:
            position = self.topLevelItemCount()
        else:
            position = self.indexOfTopLevelItem(target)
            if self.dropIndicatorPosition() == QAbstractItemView.DropIndicatorPosition.BelowItem:
                position += 1
        self.dropped.emit(dragged.data(0, _ITEM_ID), position)


class QueueDock(QDockWidget):
    def __init__(self, app, parent=None):
        super().__init__("Queue", parent)
        self.setObjectName("dock_queue")
        self.app = app
        self._stale = True  # rows not rebuilt since the last change

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)

        buttons = QHBoxLayout()
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.app.toggle_queue_pause)
        self.cancel_button = QPushButton("Cancel all")
        self.cancel_button.setToolTip("Stop the batch that is running and drop everything still queued.")
        self.cancel_button.clicked.connect(self.app.cancel_queue_clicked)
        buttons.addWidget(self.pause_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.summary = QLabel("")
        self.summary.setProperty("muted", True)
        layout.addWidget(self.summary)

        self.tree = _ItemTree()
        self.tree.setHeaderLabels(list(COLUMNS))
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree.setDragEnabled(True)
        self.tree.setAcceptDrops(True)
        self.tree.setDropIndicatorShown(True)
        self.tree.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.dropped.connect(self._on_dropped)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        self.tree.setMinimumHeight(48)
        layout.addWidget(self.tree, 1)

        self.setWidget(content)
        self.visibilityChanged.connect(self._on_visibility_changed)
        self._sync_buttons()

    # -- rebuilding ---------------------------------------------------------

    def refresh(self) -> None:
        """Called on every queue change. A dock behind another tab or closed
        only notes that it is out of date; it rebuilds when shown."""
        self._stale = True
        self._sync_buttons()
        if self.isVisible():
            self.rebuild()

    def _on_visibility_changed(self, visible: bool) -> None:
        if visible and self._stale:
            self.rebuild()

    def _eta_text(self, item) -> str:
        if item.state not in (genqueue.QUEUED, genqueue.RUNNING):
            return ""
        app = self.app
        fraction = app._queue_fraction if item.state == genqueue.RUNNING else 0.0
        seconds, complete = genqueue.GenerationQueue.item_eta_s(item, app._rate_for, fraction)
        if seconds <= 0:
            return ""
        return genqueue.format_eta(seconds) if complete else f"{genqueue.format_eta(seconds)}+"

    def item_row_text(self, item) -> tuple:
        clips = "subproject" if item.kind == genqueue.SUBPROJECT else str(item.clip_count)
        return item.title, clips, STATE_LABELS.get(item.state, item.state), self._eta_text(item)

    def rebuild(self) -> None:
        self._stale = False
        queue = self.app.generation_queue
        selected = self.selected_item_id()
        self.tree.setUpdatesEnabled(False)
        try:
            self.tree.clear()
            for item in queue.items:
                row = QTreeWidgetItem(list(self.item_row_text(item)))
                row.setData(0, _ITEM_ID, item.id)
                row.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                flags = row.flags()
                # Only a queued item is dragged; any row can be a drop target.
                row.setFlags(flags | Qt.ItemFlag.ItemIsDragEnabled if item.state == genqueue.QUEUED
                             else flags & ~Qt.ItemFlag.ItemIsDragEnabled)
                self.tree.addTopLevelItem(row)
                if item.id == selected:
                    self.tree.setCurrentItem(row)
        finally:
            self.tree.setUpdatesEnabled(True)
        self._sync_summary()

    def _sync_summary(self) -> None:
        app = self.app
        queue = app.generation_queue
        if queue.pending():
            if app.queue_active:
                self.summary.setText(app.queue_progress_text()[1])
            else:
                count = app.queue_pending_clip_count()
                eta = app.queue_eta_text()
                self.summary.setText(f"{count} clip(s) waiting" + (f", {eta}" if eta else "")
                                     + (" (paused)" if app.queue_paused else ""))
        else:
            self.summary.setText("Nothing queued. Generate adds the stale clips here." if not queue.items
                                 else "Nothing left to do.")

    def _sync_buttons(self) -> None:
        app = self.app
        pending = bool(app.generation_queue.pending())
        self.pause_button.setEnabled(pending)
        self.cancel_button.setEnabled(pending)
        if app.queue_paused:
            idle = not app.queue_active
            self.pause_button.setText(f"Resume generating {app.queue_pending_clip_count()} clips" if idle
                                      else "Resume")
        else:
            self.pause_button.setText("Pause")

    # -- items -------------------------------------------------------------------

    def selected_item_id(self):
        row = self.tree.currentItem()
        return row.data(0, _ITEM_ID) if row is not None else None

    def _item(self, item_id):
        return self.app.generation_queue.find(item_id) if item_id else None

    def _on_dropped(self, item_id: str, position: int) -> None:
        """A queued item was dropped before row `position`."""
        queue = self.app.generation_queue
        item = self._item(item_id)
        if item is None:
            return
        index = queue.index_of(item)
        self.app.move_queue_item(item, position - 1 if position > index else position)

    def _on_context_menu(self, pos) -> None:
        row = self.tree.itemAt(pos)
        menu = self.build_menu(self._item(row.data(0, _ITEM_ID)) if row is not None else None)
        if menu is not None:
            menu.exec(self.tree.viewport().mapToGlobal(pos))

    def build_menu(self, item):
        """The right-click menu for a queued `item`: None for a finished or
        running one, which has nothing to move or remove."""
        if item is None or item.state != genqueue.QUEUED:
            return None
        menu = QMenu(self)
        menu.addAction("Move to top", lambda: self.app.queue_item_to_top(item))
        menu.addAction("Remove", lambda: self.app.remove_queue_item(item))
        return menu
