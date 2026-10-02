"""Outline dock (plan 20): the chapters of the timeline's level with a status
and a length, or its markers when it has no subprojects. Everything shown
is derived (`daw/outline.py`); the one thing it writes is the "proofed"
flag, a nested clip's `status == "approved"`, through `SetFieldCommand` on
the level's undo stack, the same command the timeline block menu uses."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QDockWidget, QHeaderView, QLabel, QMenu, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from kokoro_gui.daw import outline
from kokoro_gui.daw.undo import SetFieldCommand

COLUMNS = ("Title", "Status", "Length")
# `Clip.status` a cleared "proofed" goes back to: it was generated.
CLEARED_STATUS = "generated"
_ROW = Qt.ItemDataRole.UserRole


class OutlineDock(QDockWidget):
    def __init__(self, app, parent=None):
        super().__init__("Outline", parent)
        self.setObjectName("dock_outline")
        self.app = app
        self.outline = outline.Outline(outline.EMPTY, [], 0.0, False)
        self._stale = True  # rows not rebuilt since the last change
        self._built = False

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(list(COLUMNS))
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.itemDoubleClicked.connect(self._on_double_clicked)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self.tree, 1)

        self.footer = QLabel("")
        self.footer.setProperty("muted", True)
        layout.addWidget(self.footer)

        self.setWidget(content)
        self.visibilityChanged.connect(self._on_visibility_changed)

    # -- rebuilding ---------------------------------------------------------

    def refresh(self) -> None:
        """Called wherever the timeline refreshes. A dock behind another tab
        or closed only notes that it is out of date; it rebuilds when it is
        shown."""
        self._stale = True
        if self.isVisible():
            self.rebuild()

    def _on_visibility_changed(self, visible: bool) -> None:
        if visible and self._stale:
            self.rebuild()

    def _started(self, clip):
        """Whether a subproject has any generated clip, for "not started"
        against "in progress". None when the child isn't open: the dock
        never opens children to find out."""
        child = self.app.child_project(clip)
        if child is None:
            return None
        # A subproject inside it counts as started; its own state isn't asked.
        return any(c.is_nested or any(s.audio_path for s in c.segments) for c in child.document.clips)

    def current_outline(self) -> "outline.Outline":
        level = self.app.level
        with self.app.inputs_scope():
            arrangement = self.app.build_arrangement(level)
            return outline.build(level.document, arrangement,
                                 state_of=lambda clip: self.app.nested_state(clip, level),
                                 started_of=self._started)

    def rebuild(self) -> None:
        self._stale = False
        fresh = self.current_outline()
        if self._built and fresh == self.outline:
            return
        self._built = True
        self.outline = fresh
        current = self._row_key(self.tree.currentItem())
        self.tree.setUpdatesEnabled(False)
        try:
            self.tree.clear()
            for row in fresh.rows:
                item = QTreeWidgetItem([row.title, row.status, outline.format_hms(row.duration_s, row.estimated)])
                item.setData(0, _ROW, row)
                item.setTextAlignment(2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if row.status == outline.MISSING:
                    item.setToolTip(1, "The subproject's file can't be found.")
                self.tree.addTopLevelItem(item)
                if current is not None and self._row_key(item) == current:
                    self.tree.setCurrentItem(item)
        finally:
            self.tree.setUpdatesEnabled(True)
        self.footer.setText(self._footer_text(fresh))

    @staticmethod
    def _footer_text(result) -> str:
        total = f"Total {outline.format_hms(result.total_s, result.estimated)}"
        if result.kind == outline.CHAPTERS:
            count = len(result.rows)
            return f"{count} chapter{'s' if count != 1 else ''}. {total}"
        if result.kind == outline.MARKERS:
            return f"{len(result.rows)} marker{'s' if len(result.rows) != 1 else ''}. {total}"
        return f"No subprojects or markers. {total}"

    @staticmethod
    def _row_key(item):
        row = item.data(0, _ROW) if item is not None else None
        return (row.clip_id, row.marker_id) if row is not None else None

    # -- rows ---------------------------------------------------------------

    def item_rows(self) -> list:
        """The rows the tree shows, top to bottom."""
        return [self.tree.topLevelItem(i).data(0, _ROW) for i in range(self.tree.topLevelItemCount())]

    def row_item(self, clip_id: str):
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item.data(0, _ROW).clip_id == clip_id:
                return item
        return None

    def _nested_clip(self, row):
        clip = self.app.level.document.get_clip(row.clip_id) if row.clip_id else None
        return clip if clip is not None and clip.is_nested else None

    # -- actions ------------------------------------------------------------

    def go_to(self, row) -> None:
        """Double-click: the transcript caret to the row's line and the
        playhead to its start. A caret on a chapter's placeholder line
        points the docks at that subproject, as clicking it does."""
        level = self.app.level
        offset = None
        if row.clip_id:
            extent = level.document.clip_extent(row.clip_id)
            offset = extent[0] if extent else None
        else:
            hits = [p for p in self.app.build_arrangement(level).at_time(row.start_s) if not p.clip.is_bed]
            extent = level.document.clip_extent(hits[0].clip.id) if hits else None
            offset = extent[0] if extent else None
        editor = self.app.editor
        if offset is not None and editor is not None:
            # The transcript may be showing a subproject (a caret on a
            # placeholder line focuses it, grill NP1): bring the level's
            # text back before placing the caret in it.
            self.app.set_focus(level)
            cursor = editor.textCursor()
            cursor.setPosition(max(0, min(offset, len(editor.toPlainText()))))
            editor.setTextCursor(cursor)
            editor.ensureCursorVisible()
        self.app.transport.seek(row.start_s)

    def set_proofed(self, clip_id: str, proofed: bool) -> bool:
        """One undoable step on the level's stack. False when `clip_id`
        isn't a chapter of the level."""
        document = self.app.level.document
        clip = document.get_clip(clip_id)
        if clip is None or not clip.is_nested:
            return False
        status = "approved" if proofed else CLEARED_STATUS
        if clip.status == status:
            return True
        document.undo_stack.push(SetFieldCommand("clip", clip_id, "status", status))
        self.app.schedule_save()
        self.app.refresh_timeline()
        return True

    def _on_double_clicked(self, item, _column: int) -> None:
        row = item.data(0, _ROW)
        if row is not None:
            self.go_to(row)

    def _on_context_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        row = item.data(0, _ROW) if item is not None else None
        menu = self.build_menu(row)
        if menu is not None:
            menu.exec(self.tree.viewport().mapToGlobal(pos))

    def build_menu(self, row):
        """The right-click menu for `row`: None for a marker row, which has
        nothing to act on."""
        clip = self._nested_clip(row) if row is not None else None
        if clip is None:
            return None
        menu = QMenu(self)
        if row.clip_status == "approved":
            menu.addAction("Clear proofed", lambda: self.set_proofed(clip.id, False))
        else:
            menu.addAction("Mark proofed", lambda: self.set_proofed(clip.id, True))
        menu.addSeparator()
        menu.addAction("Enter subproject", lambda: self.app.enter_subproject(clip))
        generate = menu.addAction("Generate", lambda: self.app.generate_subproject(clip))
        generate.setEnabled(row.state != "missing")
        return menu
