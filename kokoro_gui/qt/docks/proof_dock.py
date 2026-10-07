"""Proof dock (plan 21): the review list for Proof by ASR.

Run proof transcribes generated clips (`ProofMixin.run_proof`) and scores
each against its text (`daw/proof.py`). The dock lists the timeline level's
clips that matched worse than the threshold, worst first. Everything shown is
derived from `session.json["proof"]` and the clips, so there is nothing to
save here except two things the dock writes on request: "Needs rewrite" sets
`Clip.status` through `SetFieldCommand` on the level's undo stack (the same
command the timeline's Status menu uses), and "Mark OK" sets a flag on the
stored result, which is not project data and not undoable.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QDockWidget, QDoubleSpinBox, QHBoxLayout, QHeaderView, QLabel, QPushButton, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget, QComboBox,
)

from kokoro_gui.daw import proof
from kokoro_gui.daw.undo import SetFieldCommand
from kokoro_gui.qt.proofing import SCOPE_ALL, SCOPE_SELECTION, SCOPE_SUBPROJECT

COLUMNS = ("Clip", "Match", "First issue")
NEEDS_REWRITE = "needs_rewrite"
_CLIP = Qt.ItemDataRole.UserRole
LABEL_CHARS = 36


class ProofDock(QDockWidget):
    def __init__(self, app, parent=None):
        super().__init__("Proof", parent)
        self.setObjectName("dock_proof")
        self.app = app
        self._rows: list = []
        self._stale = True  # rows not rebuilt since the last change
        self._built = False

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)

        top = QHBoxLayout()
        top.addWidget(QLabel("Run proof on"))
        self.scope_combo = QComboBox()
        self.scope_combo.addItem("All generated", SCOPE_ALL)
        self.scope_combo.addItem("Selection", SCOPE_SELECTION)
        self.scope_combo.addItem("This subproject", SCOPE_SUBPROJECT)
        self.scope_combo.setToolTip("All generated: every generated clip on the timeline.\n"
                                    "Selection: the selected clips.\n"
                                    "This subproject: the clips of the selected subproject block, or of the "
                                    "subproject you are in.")
        top.addWidget(self.scope_combo, 1)
        self.run_button = QPushButton("Run proof")
        self.run_button.clicked.connect(self._on_run_clicked)
        top.addWidget(self.run_button)
        layout.addLayout(top)

        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Flag below"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(proof.MIN_THRESHOLD, proof.MAX_THRESHOLD)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(app.proof_threshold())
        self.threshold_spin.setToolTip("A clip is flagged when its audio matches its text less closely than this "
                                       "(1.00 is word for word).")
        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)
        threshold_row.addWidget(self.threshold_spin)
        threshold_row.addStretch(1)
        self.footer = QLabel("")
        self.footer.setProperty("muted", True)
        threshold_row.addWidget(self.footer)
        layout.addLayout(threshold_row)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(list(COLUMNS))
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.tree.itemClicked.connect(self._on_item_chosen)
        self.tree.itemActivated.connect(self._on_item_chosen)
        self.tree.currentItemChanged.connect(lambda *_: self._sync_buttons())
        self.tree.setMinimumHeight(48)
        layout.addWidget(self.tree, 1)

        actions = QHBoxLayout()
        self.regenerate_button = QPushButton("Regenerate")
        self.regenerate_button.clicked.connect(self.regenerate_current)
        self.ok_button = QPushButton("Mark OK")
        self.ok_button.setToolTip("Take the clip off the list. It comes back if the clip is regenerated and "
                                  "proofed again.")
        self.ok_button.clicked.connect(self.mark_current_ok)
        self.rewrite_button = QPushButton("Needs rewrite")
        self.rewrite_button.clicked.connect(self.mark_current_needs_rewrite)
        for button in (self.regenerate_button, self.ok_button, self.rewrite_button):
            actions.addWidget(button)
        layout.addLayout(actions)

        self.setWidget(content)
        self.visibilityChanged.connect(self._on_visibility_changed)
        self._sync_buttons()

    # -- rebuilding ---------------------------------------------------------

    def refresh(self) -> None:
        """Called wherever the timeline refreshes. A dock behind another tab
        or closed only notes that it is out of date; it rebuilds when it is
        shown."""
        self._stale = True
        self._sync_run_button()
        if self.isVisible():
            self.rebuild()

    def _on_visibility_changed(self, visible: bool) -> None:
        if visible and self._stale:
            self.rebuild()

    def _label(self, clip) -> str:
        document = self.app.level.document
        character = document.get_character(clip.character_id)
        name = character.name if character is not None else ""
        words = " ".join(document.clip_text(clip).split())
        words = words if len(words) <= LABEL_CHARS else words[:LABEL_CHARS - 1] + "…"
        return f"{name}: {words}" if name else words

    def current_rows(self) -> tuple:
        """`(rows, proofed)`: the flagged clips worst first as
        `(clip_id, label, ratio, issue_text)`, and how many clips on the
        level have a current result."""
        app = self.app
        rows, proofed = [], 0
        for clip in app.level.document.clips:
            entry = app.proof_entry(clip)
            if entry is None:
                continue
            proofed += 1
            if not app.is_clip_flagged(clip):
                continue
            issues = entry.get("issues") or []
            first = proof.issue_text(issues[0]) if issues else ""
            rows.append((clip.id, self._label(clip), float(entry["ratio"]), first))
        rows.sort(key=lambda row: row[2])
        return rows, proofed

    def rebuild(self) -> None:
        self._stale = False
        rows, proofed = self.current_rows()
        if self._built and rows == self._rows:
            self._set_footer(len(rows), proofed)
            return
        self._built = True
        self._rows = rows
        current = self.current_clip_id()
        self.tree.setUpdatesEnabled(False)
        try:
            self.tree.clear()
            for clip_id, label, ratio, first in rows:
                item = QTreeWidgetItem([label, f"{round(ratio * 100)}%", first])
                item.setData(0, _CLIP, clip_id)
                item.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                item.setToolTip(2, first)
                self.tree.addTopLevelItem(item)
                if clip_id == current:
                    self.tree.setCurrentItem(item)
        finally:
            self.tree.setUpdatesEnabled(True)
        self._set_footer(len(rows), proofed)
        self._sync_buttons()

    def _set_footer(self, flagged: int, proofed: int) -> None:
        if proofed == 0:
            self.footer.setText("Nothing proofed on this timeline yet.")
        else:
            self.footer.setText(f"{flagged} flagged of {proofed} proofed.")

    # -- rows ---------------------------------------------------------------

    def row_clip_ids(self) -> list:
        """The clips the list shows, top to bottom."""
        return [self.tree.topLevelItem(i).data(0, _CLIP) for i in range(self.tree.topLevelItemCount())]

    def row_item(self, clip_id: str):
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item.data(0, _CLIP) == clip_id:
                return item
        return None

    def current_clip_id(self):
        item = self.tree.currentItem()
        return item.data(0, _CLIP) if item is not None else None

    def _sync_buttons(self) -> None:
        has_row = self.current_clip_id() is not None
        for button in (self.regenerate_button, self.ok_button, self.rewrite_button):
            button.setEnabled(has_row)

    def _sync_run_button(self) -> None:
        self.run_button.setText("Cancel" if self.app.is_proofing else "Run proof")

    # -- actions ------------------------------------------------------------

    def go_to(self, clip_id: str) -> None:
        """Selects the clip, puts the transcript caret on its first line and
        the playhead on its start."""
        app = self.app
        level = app.level
        clip = level.document.get_clip(clip_id)
        if clip is None:
            return
        app.set_focus(level)
        editor = app.editor
        extent = level.document.clip_extent(clip_id)
        if editor is not None and extent is not None:
            cursor = editor.textCursor()
            cursor.setPosition(max(0, min(extent[0], len(editor.toPlainText()))))
            editor.setTextCursor(cursor)
            editor.ensureCursorVisible()
        app.selection.select_clip(clip_id)
        placed = app.build_arrangement(level).by_clip_id().get(clip_id)
        if placed is not None:
            app.transport.seek(placed.start_s)

    def _on_item_chosen(self, item, _column: int = 0) -> None:
        clip_id = item.data(0, _CLIP)
        if clip_id is not None:
            self.go_to(clip_id)

    def run(self) -> bool:
        """The Run proof button: proofs the clips the scope combo names, or
        cancels a running proof."""
        app = self.app
        if app.is_proofing:
            app.cancel_proof()
            return True
        clip_ids, message = app.clips_for_proof_scope(self.scope_combo.currentData())
        if not clip_ids:
            app.set_status(message, "warning")
            return False
        started = app.run_proof(clip_ids)
        self._sync_run_button()
        return started

    def _on_run_clicked(self) -> None:
        self.run()

    def _on_threshold_changed(self, value: float) -> None:
        self.app.set_proof_threshold(value)

    def regenerate_current(self) -> None:
        clip_id = self.current_clip_id()
        if clip_id is not None:
            self.app.set_focus(self.app.level)
            self.app.generate_clip(clip_id)

    def mark_current_ok(self) -> bool:
        clip_id = self.current_clip_id()
        return clip_id is not None and self.app.mark_proof_ok(clip_id)

    def mark_current_needs_rewrite(self) -> bool:
        """One undoable step on the level's stack: the clip's status becomes
        "Needs rewrite", which also takes it off this list."""
        clip_id = self.current_clip_id()
        document = self.app.level.document
        if clip_id is None or document.get_clip(clip_id) is None:
            return False
        document.undo_stack.push(SetFieldCommand("clip", clip_id, "status", NEEDS_REWRITE))
        self.app.schedule_save()
        self.app.refresh_timeline()
        return True
