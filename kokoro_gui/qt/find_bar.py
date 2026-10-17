"""The find bar under the transcript (plan 34): Edit > Find (Ctrl+F) and Edit >
Find and Replace (Ctrl+H).

Search covers the whole project (grill PG10): the transcript on screen, and
the transcript of every other open project (the root and the subprojects that
have been opened). The bar highlights and counts the matches of the transcript
on screen, steps through them (F3, Shift+F3, Enter, Shift+Enter), and lists
the other projects' counts below the counter; a double-click on a row shows
that project. A subproject that has not been opened is not searched, and the
bar says how many were left out.

Replace swaps the match the editor has selected. Replace all swaps every match
in every searched project: the project on screen through
`TranscriptEditor.replace_ranges`, which joins the native undo step of the
editor, and each other project as one `CompositeCommand` on its own undo stack
(`textsearch.replace_command`), so each one undoes in its own project. While
a generate runs nothing is searched or replaced.

The pure half (the search, the plan and the command) is in
`kokoro_gui/daw/textsearch.py`.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Optional

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

from kokoro_gui.daw import textsearch

SEARCH_DEBOUNCE_MS = 150
# The longest selection a Ctrl+F carries into the find field.
MAX_PREFILL = 200
BUSY_TEXT = "Paused while a generate runs"


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def summary_text(plans: list, closed: int = 0) -> str:
    """The sentence under Replace all: what the replacement does, from one
    `ReplacePlan` per searched project that has matches."""
    matches = sum(p.matches for p in plans)
    if not matches:
        return ""
    clips = sum(p.clips for p in plans)
    stale = sum(p.stale for p in plans)
    recordings = sum(p.recordings for p in plans)
    text = f"Replaces {_count(matches, 'match', 'matches')} in {_count(clips, 'clip', 'clips')}"
    if len(plans) > 1:
        text += f" across {len(plans)} projects"
    text += "."
    if stale:
        text += f" {_count(stale, 'clip', 'clips')} will turn stale."
    if recordings:
        text += f" {_count(recordings, 'recording', 'recordings')} will split."
    return text


class FindBar(QWidget):
    def __init__(self, app, editor, parent=None):
        super().__init__(parent)
        self.setObjectName("find_bar")
        self.app = app
        self.editor = editor
        self._hits: list = []
        # `[(project, hits)]` of the other open projects that have matches.
        self._others: list = []
        self._closed_children = 0
        self._busy = False
        self._select_pending = False
        self._building = True
        # True from open() to close_bar(): the bar searches and highlights
        # only then (isVisible() is false while the window is hidden).
        self._active = False

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(SEARCH_DEBOUNCE_MS)
        self._timer.timeout.connect(self._on_timer)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        top = QHBoxLayout()
        self.find_edit = QLineEdit()
        self.find_edit.setPlaceholderText("Find")
        self.find_edit.setClearButtonEnabled(True)
        self.find_edit.setMinimumWidth(80)
        top.addWidget(self.find_edit, 1)
        self.match_case_check = QCheckBox("Match case")
        self.whole_word_check = QCheckBox("Whole word")
        top.addWidget(self.match_case_check)
        top.addWidget(self.whole_word_check)
        self.counter_label = QLabel()
        self.counter_label.setProperty("muted", True)
        self.counter_label.setMinimumWidth(70)
        top.addWidget(self.counter_label)
        self.previous_btn = QPushButton("Previous")
        self.previous_btn.setToolTip("The match before this one (Shift+F3, Shift+Enter).")
        self.next_btn = QPushButton("Next")
        self.next_btn.setToolTip("The next match (F3, Enter).")
        self.close_btn = QPushButton("Close")
        self.close_btn.setToolTip("Close the find bar (Esc).")
        for button in (self.previous_btn, self.next_btn, self.close_btn):
            top.addWidget(button)
        layout.addLayout(top)

        self.replace_row = QWidget()
        replace_layout = QHBoxLayout(self.replace_row)
        replace_layout.setContentsMargins(0, 0, 0, 0)
        self.replace_edit = QLineEdit()
        self.replace_edit.setPlaceholderText("Replace with")
        self.replace_edit.setMinimumWidth(80)
        replace_layout.addWidget(self.replace_edit, 1)
        self.replace_btn = QPushButton("Replace")
        self.replace_btn.setToolTip("Replace the selected match and go to the next one.")
        self.replace_all_btn = QPushButton("Replace all")
        self.replace_all_btn.setToolTip("Replace every match in every searched project. Each project undoes on its own.")
        replace_layout.addWidget(self.replace_btn)
        replace_layout.addWidget(self.replace_all_btn)
        layout.addWidget(self.replace_row)

        self.summary_label = QLabel()
        self.summary_label.setProperty("muted", True)
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.projects_list = QListWidget()
        self.projects_list.setObjectName("find_projects")
        self.projects_list.setMaximumHeight(84)
        self.projects_list.setToolTip("Matches in the other projects. Double-click one to show it.")
        layout.addWidget(self.projects_list)
        self.closed_label = QLabel()
        self.closed_label.setProperty("muted", True)
        self.closed_label.setWordWrap(True)
        layout.addWidget(self.closed_label)

        self.find_edit.textChanged.connect(lambda _text: self._schedule(True))
        self.match_case_check.toggled.connect(lambda _on: self._schedule(True))
        self.whole_word_check.toggled.connect(lambda _on: self._schedule(True))
        self.replace_edit.textChanged.connect(lambda _text: self._update_summary())
        self.next_btn.clicked.connect(lambda: self.find_next())
        self.previous_btn.clicked.connect(lambda: self.find_previous())
        self.close_btn.clicked.connect(self.close_bar)
        self.replace_btn.clicked.connect(self.replace_current)
        self.replace_all_btn.clicked.connect(self.replace_all)
        self.projects_list.itemDoubleClicked.connect(self._show_project)
        self.find_edit.installEventFilter(self)
        self.replace_edit.installEventFilter(self)
        editor.document().contentsChanged.connect(lambda: self._schedule(False))

        self._building = False
        self.set_replace_visible(False)
        self.hide()
        self._render()

    # -- showing and hiding ------------------------------------------------------

    def open(self, replace: bool = False) -> None:
        """Shows the bar (with the replace row for Find and Replace), carries
        the editor's one-line selection into the find field, and puts the
        caret there."""
        self.set_replace_visible(replace or self.replace_row.isVisibleTo(self))
        cursor = self.editor.textCursor()
        selected = cursor.selectedText()
        self._active = True
        self.show()
        if cursor.hasSelection() and 0 < len(selected) <= MAX_PREFILL and "\u2029" not in selected \
                and "\n" not in selected:
            self.find_edit.setText(selected)
        self.find_edit.setFocus()
        self.find_edit.selectAll()
        self.refresh(select=False)

    def is_open(self) -> bool:
        return self._active

    def close_bar(self) -> None:
        self._timer.stop()
        self._active = False
        self.hide()
        self.editor.clear_find_matches()
        self.editor.setFocus()

    def set_replace_visible(self, visible: bool) -> None:
        self.replace_row.setVisible(visible)
        self._update_summary()

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.key() == Qt.Key.Key_Escape:
            self.close_bar()
            event.accept()
            return
        super().keyPressEvent(event)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt override)
        if event.type() == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if watched is self.find_edit:
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self.find_previous()
                else:
                    self.find_next()
                return True
            if watched is self.replace_edit:
                self.replace_current()
                return True
        return super().eventFilter(watched, event)

    # -- searching ----------------------------------------------------------------

    def options(self) -> tuple:
        return self.find_edit.text(), self.match_case_check.isChecked(), self.whole_word_check.isChecked()

    def hits(self) -> list:
        """The matches in the transcript on screen, as `(start, end)`."""
        return list(self._hits)

    def other_hits(self) -> list:
        """`[(project, count)]` for the other open projects with matches."""
        return [(project, len(hits)) for project, hits in self._others]

    def _schedule(self, select: bool) -> None:
        if self._building or not self._active:
            return
        self._select_pending = self._select_pending or select
        self._timer.start()

    def _on_timer(self) -> None:
        select, self._select_pending = self._select_pending, False
        self.refresh(select=select)

    def on_busy_changed(self) -> None:
        """A generate started or ended: the search stops or starts again."""
        if self._active:
            self.refresh(select=False)

    def refresh(self, select: bool = False) -> None:
        """Searches now. With `select`, the first match from the caret on
        becomes the editor's selection when none is."""
        self._timer.stop()
        self._select_pending = False
        needle, match_case, whole_word = self.options()
        self._busy = bool(self.app.is_busy())
        self._hits = []
        self._others = []
        self._closed_children = 0
        if needle and not self._busy and self._active:
            document = self.app.document
            self._hits = textsearch.find_in_document(document, needle, match_case, whole_word)
            projects = self._ordered_projects()
            for project in projects:
                if project.document is document:
                    continue
                found = textsearch.find_in_document(project.document, needle, match_case, whole_word)
                if found:
                    self._others.append((project, found))
            self._closed_children = sum(
                1 for project in projects for clip in project.document.nested_clips()
                if self.app.child_project(clip) is None)
        if select and self._hits and self._current_index() is None:
            self._select(self._first_from(self.editor.textCursor().selectionStart()))
            return
        self._render()

    def _ordered_projects(self) -> list:
        """The open projects in transcript order: the root, then each
        subproject where its line sits, depth first."""
        opened = self.app.open_projects()
        ordered: list = []

        def walk(project) -> None:
            ordered.append(project)
            document = project.document
            nested = sorted(document.nested_clips(), key=lambda clip: (document.clip_extent(clip.id) or (0, 0))[0])
            for clip in nested:
                child = self.app.child_project(clip)
                if child is not None and not any(child is seen for seen in ordered):
                    walk(child)

        walk(self.app.root)
        ordered = [project for project in ordered if any(project is item for item in opened)]
        return ordered + [project for project in opened if not any(project is item for item in ordered)]

    def _current_index(self) -> Optional[int]:
        cursor = self.editor.textCursor()
        span = (cursor.selectionStart(), cursor.selectionEnd())
        i = bisect_left(self._hits, span)
        return i if i < len(self._hits) and self._hits[i] == span else None

    def _first_from(self, position: int) -> int:
        """Index of the first match starting at or after `position`, or the
        first one."""
        i = bisect_left(self._hits, (position, -1))
        return i if i < len(self._hits) else 0

    def _render(self) -> None:
        """Highlights, the counter, the other projects' rows and the summary."""
        current = self._current_index() if self._hits else None
        if self._active:
            self.editor.set_find_matches(self._hits, current)
        needle = self.find_edit.text()
        if self._busy:
            self.counter_label.setText(BUSY_TEXT)
        elif not needle:
            self.counter_label.setText("")
        elif not self._hits:
            self.counter_label.setText("No matches")
        elif current is None:
            self.counter_label.setText(_count(len(self._hits), "match", "matches"))
        else:
            self.counter_label.setText(f"{current + 1} of {len(self._hits)}")
        has_hits = bool(self._hits)
        self.next_btn.setEnabled(has_hits)
        self.previous_btn.setEnabled(has_hits)
        self.projects_list.clear()
        for project, hits in self._others:
            item = QListWidgetItem(f"{project.title()}: {_count(len(hits), 'match', 'matches')}")
            item.setData(Qt.ItemDataRole.UserRole, project.project_id)
            self.projects_list.addItem(item)
        self.projects_list.setVisible(bool(self._others))
        self.closed_label.setText(
            f"{_count(self._closed_children, 'subproject', 'subprojects')} not opened, so not searched."
            if self._closed_children and needle and not self._busy else "")
        self.closed_label.setVisible(bool(self.closed_label.text()))
        self._update_summary()

    def _plans(self) -> list:
        """`[(project, plan)]` of replacing every match with the replace
        field's text, the project on screen first, for the projects where it
        changes something."""
        replacement = self.replace_edit.text()
        plans = []
        if self._hits:
            plan = textsearch.plan_replace(self.app.document, self._hits, replacement)
            if plan.matches:
                plans.append((self.app.focus, plan))
        for project, hits in self._others:
            plan = textsearch.plan_replace(project.document, hits, replacement)
            if plan.matches:
                plans.append((project, plan))
        return plans

    def _update_summary(self) -> None:
        shown = self.replace_row.isVisibleTo(self) and not self._busy
        plans = self._plans() if shown and (self._hits or self._others) else []
        self.summary_label.setText(summary_text([plan for _project, plan in plans]) if shown else "")
        self.summary_label.setVisible(bool(self.summary_label.text()))
        self.replace_btn.setEnabled(not self._busy and bool(self._hits))
        self.replace_all_btn.setEnabled(not self._busy and bool(plans))

    # -- stepping -----------------------------------------------------------------

    def find_next(self) -> bool:
        return self._step(forward=True)

    def find_previous(self) -> bool:
        return self._step(forward=False)

    def _step(self, forward: bool) -> bool:
        """Selects the next (or previous) match from the caret, wrapping
        round the transcript. False when there is none."""
        if self._timer.isActive() or not self._active:
            self.refresh(select=False)
        if not self._hits:
            return False
        cursor = self.editor.textCursor()
        if forward:
            index = self._first_from(cursor.selectionEnd() if cursor.hasSelection() else cursor.position())
        else:
            ends = [end for _start, end in self._hits]
            index = bisect_right(ends, cursor.selectionStart()) - 1
            if index < 0:
                index = len(self._hits) - 1
        self._select(index)
        return True

    def _select(self, index: int) -> None:
        self.editor.select_span(*self._hits[index])
        self._render()

    # -- replacing ----------------------------------------------------------------

    def replace_current(self) -> bool:
        """Replaces the selected match and moves to the next one. When no
        match is selected it only selects one."""
        if self._busy or self.app.is_busy():
            return False
        if self._timer.isActive():
            self.refresh(select=False)
        index = self._current_index()
        if index is None:
            return self.find_next()
        start, end = self._hits[index]
        replacement = self.replace_edit.text()
        self.editor.replace_ranges([(start, end, replacement)])
        cursor = self.editor.textCursor()
        cursor.setPosition(start + len(replacement))
        self.editor.setTextCursor(cursor)
        self.refresh(select=False)
        self.find_next()
        return True

    def replace_all(self) -> int:
        """Replaces every match in every searched project and returns how
        many it replaced."""
        if self.app.is_busy():
            self._busy = True
            self._render()
            return 0
        self.refresh(select=False)
        plans = self._plans()
        if not plans:
            return 0
        others = [(project, plan) for project, plan in plans if project.document is not self.app.document]
        if others:
            names = ", ".join(f"{project.title()} ({plan.matches})" for project, plan in others)
            answer = QMessageBox.question(
                self.app, "Replace all",
                summary_text([plan for _project, plan in plans]) + f"\n\nIn the other projects: {names}.\n"
                "Each project undoes its own replacement.")
            if answer != QMessageBox.StandardButton.Yes:
                return 0
        replaced = 0
        # A long replacement runs on the GUI thread, about 5 ms an edit in a
        # transcript of a thousand clips.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for project, plan in plans:
                if project.document is self.app.document:
                    replaced += self.editor.replace_ranges(plan.edits)
                else:
                    project.document.undo_stack.push(textsearch.replace_command(project.document, plan.edits))
                    replaced += plan.matches
        finally:
            QApplication.restoreOverrideCursor()
        clips = sum(plan.clips for _project, plan in plans)
        stale = sum(plan.stale for _project, plan in plans)
        message = f"Replaced {_count(replaced, 'match', 'matches')} in {_count(clips, 'clip', 'clips')}"
        if len(plans) > 1:
            message += f" across {len(plans)} projects"
        message += f"; {_count(stale, 'clip', 'clips')} turned stale." if stale else "."
        self.app.set_status(message, "success")
        self.app.schedule_save()
        self.app.refresh_timeline()
        self.refresh(select=False)
        return replaced

    def _show_project(self, item) -> None:
        project_id = item.data(Qt.ItemDataRole.UserRole)
        for project, _hits in self._others:
            if project.project_id == project_id:
                self.app.set_focus(project)
                self.refresh(select=True)
                return
