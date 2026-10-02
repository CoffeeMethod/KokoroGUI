"""The Outline dock (plan 20): chapters with status and length, proofing as
one undoable step, double-click to the line and the playhead, the marker
fallback and the dock's place in the shell."""
import pytest
from PySide6.QtCore import Qt

from kokoro_gui.daw import outline
from tests.gui_qt.test_subprojects import _generated_clip


def _two_chapters(qt_app):
    """"Intro. Alpha text. Beta text. Outro." with both middle clips moved
    into embedded subprojects (still unrendered, so "stale")."""
    document = qt_app.document
    document.text = "Intro. Alpha text. Beta text. Outro."
    qt_app.editor.load_text(document.text)
    _generated_clip(qt_app, 0, 6)
    _generated_clip(qt_app, 7, 18)
    _generated_clip(qt_app, 19, 29)
    alpha = qt_app.new_subproject(7, 18, title="Alpha")
    start = document.text.index("Beta")
    beta = qt_app.new_subproject(start, start + len("Beta text."), title="Beta")
    return alpha, beta


def _shown(qt_app, qtbot):
    qt_app.show()
    qtbot.waitExposed(qt_app)
    qt_app.outline_dock.show()
    qt_app.outline_dock.raise_()
    qt_app.flush_updates()
    qt_app.outline_dock.rebuild()
    return qt_app.outline_dock


def test_the_dock_is_registered_and_named(qt_app):
    dock = qt_app.outline_dock
    assert dock.objectName() == "dock_outline" and dock.windowTitle() == "Outline"
    assert dock in qt_app._all_docks()


def test_two_subprojects_make_two_rows(qt_app, qtbot):
    alpha, beta = _two_chapters(qt_app)
    dock = _shown(qt_app, qtbot)
    rows = dock.item_rows()
    assert [r.title for r in rows] == ["Alpha", "Beta"]
    assert [r.clip_id for r in rows] == [alpha.clip_id, beta.clip_id]
    # Moved clips kept their audio but nothing is rendered yet.
    assert [r.state for r in rows] == ["stale", "stale"]
    assert [r.status for r in rows] == [outline.IN_PROGRESS, outline.IN_PROGRESS]
    assert [dock.tree.topLevelItem(i).text(0) for i in range(2)] == ["Alpha", "Beta"]
    assert dock.tree.topLevelItem(0).text(2).startswith("~")
    assert dock.footer.text().startswith("2 chapters. Total ~")


def test_rendering_a_chapter_turns_it_done_and_gives_its_length(qt_app, qtbot):
    alpha, _beta = _two_chapters(qt_app)
    dock = _shown(qt_app, qtbot)
    assert qt_app.render_subproject(alpha)
    qt_app.wait_for_project_io()
    dock.rebuild()
    first, second = dock.item_rows()
    assert first.status == outline.DONE and first.estimated is False and first.duration_s > 0
    assert second.status == outline.IN_PROGRESS
    assert not dock.tree.topLevelItem(0).text(2).startswith("~")


def test_a_chapter_with_no_generated_clip_is_not_started(qt_app, qtbot):
    alpha, _beta = _two_chapters(qt_app)
    for clip in alpha.document.clips:
        clip.segments = []
    dock = _shown(qt_app, qtbot)
    qt_app.refresh_timeline()
    assert [r.status for r in dock.item_rows()] == [outline.NOT_STARTED, outline.IN_PROGRESS]


def test_mark_proofed_sets_approved_and_undo_clears_it(qt_app, qtbot):
    alpha, _beta = _two_chapters(qt_app)
    assert qt_app.render_subproject(alpha)
    qt_app.wait_for_project_io()
    dock = _shown(qt_app, qtbot)
    assert dock.item_rows()[0].status == outline.DONE
    stack = qt_app.document.undo_stack

    assert dock.set_proofed(alpha.clip_id, True)
    assert qt_app.document.get_clip(alpha.clip_id).status == "approved"
    assert dock.item_rows()[0].status == outline.PROOFED

    stack.undo()
    qt_app.refresh_timeline()
    assert qt_app.document.get_clip(alpha.clip_id).status != "approved"
    assert dock.item_rows()[0].status == outline.DONE

    stack.redo()
    qt_app.refresh_timeline()
    assert dock.item_rows()[0].status == outline.PROOFED

    assert dock.set_proofed(alpha.clip_id, False)
    assert qt_app.document.get_clip(alpha.clip_id).status == "generated"
    assert dock.item_rows()[0].status == outline.DONE


def test_the_menu_offers_mark_or_clear_by_the_clips_status(qt_app, qtbot):
    alpha, _beta = _two_chapters(qt_app)
    dock = _shown(qt_app, qtbot)

    def texts(row):
        return [a.text() for a in dock.build_menu(row).actions() if not a.isSeparator()]

    assert texts(dock.item_rows()[0]) == ["Mark proofed", "Enter subproject", "Generate"]
    dock.set_proofed(alpha.clip_id, True)
    assert texts(dock.item_rows()[0]) == ["Clear proofed", "Enter subproject", "Generate"]


def test_the_menu_actions_call_the_app(qt_app, qtbot, monkeypatch):
    alpha, _beta = _two_chapters(qt_app)
    dock = _shown(qt_app, qtbot)
    entered, generated = [], []
    monkeypatch.setattr(qt_app, "enter_subproject", lambda clip: entered.append(clip.id))
    monkeypatch.setattr(qt_app, "generate_subproject", lambda clip, then=None: generated.append(clip.id))
    menu = dock.build_menu(dock.item_rows()[0])
    actions = {a.text(): a for a in menu.actions()}
    actions["Enter subproject"].trigger()
    actions["Generate"].trigger()
    assert entered == generated == [alpha.clip_id]


def test_double_click_moves_the_caret_and_seeks(qt_app, qtbot, monkeypatch):
    alpha, beta = _two_chapters(qt_app)
    dock = _shown(qt_app, qtbot)
    seeks = []
    monkeypatch.setattr(qt_app.transport, "seek", lambda seconds: seeks.append(seconds))

    second = dock.tree.topLevelItem(1)
    dock.tree.itemDoubleClicked.emit(second, 0)

    # The caret landed on Beta's placeholder line: that selected it.
    assert qt_app.selection.selected_clip_id == beta.clip_id
    assert seeks == [second.data(0, Qt.ItemDataRole.UserRole).start_s] and seeks[0] > 0

    # The transcript now shows Beta's text; a double-click on the first row
    # brings the level back and lands on Alpha's line.
    first = dock.tree.topLevelItem(0)
    dock.tree.itemDoubleClicked.emit(first, 0)
    assert qt_app.selection.selected_clip_id == alpha.clip_id
    assert len(seeks) == 2 and seeks[1] == first.data(0, Qt.ItemDataRole.UserRole).start_s


def test_double_click_on_a_marker_lands_on_the_clip_under_it(qt_app, qtbot, monkeypatch):
    from kokoro_gui.daw.undo import SetFieldCommand

    qt_app.document.text = "One line. Another line."
    qt_app.editor.load_text(qt_app.document.text)
    first = _generated_clip(qt_app, 0, 9)
    second = _generated_clip(qt_app, 10, 23)
    start = qt_app.build_arrangement().by_clip_id()[second.id].start_s
    qt_app.document.undo_stack.push(SetFieldCommand("document", None, "settings", [
        {"id": "m", "seconds": start + 0.01, "name": "Second", "note": ""}], key="markers"))
    dock = _shown(qt_app, qtbot)
    seeks = []
    monkeypatch.setattr(qt_app.transport, "seek", lambda seconds: seeks.append(seconds))

    dock.tree.itemDoubleClicked.emit(dock.tree.topLevelItem(0), 0)

    assert qt_app.editor.textCursor().position() == qt_app.document.clip_extent(second.id)[0]
    assert qt_app.selection.selected_clip_id == second.id and first.id != second.id
    assert seeks == [start + 0.01]


def test_closed_dock_rebuilds_when_shown(qt_app, qtbot):
    _two_chapters(qt_app)
    dock = qt_app.outline_dock
    dock.hide()
    qt_app.refresh_timeline()
    assert dock.tree.topLevelItemCount() == 0  # nothing built while hidden
    qt_app.show()
    qtbot.waitExposed(qt_app)
    dock.show()
    dock.raise_()
    assert dock.tree.topLevelItemCount() == 2


def test_markers_stand_in_for_a_project_without_subprojects(qt_app, qtbot):
    from kokoro_gui.daw.undo import SetFieldCommand

    qt_app.document.text = "One line. Another line."
    qt_app.editor.load_text(qt_app.document.text)
    _generated_clip(qt_app, 0, 9)
    _generated_clip(qt_app, 10, 23)
    qt_app.document.undo_stack.push(SetFieldCommand("document", None, "settings", [
        {"id": "a", "seconds": 0.0, "name": "Opening", "note": ""},
        {"id": "b", "seconds": 0.1, "name": "Second part", "note": ""},
    ], key="markers"))
    dock = _shown(qt_app, qtbot)
    rows = dock.item_rows()
    assert [r.title for r in rows] == ["Opening", "Second part"]
    assert all(r.status == "" for r in rows)
    assert dock.footer.text().startswith("2 markers. Total ")
    assert dock.build_menu(rows[0]) is None


def test_an_empty_project_says_so(qt_app, qtbot):
    dock = _shown(qt_app, qtbot)
    assert dock.tree.topLevelItemCount() == 0
    assert dock.footer.text().startswith("No subprojects or markers.")
