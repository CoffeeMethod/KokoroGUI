"""Find and replace in the transcript (plan 34): the find bar, its keys, the
highlights, and `TranscriptEditor.replace_ranges` with Replace and Replace
all, over the open project and its subprojects."""
import copy

import numpy as np
import soundfile as sf
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtTest import QTest

from kokoro_gui.daw.dirty import build_segments_from_results, compute_expected_cache_hash
from kokoro_gui.qt import keymap
from kokoro_gui.qt import theme
from kokoro_gui.qt.about_dialog import shortcuts_text

CTRL = Qt.KeyboardModifier.ControlModifier
SHIFT = Qt.KeyboardModifier.ShiftModifier
LINES = ["The cat sat on the mat.", "A cat and a cat.", "No felines here.", "Concatenate it."]
TEXT = "\n".join(LINES)


def _show_active(qt_app, qtbot):
    qt_app.show()
    qtbot.waitExposed(qt_app)
    qt_app.activateWindow()
    qtbot.waitUntil(qt_app.isActiveWindow, timeout=2000)


def _load(qt_app, lines=LINES, clip_lines=(0, 1, 2)):
    """The transcript is `lines`; the lines in `clip_lines` are clips of the
    first character."""
    document, editor = qt_app.document, qt_app.editor
    document.set_plain_text("\n".join(lines))
    character = document.characters[0]
    for number in (n for n in clip_lines if n < len(lines)):
        start = sum(len(line) + 1 for line in lines[:number])
        document.assign_character_to_range(start, start + len(lines[number]), character.id)
    editor.load_text(document.text)
    editor.rehighlight()
    return document, editor


def _bar(qt_app):
    return qt_app.transcript_dock.find_bar


def _search(qt_app, needle, replacement=None, match_case=False, whole_word=False, replace_row=False):
    bar = _bar(qt_app)
    bar.open(replace=replace_row or replacement is not None)
    bar.match_case_check.setChecked(match_case)
    bar.whole_word_check.setChecked(whole_word)
    bar.find_edit.setText(needle)
    if replacement is not None:
        bar.replace_edit.setText(replacement)
    bar.refresh()
    return bar


def _selected(editor):
    return editor.textCursor().selectedText()


# -- opening and closing -------------------------------------------------------------------

def test_the_bar_starts_hidden_with_no_highlights(qt_app):
    bar = _bar(qt_app)
    assert bar.isHidden() and not bar.is_open()
    assert qt_app.editor.find_matches() == []


def test_ctrl_f_opens_the_bar_and_esc_closes_it(qt_app, qtbot):
    _load(qt_app)
    _show_active(qt_app, qtbot)
    editor = qt_app.editor
    editor.setFocus()

    QTest.keyClick(editor, Qt.Key.Key_F, CTRL)

    bar = _bar(qt_app)
    assert bar.is_open() and bar.find_edit.hasFocus()
    assert bar.replace_row.isHidden()
    bar.find_edit.setText("cat")
    bar.refresh()
    assert editor.find_matches()

    QTest.keyClick(bar.find_edit, Qt.Key.Key_Escape)

    assert not bar.is_open() and bar.isHidden()
    assert editor.find_matches() == []
    assert editor.hasFocus()


def test_ctrl_h_opens_the_bar_with_the_replace_row(qt_app, qtbot):
    _load(qt_app)
    _show_active(qt_app, qtbot)
    qt_app.editor.setFocus()

    QTest.keyClick(qt_app.editor, Qt.Key.Key_H, CTRL)

    bar = _bar(qt_app)
    assert bar.is_open() and bar.replace_row.isVisibleTo(bar)
    assert bar.find_edit.hasFocus()


def test_ctrl_f_works_with_the_timeline_focused(qt_app, qtbot):
    _load(qt_app)
    _show_active(qt_app, qtbot)
    qt_app.timeline_dock.timeline_view.setFocus()

    QTest.keyClick(qt_app.timeline_dock.timeline_view, Qt.Key.Key_F, CTRL)

    assert _bar(qt_app).is_open()


def test_the_find_keys_are_rows_of_the_keymap_and_in_the_shortcut_sheet(qt_app):
    rows = {b.id: b for b in keymap.KEYS}
    assert rows["find"].sequence == "Ctrl+F" and rows["find"].scope == keymap.APP
    assert rows["replace"].sequence == "Ctrl+H" and rows["replace"].scope == keymap.APP
    assert rows["find_next"].sequence == "F3" and rows["find_previous"].sequence == "Shift+F3"
    text = shortcuts_text(qt_app)
    for binding in (rows["find"], rows["replace"], rows["find_next"], rows["find_previous"]):
        assert binding.label in text
        assert hasattr(qt_app, f"{binding.id}_shortcut")


def test_the_selection_is_carried_into_the_find_field(qt_app):
    _document, editor = _load(qt_app)
    editor.select_span(TEXT.index("felines"), TEXT.index("felines") + 7)

    _bar(qt_app).open()

    assert _bar(qt_app).find_edit.text() == "felines"


def test_the_edit_menu_has_find_and_replace_actions(qt_app):
    qt_app.find_replace_action.trigger()
    assert _bar(qt_app).is_open() and _bar(qt_app).replace_row.isVisibleTo(_bar(qt_app))
    qt_app.find_action.trigger()
    assert _bar(qt_app).is_open()


# -- the highlights and the counter ------------------------------------------------------------

def test_typing_in_the_find_field_highlights_and_counts_after_the_debounce(qt_app, qtbot):
    _document, editor = _load(qt_app)
    bar = _bar(qt_app)
    bar.open()

    bar.find_edit.setText("cat")

    qtbot.waitUntil(lambda: editor.find_matches() != [], timeout=2000)
    spans = editor.find_matches()
    assert [TEXT[s:e].lower() for s, e in spans] == ["cat", "cat", "cat", "cat"]
    assert bar.counter_label.text() == "1 of 4"


def test_match_case_and_whole_word_change_what_is_found(qt_app):
    _load(qt_app, ["The Cat", "the cat", "concatenate cat"])
    bar = _search(qt_app, "cat")
    assert len(bar.hits()) == 4
    bar.match_case_check.setChecked(True)
    bar.refresh()
    assert len(bar.hits()) == 3
    bar.whole_word_check.setChecked(True)
    bar.refresh()
    assert len(bar.hits()) == 2


def test_the_counter_says_no_matches_and_the_buttons_wait(qt_app):
    _load(qt_app)
    bar = _search(qt_app, "zebra")
    assert bar.counter_label.text() == "No matches"
    assert not bar.next_btn.isEnabled() and not bar.previous_btn.isEnabled()
    assert qt_app.editor.find_matches() == []


def test_the_current_match_has_its_own_color(qt_app, qtbot):
    _document, editor = _load(qt_app)
    _show_active(qt_app, qtbot)
    bar = _search(qt_app, "cat")
    bar.find_next()

    colors = [sel.format.background().color().name() for sel in editor.extraSelections()]

    assert theme.current().find_current in colors
    assert colors.count(theme.current().find_match) >= 1
    assert theme.LIGHT.find_current != theme.DARK.find_current
    assert theme.LIGHT.find_match != theme.DARK.find_match


def test_matches_in_a_subproject_line_are_not_found(qt_app):
    document, editor = _load(qt_app, ["see the cat", "x"])
    child = qt_app.new_subproject(len("see the cat\n"), title="Cat chapter")
    assert child is not None
    bar = _search(qt_app, "cat")
    # The subproject is a project of its own; its line in this transcript never matches.
    assert bar.hits() == [(len("see the "), len("see the cat"))]


# -- stepping -------------------------------------------------------------------------------

def test_f3_moves_the_selection_to_the_next_match_and_shift_f3_back(qt_app, qtbot):
    _document, editor = _load(qt_app)
    _show_active(qt_app, qtbot)
    bar = _search(qt_app, "cat")
    cursor = editor.textCursor()
    cursor.setPosition(0)
    editor.setTextCursor(cursor)
    editor.setFocus()

    QTest.keyClick(editor, Qt.Key.Key_F3)
    first = editor.textCursor().selectionStart()
    assert first == TEXT.index("cat") and _selected(editor) == "cat"
    assert bar.counter_label.text() == "1 of 4"

    QTest.keyClick(editor, Qt.Key.Key_F3)
    assert editor.textCursor().selectionStart() == bar.hits()[1][0]
    assert bar.counter_label.text() == "2 of 4"

    QTest.keyClick(editor, Qt.Key.Key_F3, SHIFT)
    assert editor.textCursor().selectionStart() == first


def test_stepping_wraps_round_the_transcript(qt_app):
    _document, editor = _load(qt_app)
    bar = _search(qt_app, "cat")
    hits = bar.hits()
    last = hits[-1]
    editor.select_span(*last)

    assert bar.find_next() is True
    assert editor.textCursor().selectionStart() == hits[0][0]
    assert bar.find_previous() is True
    assert editor.textCursor().selectionStart() == last[0]


def test_f3_with_the_bar_closed_opens_it(qt_app):
    _load(qt_app)
    qt_app.find_next_match()
    assert _bar(qt_app).is_open()


def test_enter_in_the_find_field_goes_to_the_next_match_and_shift_enter_back(qt_app, qtbot):
    _document, editor = _load(qt_app)
    bar = _search(qt_app, "cat")
    cursor = editor.textCursor()
    cursor.setPosition(0)
    editor.setTextCursor(cursor)

    QTest.keyClick(bar.find_edit, Qt.Key.Key_Return)
    assert editor.textCursor().selectionStart() == bar.hits()[0][0]
    QTest.keyClick(bar.find_edit, Qt.Key.Key_Return)
    assert editor.textCursor().selectionStart() == bar.hits()[1][0]
    QTest.keyClick(bar.find_edit, Qt.Key.Key_Return, SHIFT)
    assert editor.textCursor().selectionStart() == bar.hits()[0][0]


def test_typing_in_the_transcript_refreshes_the_matches(qt_app, qtbot):
    _document, editor = _load(qt_app)
    bar = _search(qt_app, "cat")
    assert len(bar.hits()) == 4

    cursor = editor.textCursor()
    cursor.setPosition(0)
    cursor.insertText("cat cat ")

    qtbot.waitUntil(lambda: len(bar.hits()) == 6, timeout=2000)


# -- busy ----------------------------------------------------------------------------------

def test_nothing_is_searched_or_replaced_while_a_generate_runs(qt_app, monkeypatch):
    _document, editor = _load(qt_app)
    bar = _search(qt_app, "cat", replacement="dog")
    assert bar.hits()
    monkeypatch.setattr(qt_app, "is_busy", lambda: True)

    bar.on_busy_changed()

    assert bar.hits() == [] and editor.find_matches() == []
    assert "generate" in bar.counter_label.text()
    assert not bar.replace_btn.isEnabled() and not bar.replace_all_btn.isEnabled()
    before = qt_app.document.text
    assert bar.replace_all() == 0
    assert qt_app.document.text == before

    monkeypatch.setattr(qt_app, "is_busy", lambda: False)
    bar.on_busy_changed()
    assert len(bar.hits()) == 4 and bar.replace_all_btn.isEnabled()


# -- replace_ranges and Replace -----------------------------------------------------------------

def _clean_clip(qt_app, tmp_path, start, end, name):
    """A clip over `[start, end)` that is not stale: it has a segment for its text."""
    character = qt_app.document.characters[0]
    clip = qt_app.document.assign_character_to_range(start, end, character.id)
    path = str(tmp_path / f"{name}.wav")
    sf.write(path, np.full(2400, 0.2, dtype=np.float32), 24000)
    text = qt_app.document.clip_text(clip)
    expected = compute_expected_cache_hash(text, qt_app.document.effective_config_for_clip(clip))
    clip.segments = build_segments_from_results(expected, [{"text": text, "path": path, "duration": 0.1}])
    return clip


def _clean_lines(qt_app, tmp_path, lines=LINES, clip_lines=(0, 1, 2)):
    """`_load` with every clip generated, so a replacement turns it stale."""
    document, editor = qt_app.document, qt_app.editor
    document.set_plain_text("\n".join(lines))
    editor.load_text(document.text)
    clips = []
    for number in clip_lines:
        start = sum(len(line) + 1 for line in lines[:number])
        clips.append(_clean_clip(qt_app, tmp_path, start, start + len(lines[number]), f"clip{number}"))
    assert document.dirty_clips() == []
    editor.rehighlight()
    return document, editor, clips


def _runs(document):
    return copy.deepcopy(document.runs)


def test_replace_all_is_one_undo_step_that_gives_the_clips_back(qt_app, tmp_path):
    document, editor, clips = _clean_lines(qt_app, tmp_path)
    before_text, before_runs = document.text, _runs(document)
    bar = _search(qt_app, "cat", replacement="dog")
    assert bar.summary_label.text() == "Replaces 4 matches in 2 clips. 2 clips will turn stale."

    assert bar.replace_all() == 4

    assert document.text == before_text.replace("cat", "dog")
    assert editor.toPlainText() == document.text
    # The stated count is the clips that did turn stale, and they are the two that held a match.
    assert {c.id for c in document.dirty_clips()} == {clips[0].id, clips[1].id}
    assert [c.id for c in document.clips] == [c.id for c in clips]

    qt_app.undo()
    assert document.text == before_text and editor.toPlainText() == before_text
    assert document.runs == before_runs
    assert document.dirty_clips() == []

    qt_app.redo()
    assert document.text == before_text.replace("cat", "dog")
    assert editor.toPlainText() == document.text
    assert {c.id for c in document.dirty_clips()} == {clips[0].id, clips[1].id}


def test_replace_all_leaves_the_typing_history_before_it_alone(qt_app, tmp_path):
    document, editor, _clips = _clean_lines(qt_app, tmp_path)
    cursor = QTextCursor(editor.document())
    cursor.movePosition(QTextCursor.MoveOperation.End)
    cursor.insertText("!")
    typed = document.text
    bar = _search(qt_app, "felines", replacement="lynx")

    bar.replace_all()
    assert "lynx" in document.text

    qt_app.undo()
    assert document.text == typed and editor.toPlainText() == typed
    qt_app.undo()
    assert document.text == typed[:-1]


def test_a_clip_that_is_only_the_match_keeps_its_character(qt_app, tmp_path):
    document, editor, clips = _clean_lines(qt_app, tmp_path, ["Hello", "World"], (0, 1))
    bar = _search(qt_app, "Hello", replacement="Hi")

    bar.replace_all()

    assert document.text == "Hi\nWorld"
    assert [c.id for c in document.clips] == [c.id for c in clips]
    assert document.get_clip(clips[0].id).character_id == document.characters[0].id
    assert document.clip_text(clips[0]) == "Hi"
    qt_app.undo()
    assert document.clip_text(document.get_clip(clips[0].id)) == "Hello"
    assert document.dirty_clips() == []


def test_replacing_with_nothing_deletes_the_matches(qt_app, tmp_path):
    document, _editor, _clips = _clean_lines(qt_app, tmp_path)
    bar = _search(qt_app, " cat", replacement="", match_case=True)

    bar.replace_all()

    assert document.text == "The sat on the mat.\nA and a.\nNo felines here.\nConcatenate it."
    qt_app.undo()
    assert document.text == TEXT


def test_a_match_that_already_reads_as_the_replacement_changes_nothing(qt_app, tmp_path):
    document, _editor, _clips = _clean_lines(qt_app, tmp_path)
    bar = _search(qt_app, "cat", replacement="cat", match_case=True)

    assert bar.summary_label.isHidden()
    assert bar.replace_all() == 0
    assert document.dirty_clips() == []
    assert document.text == TEXT


def test_replace_goes_one_match_at_a_time(qt_app, tmp_path):
    document, editor, _clips = _clean_lines(qt_app, tmp_path)
    bar = _search(qt_app, "cat", replacement="dog")
    editor.select_span(0, 0)

    bar.replace_btn.click()  # nothing selected yet: it only finds the first match
    assert document.text == TEXT and _selected(editor) == "cat"
    bar.replace_btn.click()
    assert document.text.startswith("The dog sat") and document.text.count("cat") == 3
    assert _selected(editor) == "cat" and editor.textCursor().selectionStart() > TEXT.index("cat")
    for _ in range(3):
        bar.replace_btn.click()
    assert document.text == TEXT.replace("cat", "dog")
    assert bar.counter_label.text() == "No matches"


def test_replace_in_the_bar_is_one_undo_step_per_match(qt_app, tmp_path):
    document, editor, _clips = _clean_lines(qt_app, tmp_path)
    bar = _search(qt_app, "cat", replacement="dog")
    editor.select_span(*bar.hits()[0])

    bar.replace_btn.click()
    bar.replace_btn.click()
    assert document.text.count("dog") == 2

    qt_app.undo()
    assert document.text.count("dog") == 1
    qt_app.undo()
    assert document.text == TEXT


def test_replace_ranges_skips_overlaps_and_no_ops_and_reports_the_count(qt_app, tmp_path):
    document, editor, _clips = _clean_lines(qt_app, tmp_path, ["one two three"], (0,))

    # (2, 5) overlaps (0, 3); (9, 9) is empty; (8, 13) is already "three".
    assert editor.replace_ranges([(0, 3, "1"), (2, 5, "x"), (4, 7, "2"), (9, 9, "z"), (8, 13, "three")]) == 2
    assert document.text == "1 2 three"
    assert editor.replace_ranges([]) == 0


# -- recordings -------------------------------------------------------------------------------

def test_a_replacement_inside_a_recording_splits_it_the_way_typing_does(qt_app, tmp_path):
    from tests.gui_qt.test_filler_removal import SAYING, _recording, _times

    document, editor = qt_app.document, qt_app.editor
    _recording(qt_app, tmp_path, [SAYING])
    before_text, before_runs, before_times = document.text, _runs(document), _times(document)
    bar = _search(qt_app, "store", replacement="shop")
    assert bar.summary_label.text() == "Replaces 1 match in 1 clip. 1 recording will split."

    bar.replace_all()
    replaced = [(r.text, r.kind, r.words) for r in document.runs]
    replaced_clips = len(document.clips)
    assert document.text == before_text.replace("store", "shop")

    qt_app.undo()
    assert document.text == before_text and document.runs == before_runs
    assert _times(document) == before_times

    # The same edit typed over the selected word.
    start = before_text.index("store")
    editor.select_span(start, start + len("store"))
    editor.textCursor().insertText("shop")
    typed = [(r.text, r.kind, r.words) for r in document.runs]

    assert replaced == typed
    assert replaced_clips == len(document.clips) == 2
    # The new word has no audio of its own; the others keep their times.
    after = _times(document)
    assert "shop" not in after and "store" not in after
    assert all(after[word] == before_times[word] for word in after)


def test_one_replace_all_over_a_recording_and_a_generated_clip(qt_app, tmp_path):
    from tests.gui_qt.test_filler_removal import SAYING, _recording

    document = qt_app.document
    _recording(qt_app, tmp_path, [SAYING])
    tail = "We went home."
    start = len(document.text) + 1
    cursor = QTextCursor(qt_app.editor.document())
    cursor.movePosition(QTextCursor.MoveOperation.End)
    cursor.insertText("\n" + tail)
    clip = _clean_clip(qt_app, tmp_path, start, start + len(tail), "tail")
    before_text, before_runs = document.text, _runs(document)
    bar = _search(qt_app, "went", replacement="gone")
    assert bar.summary_label.text() == "Replaces 2 matches in 2 clips. 1 clip will turn stale. 1 recording will split."

    bar.replace_all()
    assert document.text == before_text.replace("went", "gone")
    assert document.get_clip(clip.id) in document.dirty_clips()

    qt_app.undo()
    assert document.text == before_text and document.runs == before_runs
    assert document.get_clip(clip.id) not in document.dirty_clips()


# -- subprojects ------------------------------------------------------------------------------

def _book(qt_app):
    """"The cat sat. A cat tale. Another cat story. End cat." with the two
    middle clips in subprojects "Tale" and "Story"; every clip generated."""
    from tests.gui_qt.test_subprojects import _generated_clip

    text = "The cat sat. A cat tale. Another cat story. End cat."
    qt_app.document.text = text
    qt_app.editor.load_text(text)
    for piece in ("The cat sat.", "A cat tale.", "Another cat story.", "End cat."):
        start = text.index(piece)
        _generated_clip(qt_app, start, start + len(piece))
    story_at = text.index("Another")
    story = qt_app.new_subproject(story_at, story_at + len("Another cat story."), title="Story")
    tale_at = text.index("A cat tale.")
    tale = qt_app.new_subproject(tale_at, tale_at + len("A cat tale."), title="Tale")
    assert qt_app.document.text == "The cat sat. Tale Story End cat."
    return tale, story


def test_the_bar_lists_the_matches_of_every_open_subproject(qt_app):
    tale, story = _book(qt_app)

    bar = _search(qt_app, "cat", replacement="dog")

    assert len(bar.hits()) == 2
    assert [(p.title(), n) for p, n in bar.other_hits()] == [("Tale", 1), ("Story", 1)]
    rows = [bar.projects_list.item(i).text() for i in range(bar.projects_list.count())]
    assert rows == ["Tale: 1 match", "Story: 1 match"]
    assert bar.summary_label.text() == "Replaces 4 matches in 4 clips across 3 projects. 4 clips will turn stale."
    assert tale.document.text == "A cat tale." and story.document.text == "Another cat story."


def test_replace_all_reaches_every_subproject_and_each_undoes_on_its_own_stack(qt_app, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    tale, story = _book(qt_app)
    root = qt_app.document
    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: asked.append(a[2]) or QMessageBox.StandardButton.Yes))
    bar = _search(qt_app, "cat", replacement="dog")

    assert bar.replace_all() == 4

    assert len(asked) == 1 and "Tale (1)" in asked[0] and "Story (1)" in asked[0]
    assert root.text == "The dog sat. Tale Story End dog."
    assert tale.document.text == "A dog tale." and story.document.text == "Another dog story."
    assert all(c in c_doc.dirty_clips() for c_doc in (root, tale.document, story.document)
               for c in c_doc.clips if not c.is_nested)
    assert qt_app.editor.toPlainText() == root.text
    qt_app._autosave_project_dir()
    assert tale.dirty and story.dirty

    # The editor's undo is the open project's own.
    qt_app.undo()
    assert root.text == "The cat sat. Tale Story End cat."
    assert tale.document.text == "A dog tale." and story.document.text == "Another dog story."
    # A subproject has its own stack: one undo takes back all of its matches.
    story.document.undo_stack.undo()
    assert story.document.text == "Another cat story." and tale.document.text == "A dog tale."


def test_a_subproject_undoes_with_ctrl_z_once_it_has_the_editor(qt_app):
    tale, story = _book(qt_app)
    bar = _search(qt_app, "cat", replacement="dog")
    bar.replace_all()

    qt_app.set_focus(story)
    assert qt_app.document is story.document
    qt_app.undo()
    assert story.document.text == "Another cat story."
    assert qt_app.editor.toPlainText() == "Another cat story."
    assert tale.document.text == "A dog tale."
    qt_app.redo()
    assert story.document.text == "Another dog story."


def test_declining_the_question_changes_nothing(qt_app, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    tale, story = _book(qt_app)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    bar = _search(qt_app, "cat", replacement="dog")

    assert bar.replace_all() == 0

    assert qt_app.document.text == "The cat sat. Tale Story End cat."
    assert tale.document.text == "A cat tale." and story.document.text == "Another cat story."


def test_a_double_click_on_a_row_shows_that_subproject(qt_app):
    tale, _story = _book(qt_app)
    bar = _search(qt_app, "cat")

    bar.projects_list.itemDoubleClicked.emit(bar.projects_list.item(0))

    assert qt_app.focus is tale
    assert qt_app.document is tale.document
    assert bar.hits() == [(2, 5)]
    # The root is now one of the other projects.
    assert [(p.title(), n) for p, n in bar.other_hits()] == [(qt_app.root.title(), 2), ("Story", 1)]


def test_a_subproject_that_is_not_open_is_left_out_with_a_note(qt_app, monkeypatch):
    tale, story = _book(qt_app)
    story_clip = qt_app.document.get_clip(story.clip_id)
    real = qt_app.child_project
    monkeypatch.setattr(qt_app, "child_project", lambda clip: None if clip.id == story_clip.id else real(clip))
    monkeypatch.setattr(qt_app, "open_projects", lambda: [qt_app.root, tale])

    bar = _search(qt_app, "cat", replacement="dog")

    assert [(p.title(), n) for p, n in bar.other_hits()] == [("Tale", 1)]
    assert bar.closed_label.text() == "1 subproject not opened, so not searched."
    assert not bar.closed_label.isHidden()
    bar.replace_all()
    assert story.document.text == "Another cat story."
    assert tale.document.text == "A dog tale."
