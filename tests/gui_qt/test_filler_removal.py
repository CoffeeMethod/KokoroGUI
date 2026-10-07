"""Removing filler words from an imported recording (plan 32): the editor's
`delete_ranges` as one undoable edit, and the Edit > Remove Filler Words
dialog."""
import copy

import numpy as np
import soundfile as sf

from kokoro_gui.daw import fillers, imported
from kokoro_gui.daw.undo import ImportRecordingCommand
from kokoro_gui.qt import project as project_io

RATE = 16000
# One word per half second, so every word has a time of its own.
SAYING = [("So,", 0.0, 0.4), ("um,", 0.5, 0.9), ("we", 1.0, 1.4), ("went", 1.5, 1.9), ("uh", 2.0, 2.4),
          ("to", 2.5, 2.9), ("the", 3.0, 3.4), ("store", 3.5, 3.9), ("erm.", 4.0, 4.4)]
LATER = [("Then", 6.0, 6.4), ("we", 6.5, 6.9), ("came", 7.0, 7.4), ("home", 7.5, 7.9)]


def _recording(qt_app, tmp_path, rows):
    """Imports a 10 s file and appends one recording clip per row list, as
    File > Import Audio's commit does. Returns `(source, clip_ids)`."""
    path = str(tmp_path / "talk.wav")
    sf.write(path, np.linspace(-0.2, 0.2, RATE * 10).astype(np.float32), RATE)
    stored = project_io.import_audio_file(path, qt_app.project_dir)
    source, entry = imported.source_entry(stored)
    host = qt_app.document.characters[0]
    command = ImportRecordingCommand(
        [dict(zip(("text", "words"), imported.run_from_asr_words(words, source)), character_id=host.id)
         for words in rows], {source: entry})
    qt_app.document.undo_stack.push(command)
    qt_app.editor.load_text(qt_app.document.text)
    return source, command.clip_ids


def _times(document):
    """`{word text: (start_s, end_s)}` of every timed word."""
    out = {}
    for clip in document.clips:
        for start, end, _source, start_s, end_s in imported.clip_words(document, clip):
            out[document.text[start:end]] = (start_s, end_s)
    return out


def test_delete_ranges_removes_every_range_as_one_undo_step(qt_app, tmp_path):
    document, editor = qt_app.document, qt_app.editor
    _source, (clip_id,) = _recording(qt_app, tmp_path, [SAYING])
    before_text = document.text
    before_runs = copy.deepcopy(document.runs)
    before_times = _times(document)
    hits = fillers.find_fillers(document)
    assert [h.text for h in hits] == ["um", "uh", "erm"]

    removed = editor.delete_ranges([(h.start, h.end) for h in hits])

    assert removed == 3
    assert document.text == "So, we went to the store."
    assert editor.toPlainText() == document.text
    # The words that stay keep their times, so the audio cuts line up.
    after = _times(document)
    assert set(after) == set(before_times) - {"um", "uh", "erm"}
    assert all(after[word] == before_times[word] for word in after)
    # The cuts skip the fillers' audio: "um," 0.45-0.95 s, "uh" 1.95-2.45 s and
    # "erm." from 3.95 s are not played.
    ranges = [s.range for s in document.get_clip(clip_id).segments]
    assert ranges == [[0.0, 0.45], [0.95, 1.95], [2.45, 3.95]]

    qt_app.undo()
    assert document.text == before_text and editor.toPlainText() == before_text
    assert document.runs == before_runs
    assert _times(document) == before_times

    qt_app.redo()
    assert editor.toPlainText() == document.text
    assert set(_times(document)) == set(before_times) - {"um", "uh", "erm"}


def test_one_undo_restores_a_recording_cut_in_two_clips(qt_app, tmp_path):
    document, editor = qt_app.document, qt_app.editor
    _recording(qt_app, tmp_path, [SAYING, LATER])
    before = copy.deepcopy(document.runs)
    hits = fillers.find_fillers(document)

    editor.delete_ranges([(h.start, h.end) for h in hits])
    qt_app.undo()

    assert document.runs == before and editor.toPlainText() == document.text


def test_a_typed_edit_before_the_removal_keeps_its_own_undo_step(qt_app, tmp_path):
    from PySide6.QtGui import QTextCursor

    document, editor = qt_app.document, qt_app.editor
    _recording(qt_app, tmp_path, [SAYING])
    original = document.text
    cursor = QTextCursor(editor.document())
    cursor.movePosition(QTextCursor.MoveOperation.End)
    cursor.insertText("!")
    typed = document.text
    assert typed == original + "!"

    editor.delete_ranges([(h.start, h.end) for h in fillers.find_fillers(document)])
    assert document.text == "So, we went to the store.!"

    qt_app.undo()
    assert document.text == typed and editor.toPlainText() == typed
    qt_app.undo()
    assert document.text == original and editor.toPlainText() == original
    qt_app.redo()
    qt_app.redo()
    assert document.text == "So, we went to the store.!" and editor.toPlainText() == document.text


def test_delete_ranges_merges_overlaps_and_ignores_empty_ones(qt_app, tmp_path):
    document, editor = qt_app.document, qt_app.editor
    _recording(qt_app, tmp_path, [SAYING])
    text = document.text
    start = text.index("went")

    removed = editor.delete_ranges([(start, start + 3), (start + 2, start + 5), (4, 4)])

    assert removed == 1
    assert document.text == text[:start] + text[start + 5:]
    qt_app.undo()
    assert document.text == text


# -- the dialog and the Edit menu entry ----------------------------------------


def _answer(qt_app, monkeypatch, edit=None, accept=True):
    """Patches the dialog's modal: `edit(dialog)` runs first (tick boxes),
    then the dialog is accepted or cancelled. Returns the dialogs shown."""
    shown = []

    def _ask(dialog):
        shown.append(dialog)
        if edit is not None:
            edit(dialog)
        return accept

    monkeypatch.setattr(qt_app, "_ask_fillers", _ask)
    return shown


def test_the_edit_menu_has_the_entry(qt_app):
    labels = [a.text() for a in qt_app.edit_menu.actions()]
    assert "Remove &Filler Words..." in labels


def test_the_dialog_lists_every_hit_with_the_safe_ones_checked(qt_app, tmp_path, monkeypatch):
    _recording(qt_app, tmp_path, [[("So,", 0.0, 0.4), ("like,", 0.5, 0.9), ("um", 1.0, 1.4), ("we", 1.5, 1.9),
                                   ("left", 2.0, 2.4)]])
    shown = _answer(qt_app, monkeypatch, accept=False)

    assert qt_app.remove_filler_words() == 0

    dialog, = shown
    assert [h.text for h in dialog.hits()] == ["like", "um"]
    assert [dialog.is_checked(i) for i in range(dialog.row_count())] == [False, True]
    assert dialog.remove_button.text() == "Remove 1"
    assert qt_app.document.text == "So, like, um we left"


def test_unchecked_rows_stay_and_checked_rows_go(qt_app, tmp_path, monkeypatch):
    _recording(qt_app, tmp_path, [[("So,", 0.0, 0.4), ("like,", 0.5, 0.9), ("um", 1.0, 1.4), ("we", 1.5, 1.9),
                                   ("left", 2.0, 2.4)]])
    document = qt_app.document
    before = copy.deepcopy(document.runs)
    _answer(qt_app, monkeypatch)

    assert qt_app.remove_filler_words() == 1
    assert document.text == "So, like, we left"
    assert "um" not in qt_app.editor.toPlainText().split()

    qt_app.undo()
    assert document.runs == before and qt_app.editor.toPlainText() == "So, like, um we left"


def test_checking_a_context_row_removes_it_too(qt_app, tmp_path, monkeypatch):
    _recording(qt_app, tmp_path, [[("So,", 0.0, 0.4), ("like,", 0.5, 0.9), ("we", 1.0, 1.4), ("left", 1.5, 1.9)]])
    _answer(qt_app, monkeypatch, edit=lambda dialog: dialog.set_checked(0, True))

    assert qt_app.remove_filler_words() == 1
    assert qt_app.document.text == "So, we left"


def test_cancel_and_an_empty_selection_change_nothing(qt_app, tmp_path, monkeypatch):
    _recording(qt_app, tmp_path, [SAYING])
    text = qt_app.document.text
    _answer(qt_app, monkeypatch, accept=False)
    assert qt_app.remove_filler_words() == 0 and qt_app.document.text == text

    _answer(qt_app, monkeypatch, edit=lambda dialog: dialog.set_all(False))
    assert qt_app.remove_filler_words() == 0 and qt_app.document.text == text
    assert qt_app.editor.toPlainText() == text


def test_with_no_fillers_it_says_so_and_shows_no_dialog(qt_app, tmp_path, monkeypatch):
    _recording(qt_app, tmp_path, [LATER])
    shown = _answer(qt_app, monkeypatch)

    assert qt_app.remove_filler_words() == 0

    assert shown == []


def test_generated_text_is_never_offered(qt_app, monkeypatch):
    from PySide6.QtGui import QTextCursor

    cursor = QTextCursor(qt_app.editor.document())
    cursor.insertText("Well um we went")
    qt_app.document.assign_character_to_range(0, len(qt_app.document.text), qt_app.document.characters[0].id)
    shown = _answer(qt_app, monkeypatch)

    assert qt_app.remove_filler_words() == 0 and shown == []


def test_play_reads_the_fillers_slice_of_the_recording_with_a_margin(qt_app, tmp_path, monkeypatch):
    from kokoro_gui.qt import filler_dialog

    _source, _ids = _recording(qt_app, tmp_path, [SAYING])
    played = []
    monkeypatch.setattr(filler_dialog.playback, "play_range", lambda path, a, b: played.append((path, a, b)))
    hits = fillers.find_fillers(qt_app.document)
    dialog = filler_dialog.FillerDialog(qt_app.document, hits)

    assert dialog.play(0) is True

    (path, start_s, end_s), = played
    assert path.endswith(".wav") and "imported" in path
    assert start_s == 0.45 - filler_dialog.PLAY_MARGIN_S
    assert end_s == 0.95 + filler_dialog.PLAY_MARGIN_S
