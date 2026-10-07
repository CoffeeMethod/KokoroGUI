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
