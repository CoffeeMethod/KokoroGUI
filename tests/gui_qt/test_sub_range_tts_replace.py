"""Tests for item 9 ("Sub-range TTS replacement"):
`TimelineDock.on_sub_range_tts_requested` - the dialog offering an editable
sub-range transcript + any-character picker, sequencing a `TextEditCommand`
(if the text was edited) before an `AssignCharacterCommand`, then dispatching
item 3's `generate_dirty_clips_requested()`. Mirrors
tests/gui_qt/test_timeline_drag_reassign.py's conventions (qt_app fixture,
driving the dock's handler directly, monkeypatching the blocking modal so a
test can supply canned "user typed X and picked character Y" input without a
real dialog blocking).
"""
from PySide6.QtWidgets import QComboBox, QDialog, QPlainTextEdit

from kokoro_gui.daw.models import Character


def _make_whole_document_clip(qt_app, text="The quick brown fox jumps"):
    qt_app.document.text = text
    alice = qt_app.document.characters[0]
    clip = qt_app.document.assign_character_to_range(0, len(text), alice.id)
    return clip, alice


def _accept_dialog_with(new_text, character_id):
    """A QDialog.exec replacement that reaches into the dialog's own
    QPlainTextEdit/QComboBox (found via findChild, same as the real
    on_sub_range_tts_requested does after a real exec()) to set the "user
    typed X and picked character Y" state before resolving Accepted - same
    precedent as test_timeline_drag_reassign.py's QMessageBox.exec/
    clickedButton monkeypatching."""
    def _exec(self):
        self.findChild(QPlainTextEdit).setPlainText(new_text)
        combo = self.findChild(QComboBox)
        index = combo.findData(character_id)
        if index >= 0:
            combo.setCurrentIndex(index)
        return QDialog.DialogCode.Accepted
    return _exec


def _reject_dialog(self):
    return QDialog.DialogCode.Rejected


# ---------------------------------------------------------------------------
# Same text, a different character than the clip's own (Q27).
# ---------------------------------------------------------------------------

def test_same_text_different_character_creates_sub_clip_and_leaves_remainder(qt_app, monkeypatch):
    text = "The quick brown fox jumps"
    clip, alice = _make_whole_document_clip(qt_app, text)
    bob = Character.from_preset_dict("Bob", {})
    qt_app.document.characters.append(bob)

    sub_start, sub_end = 4, 9  # "quick"
    original_fragment = text[sub_start:sub_end]
    monkeypatch.setattr(QDialog, "exec", _accept_dialog_with(original_fragment, bob.id))

    qt_app.timeline_dock.on_sub_range_tts_requested(clip.id, sub_start, sub_end)

    assert qt_app.document.text == text  # no text edit happened

    new_clip = next(
        c for c in qt_app.document.clips if c.start_offset == sub_start and c.end_offset == sub_end
    )
    assert new_clip.character_id == bob.id

    remainder = [c for c in qt_app.document.clips if c.id != new_clip.id]
    assert remainder  # at least one leftover fragment
    assert all(c.character_id == alice.id for c in remainder)
    covered = sorted((c.start_offset, c.end_offset) for c in remainder)
    assert covered == [(0, sub_start), (sub_end, len(text))]

    assert qt_app.engine.generate_dirty_clips.called


def test_chosen_character_wins_over_parent_clips_own_character(qt_app, monkeypatch):
    """Q27's core claim, isolated: the new sub-clip's character is whichever
    one was explicitly picked in the dialog, never silently inherited from
    the parent clip."""
    text = "One two three four five"
    clip, alice = _make_whole_document_clip(qt_app, text)
    carol = Character.from_preset_dict("Carol", {})
    qt_app.document.characters.append(carol)
    assert clip.character_id == alice.id

    sub_start, sub_end = 0, 3  # "One"
    original_fragment = text[sub_start:sub_end]
    monkeypatch.setattr(QDialog, "exec", _accept_dialog_with(original_fragment, carol.id))

    qt_app.timeline_dock.on_sub_range_tts_requested(clip.id, sub_start, sub_end)

    new_clip = next(
        c for c in qt_app.document.clips if c.start_offset == sub_start and c.end_offset == sub_end
    )
    assert new_clip.character_id == carol.id
    assert new_clip.character_id != alice.id


# ---------------------------------------------------------------------------
# Edited text.
# ---------------------------------------------------------------------------

def test_edited_text_updates_document_and_resyncs_transcript_editor(qt_app, monkeypatch):
    text = "The quick brown fox jumps"
    clip, alice = _make_whole_document_clip(qt_app, text)

    sub_start, sub_end = 4, 9  # "quick"
    new_fragment = "slow"  # shorter than the original "quick"
    monkeypatch.setattr(QDialog, "exec", _accept_dialog_with(new_fragment, alice.id))

    qt_app.timeline_dock.on_sub_range_tts_requested(clip.id, sub_start, sub_end)

    expected_text = text[:sub_start] + new_fragment + text[sub_end:]
    assert qt_app.document.text == expected_text

    new_end = sub_start + len(new_fragment)
    new_clip = next(
        c for c in qt_app.document.clips if c.start_offset == sub_start and c.end_offset == new_end
    )
    assert new_clip.character_id == alice.id

    assert qt_app.generation_dock.text_entry.toPlainText() == expected_text


# ---------------------------------------------------------------------------
# Undo/redo.
# ---------------------------------------------------------------------------

def test_both_pushes_are_undoable_and_undo_twice_restores_original_state(qt_app, monkeypatch):
    text = "The quick brown fox jumps"
    clip, alice = _make_whole_document_clip(qt_app, text)
    original_clip_id = clip.id

    sub_start, sub_end = 4, 9  # "quick"
    new_fragment = "slow"
    monkeypatch.setattr(QDialog, "exec", _accept_dialog_with(new_fragment, alice.id))

    qt_app.timeline_dock.on_sub_range_tts_requested(clip.id, sub_start, sub_end)

    assert qt_app.document.undo_stack.can_undo() is True
    assert qt_app.document.text != text

    qt_app.undo()  # undoes AssignCharacterCommand
    qt_app.undo()  # undoes TextEditCommand

    assert qt_app.document.text == text
    assert len(qt_app.document.clips) == 1
    restored_clip = qt_app.document.clips[0]
    assert restored_clip.id == original_clip_id
    assert restored_clip.start_offset == 0
    assert restored_clip.end_offset == len(text)


# ---------------------------------------------------------------------------
# Cancel.
# ---------------------------------------------------------------------------

def test_cancelling_dialog_pushes_nothing(qt_app, monkeypatch):
    text = "The quick brown fox jumps"
    clip, _alice = _make_whole_document_clip(qt_app, text)
    can_undo_before = qt_app.document.undo_stack.can_undo()
    clip_ids_before = {c.id for c in qt_app.document.clips}

    monkeypatch.setattr(QDialog, "exec", _reject_dialog)

    qt_app.timeline_dock.on_sub_range_tts_requested(clip.id, 4, 9)

    assert qt_app.document.undo_stack.can_undo() is can_undo_before
    assert qt_app.document.text == text
    assert {c.id for c in qt_app.document.clips} == clip_ids_before
    assert qt_app.engine.generate_dirty_clips.called is False
