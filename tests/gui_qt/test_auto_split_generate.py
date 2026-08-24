"""Tests for `QtTTSApp.auto_split_and_generate` (item 7, "Auto-split on
generation + combined-vs-separate clip generation", of the DAW-for-text
remaining-work roadmap). Mirrors tests/gui_qt/test_timeline_batch_generate.py's
conventions for driving generation and checking `engine.generate_dirty_clips`."""
from PySide6.QtWidgets import QMessageBox

from kokoro_gui.daw.models import Character


def _add_bob(qt_app) -> Character:
    """`qt_app.document` already has exactly one "Default" character
    (migration.py's fallback) - most tests here want a second, distinctly-
    named one so untagged-narration auto-clipping (Decision 2: only with
    EXACTLY ONE character) doesn't interfere with span-based assertions."""
    bob = Character.from_preset_dict("Bob", {"voice": "am_michael"})
    qt_app.document.characters.append(bob)
    return bob


def _two_speaker_text() -> str:
    return "[Default]: Hello there.\n\n[Bob]: Hi, how are you?"


# -- basic clip creation + batch-generate hookup -----------------------------


def test_auto_split_and_generate_creates_one_clip_per_span_and_calls_batch_generate(qt_app):
    bob = _add_bob(qt_app)
    default_character = qt_app.document.characters[0]
    qt_app.document.text = _two_speaker_text()

    qt_app.auto_split_and_generate()

    assert len(qt_app.document.clips) == 2
    character_ids = {c.character_id for c in qt_app.document.clips}
    assert character_ids == {default_character.id, bob.id}
    assert qt_app.engine.generate_dirty_clips.called


def test_auto_split_and_generate_is_noop_and_informs_when_nothing_to_split(qt_app, monkeypatch):
    info_calls = []
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: info_calls.append(a)))
    # Two characters (Default + Bob) + fully untagged text -> Decision 2 says
    # no auto-clipping (ambiguous which of two characters to use).
    _add_bob(qt_app)
    qt_app.document.text = "Untagged narration, no tags anywhere."

    qt_app.auto_split_and_generate()

    assert qt_app.document.clips == []
    assert not qt_app.engine.generate_dirty_clips.called
    assert info_calls


# -- undo ----------------------------------------------------------------------


def test_auto_split_and_generate_is_undoable(qt_app):
    _add_bob(qt_app)
    qt_app.document.text = _two_speaker_text()

    qt_app.auto_split_and_generate()

    assert len(qt_app.document.clips) == 2
    assert qt_app.document.undo_stack.can_undo() is True

    qt_app.undo()
    qt_app.undo()

    assert qt_app.document.clips == []


# -- one-job-at-a-time guard ---------------------------------------------------


def test_auto_split_and_generate_blocked_while_a_job_is_already_running(qt_app):
    _add_bob(qt_app)
    qt_app.document.text = _two_speaker_text()
    qt_app.cancel_btn.setEnabled(True)  # simulate a running job

    qt_app.auto_split_and_generate()

    assert qt_app.document.clips == []
    assert not qt_app.engine.generate_dirty_clips.called


# -- unmatched tag names --------------------------------------------------------


def test_unmatched_tag_name_warns_but_still_processes_matched_spans(qt_app, monkeypatch):
    warn_calls = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warn_calls.append(a)))
    bob = _add_bob(qt_app)
    qt_app.document.text = "[Carol]: Nobody matches this name.\n\n[Bob]: This one matches."

    qt_app.auto_split_and_generate()

    assert warn_calls  # the unmatched-name dialog fired
    assert len(qt_app.document.clips) == 1
    assert qt_app.document.clips[0].character_id == bob.id
    assert qt_app.engine.generate_dirty_clips.called


# -- auto_split_by_paragraph checkbox changes behavior end-to-end -------------


def test_auto_split_by_paragraph_checkbox_produces_finer_clips(qt_app):
    _add_bob(qt_app)
    text = "[Bob]: First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    qt_app.document.text = text

    assert qt_app.generation_dock.auto_split_paragraph_check.isChecked() is False
    qt_app.auto_split_and_generate()
    coarse_count = len(qt_app.document.clips)
    assert coarse_count == 1

    while qt_app.document.undo_stack.can_undo():
        qt_app.undo()
    assert qt_app.document.clips == []
    # generate_dirty_clips_requested() left the one-job-at-a-time guard
    # engaged (its dispatched future never resolves in this StubEngine-backed
    # test) - reset it so the second auto_split_and_generate() call below
    # isn't blocked by the first one's still-"running" job.
    qt_app.cancel_btn.setEnabled(False)

    qt_app.generation_dock.auto_split_paragraph_check.setChecked(True)
    assert qt_app.settings["auto_split_by_paragraph"] is True
    qt_app.document.text = text
    qt_app.auto_split_and_generate()
    fine_count = len(qt_app.document.clips)

    assert fine_count > coarse_count
    assert fine_count == 3
