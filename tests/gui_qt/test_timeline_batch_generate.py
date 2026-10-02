"""Tests for the consolidated action bar's batch dirty-scoped Generate path
(item 3, "Consolidated action bar + batch dirty-scoped generation" of the
DAW-for-text remaining-work roadmap): `QtTTSApp.on_generate_clicked`
dispatching to `TimelineDock.generate_dirty_clips_requested`, and that
dock's handling of a resolved `KokoroEngine.generate_dirty_clips` future.
Mirrors tests/gui_qt/test_timeline_clip_generate.py's conventions."""
from kokoro_gui.daw.dirty import build_segments_from_results, compute_expected_cache_hash


def _make_clip(qt_app, start=0, end=5, text="hello world"):
    qt_app.document.text = text
    character = qt_app.document.characters[0]
    return qt_app.document.assign_character_to_range(start, end, character.id)


def _make_two_dirty_clips(qt_app):
    text = "First clip text here. Second clip text here."
    qt_app.document.text = text
    character = qt_app.document.characters[0]
    split = text.index(" Second")
    clip_a = qt_app.document.assign_character_to_range(0, split, character.id)
    clip_b = qt_app.document.assign_character_to_range(split, len(text), character.id)
    return clip_a, clip_b


def _mark_not_dirty(qt_app, clip, tmp_path):
    """Populates `clip.segments` so `Document.dirty_clips()` no longer
    reports it - a fake-but-consistent "already generated" state. The file
    has to exist: a segment whose file is missing is dirty (grill TB11)."""
    text = qt_app.document.clip_text(clip)
    config = qt_app._assemble_clip_config(clip)
    expected_hash = compute_expected_cache_hash(text, config)
    path = tmp_path / "clip.wav"
    path.write_bytes(b"RIFF")
    fake_results = [{"text": text, "path": str(path), "duration": 1.0, "seg_idx": 0}]
    clip.segments = build_segments_from_results(expected_hash, fake_results)


# -- on_generate_clicked dispatch --------------------------------------------

def test_on_generate_clicked_with_clips_present_and_dirty_calls_generate_dirty_clips(qt_app):
    _make_clip(qt_app)

    qt_app.on_generate_clicked()

    assert qt_app.engine.generate_dirty_clips.called
    assert not qt_app.engine.start_conversion.called


def test_on_generate_clicked_with_no_clips_calls_start_conversion(qt_app, monkeypatch):
    assert qt_app.document.clips == []
    calls = []
    monkeypatch.setattr(qt_app, "start_conversion", lambda: calls.append(True))

    qt_app.on_generate_clicked()

    assert calls == [True]
    assert not qt_app.engine.generate_dirty_clips.called


def test_on_generate_clicked_with_clips_present_but_none_dirty_calls_neither(qt_app, monkeypatch, tmp_path):
    clip = _make_clip(qt_app)
    _mark_not_dirty(qt_app, clip, tmp_path)
    assert qt_app.document.dirty_clips() == []

    set_ui_state_calls = []
    monkeypatch.setattr(qt_app, "set_ui_state", lambda *a, **k: set_ui_state_calls.append(a))

    qt_app.on_generate_clicked()

    assert not qt_app.engine.generate_dirty_clips.called
    assert not qt_app.engine.start_conversion.called
    assert set_ui_state_calls == []


def test_generate_blocked_while_a_job_is_already_running(qt_app):
    _make_clip(qt_app)
    qt_app.transport_dock.set_busy(True)  # simulate a running job

    qt_app.timeline_dock.generate_dirty_clips_requested()

    assert not qt_app.engine.generate_dirty_clips.called


# -- batch completion handling ------------------------------------------------

def test_partial_batch_completion_populates_segments_for_succeeded_only_and_reports_status(qt_app):
    clip_a, clip_b = _make_two_dirty_clips(qt_app)
    qt_app.timeline_dock.generate_dirty_clips_requested()
    assert qt_app.engine.generate_dirty_clips.called

    text_a = qt_app.document.clip_text(clip_a)
    config_a = qt_app._assemble_clip_config(clip_a)
    expected_hash_a = compute_expected_cache_hash(text_a, config_a)

    outcomes = [
        {
            "clip_id": clip_a.id, "success": True,
            "results": [{"path": "a.wav", "text": text_a, "duration": 1.0, "seg_idx": 0}],
            "error": "", "cancelled": False,
        },
        {
            "clip_id": clip_b.id, "success": False,
            "results": [], "error": "boom", "cancelled": False,
        },
    ]
    qt_app.engine.worker.run_coro.return_value.set_result(outcomes)

    assert len(clip_a.segments) == 1
    assert clip_a.segments[0].audio_path == "a.wav"
    assert clip_a.segments[0].cache_key == expected_hash_a
    assert clip_b.segments == []  # failed clip's segments left untouched

    assert "Generated 1 of 2 clips (1 failed)" in qt_app.transport_dock.status_text()
    assert "orange" in qt_app.transport_dock.progress_bar.styleSheet()


def test_total_batch_failure_uses_error_styling(qt_app):
    clip_a, clip_b = _make_two_dirty_clips(qt_app)
    qt_app.timeline_dock.generate_dirty_clips_requested()

    outcomes = [
        {"clip_id": clip_a.id, "success": False, "results": [], "error": "boom", "cancelled": False},
        {"clip_id": clip_b.id, "success": False, "results": [], "error": "boom", "cancelled": False},
    ]
    qt_app.engine.worker.run_coro.return_value.set_result(outcomes)

    assert clip_a.segments == []
    assert clip_b.segments == []
    assert "#ff5555" in qt_app.transport_dock.progress_bar.styleSheet()


def test_fully_successful_batch_calls_schedule_save_and_refresh_timeline_once(qt_app, monkeypatch):
    clip_a, clip_b = _make_two_dirty_clips(qt_app)
    save_calls = []
    refresh_calls = []
    monkeypatch.setattr(qt_app, "schedule_save", lambda: save_calls.append(True))
    monkeypatch.setattr(qt_app, "refresh_timeline", lambda: refresh_calls.append(True))

    qt_app.timeline_dock.generate_dirty_clips_requested()

    text_a = qt_app.document.clip_text(clip_a)
    text_b = qt_app.document.clip_text(clip_b)
    outcomes = [
        {
            "clip_id": clip_a.id, "success": True,
            "results": [{"path": "a.wav", "text": text_a, "duration": 1.0, "seg_idx": 0}],
            "error": "", "cancelled": False,
        },
        {
            "clip_id": clip_b.id, "success": True,
            "results": [{"path": "b.wav", "text": text_b, "duration": 1.0, "seg_idx": 0}],
            "error": "", "cancelled": False,
        },
    ]
    qt_app.engine.worker.run_coro.return_value.set_result(outcomes)

    assert len(save_calls) == 1
    assert len(refresh_calls) == 1
    assert "Generated 2 clip(s)." in qt_app.transport_dock.status_text()


# -- Generate stale clips in selection ----------------------------------------

def _make_three_clips(qt_app):
    text = "First clip text. Middle clip text. Last clip text."
    qt_app.document.text = text
    character = qt_app.document.characters[0]
    first = text.index(" Middle")
    last = text.index(" Last")
    clips = (qt_app.document.assign_character_to_range(0, first, character.id),
             qt_app.document.assign_character_to_range(first, last, character.id),
             qt_app.document.assign_character_to_range(last, len(text), character.id))
    return text, clips, (first, last)


def _select_text(qt_app, start, end):
    from PySide6.QtGui import QTextCursor

    # Like a drag: the caret lands first (which selects the clip under it),
    # then the selection extends from there.
    qt_app.editor.load_text(qt_app.document.text)
    cursor = qt_app.editor.textCursor()
    cursor.setPosition(start)
    qt_app.editor.setTextCursor(cursor)
    cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
    qt_app.editor.setTextCursor(cursor)


def _texts_sent(qt_app):
    group = qt_app.engine.generate_dirty_clips.call_args[0][0]
    return [text for _clip_id, text, _config in group]


def test_generate_selection_sends_only_the_selected_stale_clip(qt_app):
    text, clips, (first, last) = _make_three_clips(qt_app)
    _select_text(qt_app, first + 3, last - 3)

    assert qt_app.selected_clip_ids() == {clips[1].id}
    qt_app.generate_selection()

    assert _texts_sent(qt_app) == [qt_app.document.clip_text(clips[1])]
    assert {c.id for c in qt_app.document.dirty_clips()} == {c.id for c in clips}


def test_generate_selection_covers_every_clip_the_selection_overlaps(qt_app):
    text, clips, (first, last) = _make_three_clips(qt_app)
    _select_text(qt_app, first - 2, last + 2)

    assert qt_app.selected_clip_ids() == {c.id for c in clips}


def test_selected_clip_ids_falls_back_to_the_selected_clip_then_nothing(qt_app):
    text, clips, _ = _make_three_clips(qt_app)
    qt_app.editor.load_text(text)
    assert qt_app.selected_clip_ids() == set()

    qt_app.selection.selected_clip_id = clips[2].id
    assert qt_app.selected_clip_ids() == {clips[2].id}


def test_generate_selection_with_nothing_stale_says_so_and_dispatches_nothing(qt_app, tmp_path):
    text, clips, (first, last) = _make_three_clips(qt_app)
    _mark_not_dirty(qt_app, clips[1], tmp_path)
    _select_text(qt_app, first + 3, last - 3)

    qt_app.generate_selection()

    assert not qt_app.engine.generate_dirty_clips.called
    assert "Nothing stale in the selection" in qt_app.transport_dock.status_text()


def test_generate_menu_entry_is_on_only_when_the_selection_has_a_stale_clip(qt_app, tmp_path):
    text, clips, (first, last) = _make_three_clips(qt_app)
    dock = qt_app.transport_dock

    dock._refresh_generate_menu()
    assert not dock.generate_selection_action.isEnabled()

    _select_text(qt_app, first + 3, last - 3)
    dock._refresh_generate_menu()
    assert dock.generate_selection_action.isEnabled()

    _mark_not_dirty(qt_app, clips[1], tmp_path)
    dock._refresh_generate_menu()
    assert not dock.generate_selection_action.isEnabled()


# -- notify when a long job ends ------------------------------------------------

def _finish_batch(qt_app, started_s_ago):
    import time

    clip_a, clip_b = _make_two_dirty_clips(qt_app)
    qt_app.timeline_dock.generate_dirty_clips_requested()
    qt_app.transport_dock.busy_since = time.monotonic() - started_s_ago
    outcomes = [{"clip_id": c.id, "success": False, "results": [], "error": "x", "cancelled": False}
                for c in (clip_a, clip_b)]
    qt_app.engine.worker.run_coro.return_value.set_result(outcomes)


def _watch_notifications(qt_app, monkeypatch, active=False):
    from PySide6.QtWidgets import QApplication

    calls = []
    monkeypatch.setattr(QApplication, "alert", staticmethod(lambda *a, **k: calls.append("alert")))
    monkeypatch.setattr(QApplication, "beep", staticmethod(lambda *a, **k: calls.append("beep")))
    monkeypatch.setattr(type(qt_app), "isActiveWindow", lambda self: active)
    return calls


def test_a_long_batch_alerts_and_beeps_when_the_window_is_in_the_background(qt_app, monkeypatch):
    calls = _watch_notifications(qt_app, monkeypatch)
    _finish_batch(qt_app, started_s_ago=11)
    assert calls == ["alert", "beep"]


def test_a_long_batch_stays_quiet_when_the_window_is_active(qt_app, monkeypatch):
    calls = _watch_notifications(qt_app, monkeypatch, active=True)
    _finish_batch(qt_app, started_s_ago=11)
    assert calls == []


def test_a_short_batch_stays_quiet(qt_app, monkeypatch):
    calls = _watch_notifications(qt_app, monkeypatch)
    _finish_batch(qt_app, started_s_ago=2)
    assert calls == []


def test_the_sound_option_turns_the_beep_off_but_keeps_the_alert(qt_app, monkeypatch):
    calls = _watch_notifications(qt_app, monkeypatch)
    qt_app.notify_sound_action.setChecked(False)
    assert qt_app.settings["notify_sound"] is False
    _finish_batch(qt_app, started_s_ago=11)
    assert calls == ["alert"]


def test_a_long_export_alerts_when_it_ends_in_the_background(qt_app, monkeypatch):
    import time

    calls = _watch_notifications(qt_app, monkeypatch)
    qt_app.transport_dock.set_busy(True)
    qt_app.transport_dock.busy_since = time.monotonic() - 30
    qt_app._on_export_finished(True, "Exported")
    assert calls == ["alert", "beep"]


# -- closing while a generate runs (PG2) -----------------------------------------

def test_close_while_generating_cancels_then_closes_once_the_job_ends(qt_app, monkeypatch, qtbot):
    clip_a, clip_b = _make_two_dirty_clips(qt_app)
    qt_app.timeline_dock.generate_dirty_clips_requested()
    asked = []
    monkeypatch.setattr(type(qt_app), "_ask_cancel_generate_to_quit", lambda self: asked.append(True) or True)

    qt_app.close()

    assert asked == [True]
    assert qt_app.engine.cancel.called
    assert not qt_app._closed  # waits for the engines to stop

    text_a = qt_app.document.clip_text(clip_a)
    qt_app.engine.worker.run_coro.return_value.set_result([
        {"clip_id": clip_a.id, "success": True,
         "results": [{"path": "a.wav", "text": text_a, "duration": 1.0, "seg_idx": 0}],
         "error": "", "cancelled": False},
        {"clip_id": clip_b.id, "success": False, "results": [], "error": "", "cancelled": True},
    ])

    qtbot.waitUntil(lambda: qt_app._closed, timeout=2000)
    assert len(clip_a.segments) == 1  # the clip that finished keeps its segments
    assert clip_b.segments == []


def test_close_while_generating_keeps_working_on_the_other_answer(qt_app, monkeypatch):
    _make_two_dirty_clips(qt_app)
    qt_app.timeline_dock.generate_dirty_clips_requested()
    monkeypatch.setattr(type(qt_app), "_ask_cancel_generate_to_quit", lambda self: False)

    qt_app.close()

    assert not qt_app.engine.cancel.called
    assert not qt_app._closed
    assert qt_app.is_busy()


def test_close_with_no_generate_running_does_not_ask(qt_app, monkeypatch):
    asked = []
    monkeypatch.setattr(type(qt_app), "_ask_cancel_generate_to_quit", lambda self: asked.append(True) or True)
    qt_app.close()
    assert asked == []
    assert qt_app._closed
