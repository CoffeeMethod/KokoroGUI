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
