"""Tests for per-clip Generate: the right-click "Generate" action on a
timeline clip block (kokoro_gui/qt/timeline_view.py's context menu +
kokoro_gui/qt/docks/timeline_dock.py's dispatch/completion handling)."""
from kokoro_gui.daw.dirty import compute_expected_cache_hash


def _make_clip(qt_app, start=0, end=5, text="hello world"):
    qt_app.document.text = text
    character = qt_app.document.characters[0]
    return qt_app.document.assign_character_to_range(start, end, character.id)


def _clip_block_for(qt_app, clip_id):
    from kokoro_gui.qt.timeline_view import ClipBlockItem
    for item in qt_app.timeline_dock.timeline_view._scene.items():
        if isinstance(item, ClipBlockItem) and item.clip_id == clip_id:
            return item
    return None


def test_context_menu_shows_generate_action_over_a_clip_block(qt_app):
    clip = _make_clip(qt_app)
    qt_app.refresh_timeline()
    block = _clip_block_for(qt_app, clip.id)
    pos = block.mapToScene(0, 0)
    view_pos = qt_app.timeline_dock.timeline_view.mapFromScene(pos)

    menu = qt_app.timeline_dock.timeline_view._build_context_menu(view_pos)

    assert menu is not None
    assert [a.text() for a in menu.actions()] == ["Generate"]


def test_context_menu_empty_over_lane_background(qt_app):
    _make_clip(qt_app, start=0, end=5)
    qt_app.refresh_timeline()

    # Far outside any clip block's geometry.
    menu = qt_app.timeline_dock.timeline_view._build_context_menu(qt_app.timeline_dock.timeline_view.mapFromScene(9999, 9999))

    assert menu is None


def test_generate_clip_requested_signal_carries_clip_id(qt_app):
    clip = _make_clip(qt_app)
    qt_app.refresh_timeline()
    block = _clip_block_for(qt_app, clip.id)
    pos = block.mapToScene(0, 0)
    view_pos = qt_app.timeline_dock.timeline_view.mapFromScene(pos)

    menu = qt_app.timeline_dock.timeline_view._build_context_menu(view_pos)
    received = []
    qt_app.timeline_dock.timeline_view.generateClipRequested.connect(lambda cid: received.append(cid))
    menu.actions()[0].trigger()

    assert received == [clip.id]


def test_on_generate_clip_requested_calls_engine_with_assembled_config(qt_app):
    clip = _make_clip(qt_app)
    character = qt_app.document.characters[0]
    character.preset_data["voice"] = "af_sarah"

    qt_app.timeline_dock.on_generate_clip_requested(clip.id)

    assert qt_app.engine.generate_clip_audio.called
    (chunk_data,), _kwargs = qt_app.engine.generate_clip_audio.call_args
    index, text, config = chunk_data
    assert text == qt_app.document.clip_text(clip)
    assert config["voice"] == "af_sarah"


def test_successful_generation_populates_segments_with_order_index_and_shared_cache_key(qt_app):
    clip = _make_clip(qt_app)
    qt_app.timeline_dock.on_generate_clip_requested(clip.id)

    results = [
        {"path": "a.wav", "text": "hello", "duration": 1.0, "seg_idx": 0},
        {"path": "b.wav", "text": "world", "duration": 1.0, "seg_idx": 0},
    ]
    qt_app.engine.worker.run_coro.return_value.set_result(results)

    assert [s.order_index for s in clip.segments] == [0, 1]
    assert clip.segments[0].cache_key == clip.segments[1].cache_key
    assert clip.segments[0].audio_path == "a.wav"
    assert clip.segments[1].audio_path == "b.wav"


def test_successful_generation_calls_refresh_timeline_and_schedule_save(qt_app, monkeypatch):
    clip = _make_clip(qt_app)
    refresh_calls = []
    save_calls = []
    monkeypatch.setattr(qt_app, "refresh_timeline", lambda: refresh_calls.append(True))
    monkeypatch.setattr(qt_app, "schedule_save", lambda: save_calls.append(True))

    qt_app.timeline_dock.on_generate_clip_requested(clip.id)
    qt_app.engine.worker.run_coro.return_value.set_result(
        [{"path": "a.wav", "text": "hello world", "duration": 1.0, "seg_idx": 0}]
    )

    assert refresh_calls
    assert save_calls


def test_failed_generation_with_exception_leaves_segments_unchanged(qt_app):
    clip = _make_clip(qt_app)
    qt_app.timeline_dock.on_generate_clip_requested(clip.id)

    qt_app.engine.worker.run_coro.return_value.set_exception(RuntimeError("boom"))

    assert clip.segments == []
    assert "Clip generation failed" in qt_app.transport_dock.status_text()


def test_failed_generation_with_empty_result_leaves_segments_unchanged(qt_app):
    clip = _make_clip(qt_app)
    qt_app.timeline_dock.on_generate_clip_requested(clip.id)

    qt_app.engine.worker.run_coro.return_value.set_result([])

    assert clip.segments == []
    assert "Clip generation failed" in qt_app.transport_dock.status_text()


def test_generate_blocked_while_a_job_is_already_running(qt_app):
    clip = _make_clip(qt_app)
    qt_app.transport_dock.set_busy(True)  # simulate a running whole-document job

    qt_app.timeline_dock.on_generate_clip_requested(clip.id)

    assert not qt_app.engine.generate_clip_audio.called


# --- takes (Claude/PLAN_tbaw_bundle.md 2.3, grill TB8) -----------------------

def test_generate_clip_on_a_clean_clip_sends_regenerate_and_a_dirty_one_does_not(qt_app, tmp_path):
    from kokoro_gui.daw.dirty import build_segments_from_results

    clip = _make_clip(qt_app)
    qt_app.generate_clip(clip.id)  # dirty: never generated
    (chunk_data,), _ = qt_app.engine.generate_clip_audio.call_args
    assert "regenerate" not in chunk_data[2]
    qt_app.transport_dock.set_busy(False)

    path = tmp_path / "seg.wav"
    path.write_bytes(b"RIFF")
    text = qt_app.document.clip_text(clip)
    key = qt_app.document.segment_key_fn(text, clip)
    clip.segments = build_segments_from_results(key, [{"text": text, "path": str(path), "duration": 1.0}])
    assert qt_app.document.dirty_clips() == []

    qt_app.generate_clip(clip.id)  # clean: the gutter button means regenerate
    (chunk_data,), _ = qt_app.engine.generate_clip_audio.call_args
    assert chunk_data[2]["regenerate"] is True


def test_results_stamp_the_take_key_and_version_the_engine_reports(qt_app):
    clip = _make_clip(qt_app)
    qt_app.timeline_dock.on_generate_clip_requested(clip.id)
    qt_app.engine.worker.run_coro.return_value.set_result([
        {"path": "a.wav", "text": "hello world", "duration": 1.0, "seg_idx": 0,
         "cache_key": "k-take-2", "take": 2, "engine_version": "9.9"},
    ])
    assert clip.overrides["take"] == 2
    assert clip.segments[0].cache_key == "k-take-2"
    assert clip.segments[0].engine_version == "9.9"
    assert qt_app._assemble_generation_config(clip)["take"] == 2
