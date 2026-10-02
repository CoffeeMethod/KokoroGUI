"""Resume where you left off: `session.json["view"]` (kokoro_gui/qt/resume_view.py).

The view is runtime state in the project dir, never bundled. It comes back
when the dir survives, which is a relaunch on the last project (TB13 deletes
every other clean dir on Open and New)."""
import json
import os
import zipfile

import numpy as np
import pytest
import soundfile as sf

import kokoro_gui.qt.app as qt_app_module
from kokoro_gui.daw.dirty import build_segments_from_results
from kokoro_gui.qt import project as project_io, resume_view
from kokoro_gui.qt.timeline_view import MAX_PIXELS_PER_SECOND, MIN_PIXELS_PER_SECOND


def _generated_clip(qt_app, text="hello world", seconds=1.0):
    """A clean clip whose one segment is a real wav inside the project dir."""
    qt_app.document.text = text
    character = qt_app.document.characters[0]
    clip = qt_app.document.assign_character_to_range(0, len(text), character.id)
    key = qt_app.document.segment_key_fn(text, clip)
    generated = os.path.join(qt_app.project_dir, "audio", "generated")
    os.makedirs(generated, exist_ok=True)
    path = os.path.join(generated, f"{key}_0.wav")
    sf.write(path, np.full(int(24000 * seconds), 0.2, dtype=np.float32), 24000)
    clip.segments = build_segments_from_results(key, [{"text": text, "path": path, "duration": seconds,
                                                       "cache_key": key,
                                                       "engine_version": qt_app.backend.engine_version()}])
    return clip


def _saved_project(qt_app, tmp_path, text="hello world"):
    clip = _generated_clip(qt_app, text)
    qt_app.save_project_as(str(tmp_path / "story"))
    qt_app.wait_for_project_io()
    assert not qt_app.is_project_dirty()
    return clip, qt_app.project_path


def _relaunch(qt_app, qtbot):
    """Closes the window (keeping the last project's dir, TB13) and starts
    another against the same config: the launch resumes that project. The
    new window is shown, since a scroll bar has no range until it is laid
    out."""
    qt_app.close()
    second = qt_app_module.QtTTSApp()
    second.wait_for_project_io()
    second.show()
    qtbot.wait(2 * resume_view.SCROLL_RETRY_MS)  # the scroll restore, and its retry
    return second


# -- clean_view ---------------------------------------------------------------------------


def _has(*ids):
    return lambda clip_id: clip_id in ids


def test_clean_view_keeps_valid_values():
    raw = {"playhead_s": 12.5, "zoom": 80, "timeline_scroll_x": 300, "transcript_scroll_y": 40,
           "selected_clip_id": "c1"}
    assert resume_view.clean_view(raw, _has("c1")) == {
        "playhead_s": 12.5, "zoom": 80.0, "timeline_scroll_x": 300, "transcript_scroll_y": 40,
        "selected_clip_id": "c1"}
    assert resume_view.clean_view({"zoom": MIN_PIXELS_PER_SECOND}, _has())["zoom"] == MIN_PIXELS_PER_SECOND
    assert resume_view.clean_view({"zoom": MAX_PIXELS_PER_SECOND}, _has())["zoom"] == MAX_PIXELS_PER_SECOND


@pytest.mark.parametrize("raw", ["garbage", 7, None, [], [1, 2], True, 3.5])
def test_clean_view_gives_nothing_for_a_non_dict(raw):
    assert resume_view.clean_view(raw, _has("c1")) == {}


@pytest.mark.parametrize("key, bad", [
    ("playhead_s", -1), ("playhead_s", float("nan")), ("playhead_s", float("inf")), ("playhead_s", "3"),
    ("playhead_s", True), ("playhead_s", 1e12), ("playhead_s", None), ("playhead_s", [1]),
    ("zoom", 0), ("zoom", -5), ("zoom", MIN_PIXELS_PER_SECOND / 2), ("zoom", MAX_PIXELS_PER_SECOND + 1),
    ("zoom", "50"), ("zoom", float("nan")), ("zoom", False),
    ("timeline_scroll_x", -1), ("timeline_scroll_x", 10 ** 12), ("timeline_scroll_x", "7"),
    ("timeline_scroll_x", float("nan")), ("timeline_scroll_x", True),
    ("transcript_scroll_y", -1), ("transcript_scroll_y", 10 ** 12), ("transcript_scroll_y", {}),
    ("selected_clip_id", 5), ("selected_clip_id", ""), ("selected_clip_id", None),
    ("selected_clip_id", "gone"), ("selected_clip_id", "c" * 500), ("selected_clip_id", ["c1"]),
])
def test_clean_view_drops_a_bad_value_and_keeps_the_rest(key, bad):
    raw = {"playhead_s": 1.0, "zoom": 50.0, "timeline_scroll_x": 10, "transcript_scroll_y": 5,
           "selected_clip_id": "c1", key: bad}
    view = resume_view.clean_view(raw, _has("c1"))
    assert key not in view
    assert set(view) == {"playhead_s", "zoom", "timeline_scroll_x", "transcript_scroll_y",
                         "selected_clip_id"} - {key}


# -- session.json keys survive a reopen of the same file only ---------------------------


def test_reopening_the_same_file_keeps_the_runtime_state_keys(tmp_path, isolated_dirs):
    from kokoro_gui.daw.models import Character, Document

    project_dir, project_id = project_io.create_project_dir()
    doc = Document.from_plain_text("hello", characters=[Character.from_preset_dict("A", {})])
    path = str(tmp_path / "proj.tbaw")
    project_io.save_project(doc, path, {}, project_dir, project_id)
    session = project_io.read_session(project_dir)
    session.update({"loop_s": [1.0, 2.0], "monitor": "original", "view": {"zoom": 90.0}})
    project_io.write_session(project_dir, session)

    info = project_io.inspect_bundle(path)
    project_io.extract_small(info, project_dir)
    project_io.finish_open(info, project_dir)
    kept = project_io.read_session(project_dir)
    assert kept["loop_s"] == [1.0, 2.0] and kept["monitor"] == "original" and kept["view"] == {"zoom": 90.0}
    assert kept["dirty"] is False

    # Another dir (a bundle opened from somewhere else) starts with none.
    other = str(tmp_path / "other")
    project_io.extract_small(info, other)
    project_io.finish_open(info, other)
    assert not set(project_io.RUNTIME_STATE_KEYS) & set(project_io.read_session(other))


def test_the_view_is_never_bundled(qt_app, tmp_path):
    clip, path = _saved_project(qt_app, tmp_path)
    qt_app.selection.select_clip(clip.id)
    qt_app._remember_view()
    assert "view" in project_io.read_session(qt_app.project_dir)
    qt_app.save_project()
    qt_app.wait_for_project_io()
    with zipfile.ZipFile(path) as zf:
        assert "session.json" not in zf.namelist()
        assert b"timeline_scroll_x" not in b"".join(zf.read(n) for n in zf.namelist() if n.endswith(".json"))


# -- remembering --------------------------------------------------------------------------


def test_a_zoom_or_selection_change_writes_the_view_after_the_debounce(qt_app, tmp_path):
    clip, _path = _saved_project(qt_app, tmp_path)
    view = qt_app.timeline_dock.timeline_view
    assert not qt_app._view_timer.isActive()

    view.set_zoom(77.0)
    qt_app.selection.select_clip(clip.id)
    assert qt_app._view_timer.isActive()
    assert qt_app._view_timer.interval() == qt_app_module.VIEW_REMEMBER_DEBOUNCE_MS == 2000
    # Nothing is written per change; the timer's timeout does it, once.
    assert "view" not in project_io.read_session(qt_app.project_dir)

    qt_app._view_timer.timeout.emit()

    assert not qt_app._view_timer.isActive()
    saved = project_io.read_session(qt_app.project_dir)["view"]
    assert saved["zoom"] == 77.0 and saved["selected_clip_id"] == clip.id
    assert set(saved) == {"playhead_s", "zoom", "timeline_scroll_x", "transcript_scroll_y", "selected_clip_id"}


def test_a_subproject_level_does_not_write_the_view(qt_app, tmp_path, monkeypatch):
    _saved_project(qt_app, tmp_path)
    project_dir = qt_app.project_dir
    qt_app._remember_view()
    before = project_io.read_session(project_dir)
    qt_app.timeline_dock.timeline_view.set_zoom(33.0)

    monkeypatch.setattr(qt_app, "level", object())
    qt_app._remember_view()
    assert project_io.read_session(project_dir) == before

    monkeypatch.undo()
    monkeypatch.setattr(qt_app, "focus", object())
    qt_app._remember_view()
    assert project_io.read_session(project_dir) == before


def test_closing_the_current_project_remembers_the_view_first(qt_app, tmp_path):
    clip, _path = _saved_project(qt_app, tmp_path)
    qt_app.timeline_dock.timeline_view.set_zoom(150.0)
    qt_app.selection.select_clip(clip.id)
    project_dir = qt_app.project_dir
    seen = []

    def _then():
        seen.append(project_io.read_session(project_dir)["view"])

    # `_close_current_project` writes before it tears the project down.
    qt_app._close_current_project(_then)
    assert seen and seen[0]["zoom"] == 150.0 and seen[0]["selected_clip_id"] == clip.id


# -- the round trip -----------------------------------------------------------------------


def test_relaunching_restores_zoom_selection_scroll_and_playhead(qt_app, qtbot, tmp_path):
    clip, path = _saved_project(qt_app, tmp_path, text="a line of the story\n" * 200)
    qt_app.editor.rebind_document()  # the editor shows the text set on the document
    qt_app.show()
    qtbot.wait(50)
    qt_app._rebuild_transport_schedule()
    view = qt_app.timeline_dock.timeline_view
    view.set_zoom(300.0)
    qt_app.selection.select_clip(clip.id)  # scrolls the block into view, so scroll after it
    view.horizontalScrollBar().setValue(450)
    assert view.horizontalScrollBar().value() == 450
    transcript = qt_app.editor.verticalScrollBar()
    transcript.setValue(transcript.maximum() // 2)
    transcript_y = transcript.value()
    assert transcript_y > 0
    qt_app.transport.seek(0.5)
    assert qt_app.transport.position() == pytest.approx(0.5)

    second = _relaunch(qt_app, qtbot)
    try:
        assert second.project_path == path
        assert second.timeline_dock.timeline_view.zoom == 300.0
        assert second.selection.selected_clip_id == clip.id
        assert second.timeline_dock.timeline_view.horizontalScrollBar().value() == 450
        assert second.editor.verticalScrollBar().value() == transcript_y
        # The playhead waits for the transport's schedule, then lands.
        qtbot.waitUntil(lambda: second._resume_playhead_s is None, timeout=5000)
        assert second.transport.position() == pytest.approx(0.5)
    finally:
        second.close()


def test_the_close_keeps_the_playhead_the_stopped_transport_lost(qt_app, tmp_path):
    _saved_project(qt_app, tmp_path)
    qt_app._rebuild_transport_schedule()
    qt_app.transport.seek(0.75)
    project_dir = qt_app.project_dir

    # The window writes first, then stops the transport, which rewinds it.
    qt_app.close()
    assert project_io.read_session(project_dir)["view"]["playhead_s"] == pytest.approx(0.75)


def test_a_second_write_after_the_stop_can_keep_the_stored_playhead(qt_app, tmp_path):
    _saved_project(qt_app, tmp_path)
    qt_app._rebuild_transport_schedule()
    qt_app.transport.seek(0.75)
    qt_app._remember_view()
    qt_app.transport.stop()

    qt_app._remember_view(keep_playhead=True)
    assert project_io.read_session(qt_app.project_dir)["view"]["playhead_s"] == pytest.approx(0.75)
    qt_app._remember_view()
    assert project_io.read_session(qt_app.project_dir)["view"]["playhead_s"] == 0.0


def test_a_clip_that_no_longer_exists_is_ignored_on_resume(qt_app, tmp_path):
    clip, _path = _saved_project(qt_app, tmp_path)
    qt_app.timeline_dock.timeline_view.set_zoom(200.0)
    qt_app.selection.select_clip(clip.id)
    project_dir = qt_app.project_dir
    qt_app.close()
    session = project_io.read_session(project_dir)
    session["view"]["selected_clip_id"] = "not-a-clip"
    project_io.write_session(project_dir, session)

    second = qt_app_module.QtTTSApp()
    try:
        second.wait_for_project_io()
        assert second.selection.selected_clip_id is None
        assert second.timeline_dock.timeline_view.zoom == 200.0
    finally:
        second.close()


@pytest.mark.parametrize("garbage", [
    "garbage", 5, None, [], {"zoom": "wide", "playhead_s": [], "selected_clip_id": 9, "timeline_scroll_x": {}},
    {"zoom": 1e308, "timeline_scroll_x": -5, "transcript_scroll_y": 1e99, "playhead_s": float("nan")},
])
def test_a_garbage_view_opens_without_error(qt_app, tmp_path, garbage):
    clip, path = _saved_project(qt_app, tmp_path)
    project_dir = qt_app.project_dir
    qt_app.close()
    session = project_io.read_session(project_dir)
    session["view"] = garbage
    # json.dump writes NaN as a bare NaN, which read_session accepts back.
    project_io.write_session(project_dir, session)
    assert "view" in json.loads(open(os.path.join(project_dir, "session.json"), encoding="utf-8").read())

    second = qt_app_module.QtTTSApp()
    try:
        second.wait_for_project_io()
        assert second.project_path == path
        assert second.document.get_clip(clip.id) is not None
        zoom = second.timeline_dock.timeline_view.zoom
        assert MIN_PIXELS_PER_SECOND <= zoom <= MAX_PIXELS_PER_SECOND
        assert second.selection.selected_clip_id is None
        assert second._resume_playhead_s is None
    finally:
        second.close()
