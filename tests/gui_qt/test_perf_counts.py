"""Operation counts on the GUI's hot paths (Claude/PLAN_performance.md).

Counts, not timings: each test pins how much work one paint or one edit
does, so a change that brings back a whole-document pass per paint or per
keystroke fails here instead of showing up as lag on a long project.
`tests/daw/test_perf_counts.py` holds the headless half, which CI runs.
"""
import os

import pytest
from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QStyleOptionGraphicsItem

from kokoro_gui.daw import dirty
from kokoro_gui.daw.dirty import build_segments_from_results
from kokoro_gui.qt.timeline_view import RULER_HEIGHT_PX, ClipBlockItem, _RulerItem
from kokoro_gui.qt.transcript_editor import ClipHighlighter, TranscriptGutter

LINE = "A line of narration for one clip."


def _paint_ruler(ruler, exposed: QRectF) -> None:
    image = QImage(int(exposed.width()) + 1, int(RULER_HEIGHT_PX) + 1, QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    painter.translate(-exposed.left(), 0)
    option = QStyleOptionGraphicsItem()
    option.exposedRect = exposed
    try:
        ruler.paint(painter, option)
    finally:
        painter.end()


def test_ruler_paints_only_the_ticks_in_its_exposed_rect(qapp):
    ruler = _RulerItem()
    ruler.set_span(200_000.0, 20.0)  # 10,000 s at 20 px/s: a tick every 5 s
    labels = []
    ruler.label_for = lambda seconds: labels.append(seconds) or ""

    _paint_ruler(ruler, QRectF(100_000.0, 0, 1_000.0, RULER_HEIGHT_PX))

    assert 0 < len(labels) <= 1_000 / (5 * 20) + 3
    assert min(labels) <= 100_000 / 20 <= max(labels)


def test_ruler_still_labels_the_first_tick_at_zero(qapp):
    ruler = _RulerItem()
    ruler.set_span(2_000.0, 20.0)
    labels = []
    ruler.label_for = lambda seconds: labels.append(seconds) or ""

    _paint_ruler(ruler, QRectF(0, 0, 2_000.0, RULER_HEIGHT_PX))

    assert labels[0] == 0.0
    assert labels == sorted(labels) and max(labels) >= 95.0


def _project(qt_app, count: int = 12):
    """`count` one-line clips, every other one generated with a real wav in
    the project dir under its key."""
    import numpy as np
    import soundfile as sf

    doc = qt_app.document
    text = "\n".join(LINE for _ in range(count))
    qt_app.editor.load_text(text)
    doc.text = text
    character = doc.characters[0]
    generated = os.path.join(qt_app.project_dir, "audio", "generated")
    os.makedirs(generated, exist_ok=True)
    clips = []
    for i in range(count):
        start = i * (len(LINE) + 1)
        clip = doc.assign_character_to_range(start, start + len(LINE), character.id)
        clips.append(clip)
        if i % 2 == 0:
            key = doc.segment_key_fn(LINE, clip)
            path = os.path.join(generated, f"{key}_0.wav")
            sf.write(path, np.full(2400, 0.2, dtype=np.float32), 24000)
            clip.segments = build_segments_from_results(key, [{"text": LINE, "path": path, "duration": 0.1,
                                                               "cache_key": key,
                                                               "engine_version": qt_app.backend.engine_version()}])
    qt_app.editor.rehighlight()
    qt_app.refresh_timeline()
    return clips


@pytest.fixture
def counted_dirty_checks(monkeypatch):
    calls = []
    real = dirty.is_clip_dirty

    def counting(clip, *args, **kwargs):
        calls.append(clip.id)
        return real(clip, *args, **kwargs)

    monkeypatch.setattr(dirty, "is_clip_dirty", counting)
    return calls


def test_a_gutter_paint_reruns_no_dirty_check(qt_app, counted_dirty_checks, monkeypatch):
    from kokoro_gui.daw import revision

    monkeypatch.setattr(revision, "VERIFY", False)  # verify mode reruns every check on purpose
    _project(qt_app)
    gutter = qt_app.editor.findChild(TranscriptGutter)
    gutter.repaint()
    counted_dirty_checks.clear()
    for _ in range(3):
        gutter.repaint()
    assert counted_dirty_checks == []


def test_a_keystroke_leaves_the_timeline_to_the_debounce(qt_app, monkeypatch):
    _project(qt_app)
    refreshes = []
    real = qt_app.timeline_dock.refresh
    monkeypatch.setattr(qt_app.timeline_dock, "refresh", lambda: refreshes.append(1) or real())

    cursor = qt_app.editor.textCursor()
    cursor.setPosition(3)
    cursor.insertText("x")
    assert refreshes == []
    qt_app.flush_updates()
    assert refreshes == [1]


def test_a_timeline_refresh_after_a_one_clip_edit_reloads_one_waveform(qt_app, monkeypatch):
    clips = _project(qt_app)
    sources = []
    real = ClipBlockItem.set_waveform_source
    monkeypatch.setattr(ClipBlockItem, "set_waveform_source",
                        lambda self, *a, **k: sources.append(self.clip_id) or real(self, *a, **k))

    qt_app.refresh_timeline()
    assert sources == []  # nothing changed

    target = clips[4]
    target.fx_override = {"reverb_enabled": True, "reverb_room_size": 0.5}  # a new post key
    qt_app.refresh_timeline()
    assert sources == [target.id]


def test_rehighlight_after_a_one_clip_change_repaints_that_clips_line(qt_app, monkeypatch):
    clips = _project(qt_app)
    painted = []
    real = ClipHighlighter.rehighlightBlock
    monkeypatch.setattr(ClipHighlighter, "rehighlightBlock",
                        lambda self, block: painted.append(block.blockNumber()) or real(self, block))

    qt_app.editor.rehighlight()
    assert painted == []  # nothing changed

    qt_app.document.characters[0].highlight_color = "#123456"  # every clip's tint
    qt_app.editor.rehighlight()
    # Qt highlights the next block itself while block states keep changing,
    # so the pass asks for block 0 and the rest follow.
    highlighter = qt_app.editor._highlighter
    block = qt_app.editor.document().begin()
    while block.isValid():
        start = block.position()
        spec = highlighter.block_spec(start, start + block.length() - 1)
        assert block.userState() == highlighter.spec_state(spec)
        assert spec[0][0][3] == "#123456"
        block = block.next()
    painted.clear()

    assert clips[2].id not in qt_app.document.dirty_ids()
    clips[2].segments = []  # one generated clip goes stale
    qt_app.editor.rehighlight()
    assert painted == [2]


def test_a_file_deleted_behind_the_apps_back_counts_after_force_refresh(qt_app):
    clips = _project(qt_app)
    clean = clips[0]
    assert clean.id not in qt_app.document.dirty_ids()

    os.remove(clean.segments[0].audio_path)
    assert clean.id not in qt_app.document.dirty_ids()  # remembered until the files are re-checked

    qt_app.force_refresh_action.trigger()
    assert clean.id in qt_app.document.dirty_ids()


def test_force_refresh_is_in_the_options_menu(qt_app):
    assert qt_app.force_refresh_action in qt_app.options_menu.actions()
