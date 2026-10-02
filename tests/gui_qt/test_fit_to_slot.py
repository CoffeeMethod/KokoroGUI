"""Fit to slot (phase 5, D4): TimelineDock's per-clip loop over
`overrides["speed"]` for an engine with a speed control, the
`overrides["time_stretch"]` path for one without, "Fit all over slot", and
the transcript's reading-rate underline. The StubEngine's generate is
replaced by a fake whose output length scales with 1/speed; its futures
complete at once, so a whole fit runs inside the call that starts it. The
`qt_app` fixture runs in a tmp dir, where the fake writes its files."""
import concurrent.futures
import os

import numpy as np
import pytest
import soundfile as sf

from kokoro_gui.daw import fit
from kokoro_gui.engines.base import EngineCapabilities

RATE = 8000


def _clip(qt_app, text, target, start=None):
    """A clip over `text` (appended to the document) with a target."""
    document = qt_app.document
    offset = len(document.text) if start is None else start
    if start is None:
        document.text = document.text + ("\n\n" if document.text else "") + text
        offset = len(document.text) - len(text)
    clip = document.assign_character_to_range(offset, offset + len(text), document.characters[0].id)
    clip.overrides["target_duration_s"] = target
    return clip


def _fake_generate(qt_app, natural_for, fail_on=None):
    """Every generate returns one segment `natural_for(text) / speed ** 0.8
    + 0.2` seconds long (a pause that doesn't speed up, so one pass isn't
    exact). Returns the list of speeds generated at."""
    engine = qt_app.engine
    speeds = []
    engine.generate_clip_audio.side_effect = lambda chunk: chunk

    def run_coro(chunk):
        _index, text, config = chunk
        speed = float(config["speed"])
        speeds.append(speed)
        future = concurrent.futures.Future()
        if fail_on is not None and len(speeds) == fail_on:
            future.set_exception(RuntimeError("boom"))
            return future
        duration = (natural_for(text) - 0.2) / speed ** 0.8 + 0.2
        # A real file (a missing one is dirty) of the reported length.
        path = os.path.abspath(f"take_{len(speeds)}.wav")
        sf.write(path, np.zeros(int(round(duration * RATE)), dtype=np.float32), RATE)
        future.set_result([{"path": path, "text": text, "duration": duration,
                            "onset_s": 0.0, "tail_s": 0.0, "seg_idx": 0}])
        return future

    engine.worker.run_coro = run_coro
    return speeds


def _ratio(qt_app, clip):
    return qt_app.clip_duration_s(clip) / fit.target_duration_s(clip)


def test_speed_fit_converges_within_three_passes_and_the_tolerance(qt_app):
    clip = _clip(qt_app, "A line that runs long.", 2.4)
    speeds = _fake_generate(qt_app, lambda _text: 3.0)

    assert qt_app.timeline_dock.on_fit_to_slot_requested(clip.id)

    # The first generate is at the clip's own speed, then the speed passes.
    assert speeds[0] == 1.0
    assert 1 <= len(speeds) - 1 <= fit.MAX_FIT_PASSES
    assert abs(_ratio(qt_app, clip) - 1.0) <= fit.FIT_TOLERANCE
    assert clip.overrides["speed"] == speeds[-1]
    # A speed change is a new key: no pass asked for a take bump.
    assert "take" not in clip.overrides
    assert not qt_app.timeline_dock._fits and not qt_app.is_busy()
    assert "Fit to slot" in qt_app.transport_dock.status_text()


def test_a_fit_is_one_undo_step(qt_app):
    clip = _clip(qt_app, "A line that runs long.", 2.4)
    _fake_generate(qt_app, lambda _text: 3.0)
    stack = qt_app.document.undo_stack
    steps = len(stack._undo)

    qt_app.timeline_dock.on_fit_to_slot_requested(clip.id)

    assert len(stack._undo) == steps + 1
    stack.undo()
    assert "speed" not in clip.overrides
    stack.redo()
    assert clip.overrides["speed"] > 1.0


def test_speed_is_clamped_and_a_line_that_cannot_fit_needs_a_rewrite(qt_app):
    long = _clip(qt_app, "Far too much to say here.", 1.0)
    short = _clip(qt_app, "Hi.", 6.0)
    speeds = _fake_generate(qt_app, lambda text: 3.0)

    qt_app.timeline_dock.on_fit_to_slot_requested(long.id)
    assert long.overrides["speed"] == fit.SPEED_MAX
    # Clamped on the first pass; a second at the same speed would be the same.
    assert speeds == [1.0, fit.SPEED_MAX]
    assert long.status == "needs_rewrite"

    del speeds[:]
    qt_app.timeline_dock.on_fit_to_slot_requested(short.id)
    assert short.overrides["speed"] == fit.SPEED_MIN
    assert short.status == "generated"


def test_a_clip_that_already_fits_is_left_alone(qt_app):
    clip = _clip(qt_app, "Just right.", 3.0)
    speeds = _fake_generate(qt_app, lambda _text: 3.0)

    qt_app.timeline_dock.on_fit_to_slot_requested(clip.id)
    assert speeds == [1.0]
    del speeds[:]
    steps = len(qt_app.document.undo_stack._undo)
    qt_app.timeline_dock.on_fit_to_slot_requested(clip.id)  # clean now: no generate at all
    assert speeds == []
    assert "speed" not in clip.overrides
    assert len(qt_app.document.undo_stack._undo) == steps


def test_a_failed_pass_puts_back_the_speed_that_rendered(qt_app):
    clip = _clip(qt_app, "A line that runs long.", 2.4)
    _fake_generate(qt_app, lambda _text: 3.0, fail_on=2)
    stack = qt_app.document.undo_stack
    steps = len(stack._undo)

    qt_app.timeline_dock.on_fit_to_slot_requested(clip.id)

    assert "speed" not in clip.overrides
    assert len(stack._undo) == steps
    assert not qt_app.timeline_dock._fits and not qt_app.is_busy()


def _no_speed_control(qt_app, monkeypatch, clip):
    backend = qt_app.backend_for(clip)
    monkeypatch.setattr(type(backend), "capabilities", EngineCapabilities(supports_speed=False))


def test_without_a_speed_control_the_fit_time_stretches(qt_app, monkeypatch):
    clip = _clip(qt_app, "A line that runs long.", 2.7)
    _no_speed_control(qt_app, monkeypatch, clip)
    speeds = _fake_generate(qt_app, lambda _text: 3.0)

    qt_app.timeline_dock.on_fit_to_slot_requested(clip.id)

    assert speeds == [1.0]  # generated once, never at another speed
    assert "speed" not in clip.overrides
    assert clip.overrides["time_stretch"] == pytest.approx(3.0 / 2.7, abs=1e-4)
    # The factor reaches the post config, so the rendered length is the target.
    assert qt_app.post_config_for_clip(clip)["time_stretch"] == clip.overrides["time_stretch"]
    assert qt_app.clip_duration_s(clip) == pytest.approx(2.7, abs=1e-3)
    assert clip.status == "generated"
    # A post key: the clip stays clean.
    assert clip not in qt_app.document.dirty_clips()


def test_a_stretch_past_the_cap_is_capped_and_needs_a_rewrite(qt_app, monkeypatch):
    clip = _clip(qt_app, "A line that runs long.", 2.0)
    _no_speed_control(qt_app, monkeypatch, clip)
    _fake_generate(qt_app, lambda _text: 3.0)
    stack = qt_app.document.undo_stack
    steps = len(stack._undo)

    qt_app.timeline_dock.on_fit_to_slot_requested(clip.id)

    assert clip.overrides["time_stretch"] == fit.STRETCH_MAX
    assert clip.status == "needs_rewrite"
    assert len(stack._undo) == steps + 1
    stack.undo()
    assert "time_stretch" not in clip.overrides and clip.status == "generated"


def test_fit_all_over_slot_fits_only_the_clips_over_their_target(qt_app):
    over_a = _clip(qt_app, "First long line here.", 2.3)
    over_b = _clip(qt_app, "Second long line.", 2.5)
    fine = _clip(qt_app, "Fits.", 3.0)
    no_target = _clip(qt_app, "No target at all.", 1.0)
    del no_target.overrides["target_duration_s"]
    speeds = _fake_generate(qt_app, lambda _text: 3.0)
    for clip in (over_a, over_b, fine, no_target):
        qt_app.timeline_dock.on_generate_clip_requested(clip.id)
    assert qt_app.timeline_dock.clips_over_slot() == [over_a, over_b]
    del speeds[:]

    assert qt_app.timeline_dock.fit_all_over_slot() == 2

    for clip in (over_a, over_b):
        assert abs(_ratio(qt_app, clip) - 1.0) <= fit.FIT_TOLERANCE
    assert "speed" not in fine.overrides and "speed" not in no_target.overrides
    assert qt_app.timeline_dock.clips_over_slot() == []
    assert "Fitted 2 clips" in qt_app.transport_dock.status_text()


def test_the_header_button_shows_with_a_target_and_runs_fit_all(qt_app, monkeypatch):
    dock = qt_app.timeline_dock
    dock.refresh()
    assert dock.fit_all_button.isHidden()
    _clip(qt_app, "A line.", 2.0)
    dock.refresh()
    assert not dock.fit_all_button.isHidden()

    calls = []
    monkeypatch.setattr(dock, "fit_all_over_slot", lambda: calls.append(True))
    dock.fit_all_button.click()
    assert calls == [True]


def test_audio8_has_no_speed_control_and_kokoro_does():
    from kokoro_gui.engines.audio8_tts import Audio8BackendAdapter
    from kokoro_gui.engines.kokoro import KokoroBackendAdapter

    assert Audio8BackendAdapter.capabilities.supports_speed is False
    assert KokoroBackendAdapter.capabilities.supports_speed is True


# -- the reading-rate underline ----------------------------------------------------


def test_a_dirty_clip_whose_text_reads_past_its_target_is_wave_underlined(qt_app):
    from PySide6.QtGui import QTextCharFormat

    text = "This line has far too many words for such a short slot."
    clip = _clip(qt_app, text, 1.0)
    qt_app.editor.load_text(qt_app.document.text)
    highlighter = qt_app.editor._highlighter

    assert highlighter.rate_levels() == {clip.id: "far_over"}
    block = qt_app.editor.document().firstBlock()
    formats = block.layout().formats()
    assert any(f.format.underlineStyle() == QTextCharFormat.UnderlineStyle.WaveUnderline for f in formats)

    # Room enough at the fallback pace: no warning.
    clip.overrides["target_duration_s"] = 60.0
    qt_app.editor.rehighlight()
    assert highlighter.rate_levels() == {}


def test_reading_rate_learns_the_pace_from_generated_clips(qt_app):
    from kokoro_gui.daw.models import Segment

    generated = _clip(qt_app, "x" * 40, 99.0)
    generated.segments = [Segment(text="x" * 40, audio_path="a.wav", duration=2.0)]
    rates = fit.speaking_rates(qt_app.document)
    assert rates[generated.character_id] == pytest.approx(20.0)
    assert rates[None] == pytest.approx(20.0)

    typed = _clip(qt_app, "y" * 30, 1.0)
    assert fit.reading_rate_ratio(qt_app.document, typed, rates) == pytest.approx(1.5)


def _stack_depth() -> int:
    import sys

    depth, frame = 0, sys._getframe()
    while frame is not None:
        depth, frame = depth + 1, frame.f_back
    return depth


def test_fit_all_of_many_stretch_fits_does_not_recurse(qt_app, monkeypatch):
    """An Audio8 fit ends inside the call that starts it; a few hundred of
    them in one "Fit all over slot" run one after another, not nested."""
    from types import SimpleNamespace

    from kokoro_gui.daw.models import Clip, Run

    document = qt_app.document
    character = document.characters[0]
    runs = []
    for i in range(300):
        clip = Clip(character_id=character.id, overrides={"target_duration_s": 1.9},
                    timeline_timestamp=float(i * 3), pinned=True)
        document.clips.append(clip)
        if runs:
            runs.append(Run(text="\n\n"))
        runs.append(Run(text=f"line {i}", clip_id=clip.id, kind="generated"))
    document.runs = runs
    dock = qt_app.timeline_dock
    # Every clip is clean and renders 2 s long, and there is no speed control.
    monkeypatch.setattr(type(document), "dirty_clips", lambda self: [])
    monkeypatch.setattr(qt_app, "clip_duration_s", lambda clip, project=None: 2.0)
    no_speed = SimpleNamespace(capabilities=EngineCapabilities(supports_speed=False))
    monkeypatch.setattr(qt_app, "backend_for", lambda clip, project=None: no_speed)
    monkeypatch.setattr(qt_app, "refresh_timeline", lambda *a, **k: None)
    monkeypatch.setattr(dock, "_ripple_shifts", lambda *a, **k: [])
    depths = []
    finish = dock._finish_fit

    def _finish(job, failed=False):
        depths.append(_stack_depth())
        finish(job, failed)

    monkeypatch.setattr(dock, "_finish_fit", _finish)

    assert dock.fit_all_over_slot() == 300

    assert len(depths) == 300
    assert max(depths) - min(depths) < 5
    assert {c.overrides.get("time_stretch") for c in document.clips} == {round(2.0 / 1.9, 4)}
    assert "Fitted 300 clips" in qt_app.transport_dock.status_text()
