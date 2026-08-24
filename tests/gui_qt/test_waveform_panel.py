"""Tests for kokoro_gui/qt/waveform_view.py's WaveformPanel - Play/Stop +
QTimer-driven playhead. Monkeypatches playback.play/stop/AVAILABLE rather
than touching real audio (Workstream 3 spike)."""
import numpy as np
import soundfile as sf

import playback
from kokoro_gui.qt.waveform_view import WaveformPanel


def _write_tone_wav(path, sample_rate=8000, seconds=0.5, freq=440):
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    data = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), data, sample_rate)


def _panel_with_audio(qtbot, tmp_path):
    path = tmp_path / "tone.wav"
    _write_tone_wav(path)
    panel = WaveformPanel()
    qtbot.addWidget(panel)
    panel.show()
    panel.resize(200, 120)
    qtbot.wait(10)
    panel.load_audio(str(path))
    return panel


def test_play_button_calls_playback_play(qtbot, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(playback, "play", lambda path, blocking=False: calls.append((path, blocking)))
    monkeypatch.setattr(playback, "AVAILABLE", True)
    panel = _panel_with_audio(qtbot, tmp_path)

    panel._on_play_clicked()

    assert len(calls) == 1
    assert calls[0][0] == panel._view._loaded_path
    assert calls[0][1] is False


def test_play_starts_timer_when_playback_available(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(playback, "play", lambda *a, **k: None)
    monkeypatch.setattr(playback, "AVAILABLE", True)
    panel = _panel_with_audio(qtbot, tmp_path)

    panel._on_play_clicked()

    assert panel._timer.isActive()


def test_play_does_not_start_timer_when_playback_unavailable(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(playback, "play", lambda *a, **k: None)
    monkeypatch.setattr(playback, "AVAILABLE", False)
    panel = _panel_with_audio(qtbot, tmp_path)

    panel._on_play_clicked()

    assert not panel._timer.isActive()


def test_play_without_loaded_audio_is_a_noop(qtbot, monkeypatch):
    calls = []
    monkeypatch.setattr(playback, "play", lambda *a, **k: calls.append(a))
    panel = WaveformPanel()
    qtbot.addWidget(panel)

    panel._on_play_clicked()

    assert calls == []
    assert not panel._timer.isActive()


def test_stop_button_calls_playback_stop_and_stops_timer(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(playback, "play", lambda *a, **k: None)
    monkeypatch.setattr(playback, "AVAILABLE", True)
    stop_calls = []
    monkeypatch.setattr(playback, "stop", lambda: stop_calls.append(True))
    panel = _panel_with_audio(qtbot, tmp_path)
    panel._on_play_clicked()
    assert panel._timer.isActive()

    panel._on_stop_clicked()

    assert stop_calls == [True]
    assert not panel._timer.isActive()
    assert not panel._playhead_item.isVisible()


def test_timer_tick_moves_playhead_item(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(playback, "play", lambda *a, **k: None)
    monkeypatch.setattr(playback, "AVAILABLE", True)
    panel = _panel_with_audio(qtbot, tmp_path)
    panel._on_play_clicked()

    # Simulate 1/4 of the way through playback by moving _play_started into
    # the past, rather than waiting on the real timer/clock.
    import time
    panel._play_started = time.monotonic() - (panel._view.duration / 4)

    panel._on_timer_tick()

    expected_x = panel._view.viewport().width() / 4
    line = panel._playhead_item.line()
    assert line.x1() == line.x2()
    assert abs(line.x1() - expected_x) < 1.0
    assert panel._playhead_item.isVisible()


def test_timer_stops_after_duration_elapsed(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(playback, "play", lambda *a, **k: None)
    monkeypatch.setattr(playback, "AVAILABLE", True)
    panel = _panel_with_audio(qtbot, tmp_path)
    panel._on_play_clicked()

    import time
    panel._play_started = time.monotonic() - (panel._view.duration * 2)

    panel._on_timer_tick()

    assert not panel._timer.isActive()
