"""Tests for kokoro_gui/qt/waveform_view.py's WaveformItem/WaveformView -
needs qtbot/offscreen but not the full qt_app (QtTTSApp) fixture, since this
widget has no dependency on the app at all (Workstream 3 spike)."""
import numpy as np
import soundfile as sf

from kokoro_gui.qt.waveform_view import WaveformItem, WaveformView


def _write_silent_wav(path, sample_rate=8000, seconds=0.25):
    data = np.zeros(int(sample_rate * seconds), dtype=np.float32)
    sf.write(str(path), data, sample_rate)


def _write_tone_wav(path, sample_rate=8000, seconds=0.25, freq=440):
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    data = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), data, sample_rate)


def test_widget_constructs_without_crash_offscreen(qtbot):
    view = WaveformView()
    qtbot.addWidget(view)
    assert view.duration == 0.0


def test_bounding_rect_matches_set_peaks_dimensions():
    item = WaveformItem()
    peaks = np.zeros((10, 2), dtype=np.float32)
    item.set_peaks(peaks, width=200.0, height=80.0)
    rect = item.boundingRect()
    assert rect.width() == 200.0
    assert rect.height() == 80.0


def test_load_audio_sets_peaks_and_duration(qtbot, tmp_path):
    path = tmp_path / "tone.wav"
    _write_tone_wav(path)
    view = WaveformView()
    qtbot.addWidget(view)
    view.show()
    view.resize(300, 100)
    qtbot.wait(10)

    view.load_audio(str(path))

    assert view.duration > 0
    bucket_count = max(1, view.viewport().width())
    assert view.waveform_item._peaks.shape == (bucket_count, 2)


def test_resize_recomputes_bucket_count(qtbot, tmp_path):
    # QGraphicsView.resize() only actually changes viewport geometry once
    # the widget has been shown at least once - a plain resize() on a
    # never-shown widget is a no-op even under the offscreen platform.
    path = tmp_path / "tone.wav"
    _write_tone_wav(path)
    view = WaveformView()
    qtbot.addWidget(view)
    view.show()
    view.resize(150, 80)
    qtbot.wait(10)
    view.load_audio(str(path))
    first_count = view.waveform_item._peaks.shape[0]

    view.resize(400, 80)
    qtbot.wait(10)

    second_count = view.waveform_item._peaks.shape[0]
    assert second_count != first_count
    assert second_count == max(1, view.viewport().width())


def test_load_audio_with_silent_file_does_not_crash(qtbot, tmp_path):
    path = tmp_path / "silence.wav"
    _write_silent_wav(path)
    view = WaveformView()
    qtbot.addWidget(view)
    view.show()
    view.resize(200, 80)
    qtbot.wait(10)

    view.load_audio(str(path))

    assert np.all(view.waveform_item._peaks == 0.0)
