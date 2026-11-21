"""kokoro_gui/audio/recorder.py: the guide take's input (plan 31), on a fake stream."""
import numpy as np
import pytest

import playback
from kokoro_gui.audio import recorder
from kokoro_gui.audio.recorder import Recorder, RecorderError


class FakeInputStream:
    """Stands in for `sounddevice.InputStream`: `feed` calls the callback
    the way PortAudio would."""

    def __init__(self, sample_rate, channels, device, callback):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.callback = callback
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True

    def feed(self, block, status=0):
        self.callback(np.asarray(block, dtype=np.float32), len(block), None, status)


def _factory(made):
    def factory(sample_rate, channels, device, callback):
        stream = FakeInputStream(sample_rate, channels, device, callback)
        made.append(stream)
        return stream
    return factory


def test_stop_returns_the_blocks_joined_in_order():
    made = []
    rec = Recorder(sample_rate=8000, device=3, stream_factory=_factory(made))
    rec.start()
    stream = made[0]
    assert stream.started and stream.device == 3 and stream.sample_rate == 8000 and stream.channels == 1
    stream.feed(np.full((4, 1), 0.25))
    stream.feed(np.full((2, 1), -0.5))
    assert rec.recording
    assert rec.seconds() == pytest.approx(6 / 8000)
    samples = rec.stop()
    assert samples.dtype == np.float32 and samples.ndim == 1
    np.testing.assert_allclose(samples, [0.25] * 4 + [-0.5] * 2)
    assert stream.closed and not rec.recording


def test_a_block_is_copied_so_portaudio_can_reuse_its_buffer():
    made = []
    rec = Recorder(stream_factory=_factory(made))
    rec.start()
    buffer = np.full((3, 1), 0.1, dtype=np.float32)
    made[0].callback(buffer, 3, None, 0)
    buffer[:] = 0.9
    np.testing.assert_allclose(rec.stop(), [0.1, 0.1, 0.1], rtol=1e-6)


def test_level_is_the_peak_of_the_latest_block_and_zero_once_stopped():
    made = []
    rec = Recorder(stream_factory=_factory(made))
    assert rec.level() == 0.0
    rec.start()
    made[0].feed([[0.1], [-0.7], [0.3]])
    assert rec.level() == pytest.approx(0.7)
    made[0].feed([[0.2], [0.05]])
    assert rec.level() == pytest.approx(0.2)
    rec.stop()
    assert rec.level() == 0.0


def test_stop_without_input_is_empty_and_a_second_start_clears_the_last_take():
    made = []
    rec = Recorder(stream_factory=_factory(made))
    rec.start()
    assert rec.stop().size == 0
    rec.start()
    made[1].feed([[0.5]])
    rec.stop()
    rec.start()
    made[2].feed([[0.25]])
    np.testing.assert_allclose(rec.stop(), [0.25])


def test_stereo_input_keeps_the_first_channel():
    made = []
    rec = Recorder(channels=2, stream_factory=_factory(made))
    rec.start()
    made[0].feed([[0.1, 0.9], [0.2, 0.8]])
    np.testing.assert_allclose(rec.stop(), [0.1, 0.2], rtol=1e-6)


def test_an_overflow_flag_is_counted():
    made = []
    rec = Recorder(stream_factory=_factory(made))
    rec.start()
    made[0].feed([[0.0]], status=1)
    assert rec.overflows == 1


def test_a_stream_that_will_not_open_raises_recorder_error():
    def broken(*_args):
        raise OSError("device busy")

    rec = Recorder(stream_factory=broken)
    with pytest.raises(RecorderError, match="device busy"):
        rec.start()
    assert not rec.recording


def test_without_portaudio_nothing_is_available_and_start_refuses(monkeypatch):
    monkeypatch.setattr(playback, "AVAILABLE", False)
    monkeypatch.setattr(playback, "sd", None)
    assert recorder.available() is False
    assert recorder.input_devices() == []
    with pytest.raises(RecorderError):
        Recorder().start()


def test_the_default_factory_opens_an_input_stream_with_the_playback_block_size(monkeypatch):
    calls = {}

    class FakeSd:
        @staticmethod
        def InputStream(**kwargs):
            calls.update(kwargs)
            return FakeInputStream(kwargs["samplerate"], kwargs["channels"], kwargs["device"], kwargs["callback"])

    monkeypatch.setattr(playback, "AVAILABLE", True)
    monkeypatch.setattr(playback, "sd", FakeSd)
    monkeypatch.setattr(playback, "stream_blocksize", lambda rate: 2048 if rate == 48000 else -1)
    rec = Recorder(sample_rate=48000, device=1)
    rec.start()
    assert calls["blocksize"] == 2048 and calls["dtype"] == "float32" and calls["device"] == 1
    rec.stop()


def test_input_devices_lists_only_devices_with_an_input(monkeypatch):
    class FakeSd:
        @staticmethod
        def query_devices():
            return [{"name": "Speakers", "max_input_channels": 0, "default_samplerate": 48000.0},
                    {"name": "Mic", "max_input_channels": 1, "default_samplerate": 44100.0}]

    monkeypatch.setattr(playback, "AVAILABLE", True)
    monkeypatch.setattr(playback, "sd", FakeSd)
    assert recorder.input_devices() == [(1, "Mic", 44100)]


def test_count_in_is_three_beeps_a_second_apart_with_silence_between():
    rate = 8000
    clicks = recorder.count_in(rate)
    assert clicks.dtype == np.float32
    assert len(clicks) == int(recorder.COUNT_IN_S * rate)
    beep = int(recorder.BEEP_S * rate)
    for beat in range(recorder.BEEPS):
        start = int(beat * recorder.BEAT_S * rate)
        assert np.abs(clicks[start:start + beep]).max() > 0.3
        assert np.abs(clicks[start + beep:start + rate]).max() == 0.0
    assert np.abs(clicks).max() <= recorder.BEEP_GAIN + 1e-6
    # The ramps keep each beep's edges quiet.
    assert abs(clicks[0]) < 0.01


def test_play_array_uses_the_stream_block_size(monkeypatch):
    calls = []

    class FakeSd:
        @staticmethod
        def play(data, rate, blocksize=None):
            calls.append((len(data), rate, blocksize))

    monkeypatch.setattr(playback, "AVAILABLE", True)
    monkeypatch.setattr(playback, "sd", FakeSd)
    monkeypatch.setattr(playback, "stream_blocksize", lambda rate: 512)
    playback.play_array(np.zeros(10, dtype=np.float32), 48000)
    assert calls == [(10, 48000, 512)]
