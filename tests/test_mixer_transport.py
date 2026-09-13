"""Tests for kokoro_gui/audio/mixer.py and kokoro_gui/audio/transport.py:
the block-mixing arithmetic and the transport state machine, driven with a
fake output stream so no audio device is ever opened (same rule as
conftest.py's `playback` mock)."""
import numpy as np
import pytest
import soundfile as sf

from kokoro_gui.audio import mixer
from kokoro_gui.audio.transport import ScheduledClip, Transport, render_block_for_test

pytest.importorskip("PySide6")


def _write(path, samples, rate=8000):
    sf.write(str(path), samples.astype(np.float32), rate)
    return str(path)


class FakeStream:
    instances = []

    def __init__(self, sample_rate, callback):
        self.sample_rate = sample_rate
        self.callback = callback
        self.started = False
        self.closed = False
        FakeStream.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True

    def pull(self, frames):
        out = np.zeros((frames, 1), dtype=np.float32)
        self.callback(out, frames)
        return out[:, 0]


@pytest.fixture
def fake_factory():
    FakeStream.instances = []
    yield lambda rate, cb: FakeStream(rate, cb)
    FakeStream.instances = []


@pytest.fixture
def make_transport(fake_factory):
    """Every `Transport` a test builds is stopped at teardown. A test that
    leaves one playing (the loop test does) leaves its 30Hz `QTimer`
    running; a later test's `processEvents` then ticks it while the
    object is being garbage-collected, which crashed the Windows CI leg
    with an access violation."""
    made = []

    def _make():
        transport = Transport(stream_factory=fake_factory)
        made.append(transport)
        return transport

    yield _make
    for transport in made:
        transport.stop()


# -- mixer -----------------------------------------------------------------------


def test_mix_block_sums_overlapping_clips_and_clips_to_unit_range():
    a = mixer.LoadedClip("a", start_frame=0, samples=np.full(10, 0.6, dtype=np.float32))
    b = mixer.LoadedClip("b", start_frame=5, samples=np.full(10, 0.6, dtype=np.float32))

    out = mixer.mix_block([a, b], 0, 20)

    assert np.allclose(out[:5], 0.6)
    assert np.allclose(out[5:10], 1.0)  # 1.2 clipped
    assert np.allclose(out[10:15], 0.6)
    assert np.allclose(out[15:], 0.0)


def test_mix_block_applies_gain_and_partial_overlap():
    a = mixer.LoadedClip("a", start_frame=3, samples=np.ones(4, dtype=np.float32), gain=0.5)

    out = mixer.mix_block([a], 5, 4)  # block covers frames 5..9, clip covers 3..7

    assert np.allclose(out, [0.5, 0.5, 0.0, 0.0])


def test_load_clip_samples_resamples_and_downmixes(tmp_path):
    stereo = np.stack([np.ones(800), np.zeros(800)], axis=1)
    path = _write(tmp_path / "s.wav", stereo, rate=8000)

    mono = mixer.load_clip_samples(path, 16000)

    assert mono.ndim == 1
    assert abs(len(mono) - 1600) <= 2
    assert np.allclose(mono[100:1500], 0.5, atol=0.05)


def test_total_frames_is_the_latest_end():
    a = mixer.LoadedClip("a", 0, np.zeros(10, dtype=np.float32))
    b = mixer.LoadedClip("b", 5, np.zeros(10, dtype=np.float32))
    assert mixer.total_frames([a, b]) == 15
    assert mixer.total_frames([]) == 0


# -- transport --------------------------------------------------------------------


def test_transport_load_positions_clips_and_reports_duration(tmp_path, make_transport):
    path = _write(tmp_path / "a.wav", np.ones(8000), rate=8000)  # 1s
    transport = make_transport()

    transport.load([ScheduledClip("c1", 2.0, path)], sample_rate=8000, total_duration_s=5.0)

    clips = transport.loaded_clips()
    assert clips[0].start_frame == 16000
    assert transport.duration() == 5.0
    assert transport.state == "stopped"


def test_transport_callback_mixes_at_the_right_frames(tmp_path, make_transport):
    path = _write(tmp_path / "a.wav", np.full(8000, 0.5), rate=8000)
    transport = make_transport()
    transport.load([ScheduledClip("c1", 1.0, path)], sample_rate=8000)

    silence = render_block_for_test(transport, 4000)  # frames 0..4000: before the clip
    assert np.allclose(silence, 0.0)
    transport.seek(1.0)
    block = render_block_for_test(transport, 4000)  # frames 8000..12000: inside
    assert np.allclose(block, 0.5)
    assert abs(transport.position() - 1.5) < 1e-6


def test_transport_play_pause_stop_state_machine(tmp_path, make_transport):
    path = _write(tmp_path / "a.wav", np.ones(8000), rate=8000)
    transport = make_transport()
    transport.load([ScheduledClip("c1", 0.0, path)], sample_rate=8000)
    states = []
    transport.stateChanged.connect(states.append)

    transport.play()
    assert transport.is_playing
    assert FakeStream.instances[-1].started
    transport.pause()
    assert transport.state == "paused"
    assert FakeStream.instances[-1].closed
    transport.play()
    transport.stop()
    assert transport.state == "stopped"
    assert transport.position() == 0.0
    assert states == ["playing", "paused", "playing", "stopped"]


def test_transport_reaching_the_end_stops_and_emits_finished(tmp_path, make_transport):
    path = _write(tmp_path / "a.wav", np.ones(800), rate=8000)  # 0.1s
    transport = make_transport()
    transport.load([ScheduledClip("c1", 0.0, path)], sample_rate=8000)
    finished = []
    transport.finished.connect(lambda: finished.append(True))

    transport.play()
    FakeStream.instances[-1].pull(1000)  # past the end
    transport.process_pending()

    assert finished == [True]
    assert transport.state == "stopped"
    assert transport.position() == 0.1


def test_transport_loop_wraps_instead_of_stopping(tmp_path, make_transport):
    path = _write(tmp_path / "a.wav", np.ones(800), rate=8000)
    transport = make_transport()
    transport.load([ScheduledClip("c1", 0.0, path)], sample_rate=8000)
    transport.loop = True

    transport.play()
    FakeStream.instances[-1].pull(1000)
    transport.process_pending()

    assert transport.is_playing
    assert transport.position() == 200 / 8000


def test_transport_play_with_nothing_loaded_is_a_noop(make_transport):
    transport = make_transport()
    transport.play()
    assert transport.state == "stopped"
    assert FakeStream.instances == []


def test_transport_reload_keeps_position_and_skips_unreadable_paths(tmp_path, make_transport):
    path = _write(tmp_path / "a.wav", np.ones(8000), rate=8000)
    transport = make_transport()
    transport.load([ScheduledClip("c1", 0.0, path)], sample_rate=8000)
    transport.seek(0.5)

    transport.reload([ScheduledClip("c1", 0.0, path), ScheduledClip("c2", 1.0, str(tmp_path / "missing.wav")),
                      ScheduledClip("c3", 2.0, None)], sample_rate=8000)

    assert [c.clip_id for c in transport.loaded_clips()] == ["c1"]
    assert abs(transport.position() - 0.5) < 1e-6
