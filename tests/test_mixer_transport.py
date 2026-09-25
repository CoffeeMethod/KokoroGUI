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

    assert out.shape == (4, 2)
    assert np.allclose(out[:, 0], [0.5, 0.5, 0.0, 0.0])
    assert np.allclose(out[:, 1], out[:, 0])


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


# -- phase 2: stereo, pan, fades, automation, loop region ------------------------


def test_centred_clip_is_equal_in_both_columns_and_unity():
    a = mixer.LoadedClip("a", 0, np.full(10, 0.5, dtype=np.float32))
    out = mixer.mix_block([a], 0, 10)
    assert np.allclose(out[:, 0], 0.5)
    assert np.allclose(out[:, 1], 0.5)


def test_pan_gains_are_constant_power():
    for pan in (-1.0, -0.3, 0.0, 0.5, 1.0):
        left, right = mixer.pan_gains(pan)
        assert abs(left ** 2 + right ** 2 - 2.0) < 1e-9
    assert mixer.pan_gains(0.0) == pytest.approx((1.0, 1.0))


def test_hard_left_pan_silences_the_right_column():
    left, right = mixer.pan_gains(-1.0)
    a = mixer.LoadedClip("a", 0, np.full(10, 0.5, dtype=np.float32), gain_l=left, gain_r=right)
    out = mixer.mix_block([a], 0, 10)
    assert np.allclose(out[:, 1], 0.0)
    assert np.allclose(out[:, 0], min(1.0, 0.5 * left))


def test_fade_in_ramp_reads_half_at_its_midpoint():
    samples = np.ones(1000, dtype=np.float32)
    a = mixer.LoadedClip("a", 0, samples, fade_in_frames=100)
    out = mixer.mix_block([a], 0, 200)
    assert out[0, 0] == 0.0
    assert out[50, 0] == pytest.approx(0.5)
    assert np.allclose(out[100:, 0], 1.0)
    assert np.all(samples == 1.0)  # the memoized render is never touched


def test_fade_out_ramp_ends_near_zero_and_blocks_agree():
    a = mixer.LoadedClip("a", 0, np.ones(1000, dtype=np.float32), fade_out_frames=100)
    whole = mixer.mix_block([a], 0, 1000)[:, 0]
    assert whole[899] == pytest.approx(1.0)
    assert whole[950] == pytest.approx(0.5)
    assert whole[999] == pytest.approx(0.01)
    split = np.concatenate([mixer.mix_block([a], 0, 930)[:, 0], mixer.mix_block([a], 930, 70)[:, 0]])
    assert np.allclose(whole, split)


def test_automation_interpolates_linearly_in_absolute_frames():
    automation = mixer.automation_arrays([[0.0, 0.0], [100 / 8000, 1.0]], 8000)
    a = mixer.LoadedClip("a", 0, np.ones(200, dtype=np.float32), automation=automation)
    out = mixer.mix_block([a], 0, 200)[:, 0]
    assert out[50] == pytest.approx(0.5)
    assert np.allclose(out[100:], 1.0)


def test_automation_arrays_sort_and_clamp():
    times, gains = mixer.automation_arrays([[2.0, 5.0], [1.0, -1.0]], 10)
    assert list(times) == [10.0, 20.0]
    assert list(gains) == [0.0, 2.0]
    assert mixer.automation_arrays([], 10) is None


def test_transport_loop_range_wraps_inside_the_region(tmp_path, make_transport):
    samples = np.concatenate([np.full(8000, 0.25), np.full(8000, 0.75)])
    path = _write(tmp_path / "a.wav", samples, rate=8000)
    transport = make_transport()
    transport.load([ScheduledClip("c1", 0.0, path)], sample_rate=8000)
    transport.set_loop_range_s(0.5, 1.0)  # frames 4000..8000
    transport.seek(0.75)  # frame 6000

    block = render_block_for_test(transport, 4000)

    assert np.allclose(block[:2000, 0], 0.25)
    assert np.allclose(block[2000:, 0], 0.25)  # wrapped to 4000, still the first half
    assert transport.position() == pytest.approx((4000 + 2000) / 8000)
    transport.set_loop_range_s(None)
    assert transport.loop_range is None


def test_transport_scheduled_clip_carries_fades_pan_and_automation(tmp_path, make_transport):
    path = _write(tmp_path / "a.wav", np.ones(8000), rate=8000)
    transport = make_transport()
    transport.load([ScheduledClip("c1", 0.0, path, pan=1.0, fade_in_s=0.5, automation=((0.0, 1.0),))],
                   sample_rate=8000)
    clip = transport.loaded_clips()[0]
    assert clip.fade_in_frames == 4000
    assert clip.gain_l == pytest.approx(0.0, abs=1e-9)
    assert clip.automation is not None


# -- mix plan (track controls) ---------------------------------------------------


def _mix_doc(**settings):
    from kokoro_gui.daw.arrangement import compute_arrangement
    from kokoro_gui.daw.models import Character, Clip, Document, Run, Track

    alice, bob = Character.from_preset_dict("Alice", {}), Character.from_preset_dict("Bob", {})
    ta, tb = Track(name="A", character_id=alice.id), Track(name="B", character_id=bob.id, order_index=1)
    ca, cb = Clip(character_id=alice.id, track_id=ta.id), Clip(character_id=bob.id, track_id=tb.id)
    doc = Document(runs=[Run(text="x" * 10, clip_id=ca.id), Run(text="y" * 10, clip_id=cb.id)],
                   clips=[ca, cb], tracks=[ta, tb], characters=[alice, bob],
                   settings={"gap_s": 0.0, **settings})

    def plan():
        from kokoro_gui.daw.mixplan import clip_mixes

        return clip_mixes(doc, compute_arrangement(doc, chars_per_second=10.0))

    return doc, ta, tb, ca, cb, plan


def test_mute_leaves_the_track_out_and_the_fader_scales():
    _doc, ta, tb, ca, cb, plan = _mix_doc()
    ta.mute = True
    tb.gain = 0.5
    mixes = plan()
    assert ca.id not in mixes
    assert mixes[cb.id].gain == 0.5


def test_solo_isolates_the_soloed_track():
    _doc, ta, _tb, ca, cb, plan = _mix_doc()
    ta.solo = True
    mixes = plan()
    assert set(mixes) == {ca.id}


def test_pan_and_automation_come_from_the_track():
    _doc, ta, _tb, ca, _cb, plan = _mix_doc()
    ta.pan = -0.5
    ta.automation = [[0.0, 1.0], [1.0, 0.0]]
    mix = plan()[ca.id]
    assert mix.pan == -0.5
    assert mix.automation == ((0.0, 1.0), (1.0, 0.0))


def test_auto_crossfade_only_when_the_option_is_on():
    from kokoro_gui.daw.mixplan import AUTO_CROSSFADE_S

    doc, _ta, _tb, ca, cb, plan = _mix_doc()
    cb.timeline_timestamp = 0.5  # starts inside ca (0..1s)
    assert plan()[cb.id].fade_in_s == 0.0
    assert plan()[ca.id].fade_out_s == 0.0

    doc.settings["auto_crossfade"] = True
    mixes = plan()
    assert mixes[cb.id].fade_in_s == AUTO_CROSSFADE_S
    assert mixes[ca.id].fade_out_s == AUTO_CROSSFADE_S
    cb.fade_in_s = 0.2
    assert plan()[cb.id].fade_in_s == 0.2
    assert cb.fade_in_s == 0.2  # never written by the option
