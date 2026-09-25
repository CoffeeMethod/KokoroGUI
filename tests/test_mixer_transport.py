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


def test_transport_scheduled_clip_with_a_slice_plays_only_that_range(tmp_path, make_transport):
    ramp = (np.arange(8000) / 8000.0).astype(np.float32)
    path = str(tmp_path / "ramp.wav")
    sf.write(path, ramp, 8000, subtype="FLOAT")
    transport = make_transport()
    transport.load([ScheduledClip("c1", 1.0, path, slice=(0.25, 0.5))], sample_rate=8000)

    clip = transport.loaded_clips()[0]
    assert len(clip.samples) == 2000 and clip.start_frame == 8000
    assert transport.duration() == 1.25
    transport.seek(1.0)
    block = render_block_for_test(transport, 2000)
    assert np.allclose(block[:, 0], ramp[2000:4000], atol=1e-6)
    assert np.allclose(block[:, 1], ramp[2000:4000], atol=1e-6)
    assert np.allclose(render_block_for_test(transport, 100), 0.0)


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


def test_a_duck_track_is_ducked_and_a_bed_never_feeds_the_sidechain():
    from kokoro_gui.daw.models import Clip, Run

    doc, _ta, tb, ca, cb, plan = _mix_doc()
    tb.duck = True
    bed = Clip(source="imported", original_audio_path="bed.wav", track_id=None)
    doc.clips.append(bed)
    doc.runs.append(Run("bed", bed.id, "placeholder"))
    mixes = plan()
    assert (mixes[ca.id].duck, mixes[ca.id].sidechain) == (False, True)
    assert (mixes[cb.id].duck, mixes[cb.id].sidechain) == (True, False)
    assert (mixes[bed.id].duck, mixes[bed.id].sidechain) == (False, False)


# -- ducking (phase 5 P2) ---------------------------------------------------------

DUCK_RATE = 8000


def _bed(samples, **kwargs):
    return mixer.LoadedClip("bed", start_frame=0, samples=samples.astype(np.float32), duck=True,
                            sidechain=False, **kwargs)


def _speech(samples, start_frame=0, **kwargs):
    return mixer.LoadedClip("speech", start_frame=start_frame, samples=samples.astype(np.float32), **kwargs)


def _blockwise(clips, total, block, duck):
    out = np.zeros((total, mixer.CHANNELS), dtype=np.float32)
    for start in range(0, total, block):
        n = min(block, total - start)
        mixer.mix_block(clips, start, n, out=out[start:start + n], duck=duck)
    return out


def test_a_bed_alone_plays_at_unity():
    bed = _bed(np.full(DUCK_RATE, 0.3))
    out = _blockwise([bed], DUCK_RATE, 512, mixer.DuckState(DUCK_RATE))
    assert np.allclose(out, 0.3)


def test_a_bed_under_full_scale_speech_is_down_12_db_after_the_attack_and_recovers_after_release():
    rate = DUCK_RATE
    # Speech hard left at full scale for 1 s, then 4 s of silence; the bed
    # plays centred, so the right column is the bed alone.
    speech = _speech(np.ones(rate), gain_l=1.0, gain_r=0.0)
    bed = _bed(np.full(5 * rate, 0.1))
    duck = mixer.DuckState(rate)
    out = _blockwise([speech, bed], 5 * rate, 512, duck)
    right = out[:, 1]

    assert right[0] == pytest.approx(0.1)  # nothing heard yet at the first frame
    after_attack = right[int(0.05 * rate):rate]
    assert np.allclose(20 * np.log10(after_attack / 0.1), -12.0, atol=0.01)
    # Still down just after the speech stops (the release is slow)...
    assert right[rate + int(0.1 * rate)] < 0.05
    # ...and back to unity well after it.
    assert right[-1] == pytest.approx(0.1, rel=0.01)


def test_duck_depth_follows_the_setting():
    rate = DUCK_RATE
    speech = _speech(np.ones(rate), gain_l=1.0, gain_r=0.0)
    bed = _bed(np.full(rate, 0.1))
    out = _blockwise([speech, bed], rate, 512, mixer.DuckState(rate, duck_db=-6.0))
    assert 20 * np.log10(out[-1, 1] / 0.1) == pytest.approx(-6.0, abs=0.01)


def test_duck_state_carries_across_blocks_so_block_size_does_not_matter():
    rate = DUCK_RATE
    rng = np.random.default_rng(0)
    speech = _speech(rng.uniform(-0.5, 0.5, 2 * rate) * (np.arange(2 * rate) % 3000 < 1500), start_frame=300)
    bed = _bed(rng.uniform(-0.2, 0.2, 3 * rate))
    total = 3 * rate
    one = _blockwise([speech, bed], total, total, mixer.DuckState(rate))
    for block in (512, 333, 40, 1):
        other = _blockwise([speech, bed], total, block, mixer.DuckState(rate))
        assert np.array_equal(one, other), block


def test_without_a_duck_state_ducked_clips_mix_plainly():
    speech = _speech(np.ones(100), gain_l=1.0, gain_r=0.0)
    bed = _bed(np.full(100, 0.1))
    out = mixer.mix_block([speech, bed], 0, 100)
    assert np.allclose(out[:, 1], 0.1)


def test_a_bed_does_not_duck_another_bed():
    # A bed on an unducked track isn't speech: it doesn't feed the sidechain.
    other = mixer.LoadedClip("other", 0, np.ones(DUCK_RATE, dtype=np.float32), gain_l=1.0, gain_r=0.0,
                             sidechain=False)
    bed = _bed(np.full(DUCK_RATE, 0.1))
    out = _blockwise([other, bed], DUCK_RATE, 512, mixer.DuckState(DUCK_RATE))
    assert np.allclose(out[:, 1], 0.1)


def test_transport_carries_the_duck_state_and_resets_it_on_seek(tmp_path, make_transport):
    rate = DUCK_RATE
    speech_path = _write(tmp_path / "speech.wav", np.full(rate, 0.9), rate=rate)
    bed_path = _write(tmp_path / "bed.wav", np.full(2 * rate, 0.1), rate=rate)
    schedule = [ScheduledClip("s", 0.0, speech_path, pan=-1.0),
                ScheduledClip("b", 0.0, bed_path, duck=True, sidechain=False)]
    transport = make_transport()
    transport.load(schedule, sample_rate=rate, duck_db=-12.0)

    played = np.concatenate([render_block_for_test(transport, 512) for _ in range(2 * rate // 512)])
    loaded = transport.loaded_clips()
    expected = _blockwise(loaded, len(played), len(played), mixer.DuckState(rate, -12.0))
    assert np.array_equal(played, expected)
    assert played[rate - 1, 1] < 0.03

    transport.seek(0.0)
    assert np.array_equal(render_block_for_test(transport, 512), expected[:512])


def test_mixdown_matches_the_transport_with_a_ducked_bed(tmp_path, make_transport):
    from kokoro_gui.daw.arrangement import compute_arrangement
    from kokoro_gui.daw.beds import playable_segments
    from kokoro_gui.daw.mixdown import mixdown
    from kokoro_gui.daw.mixplan import clip_mixes
    from kokoro_gui.daw.models import Character, Clip, Document, Run, Segment, Track

    rate = DUCK_RATE
    rng = np.random.default_rng(1)
    speech_path = str(tmp_path / "speech.wav")
    sf.write(speech_path, rng.uniform(-0.6, 0.6, rate).astype(np.float32), rate, subtype="FLOAT")
    bed_path = str(tmp_path / "bed.wav")
    sf.write(bed_path, rng.uniform(-0.2, 0.2, 3 * rate).astype(np.float32), rate, subtype="FLOAT")

    narrator = Character.from_preset_dict("N", {})
    voice = Track(name="N", character_id=narrator.id, pan=-1.0)
    music = Track(name="Music", role="music", order_index=1, duck=True)
    speech = Clip(character_id=narrator.id, track_id=voice.id, timeline_timestamp=0.5,
                  segments=[Segment(0, "hi", "k", speech_path, 1.0)])
    bed = Clip(source="imported", original_audio_path=bed_path, track_id=music.id, timeline_timestamp=0.0,
               pinned=True)
    doc = Document(runs=[Run("hi", speech.id, "generated"), Run("\n\n"), Run("bed", bed.id, "placeholder")],
                   clips=[speech, bed], tracks=[voice, music], characters=[narrator],
                   settings={"duck_db": -9.0})
    arrangement = compute_arrangement(doc, chars_per_second=10.0)

    out = str(tmp_path / "mix.wav")
    mixdown(doc, out, "wav", rate, arrangement=arrangement)
    exported, _rate = sf.read(out, dtype="float32", always_2d=True)

    mixes = clip_mixes(doc, arrangement)
    schedule = []
    for placed in arrangement.placed:
        mix = mixes[placed.clip.id]
        offset = placed.start_s
        for segment in playable_segments(placed.clip):
            schedule.append(ScheduledClip(placed.clip.id, offset, segment.audio_path, slice=segment.range,
                                          gain=mix.gain, pan=mix.pan, duck=mix.duck, sidechain=mix.sidechain))
            offset += segment.duration
    assert [(s.duck, s.sidechain) for s in schedule] == [(False, True), (True, False)]
    transport = make_transport()
    transport.load(schedule, sample_rate=rate, total_duration_s=arrangement.total_duration_s, duck_db=-9.0)
    played = np.concatenate([render_block_for_test(transport, 512) for _ in range(len(exported) // 512 + 1)])
    played = played[:len(exported)]

    assert len(exported) == 3 * rate
    # The wav is 16-bit.
    assert np.allclose(exported, played, atol=2.0 / 32768)
    # The bed (alone in the right column: the speech is panned hard left)
    # really was ducked while the speech played.
    assert np.abs(exported[int(1.2 * rate):int(1.4 * rate), 1]).max() < 0.2 * 10 ** (-9 / 20) + 1e-3
    assert np.abs(exported[int(2.5 * rate):, 1]).max() > 0.15
