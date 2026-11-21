"""Record a guide take (plan 31, grill PG8): a (fake) recording becomes the
clip's active take with its file under audio/imported, one undo removes it,
it survives save and reopen, and the clip never reads as stale. The panel
drives a fake input stream; no test opens a real audio device."""
import os

import numpy as np
import pytest
import soundfile as sf

import playback
from kokoro_gui.audio import recorder
from kokoro_gui.daw.dirty import build_segments_from_results
from kokoro_gui.qt import guide_take

RATE = 8000


def _generated_clip(qt_app, text="Hello there.", seconds=0.5):
    """A clean clip whose one segment is a real wav named by its key."""
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


def _recording(seconds_after_count_in=1.5):
    beeps = recorder.count_in(RATE)
    line = np.full(int(RATE * seconds_after_count_in), 0.3, dtype=np.float32)
    return np.concatenate([beeps, line])


def _stale_ids(qt_app):
    return {clip.id for clip in qt_app.document.dirty_clips()}


def test_a_recording_becomes_the_active_take_with_its_file_under_audio_imported(qt_app):
    clip = _generated_clip(qt_app)
    generated = list(clip.segments)
    assert clip.id not in _stale_ids(qt_app)

    segment = guide_take.store_guide_take(qt_app, clip.id, _recording(), RATE)

    assert segment is not None and clip.segments == [segment]
    assert clip.overrides["take"] == 1 and clip.takes == {0: generated}
    imported = os.path.join(qt_app.project_dir, "audio", "imported")
    assert os.path.dirname(segment.audio_path) == imported and os.path.isfile(segment.audio_path)
    assert segment.cache_key is None and segment.raw
    assert segment.range == [recorder.COUNT_IN_S, pytest.approx(recorder.COUNT_IN_S + 1.5)]
    assert segment.duration == pytest.approx(1.5)
    assert segment.text == "Hello there."
    assert clip.id not in _stale_ids(qt_app)


def test_the_arrangement_plays_the_take_without_the_count_in(qt_app):
    clip = _generated_clip(qt_app)
    guide_take.store_guide_take(qt_app, clip.id, _recording(1.5), RATE)
    placed = qt_app.build_arrangement().by_clip_id()[clip.id]
    assert placed.duration_s == pytest.approx(1.5, abs=0.01)


def test_one_undo_removes_the_take_and_brings_back_the_generated_one(qt_app):
    clip = _generated_clip(qt_app)
    generated = list(clip.segments)
    guide_take.store_guide_take(qt_app, clip.id, _recording(), RATE)

    qt_app.document.undo_stack.undo()

    assert clip.segments == generated and clip.takes == {} and "take" not in clip.overrides
    assert clip.id not in _stale_ids(qt_app)


def test_the_take_is_still_not_stale_after_the_text_changes(qt_app):
    clip = _generated_clip(qt_app)
    guide_take.store_guide_take(qt_app, clip.id, _recording(), RATE)
    new_text = "Hello friend."
    qt_app.document.replace_text(6, len("there"), len("friend"), new_text)
    assert qt_app.document.clip_text(clip) == new_text
    assert clip.id not in _stale_ids(qt_app)
    # The generated take, made active again, is stale for the new text.
    qt_app.timeline_dock.on_take_selected(clip.id, 0)
    assert clip.id in _stale_ids(qt_app)


def test_the_take_survives_save_and_reopen(qt_app, tmp_path):
    clip = _generated_clip(qt_app)
    guide_take.store_guide_take(qt_app, clip.id, _recording(), RATE)
    path = str(tmp_path / "guide.tbaw")
    qt_app.save_project_as(path)
    qt_app.wait_for_project_io()

    qt_app.open_project(path)
    qt_app.wait_for_project_io()

    reopened = qt_app.document.clips[0]
    segment = reopened.segments[0]
    assert segment.cache_key is None and os.path.isfile(segment.audio_path)
    assert os.path.dirname(segment.audio_path) == os.path.join(qt_app.project_dir, "audio", "imported")
    assert segment.range == [recorder.COUNT_IN_S, pytest.approx(recorder.COUNT_IN_S + 1.5)]
    assert reopened.overrides["take"] == 1 and list(reopened.takes) == [0]
    assert reopened.id not in _stale_ids(qt_app)


def test_a_recording_shorter_than_the_count_in_keeps_nothing(qt_app):
    clip = _generated_clip(qt_app)
    before = list(clip.segments)
    assert guide_take.store_guide_take(qt_app, clip.id, recorder.count_in(RATE)[:RATE], RATE) is None
    assert clip.segments == before and clip.takes == {}
    assert not os.path.isdir(os.path.join(qt_app.project_dir, "audio", "imported"))


def test_a_generation_landing_on_the_recorded_takes_index_parks_it(qt_app):
    clip = _generated_clip(qt_app)
    segment = guide_take.store_guide_take(qt_app, clip.id, _recording(), RATE)
    # Regenerating the active (recorded) take: the engine lands on take 1.
    text = qt_app.document.clip_text(clip)
    path = os.path.join(qt_app.project_dir, "audio", "generated", "new_0.wav")
    sf.write(path, np.zeros(2400, dtype=np.float32), 24000)
    qt_app.timeline_dock._apply_results(clip, [{"text": text, "path": path, "duration": 0.1, "cache_key": "new",
                                                "take": 1}])
    assert clip.segments[0].cache_key == "new" and clip.overrides["take"] == 1
    assert clip.takes[2] == [segment]


# -- the panel ----------------------------------------------------------------------


class FakeInputStream:
    def __init__(self, sample_rate, channels, device, callback):
        self.sample_rate = sample_rate
        self.device = device
        self.callback = callback
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        pass

    def feed(self, samples):
        block = np.asarray(samples, dtype=np.float32).reshape(-1, 1)
        self.callback(block, len(block), None, 0)


@pytest.fixture
def fake_audio(qt_app, monkeypatch):
    """A fake input stream, the count-in captured instead of played, and
    the transport's seek and play recorded instead of opening a stream."""
    streams, beeps, transport = [], [], []

    def factory(sample_rate, channels, device, callback):
        stream = FakeInputStream(sample_rate, channels, device, callback)
        streams.append(stream)
        return stream

    monkeypatch.setattr(recorder, "default_stream_factory", factory)
    monkeypatch.setattr(recorder, "available", lambda: True)
    monkeypatch.setattr(recorder, "input_devices", lambda: [(2, "Test mic", RATE)])
    monkeypatch.setattr(playback, "play_array", lambda data, rate: beeps.append((len(data), rate)))
    monkeypatch.setattr(qt_app.transport, "seek", lambda s: transport.append(("seek", s)))
    monkeypatch.setattr(qt_app.transport, "play", lambda: transport.append(("play",)))
    monkeypatch.setattr(qt_app.transport, "pause", lambda: transport.append(("pause",)))
    return streams, beeps, transport


def _place_clip_at(qt_app, clip, seconds):
    clip.timeline_timestamp = seconds
    clip.pinned = True


def test_the_panel_records_with_a_count_in_and_pre_roll_into_a_take(qt_app, qtbot, fake_audio):
    streams, beeps, transport = fake_audio
    clip = _generated_clip(qt_app)
    _place_clip_at(qt_app, clip, 5.0)
    panel = qt_app.record_guide_take(clip.id)
    assert panel is not None and panel.isVisible()
    assert panel.device_combo.currentData() is None  # the default input first
    panel.device_combo.setCurrentIndex(panel.device_combo.findData(2))
    panel.preroll_spin.setValue(2.0)

    panel.record_btn.click()

    assert panel.is_recording() and streams[0].started and streams[0].device == 2
    assert beeps == [(int(recorder.COUNT_IN_S * RATE), RATE)]
    # The pre-roll starts 2 s before the clip, 1 s into the 3 s count-in.
    assert panel.transport_delay_ms() == 1000
    panel.start_transport()
    assert transport == [("seek", pytest.approx(3.0)), ("play",)]

    streams[0].feed(_recording(1.0))
    panel.update_level()
    assert panel.level_bar.value() > 0

    panel.stop_btn.click()

    assert not panel.is_recording() and ("pause",) in transport
    assert clip.overrides["take"] == 1 and clip.segments[0].range[0] == recorder.COUNT_IN_S
    assert clip.segments[0].duration == pytest.approx(1.0)


def test_a_clip_near_the_start_gets_only_the_pre_roll_there_is(qt_app, qtbot, fake_audio):
    _streams, _beeps, transport = fake_audio
    clip = _generated_clip(qt_app)
    _place_clip_at(qt_app, clip, 0.5)
    panel = qt_app.record_guide_take(clip.id)
    panel.preroll_spin.setValue(2.0)
    panel.record_btn.click()
    assert panel.transport_delay_ms() == 2500
    panel.start_transport()
    assert transport[0] == ("seek", pytest.approx(0.0))
    panel.stop_btn.click()


def test_recording_is_refused_while_the_project_is_saving(qt_app, qtbot, fake_audio, monkeypatch):
    streams, _beeps, _transport = fake_audio
    clip = _generated_clip(qt_app)
    panel = qt_app.record_guide_take(clip.id)
    monkeypatch.setattr(qt_app, "is_io_busy", lambda: True)
    panel.record_btn.click()
    assert not panel.is_recording() and streams == []


def test_a_stop_during_a_save_keeps_the_take_until_the_save_ends(qt_app, qtbot, fake_audio, monkeypatch):
    streams, _beeps, _transport = fake_audio
    clip = _generated_clip(qt_app)
    panel = qt_app.record_guide_take(clip.id)
    panel.device_combo.setCurrentIndex(panel.device_combo.findData(2))
    panel.record_btn.click()
    streams[0].feed(_recording(1.0))
    busy = [True]
    monkeypatch.setattr(qt_app, "is_io_busy", lambda: busy[0])
    panel.stop_btn.click()
    assert "take" not in clip.overrides and panel.has_pending_take()
    busy[0] = False
    panel.commit_pending()
    assert clip.overrides["take"] == 1 and not panel.has_pending_take()


def test_record_needs_a_speech_clip(qt_app, fake_audio):
    qt_app.selection.clear()
    assert qt_app.record_guide_take() is None


def test_the_r_key_and_the_block_menu_open_the_panel_on_the_clip(qt_app, qtbot, fake_audio):
    clip = _generated_clip(qt_app)
    qt_app.selection.select_clip(clip.id)
    qt_app.record_guide_take_key()
    assert qt_app.guide_take_panel is not None and qt_app.guide_take_panel.clip_id == clip.id
    qt_app.guide_take_panel.close()
    qt_app.timeline_dock.timeline_view.recordTakeRequested.emit(clip.id)
    assert qt_app.guide_take_panel.isVisible() and qt_app.guide_take_panel.clip_id == clip.id


def test_the_transport_docks_rec_button_opens_the_panel(qt_app, fake_audio):
    clip = _generated_clip(qt_app)
    qt_app.selection.select_clip(clip.id)
    qt_app.transport_dock.record_btn.click()
    assert qt_app.guide_take_panel.isVisible() and qt_app.guide_take_panel.clip_id == clip.id


def test_the_take_menu_marks_a_recorded_take(qt_app, fake_audio):
    from PySide6.QtWidgets import QMenu

    clip = _generated_clip(qt_app)
    guide_take.store_guide_take(qt_app, clip.id, _recording(), RATE)
    menu = QMenu()
    qt_app.timeline_dock.timeline_view._document = qt_app.document
    qt_app.timeline_dock.timeline_view._add_take_menu(menu, clip)
    take_menu = menu.actions()[0].menu()
    labels = [a.text() for a in take_menu.actions()]
    assert labels[0].startswith("Take 1 (") and labels[1].startswith("Take 2 (recorded, ")
