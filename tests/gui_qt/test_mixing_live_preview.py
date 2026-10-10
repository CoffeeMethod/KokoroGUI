"""The Mixing dock's Live preview: it follows the ratio slider and the voice
and operation combos, only the latest request plays, and nothing is written
into the voice store (the blend is made in memory)."""
import asyncio
import os
from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest

import playback
import kokoro_gui.qt.app  # noqa: F401 - the docks package needs the app imported first
from kokoro_gui.qt.docks.mixing_dock import first_sentence


@pytest.fixture(autouse=True)
def _no_audio(monkeypatch):
    monkeypatch.setattr(playback, "play", MagicMock())
    monkeypatch.setattr(playback, "stop", MagicMock())


@pytest.fixture
def pending(qt_app, monkeypatch):
    """`backend.run` hands back futures the test resolves itself."""
    queued = []

    def fake_run(coro):
        coro.close()
        future = Future()
        queued.append(future)
        return future

    monkeypatch.setattr(qt_app.backend, "run", fake_run, raising=False)
    return queued


def test_first_sentence():
    assert first_sentence("One here. Two there.") == "One here."
    assert first_sentence("No end") == "No end"
    assert first_sentence("") == ""
    assert first_sentence("これは。もう一つ。") == "これは。"


def test_live_is_off_until_checked(qt_app, pending):
    dock = qt_app.mixing_dock
    assert not dock.live_check.isChecked()
    dock.ratio_slider.setValue(20)
    dock.ratio_slider.sliderReleased.emit()
    dock.op_combo.setCurrentText("add")
    assert pending == [] and not dock._live_timer.isActive()


def test_three_quick_releases_play_only_the_last_result(qt_app, pending):
    dock = qt_app.mixing_dock
    dock.live_check.setChecked(True)
    assert len(pending) == 1  # turning Live on speaks the Mix as it is
    for value in (20, 40, 60):
        dock.ratio_slider.setValue(value)
        dock.ratio_slider.sliderReleased.emit()
    assert len(pending) == 4
    last_path = dock._preview_path
    assert not dock._live_timer.isActive()

    for future in pending:  # they finish in order
        future.set_result((True, ""))

    playback.play.assert_called_once_with(last_path)
    assert os.path.exists(last_path)


def test_a_late_older_result_does_not_play_or_leave_a_file(qt_app, pending):
    dock = qt_app.mixing_dock
    dock.live_check.setChecked(True)
    first_path = dock._preview_path
    dock.ratio_slider.setValue(70)
    dock.ratio_slider.sliderReleased.emit()
    second_path = dock._preview_path
    assert first_path != second_path

    pending[1].set_result((True, ""))   # the newer one first
    open(first_path, "wb").close()      # the older one's engine writes its file late
    pending[0].set_result((True, ""))

    playback.play.assert_called_once_with(second_path)
    assert not os.path.exists(first_path)


def test_a_new_live_request_stops_the_sound_before_it(qt_app, pending):
    dock = qt_app.mixing_dock
    dock.live_check.setChecked(True)
    playback.stop.reset_mock()
    dock.ratio_slider.setValue(10)
    dock.ratio_slider.sliderReleased.emit()
    assert playback.stop.call_count == 1
    pending[1].set_result((True, ""))
    assert playback.stop.call_count == 2  # again just before the new one plays


def test_a_failed_latest_preview_says_why(qt_app, pending):
    dock = qt_app.mixing_dock
    dock.live_check.setChecked(True)
    pending[0].set_result((False, "Failed to load one of the voices."))
    assert dock.mix_status_label.text() == "Preview failed: Failed to load one of the voices."
    playback.play.assert_not_called()


def test_keyboard_steps_wait_for_a_pause(qt_app, qtbot, pending):
    dock = qt_app.mixing_dock
    dock.live_check.setChecked(True)
    for value in (31, 32, 33):
        dock.ratio_slider.setValue(value)
    assert len(pending) == 1 and dock._live_timer.isActive()
    qtbot.waitUntil(lambda: len(pending) == 2, timeout=3000)
    assert not dock._live_timer.isActive()


def test_a_release_after_a_change_does_not_repeat_it(qt_app, pending):
    dock = qt_app.mixing_dock
    dock.live_check.setChecked(True)
    dock.ratio_slider.setValue(35)
    dock.ratio_slider.sliderReleased.emit()
    dock.ratio_slider.sliderReleased.emit()
    assert len(pending) == 2


def test_changing_a_voice_or_the_operation_previews_after_the_pause(qt_app, qtbot, pending):
    dock = qt_app.mixing_dock
    dock.live_check.setChecked(True)
    dock.op_combo.setCurrentText("add")
    assert dock._live_timer.isActive()
    qtbot.waitUntil(lambda: len(pending) == 2, timeout=3000)
    dock.voice_b_combo.setCurrentIndex((dock.voice_b_combo.currentIndex() + 1) % dock.voice_b_combo.count())
    qtbot.waitUntil(lambda: len(pending) == 3, timeout=3000)


def test_the_plain_preview_button_still_works_with_live_off(qt_app, pending):
    dock = qt_app.mixing_dock
    dock.preview_mix()
    assert len(pending) == 1
    pending[0].set_result((True, ""))
    playback.play.assert_called_once_with(dock._preview_path)


def _sync_run(coro):
    future = Future()
    future.set_result(asyncio.run(coro))
    return future


def test_live_speaks_only_the_first_sentence(qt_app, monkeypatch):
    backend = qt_app.backend
    spoken = []

    async def mix_tensor(v1, v2, ratio, op="mix"):
        return True, "", object()

    async def preview_mix(tensor, voice_name, text, output_path, lang_code):
        spoken.append(text)
        return True

    monkeypatch.setattr(backend, "mix_tensor", mix_tensor, raising=False)
    monkeypatch.setattr(backend, "preview_mix", preview_mix, raising=False)
    monkeypatch.setattr(backend, "run", _sync_run, raising=False)
    monkeypatch.setattr(backend, "preview_text", lambda lang: "First one. Second one.", raising=False)

    dock = qt_app.mixing_dock
    dock.preview_mix()
    dock.live_check.setChecked(True)
    assert spoken == ["First one. Second one.", "First one."]


def test_a_live_preview_writes_nothing_into_the_voice_store(qt_app, fake_pipeline, tmp_path, monkeypatch):
    """Through the real engine code: the blend and the speech both happen in
    memory, and `mix_voices` (which saves) is never called."""
    import kokoro_engine
    from kokoro_gui.engine import runtime
    from kokoro_gui.engine.runner import EngineRunner
    from kokoro_gui.engine.voices import VoiceMixingMixin

    class MixingEngine(VoiceMixingMixin, EngineRunner):
        pass

    store = tmp_path / "custom_voices"  # qt_app made it
    monkeypatch.setattr(runtime, "CUSTOM_VOICES_DIR", str(store))
    engine = MixingEngine(kokoro_engine.KokoroModel())
    try:
        saved = MagicMock(side_effect=AssertionError("a live preview must not save a Mix"))
        monkeypatch.setattr(MixingEngine, "mix_voices", saved)
        engine.pipeline = fake_pipeline
        backend = qt_app.backend
        monkeypatch.setattr(backend, "_engine", engine)
        monkeypatch.setattr(backend, "run", _sync_run, raising=False)
        before = sorted(os.listdir(store))

        dock = qt_app.mixing_dock
        dock.live_check.setChecked(True)
        for value in (25, 50, 75):
            dock.ratio_slider.setValue(value)
            dock.ratio_slider.sliderReleased.emit()
        dock.preview_mix()

        assert sorted(os.listdir(store)) == before == []
        played = playback.play.call_args[0][0]
        assert played == dock._preview_path and os.path.getsize(played) > 0
        saved.assert_not_called()
    finally:
        engine.worker.stop()
