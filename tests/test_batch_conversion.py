"""Tests for process_chunk_task + _process_text_async, the batch conversion
pipeline (kokoro_engine.py:568-705, 910-1061). caching=False throughout
(via make_config's default) except where noted."""
import asyncio
import json

import pytest

import kokoro_engine
from kokoro_gui.engine import stats as generation_stats


def test_process_chunk_task_writes_named_part_files(engine, fake_pipeline, make_config, isolated_dirs):
    config = make_config(filename="myrun", time_id="20260101000000")
    results = engine.process_chunk_task((3, "Hello there.", config), None)

    assert len(results) == 1
    expected = isolated_dirs.out_dir / "myrun_20260101000000_part3_0.wav"
    assert expected.exists()
    assert results[0]["path"] == str(expected)


def test_process_text_async_combine_true_writes_combined_file(engine, fake_pipeline, make_config, isolated_dirs):
    config = make_config(combine=True, filename="run", time_id="1")
    asyncio.run(engine._process_text_async("Hello world.", config))

    assert (isolated_dirs.out_dir / "run_1_combined.wav").exists()


def test_process_text_async_export_subtitles_writes_srt(engine, fake_pipeline, make_config, isolated_dirs):
    config = make_config(export_subtitles=True, filename="run", time_id="1")
    asyncio.run(engine._process_text_async("Hello world.", config))

    srt_path = isolated_dirs.out_dir / "run_1_combined.srt"
    assert srt_path.exists()
    assert "-->" in srt_path.read_text(encoding="utf-8")


def test_process_text_async_separate_false_deletes_part_files(engine, fake_pipeline, make_config, isolated_dirs):
    config = make_config(separate=False, combine=True, filename="run", time_id="1")
    asyncio.run(engine._process_text_async("Hello world.", config))

    assert list(isolated_dirs.out_dir.glob("run_1_part*")) == []
    assert (isolated_dirs.out_dir / "run_1_combined.wav").exists()


def test_process_text_async_no_text_calls_on_finish_and_status(engine, fake_pipeline, make_config, callback_recorder):
    config = make_config()
    asyncio.run(engine._process_text_async("   ", config))

    assert callback_recorder.finished.is_set()
    assert any("No text" in msg for msg, _ in callback_recorder.statuses)


def test_process_text_async_chunk_exception_does_not_abort_batch(
    engine, fake_pipeline, make_config, isolated_dirs, monkeypatch, callback_recorder
):
    config = make_config(filename="run", time_id="1", num_threads=1)
    # Two multispeaker segments guarantee two chunks even with num_threads=1
    # (a single unmarked segment would be merged into one smart_split chunk).
    text = "[SpeakerA]: First paragraph.\n\n[SpeakerB]: Second paragraph."

    real_task = engine.process_chunk_task
    call_count = {"n": 0}

    def flaky(chunk_data, progress_callback):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("boom")
        return real_task(chunk_data, progress_callback)

    monkeypatch.setattr(engine, "process_chunk_task", flaky)
    asyncio.run(engine._process_text_async(text, config))

    assert any("Error in chunk" in msg for msg, is_err in callback_recorder.statuses if is_err)
    assert (isolated_dirs.out_dir / "run_1_combined.wav").exists()


def test_multispeaker_preset_and_fx_preset_layering(engine, fake_pipeline, make_config, isolated_dirs, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    presets_dir = tmp_path / "presets"
    fx_dir = presets_dir / "fx"
    presets_dir.mkdir(exist_ok=True)
    fx_dir.mkdir(exist_ok=True)

    (presets_dir / "Narrator.json").write_text(json.dumps({"voice": "am_adam", "speed": 1.25}), encoding="utf-8")
    (fx_dir / "Radio.json").write_text(json.dumps({"reverb_enabled": True, "reverb_room_size": 0.9}), encoding="utf-8")

    config = make_config(filename="run", time_id="1")
    text = "[Narrator:Radio]: Hello from the narrator."

    captured = {}
    real_task = engine.process_chunk_task

    def spy(chunk_data, progress_callback):
        captured["config"] = chunk_data[2]
        return real_task(chunk_data, progress_callback)

    monkeypatch.setattr(engine, "process_chunk_task", spy)
    asyncio.run(engine._process_text_async(text, config))

    assert captured["config"]["voice"] == "am_adam"
    assert captured["config"]["speed"] == 1.25
    assert captured["config"]["reverb_enabled"] is True
    assert captured["config"]["apply_fx"] is True


@pytest.mark.parametrize("fx_data,expected_ir", [
    ({"convolution_ir": "Hall", "convolution_mix": 0.7}, "Hall"),
    ({"convolution_ir": {"path": "/etc/passwd"}, "convolution_mix": "loud"}, None),
    ({"convolution_ir": ["Hall"], "reverb_room_size": "big"}, None),
], ids=["string", "dict", "list"])
def test_fx_preset_merge_accepts_only_a_string_impulse_response(engine, fake_pipeline, make_config, isolated_dirs,
                                                                 monkeypatch, tmp_path, fx_data, expected_ir):
    monkeypatch.chdir(tmp_path)
    fx_dir = tmp_path / "presets" / "fx"
    fx_dir.mkdir(parents=True, exist_ok=True)
    (fx_dir / "Room.json").write_text(json.dumps(fx_data), encoding="utf-8")

    captured = {}
    real_task = engine.process_chunk_task

    def spy(chunk_data, progress_callback):
        captured["config"] = chunk_data[2]
        return real_task(chunk_data, progress_callback)

    monkeypatch.setattr(engine, "process_chunk_task", spy)
    asyncio.run(engine._process_text_async("[Narrator:Room]: Hello.", make_config(filename="run", time_id="1")))

    assert captured["config"].get("convolution_ir") == expected_ir
    if expected_ir is None:
        assert not isinstance(captured["config"].get("convolution_mix"), str)
        assert not isinstance(captured["config"].get("reverb_room_size"), str)


def test_process_text_async_records_generation_stats_under_engine_id(engine, fake_pipeline, make_config, isolated_dirs):
    text = "Hello there, this is a short test sentence."
    config = make_config(engine_id="kokoro", filename="run", time_id="1")
    asyncio.run(engine._process_text_async(text, config))

    with open(kokoro_engine.STATS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    entry = data["kokoro"][-1]
    assert entry["chars"] == len(text)
    assert entry["words"] == len(text.split())
    assert entry["duration"] > 0


def test_process_text_async_keeps_stats_separate_per_engine_id(engine, fake_pipeline, make_config, isolated_dirs):
    text_a = "Text for engine one."
    text_b = "A distinctly longer piece of text used for engine two."
    asyncio.run(engine._process_text_async(text_a, make_config(engine_id="engine-a", time_id="1")))
    asyncio.run(engine._process_text_async(text_b, make_config(engine_id="engine-b", time_id="2")))

    with open(kokoro_engine.STATS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["engine-a"][-1]["chars"] == len(text_a)
    assert data["engine-b"][-1]["chars"] == len(text_b)


def test_process_text_async_seeds_initial_eta_from_history(engine, fake_pipeline, make_config, isolated_dirs, callback_recorder):
    # Prime history for this engine_id well before the run starts, so the
    # very first on_progress call (percent 0) already carries a real ETA
    # instead of "--:--".
    generation_stats.record_generation("kokoro", chars=1000, words=180, duration=10.0)

    config = make_config(engine_id="kokoro", filename="run", time_id="1")
    asyncio.run(engine._process_text_async("Hello there, this is a short test sentence.", config))

    first_percent, first_elapsed, first_eta, first_detail = callback_recorder.progresses[0]
    assert first_percent == 0
    assert first_elapsed == 0.0
    assert first_eta != "--:--"


def test_process_text_async_no_history_still_shows_no_initial_eta(engine, fake_pipeline, make_config, isolated_dirs, callback_recorder):
    config = make_config(engine_id="brand-new-engine", filename="run", time_id="1")
    asyncio.run(engine._process_text_async("Hello there, this is a short test sentence.", config))

    # Without history, the first callback is a real chunk-progress tick
    # (not the history-seeded pre-generation one), so it should carry actual
    # progress rather than the percent==0 placeholder.
    first_percent, *_ = callback_recorder.progresses[0]
    assert first_percent > 0


def test_full_batch_conversion_leaves_inspectable_output(engine, fake_pipeline, make_config, timestamped_output_dir):
    config = make_config(out_dir=str(timestamped_output_dir), filename="sample", time_id="smoke")
    asyncio.run(engine._process_text_async("This audio should be inspectable by a human.", config))

    combined = timestamped_output_dir / "sample_smoke_combined.wav"
    assert combined.exists()
    assert combined.stat().st_size > 0
