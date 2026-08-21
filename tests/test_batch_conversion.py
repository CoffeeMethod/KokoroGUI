"""Tests for process_chunk_task + _process_text_async, the batch conversion
pipeline (kokoro_engine.py:568-705, 910-1061). caching=False throughout
(via make_config's default) except where noted."""
import asyncio
import json


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


def test_full_batch_conversion_leaves_inspectable_output(engine, fake_pipeline, make_config, timestamped_output_dir):
    config = make_config(out_dir=str(timestamped_output_dir), filename="sample", time_id="smoke")
    asyncio.run(engine._process_text_async("This audio should be inspectable by a human.", config))

    combined = timestamped_output_dir / "sample_smoke_combined.wav"
    assert combined.exists()
    assert combined.stat().st_size > 0
