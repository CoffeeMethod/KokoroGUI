"""Tests for _process_jit_async, the real-time/JIT pipeline
(kokoro_engine.py:747-908). caching=False throughout via make_config."""
import asyncio


def test_jit_conversion_writes_combined_jit_output_file(engine, fake_pipeline, make_config, isolated_dirs):
    config = make_config(filename="jitrun", time_id="1")
    asyncio.run(engine._process_jit_async("Hello world. This is JIT.", config))

    assert (isolated_dirs.out_dir / "jitrun_1_jit_output.wav").exists()


def test_jit_conversion_cancel_writes_remaining_txt(engine, fake_pipeline, make_config, isolated_dirs):
    config = make_config(filename="jitrun", time_id="1")
    engine.cancel_event.set()  # cancel before starting -> deterministic "nothing played" branch

    asyncio.run(engine._process_jit_async("Hello world. This is JIT playback text.", config))

    remaining = isolated_dirs.out_dir / "jitrun_1_remaining.txt"
    assert remaining.exists()
    content = remaining.read_text(encoding="utf-8")
    assert "Hello world" in content

    assert not (isolated_dirs.out_dir / "jitrun_1_jit_output.wav").exists()


def test_jit_conversion_calls_on_finish_even_when_no_text(engine, fake_pipeline, make_config, callback_recorder):
    config = make_config()
    asyncio.run(engine._process_jit_async("   ", config))

    assert callback_recorder.finished.is_set()


def test_jit_conversion_leaves_inspectable_output(engine, fake_pipeline, make_config, timestamped_output_dir):
    config = make_config(out_dir=str(timestamped_output_dir), filename="jitsmoke", time_id="1")
    asyncio.run(engine._process_jit_async("Hello from the JIT smoke test.", config))

    assert (timestamped_output_dir / "jitsmoke_1_jit_output.wav").exists()
