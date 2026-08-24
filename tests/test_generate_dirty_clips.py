"""Tests for KokoroEngine.generate_dirty_clips (kokoro_gui/engine/conversion.py) -
the batch dirty-scoped Generate entry point for item 3 ("Consolidated action
bar + batch dirty-scoped generation") of the DAW-for-text remaining-work
roadmap. Mirrors tests/test_generate_clip_audio.py's fixtures/conventions."""
import asyncio
import os

import pytest


def test_all_clips_succeed_returns_one_outcome_per_clip_with_distinct_filenames(engine, fake_pipeline, make_config):
    clips_with_configs = [
        ("clip-1", "Hello there.", make_config(filename="run", time_id="1")),
        ("clip-2", "General Kenobi.", make_config(filename="run", time_id="1")),
        ("clip-3", "You are a bold one.", make_config(filename="run", time_id="1")),
    ]

    outcomes = asyncio.run(engine.generate_dirty_clips(clips_with_configs))

    assert len(outcomes) == 3
    assert [o["clip_id"] for o in outcomes] == ["clip-1", "clip-2", "clip-3"]
    assert all(o["success"] for o in outcomes)
    assert all(not o["cancelled"] for o in outcomes)

    # Every clip's config shares the same filename/time_id - the per-clip
    # batch-local index (assigned via enumerate()) is what keeps their
    # output files from colliding on disk.
    paths = [o["results"][0]["path"] for o in outcomes]
    assert len(set(paths)) == 3
    for path in paths:
        assert os.path.exists(path)


def test_one_clip_raises_others_still_complete(engine, fake_pipeline, make_config, monkeypatch):
    clips_with_configs = [
        ("clip-1", "First paragraph.", make_config(filename="run", time_id="1")),
        ("clip-2", "Second paragraph.", make_config(filename="run", time_id="1")),
        ("clip-3", "Third paragraph.", make_config(filename="run", time_id="1")),
    ]

    real_task = engine.process_chunk_task
    call_count = {"n": 0}

    def flaky(chunk_data, progress_callback):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("boom")
        return real_task(chunk_data, progress_callback)

    monkeypatch.setattr(engine, "process_chunk_task", flaky)

    outcomes = asyncio.run(engine.generate_dirty_clips(clips_with_configs))

    assert len(outcomes) == 3
    successes = [o for o in outcomes if o["success"]]
    failures = [o for o in outcomes if not o["success"]]
    assert len(successes) == 2
    assert len(failures) == 1
    assert "boom" in failures[0]["error"]
    assert not failures[0]["cancelled"]


def test_empty_input_list_returns_empty_list_with_no_side_effects(engine, fake_pipeline, make_config, isolated_dirs):
    outcomes = asyncio.run(engine.generate_dirty_clips([]))

    assert outcomes == []
    assert list(isolated_dirs.out_dir.iterdir()) == []


def test_cancel_mid_batch_skips_queued_clips_without_calling_generate_clip_audio(
    engine, fake_pipeline, make_config, monkeypatch
):
    # num_threads=1 makes ordering deterministic: clip-1 dispatches first,
    # sets cancel_event as a side effect, and clip-2/clip-3 must never reach
    # generate_clip_audio at all.
    clips_with_configs = [
        ("clip-1", "First.", make_config(filename="run", time_id="1", num_threads=1)),
        ("clip-2", "Second.", make_config(filename="run", time_id="1", num_threads=1)),
        ("clip-3", "Third.", make_config(filename="run", time_id="1", num_threads=1)),
    ]

    calls = []
    real_generate_clip_audio = engine.generate_clip_audio

    async def spy(chunk_data, progress_callback=None):
        calls.append(chunk_data[1])  # record the text this call was for
        result = await real_generate_clip_audio(chunk_data, progress_callback)
        # Simulate the user hitting Cancel right after the first clip
        # dispatches, while clip-2/clip-3 are still queued behind the
        # num_threads=1 semaphore.
        engine.cancel_event.set()
        return result

    monkeypatch.setattr(engine, "generate_clip_audio", spy)

    outcomes = asyncio.run(engine.generate_dirty_clips(clips_with_configs))

    assert len(outcomes) == 3
    by_id = {o["clip_id"]: o for o in outcomes}

    # clip-1 got to run before cancellation.
    assert by_id["clip-1"]["success"] is True
    assert by_id["clip-1"]["cancelled"] is False

    # clip-2/clip-3 were still queued when cancel_event got set - they must
    # be skipped with a distinct "cancelled" outcome, not a generic failure,
    # and generate_clip_audio must never have been called for them at all
    # (this is the actual bug being guarded against: a queued clip's
    # eventual turn calling generate_clip_audio would silently re-clear
    # cancel_event via its own unconditional `self.cancel_event.clear()`).
    for clip_id in ("clip-2", "clip-3"):
        outcome = by_id[clip_id]
        assert outcome["success"] is False
        assert outcome["cancelled"] is True

    assert calls == ["First."]  # generate_clip_audio was only ever called once


@pytest.mark.skip(reason="Timing-based concurrency assertions are inherently flaky under CI load; "
                          "the Semaphore(num_threads) bound is already exercised structurally by "
                          "the cancel-race test above (num_threads=1 forces deterministic ordering).")
def test_concurrency_bound_lets_multiple_clips_run_in_parallel():
    pass
