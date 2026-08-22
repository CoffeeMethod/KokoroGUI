"""Opt-in integration tests using the REAL Kokoro pipeline (no mocking).

Skipped by default (see pytest.ini's `addopts = -m "not integration"`).
Run explicitly with:
    pytest -m integration tests/integration -s

(the `-s` shows the spoken sample text in the terminal as each test runs;
without it, pytest still writes a matching `*_transcript.txt` next to
each `.wav` for the same purpose.)

Requires the espeak-ng backend (via the `espeakng_loader` package, a
transitive dependency of kokoro/misaki that bundles its own espeak-ng.dll
and data dir - no system-wide espeak-ng install/PATH entry needed) plus
torch/kokoro model weights. Real audio is always written under the shared
timestamped_output_dir fixture so it persists for manual inspection (never
tmp_path, which pytest auto-cleans).

Since this is real synthesis, correctness can't be asserted automatically -
each test speaks a short, self-describing sample naming the voice and mode
in use, so a human listening to the output can actually confirm it sounds
right (clear pronunciation, correct voice, no glitches/silence).
"""
import asyncio
import os

import pytest

import kokoro_engine
from kokoro_engine import KokoroEngine


def _espeak_available():
    # kokoro_engine never shells out to an `espeak-ng` CLI - phonemization
    # goes through misaki -> phonemizer's EspeakWrapper, pointed at the DLL
    # and data dir that `espeakng_loader` bundles/resolves. That's the
    # actual runtime dependency, so check for it directly instead of
    # probing PATH for a binary the app doesn't use.
    try:
        import espeakng_loader
        return (
            os.path.isfile(espeakng_loader.get_library_path())
            and os.path.isdir(espeakng_loader.get_data_path())
        )
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _espeak_available(), reason="espeakng_loader library/data not found"),
]


def _sample_text(mode, voice):
    # A pangram + digits is a standard TTS smoke-test phrase: it exercises
    # every letter and reads unambiguously by ear, so a mispronunciation or
    # glitch is easy for a human to catch. Naming the mode/voice lets the
    # listener confirm the right code path actually produced this file.
    return (
        f"This is a Kokoro {mode} integration test, spoken by the {voice} voice. "
        "Please confirm this is clear and understandable: "
        "the quick brown fox jumps over the lazy dog. One, two, three, four, five."
    )


def _write_transcript(out_dir, stem, text):
    path = out_dir / f"{stem}_transcript.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _ensure_output_dir(out_dir):
    # timestamped_output_dir is created by the conftest fixture, but these
    # tests write real audio that's meant to persist for manual inspection,
    # so guard against it having been removed (or never created) out from
    # under us rather than failing deep inside the engine's file write.
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def test_real_batch_conversion_produces_playable_audio(real_engine, timestamped_output_dir, make_config):
    timestamped_output_dir = _ensure_output_dir(timestamped_output_dir)
    success = asyncio.run(real_engine.init_pipeline_async(lang_code="a"))
    assert success, "Real pipeline failed to init - check espeak-ng/kokoro install"

    voice = "af_heart"
    text = _sample_text("batch conversion", voice)
    config = make_config(
        out_dir=str(timestamped_output_dir), voice=voice,
        filename="realbatch", time_id="1",
    )
    asyncio.run(real_engine._process_text_async(text, config))

    combined = timestamped_output_dir / "realbatch_1_combined.wav"
    assert combined.exists()
    assert combined.stat().st_size > 1000

    transcript = _write_transcript(timestamped_output_dir, "realbatch_1", text)
    print(f"\n[HUMAN CONFIRMATION NEEDED] Listen to {combined}")
    print(f"It should say: {text}")
    print(f"(transcript saved to {transcript})")


def test_real_jit_conversion_with_playback_mocked(real_engine, timestamped_output_dir, make_config):
    timestamped_output_dir = _ensure_output_dir(timestamped_output_dir)
    success = asyncio.run(real_engine.init_pipeline_async(lang_code="a"))
    assert success, "Real pipeline failed to init - check espeak-ng/kokoro install"

    voice = "af_heart"
    text = _sample_text("real-time JIT", voice)
    config = make_config(
        out_dir=str(timestamped_output_dir), voice=voice,
        filename="realjit", time_id="1",
    )
    asyncio.run(real_engine._process_jit_async(text, config))

    jit_output = timestamped_output_dir / "realjit_1_jit_output.wav"
    assert jit_output.exists()
    assert jit_output.stat().st_size > 1000

    transcript = _write_transcript(timestamped_output_dir, "realjit_1", text)
    print(f"\n[HUMAN CONFIRMATION NEEDED] Listen to {jit_output}")
    print(f"It should say: {text}")
    print(f"(transcript saved to {transcript})")
