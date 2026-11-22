"""A project's own bundled voice resolves the same file for synthesis and for
the segment key on every generation path (plan 46, F3): clip generation, the
whole-document path, JIT and Preview. Covers Kokoro (`.pt` mixes) and Audio8
(reference wavs) with only a project-local voice, and with a project voice and
a global voice that share a name. Synthesis is stubbed; the test reads the
config the engine hands to it."""
import asyncio
import os

import numpy as np
import pytest
import soundfile as sf
import torch

from kokoro_gui.engine import caching
from kokoro_gui.engines import audio8_tts


def _kokoro_voice(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(torch.full((2,), float(value)), path)


def _audio8_voice(path, freq):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, (0.1 * np.sin(2 * np.pi * freq * np.arange(1600) / 16000)).astype(np.float32), 16000)


@pytest.fixture(params=["kokoro", "audio8"])
def rig(request, tmp_path, isolated_dirs, monkeypatch):
    """(engine, project_dir, put_project_voice, put_global_voice, project voice path, global voice path)."""
    project_dir = str(tmp_path / "proj")
    os.makedirs(project_dir)
    if request.param == "kokoro":
        from kokoro_engine import KokoroEngine

        engine = KokoroEngine()
        project_path = os.path.join(project_dir, "engines", "kokoro", "voices", "Eve.pt")
        global_path = os.path.join(str(isolated_dirs.custom_voices), "Eve.pt")
        put = _kokoro_voice
    else:
        monkeypatch.setattr(audio8_tts, "AUDIO8_REFS_DIR", str(tmp_path / "refs"))
        engine = audio8_tts.Audio8Engine()
        project_path = os.path.join(project_dir, "engines", "audio8", "refs", "Eve.wav")
        global_path = os.path.join(str(tmp_path / "refs"), "Eve.wav")
        put = _audio8_voice
    yield engine, project_dir, put, project_path, global_path
    engine.worker.stop()


def _config(project_dir, tmp_path):
    return {"voice": "Eve", "speed": 1.0, "lang_code": "a", "out_dir": str(tmp_path / "out"), "format": "wav",
            "filename": "f", "time_id": "t", "project_dir": project_dir, "caching": False,
            "num_threads": 1, "lexicon": {}}


def _capture_clip(engine, config):
    seen = []
    engine.process_chunk_task = lambda chunk, cb: seen.append(chunk[2]) or []
    asyncio.run(engine.generate_clip_audio((0, "Hello there.", config)))
    return seen[0]["voice"]


def _capture_whole(engine, config):
    seen = []

    async def fake(text, cfg):
        seen.append(dict(cfg))

    engine._process_text_async = fake
    engine.worker.run_coro = lambda coro: asyncio.run(coro)
    engine.start_conversion("Hello there.", config)
    return seen[0]["voice"]


def _capture_jit(engine, config):
    seen = []

    async def fake(text, cfg):
        seen.append(dict(cfg))

    engine._process_jit_async = fake
    engine.worker.run_coro = lambda coro: asyncio.run(coro)
    engine.start_jit_conversion("Hello there.", config)
    return seen[0]["voice"]


PATHS = {"clip": _capture_clip, "whole": _capture_whole, "jit": _capture_jit}


@pytest.mark.parametrize("path_name", sorted(PATHS))
def test_a_project_only_voice_resolves_on_every_generation_path(rig, tmp_path, path_name):
    engine, project_dir, put, project_path, _global_path = rig
    put(project_path, 1)

    voice = PATHS[path_name](engine, _config(project_dir, tmp_path))

    assert os.path.abspath(voice) == os.path.abspath(project_path)


@pytest.mark.parametrize("path_name", sorted(PATHS))
def test_a_project_voice_shadows_a_global_one_of_the_same_name(rig, tmp_path, path_name):
    engine, project_dir, put, project_path, global_path = rig
    put(project_path, 1)
    put(global_path, 7 if "wav" not in project_path else 440)

    voice = PATHS[path_name](engine, _config(project_dir, tmp_path))

    assert os.path.abspath(voice) == os.path.abspath(project_path)


def test_synthesis_input_and_expected_key_agree(rig, tmp_path):
    engine, project_dir, put, project_path, global_path = rig
    put(project_path, 1)
    put(global_path, 7 if "wav" not in project_path else 440)
    config = _config(project_dir, tmp_path)
    expected = caching.segment_key("Hello there.", config, engine, engine_version="v")

    resolved = _capture_clip(engine, dict(config))
    from_synthesis_input = caching.segment_key("Hello there.", {**config, "voice": resolved}, engine,
                                               engine_version="v")
    global_only = caching.segment_key("Hello there.", {**config, "project_dir": None}, engine, engine_version="v")

    assert from_synthesis_input == expected
    assert expected != global_only  # the collision is real: the global copy would key differently


def test_preview_resolves_the_project_voice(rig, tmp_path):
    engine, project_dir, put, project_path, global_path = rig
    put(project_path, 1)
    put(global_path, 7 if "wav" not in project_path else 440)
    seen = []

    class Synthesis:
        audio = np.zeros(2400, dtype=np.float32)

    engine._synthesize = lambda text, cfg, cancellable=True: seen.append(cfg["voice"]) or Synthesis()
    asyncio.run(engine.generate_preview("Hello there.", "Eve", 1.0, str(tmp_path / "p.wav"),
                                        extra_config={"project_dir": project_dir}))

    assert os.path.abspath(seen[0]) == os.path.abspath(project_path)
