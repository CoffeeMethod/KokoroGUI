"""LuxTTS API, reference and cache behavior without model downloads."""
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
import torch

from kokoro_gui.engine.caching import segment_key
from kokoro_gui.engine.runner import EngineRunner
from kokoro_gui.engines import luxtts
from kokoro_gui.engines.voice_store import ReferenceStore


@pytest.fixture
def lux(tmp_path, isolated_dirs):
    wav = tmp_path / "ref.wav"
    sf.write(str(wav), np.full(3 * 16000, 0.1, dtype=np.float32), 16000)
    fake = SimpleNamespace(encode_prompt=MagicMock(return_value={"encoded": True}),
                           generate_speech=MagicMock(return_value=torch.ones(1, 4800)))
    loader = MagicMock(return_value=fake)
    model = luxtts.LuxTTSModel(loader=loader)
    return model, fake, loader, str(wav)


def test_reference_encoding_reused_and_generation_settings_forwarded(lux):
    model, fake, loader, wav = lux
    params = {"num_steps": 7, "guidance_scale": 2.0, "t_shift": 0.7, "return_smooth": True}
    result = model.synthesize("Hello.", wav, 1.2, "en", params)
    model.synthesize("Next segment.", wav, 1.0, "en", params)
    assert result.audio.dtype == np.float32 and result.audio.shape == (4800,)
    assert result.words == [] and model.sample_rate == 48000
    loader.assert_called_once_with("cpu")
    fake.encode_prompt.assert_called_once_with(wav, duration=5.0, rms=0.01)
    assert fake.generate_speech.call_args_list[0].kwargs == {
        "speed": 1.2, "num_steps": 7, "guidance_scale": 2.0, "t_shift": 0.7, "return_smooth": True,
    }


def test_reference_cache_tracks_audio_and_encoding_settings(lux):
    model, fake, _, wav = lux
    model.synthesize("Hello.", wav, 1, "en", {})
    model.synthesize("Hello.", wav, 1, "en", {"ref_rms": 0.02})
    model.synthesize("Hello.", wav, 1, "en", {"ref_duration": 3})
    sf.write(wav, np.full(4 * 16000, 0.2, dtype=np.float32), 16000)
    model.synthesize("Hello.", wav, 1, "en", {})
    assert fake.encode_prompt.call_count == 4


def test_device_reload_clears_prompts(lux):
    model, fake, loader, wav = lux
    model.load("en", "cpu")
    model.synthesize("Hello.", wav, 1, "en", {})
    model.load("en", "cpu")
    assert loader.call_count == 1
    model.load("en", "cuda")
    assert not model._prompts
    model.synthesize("Hello.", wav, 1, "en", {})
    assert fake.encode_prompt.call_count == 2
    assert loader.call_args.args == ("cuda",)


def test_auto_device_and_invalid_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert luxtts._device("auto") == "cuda"
    assert luxtts._device("cpu") == "cpu"
    with pytest.raises(ValueError, match="Unsupported"):
        luxtts._device("invalid")


def test_cancel_before_and_after_encoding(lux):
    model, fake, loader, wav = lux
    cancel = threading.Event()
    cancel.set()
    assert model.synthesize("Hello.", wav, 1, "en", {"cancel_event": cancel}).audio.size == 0
    loader.assert_not_called()
    cancel.clear()
    fake.encode_prompt.side_effect = lambda *a, **k: cancel.set() or {}
    assert model.synthesize("Hello.", wav, 1, "en", {"cancel_event": cancel}).audio.size == 0
    fake.generate_speech.assert_not_called()


@pytest.mark.parametrize("params", [{"num_steps": 0}, {"num_steps": 2.5}, {"t_shift": float("nan")},
                                  {"ref_rms": 0}, {"ref_duration": 2}])
def test_invalid_parameters_fail_before_loading(lux, params):
    model, _, loader, wav = lux
    with pytest.raises(ValueError, match="LuxTTS"):
        model.synthesize("Hello.", wav, 1, "en", params)
    loader.assert_not_called()


def test_missing_and_short_reference(lux, tmp_path):
    model, _, loader, _ = lux
    with pytest.raises(ValueError, match="save a LuxTTS"):
        model.synthesize("Hello.", None, 1, "en", {})
    wav = tmp_path / "short.wav"
    sf.write(str(wav), np.zeros(1600), 16000)
    with pytest.raises(ValueError, match="3 seconds"):
        model.synthesize("Hello.", str(wav), 1, "en", {})
    loader.assert_not_called()


def test_segment_key_includes_every_output_setting_and_reference(lux):
    model, _, _, wav = lux
    runner = EngineRunner(model)
    backend = luxtts.LuxTTSBackendAdapter(engine=runner)
    try:
        config = {"voice": wav, "lang_code": "en"}
        first = segment_key("Hello.", config, backend)
        changes = {"num_steps": 8, "guidance_scale": 2, "t_shift": 0.5, "ref_duration": 3,
                   "ref_rms": 0.02, "return_smooth": True}
        for key, value in changes.items():
            assert segment_key("Hello.", {**config, key: value}, backend) != first
        sf.write(wav, np.full(4 * 16000, 0.2), 16000)
        assert segment_key("Hello.", config, backend) != first
    finally:
        runner.worker.stop()


def test_audio_only_store_and_project_precedence(lux, tmp_path):
    model, _, _, wav = lux
    store = luxtts.LuxTTSReferenceStore
    store.save_reference("Voice", wav, "")
    # Saving an existing reference to itself is valid.
    store.save_reference("Voice", store.find_wav("Voice"), "")
    assert store.list_references() == ["Voice"]
    project = tmp_path / "project"
    refs = project / "engines" / "luxtts" / "refs"
    refs.mkdir(parents=True)
    sf.write(str(refs / "Voice.wav"), np.zeros(3 * 16000), 16000)
    assert store.find_wav("Voice", str(project)) == str(refs / "Voice.wav")
    assert store.list_references(str(project)) == ["Voice"]
    backend = luxtts.LuxTTSBackendAdapter(engine=SimpleNamespace())
    assets, _ = backend.collect_project_assets({"Voice"}, str(project))
    assert assets[0].source_path == str(refs / "Voice.wav")
    assert assets[0].bundle_path == "engines/luxtts/refs/Voice.wav"
    assert model.resolve_voice_path("Voice", str(project)) == str(refs / "Voice.wav")
    audio8 = ReferenceStore("audio8", global_dir=lambda: str(refs))
    assert audio8.list_references() == []  # Still needs its transcript sidecar.


def test_missing_dependency_error(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "zipvoice.luxvoice", None)
    with pytest.raises(RuntimeError, match="requirements-luxtts.txt"):
        luxtts._load_lux("cpu")


def test_preview_and_clip_use_project_reference_and_model_settings(lux, tmp_path, make_config):
    import asyncio

    model, fake, _, wav = lux
    luxtts.LuxTTSReferenceStore.save_reference("Voice", wav, "")
    project = tmp_path / "project"
    refs = project / "engines" / "luxtts" / "refs"
    refs.mkdir(parents=True)
    local_wav = str(refs / "Voice.wav")
    sf.write(local_wav, np.full(3 * 16000, 0.2), 16000)
    runner = EngineRunner(model)
    try:
        config = make_config(voice="Voice", project_dir=str(project), lang_code="en", num_steps=8)
        assert asyncio.run(runner.generate_preview("Hello.", "Voice", 1, str(tmp_path / "preview.wav"), config,
                                                   lang_code="en"))
        fake.encode_prompt.assert_called_once_with(local_wav, duration=5.0, rms=0.01)
        assert fake.generate_speech.call_args.kwargs["num_steps"] == 8
        result = asyncio.run(runner.generate_clip_audio((0, "Hello.", config)))
        assert result
        assert fake.generate_speech.call_count == 2
        assert fake.encode_prompt.call_count == 1
        assert sf.info(str(tmp_path / "preview.wav")).samplerate == 48000
    finally:
        runner.worker.stop()


def test_relative_paths_do_not_escape_voice_store(lux):
    model, _, _, _ = lux
    assert model.resolve_voice_path("../outside.wav") == "outside.wav"


def test_edited_transcript_reencodes_prompt_and_changes_segment_key(lux, monkeypatch):
    from pathlib import Path

    model, fake, _, wav = lux
    encoder = MagicMock(return_value={"edited": True})
    monkeypatch.setattr(luxtts, "_encode_prompt", encoder)
    config = {"voice": wav, "lang_code": "en"}
    backend = SimpleNamespace(id="luxtts", resolve_voice_file=model.resolve_voice_path,
                              cache_key_extra=model.cache_key_extra, engine_version=lambda: "test")
    first = segment_key("Hello.", config, backend)
    model.synthesize("Hello.", wav, 1, "en", {})
    Path(wav).with_suffix(".txt").write_text("Reviewed reference words.", encoding="utf-8")
    assert segment_key("Hello.", config, backend) != first
    model.synthesize("Hello.", wav, 1, "en", {})
    assert encoder.call_count == 2
    assert encoder.call_args.args[-1] == "Reviewed reference words."


def test_reviewed_text_uses_upstream_audio_processing_without_asr(monkeypatch):
    import sys

    process = MagicMock(side_effect=lambda path, transcriber, *args, **kwargs:
                        (transcriber(None)["text"], 1, 2, 3))
    monkeypatch.setitem(sys.modules, "zipvoice.modeling_utils", SimpleNamespace(process_audio=process))
    lux = SimpleNamespace(tokenizer=object(), feature_extractor=object(), device="cpu")
    result = luxtts._encode_prompt(lux, "ref.wav", luxtts.DEFAULTS, "Corrected words.")
    assert result == {"prompt_tokens": "Corrected words.", "prompt_features_lens": 1,
                      "prompt_features": 2, "prompt_rms": 3}
    assert process.call_args.kwargs == {"target_rms": 0.01, "duration": 5.0}


def test_project_transcript_wins_over_global(lux, tmp_path):
    model, _, _, wav = lux
    store = luxtts.LuxTTSReferenceStore
    store.save_reference("Voice", wav, "Global words.")
    refs = tmp_path / "project" / "engines" / "luxtts" / "refs"
    refs.mkdir(parents=True)
    sf.write(str(refs / "Voice.wav"), np.zeros(48000), 16000)
    (refs / "Voice.txt").write_text("Project words.", encoding="utf-8")
    assert model.cache_key_extra({"voice": "Voice", "project_dir": str(tmp_path / "project")})[
        "ref_transcript"] == "Project words."
