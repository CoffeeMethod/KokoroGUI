"""The contract every engine keeps (ENGINE_AGNOSTIC plan, C6), run over the
built-ins and a plugin loaded through the entry point, each on a fake model
so nothing downloads: Kokoro on `fake_pipeline`, Audio8 with
`generate_segment` patched, Dummy and the plugin as they are (the plugin's
adapter says how: `make_contract_engine()`).

A new engine passes these with one module (a model and an adapter) and,
when its model needs weights, a fake here or its own `make_contract_engine`.
"""
import numpy as np
import pytest

from kokoro_gui.engine.caching import segment_key
from kokoro_gui.engines import registry

ENGINES = ["audio8", "dummy", "kokoro", "toneclone"]


def _tone(n=2205):
    return (0.1 * np.sin(2 * np.pi * 220 * np.arange(n) / 44100)).astype(np.float32)


@pytest.fixture
def contract_backend(request, isolated_dirs, fake_pipeline, toneclone_plugin, monkeypatch, tmp_path):
    """The adapter for `request.param` on its fake model, plus a voice of
    its own kind saved in the isolated stores. Yields `(backend, voice)`."""
    engine_id = request.param
    if engine_id == "kokoro":
        from kokoro_engine import KokoroEngine

        engine = KokoroEngine()
    elif engine_id == "audio8":
        from kokoro_gui.engines import audio8_tts

        monkeypatch.setattr(audio8_tts, "AUDIO8_REFS_DIR", str(tmp_path / "audio8_refs"))
        engine = audio8_tts.Audio8Engine()
        monkeypatch.setattr(engine, "generate_segment", lambda *a, **k: _tone())
    elif engine_id == "dummy":
        from kokoro_gui.engines.dummy import DummyEngine

        engine = DummyEngine()
    else:
        engine = registry.get_factory(engine_id).make_contract_engine()
    backend = registry.get_engine(engine_id, engine=engine)

    voice = None
    if backend.voice_kind == "embedding":
        voice = "contract_mix"
        with open(f"{backend.voice_store.global_dir}/{voice}{backend.voice_store.extension}", "wb") as f:
            f.write(b"voice bytes")
    elif backend.voice_kind == "reference":
        import soundfile as sf

        voice = "contract_ref"
        wav = tmp_path / "ref.wav"
        sf.write(str(wav), _tone(), 44100)
        backend.voice_store.save_reference(voice, str(wav), "What the reference says.")
    else:
        voice = backend.get_voices(None)[0].id
    try:
        yield backend, voice
    finally:
        engine.worker.stop()


def test_every_registered_engine_is_covered(toneclone_plugin):
    assert set(registry.list_engines()) <= set(ENGINES)


@pytest.mark.parametrize("contract_backend", ENGINES, indirect=True)
def test_the_schema_has_a_language_and_a_voice(contract_backend):
    backend, _voice = contract_backend
    keys = {f.key for f in backend.get_config_schema()}
    assert {"lang_code", "voice"} <= keys
    assert backend.get_languages()
    assert registry.get_config_schema(backend.id) == backend.get_config_schema()


@pytest.mark.parametrize("contract_backend", ENGINES, indirect=True)
def test_synthesize_returns_mono_float32_at_the_models_rate(contract_backend):
    backend, voice = contract_backend
    engine = backend.engine
    lang_code = backend.get_languages()[0][1]
    synthesis = engine._synthesize("Hello there.", {"voice": engine.resolve_voice_path(voice), "speed": 1.0,
                                                    "lang_code": lang_code})
    audio = np.asarray(synthesis.audio)
    assert audio.dtype == np.float32 and audio.ndim == 1 and len(audio) > 0
    assert backend.sample_rate == engine.SAMPLE_RATE == engine.model.sample_rate
    assert isinstance(synthesis.words, list)


@pytest.mark.parametrize("contract_backend", ENGINES, indirect=True)
def test_project_assets_live_under_the_engines_own_folder(contract_backend, tmp_path):
    backend, voice = contract_backend
    assets, meta = backend.collect_project_assets({voice}, str(tmp_path / "project"))
    assert isinstance(meta, dict)
    assert all(a.bundle_path.startswith(f"engines/{backend.id}/") for a in assets)
    if backend.voice_kind != "named":
        assert assets, "a voice of the engine's own kind is bundled"


@pytest.mark.parametrize("contract_backend", ENGINES, indirect=True)
def test_the_segment_key_is_stable(contract_backend):
    backend, voice = contract_backend
    config = {"voice": voice, "speed": 1.0, "lang_code": backend.get_languages()[0][1]}
    first = segment_key("The same words.", config, backend)
    assert segment_key("The same words.", config, backend) == first
    assert segment_key("Other words.", config, backend) != first


@pytest.mark.parametrize("contract_backend", ENGINES, indirect=True)
def test_opening_a_project_loads_no_model(contract_backend, monkeypatch, tmp_path):
    backend, _voice = contract_backend

    def refuse(*_args, **_kwargs):
        raise AssertionError("on_project_opened loaded the model")

    monkeypatch.setattr(backend.engine.model, "load", refuse)
    backend.on_project_opened(str(tmp_path), {})
    assert backend.project_dir == str(tmp_path)
