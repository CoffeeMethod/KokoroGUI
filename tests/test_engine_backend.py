"""Tests for the engine backend abstraction (kokoro_gui/engines/) -
PLAN_qt_and_engine_abstraction.md workstream 1."""
import asyncio
from unittest.mock import MagicMock

import pytest

from kokoro_gui.engines import registry
from kokoro_gui.engines.base import ConfigField, EngineCapabilities, VoiceInfo
from kokoro_gui.engines.kokoro import KokoroBackendAdapter, OUTPUT_FORMAT_CHOICES, SPLIT_PATTERN_CHOICES


def test_kokoro_registered_by_default():
    assert "kokoro" in registry.list_engines()


def test_get_engine_wraps_the_given_engine_instance(engine):
    backend = registry.get_engine("kokoro", engine=engine)
    assert isinstance(backend, KokoroBackendAdapter)
    assert backend._engine is engine


def test_get_engine_unknown_id_raises():
    with pytest.raises(KeyError):
        registry.get_engine("does-not-exist")


def test_capabilities_reflect_kokoro_shape():
    caps = KokoroBackendAdapter.capabilities
    assert isinstance(caps, EngineCapabilities)
    assert caps.supports_voice_mixing is True
    assert caps.supports_multi_speaker_script is True
    assert caps.is_local_model is True
    assert caps.supports_jit_streaming is True


def test_config_schema_covers_todays_actual_fields(engine):
    backend = registry.get_engine("kokoro", engine=engine)
    schema = backend.get_config_schema()
    assert all(isinstance(f, ConfigField) for f in schema)

    keys = {f.key for f in schema}
    assert keys == {
        "lang_code", "voice", "speed", "pitch", "split_pattern",
        "format", "num_threads", "caching", "lexicon",
    }


def test_config_schema_split_pattern_and_format_choices(engine):
    backend = registry.get_engine("kokoro", engine=engine)
    by_key = {f.key: f for f in backend.get_config_schema()}

    assert by_key["split_pattern"].choices == SPLIT_PATTERN_CHOICES
    assert by_key["format"].choices == OUTPUT_FORMAT_CHOICES
    # voice/lang_code are GUI-resolved (dynamic), not schema-fixed.
    assert by_key["voice"].choices is None
    assert by_key["lang_code"].choices is None


def test_get_voices_empty_when_no_custom_voices(engine, isolated_dirs):
    backend = registry.get_engine("kokoro", engine=engine)
    assert backend.get_voices() == []


def test_get_voices_lists_custom_voice_files(engine, isolated_dirs):
    (isolated_dirs.custom_voices / "MyMix.pt").write_bytes(b"not a real tensor")

    backend = registry.get_engine("kokoro", engine=engine)
    voices = backend.get_voices()

    assert voices == [VoiceInfo(id="MyMix", display_name="MyMix", lang_code=None, is_custom=True)]


def test_mix_voices_delegates_to_wrapped_engine(engine, monkeypatch):
    backend = registry.get_engine("kokoro", engine=engine)

    async def fake_mix_voices(v1, v2, ratio, new_name, op="mix"):
        return (True, (v1, v2, ratio, new_name, op), None)

    monkeypatch.setattr(engine, "mix_voices", fake_mix_voices)

    ok, payload, _ = asyncio.run(backend.mix_voices("af_heart", "af_bella", 0.5, "blend", op="add"))
    assert ok is True
    assert payload == ("af_heart", "af_bella", 0.5, "blend", "add")


def test_cancel_delegates_to_wrapped_engine(engine):
    backend = registry.get_engine("kokoro", engine=engine)
    engine.cancel = MagicMock()

    backend.cancel()

    engine.cancel.assert_called_once_with()


def test_tts_app_wires_a_backend_and_shows_mixing_tab_when_capable(tts_app):
    """Mixing tab in create_widgets (gui.py) is gated on
    backend.capabilities.supports_voice_mixing - kokoro supports it, so the
    tab and its widgets (mixing_tab.py's build_mixing_tab) must still exist."""
    assert tts_app.backend.id == "kokoro"
    assert tts_app.backend.capabilities.supports_voice_mixing is True
    assert hasattr(tts_app, "mix_combo_a")
    assert hasattr(tts_app, "mix_combo_b")
