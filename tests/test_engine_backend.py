"""Tests for the engine backend abstraction (kokoro_gui/engines/) -
PLAN_qt_and_engine_abstraction.md workstream 1."""
import asyncio
from unittest.mock import MagicMock

import pytest

from kokoro_gui.engines import registry
from kokoro_gui.engines.base import ConfigField, EngineCapabilities, VoiceInfo
from kokoro_gui.engines.dummy import DummyBackendAdapter, DummyEngine
from kokoro_gui.engines.kokoro import LANGUAGES, VOICE_DB, KokoroBackendAdapter, OUTPUT_FORMAT_CHOICES


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
        "lang_code", "voice", "speed", "pitch", "segment_target_words", "segment_at_paragraphs", "segment_at_sentences", "segment_at_pauses",
        "format", "num_threads", "caching", "lexicon",
    }


def test_config_schema_segmentation_and_format_fields(engine):
    backend = registry.get_engine("kokoro", engine=engine)
    by_key = {f.key: f for f in backend.get_config_schema()}

    assert by_key["segment_target_words"].default == 40
    assert all(by_key[k].default is True for k in ("segment_at_paragraphs", "segment_at_sentences",
                                                    "segment_at_pauses"))
    assert by_key["format"].choices == OUTPUT_FORMAT_CHOICES
    # The voice list depends on the language and the disk; the languages don't.
    assert by_key["voice"].choices is None
    assert by_key["lang_code"].choices == LANGUAGES


def test_kokoro_languages_come_from_the_adapter(engine):
    backend = registry.get_engine("kokoro", engine=engine)
    assert backend.get_languages() == LANGUAGES
    assert ("British English", "b") in backend.get_languages()


def test_kokoro_lists_the_builtin_voices_of_a_language_first(engine, isolated_dirs):
    backend = registry.get_engine("kokoro", engine=engine)
    voices = backend.get_voices("b")
    assert [v.id for v in voices] == VOICE_DB["b"]
    assert all(not v.is_custom and v.lang_code == "b" for v in voices)
    assert len(backend.get_voices()) == sum(len(names) for names in VOICE_DB.values())


def test_get_voices_has_no_custom_voices_without_files(engine, isolated_dirs):
    backend = registry.get_engine("kokoro", engine=engine)
    assert [v for v in backend.get_voices("a") if v.is_custom] == []


def test_get_voices_lists_custom_voice_files_after_the_builtins(engine, isolated_dirs):
    (isolated_dirs.custom_voices / "MyMix.pt").write_bytes(b"not a real tensor")

    backend = registry.get_engine("kokoro", engine=engine)
    voices = backend.get_voices("a")

    assert voices[-1] == VoiceInfo(id="MyMix", display_name="MyMix", lang_code=None, is_custom=True)
    assert [v.id for v in voices[:-1]] == VOICE_DB["a"]


def test_schemas_are_readable_without_building_a_backend():
    assert registry.get_config_schema("kokoro") == KokoroBackendAdapter.get_config_schema()
    assert {f.key for f in registry.get_config_schema("audio8")} >= {"lang_code", "temperature"}
    assert registry.get_config_schema("does-not-exist") == []


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


def test_dummy_registered_and_shaped_like_a_real_backend():
    assert "dummy" in registry.list_engines()
    assert DummyBackendAdapter.capabilities.supports_voice_mixing is False

    backend = registry.get_engine("dummy")
    keys = {f.key for f in backend.get_config_schema()}
    assert keys == {
        "lang_code", "voice", "speed", "pitch", "segment_target_words", "segment_at_paragraphs", "segment_at_sentences", "segment_at_pauses",
        "format", "num_threads", "caching",
    }
    assert backend.get_voices() == [VoiceInfo(id="dummy", display_name="Dummy Tone", lang_code=None, is_custom=False)]
    assert backend.get_languages() == LANGUAGES


def test_dummy_engine_produces_real_nonsilent_audio(tmp_path):
    """Sanity check that DummyEngine's fake pipeline actually writes audible
    (non-silent) audio through the same process_chunk_task shape as
    CachingMixin, exercising the generic FX/write path with no cache."""
    import numpy as np
    import soundfile as sf

    engine = DummyEngine()
    try:
        config = {
            "lang_code": "a", "voice": "dummy", "speed": 1.0,
            "filename": "out", "time_id": "1", "out_dir": str(tmp_path), "format": "wav",
            "apply_fx": False,
        }
        files = engine.process_chunk_task((0, "Hello there.", config), None)
        assert len(files) == 1
        data, sr = sf.read(files[0]["path"])
        assert sr == 24000
        assert np.max(np.abs(data)) > 0.01
    finally:
        engine.worker.stop()


# Engine-picker switch behavior (switch to dummy, mixing-dock visibility,
# refuse-while-job-running, etc.) is covered by
# tests/gui_qt/test_qt_engine_backend.py now that the Tk frontend has been
# retired - see PLAN_qt_and_engine_abstraction.md.


def test_the_other_engines_load_without_the_kokoro_package():
    """ENGINE_AGNOSTIC C1: with `kokoro` unimportable, Dummy and Audio8
    register and Kokoro is marked unavailable instead of failing the import."""
    import subprocess
    import sys
    from pathlib import Path

    from tests.conftest import strip_ansi

    code = (
        "import sys; sys.modules['kokoro'] = None; "
        "from kokoro_gui.engines import registry; "
        "print(registry.list_engines(), registry.list_all_engines(), "
        "'kokoro' in (registry.unavailable_reason('kokoro') or ''))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            cwd=str(Path(__file__).resolve().parent.parent))
    assert result.returncode == 0, result.stderr
    assert strip_ansi(result.stdout).strip() == "['audio8', 'dummy'] ['audio8', 'dummy', 'kokoro'] True"


def test_the_engine_package_imports_without_kokoro():
    import subprocess
    import sys
    from pathlib import Path

    code = "import kokoro_gui.engine, sys; print('kokoro' in sys.modules, 'kokoro_engine' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            cwd=str(Path(__file__).resolve().parent.parent))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("False False")


def test_an_engine_that_fails_to_load_is_named_but_not_listed():
    registry.mark_unavailable("ghost-engine", "Ghost TTS", "ImportError: No module named 'ghost'")
    try:
        assert "ghost-engine" not in registry.list_engines()
        assert "ghost-engine" in registry.list_all_engines()
        assert registry.get_display_name("ghost-engine") == "Ghost TTS"
        assert "ghost" in registry.unavailable_reason("ghost-engine")
    finally:
        registry.unregister_engine("ghost-engine")
    assert registry.unavailable_reason("ghost-engine") is None


def test_a_shared_model_is_called_one_at_a_time():
    """`concurrency = "shared"` (Audio8): the runner serializes synthesize
    calls however many worker threads a batch uses."""
    import threading
    import time

    import numpy as np

    from kokoro_gui.engine.runner import EngineRunner, ModelBase
    from kokoro_gui.engines.base import Synthesis

    class Slow(ModelBase):
        engine_id = "slow"
        concurrency = "shared"

        def __init__(self):
            self.active = 0
            self.peak = 0
            self.guard = threading.Lock()

        def synthesize(self, text, voice, speed, lang_code, params):
            with self.guard:
                self.active += 1
                self.peak = max(self.peak, self.active)
            time.sleep(0.02)
            with self.guard:
                self.active -= 1
            return Synthesis(np.ones(10, dtype=np.float32), [])

    runner = EngineRunner(Slow())
    try:
        threads = [threading.Thread(target=runner._synthesize, args=("hi", {"voice": "v"})) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert runner.model.peak == 1
    finally:
        runner.worker.stop()


def test_a_missing_engine_says_it_isnt_installed():
    from kokoro_gui.engines.missing import MissingBackend, MissingEngineError

    registry.mark_unavailable("gone", "Gone TTS", "ImportError: No module named 'gone'")
    try:
        backend = MissingBackend("gone", registry.unavailable_reason("gone"))
        assert backend.message == "Gone TTS isn't installed"
        assert backend.display_name == "Gone TTS (not installed)"
        assert backend.get_voices() == [] and backend.engine_version() is None
        assert not backend.is_ready()
        with pytest.raises(MissingEngineError):
            backend.generate_clip((0, "hello", {})).result()
    finally:
        registry.unregister_engine("gone")
