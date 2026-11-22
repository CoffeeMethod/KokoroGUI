"""kokoro_gui/core/config.py: the generation and clip configs, built from
plain inputs with no Settings tab."""
import os

import pytest

from kokoro_gui.core import config as core_config
from kokoro_gui.core.config import GenerationInputs, OutputInputs
from kokoro_gui.core.engines import EngineSet, engine_settings
from kokoro_gui.core.fx import FxResolution
from kokoro_gui.daw.models import Character, Clip, Document
from kokoro_gui.engines.base import ConfigField, ConfigFieldType, EngineCapabilities
from kokoro_gui.qt import project as project_io, spec


class ModelBackend:
    """A backend with a "Model" schema group, like Audio8's sampling knobs."""
    id = "modelish"
    capabilities = EngineCapabilities()

    @classmethod
    def get_config_schema(cls):
        return [ConfigField("temperature", "Temperature", ConfigFieldType.FLOAT, default=0.7, group="Model"),
                ConfigField("voice", "Voice", ConfigFieldType.CHOICE, default="a", group="Generation")]


@pytest.fixture
def engines(isolated_dirs):
    engine_set = EngineSet("dummy", "dummy")
    yield engine_set
    for backend in engine_set.resident.values():
        backend.stop()


def _scene(preset_data=None, overrides=None, backend_id="dummy", variants=None):
    character = Character(name="Sam", preset_data=dict(preset_data or {}), backend_id=backend_id,
                          variants=dict(variants or {}))
    clip = Clip(character_id=character.id, overrides=dict(overrides or {}))
    return Document(runs=[], clips=[clip], characters=[character]), clip


def test_defaults_come_from_settings_defaults():
    inputs = GenerationInputs.defaults()
    defaults = spec.SETTINGS_DEFAULTS
    assert (inputs.speed, inputs.pitch, inputs.volume) == (defaults["speed"], defaults["pitch"], defaults["volume"])
    assert (inputs.normalize, inputs.trim_silence, inputs.caching) == (False, False, True)
    assert inputs.segmentation == {key: defaults[key] for key in spec.SEGMENTATION_KEYS}
    output = OutputInputs.defaults()
    assert (output.format, output.out_dir, output.filename) == ("wav", "audio_output", "output")


def test_segmentation_config_fills_missing_keys():
    config = core_config.segmentation_config({"segment_target_words": 15, "unrelated": 1})
    assert config["segment_target_words"] == 15
    assert set(config) == set(spec.SEGMENTATION_KEYS)


def test_character_values_override_the_defaults(engines):
    document, clip = _scene({"voice": "custom", "speed": 1.3, "pitch": 2.0})
    backend = engines.get("dummy")
    config = core_config.generation_config(GenerationInputs.defaults(), engine_settings({}, "dummy"),
                                           document, clip, backend, None)
    assert (config["voice"], config["speed"], config["pitch"]) == ("custom", 1.3, 2.0)
    assert config["engine_id"] == "dummy"
    assert config["lang_code"] == "a"


def test_clip_overrides_win_over_the_character(engines):
    document, clip = _scene({"speed": 1.3}, {"speed": 0.8})
    config = core_config.generation_config(GenerationInputs.defaults(), engine_settings({}, "dummy"),
                                           document, clip, engines.get("dummy"), None)
    assert config["speed"] == 0.8


def test_variant_applies_only_on_a_cloning_engine(engines, toneclone_plugin):
    document, clip = _scene({"voice": "base"}, {"variant": "angry"}, backend_id="toneclone",
                            variants={"angry": "base_angry"})
    clone = engines.get("toneclone")
    assert core_config.variant_voice(document, clip, clone) == "base_angry"
    config = core_config.generation_config(GenerationInputs.defaults(), engine_settings({}, "toneclone"),
                                           document, clip, clone, None)
    assert config["voice"] == "base_angry"

    dummy = engines.get("dummy")
    assert core_config.variant_voice(document, clip, dummy) is None
    config = core_config.generation_config(GenerationInputs.defaults(), engine_settings({}, "dummy"),
                                           document, clip, dummy, None)
    assert config["voice"] == "base"


def test_model_group_fields_come_from_engine_settings():
    document, clip = _scene(backend_id="modelish")
    inputs = GenerationInputs.defaults()
    config = core_config.generation_config(inputs, {}, document, clip, ModelBackend(), None)
    assert config["temperature"] == 0.7
    config = core_config.generation_config(inputs, {"temperature": 0.2}, document, clip, ModelBackend(), None)
    assert config["temperature"] == 0.2


def test_take_is_read_off_the_overrides(engines):
    document, clip = _scene(overrides={"take": 3})
    config = core_config.generation_config(GenerationInputs.defaults(), engine_settings({}, "dummy"),
                                           document, clip, engines.get("dummy"), None)
    assert config["take"] == 3
    document, clip = _scene()
    config = core_config.generation_config(GenerationInputs.defaults(), engine_settings({}, "dummy"),
                                           document, clip, engines.get("dummy"), None)
    assert config["take"] == 0


def test_lexicon_is_normalized():
    document, clip = _scene(backend_id="modelish")
    inputs = GenerationInputs.defaults()
    assert core_config.generation_config(inputs, {}, document, clip, ModelBackend(), None)["lexicon"] == []


def _clip_config(document, clip, backend, project_dir, fx=None):
    return core_config.clip_config(GenerationInputs.defaults(), OutputInputs.defaults(),
                                   engine_settings({}, backend.id), document, clip, backend, project_dir,
                                   {}, fx or FxResolution())


def test_segment_naming_and_out_dir_only_with_a_project_dir(engines, tmp_path):
    document, clip = _scene()
    backend = engines.get("dummy")
    without = _clip_config(document, clip, backend, None)
    assert "segment_naming" not in without
    assert without["out_dir"] == "audio_output"
    assert without["project_dir"] is None

    project_dir = str(tmp_path / "proj")
    with_dir = _clip_config(document, clip, backend, project_dir)
    assert with_dir["segment_naming"] == "cache_key"
    assert with_dir["out_dir"] == os.path.join(project_dir, *project_io.AUDIO_GENERATED.split("/"))
    assert with_dir["format"] == project_io.bundle_options({})["audio_format"]
    assert with_dir["project_dir"] == project_dir


def test_clip_config_carries_fx_trim_rename_and_stretch(engines):
    document, clip = _scene({"trim": True}, {"time_stretch": 1.1})
    fx = FxResolution(values={"eq_bass": 3.0}, apply_fx=False)
    config = _clip_config(document, clip, engines.get("dummy"), None, fx)
    assert config["trim_silence"] is True and "trim" not in config
    assert config["eq_bass"] == 3.0
    assert config["apply_fx"] is False
    assert config["time_stretch"] == 1.1
    assert config["voice"] == "dummy"


def test_post_config_is_the_post_subset(engines):
    document, clip = _scene()
    config = _clip_config(document, clip, engines.get("dummy"), None, FxResolution(values={"eq_bass": 3.0}))
    post_config = core_config.post_config(config)
    assert post_config["eq_bass"] == 3.0
    assert "voice" not in post_config
