"""kokoro_gui/core/engines.py: per-engine settings, default voices, new
characters and the resident engine set, with no Qt."""
import pytest

from kokoro_gui.core import engines as core_engines
from kokoro_gui.core.engines import EngineSet
from kokoro_gui.daw.models import Character, Clip, Document
from kokoro_gui.engines.missing import MissingBackend


@pytest.fixture
def engine_set(isolated_dirs):
    built = []
    engines = EngineSet("dummy", "dummy", on_new_backend=built.append)
    engines.built = built
    yield engines
    for backend in engines.resident.values():
        backend.stop()


def test_engine_settings_are_schema_defaults_without_stored_values():
    assert core_engines.engine_settings({}, "dummy") == {"lang_code": "a", "voice": "dummy", "num_threads": 1}


def test_engine_settings_overlay_stored_values_and_drop_unknown_keys():
    settings = {"engines": {"dummy": {"voice": "other", "num_threads": 4, "gone": 1}}}
    assert core_engines.engine_settings(settings, "dummy") == {"lang_code": "a", "voice": "other", "num_threads": 4}


def test_engine_settings_memo_keeps_the_field_list():
    memo = {}
    core_engines.engine_settings({}, "dummy", memo)
    assert ("per_engine", "dummy") in memo
    memo[("per_engine", "dummy")] = ()
    assert core_engines.engine_settings({}, "dummy", memo) == {}


def test_engine_settings_of_a_cloning_engine(toneclone_plugin):
    assert core_engines.engine_settings({}, "toneclone") == {"lang_code": "en", "voice": None, "num_threads": 1}


def test_default_voice_is_the_stored_voice_when_listed(engine_set):
    backend = engine_set.get("dummy")
    assert core_engines.default_voice(backend, {"lang_code": "a", "voice": "dummy"}) == "dummy"


def test_a_stored_voice_not_in_the_list_falls_back_to_the_schema_default(engine_set):
    backend = engine_set.get("dummy")
    assert core_engines.default_voice(backend, {"lang_code": "a", "voice": "gone"}) == "dummy"


def test_voice_names_lists_each_voice_once(engine_set):
    assert core_engines.voice_names(engine_set.get("dummy"), "a") == ["dummy"]


def test_engine_set_builds_once_and_calls_back_once(engine_set):
    first = engine_set.get("dummy")
    assert engine_set.get("dummy") is first
    assert engine_set.get(None) is first  # no id: the primary engine
    assert engine_set.built == [first]
    assert engine_set.resident == {"dummy": first}


def test_engine_set_unknown_id_is_none_and_a_character_gets_a_missing_backend(engine_set):
    assert engine_set.get("no-such-engine") is None
    character = Character(name="Ghost", backend_id="no-such-engine")
    backend = engine_set.for_character(character)
    assert isinstance(backend, MissingBackend)
    assert engine_set.for_character(character) is backend
    assert "no-such-engine" not in engine_set.resident


def test_engine_set_for_clip_follows_the_character_and_defaults(engine_set):
    character = Character(name="Narrator", backend_id="dummy")
    clip = Clip(character_id=character.id)
    document = Document(runs=[], clips=[clip], characters=[character])
    assert engine_set.for_clip(document, clip) is engine_set.get("dummy")
    assert engine_set.for_clip(document, None) is engine_set.get("dummy")


def test_engine_set_default_engine_id_may_be_a_callable(isolated_dirs):
    current = {"id": "dummy"}
    engines = EngineSet(lambda: current["id"], "dummy")
    try:
        assert engines.default_engine_id == "dummy"
        current["id"] = "other"
        assert engines.default_engine_id == "other"
    finally:
        for backend in engines.resident.values():
            backend.stop()


def test_new_character_takes_the_engine_and_its_default_voice(engine_set):
    backend = engine_set.get("dummy")
    character = core_engines.new_character("Sam", "#ff0000", backend, core_engines.engine_settings({}, "dummy"))
    assert character.name == "Sam"
    assert character.backend_id == "dummy"
    assert character.preset_data == {"voice": "dummy"}
    assert character.highlight_color == "#ff0000"


def test_new_character_on_a_cloning_engine_without_references_has_no_voice(toneclone_plugin, isolated_dirs):
    engines = EngineSet("toneclone", "toneclone")
    try:
        backend = engines.get("toneclone")
        character = core_engines.new_character(
            "Clone", "#00ff00", backend, core_engines.engine_settings({}, "toneclone"))
        assert character.backend_id == "toneclone"
        assert "voice" not in character.preset_data
    finally:
        for backend in engines.resident.values():
            backend.stop()
