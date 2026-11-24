"""kokoro_gui/headless/script.py: a script becomes a Document. No model loads;
the backend is a stub that lists voices, plus one test on the real Dummy
adapter."""
from types import SimpleNamespace

import pytest

from kokoro_gui.core.engines import EngineSet
from kokoro_gui.headless.script import build_document


class StubBackend:
    id = "stub"

    def __init__(self, voices=("v1", "v2")):
        self._voices = voices

    def get_voices(self, lang_code=None):
        return [SimpleNamespace(id=v) for v in self._voices]

    def get_config_schema(self):
        return []


@pytest.fixture
def backend():
    return StubBackend()


def build(script, backend, cast=None, **kwargs):
    return build_document(script, backend, cast, **kwargs)


def clips_of(document):
    """`(character name, clip text)` per clip, in text order."""
    ordered = sorted(document.clips, key=lambda c: document.clip_extent(c.id)[0])
    return [(document.get_character(c.character_id).name, document.clip_text(c)) for c in ordered]


def clip_after(document, text):
    return next(c for c in document.clips if document.clip_text(c) == text)


def test_untagged_paragraphs_are_one_narrator_clip_each(backend):
    document, warnings = build("One.\n\nTwo.\n\nThree.", backend)
    assert clips_of(document) == [("Narrator", "One."), ("Narrator", "Two."), ("Narrator", "Three.")]
    assert document.text == "One.\n\nTwo.\n\nThree."
    assert warnings == []


def test_tags_are_gone_from_the_document_text(backend):
    document, _ = build("[Sam]: Hello there.\n\n[Ann:Radio]: Hi.", backend, {"Sam": "v1", "Ann": "v2"})
    assert "[" not in document.text
    assert document.text == "Hello there.\n\nHi."
    assert clips_of(document) == [("Sam", "Hello there."), ("Ann", "Hi.")]


def test_each_paragraph_of_a_speaker_block_is_a_clip(backend):
    script = "[Sam]: First.\n\nSecond.\n\n[Ann]: Third.\n\nFourth."
    document, _ = build(script, backend, {"Sam": "v1", "Ann": "v2"})
    assert clips_of(document) == [("Sam", "First."), ("Sam", "Second."), ("Ann", "Third."), ("Ann", "Fourth.")]


def test_a_tag_fx_name_sets_the_clip_override(backend):
    document, _ = build("[Sam:Radio]: Over.\n\nStill Sam.\n\n[Sam]: Plain.", backend, {"Sam": "v1"})
    assert clip_after(document, "Over.").overrides == {"fx_preset": "Radio"}
    assert clip_after(document, "Still Sam.").overrides == {"fx_preset": "Radio"}
    assert clip_after(document, "Plain.").overrides == {}


def test_an_unknown_fx_name_warns_and_sets_nothing_when_the_list_is_given(backend):
    document, warnings = build("[Sam:Gone]: Hi.\n\n[Sam:Radio]: Yo.", backend, {"Sam": "v1"}, known_fx={"Radio"})
    assert clip_after(document, "Hi.").overrides == {}
    assert clip_after(document, "Yo.").overrides == {"fx_preset": "Radio"}
    assert len(warnings) == 1 and "Gone" in warnings[0]


def test_tag_options_become_overrides_and_bad_ones_warn(backend):
    document, warnings = build("[Sam, overlap:0.3]: Hi.\n\n[Ann, bogus:1]: Yo.", backend,
                               {"Sam": "v1", "Ann": "v2"})
    assert clip_after(document, "Hi.").overrides == {"overlap_s": 0.3}
    assert clip_after(document, "Yo.").overrides == {}
    assert any("bogus" in w for w in warnings)


def test_a_pause_is_the_next_clips_gap_before(backend):
    document, _ = build("One.\n\n[pause:1.5]\n\nTwo.\n\nThree.", backend)
    assert document.text == "One.\n\nTwo.\n\nThree."
    assert clip_after(document, "One.").gap_before_s is None
    assert clip_after(document, "Two.").gap_before_s == 1.5
    assert clip_after(document, "Three.").gap_before_s is None


def test_a_pause_inside_a_paragraph_splits_it(backend):
    document, _ = build("Before [pause:2] after.", backend)
    assert clips_of(document) == [("Narrator", "Before"), ("Narrator", "after.")]
    assert clip_after(document, "after.").gap_before_s == 2.0


def test_a_pause_before_a_tag_lands_on_that_speakers_clip(backend):
    document, _ = build("[Sam]: Hi.\n\n[pause:3]\n\n[Ann]: Yo.", backend, {"Sam": "v1", "Ann": "v2"})
    assert clip_after(document, "Yo.").gap_before_s == 3.0


def test_a_pause_at_the_start_is_the_first_clips_gap(backend):
    document, warnings = build("[pause:1]\n\nHello.", backend)
    assert clip_after(document, "Hello.").gap_before_s == 1.0
    assert warnings == []


def test_a_pause_at_the_end_warns_and_adds_no_clip(backend):
    document, warnings = build("Hello.\n\n[pause:1]", backend)
    assert clips_of(document) == [("Narrator", "Hello.")]
    assert document.text == "Hello."
    assert len(warnings) == 1 and "pause" in warnings[0]


def test_two_pauses_in_a_row_keep_the_last(backend):
    document, _ = build("A.\n\n[pause:1][pause:2.5]\n\nB.", backend)
    assert clip_after(document, "B.").gap_before_s == 2.5


def test_an_unmapped_speaker_gets_the_default_voice_and_a_warning(backend):
    document, warnings = build("[Zed]: Hi.", backend, {})
    assert document.get_character_by_name("Zed").preset_data == {"voice": "v1"}
    assert len(warnings) == 1 and "Zed" in warnings[0]


def test_the_narrator_needs_no_cast_entry_and_no_warning(backend):
    document, warnings = build("Plain text.", backend)
    assert document.get_character_by_name("Narrator").preset_data == {"voice": "v1"}
    assert warnings == []


def test_the_cast_can_set_the_narrator(backend):
    document, _ = build("Plain text.", backend, {"Narrator": "v2"})
    assert document.get_character_by_name("Narrator").preset_data == {"voice": "v2"}


def test_a_mapped_speaker_with_a_voice_string_has_no_warning(backend):
    document, warnings = build("[Sam]: Hi.", backend, {"Sam": "v2"})
    assert document.get_character_by_name("Sam").preset_data == {"voice": "v2"}
    assert warnings == []


def test_agent_settings_go_in_the_character_preset(backend):
    cast = {"Sam": {"voice": "v2", "speed": 1.2, "pitch": -2, "fx_preset": "Radio", "lang_code": "b", "x": 1}}
    document, _ = build("[Sam]: Hi.", backend, cast)
    character = document.get_character_by_name("Sam")
    assert character.preset_data == {"voice": "v2", "speed": 1.2, "pitch": -2, "fx_preset": "Radio"}
    assert character.backend_id == "stub"


def test_a_cast_mapping_without_a_voice_gets_the_default_voice(backend):
    document, warnings = build("[Sam]: Hi.", backend, {"Sam": {"speed": 1.5}})
    assert document.get_character_by_name("Sam").preset_data == {"voice": "v1", "speed": 1.5}
    assert warnings == []


def test_cast_names_match_ignoring_case_and_spaces(backend):
    document, warnings = build("[ sam ]: Hi.\n\n[SAM]: Again.", backend, {"Sam": "v2"})
    assert len(document.characters) == 1
    assert document.characters[0].preset_data == {"voice": "v2"}
    assert warnings == []


def test_gaps_are_zero_in_the_settings(backend):
    document, _ = build("A.\n\nB.", backend)
    assert document.settings["gap_s"] == 0
    assert document.settings["paragraph_gap_s"] == 0


def test_one_track_per_character(backend):
    script = "Intro.\n\n[Sam]: A.\n\n[Ann]: B.\n\n[Sam]: C."
    document, _ = build(script, backend, {"Sam": "v1", "Ann": "v2"})
    assert [t.name for t in document.tracks] == ["Narrator", "Sam", "Ann"]
    assert len(document.characters) == 3
    for clip in document.clips:
        track = next(t for t in document.tracks if t.id == clip.track_id)
        assert track.character_id == clip.character_id


def test_the_same_speaker_twice_is_one_character(backend):
    document, _ = build("[Sam]: A.\n\n[Ann]: B.\n\n[Sam]: C.", backend, {"Sam": "v1", "Ann": "v2"})
    assert [c.name for c in document.characters] == ["Sam", "Ann"]
    assert clips_of(document) == [("Sam", "A."), ("Ann", "B."), ("Sam", "C.")]


def test_characters_get_distinct_colors_in_first_use_order(backend):
    document, _ = build("[A]: a\n\n[B]: b\n\n[C]: c", backend)
    colors = [c.highlight_color for c in document.characters]
    assert len(set(colors)) == 3


def test_clips_are_not_generated_and_are_stale(backend):
    document, _ = build("Hi.", backend)
    assert all(not c.segments for c in document.clips)
    assert {c.id for c in document.dirty_clips()} == {c.id for c in document.clips}


@pytest.mark.parametrize("script", ["", "   \n\n  ", None, "\n"])
def test_an_empty_script_is_an_empty_document(backend, script):
    document, warnings = build(script, backend)
    assert document.text == ""
    assert document.clips == [] and document.characters == [] and document.tracks == []
    assert document.settings["gap_s"] == 0
    assert warnings == []


def test_only_tags_gives_no_clips_and_no_characters(backend):
    document, _ = build("[Sam]:\n\n[Ann:Radio]:  ", backend, {"Sam": "v1"})
    assert document.text == ""
    assert document.clips == [] and document.characters == []


def test_only_pauses_gives_no_clips(backend):
    document, warnings = build("[pause:1] [pause:2]", backend)
    assert document.clips == []
    assert len(warnings) == 1


def test_crlf_and_cr_line_endings(backend):
    document, _ = build("[Sam]: One.\r\n\r\nTwo.\r\rThree.", backend, {"Sam": "v1"})
    assert document.text == "One.\n\nTwo.\n\nThree."
    assert "\r" not in document.text


def test_lines_with_only_spaces_still_split_paragraphs(backend):
    document, _ = build("One.\n   \nTwo.", backend)
    assert clips_of(document) == [("Narrator", "One."), ("Narrator", "Two.")]


def test_a_single_newline_stays_inside_the_paragraph(backend):
    document, _ = build("Line one\nline two.", backend)
    assert clips_of(document) == [("Narrator", "Line one\nline two.")]


@pytest.mark.parametrize("name", ["Dr. Smith", "O'Brien", "Smith, John", "Mary-Jane #2", "Ann (old)", "a/b\\c",
                                  "..", "サム", "Zøe \U0001f600"])
def test_odd_speaker_names(backend, name):
    document, warnings = build(f"[{name}]: Hello.", backend, {name: "v2"})
    assert clips_of(document) == [(name, "Hello.")]
    assert document.characters[0].preset_data == {"voice": "v2"}
    assert warnings == []


def test_a_very_long_speaker_name_is_not_a_tag(backend):
    name = "x" * 101
    document, _ = build(f"[{name}]: Hello.", backend)
    assert clips_of(document) == [("Narrator", f"[{name}]: Hello.")]


def test_unicode_text_is_kept(backend):
    document, _ = build("[Zoë]: ¡Hola! 日本語 \U0001f600", backend, {"Zoë": "v1"})
    assert document.text == "¡Hola! 日本語 \U0001f600"


def test_clip_offsets_cover_exact_paragraphs(backend):
    document, _ = build("[Sam]: Alpha.\n\nBeta.\n\n[pause:1]\n\nGamma.", backend, {"Sam": "v1"})
    for clip in document.clips:
        start, end = document.clip_extent(clip.id)
        assert document.text[start:end] == document.clip_text(clip)
    assert document.text == "Alpha.\n\nBeta.\n\nGamma."


def test_the_stored_engine_settings_decide_the_default_voice():
    backend = StubBackend(("v1", "v2"))
    document, _ = build("[Zed]: Hi.", backend, engine_settings={"voice": "v2"})
    assert document.get_character_by_name("Zed").preset_data == {"voice": "v2"}


def test_the_script_and_cast_are_not_modified(backend):
    cast = {"Sam": {"voice": "v1"}}
    build("[Sam]: Hi.", backend, cast)
    assert cast == {"Sam": {"voice": "v1"}}


def test_hostile_brackets_stay_fast(backend):
    import time
    script = "[" * 100_000 + "\n\n" + "[a]: b [pause:1]" * 300
    start = time.perf_counter()
    build(script, backend)
    assert time.perf_counter() - start < 10


def test_with_the_real_dummy_adapter(isolated_dirs):
    engines = EngineSet("dummy", "dummy")
    try:
        backend = engines.get("dummy")
        document, warnings = build_document("Hello.\n\n[Sam:Radio]: Hi.", backend, {"Sam": "dummy"})
    finally:
        for adapter in engines.resident.values():
            adapter.stop()
    assert warnings == []
    assert {c.backend_id for c in document.characters} == {"dummy"}
    assert document.get_character_by_name("Narrator").preset_data == {"voice": "dummy"}
    assert clip_after(document, "Hi.").overrides == {"fx_preset": "Radio"}
