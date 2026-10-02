"""Tests for KokoroEngine.apply_lexicon (kokoro_engine.py:87-108)."""


def test_apply_lexicon_case_insensitive_replace(engine):
    result = engine.apply_lexicon("Hello WORLD", {"world": "planet"})
    assert result == "Hello planet"


def test_apply_lexicon_empty_dict_returns_unchanged(engine):
    assert engine.apply_lexicon("Hello world", {}) == "Hello world"


def test_apply_lexicon_falsy_input_returns_text_as_is(engine):
    assert engine.apply_lexicon("Hello world", None) == "Hello world"


def test_apply_lexicon_skips_falsy_keys(engine):
    result = engine.apply_lexicon("Hello world", {"": "ignored", "world": "planet"})
    assert result == "Hello planet"


def test_apply_lexicon_caches_compiled_regex(engine):
    lexicon = {"hello": "hi"}
    engine.apply_lexicon("hello there", lexicon)
    pattern1 = engine._lexicon_cache["hello"]

    engine.apply_lexicon("hello again", lexicon)
    pattern2 = engine._lexicon_cache["hello"]

    assert pattern1 is pattern2


def test_apply_lexicon_multiple_rules_applied(engine):
    result = engine.apply_lexicon("The cat sat on the mat", {"cat": "dog", "mat": "rug"})
    assert result == "The dog sat on the rug"


# -- spans (phase 2, C1: word highlight through a lexicon) -----------------------


def test_apply_lexicon_with_spans_maps_offsets_back():
    from kokoro_gui.engine.lexicon import apply_lexicon, original_offset

    text = "Dr Who met Dr No."
    spoken, spans = apply_lexicon(text, {"Dr": "Doctor"}, with_spans=True)
    assert spoken == "Doctor Who met Doctor No."
    assert apply_lexicon(text, {"Dr": "Doctor"}) == spoken
    assert original_offset(spans, spoken.index("Who")) == text.index("Who")
    assert original_offset(spans, spoken.index("No")) == text.index("No")
    assert original_offset(spans, 3) == 0  # inside "Doctor" -> the "Dr" it replaced
    assert original_offset(spans, spoken.index("met")) == text.index("met")


def test_apply_lexicon_spans_without_a_lexicon_are_identity():
    from kokoro_gui.engine.lexicon import apply_lexicon, original_offset

    spoken, spans = apply_lexicon("plain", {}, with_spans=True)
    assert spoken == "plain"
    assert [original_offset(spans, i) for i in range(5)] == [0, 1, 2, 3, 4]


def test_apply_lexicon_spans_survive_chained_rules():
    from kokoro_gui.engine.lexicon import apply_lexicon, original_offset

    text = "use SQL now"
    spoken, spans = apply_lexicon(text, {"SQL": "S Q L", "now": "right now"}, with_spans=True)
    assert spoken == "use S Q L right now"
    assert original_offset(spans, spoken.index("right")) == text.index("now")
    assert original_offset(spans, spoken.index("S Q L") + 2) == text.index("SQL")
