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
