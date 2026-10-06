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
    pattern1 = engine._lexicon_cache[("literal", False, "hello")]

    engine.apply_lexicon("hello again", lexicon)
    pattern2 = engine._lexicon_cache[("literal", False, "hello")]

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



# -- rules with modes (plan 23) ---------------------------------------------------


LEGACY = {
    "Dr": "Doctor", "Mr": "Mister", "Mrs": "Missus", "Al": "Albert", "SQL": "sequel",
    "API": "A P I", "NASA": "nassa", "gif": "jiff", "cat": "dog", "dog": "wolf",
    "Doctor": "Doc", "colour": "color", "e.g.": "for example", "$": "dollars ",
    "a+b": "a plus b", "(1)": "one", "x*": "ex star", "path\\to": "path to",
    "Marta": "Marrta", "id": "I D \\g<0>",
    "name": "<\\g<0>>", "\\d": "digit", "tab": "a\\tb",
}
PARAGRAPH = (
    "Dr Al met Mr and Mrs Marta. Also Doctor Who, the cat and the dog, in colour; e.g. a+b (1) "
    "used SQL, an API from NASA, a gif for $5. x*y: the id and name of Al's path\\to tab. DR AL."
)


def test_normalize_rules_turns_a_dict_into_literal_case_insensitive_rules():
    from kokoro_gui.engine.lexicon import normalize_rules

    assert normalize_rules({"a": "b", "": "x", "c": "d"}) == [
        {"find": "a", "replace": "b", "mode": "literal", "case": False},
        {"find": "c", "replace": "d", "mode": "literal", "case": False},
    ]


def test_normalize_rules_validates_a_list_and_fills_defaults():
    from kokoro_gui.engine.lexicon import normalize_rules

    rules = normalize_rules([
        {"find": "a", "replace": "b"},
        {"find": "", "replace": "x"},
        {"replace": "x"},
        {"find": "c", "replace": 5},
        "junk",
        {"find": "d", "replace": "e", "mode": "word", "case": True, "extra": 1},
        {"find": "f", "replace": "g", "mode": "bogus", "case": "yes"},
    ])
    assert rules == [
        {"find": "a", "replace": "b", "mode": "literal", "case": False},
        {"find": "d", "replace": "e", "mode": "word", "case": True},
        {"find": "f", "replace": "g", "mode": "literal", "case": False},
    ]
    assert normalize_rules(None) == normalize_rules("text") == normalize_rules(3) == []


def test_a_legacy_dict_and_its_migrated_list_speak_the_same_text():
    from kokoro_gui.engine.lexicon import apply_lexicon, normalize_rules

    assert len(LEGACY) >= 20
    rules = normalize_rules(LEGACY)
    assert apply_lexicon(PARAGRAPH, LEGACY) == apply_lexicon(PARAGRAPH, rules)
    assert apply_lexicon(PARAGRAPH, LEGACY, with_spans=True) == apply_lexicon(PARAGRAPH, rules, with_spans=True)
    assert apply_lexicon(PARAGRAPH, LEGACY) != PARAGRAPH


def test_a_migrated_rule_still_reads_its_replacement_as_a_template():
    from kokoro_gui.engine.lexicon import apply_lexicon, normalize_rules

    legacy = {"name": "<\\g<0>>", "tab": "a\\tb"}
    assert apply_lexicon("a name", normalize_rules(legacy)) == apply_lexicon("a name", legacy) == "a <name>"
    assert apply_lexicon("tab", normalize_rules(legacy)) == "a\tb"


def test_the_signature_is_the_same_for_a_dict_and_its_list():
    from kokoro_gui.engine.lexicon import lexicon_signature, normalize_rules

    assert lexicon_signature(LEGACY) == lexicon_signature(normalize_rules(LEGACY))
    assert lexicon_signature({"a": "b"}) != lexicon_signature([{"find": "a", "replace": "b", "mode": "word"}])
    assert lexicon_signature(None) == lexicon_signature({}) == lexicon_signature([])


def test_word_mode_leaves_a_longer_word_alone():
    from kokoro_gui.engine.lexicon import apply_lexicon

    literal = [{"find": "Al", "replace": "Albert", "mode": "literal", "case": False}]
    word = [{"find": "Al", "replace": "Albert", "mode": "word", "case": False}]
    assert apply_lexicon("Al said Also", literal) == "Albert said Albertso"
    assert apply_lexicon("Al said it is Also Valid, al.", word) == "Albert said it is Also Valid, Albert."


def test_word_mode_checks_only_the_sides_that_end_in_a_word_character():
    from kokoro_gui.engine.lexicon import apply_lexicon

    rules = [{"find": "Dr.", "replace": "Doctor", "mode": "word", "case": False}]
    assert apply_lexicon("Dr. Who and Mydr. Who", rules) == "Doctor Who and Mydr. Who"
    assert apply_lexicon("Dr.Who", rules) == "DoctorWho"


def test_match_case_skips_other_casing():
    from kokoro_gui.engine.lexicon import apply_lexicon

    rules = [{"find": "US", "replace": "United States", "mode": "word", "case": True}]
    assert apply_lexicon("US and us", rules) == "United States and us"


def test_regex_mode_expands_groups():
    from kokoro_gui.engine.lexicon import apply_lexicon

    rules = [{"find": r"(\d+)(st|nd|rd|th)", "replace": r"\1 \2", "mode": "regex", "case": False}]
    assert apply_lexicon("the 3rd and 21st", rules) == "the 3 rd and 21 st"
    text, spans = apply_lexicon("the 3rd", rules, with_spans=True)
    assert text == "the 3 rd"
    assert spans[-1][1] == len("the 3rd")


def test_regex_mode_empty_matches_agree_between_the_plain_and_span_paths():
    from kokoro_gui.engine.lexicon import apply_lexicon, original_offset

    rules = [{"find": r"\b", "replace": "|", "mode": "regex", "case": False}]
    plain = apply_lexicon("ab cd", rules)
    spoken, spans = apply_lexicon("ab cd", rules, with_spans=True)
    assert plain == spoken == "|ab| |cd|"
    assert original_offset(spans, spoken.index("cd")) == 3


def test_an_invalid_regex_is_skipped(capsys):
    from kokoro_gui.engine.lexicon import apply_lexicon

    rules = [{"find": "(", "replace": "x", "mode": "regex", "case": False},
             {"find": "b", "replace": "B", "mode": "literal", "case": False}]
    cache = {}
    assert apply_lexicon("abc", rules, cache) == "aBc"
    assert "Lexicon error" in capsys.readouterr().out
    assert cache[("regex", False, "(")] is None
    assert apply_lexicon("abc", rules, cache) == "aBc"


def test_check_rule_names_the_problem():
    from kokoro_gui.engine.lexicon import check_rule

    assert check_rule("(", "x", "regex") is not None
    assert check_rule("(a)", r"\1", "regex") is None
    assert check_rule("(a)", r"\2", "regex") is not None
    assert check_rule("(?P<n>a)", r"\g<m>", "regex") is not None
    assert check_rule("a", "plain", "literal") is None


def test_rules_apply_in_order():
    from kokoro_gui.engine.lexicon import apply_lexicon

    one = [{"find": "a", "replace": "b", "mode": "literal", "case": False},
           {"find": "b", "replace": "c", "mode": "literal", "case": False}]
    assert apply_lexicon("a", one) == "c"
    assert apply_lexicon("a", list(reversed(one))) == "b"
