"""Tests for kokoro_gui/daw/harvest.py: which words of a transcript are worth a
pronunciation rule, where they are, and what the rule for one looks like
(plan 22). No Qt."""
from kokoro_gui.daw.harvest import ACRONYM, DIGITS, NAME, UNKNOWN, candidates, whole_word_rule
from kokoro_gui.engine.lexicon import apply_lexicon


def _by_word(found):
    return {c.word: c for c in found}


def test_a_capital_that_opens_a_sentence_is_not_a_name():
    assert candidates("Marcus walked in. Then he saw nothing. Later it rained.") == []


def test_a_mid_sentence_name_is_found_with_its_count_and_offset():
    text = "He met Marcus. Marcus smiled at the stranger, and Marcus left."
    found = _by_word(candidates(text))
    marcus = found["Marcus"]
    assert marcus.kinds == (NAME,)
    # Every occurrence of the spelling counts once one qualifies.
    assert marcus.count == 3
    assert marcus.first_offset == text.index("Marcus")
    assert text[marcus.first_offset:marcus.first_offset + 6] == "Marcus"
    assert "Marcus" in marcus.context


def test_sentence_starts_after_quotes_and_line_breaks_are_skipped():
    text = 'She said, "Tomas knows." "Marcus agrees."\nPetra waved.\nHe nodded at Iris.'
    assert set(_by_word(candidates(text))) == {"Iris"}


def test_a_name_after_a_title_or_an_initial_is_found():
    found = _by_word(candidates("We asked Dr. Okafor and J. Rowan about it."))
    assert set(found) == {"Okafor", "Rowan"}


def test_contractions_and_i_are_not_names_and_a_possessive_is_the_name():
    found = _by_word(candidates("Then I'm sure I'll go, but Marcus's dog won't."))
    assert set(found) == {"Marcus"}


def test_acronyms_are_found_but_a_shouted_heading_is_not():
    found = _by_word(candidates("The NASA team and the BBC agreed.\nCHAPTER ONE\n"))
    assert set(found) == {"NASA", "BBC"}
    assert found["NASA"].kinds == (ACRONYM,)


def test_digit_tokens_are_found_trimmed_of_punctuation():
    text = "In 1999, the 3rd copy cost $4.50. It flew a B-52 (twice)."
    found = _by_word(candidates(text))
    assert set(found) == {"1999", "3rd", "$4.50", "B-52"}
    assert all(c.kinds == (DIGITS,) for c in found.values())
    assert found["$4.50"].first_offset == text.index("$4.50")


def test_a_word_glued_to_a_digit_token_is_not_a_second_candidate():
    found = _by_word(candidates("We saw Sam fly the B-52 and the 3rd one."))
    assert set(found) == {"Sam", "B-52", "3rd"}


def test_unknown_words_need_a_dictionary_and_skip_known_ones():
    text = "The zephyrine wind and a plain wind blew over Quillon."
    assert set(_by_word(candidates(text))) == {"Quillon"}
    known = {"the", "wind", "and", "a", "plain", "blew", "over"}
    found = _by_word(candidates(text, known))
    assert set(found) == {"zephyrine", "Quillon"}
    assert found["zephyrine"].kinds == (UNKNOWN,)
    assert found["Quillon"].kinds == (NAME, UNKNOWN)


def test_unknown_words_group_by_lowercase_across_sentence_starts():
    known = {"is", "here", "the"}
    found = _by_word(candidates("Zephyr is here. The zephyr is here.", known))
    assert found["Zephyr"].count == 2
    assert found["Zephyr"].kinds == (UNKNOWN,)


def test_a_word_the_lexicon_already_rewrites_is_skipped():
    text = "He met Marcus and Petra in 1999 near the USA."
    rules = [
        {"find": "marcus", "replace": "Mar-kus", "mode": "literal", "case": False},
        {"find": r"\d{4}", "replace": "nineteen", "mode": "regex", "case": False},
        {"find": "USA", "replace": "U S A", "mode": "word", "case": True},
    ]
    assert set(_by_word(candidates(text, lexicon=rules))) == {"Petra"}
    # The old dict shape works too.
    assert set(_by_word(candidates(text, lexicon={"Petra": "Pay-tra"}))) == {"Marcus", "1999", "USA"}


def test_ignored_names_are_skipped_case_insensitively():
    text = "He told Bella about Marcus."
    assert set(_by_word(candidates(text, ignore=["bella"]))) == {"Marcus"}


def test_tags_and_pause_markers_are_not_scanned_and_keep_offsets_right():
    text = "[Narrator:Echo]: We met Marcus [pause:1.5] in 1999.\n[Bob]: Petra came too."
    found = _by_word(candidates(text))
    assert set(found) == {"Marcus", "1999"}
    for c in found.values():
        assert text[c.first_offset:c.first_offset + len(c.word)] == c.word
        assert "[" not in c.context


def test_a_word_right_after_a_tag_starts_a_sentence():
    assert candidates("Then [Bob]: Marcus spoke.") == []


def test_results_are_most_frequent_first():
    text = "We met Iris. We met Petra. We met Petra and Iris and Petra."
    assert [c.word for c in candidates(text)] == ["Petra", "Iris"]


def test_context_is_about_sixty_characters_cut_at_word_boundaries():
    filler = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda "
    text = filler * 3 + "then Marcus came " + filler * 3
    context = candidates(text)[0].context
    assert "Marcus" in context
    assert 30 <= len(context) <= 70
    assert not context.startswith(" ") and "  " not in context
    assert all(word in text.split() for word in context.split())


def test_whole_word_rule_is_whole_word_and_matches_case_for_names():
    rule = whole_word_rule("Al", "Ahl", (NAME,))
    assert rule == {"find": "Al", "replace": "Ahl", "mode": "word", "case": True}
    assert apply_lexicon("Al saw Also and al.", [rule]) == "Ahl saw Also and al."
    loose = whole_word_rule("1999", "nineteen ninety-nine", (DIGITS,))
    assert loose["case"] is False
    assert apply_lexicon("In 1999.", [loose]) == "In nineteen ninety-nine."


def test_whole_word_rule_speaks_a_backslash_as_typed():
    rule = whole_word_rule("AC", "A\\C")
    assert apply_lexicon("The AC unit", [rule]) == "The A\\C unit"
