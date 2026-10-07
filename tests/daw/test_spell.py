"""Tests for kokoro_gui/daw/spell.py: the language map, the dictionary's
lowercase and apostrophe rules, its answer cache, and which words of a line
get underlined (plan 22, part B). No Qt. The word list is a fake except in the
one test that loads the real English list."""
import pytest

from kokoro_gui.daw import spell
from kokoro_gui.daw.spell import Dictionary, dictionary_for, language_for, unknown_spans

WORDS = {"the", "dog", "barked", "at", "it", "don't", "marcus", "and"}


class FakeList:
    """A word list that counts the questions it is asked."""

    def __init__(self, words):
        self.words, self.asked = set(words), []

    def __contains__(self, word):
        self.asked.append(word)
        return word in self.words


def _dictionary(extra=(), words=WORDS):
    return Dictionary("en", extra, checker=FakeList(words))


def test_kokoros_letters_and_two_letter_codes_map_to_a_language():
    assert [language_for(c) for c in ("a", "b", "e", "f", "i", "p", "en", "de")] == [
        "en", "en", "es", "fr", "it", "pt", "en", "de"]
    assert language_for("j") is None and language_for("z") is None
    assert language_for("English") is None and language_for(None) is None


def test_a_word_is_compared_in_lowercase_with_straight_apostrophes_and_no_possessive():
    d = _dictionary()
    assert "Dog" in d and "DOG" in d
    assert "Don’t" in d
    assert "dog's" in d and "Dog’s" in d
    assert "dgo" not in d and "s" not in d


def test_extra_words_count_as_known_without_touching_the_list():
    shared = FakeList(WORDS)
    d = Dictionary("en", ["Zephyra", "  "], checker=shared)
    assert "zephyra" in d and "Zephyra's" in d
    assert "Zephyra" not in shared.words
    assert "zephyra" not in Dictionary("en", checker=shared)


def test_an_answer_is_asked_of_the_list_once():
    shared = FakeList(WORDS)
    d = Dictionary("en", checker=shared)
    for _ in range(3):
        assert "dog" in d and "qwxz" not in d
    assert shared.asked == ["dog", "qwxz"]


def test_unknown_spans_underline_only_unknown_words():
    text = "The dog barked at Quillon, qwxz."
    spans = unknown_spans(text, _dictionary())
    assert [text[a:b] for a, b in spans] == ["Quillon", "qwxz"]


def test_unknown_spans_skip_tags_acronyms_single_letters_and_digit_words():
    text = "[Zorblax:Radio]: the NASA dog [pause:1.5] x barked 3rd and B52 at 4th."
    assert unknown_spans(text, _dictionary()) == []


def test_unknown_spans_read_a_possessive_as_the_word():
    text = "Marcus's dog"
    assert unknown_spans(text, _dictionary()) == []
    other = "Quillon's dog"
    assert [other[a:b] for a, b in unknown_spans(other, _dictionary())] == ["Quillon's"]


def test_dictionary_for_is_none_for_a_language_without_a_list(monkeypatch):
    assert dictionary_for("j") is None
    monkeypatch.setattr(spell, "_checker", lambda language: None)
    assert dictionary_for("a") is None


def test_the_real_english_list_knows_common_words_and_extra_names():
    pytest.importorskip("spellchecker")
    assert spell.available()
    d = dictionary_for("a", extra=["Qwxzv"])
    assert "hello" in d and "Qwxzv" in d and "zephyrine" not in d
    assert dictionary_for("a") is not None and "qwxzv" not in dictionary_for("a")
