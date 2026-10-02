"""Tests for kokoro_gui/daw/wordalign.py (phase 2, C1)."""
from kokoro_gui.daw.wordalign import align


def test_exact_match_takes_the_asr_times():
    asr = [("hello", 0.0, 0.4), ("brave", 0.5, 0.8), ("world", 0.9, 1.3)]
    assert align("Hello brave world.", asr) == [
        ["Hello", 0.0, 0.4], ["brave", 0.5, 0.8], ["world.", 0.9, 1.3],
    ]


def test_a_word_the_asr_dropped_interpolates_between_neighbours():
    asr = [("hello", 0.0, 0.4), ("world", 1.0, 1.4)]
    words = align("Hello brave world", asr)
    assert words[0] == ["Hello", 0.0, 0.4]
    assert words[1] == ["brave", 0.4, 1.0]
    assert words[2] == ["world", 1.0, 1.4]


def test_an_inserted_asr_word_is_ignored():
    asr = [("hello", 0.0, 0.4), ("um", 0.45, 0.6), ("world", 0.7, 1.0)]
    assert align("Hello world", asr) == [["Hello", 0.0, 0.4], ["world", 0.7, 1.0]]


def test_a_number_read_as_words_still_places_the_text_digit():
    asr = [("it", 0.0, 0.1), ("costs", 0.1, 0.4), ("twenty", 0.5, 0.8), ("dollars", 0.9, 1.2)]
    words = align("It costs 20 dollars", asr)
    assert [w[0] for w in words] == ["It", "costs", "20", "dollars"]
    assert words[2][1] == 0.4 and words[2][2] == 0.9
    assert words[3] == ["dollars", 0.9, 1.2]


def test_unmatched_edges_and_empty_inputs():
    assert align("", [("a", 0.0, 1.0)]) == []
    assert align("Hello", []) == []
    words = align("Well hello", [("hello", 0.3, 0.6)])
    assert words[0] == ["Well", 0.3, 0.3]


def test_nothing_matching_spreads_the_text_over_the_asr_span():
    words = align("alpha beta", [("x", 0.0, 1.0), ("y", 1.0, 2.0)])
    assert words == [["alpha", 0.0, 1.0], ["beta", 1.0, 2.0]]
