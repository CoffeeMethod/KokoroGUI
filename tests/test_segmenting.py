"""Tests for kokoro_gui/engine/segmenting.py: a word target, ranked boundary
toggles (paragraph > sentence > pause > word), a hard limit at 2x the
target (grill PR6)."""
from kokoro_gui.engine.caching import split_segments
from kokoro_gui.engine.segmenting import split_text


def _words(n, start=1, end="."):
    return " ".join(f"w{i}" for i in range(start, start + n)) + end


def _counts(pieces):
    return [len(p.split()) for p in pieces]


def test_caching_split_segments_is_the_segmenter():
    text = _words(30)
    assert split_segments(text, {"segment_target_words": 10}) == split_text(text, {"segment_target_words": 10})


def test_empty_and_blank_text():
    assert split_text("", {}) == []
    assert split_text("  \n\n \t", {}) == []


def test_text_within_one_and_a_half_targets_stays_whole():
    text = _words(12) + " " + _words(3, start=13)
    assert split_text(text, {"segment_target_words": 10}) == [text]


def test_sentences_pack_toward_the_target():
    text = " ".join(_words(5, start=i) for i in range(1, 60, 5))  # 12 five-word sentences
    pieces = split_text(text, {"segment_target_words": 20})
    assert _counts(pieces) == [20, 20, 20]
    assert all(p.endswith(".") for p in pieces)


def test_paragraph_beats_a_sentence_nearer_the_target():
    # T=10: a sentence end at 10 words, a paragraph end at 13.
    text = _words(10) + " " + _words(3, start=11) + "\n\n" + _words(10, start=14)
    pieces = split_text(text, {"segment_target_words": 10})
    assert _counts(pieces)[0] == 13


def test_sentence_beats_a_pause_nearer_the_target():
    text = "w1 w2 w3 w4 w5 w6 w7. w8 w9 w10, w11 w12 w13 w14 w15 w16 w17 w18 w19 w20."
    pieces = split_text(text, {"segment_target_words": 10})
    assert pieces[0] == "w1 w2 w3 w4 w5 w6 w7."


def test_a_long_sentence_is_cut_at_a_pause_not_mid_phrase():
    text = "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12, w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24."
    pieces = split_text(text, {"segment_target_words": 10})
    assert pieces == ["w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12,",
                      "w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24."]


def test_a_sentence_runs_up_to_twice_the_target_before_it_is_cut():
    text = _words(19) + " " + _words(10, start=20)
    pieces = split_text(text, {"segment_target_words": 10})
    assert pieces[0] == _words(19)


def test_with_no_boundary_it_cuts_between_words_at_the_target():
    # 50 words, no punctuation: word cuts at the target until the end of
    # the text is within 2x, since the end is itself a boundary.
    text = " ".join(f"w{i}" for i in range(1, 51))
    pieces = split_text(text, {"segment_target_words": 10})
    assert _counts(pieces) == [10, 10, 10, 20]
    assert " ".join(pieces) == text


def test_never_splits_inside_a_word():
    text = "supercalifragilistic " * 30
    for piece in split_text(text, {"segment_target_words": 7}):
        assert all(word == "supercalifragilistic" for word in piece.split())


def test_abbreviations_and_initials_are_not_sentence_ends():
    text = "Mr. Smith met Dr. J. Watson at St. Pancras on the morning train. " + _words(10, start=1)
    pieces = split_text(text, {"segment_target_words": 10, "segment_at_pauses": False})
    assert pieces[0] == "Mr. Smith met Dr. J. Watson at St. Pancras on the morning train."


def test_closing_quotes_after_a_full_stop_still_end_a_sentence():
    text = 'He said "we leave at dawn tomorrow, early." ' + _words(12)
    pieces = split_text(text, {"segment_target_words": 8})
    assert pieces[0] == 'He said "we leave at dawn tomorrow, early."'


def test_sentences_off_a_full_stop_still_counts_as_a_pause():
    text = _words(10) + " " + _words(10, start=11)
    pieces = split_text(text, {"segment_target_words": 10, "segment_at_sentences": False})
    assert pieces == [_words(10), _words(10, start=11)]


def test_all_toggles_off_cuts_only_between_words_at_the_target():
    text = _words(5) + "\n\n" + _words(25, start=6)
    off = {"segment_target_words": 10, "segment_at_paragraphs": False,
           "segment_at_sentences": False, "segment_at_pauses": False}
    assert _counts(split_text(text, off)) == [10, 20]
    assert _counts(split_text(text, {"segment_target_words": 10}))[0] == 5  # the paragraph, early


def test_a_line_break_is_a_pause():
    text = "w1 w2 w3 w4 w5 w6 w7 w8\nw9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20"
    pieces = split_text(text, {"segment_target_words": 10})
    assert pieces[0] == "w1 w2 w3 w4 w5 w6 w7 w8"


def test_a_short_tail_joins_the_piece_before_it():
    # Sentence ends after 12 and 16 words: the best cut (12) would leave 4,
    # under half the target, so all 16 (within 2x) stay one piece.
    text = _words(12) + " " + _words(4, start=13)
    pieces = split_text(text, {"segment_target_words": 10})
    assert _counts(pieces) == [16]


def test_whitespace_inside_a_piece_is_collapsed():
    assert split_text("Hello   there,\n\tfriend.", {}) == ["Hello there, friend."]


def test_target_is_clamped():
    text = " ".join(f"w{i}" for i in range(1, 21))
    assert _counts(split_text(text, {"segment_target_words": 1})) == [5, 5, 10]  # clamped to 5
    assert split_text(text, {"segment_target_words": "junk"}) == [text]
