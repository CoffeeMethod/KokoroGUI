"""Tests for kokoro_gui/daw/proof.py (plan 21): word normalization, the
score and its issues, the threshold, what a clip is proofed against, and
checking a stored result on the way back in."""
import pytest

from kokoro_gui.daw import dirty, proof
from kokoro_gui.daw.models import Clip, Segment


def _clip(*texts, keys=None):
    segments = [Segment(order_index=i, text=t, cache_key=(keys[i] if keys else f"k{i}"))
                for i, t in enumerate(texts)]
    return Clip(segments=segments)


def test_identical_text_scores_one_and_has_no_issues():
    result = proof.score("The old mill stood by the river.", ["the", "old", "mill", "stood", "by", "the", "river"])
    assert result.ratio == 1.0 and result.issues == ()
    assert not proof.flagged(result)


def test_punctuation_and_case_are_ignored():
    assert proof.score("Hello, WORLD!", ["hello", "world."]).ratio == 1.0
    assert proof.score("Don't stop", ["don’t", "stop"]).ratio == 1.0
    assert proof.score("a well-known fact", ["a", "well", "known", "fact"]).ratio == 1.0


def test_a_dropped_phrase_is_a_dropped_issue():
    result = proof.score("It was the old mill by the river", ["it", "was", "by", "the", "river"])
    assert result.issues == (("dropped", "the old mill", ""),)
    assert result.ratio < 1.0
    assert proof.issue_text(result.issues[0]) == "dropped: 'the old mill'"


def test_a_repeated_word_is_an_added_issue():
    result = proof.score("go to the store", ["go", "to", "to", "the", "store"])
    assert result.issues == (("added", "", "to"),)
    assert proof.issue_text(result.issues[0]) == "added: 'to'"


def test_a_misheard_word_is_a_changed_issue():
    result = proof.score("the old mill", ["the", "old", "meal"])
    assert result.issues == (("changed", "mill", "meal"),)
    assert proof.issue_text(result.issues[0]) == "changed: 'mill' heard as 'meal'"


def test_heard_words_may_be_the_asr_triples():
    heard = [("Hello", 0.0, 0.4), ("world", 0.5, 0.9)]
    assert proof.score("hello world", heard).ratio == 1.0


def test_nothing_heard_is_a_zero_and_two_empty_sides_match():
    assert proof.score("some words here", []).ratio == 0.0
    assert proof.score("some words here", []).issues == (("dropped", "some words here", ""),)
    assert proof.score("", []).ratio == 1.0
    assert proof.score("", ["uh"]).ratio == 0.0


def test_a_year_matches_the_same_year_read_in_words():
    assert proof.score("In 1999 it ended.", ["in", "nineteen", "ninety", "nine", "it", "ended"]).ratio == 1.0
    assert proof.score("In 1999 it ended.", ["in", "1999", "it", "ended"]).ratio == 1.0
    assert proof.score("Back in 2007", ["back", "in", "two", "thousand", "seven"]).ratio == 1.0


@pytest.mark.parametrize("digits, words", [
    ("0", "zero"), ("7", "seven"), ("13", "thirteen"), ("21", "twenty one"), ("40", "forty"),
    ("100", "one hundred"), ("1000", "one thousand"), ("1905", "nineteen oh five"),
    ("1900", "nineteen hundred"), ("2000", "two thousand"), ("2024", "twenty twenty four"),
])
def test_numbers_are_spelled_by_the_table(digits, words):
    assert proof.normalize_words(digits) == words.split()


@pytest.mark.parametrize("token", ["101", "999", "2100", "12345", "007", "1st"])
def test_a_number_outside_the_table_stays_as_digits(token):
    assert proof.normalize_words(token) == [token]


def test_a_decimal_is_two_numbers_on_both_sides():
    assert proof.normalize_words("3.5") == ["three", "five"]
    assert proof.score("pi is 3.14", ["pi", "is", "3.14"]).ratio == 1.0


def test_flagged_compares_with_the_threshold():
    result = proof.ProofResult(0.9, ())
    assert proof.flagged(result) and not proof.flagged(result, threshold=0.85)
    assert not proof.flagged(proof.ProofResult(0.92, ()))


def test_clamp_threshold_keeps_a_usable_value():
    assert proof.clamp_threshold(0.8) == 0.8
    assert proof.clamp_threshold(5) == 1.0 and proof.clamp_threshold(0.1) == 0.5
    for junk in ("x", None, True, float("nan"), float("inf")):
        assert proof.clamp_threshold(junk) == proof.DEFAULT_THRESHOLD


def test_the_expected_side_is_what_the_engine_spoke_so_the_lexicon_is_applied():
    # Generation stores Segment.text over the spoken text: tags and pause
    # markers gone, the lexicon applied. Proofing compares against that.
    spoken = dirty.spoken_text("Dr. Smith [Alice]: arrived.", {"lexicon": {"Dr.": "Doctor"}})
    clip = _clip(spoken)
    assert "Doctor" in proof.expected_text(clip)
    heard = proof.score(proof.expected_text(clip), ["doctor", "smith", "arrived"])
    unspoken = proof.score("Dr. Smith [Alice]: arrived.", ["doctor", "smith", "arrived"])
    assert heard.ratio > unspoken.ratio


def test_expected_text_joins_the_segments_in_order():
    clip = _clip("first part.", "second part.")
    clip.segments.reverse()
    assert proof.expected_text(clip) == "first part. second part."


def test_segment_keys_change_with_a_regenerate():
    clip = _clip("a", "b", keys=["x", "x"])
    before = proof.segment_keys(clip)
    assert before == ["x:0", "x:1"]
    clip.segments = [Segment(order_index=0, text="a", cache_key="y")]
    assert proof.segment_keys(clip) != before


def test_an_entry_is_current_only_for_the_audio_it_was_scored_on():
    clip = _clip("hello there")
    entry = proof.make_entry(proof.score("hello there", ["hello"]), proof.segment_keys(clip))
    assert proof.is_current(entry, clip)
    assert proof.is_flagged(entry, clip)
    clip.segments[0].cache_key = "regenerated"
    assert not proof.is_current(entry, clip) and not proof.is_flagged(entry, clip)


def test_an_entry_marked_ok_is_not_flagged_and_a_good_score_never_is():
    clip = _clip("hello there")
    entry = proof.make_entry(proof.score("hello there", ["hello"]), proof.segment_keys(clip))
    entry["ok"] = True
    assert not proof.is_flagged(entry, clip)
    good = proof.make_entry(proof.score("hello there", ["hello", "there"]), proof.segment_keys(clip))
    assert not proof.is_flagged(good, clip)
    assert proof.is_flagged(good, clip, threshold=1.0) is False  # 1.0 is not below 1.0


def test_make_entry_round_trips_through_json_and_clean_results():
    import json

    clip = _clip("hello there")
    entry = proof.make_entry(proof.score("hello there", ["hullo", "there"]), proof.segment_keys(clip))
    stored = json.loads(json.dumps({"c1": entry}))
    assert proof.clean_results(stored) == {"c1": entry}
    assert proof.entry_result(stored["c1"]).issues == (("changed", "hello", "hullo"),)


def test_make_entry_caps_the_issues():
    expected = " ".join(f"w{i}" for i in range(0, 200, 2))
    heard = [f"x{i}" for i in range(0, 200, 2)]
    entry = proof.make_entry(proof.score(expected, heard), ["k:0"])
    assert len(entry["issues"]) <= proof.MAX_ISSUES


def test_clean_results_drops_what_is_not_a_result():
    good = {"ratio": 0.5, "issues": [["dropped", "a", ""]], "segment_keys": ["k:0"], "ok": True}
    raw = {
        "good": good,
        "": good,
        "no_ratio": {"segment_keys": ["k:0"]},
        "string_ratio": {"ratio": "0.5", "segment_keys": ["k:0"]},
        "bool_ratio": {"ratio": True, "segment_keys": ["k:0"]},
        "nan_ratio": {"ratio": float("nan"), "segment_keys": ["k:0"]},
        "out_of_range": {"ratio": 1.5, "segment_keys": ["k:0"]},
        "keys_not_a_list": {"ratio": 0.5, "segment_keys": "k:0"},
        "keys_not_strings": {"ratio": 0.5, "segment_keys": [1]},
        "not_a_dict": [0.5],
        7: good,
    }
    assert set(proof.clean_results(raw)) == {"good"}
    assert proof.clean_results(raw)["good"]["ok"] is True
    assert proof.clean_results(["x"]) == {} and proof.clean_results(None) == {}


def test_clean_results_repairs_a_damaged_issue_list_and_ok_flag():
    entry = {"ratio": 0.4, "segment_keys": ["k:0"], "ok": "yes",
             "issues": [["dropped", "a", "b"], ["bogus", "a", "b"], ["added", 1, "b"], "text", ["added", "x"]]}
    cleaned = proof.clean_results({"c": entry})["c"]
    assert cleaned["issues"] == [["dropped", "a", "b"]] and cleaned["ok"] is False
    assert proof.clean_results({"c": {**entry, "issues": 5}})["c"]["issues"] == []
