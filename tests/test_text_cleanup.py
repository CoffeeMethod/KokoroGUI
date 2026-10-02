"""kokoro_gui/engine/text_cleanup.py: the import wizard's cleanup rules."""
import pytest

from kokoro_gui.engine import text_cleanup
from kokoro_gui.engine.text_cleanup import RULES, apply_rules, default_rule_ids, enabled_rule_ids, guess_skip


def _rule(rule_id):
    return next(rule for rule in RULES if rule.id == rule_id).apply


def test_rules_are_ordered_and_have_unique_ids():
    ids = [rule.id for rule in RULES]
    assert ids == ["page_numbers", "hyphen_joins", "soft_wraps", "scene_breaks",
                   "collapse_blank_lines", "strip_whitespace"]
    assert len(set(ids)) == len(ids)
    assert default_rule_ids() == [i for i in ids if i != "soft_wraps"]


def test_page_numbers_drops_digit_roman_and_page_lines():
    text = "End of a page.\n12\nvii\nXIV\nPage 13\nPage 14 of 300\n- 15 -\nThe next page starts."
    assert _rule("page_numbers")(text) == "End of a page.\nThe next page starts."


def test_page_numbers_keeps_numbers_inside_sentences_and_words():
    text = "He was 12.\nShe paid 300 for it.\nmild\nChapter 4\n"
    assert _rule("page_numbers")(text) == text


def test_page_numbers_drops_a_last_line_without_a_newline():
    assert _rule("page_numbers")("Text.\n42") == "Text.\n"


def test_hyphen_joins_rejoins_a_split_word():
    assert _rule("hyphen_joins")("an exam-\nple of it") == "an example of it"
    assert _rule("hyphen_joins")("an exam-  \r\n  ple") == "an example"


def test_hyphen_joins_leaves_a_hyphen_before_a_capital():
    # "well-\nKnown" could be a proper noun after a real hyphen; the rule only
    # joins when a lowercase letter follows the break.
    assert _rule("hyphen_joins")("the well-\nKnown author") == "the well-\nKnown author"
    assert _rule("hyphen_joins")("a dash -\nhere and 5-\n6") == "a dash -\nhere and 5-\n6"


def test_soft_wraps_joins_lines_inside_a_paragraph():
    text = "The road ran\nalong the river\nand then stopped.\n\nNext paragraph."
    assert _rule("soft_wraps")(text) == "The road ran along the river and then stopped.\n\nNext paragraph."


def test_soft_wraps_keeps_a_break_after_sentence_end():
    text = 'She said "Go."\nHe went.\nThen it rained\nhard.'
    assert _rule("soft_wraps")(text) == 'She said "Go."\nHe went.\nThen it rained hard.'


def test_soft_wraps_does_not_join_a_scene_break_or_pause_marker():
    text = "It ended\n* * *\nIt began\n[pause:1.5]\nAgain"
    assert _rule("soft_wraps")(text) == "It ended\n* * *\nIt began\n[pause:1.5]\nAgain"


@pytest.mark.parametrize("line", ["*", "* * *", "***", "#", "# # #", "~", "~~~", "—", "— — —"])
def test_scene_breaks_become_a_pause_marker_paragraph(line):
    text = f"The door shut.\n\n{line}\n\nMorning came."
    assert _rule("scene_breaks")(text) == "The door shut.\n\n[pause:1.5]\n\nMorning came."


def test_scene_break_without_blank_lines_gets_its_own_paragraph():
    assert _rule("scene_breaks")("One.\n* * *\nTwo.") == "One.\n\n[pause:1.5]\n\nTwo."


def test_scene_breaks_leave_ordinary_lines_alone():
    text = "*Italic* start\n# not a rule here\nAnd --- dashes"
    assert _rule("scene_breaks")(text) == text


def test_collapse_blank_lines_keeps_one_blank_line():
    assert _rule("collapse_blank_lines")("a\n\n\n\nb\n  \n \nc\n\nd") == "a\n\nb\n\nc\n\nd"


def test_strip_whitespace_trims_line_ends_only():
    assert _rule("strip_whitespace")("  a  \nb\t\n\nc ") == "  a\nb\n\nc"


def test_apply_rules_runs_only_the_named_ids_in_rule_order():
    text = "Intro\n3\n* * *\n\n\n\nan exam-\nple  "
    assert apply_rules(text, []) == text
    assert apply_rules(text, ["page_numbers"]) == "Intro\n* * *\n\n\n\nan exam-\nple  "
    assert apply_rules(text, default_rule_ids()) == "Intro\n\n[pause:1.5]\n\nan example"
    assert apply_rules(text, ["nonsense"]) == text


def test_scene_break_marker_survives_the_collapse_rule():
    assert apply_rules("A.\n\n\n***\n\n\nB.", ["scene_breaks", "collapse_blank_lines"]) == "A.\n\n[pause:1.5]\n\nB."


@pytest.mark.parametrize("title", [
    "Copyright", "Copyright (c) 2021", "COPYRIGHT PAGE", "Contents", "Table of Contents", "Also by Jane Doe",
    "Acknowledgments", "Acknowledgements", "About the Author", "Dedication"])
def test_guess_skip_flags_front_and_back_matter_by_title(title):
    assert guess_skip(title, "x" * 500) is True


@pytest.mark.parametrize("title", ["Chapter 1", "The Contents of the Box", "Arrival", ""])
def test_guess_skip_keeps_story_sections(title):
    assert guess_skip(title, "It was a long road home. " * 20) is False


def test_guess_skip_flags_a_short_section_of_digits_and_punctuation():
    assert guess_skip("Chapter 2", "1 . . . . . 5\n2 . . . . . 9") is True
    assert guess_skip("Chapter 3", "* * *") is True
    assert guess_skip("Chapter 4", "") is True


def test_guess_skip_keeps_a_short_real_section_and_a_long_numeric_one():
    assert guess_skip("Prologue", "He left.") is False
    assert guess_skip("Tables", "1 2 3 4 5 6 7 8 9 " * 30) is False


def test_enabled_rule_ids_reads_the_saved_dict():
    assert enabled_rule_ids(None) == default_rule_ids()
    assert enabled_rule_ids("junk") == default_rule_ids()
    saved = {"page_numbers": False, "soft_wraps": True, "hyphen_joins": "no", "unknown": True}
    got = enabled_rule_ids(saved)
    assert "page_numbers" not in got and "soft_wraps" in got and "hyphen_joins" in got
    assert text_cleanup.RULE_IDS == tuple(r.id for r in RULES)
