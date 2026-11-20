"""Dialogue attribution (plan 26): `daw/attribution.py`'s quote finder,
speech-tag reader and planner."""
from kokoro_gui.daw.attribution import HIGH, LOW, Quote, find_quotes, find_tag

NAMES = ["Marta", "Jon"]


def _spans(text):
    return [text[q.start:q.end] for q in find_quotes(text)]


def _tag(text, names=NAMES, index=0):
    return find_tag(text, find_quotes(text)[index], names)


# -- find_quotes -----------------------------------------------------------------


def test_straight_quotes_pair_up():
    text = 'He said "yes" and then "no".'
    assert _spans(text) == ['"yes"', '"no"']
    quote = find_quotes(text)[0]
    assert text[quote.inner_start:quote.inner_end] == "yes"


def test_curly_double_quotes():
    assert _spans("“Hello,” she said. “Goodbye.”") == ["“Hello,”", "“Goodbye.”"]


def test_curly_single_quotes_skip_apostrophes():
    text = "‘I don’t know,’ said Marta. The dogs’ bowls were full."
    assert _spans(text) == ["‘I don’t know,’"]


def test_apostrophe_inside_double_quotes_is_part_of_the_quote():
    assert _spans('"Don\'t go," said Jon.') == ['"Don\'t go,"']


def test_single_quotes_inside_double_quotes_are_part_of_it():
    text = "“She said ‘run’ and I ran,” said Jon."
    assert _spans(text) == ["“She said ‘run’ and I ran,”"]


def test_an_opening_mark_inside_a_word_is_not_a_quote():
    assert find_quotes("rock‘n’roll") == []


def test_continuation_across_paragraphs_is_one_quote():
    text = "“The first part of a long speech.\n\n“And the second part,” said Marta."
    quotes = find_quotes(text)
    assert len(quotes) == 1
    assert quotes[0] == Quote(0, text.index("”") + 1, 1, text.index("”"))


def test_continuation_with_straight_quotes():
    text = '"One.\n"Two," said Jon.\n"Three."'
    assert _spans(text) == ['"One.\n"Two,"', '"Three."']


def test_unclosed_quote_ends_at_its_paragraph():
    text = '"Never closed, said Jon.\nNarration here. "Next."'
    assert _spans(text) == ['"Never closed, said Jon.', '"Next."']


def test_unclosed_quote_at_the_end_of_the_text():
    assert _spans('Then: "the end  ') == ['"the end']


def test_no_quotes():
    assert find_quotes("Plain narration.\n\nMore of it.") == []


# -- find_tag --------------------------------------------------------------------


def test_tag_after_verb_then_name():
    tag = _tag('"Hello," said Marta.')
    assert (tag.speaker, tag.pronoun, tag.confidence) == ("Marta", None, HIGH)
    assert tag.text == "said Marta"


def test_tag_after_name_then_verb():
    tag = _tag('"Where?" Jon asked.')
    assert (tag.speaker, tag.confidence) == ("Jon", HIGH)


def test_tag_after_with_an_adverb():
    assert _tag('"Fine," Marta quietly replied.').speaker == "Marta"


def test_tag_before_the_quote():
    tag = _tag('Marta said, "Come in."')
    assert (tag.speaker, tag.confidence) == ("Marta", HIGH)


def test_tag_before_with_an_object():
    assert _tag('Jon told her: "Wait."').speaker == "Jon"


def test_tag_between_two_parts_of_one_sentence():
    text = '"Well," said Jon, "I suppose so."'
    assert _tag(text, index=0).speaker == "Jon"
    assert _tag(text, index=1).speaker == "Jon"


def test_tag_after_wins_over_tag_before():
    assert _tag('Marta said nothing. Jon said, "No," Marta replied.').speaker == "Marta"


def test_pronoun_tag_is_low_confidence():
    tag = _tag('"Go," she whispered.')
    assert (tag.speaker, tag.pronoun, tag.confidence) == (None, "she", LOW)


def test_pronoun_tag_verb_first():
    assert _tag('"Go," said he.').pronoun == "he"


def test_name_that_is_not_a_character_is_ignored():
    tag = _tag('"Hi," said Bob.')
    assert (tag.speaker, tag.other, tag.confidence) == (None, "Bob", LOW)


def test_names_match_case_insensitively_and_return_the_given_entry():
    assert _tag('"Hi," said MARTA.', names=["marta"]).speaker == "marta"


def test_first_name_in_a_full_name_tag():
    assert _tag('"Hi," said Marta Lind.').speaker == "Marta"


def test_honorific_is_skipped():
    assert _tag('"Quite," said Mr. Holmes.', names=["Holmes"]).speaker == "Holmes"


def test_no_tag_after_a_full_stop():
    assert _tag('"Hi." The door opened. Marta said nothing.') is None


def test_no_tag_on_a_plain_beat():
    assert _tag('Marta smiled. "Hello."') is None


def test_tag_does_not_cross_a_paragraph():
    assert _tag('Marta said\n"Hello."') is None


def test_lowercase_noun_is_no_name():
    assert _tag('"Hi," said the old man.') is None


def test_name_before_another_clause_does_not_count():
    assert _tag('She looked at Marta and said, "No."') is None
