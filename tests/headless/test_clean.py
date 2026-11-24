"""kokoro_gui/headless/clean.py: markdown to speakable text, and that it
stays fast on hostile input."""
import time

import pytest

from kokoro_gui.headless.clean import clean_markdown


@pytest.mark.parametrize("markdown, expected", [
    ("# Title", "Title."),
    ("## Sub ##", "Sub."),
    ("###### Deep #", "Deep."),
    ("#hashtag", "#hashtag"),
    ("####### seven", "####### seven"),
    ("**bold** and *italic* and ***both***", "bold and italic and both"),
    ("__bold__ and _italic_", "bold and italic"),
    ("snake_case_name stays", "snake_case_name stays"),
    ("2 * 3 = 6", "2 * 3 = 6"),
    ("~~gone~~ stays", "gone stays"),
    ("[the docs](https://example.com/a_b) here", "the docs here"),
    ('[titled](https://example.com "A title") ok', "titled ok"),
    ("![a cat](cat.png) sits", "a cat sits"),
    ("[![badge](b.png)](https://x.io)", "badge"),
    ("[ref link][1] text", "ref link text"),
    ("[1]: https://example.com/ref", ""),
    ("Note[^1] here", "Note here"),
    ("Use `foo()` now", "Use foo() now"),
    ("Use ``a ` b`` now", "Use a ` b now"),
    ("stray ` tick", "stray tick"),
    ("- one\n- two", "one.\ntwo."),
    ("* one\n+ two\n1. three\n2) four", "one.\ntwo.\nthree.\nfour."),
    ("- [x] done\n- [ ] todo", "done.\ntodo."),
    ("> quoted\n> > nested", "quoted\nnested"),
    ("> - item", "item."),
    ("a <b>bold</b> <br/> c", "a bold c"),
    ("a <!-- hidden\nacross --> b", "a b"),
    ("<!-- unclosed", "<!-- unclosed"),
    ("see <https://example.com/x> now", "see example.com now"),
    ("see https://www.example.com/path?q=1#top.", "see example.com."),
    ("(www.example.org/x)", "(example.org)"),
    ("AT&amp;T &lt;ok&gt;", "AT&T <ok>"),
    ("escaped \\*star\\* and \\# hash", "escaped *star* and # hash"),
    ("Title\n=====\ntext", "Title\n\ntext"),
    ("above\n\n---\n\nbelow", "above\n\nbelow"),
])
def test_cases(markdown, expected):
    assert clean_markdown(markdown) == expected


def test_fenced_code_is_dropped_by_default():
    text = "before\n\n```python\nprint('x')\n```\n\nafter"
    assert clean_markdown(text) == "before\n\nafter"


def test_fenced_code_with_tildes_and_longer_fences():
    text = "a\n\n~~~\ncode\n~~~\n\n````\n```\ninner\n```\n````\n\nb"
    assert clean_markdown(text) == "a\n\nb"


def test_unclosed_fence_drops_the_rest():
    assert clean_markdown("a\n\n```\ncode\nmore") == "a"


def test_fenced_code_can_be_kept_without_its_fences():
    assert clean_markdown("a\n\n```py\nx = 1\n```\n\nb", code_blocks="keep") == "a\n\nx = 1\n\nb"


def test_a_bad_code_blocks_value_is_refused():
    with pytest.raises(ValueError):
        clean_markdown("x", code_blocks="maybe")


def test_table_cells_are_joined_and_the_rule_row_dropped():
    table = "| Name | Age |\n| --- | :-: |\n| Ann | 30 |\n| Bo | 41 |"
    assert clean_markdown(table) == "Name, Age.\nAnn, 30.\nBo, 41."


def test_table_without_outer_pipes():
    assert clean_markdown("a | b\n--- | ---\n1 | 2") == "a, b.\n1, 2."


def test_headings_sit_in_their_own_paragraph():
    assert clean_markdown("intro\n# Head\nbody") == "intro\n\nHead.\n\nbody"


def test_end_punctuation_is_not_doubled():
    assert clean_markdown("# Done!\n- really?") == "Done!\n\nreally?"


def test_crlf_and_cr():
    assert clean_markdown("# A\r\n\r\nb\r- c") == "A.\n\nb\nc."


def test_empty_and_none():
    assert clean_markdown("") == ""
    assert clean_markdown(None) == ""
    assert clean_markdown("```\nonly code\n```") == ""


def test_unicode_survives():
    assert clean_markdown("**Café** 日本語 _ü_ \U0001f600") == "Café 日本語 ü \U0001f600"


def test_blank_runs_collapse():
    assert clean_markdown("a\n\n\n\n\nb") == "a\n\nb"


def test_speaker_tags_and_pause_markers_pass_through():
    text = "[Sam:Radio]: **Hello** there.\n\n[pause:1.5]\n\n[Ann]: Hi [pause:2][pause:3] ok"
    assert clean_markdown(text) == "[Sam:Radio]: Hello there.\n\n[pause:1.5]\n\n[Ann]: Hi [pause:2][pause:3] ok"


# Hostile input. Each case is 200 KB or more; a quadratic regex takes
# minutes there, a linear pass well under a second. The bound is loose so a
# slow CI machine doesn't fail it.
_BIG = 200_000
HOSTILE = {
    "open brackets": "[" * _BIG,
    "open brackets then close": "[" * _BIG + "](x)",
    "image openers": "![" * _BIG,
    "link openers": "](" * _BIG,
    "empty links": "[]()" * (_BIG // 4),
    "backticks": "`" * _BIG,
    "alternating backticks": "`a``b```c" * (_BIG // 9),
    "stars": "*" * _BIG,
    "star words": "*a" * _BIG,
    "underscores": "_" * _BIG,
    "tildes": "~" * _BIG,
    "angle openers": "<a" * _BIG,
    "long open tag": "<a " + "x" * _BIG,
    "comment openers": "<!--" * (_BIG // 4),
    "url prefixes": "http://" * (_BIG // 7),
    "www prefixes": "www." * (_BIG // 4),
    "long url": "https://" + "a" * _BIG,
    "spaces then text": "# a" + " " * _BIG + "x",
    "trailing hashes": "# a " + "# " * _BIG,
    "spaces between lines": ("\n" + " " * 50) * (_BIG // 50),
    "pipes": "|" * _BIG,
    "pipe rows": "| a |\n" * (_BIG // 6),
    "quote markers": ">" * _BIG,
    "one line of everything": "[a](b) `c` **d** <e> http://f " * (_BIG // 30),
    "ref openers": "[a][" * (_BIG // 4),
    "fence openers": "```\n" * (_BIG // 4),
    "escapes": "\\" * _BIG,
    "entities": "&amp;" * (_BIG // 5),
}


@pytest.mark.parametrize("name", sorted(HOSTILE))
def test_hostile_input_is_linear(name):
    text = HOSTILE[name]
    start = time.perf_counter()
    result = clean_markdown(text)
    elapsed = time.perf_counter() - start
    assert isinstance(result, str)
    assert elapsed < 3.0, f"{name} took {elapsed:.1f}s"


@pytest.mark.parametrize("bad", ["\x00", "\ud800", "￼", "﻿", "\x01", "\x0b", "\x85"])
def test_unsafe_characters_are_dropped(bad):
    assert clean_markdown(f"a{bad}b") == "ab"
    assert clean_markdown(f"**a{bad}b**") == "ab"


def test_separators_become_newlines():
    assert clean_markdown("a  b") == "a\n\nb"


@pytest.mark.parametrize("char", ["", "", "", "", "", ""])
def test_a_private_use_character_is_kept_or_dropped_never_changed(char):
    out = clean_markdown(f"x{char}y " + r"\*z\* \# " + char)
    assert out in (f"x{char}y *z* # {char}", "xy *z* #")


def test_a_private_use_character_beside_an_escape_does_not_turn_into_punctuation():
    # U+E000 stood for a backslash, U+E001 for a backtick.
    assert clean_markdown("") in ("", "")


@pytest.mark.parametrize("entity", ["&#xE000;", "&#57344;", "&#xE001;", "&#0;", "&#xD800;", "&#xFFFC;", "&#1;"])
def test_an_entity_naming_a_stand_in_or_unsafe_character_becomes_nothing_odd(entity):
    out = clean_markdown(f"a{entity}b")
    assert out in ("ab", "a�b")


def test_an_entity_cannot_build_a_speaker_tag_or_pause_marker():
    assert clean_markdown("&#91;Sam&#93;: hi &lbrack;pause:9&rsqb;") == "(Sam): hi (pause:9)"
    assert "[" not in clean_markdown("&#x5b;Sam&#x5d;: hi")
