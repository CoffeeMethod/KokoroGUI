"""Cleanup rules for text pulled out of a book, and a guess at which
sections to skip. Pure: strings in, strings out, no Qt and no engine.

The import dialog (`kokoro_gui/qt/import_dialog.py`) shows each rule as a
checkbox and previews the result; `apply_rules` runs the ticked ones in the
order of `RULES`. Nothing here runs unless the user ticks it, and nothing
touches a project: the cleaned text goes into the transcript like any other
import.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable

PAUSE_MARKER = "[pause:1.5]"

# A line that is only a page number: digits ("12", "- 12 -"), "Page 12" or
# "Page 12 of 300", or a roman numeral in one case ("iv", "XIV"). A word that
# happens to be a valid numeral ("mix", "Liv") alone on a line goes too; the
# preview shows it and the rule can be turned off.
_ROMAN_UPPER = r"(?=[IVXLCDM])M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
_ROMAN_LOWER = r"(?=[ivxlcdm])m{0,3}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})"
_PAGE_NUMBER_LINE = re.compile(
    r"^[ \t]*(?:[-–—][ \t]*)?"
    rf"(?:\d{{1,6}}|(?i:page[ \t]+\d{{1,6}}(?:[ \t]+of[ \t]+\d{{1,6}})?)|{_ROMAN_UPPER}|{_ROMAN_LOWER})"
    r"(?:[ \t]*[-–—])?[ \t]*(?:\r?\n|\Z)",
    re.MULTILINE,
)

# A letter, a hyphen, a line break, a letter. Only a lowercase letter after the
# break makes it a split word ("exam-\nple"); "well-\nKnown" keeps its hyphen.
_HYPHEN_BREAK = re.compile(r"([^\W\d_])-[ \t]*\r?\n[ \t]*([^\W\d_])")

# A line of only `*`, `#`, `~` or em dashes (spaces between are fine).
_SCENE_BREAK_LINE = re.compile(r"[*#~—](?:[ \t]*[*#~—])*")

# A line that ends a sentence: `.`, `!`, `?` or an ellipsis, then any closing
# quotes or brackets.
_SENTENCE_END = re.compile(r"[.!?…][\"'”’)\]*_]*$")

_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANK_RUN = re.compile(r"(?:[ \t]*\r?\n){3,}")


@dataclass(frozen=True)
class CleanupRule:
    id: str
    label: str
    default_on: bool
    apply: Callable[[str], str]


def _page_numbers(text: str) -> str:
    return _PAGE_NUMBER_LINE.sub("", text)


def _hyphen_joins(text: str) -> str:
    def join(match):
        if match.group(2).islower():
            return match.group(1) + match.group(2)
        return match.group(0)

    return _HYPHEN_BREAK.sub(join, text)


def _is_break_line(stripped: str) -> bool:
    return stripped == PAUSE_MARKER or bool(_SCENE_BREAK_LINE.fullmatch(stripped))


def _soft_wraps(text: str) -> str:
    """Joins a line to the one before it with a space unless the earlier line
    ends a sentence, a blank line sits between them, or either is a scene
    break. A heading with no punctuation followed straight by text gets
    joined to it: the rule is for PDFs, where every line is a wrap."""
    lines: list = []
    open_line = False  # the last kept line is text that may continue
    for line in text.split("\n"):
        stripped = line.strip()
        if open_line and stripped and not _is_break_line(stripped):
            lines[-1] = lines[-1].rstrip() + " " + stripped
        else:
            lines.append(line)
        open_line = bool(stripped) and not _is_break_line(stripped) and not _SENTENCE_END.search(stripped)
    return "\n".join(lines)


def _scene_breaks(text: str) -> str:
    """Each scene-break line becomes `[pause:1.5]` in a paragraph of its own
    (the auto-split path turns the marker into a gap)."""
    lines = text.split("\n")
    out: list = []
    after_break = False
    for line in lines:
        stripped = line.strip()
        if _SCENE_BREAK_LINE.fullmatch(stripped):
            if out and out[-1].strip():
                out.append("")
            out.append(PAUSE_MARKER)
            after_break = True
            continue
        if after_break and stripped:
            out.append("")
        after_break = False
        out.append(line)
    return "\n".join(out)


def _collapse_blank_lines(text: str) -> str:
    return _BLANK_RUN.sub("\n\n", text)


def _strip_whitespace(text: str) -> str:
    return _TRAILING_SPACE.sub("", text)


RULES: list = [
    CleanupRule("page_numbers", "Drop page numbers (lines of only a number, a roman numeral or \"Page N\")",
                True, _page_numbers),
    CleanupRule("hyphen_joins", "Join words split by a hyphen at a line break", True, _hyphen_joins),
    CleanupRule("soft_wraps", "Join wrapped lines inside a paragraph (for PDFs)", False, _soft_wraps),
    CleanupRule("scene_breaks", "Turn \"* * *\" lines into a 1.5 second pause", True, _scene_breaks),
    CleanupRule("collapse_blank_lines", "Collapse runs of blank lines to one", True, _collapse_blank_lines),
    CleanupRule("strip_whitespace", "Strip trailing spaces", True, _strip_whitespace),
]

RULE_IDS = tuple(rule.id for rule in RULES)


def default_rule_ids() -> list:
    """The ids of the rules that start ticked."""
    return [rule.id for rule in RULES if rule.default_on]


def enabled_rule_ids(saved) -> list:
    """The ids to tick from a saved `{rule_id: bool}` (the `import_rules`
    setting). A rule the dict doesn't mention keeps its default; anything
    that isn't a dict gives the defaults."""
    saved = saved if isinstance(saved, dict) else {}
    return [rule.id for rule in RULES
            if (saved[rule.id] if isinstance(saved.get(rule.id), bool) else rule.default_on)]


def apply_rules(text: str, enabled_ids: Iterable[str]) -> str:
    """`text` through the rules named in `enabled_ids`, in `RULES` order.
    An unknown id is ignored."""
    enabled = set(enabled_ids)
    for rule in RULES:
        if rule.id in enabled:
            text = rule.apply(text)
    return text


_SKIP_TITLE = re.compile(
    r"^\W*(?:copyright|(?:table\s+of\s+)?contents|also\s+by|acknowledg(?:e)?ments?|about\s+the\s+authors?|dedication)\b",
    re.IGNORECASE,
)
_SHORT_SECTION_CHARS = 200


def guess_skip(title: str, text: str) -> bool:
    """True when a section is probably not part of the story, so the import
    dialog starts it unticked: a title that begins with copyright, contents,
    table of contents, also by, acknowledgments, about the author or
    dedication, or a section under 200 characters that is mostly digits and
    punctuation (a page of numbers, a row of asterisks)."""
    if _SKIP_TITLE.match(title or ""):
        return True
    body = "".join((text or "").split())
    if len(text or "") >= _SHORT_SECTION_CHARS:
        return False
    if not body:
        return True
    letters = sum(1 for ch in body if ch.isalpha())
    return letters * 2 < len(body)
