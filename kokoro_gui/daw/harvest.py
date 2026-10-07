"""Words worth a pronunciation rule (plan 22).

`candidates(text, known, lexicon, ignore)` reads a transcript and lists the
words a synthesis engine is most likely to get wrong: names, acronyms,
anything with a digit, and (with a dictionary) words the dictionary doesn't
know. The Find Words to Check dialog shows them, lets the user type how each
should be said, and adds the answers as whole-word rules (`whole_word_rule`).

Four kinds, a candidate holds one or two of them:

- "name": a capitalised word that is not the first word of a sentence. A
  sentence start is a line start, or a word after `. ! ? " “ …` (not after a
  title like "Dr." or an initial like "J."). Contractions ("I'm", "don't")
  are not names, and neither is a title ("Dr", "Mrs"). A possessive ("Marcus's") counts as the name.
- "acronym": a word of two or more capitals ("NASA"), unless its whole line is
  in capitals (a heading).
- "digits": any token with a digit (`1999`, `3rd`, `$4.50`, `B-52`).
- "unknown": a word `known` doesn't contain. Skipped when `known` is None.

A name's or acronym's count is every occurrence of that exact spelling once one
of them qualifies. A word the lexicon already rewrites (any rule, in any mode,
changes it, or a plain rule has it as its Find) and a name in `ignore` (the
project's characters) are left out. Tags (`[Name:FX]:`) and `[pause:x]` markers
are not scanned and never appear in a context.

Qt-free, like the rest of `kokoro_gui/daw/`.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Container, Iterable, Optional

from kokoro_gui.engine.lexicon import apply_lexicon, normalize_rules
from kokoro_gui.engine.text_extraction import PAUSE_MARKER_PATTERN

NAME = "name"
ACRONYM = "acronym"
DIGITS = "digits"
UNKNOWN = "unknown"
KINDS = (NAME, ACRONYM, DIGITS, UNKNOWN)

# Characters of transcript a context shows, counting both sides of the word.
CONTEXT_CHARS = 60

# The `[Name]:` / `[Name:FX]:` tag syntax of `text_extraction`, which keeps
# its own copy for the same reason: this module can't change what the
# conversion paths read.
_TAG = r"\[[^\]\n]{1,100}\]:\s*"
_MARKUP = re.compile(f"{_TAG}|{PAUSE_MARKER_PATTERN}")
_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
_CHUNK = re.compile(r"[^\s—–]*\d[^\s—–]*")
_APOSTROPHES = re.compile(r"['’]")
_LEAD = "(\"'“‘[{"
_TRAIL = ".,;:!?)\"'”’]}…"
# What ends a sentence, so the next word is a sentence start.
_SENTENCE_END = ".!?\"“…"
# A title or an initial before a full stop does not end a sentence.
_TITLES = frozenset("mr mrs ms mx dr prof sr jr st mt capt col gen lt sgt rev hon fr mme mlle sen rep gov pres".split())
_BEFORE_STOP = re.compile(r"(\w+)\.$")


@dataclass(frozen=True)
class Candidate:
    """`word` as it should be shown and added, `count` its occurrences,
    `kinds` a tuple from `KINDS`, `first_offset` the offset of its first
    qualifying occurrence in the transcript text given to `candidates`, and
    `context` about `CONTEXT_CHARS` characters around it (markup removed,
    whitespace collapsed)."""

    word: str
    count: int
    kinds: tuple
    first_offset: int
    context: str


def whole_word_rule(word: str, say_as: str, kinds: Iterable[str] = ()) -> dict:
    """The lexicon rule for `word` said as `say_as`: a whole-word rule, case
    sensitive for a name or an acronym ("Will" is not "will"). The stored
    replacement is a `re.sub` template, so backslashes are doubled and what
    was typed is what is spoken (the Lexicon dock does the same)."""
    kinds = tuple(kinds)
    return {"find": word, "replace": say_as.replace("\\", "\\\\"), "mode": "word",
            "case": NAME in kinds or ACRONYM in kinds}


def _blank(match: re.Match, boundary: bool) -> str:
    """The text of a markup match as spaces (newlines kept), so every offset
    stays. With `boundary` the first character is a full stop: the word after a
    tag starts a sentence."""
    body = "".join("\n" if c == "\n" else " " for c in match.group(0))
    return "." + body[1:] if boundary else body


def _covered(word: str, rules: list) -> bool:
    """True when the lexicon already speaks `word` differently."""
    if not rules:
        return False
    folded = word.casefold()
    if any(rule["mode"] != "regex" and rule["find"].casefold() == folded for rule in rules):
        return True
    return apply_lexicon(word, rules) != word


def _starts_sentence(text: str, start: int) -> bool:
    """True when the word at `start` opens a sentence or a line."""
    i = start - 1
    while i >= 0 and text[i] in " \t":
        i -= 1
    if i < 0 or text[i] == "\n":
        return True
    if text[i] not in _SENTENCE_END:
        return False
    if text[i] != ".":
        return True
    before = _BEFORE_STOP.search(text[max(0, i - 12):i + 1])
    if before is None:
        return True
    prior = before.group(1)
    return not (prior.lower() in _TITLES or (len(prior) == 1 and prior.isupper()))


def _context(plain: str, start: int, end: int) -> str:
    """About `CONTEXT_CHARS` characters of `plain` around `[start, end)`, cut
    at word boundaries, whitespace collapsed."""
    pad = max(0, (CONTEXT_CHARS - (end - start)) // 2)
    lo, hi = max(0, start - pad), min(len(plain), end + pad)
    while lo > 0 and lo < start and not plain[lo - 1].isspace():
        lo += 1
    while hi < len(plain) and hi > end and not plain[hi].isspace():
        hi -= 1
    return " ".join(plain[lo:hi].split())


def _digit_tokens(plain: str) -> list:
    """`(token, start, end)` for each whitespace-separated chunk with a digit,
    with the punctuation around it trimmed."""
    out = []
    for match in _CHUNK.finditer(plain):
        chunk, start, end = match.group(0), match.start(), match.end()
        lead = len(chunk) - len(chunk.lstrip(_LEAD))
        chunk = chunk[lead:]
        start += lead
        trimmed = chunk.rstrip(_TRAIL)
        end -= len(chunk) - len(trimmed)
        if any(c.isdigit() for c in trimmed):
            out.append((trimmed, start, end))
    return out


def candidates(text: str, known: Optional[Container[str]] = None, lexicon=None,
               ignore: Iterable[str] = ()) -> list:
    """The `Candidate`s in `text`, most frequent first. `known` is a
    container of lowercase words (`word in known`), None to skip the
    "unknown" kind. `lexicon` is the rule list (or the old dict) whose words are
    left out, `ignore` more words to leave out (case-insensitive)."""
    scan = _MARKUP.sub(lambda m: _blank(m, True), text)
    plain = _MARKUP.sub(lambda m: _blank(m, False), text)
    rules = normalize_rules(lexicon)
    skip = {name.casefold() for name in ignore if isinstance(name, str)}

    digit_tokens = _digit_tokens(plain)
    digit_spans = [(start, end) for _token, start, end in digit_tokens]

    spelling_count: Counter = Counter()
    first_any: dict = {}
    qualified: dict = {}  # spelling -> (kind, offset of the first qualifying occurrence)
    span_index = 0
    for match in _WORD.finditer(scan):
        start, end = match.span()
        # A word glued to a digit chunk ("3rd", "B-52") belongs to the token.
        while span_index < len(digit_spans) and digit_spans[span_index][1] <= start:
            span_index += 1
        if span_index < len(digit_spans) and digit_spans[span_index][0] < end:
            continue
        word = match.group(0)
        if word[-2:] in ("'s", "’s", "'S", "’S") and len(word) > 2:
            word, end = word[:-2], end - 2
        if _APOSTROPHES.search(word) and all(p.islower() for p in _APOSTROPHES.split(word)[1:]):
            continue  # a contraction: don't, I'm
        letters = len(_APOSTROPHES.sub("", word))
        if letters < 2:
            continue
        spelling_count[word] += 1
        first_any.setdefault(word, start)
        if word in qualified:
            continue
        if word.isupper():
            line_start = scan.rfind("\n", 0, start) + 1
            line_end = scan.find("\n", end)
            line = scan[line_start:len(scan) if line_end < 0 else line_end]
            if any(c.islower() for c in line):
                qualified[word] = (ACRONYM, start)
        elif word[0].isupper() and word.lower() not in _TITLES and not _starts_sentence(scan, start):
            qualified[word] = (NAME, start)

    found: dict = {}  # shown word -> [count, kinds list, first offset]
    for word, (kind, offset) in qualified.items():
        kinds = [kind]
        if known is not None and word.lower() not in known and kind == NAME:
            kinds.append(UNKNOWN)
        found[word] = [spelling_count[word], kinds, offset]

    if known is not None:
        grouped: dict = {}  # lowercase -> [count, first offset, spelling]
        for word, count in spelling_count.items():
            if word in qualified or word.isupper() or word.lower() in known:
                continue
            slot = grouped.setdefault(word.lower(), [0, first_any[word], word])
            slot[0] += count
            if first_any[word] < slot[1]:
                slot[1], slot[2] = first_any[word], word
        for _low, (count, offset, spelling) in grouped.items():
            found[spelling] = [count, [UNKNOWN], offset]

    for token, start, _end in digit_tokens:
        slot = found.setdefault(token, [0, [DIGITS], start])
        slot[0] += 1
        slot[2] = min(slot[2], start)

    out = []
    for word, (count, kinds, offset) in found.items():
        if word.casefold() in skip or _covered(word, rules):
            continue
        out.append(Candidate(word=word, count=count, kinds=tuple(k for k in KINDS if k in kinds),
                             first_offset=offset, context=_context(plain, offset, offset + len(word))))
    out.sort(key=lambda c: (-c.count, c.first_offset))
    return out
