"""Dialogue attribution (plan 26): who speaks each quote in a transcript.

Pure planning, no Qt, like `auto_split.py`. `find_quotes` finds the quoted
speech, `find_tag` reads the speech tag next to one quote (`said Marta`,
`Marta asked`, `she whispered`), and `plan` turns the whole thing into
`Assignment`s: a character per quote where it can tell, the narrator for the
text between quotes when one was chosen. `qt/attribution_dialog.py` shows the
plan for review and applies it as `AssignCharacterCommand`s in one
`CompositeCommand`, so the result is ordinary clips and one Undo.

The heuristics guess. A quote with no tag, a pronoun tag (`she said`) or a
name that is no character comes back unassigned or with a lower confidence,
and the reviewer settles it.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Optional

from kokoro_gui.i18n import tr

HIGH = "high"
MEDIUM = "medium"
LOW = "low"

# Verbs a speech tag uses. Past tense, plus the present forms of the common
# ones for books told in the present.
SPEECH_VERBS = (
    "said", "says", "asked", "asks", "replied", "replies", "answered", "answers", "whispered", "whispers",
    "shouted", "shouts", "called", "calls", "cried", "cries", "murmured", "murmurs", "added", "adds",
    "continued", "continues", "began", "begins", "told", "tells", "snapped", "snaps", "muttered", "mutters",
    "yelled", "yells", "sighed", "sighs", "laughed", "laughs", "exclaimed", "demanded", "insisted",
    "explained", "repeated", "agreed", "admitted", "protested", "suggested", "warned", "wondered",
    "growled", "hissed", "screamed", "breathed", "corrected", "offered", "pleaded", "retorted",
    "responded", "interrupted", "stammered", "whimpered", "gasped", "groaned", "chuckled", "grumbled",
    "mumbled", "sobbed", "ventured", "announced", "declared", "observed", "remarked", "conceded",
)
PRONOUNS = ("he", "she", "they", "i", "we")
HONORIFICS = ("mr", "mrs", "ms", "mx", "dr", "prof", "st", "sir", "lady", "lord", "miss", "aunt", "uncle")

# How far from a quote a tag may start, in characters.
TAG_WINDOW = 80

_MARKS = re.compile("[\"“”‘’\n]")
_DOUBLE_OPEN = ('"', "“")
_DOUBLE_CLOSE = ('"', "”")
_SINGLE_OPEN = "‘"
_SINGLE_CLOSE = "’"
# Marks that end a tag window: any double quote, an opening single quote.
_WINDOW_STOP = re.compile("[\"“”‘\n]")

_VERB = "(?i:" + "|".join(SPEECH_VERBS) + ")"
_UPPER = "[A-ZÀ-ÖØ-Þ]"
_TOKEN = _UPPER + "[\\w’'-]*"
_HONOR = "(?:" + "|".join(h.capitalize() for h in HONORIFICS) + ")\\.?"
_WHO = (f"(?:(?:{_HONOR}\\s+)?{_TOKEN}(?:\\s+{_TOKEN})?"
        "|[Hh]e|[Ss]he|[Tt]hey|I|[Ww]e)")
# After the quote: `, said Marta` or `, Marta said` at the start of the window.
_AFTER_VERB_WHO = re.compile(rf"^[\s,—–-]*(?P<tag>(?P<verb>{_VERB})\s+(?P<who>{_WHO}))(?![\w’'])")
_AFTER_WHO_VERB = re.compile(
    rf"^[\s,—–-]*(?P<tag>(?P<who>{_WHO})\s+(?:\w+ly\s+)?(?P<verb>{_VERB}))(?![\w’'])")
# Before the quote: `Marta said, ` or `Said Marta: ` at the end of the window.
_BEFORE_WHO_VERB = re.compile(
    rf"(?<![\w’'])(?P<tag>(?P<who>{_WHO})\s+(?:\w+ly\s+)?(?P<verb>{_VERB})(?:\s+\w+){{0,2}})\s*[,:]?\s*$")
_BEFORE_VERB_WHO = re.compile(
    rf"(?<![\w’'])(?P<tag>(?P<verb>{_VERB})\s+(?P<who>{_WHO}))\s*[,:]?\s*$")


@dataclass(frozen=True)
class Quote:
    """A stretch of quoted speech. `[start, end)` holds the quote marks,
    `[inner_start, inner_end)` the words between them. A quote that runs on
    into the next paragraph (the paragraph ends without a closing mark and
    the next opens with one) is one `Quote` over both."""
    start: int
    end: int
    inner_start: int
    inner_end: int


@dataclass(frozen=True)
class Tag:
    """A speech tag next to a quote. `speaker` is the entry of `names` the
    tag named, `pronoun` the pronoun it used instead, `other` a capitalised
    name that is in neither. `text` is the tag as written."""
    speaker: Optional[str]
    pronoun: Optional[str]
    confidence: str
    text: str = ""
    other: Optional[str] = None


@dataclass(frozen=True)
class Assignment:
    """One planned range: a whole quote, or a stretch of narration.
    `character_id` None means the plan couldn't tell."""
    start: int
    end: int
    character_id: Optional[str]
    confidence: str
    reason: str
    is_quote: bool = True


# -- quotes --------------------------------------------------------------------


def _is_letter(ch: str) -> bool:
    return ch.isalpha()


def _paragraph_end(text: str, pos: int) -> int:
    """The end of the paragraph `pos` is in, trailing spaces trimmed."""
    nl = text.find("\n", pos)
    end = len(text) if nl < 0 else nl
    while end > pos and text[end - 1] in " \t\r":
        end -= 1
    return end


def _next_content(text: str, pos: int) -> int:
    """The first offset at or after `pos` that is not whitespace."""
    n = len(text)
    while pos < n and text[pos].isspace():
        pos += 1
    return pos


def find_quotes(text: str) -> list:
    """Every quote in `text`, in order: straight `"..."`, curly `“...”`,
    and `‘...’` when the closing mark isn't an apostrophe (a `’`
    between two letters, as in `don’t`). Quotes inside a quote are part
    of it. A quote whose closing mark is missing ends at its paragraph's
    end, unless the next paragraph opens with the same kind of mark: then
    the speech continues there (the typographic convention for a speech of
    several paragraphs). One pass over the marks."""
    quotes: list = []
    kind = None  # "d" or "s" while inside a quote
    start = 0
    skip_until = -1
    n = len(text)
    for m in _MARKS.finditer(text):
        i = m.start()
        if i < skip_until:
            continue
        c = text[i]
        if c == "\n":
            if kind is None:
                continue
            j = _next_content(text, i)
            opener = text[j] if j < n else ""
            if (kind == "d" and opener in _DOUBLE_OPEN) or (kind == "s" and opener == _SINGLE_OPEN):
                skip_until = j + 1
                continue
            end = _paragraph_end(text, start)
            quotes.append(Quote(start, end, start + 1, end))
            kind = None
            continue
        if kind == "d":
            if c in _DOUBLE_CLOSE:
                quotes.append(Quote(start, i + 1, start + 1, i))
                kind = None
            continue
        if kind == "s":
            if c == _SINGLE_CLOSE:
                before = text[i - 1] if i > 0 else ""
                after = text[i + 1] if i + 1 < n else ""
                if not (_is_letter(before) and _is_letter(after)):
                    quotes.append(Quote(start, i + 1, start + 1, i))
                    kind = None
            continue
        if c in _DOUBLE_OPEN:
            kind, start = "d", i
        elif c == _SINGLE_OPEN:
            before = text[i - 1] if i > 0 else ""
            if not before.isalnum():
                kind, start = "s", i
    if kind is not None:
        end = _paragraph_end(text, start)
        quotes.append(Quote(start, end, start + 1, end))
    return quotes


# -- tags ----------------------------------------------------------------------


@lru_cache(maxsize=64)
def _name_lookup(names: tuple) -> dict:
    return {name.strip().lower(): name for name in names if name and name.strip()}


def _resolve(who: str, lookup: dict):
    """`(speaker, pronoun, other)` for the words a tag named."""
    if who.lower() in PRONOUNS:
        return None, who.lower(), None
    if who.lower() in lookup:
        return lookup[who.lower()], None, None
    tokens = [t for t in who.split() if t.rstrip(".").lower() not in HONORIFICS]
    for token in tokens:
        if token.lower() in lookup:
            return lookup[token.lower()], None, None
    return None, None, who


def _window_after(text: str, quote: Quote) -> str:
    hi = min(len(text), quote.end + TAG_WINDOW)
    stop = _WINDOW_STOP.search(text, quote.end, hi)
    return text[quote.end:stop.start() if stop else hi]


def _window_before(text: str, quote: Quote) -> str:
    lo = max(0, quote.start - TAG_WINDOW)
    window = text[lo:quote.start]
    cut = -1
    for k in range(len(window) - 1, -1, -1):
        c = window[k]
        if c in "\n\"“”‘":
            cut = k
            break
        if c == _SINGLE_CLOSE:
            before = window[k - 1] if k > 0 else ""
            after = window[k + 1] if k + 1 < len(window) else ""
            if not (_is_letter(before) and _is_letter(after)):
                cut = k
                break
    return window[cut + 1:]


def _tag_from(match, lookup: dict) -> Optional[Tag]:
    speaker, pronoun, other = _resolve(match.group("who"), lookup)
    text = " ".join(match.group("tag").split())
    if speaker is not None:
        return Tag(speaker, None, HIGH, text)
    if pronoun is not None:
        return Tag(None, pronoun, LOW, text)
    return Tag(None, None, LOW, text, other)


def find_tag(text: str, quote: Quote, names: Iterable[str]) -> Optional[Tag]:
    """The speech tag right after `quote` (`, said Marta`, `, Marta said`)
    or right before it (`Marta said, `), within the same sentence and
    paragraph. After wins when both are there. `names` are the names that
    count as speakers, matched case-insensitively; the returned `speaker`
    is the entry of `names` as given. A tag with a pronoun has `pronoun`
    set and low confidence; one with a capitalised name not in `names` has
    `other` set and low confidence. None when there is no tag."""
    lookup = _name_lookup(tuple(names))
    after = _window_after(text, quote)
    for pattern in (_AFTER_VERB_WHO, _AFTER_WHO_VERB):
        match = pattern.match(after)
        if match:
            return _tag_from(match, lookup)
    before = _window_before(text, quote)
    for pattern in (_BEFORE_WHO_VERB, _BEFORE_VERB_WHO):
        match = pattern.search(before)
        if match:
            return _tag_from(match, lookup)
    return None


# -- plan ----------------------------------------------------------------------


def name_variants(characters) -> dict:
    """`{lowercased name: character id}` for each character's full name,
    its first name and, for a name of several words, its last. An honorific
    (`Mr.`) is never a variant. A short form two characters share is
    dropped; a full name always wins."""
    full: dict = {}
    short: dict = {}
    for character in characters:
        name = (character.name or "").strip()
        if not name:
            continue
        full.setdefault(name.lower(), character.id)
        tokens = [t for t in name.split() if t.rstrip(".").lower() not in HONORIFICS]
        variants = {tokens[0], tokens[-1]} if tokens else set()
        for variant in variants:
            short.setdefault(variant.lower(), set()).add(character.id)
    out = {key: ids.pop() for key, ids in short.items() if len(ids) == 1}
    out.update(full)
    return out


class _Unit:
    """The quotes of one paragraph (or of the paragraphs one long speech
    runs across): the speaker the tags in it name, if it is one."""
    __slots__ = ("first", "last", "rows", "speakers", "others")

    def __init__(self, first: int, last: int):
        self.first = first
        self.last = last
        self.rows: list = []  # [quote, tag, character_id, confidence, reason]
        self.speakers: set = set()
        self.others: set = set()

    def speaker(self) -> Optional[str]:
        if len(self.speakers) == 1 and not self.others:
            return next(iter(self.speakers))
        return None

    def is_open(self) -> bool:
        return not self.speakers and not self.others


def _paragraph_starts(text: str) -> list:
    """The start offset of every paragraph that isn't blank, in order."""
    starts = []
    pos = 0
    for line in text.split("\n"):
        if line.strip():
            starts.append(pos)
        pos += len(line) + 1
    return starts


def _narration(text: str, lo: int, hi: int) -> list:
    """`(start, end)` for each paragraph's piece of `text[lo:hi]` with a
    letter or digit in it, whitespace trimmed."""
    pieces = []
    pos = lo
    while pos < hi:
        nl = text.find("\n", pos, hi)
        end = hi if nl < 0 else nl
        a, b = pos, end
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if any(ch.isalnum() for ch in text[a:b]):
            pieces.append((a, b))
        pos = end + 1
    return pieces


def _tag_reason(tag: Optional[Tag]) -> str:
    if tag is None:
        return tr("No tag")
    if tag.speaker is not None:
        return tr("Tag: {text}", text=tag.text)
    if tag.pronoun is not None:
        return tr("Pronoun tag: {text}", text=tag.text)
    return tr("Not a character: {other}", other=tag.other)


def plan(text: str, characters, narrator_id: Optional[str] = None, start: int = 0,
         end: Optional[int] = None) -> list:
    """Who speaks each quote in `text[start:end]`, as `Assignment`s sorted by
    `start`, with offsets into `text`.

    1. A quote whose tag names a character goes to that character (high).
    2. Untagged and pronoun-tagged quotes in a paragraph whose tags name one
       character go to that character (medium).
    3. In a run of consecutive paragraphs that all hold quotes, with exactly
       two speakers named in the run, a paragraph nobody is named in takes
       turns from the nearest named one: A, B, A, B (medium). A third
       speaker, or a name that is no character, stops the guess.
    4. With `narrator_id`, each paragraph's text outside the quotes (the
       tags included) goes to the narrator, one assignment per paragraph.
    5. A quote nothing settles keeps `character_id` None (low).
    """
    end = len(text) if end is None else end
    scope = text[start:end]
    names_map = name_variants(characters)
    names = tuple(names_map)
    by_id = {c.id: c for c in characters}
    paragraphs = _paragraph_starts(scope)

    def paragraph_of(offset: int) -> int:
        return bisect.bisect_right(paragraphs, offset) - 1

    units: list = []
    for quote in find_quotes(scope):
        first, last = paragraph_of(quote.start), paragraph_of(max(quote.start, quote.end - 1))
        if units and first <= units[-1].last:
            unit = units[-1]
            unit.last = max(unit.last, last)
        else:
            unit = _Unit(first, last)
            units.append(unit)
        tag = find_tag(scope, quote, names)
        character_id = names_map.get(tag.speaker.lower()) if tag and tag.speaker else None
        if character_id is not None:
            unit.speakers.add(character_id)
            unit.rows.append([quote, tag, character_id, HIGH, _tag_reason(tag)])
            continue
        if tag is not None and tag.other:
            unit.others.add(tag.other)
        unit.rows.append([quote, tag, None, LOW, _tag_reason(tag)])

    # Step 2: one speaker named in the paragraph speaks all of it.
    for unit in units:
        speaker = unit.speaker()
        if speaker is None:
            continue
        for row in unit.rows:
            if row[2] is None:
                name = by_id[speaker].name
                row[2], row[3] = speaker, MEDIUM
                row[4] = tr("Same paragraph as {name}", name=name) + (f" ({row[1].text})" if row[1] else "")

    # Step 3: turn-taking inside a run of quoted paragraphs.
    run: list = []
    for unit in units + [None]:
        if unit is not None and (not run or unit.first == run[-1].last + 1):
            run.append(unit)
            continue
        _alternate(run, by_id)
        run = [unit] if unit is not None else []

    out = []
    for unit in units:
        for quote, _tag, character_id, confidence, reason in unit.rows:
            out.append(Assignment(start + quote.start, start + quote.end, character_id, confidence, reason))

    if narrator_id is not None:
        cursor = 0
        quotes = [(a.start - start, a.end - start) for a in out]
        for q_start, q_end in quotes + [(len(scope), len(scope))]:
            for a, b in _narration(scope, cursor, q_start):
                out.append(Assignment(start + a, start + b, narrator_id, HIGH, "Narration", is_quote=False))
            cursor = max(cursor, q_end)
    out.sort(key=lambda a: a.start)
    return out


def _alternate(run: list, by_id: dict) -> None:
    """Fills the open paragraphs of `run` by turn-taking (step 3 of `plan`)."""
    if len(run) < 2:
        return
    speakers = set()
    others = set()
    for unit in run:
        speakers |= unit.speakers
        others |= unit.others
    if len(speakers) != 2 or others:
        return
    anchors = [(i, unit.speaker()) for i, unit in enumerate(run) if unit.speaker() is not None]
    if not anchors:
        return
    a, b = sorted(speakers)
    for i, unit in enumerate(run):
        if not unit.is_open():
            continue
        k, anchor = min(anchors, key=lambda pair: (abs(pair[0] - i), pair[0] > i))
        speaker = anchor if (i - k) % 2 == 0 else (b if anchor == a else a)
        other = b if speaker == a else a
        for row in unit.rows:
            row[2], row[3] = speaker, MEDIUM
            row[4] = tr("Turn-taking with {name}", name=by_id[other].name) + (f" ({row[1].text})" if row[1] else "")
