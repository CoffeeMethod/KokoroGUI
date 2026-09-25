"""Where text is cut before synthesis: one piece per pipeline call, one
file and one `Segment` per piece.

Why it matters: Kokoro takes about 510 phoneme tokens per call and cuts
longer input wherever the limit lands, and every seam resets the voice a
little (a short pause, a new intonation contour). So the app decides the
seams and puts them where a reader would pause anyway.

The rule (grill PR6). Aim for `segment_target_words` words per piece. A
piece ends at the strongest enabled boundary near that target:

    3 paragraph   a blank line
    2 sentence    `.`, `!`, `?` or `...` (abbreviations and initials excluded)
    1 pause       `,` `;` `:` a dash, or a single line break
    0 word        any space

Each of the first three is a toggle (`segment_at_paragraphs`,
`segment_at_sentences`, `segment_at_pauses`, all on by default). A boundary
counts at the strongest *enabled* level at or below its own, so with
sentences off a full stop still counts as a pause. The search runs in two
bands: first within 0.5x to 1.5x the target, strongest boundary wins, ties
go to the one nearest the target; then out to 2x, the hard limit. Only
when neither band has any enabled boundary does a piece end at a plain
word break, the one nearest the target. A piece never splits a word.

Pure over its inputs; `caching.split_segments` is the name everything else
imports.
"""
import math
import re

DEFAULT_TARGET_WORDS = 40
MIN_TARGET_WORDS = 5
MAX_TARGET_WORDS = 200
# A piece may run to this multiple of the target to reach a boundary; past
# it, a sentence is cut (at a pause if it has one, else between words).
HARD_LIMIT_FACTOR = 2.0
_BAND_LOW = 0.5
_BAND_HIGH = 1.5

PARAGRAPH, SENTENCE, PAUSE, WORD = 3, 2, 1, 0

BOUNDARY_KEYS = {
    PARAGRAPH: "segment_at_paragraphs",
    SENTENCE: "segment_at_sentences",
    PAUSE: "segment_at_pauses",
}

_TOKEN_RE = re.compile(r"\S+")
_CLOSERS = "\"'”’)]}»"
_SENTENCE_END_RE = re.compile(r"(?:[.!?]+|…)[" + re.escape(_CLOSERS) + r"]*$")
_PAUSE_END_RE = re.compile(r"[,;:—–][" + re.escape(_CLOSERS) + r"]*$")
_DASH_TOKENS = {"-", "--", "–", "—"}
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "st", "sr", "jr", "vs", "etc", "e.g", "i.e",
    "mt", "no", "fig", "vol", "ch", "approx", "dept", "inc", "ltd", "co",
}


def _natural_level(token, next_token, gap):
    """How strong the boundary after `token` is, before the toggles."""
    if "\n" in gap and re.search(r"\n[^\S\n]*\n", gap):
        return PARAGRAPH
    if _SENTENCE_END_RE.search(token):
        word = token.rstrip(_CLOSERS).rstrip(".!?…").lower()
        is_initial = len(word) == 1 and word.isalpha() and token.rstrip(_CLOSERS).endswith(".")
        if word not in _ABBREVIATIONS and not is_initial:
            return SENTENCE
    if "\n" in gap or _PAUSE_END_RE.search(token) or next_token in _DASH_TOKENS:
        return PAUSE
    return WORD


def _enabled_level(level, enabled):
    """The strongest enabled level at or below `level`, else WORD."""
    for candidate in range(level, WORD, -1):
        if candidate in enabled:
            return candidate
    return WORD


def target_words(config):
    try:
        value = int(config.get("segment_target_words") or DEFAULT_TARGET_WORDS)
    except (TypeError, ValueError):
        value = DEFAULT_TARGET_WORDS
    return max(MIN_TARGET_WORDS, min(MAX_TARGET_WORDS, value))


def enabled_levels(config):
    return {level for level, key in BOUNDARY_KEYS.items() if config.get(key, True)}


def _choose_cut(levels, start, n, target):
    """The index of the last word of the piece that starts at `start`."""
    low = start + max(1, math.ceil(target * _BAND_LOW)) - 1
    high = start + math.floor(target * _BAND_HIGH) - 1
    hard = start + max(1, math.floor(target * HARD_LIMIT_FACTOR)) - 1
    ideal = start + target - 1

    def best(lo, hi):
        lo, hi = max(lo, start), min(hi, n - 1)
        found = None
        for i in range(lo, hi + 1):
            if levels[i] == WORD:
                continue
            key = (levels[i], -abs(i - ideal))
            if found is None or key > found[0]:
                found = (key, i)
        return None if found is None else found[1]

    cut = best(low, high)
    if cut is None:
        cut = best(high + 1, hard)
    if cut is None:
        cut = best(start, low - 1)  # a boundary early is better than a word break
    if cut is None:
        cut = min(ideal, n - 1)  # no enabled boundary anywhere near: word break
    return cut


def split_text(text, config):
    """The pieces of `text`, each a run of whole words joined by single
    spaces. Empty or blank text gives `[]`."""
    matches = list(_TOKEN_RE.finditer(text or ""))
    if not matches:
        return []
    tokens = [m.group() for m in matches]
    n = len(tokens)
    target = target_words(config)
    enabled = enabled_levels(config)

    levels = []
    for i, token in enumerate(tokens):
        if i == n - 1:
            levels.append(PARAGRAPH)  # the end of the text
            continue
        gap = text[matches[i].end():matches[i + 1].start()]
        levels.append(_enabled_level(_natural_level(token, tokens[i + 1], gap), enabled))

    pieces = []
    start = 0
    while start < n:
        remaining = n - start
        if remaining <= math.floor(target * _BAND_HIGH):
            cut = n - 1
        else:
            cut = _choose_cut(levels, start, n, target)
            tail = n - 1 - cut
            if 0 < tail < math.ceil(target * _BAND_LOW) and remaining <= math.floor(target * HARD_LIMIT_FACTOR):
                cut = n - 1  # don't leave a scrap of a few words on its own
        pieces.append(" ".join(tokens[start:cut + 1]))
        start = cut + 1
    return pieces
