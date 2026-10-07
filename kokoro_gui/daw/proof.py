"""Proof by ASR (plan 21): score how closely a clip's audio matches its text.

The caller transcribes each segment of a clip (Whisper, off the GUI thread)
and hands the heard words here. `score` compares them with the words the
engine was asked to speak, which are the stored `Segment.text`s: markup and
the lexicon are already applied to those, so a lexicon rewrite is not a
mismatch. Both sides go through `normalize_words` first.

Results live in `session.json["proof"]`, never in a `Clip` or the bundle.
An entry is valid while the clip's active-take segment keys equal the ones it
was scored against (`segment_keys`), so a regenerate drops it by itself.
`clean_results` type-checks what is read back, since the file is untrusted
input like the rest of `session.json`.

Qt-free and model-free.
"""
from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field

DEFAULT_THRESHOLD = 0.92
MIN_THRESHOLD, MAX_THRESHOLD = 0.5, 1.0
# Stored issues per clip, and the longest phrase kept in one.
MAX_ISSUES = 20
MAX_ISSUE_CHARS = 120
# Entries kept in `session.json["proof"]`; a book has a few thousand clips.
MAX_ENTRIES = 50_000
MAX_KEYS = 500

DROPPED, ADDED, CHANGED = "dropped", "added", "changed"
KINDS = (DROPPED, ADDED, CHANGED)

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
         "seventeen eighteen nineteen").split()
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}
_WORD = re.compile(r"\w+", re.UNICODE)
_APOSTROPHES = "'’‘`"
# Longest run of digits spelled out. A longer one stays as digits.
_MAX_DIGITS = 4


def _below_hundred(n: int) -> list:
    if n < 20:
        return [_ONES[n]]
    tens, ones = divmod(n, 10)
    return [_TENS[tens]] + ([_ONES[ones]] if ones else [])


def spell_number(token: str) -> list:
    """The words for a run of digits: 0 to 100 as a number ("21" is "twenty
    one"), 1000 to 2099 as a year ("1999" is "nineteen ninety nine", "2007"
    is "two thousand seven"), and anything else, including 101 to 999 and
    anything over four digits, left as the digits themselves. The rule is
    one fixed table, not a number reader: both sides of a comparison go
    through it, so only a number the speaker reads some other way ("1999"
    as "one thousand nine hundred ninety nine") shows up as a mismatch."""
    if not token.isdigit() or len(token) > _MAX_DIGITS:
        return [token]
    n = int(token)
    if token != str(n):  # "007" is not the number 7
        return [token]
    if n <= 100:
        return ["one", "hundred"] if n == 100 else _below_hundred(n)
    if 1000 <= n <= 2099:
        head, tail = divmod(n, 100)
        if n < 2000 and tail == 0 and head == 10:
            return ["one", "thousand"]
        if 2000 <= n <= 2009:
            return ["two", "thousand"] + ([_ONES[tail]] if tail else [])
        words = _below_hundred(head)
        if tail == 0:
            return words + ["hundred"]
        return words + (["oh"] if tail < 10 else []) + _below_hundred(tail)
    return [token]


def normalize_words(text: str) -> list:
    """`text` as comparable words: lower case, punctuation and apostrophes
    dropped, a hyphen or dash a word break ("ninety-nine" is two words),
    and digits spelled by `spell_number`."""
    cleaned = (text or "").lower()
    for mark in _APOSTROPHES:
        cleaned = cleaned.replace(mark, "")
    words: list = []
    for token in _WORD.findall(cleaned):
        token = token.replace("_", "")
        if token:
            words.extend(spell_number(token))
    return words


@dataclass(frozen=True)
class ProofResult:
    """`ratio` is 1.0 for an exact match and 0.0 for nothing in common.
    `issues` is `[(kind, expected, heard)]` in reading order: `kind` is
    "dropped" (in the text, not heard), "added" (heard, not in the text: a
    repeat or a hallucination) or "changed" (heard as something else)."""
    ratio: float
    issues: tuple = field(default_factory=tuple)


def _clip_phrase(words) -> str:
    phrase = " ".join(words)
    return phrase if len(phrase) <= MAX_ISSUE_CHARS else phrase[:MAX_ISSUE_CHARS - 1] + "…"


def score(expected_text: str, heard_words) -> ProofResult:
    """Compares `expected_text` with the words the ASR heard. `heard_words`
    is a list of words, or of the `(word, start_s, end_s)` triples
    `asr.transcribe_wav_words` returns. Two empty sides match."""
    expected = normalize_words(expected_text)
    heard: list = []
    for item in heard_words or []:
        word = item if isinstance(item, str) else (item[0] if item else "")
        heard.extend(normalize_words(str(word)))
    if not expected and not heard:
        return ProofResult(1.0, ())
    matcher = difflib.SequenceMatcher(a=expected, b=heard, autojunk=False)
    issues = []
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        if op == "equal":
            continue
        kind = {"delete": DROPPED, "insert": ADDED}.get(op, CHANGED)
        issues.append((kind, _clip_phrase(expected[a0:a1]), _clip_phrase(heard[b0:b1])))
    return ProofResult(round(matcher.ratio(), 4), tuple(issues))


def flagged(result: ProofResult, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """True when the match is below `threshold`."""
    return result.ratio < threshold


def clamp_threshold(value) -> float:
    """A usable threshold from a stored setting; the default for junk."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return DEFAULT_THRESHOLD
    return max(MIN_THRESHOLD, min(MAX_THRESHOLD, float(value)))


def issue_text(issue) -> str:
    """One issue as a line: `dropped: 'the old mill'`, `added: 'uh'`,
    `changed: 'mill' heard as 'meal'`."""
    kind, expected, heard = issue
    if kind == DROPPED:
        return f"dropped: '{expected}'"
    if kind == ADDED:
        return f"added: '{heard}'"
    return f"changed: '{expected}' heard as '{heard}'"


def segment_keys(clip) -> list:
    """The identity of a clip's active-take audio: one `<key>:<index>` per
    segment in order. A regenerate (a new take) or any edit that changes a
    segment's key changes this list."""
    return [f"{s.cache_key or ''}:{s.order_index}" for s in sorted(clip.segments, key=lambda s: s.order_index)]


def expected_text(clip) -> str:
    """What the engine was asked to speak for `clip`: its segments' stored
    texts in order. Those are over the spoken text (inline tags, pause
    markers and a leading `[Name:FX]:` tag removed, lexicon applied), so
    nothing else needs undoing here."""
    return " ".join(s.text for s in sorted(clip.segments, key=lambda s: s.order_index) if s.text)


def make_entry(result: ProofResult, keys: list) -> dict:
    """The `session.json["proof"][clip_id]` record for a scored clip."""
    return {"ratio": result.ratio, "issues": [list(i) for i in result.issues[:MAX_ISSUES]],
            "segment_keys": list(keys), "ok": False}


def entry_result(entry: dict) -> ProofResult:
    return ProofResult(float(entry["ratio"]), tuple(tuple(i) for i in entry.get("issues", ())))


def is_current(entry, clip) -> bool:
    """Whether `entry` was scored against the audio `clip` has now."""
    return isinstance(entry, dict) and entry.get("segment_keys") == segment_keys(clip)


def is_flagged(entry, clip, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """A current entry below the threshold that the user hasn't marked OK."""
    return (is_current(entry, clip) and not entry.get("ok")
            and flagged(entry_result(entry), threshold))


def _clean_issue(raw):
    if (not isinstance(raw, (list, tuple)) or len(raw) != 3 or raw[0] not in KINDS
            or not all(isinstance(part, str) for part in raw)):
        return None
    return (raw[0], raw[1][:MAX_ISSUE_CHARS], raw[2][:MAX_ISSUE_CHARS])


def clean_results(raw) -> dict:
    """The valid part of a `proof` value read from `session.json`:
    `{clip_id: entry}`. An entry needs a finite `ratio` in 0..1 and a list
    of string `segment_keys`; `issues` and `ok` are checked or dropped.
    Anything that isn't a dict gives {}."""
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for clip_id, entry in raw.items():
        if len(out) >= MAX_ENTRIES:
            break
        if not isinstance(clip_id, str) or not 0 < len(clip_id) <= 200 or not isinstance(entry, dict):
            continue
        ratio = entry.get("ratio")
        keys = entry.get("segment_keys")
        if (isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio)
                or not 0.0 <= ratio <= 1.0):
            continue
        if not isinstance(keys, list) or len(keys) > MAX_KEYS or not all(isinstance(k, str) for k in keys):
            continue
        raw_issues = entry.get("issues")
        raw_issues = raw_issues[:MAX_ISSUES] if isinstance(raw_issues, list) else []
        issues = [i for i in map(_clean_issue, raw_issues) if i is not None]
        out[clip_id] = {"ratio": float(ratio), "issues": [list(i) for i in issues], "segment_keys": list(keys),
                        "ok": entry.get("ok") is True}
    return out
