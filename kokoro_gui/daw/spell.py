"""Spelling for the transcript (plan 22, part B).

`dictionary_for(lang_code, extra)` is a word list for the active character's
language, built on `pyspellchecker` (pure Python, imported on first use), or
None when the package is missing or the engine's language has no list. The
transcript underlines the words it doesn't contain (`unknown_spans`), and the
Find Words to Check dialog uses it as its "unknown" test.

A `Dictionary` answers `word in dictionary` for a lowercase word. The word
lists are shared by language and loaded once; `extra` (the project's character
names, the lexicon's finds) belongs to the one `Dictionary` and is never added
to the shared list. Answers are remembered per `Dictionary`, so a repaint of
the transcript asks the word list nothing it was asked before.

Qt-free, like the rest of `kokoro_gui/daw/`.
"""
from __future__ import annotations

import threading
from typing import Iterable, Optional

from kokoro_gui.daw.harvest import WORD, blank_markup

# Engine `lang_code` -> a `pyspellchecker` language. Kokoro's letters first,
# then the two-letter codes other engines use. Anything else has no list.
LANGUAGES = {
    "a": "en", "b": "en", "en": "en",
    "e": "es", "es": "es",
    "f": "fr", "fr": "fr",
    "i": "it", "it": "it",
    "p": "pt", "pt": "pt",
    "de": "de",
}

_checkers: dict = {}
_lock = threading.Lock()


def language_for(lang_code) -> Optional[str]:
    """The word-list language for an engine's `lang_code`, else None."""
    return LANGUAGES.get(lang_code) if isinstance(lang_code, str) else None


def available() -> bool:
    """True when `pyspellchecker` can be imported."""
    try:
        import spellchecker  # noqa: F401
    except ImportError:
        return False
    return True


def _checker(language: str):
    """The shared word list for `language`, or None when it can't be built."""
    with _lock:
        if language not in _checkers:
            try:
                from spellchecker import SpellChecker

                _checkers[language] = SpellChecker(language=language)
            except Exception:  # noqa: BLE001 - a missing package or word list: no spelling
                _checkers[language] = None
        return _checkers[language]


class Dictionary:
    """`word in dictionary`: true for a word the language list or `extra`
    holds, compared in lowercase with a curly apostrophe made straight and a
    possessive "'s" dropped."""

    def __init__(self, language: str, extra: Iterable[str] = (), checker=None):
        self.language = language
        self._checker = checker if checker is not None else _checker(language)
        self._extra = frozenset(self._fold(word) for word in extra if isinstance(word, str) and word.strip())
        self._answers: dict = {}

    @staticmethod
    def _fold(word: str) -> str:
        word = word.lower().replace("’", "'")
        return word[:-2] if word.endswith("'s") and len(word) > 2 else word

    @property
    def extra(self) -> frozenset:
        return self._extra

    def __contains__(self, word) -> bool:
        if not isinstance(word, str):
            return False
        folded = self._fold(word)
        answer = self._answers.get(folded)
        if answer is None:
            answer = folded in self._extra or (self._checker is not None and folded in self._checker)
            self._answers[folded] = answer
        return answer


def dictionary_for(lang_code, extra: Iterable[str] = ()) -> Optional[Dictionary]:
    """A `Dictionary` for the engine language `lang_code`, or None without
    `pyspellchecker` or a list for the language."""
    language = language_for(lang_code)
    if language is None or _checker(language) is None:
        return None
    return Dictionary(language, extra)


def unknown_spans(text: str, dictionary) -> list:
    """`(start, end)` of each word of `text` the dictionary doesn't contain.
    Tags and pause markers are not read. A word of capitals only (an
    acronym), a single letter and a word glued to a digit ("3rd") are left
    alone."""
    out = []
    plain = blank_markup(text)
    for match in WORD.finditer(plain):
        start, end = match.span()
        word = match.group(0)
        if len(word) < 2 or word.isupper():
            continue
        before = plain[start - 1] if start else " "
        after = plain[end] if end < len(plain) else " "
        if before.isdigit() or after.isdigit():
            continue
        if word not in dictionary:
            out.append((start, end))
    return out
