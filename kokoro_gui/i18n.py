"""Interface translation. No Qt import: `daw/`, `core/` and the engines use it too.

The English text in the source is the key, so English never reads a file and
a string with no translation shows in English. `kokoro_gui/locales/<code>.json`
holds one language:

    {"language": "fr", "name": "Français",
     "strings": {"Play": "Lire"},
     "plurals": {"{n} clips": {"one": "{n} clip", "other": "{n} clips"}}}

`locales/en.json` is the template. `scripts/i18n.py` rewrites it from the
source, adds new keys to the other files and checks them.

- `tr("Saved {name}.", name=name)` translates, then fills the placeholders.
  The text must be a plain string literal, never an f-string, or
  `scripts/i18n.py` can't find it.
- `ntr("{n} clip", "{n} clips", count)` picks the plural form.
- `N_("Play")` marks a string in a table built at import time, before the
  language is known. Whoever shows it calls `tr()` on it.

Only `main.py` picks a language (`qt/localize.py`), so tests and the command
line stay in English. What a project stores, the names of characters and
tracks included, is never translated: it has to read the same on every
machine.
"""
from __future__ import annotations

import json
import os
import re
import string
from typing import Optional

LOCALES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locales")
SOURCE_LANGUAGE = "en"
# The setting's value for "whatever the operating system uses".
SYSTEM_LANGUAGE = "system"

MAX_CATALOG_BYTES = 8 * 1024 * 1024
_CODE_RE = re.compile(r"[a-z]{2,3}(?:_[A-Za-z0-9]{2,8})*")

_language = SOURCE_LANGUAGE
_strings: dict = {}
_plurals: dict = {}


def N_(text: str) -> str:
    """Marks `text` for `scripts/i18n.py` and returns it unchanged."""
    return text


def tr(text: str, /, **values) -> str:
    """`text` in the current language, with its `{placeholders}` filled from
    `values`. Without `values` the text is returned as it is, so a brace in
    it is a plain brace."""
    translated = _strings.get(text, text)
    if not values:
        return translated
    return _fill(translated, text, values)


def ntr(one: str, other: str, n: int, /, **values) -> str:
    """The form of "`one` or `other`" that fits the count `n`, which fills
    `{n}`. `other` is the key in a language file."""
    source = one if n == 1 else other
    translated = source
    forms = _plurals.get(other)
    if forms:
        translated = forms.get(plural_form(_language, n)) or forms.get("other") or source
    return _fill(translated, source, {**values, "n": n})


def _fill(translated: str, source: str, values: dict) -> str:
    try:
        return translated.format(**values)
    except (KeyError, IndexError, ValueError, AttributeError, TypeError):
        if translated == source:
            raise
        return source.format(**values)  # a bad translation must not stop the window


def language() -> str:
    """The current language's code."""
    return _language


# -- plural rules -------------------------------------------------------------
# The Unicode CLDR categories for whole numbers. A language missing here uses
# "one" for 1 and "other" for the rest.

def _none(n):
    return "other"


def _zero_one(n):
    return "one" if n in (0, 1) else "other"


def _east_slavic(n):
    if n % 10 == 1 and n % 100 != 11:
        return "one"
    return "few" if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14) else "many"


def _polish(n):
    if n == 1:
        return "one"
    return "few" if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14) else "many"


def _czech(n):
    return "one" if n == 1 else "few" if n in (2, 3, 4) else "other"


def _south_slavic(n):
    if n % 10 == 1 and n % 100 != 11:
        return "one"
    return "few" if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14) else "other"


def _slovenian(n):
    return {1: "one", 2: "two", 3: "few", 4: "few"}.get(n % 100, "other")


def _romanian(n):
    if n == 1:
        return "one"
    return "few" if n == 0 or 1 <= n % 100 <= 19 else "other"


def _lithuanian(n):
    if n % 10 == 1 and not 11 <= n % 100 <= 19:
        return "one"
    return "few" if n % 10 and not 11 <= n % 100 <= 19 else "other"


def _hebrew(n):
    return {1: "one", 2: "two"}.get(n, "other")


def _arabic(n):
    if n in (0, 1, 2):
        return ("zero", "one", "two")[n]
    if 3 <= n % 100 <= 10:
        return "few"
    return "many" if n % 100 >= 11 else "other"


def _one_other(n):
    return "one" if n == 1 else "other"


_RULES = (
    (("ja", "zh", "ko", "vi", "th", "id", "ms", "lo", "my"), _none, ("other",)),
    (("fr", "pt", "hi", "fa", "bn", "hy"), _zero_one, ("one", "other")),
    (("ru", "uk", "be"), _east_slavic, ("one", "few", "many")),
    (("pl",), _polish, ("one", "few", "many")),
    (("cs", "sk"), _czech, ("one", "few", "other")),
    (("hr", "sr", "bs"), _south_slavic, ("one", "few", "other")),
    (("sl",), _slovenian, ("one", "two", "few", "other")),
    (("ro",), _romanian, ("one", "few", "other")),
    (("lt",), _lithuanian, ("one", "few", "other")),
    (("he",), _hebrew, ("one", "two", "other")),
    (("ar",), _arabic, ("zero", "one", "two", "few", "many", "other")),
)


def _rule(code: str):
    base = code.split("_")[0]
    for codes, rule, forms in _RULES:
        if base in codes:
            return rule, forms
    return _one_other, ("one", "other")


def plural_form(code: str, n: int) -> str:
    return _rule(code)[0](abs(int(n)))


def plural_forms(code: str) -> tuple:
    """The categories a language file's plural entries carry."""
    return _rule(code)[1]


# -- language files -----------------------------------------------------------

def placeholders(text: str) -> Optional[set]:
    """The names in `text`'s `{placeholders}`, or None when it isn't a valid
    format string. A name that isn't a plain identifier counts as invalid,
    so a file can't reach into an argument's attributes."""
    names = set()
    try:
        for _literal, name, spec, _conversion in string.Formatter().parse(text):
            if name is None:
                continue
            if not name.isidentifier() or (spec and "{" in spec):
                return None
            names.add(name)
    except ValueError:
        return None
    return names


def entry_problem(source: str, translated, *, plural: bool = False) -> Optional[str]:
    """Why `translated` can't stand in for `source`, or None. A source that
    isn't a format string (a lone brace) has nothing to compare."""
    if not isinstance(translated, str):
        return "not text"
    want = placeholders(source)
    if not want:
        return None
    got = placeholders(translated)
    if got is None:
        return "its placeholders can't be read"
    missing = want - got - ({"n"} if plural else set())
    if missing:
        return "missing " + ", ".join("{%s}" % name for name in sorted(missing))
    extra = got - want
    if extra:
        return "unknown " + ", ".join("{%s}" % name for name in sorted(extra))
    return None


def catalog_path(code: str, directory: Optional[str] = None) -> str:
    if not _CODE_RE.fullmatch(code):
        raise ValueError(f"not a language code: {code!r}")
    return os.path.join(directory or LOCALES_DIR, f"{code}.json")


def read_catalog(path: str) -> dict:
    """The parsed file with its parts checked for type, or `ValueError`."""
    if os.path.getsize(path) > MAX_CATALOG_BYTES:
        raise ValueError("file too large")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    for key in ("strings", "plurals"):
        if not isinstance(data.setdefault(key, {}), dict):
            raise ValueError(f"{key!r} is not an object")
    return data


def available_languages(directory: Optional[str] = None) -> list:
    """`(code, name)` of every language file, English first. The name is the
    language's own ("Deutsch"), or its code when the file has none."""
    directory = directory or LOCALES_DIR
    found = []
    try:
        entries = sorted(os.listdir(directory))
    except OSError:
        entries = []
    for entry in entries:
        code, ext = os.path.splitext(entry)
        if ext != ".json" or code == SOURCE_LANGUAGE or not _CODE_RE.fullmatch(code):
            continue
        try:
            name = read_catalog(os.path.join(directory, entry)).get("name")
        except (OSError, ValueError, RecursionError) as e:
            print(f"Skipping unreadable language file {entry}: {e}")
            continue
        found.append((code, name if isinstance(name, str) and name.strip() else code))
    return [(SOURCE_LANGUAGE, "English")] + found


def match_language(wanted, directory: Optional[str] = None) -> str:
    """The first of `wanted` (locale names such as "pt-BR", best first) that
    has a file, its base language counting ("pt"), else English."""
    codes = {code.lower(): code for code, _name in available_languages(directory)}
    for name in wanted:
        parts = str(name).replace("-", "_").lower().split("_")
        while parts:
            code = codes.get("_".join(parts))
            if code is not None:
                return code
            parts.pop()
    return SOURCE_LANGUAGE


def set_language(code: str, directory: Optional[str] = None) -> str:
    """Switches to `code` and returns the language now in use, which is
    English when the file is missing or can't be read. An empty entry, or
    one whose placeholders don't match the English, is left out, so that
    string shows in English."""
    global _language, _strings, _plurals
    _language, _strings, _plurals = SOURCE_LANGUAGE, {}, {}
    if code == SOURCE_LANGUAGE:
        return _language
    try:
        data = read_catalog(catalog_path(code, directory))
    except (OSError, ValueError, RecursionError) as e:
        print(f"Couldn't load the language {code!r}, using English: {e}")
        return _language
    strings, plurals, skipped = {}, {}, 0
    for source, translated in data["strings"].items():
        if translated == "":
            continue
        if entry_problem(source, translated) is None:
            strings[source] = translated
        else:
            skipped += 1
    for source, forms in data["plurals"].items():
        if not isinstance(forms, dict):
            skipped += 1
            continue
        filled = {form: text for form, text in forms.items() if text != ""}
        good = {form: text for form, text in filled.items() if entry_problem(source, text, plural=True) is None}
        skipped += len(filled) - len(good)
        if good:
            plurals[source] = good
    if skipped:
        print(f"{code}.json: {skipped} bad entries show in English; run scripts/i18n.py check")
    _language, _strings, _plurals = code, strings, plurals
    return _language
