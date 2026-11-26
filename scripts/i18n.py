"""Keeps the files in kokoro_gui/locales in step with the source.

    python scripts/i18n.py extract            rewrite en.json, and add new keys to the other languages
    python scripts/i18n.py new fr "Français"  start a language file with every string empty
    python scripts/i18n.py check [fr ...]     report what is missing or wrong, exit 1 on a problem

The strings come from the first argument of `tr`, `ntr` and `N_`, in the order
the files show them, so strings from one dialog sit together. An empty string
in a language file shows in English. `extract` drops a key the source no longer
has and keeps every translation whose English text is unchanged.

docs: CONTRIBUTING.md, "Translations".
"""
from __future__ import annotations

import ast
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kokoro_gui import i18n  # noqa: E402

SOURCES = ("kokoro_gui/qt", "kokoro_gui/daw", "kokoro_gui/audio", "kokoro_gui/core", "kokoro_gui/engine",
           "kokoro_gui/engines", "kokoro_engine.py")


class ExtractError(Exception):
    pass


def source_files() -> list:
    files = []
    for entry in SOURCES:
        path = os.path.join(ROOT, entry)
        if os.path.isfile(path):
            files.append(path)
            continue
        for folder, dirs, names in os.walk(path):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            files.extend(os.path.join(folder, name) for name in sorted(names) if name.endswith(".py"))
    return files


def _literal(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def scan(path: str, strings: dict, plurals: dict) -> None:
    """Adds the strings of one file. `tr(NAME)` and `tr(row[0])` pass a string
    some `N_` marked, so only a literal is taken. An f-string is refused: its
    text changes at run time, so no file could hold it."""
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
    where = os.path.relpath(path, ROOT).replace(os.sep, "/")
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id in ("tr", "ntr", "N_") and n.args]
    for call in sorted(calls, key=lambda n: (n.lineno, n.col_offset)):
        kind = call.func.id
        if any(isinstance(a, (ast.JoinedStr, ast.BinOp)) for a in call.args[:2 if kind == "ntr" else 1]):
            raise ExtractError(f"{where}:{call.lineno}: {kind}() needs a plain string literal")
        if kind == "ntr":
            one, other = _literal(call.args[0]), _literal(call.args[1])
            if one is None or other is None:
                raise ExtractError(f"{where}:{call.lineno}: ntr() needs two string literals")
            if plurals.setdefault(other, {"one": one, "other": other})["one"] != one:
                raise ExtractError(f"{where}:{call.lineno}: {other!r} has two singular forms")
            continue
        text = _literal(call.args[0])
        if text:
            strings.setdefault(text, text)
        elif kind == "N_":
            raise ExtractError(f"{where}:{call.lineno}: N_() needs a string literal")


def template() -> dict:
    """What `en.json` holds."""
    strings, plurals = {}, {}
    for path in source_files():
        scan(path, strings, plurals)
    return {"language": i18n.SOURCE_LANGUAGE, "name": "English", "strings": strings, "plurals": plurals}


def write(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.write("\n")


def merged(source: dict, code: str, name: str, old: dict) -> dict:
    """`code`'s file for `source`: every key, with the translation `old` has."""
    strings = {key: old.get("strings", {}).get(key, "") for key in source["strings"]}
    plurals = {}
    for key in source["plurals"]:
        before = old.get("plurals", {}).get(key)
        before = before if isinstance(before, dict) else {}
        plurals[key] = {form: before.get(form, "") for form in i18n.plural_forms(code)}
    return {"language": code, "name": name, "strings": strings, "plurals": plurals}


def language_codes() -> list:
    return [code for code, _name in i18n.available_languages() if code != i18n.SOURCE_LANGUAGE]


def extract() -> int:
    source = template()
    write(i18n.catalog_path(i18n.SOURCE_LANGUAGE), source)
    print(f"en: {len(source['strings'])} strings, {len(source['plurals'])} plurals")
    for code in language_codes():
        path = i18n.catalog_path(code)
        old = i18n.read_catalog(path)
        write(path, merged(source, code, old.get("name") or code, old))
        print(f"{code}: {missing_count(i18n.read_catalog(path))} to translate")
    return 0


def new(code: str, name: str) -> int:
    path = i18n.catalog_path(code)
    if os.path.exists(path):
        print(f"{path} already exists")
        return 1
    write(path, merged(template(), code, name, {}))
    print(f"Wrote {path}")
    return 0


def missing_count(data: dict) -> int:
    empty = sum(1 for text in data["strings"].values() if text == "")
    return empty + sum(1 for forms in data["plurals"].values()
                       if isinstance(forms, dict) and any(text == "" for text in forms.values()))


def _edges(text: str) -> tuple:
    return text[:len(text) - len(text.lstrip())], text[len(text.rstrip()):]


def problems(code: str, data: dict, source: dict) -> list:
    """What is wrong in a language file, one line each. An empty string is
    not a problem, it is work left."""
    found = []

    def look(key, text, plural=False):
        if text == "":
            return
        issue = i18n.entry_problem(key, text, plural=plural)
        if issue is None and _edges(key) != _edges(text):
            issue = "the spaces or line breaks at its ends differ"  # it is joined to other text
        if issue is not None:
            found.append(f"{code}: {key!r}: {issue}")

    for key in source["strings"]:
        if key not in data["strings"]:
            found.append(f"{code}: {key!r}: not in the file, run extract")
        else:
            look(key, data["strings"][key])
    for key in source["plurals"]:
        forms = data["plurals"].get(key)
        if not isinstance(forms, dict) or set(forms) != set(i18n.plural_forms(code)):
            found.append(f"{code}: {key!r}: needs the forms {', '.join(i18n.plural_forms(code))}")
            continue
        for text in forms.values():
            look(key, text, plural=True)
    for part in ("strings", "plurals"):
        found.extend(f"{code}: {key!r}: no longer in the source, run extract"
                     for key in data[part] if key not in source[part])
    return found


def check(codes: list) -> int:
    source = template()
    found = []
    if i18n.read_catalog(i18n.catalog_path(i18n.SOURCE_LANGUAGE)) != source:
        found.append("en: en.json is out of date, run extract")
    for code in codes or language_codes():
        data = i18n.read_catalog(i18n.catalog_path(code))
        found.extend(problems(code, data, source))
        print(f"{code}: {missing_count(data)} of {len(source['strings']) + len(source['plurals'])} to translate")
    for line in found:
        print(line)
    return 1 if found else 0


def main(argv: list) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        if argv[:1] == ["extract"] and len(argv) == 1:
            return extract()
        if argv[:1] == ["new"] and len(argv) == 3:
            return new(argv[1], argv[2])
        if argv[:1] == ["check"]:
            return check(argv[1:])
    except (ExtractError, OSError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
