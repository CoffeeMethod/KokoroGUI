"""Project templates and audiobook credits (plan 25).

A template is the shape of a project without its content: the project
settings (gaps, ducking, track layout), the library ids of its linked
characters, and one section per subproject, each with a title and, if the
author kept it, a short text such as an intro or an ad read. A podcast saves
episode 13 as a template and starts episode 14 from it.

Stored as `<TEMPLATES_DIR>/<name>.json`, one file per template, beside
`characters/`. Nothing here touches a `.tbaw`: a template is a plain JSON
file the app reads and writes on its own, and "New from Template" builds an
ordinary new project from it (`SubprojectsMixin.new_from_template`).

The files are user data and are read as untrusted input, like
`config_qt.json`: a size cap on the file, a type check on every field,
unknown keys ignored, the name taken through a basename-and-unsafe-character
rule, and the settings stripped of anything path-like.

`credit_texts` writes the opening and closing credit lines an audiobook
needs. Qt-free like the rest of `kokoro_gui/daw/`.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
from dataclasses import dataclass, field

from kokoro_gui.daw import markers as marker_ops
from kokoro_gui.daw.models import SOURCES_KEY
from kokoro_gui.daw.reference import SOURCE_TRACK_KEY

# `templates/` beside `characters/`, relative to the working directory like
# the other stores. Read at call time, so tests monkeypatch it.
TEMPLATES_DIR = "templates"

FORMAT = "kokorogui-template"
VERSION = 1

MAX_SECTIONS = 500
MAX_TEXT_CHARS = 1_000_000  # all section texts together
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_NAME_CHARS = 80
MAX_TITLE_CHARS = 200
MAX_CHARACTERS = 500
MAX_SETTINGS_DEPTH = 6
KEEP_TEXT_LIMIT = 2000  # Save as Template ticks "keep its text" under this many characters

# Document.settings keys that never go into a template: the markers belong to
# one timeline, `sources` and the source track name files on this machine.
EXCLUDED_SETTINGS = frozenset({marker_ops.MARKERS_KEY, SOURCES_KEY, SOURCE_TRACK_KEY})

_log = logging.getLogger(__name__)

_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = frozenset({"con", "prn", "aux", "nul", *(f"com{n}" for n in range(1, 10)),
                       *(f"lpt{n}" for n in range(1, 10))})
_PATH_KEY = re.compile(r"(^|_)(path|file|dir|folder)$", re.IGNORECASE)
_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass
class Section:
    title: str
    text: str = ""


@dataclass
class Template:
    name: str
    settings: dict = field(default_factory=dict)
    characters: list = field(default_factory=list)
    sections: list = field(default_factory=list)
    # The file stem this template was read from or written to; "" before it is saved.
    stem: str = ""


def safe_name(name) -> str | None:
    """`name` as a bare file stem, or None when nothing usable is left. The
    last path component only (`../x` is `x`), unsafe characters replaced by
    `_`, no leading dot, no Windows device name, at most `MAX_NAME_CHARS`."""
    if not isinstance(name, str):
        return None
    stem = re.split(r"[\\/]", name.strip())[-1]
    stem = _UNSAFE.sub("_", stem).strip().rstrip(". ")
    stem = stem[:MAX_NAME_CHARS].rstrip(". ")
    if not stem or stem.startswith("."):
        return None
    if stem.split(".")[0].casefold() in _RESERVED:
        return None
    return stem


def _is_path_like(value) -> bool:
    return isinstance(value, str) and (os.path.isabs(value) or value.startswith(("\\\\", "//"))
                                       or bool(_DRIVE.match(value)))


def clean_settings(settings, _depth: int = 0):
    """A copy of `Document.settings` fit for a template: JSON values only,
    no excluded key, no key that names a file, no value that is an absolute
    path. Anything else is dropped. Used on the way in and on the way out."""
    if not isinstance(settings, dict) or _depth > MAX_SETTINGS_DEPTH:
        return {}
    out = {}
    for key, value in settings.items():
        if not isinstance(key, str) or (_depth == 0 and key in EXCLUDED_SETTINGS) or _PATH_KEY.search(key):
            continue
        if isinstance(value, dict):
            value = clean_settings(value, _depth + 1)
        elif isinstance(value, (list, tuple)):
            value = _clean_list(value, _depth + 1)
            if value is None:
                continue
        elif isinstance(value, bool) or value is None or isinstance(value, (int, float, str)):
            if _is_path_like(value):
                continue
        else:
            continue
        out[key] = copy.deepcopy(value)
    return out


def _clean_list(values, depth: int):
    if depth > MAX_SETTINGS_DEPTH:
        return None
    out = []
    for value in values:
        if isinstance(value, dict):
            out.append(clean_settings(value, depth + 1))
        elif isinstance(value, (list, tuple)):
            inner = _clean_list(value, depth + 1)
            if inner is not None:
                out.append(inner)
        elif isinstance(value, bool) or value is None or isinstance(value, (int, float, str)):
            if not _is_path_like(value):
                out.append(value)
    return out


def _store_root() -> str:
    return TEMPLATES_DIR


def _path_for(stem: str) -> str:
    return os.path.join(_store_root(), f"{stem}.json")


def _text(value, limit: int) -> str:
    return value[:limit] if isinstance(value, str) else ""


def template_from_dict(data, stem: str = "") -> Template | None:
    """A validated `Template` from parsed JSON, or None when it isn't one.
    Wrong-typed fields fall back to empty, unknown keys are ignored, sections
    stop at `MAX_SECTIONS` and the texts at `MAX_TEXT_CHARS` together."""
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        return None
    version = data.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version > VERSION:
        return None
    name = _text(data.get("name"), MAX_NAME_CHARS).strip() or stem
    if not name:
        return None
    characters = []
    raw_characters = data.get("characters")
    for library_id in raw_characters if isinstance(raw_characters, list) else []:
        cleaned = os.path.basename(library_id.strip()) if isinstance(library_id, str) else ""
        if cleaned and not cleaned.startswith(".") and cleaned not in characters:
            characters.append(cleaned)
        if len(characters) >= MAX_CHARACTERS:
            break
    sections = []
    budget = MAX_TEXT_CHARS
    raw_sections = data.get("sections")
    for raw in raw_sections if isinstance(raw_sections, list) else []:
        if len(sections) >= MAX_SECTIONS:
            break
        if not isinstance(raw, dict):
            continue
        title = " ".join(_text(raw.get("title"), MAX_TITLE_CHARS).split())
        text = _text(raw.get("text"), budget)
        budget -= len(text)
        sections.append(Section(title=title, text=text))
    return Template(name=name, settings=clean_settings(data.get("settings")), characters=characters,
                    sections=sections, stem=stem)


def template_to_dict(template: Template) -> dict:
    return {
        "format": FORMAT,
        "version": VERSION,
        "name": template.name,
        "settings": clean_settings(template.settings),
        "characters": list(template.characters),
        "sections": [{"title": s.title, "text": s.text} for s in template.sections],
    }


def build_template(name: str, settings: dict, characters, sections) -> Template:
    """A `Template` from an open project's parts: its `Document.settings`,
    the library ids of its linked characters, and `(title, text)` sections.
    Applies the same limits a load does, so what is saved reads back whole."""
    data = {"format": FORMAT, "version": VERSION, "name": name, "settings": settings,
            "characters": list(characters),
            "sections": [{"title": t, "text": x} for t, x in sections]}
    return template_from_dict(data, safe_name(name) or "") or Template(name=name)


def save_template(name: str, settings: dict, characters, sections) -> str | None:
    """Writes the template as `<TEMPLATES_DIR>/<safe name>.json` (a `.tmp`
    and `os.replace`, so a reader never sees half a file) and returns the file
    stem, or None when `name` leaves nothing usable. An existing template of
    that name is replaced."""
    stem = safe_name(name)
    if stem is None:
        return None
    template = build_template(name.strip(), settings, characters, sections)
    template.stem = stem
    root = _store_root()
    os.makedirs(root, exist_ok=True)
    path = _path_for(stem)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(template_to_dict(template), f, indent=2)
    os.replace(tmp, path)
    return stem


def _read(path: str, stem: str) -> Template | None:
    try:
        if os.path.getsize(path) > MAX_FILE_BYTES:
            raise ValueError("file too large")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        template = template_from_dict(data, stem)
        if template is None:
            raise ValueError("not a template")
    except (OSError, ValueError, TypeError, RecursionError) as e:
        _log.warning("Skipping unreadable template %s: %s", path, e)
        return None
    return template


def load_template(name) -> Template | None:
    """The template saved under `name` (its file stem), or None."""
    stem = safe_name(name)
    if stem is None:
        return None
    path = _path_for(stem)
    if not os.path.isfile(path):
        return None
    root = os.path.realpath(_store_root())
    if not os.path.realpath(path).startswith(root + os.sep):
        return None  # a link that leaves the templates folder
    return _read(path, stem)


def list_templates() -> list:
    """Every readable template, sorted by name. A file that isn't a template
    is skipped and logged."""
    root = _store_root()
    if not os.path.isdir(root):
        return []
    real_root = os.path.realpath(root)
    found = []
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if not entry.endswith(".json") or not os.path.isfile(path):
            continue
        stem = safe_name(entry[: -len(".json")])
        if stem is None or stem != entry[: -len(".json")]:
            continue
        if not os.path.realpath(path).startswith(real_root + os.sep):
            continue
        template = _read(path, stem)
        if template is not None:
            found.append(template)
    found.sort(key=lambda t: (t.name.casefold(), t.stem))
    return found


def delete_template(name) -> bool:
    stem = safe_name(name)
    if stem is None:
        return False
    try:
        os.remove(_path_for(stem))
    except FileNotFoundError:
        return False
    return True


# -- credits ---------------------------------------------------------------------

CREDIT_FIELDS = ("title", "subtitle", "author", "narrator", "publisher", "year")
OPENING_TITLE = "Opening Credits"
CLOSING_TITLE = "Closing Credits"
_ENDINGS = (".", "!", "?")


def clean_credit_fields(fields) -> dict:
    """`fields` as `{field: str}` for every name in `CREDIT_FIELDS`: stripped,
    one line, wrong types read as empty."""
    out = {}
    source = fields if isinstance(fields, dict) else {}
    for name in CREDIT_FIELDS:
        value = source.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = str(int(value))
        out[name] = " ".join(value.split())[:MAX_TITLE_CHARS] if isinstance(value, str) else ""
    return out


def _sentence(text: str) -> str:
    return text if text.endswith(_ENDINGS) else text + "."


def credit_texts(fields) -> tuple:
    """`(opening, closing)` credit lines for an audiobook. A field left empty
    drops its sentence; each sentence ends in a full stop, so the first line
    never reads as a chapter heading (`arrangement.is_heading_text`).

    Opening: "Title, Subtitle. Written by Author. Narrated by Narrator."
    Closing: "You have been listening to Title. Written by Author. Narrated
    by Narrator. Copyright Year by Publisher. The end."

    The wording follows ACX's audio submission requirements (read 2026-10-09,
    help.acx.com): opening credits state the title, author and narrator;
    closing credits indicate finality, suggested as "You have been listening
    to ...", the same three facts and "The End." ACX also asks for the opening
    and closing credits as separate files, which is why Add Credits makes two
    subprojects.
    """
    f = clean_credit_fields(fields)
    title, subtitle = f["title"], f["subtitle"]
    name = title
    if title and subtitle:
        name = f"{title} {subtitle}" if title.endswith(_ENDINGS) else f"{title}, {subtitle}"
    elif subtitle:
        name = subtitle
    by = [_sentence(f"Written by {f['author']}")] if f["author"] else []
    reader = [_sentence(f"Narrated by {f['narrator']}")] if f["narrator"] else []
    opening = [_sentence(name)] if name else []
    closing = [_sentence(f"You have been listening to {title}")] if title else []
    if f["year"] and f["publisher"]:
        rights = [_sentence(f"Copyright {f['year']} by {f['publisher']}")]
    elif f["year"]:
        rights = [_sentence(f"Copyright {f['year']}")]
    elif f["publisher"]:
        rights = [_sentence(f"Copyright by {f['publisher']}")]
    else:
        rights = []
    opening_text = " ".join([*opening, *by, *reader]) if (opening or by or reader) else ""
    closing_text = " ".join([*closing, *by, *reader, *rights, "The end."])
    return opening_text, closing_text
