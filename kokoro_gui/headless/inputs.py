"""Files a caller names: a script, and a `config_qt.json`. Both are untrusted."""
from __future__ import annotations

import os
from typing import Optional

from kokoro_gui.engine import text_extraction
from kokoro_gui.engines import registry
from kokoro_gui.engines.base import per_engine_fields
from kokoro_gui.qt import settings as qt_settings

from . import paths
from .clean import clean_markdown
from .errors import HeadlessError

SCRIPT_SUFFIXES = (".txt", ".md")


def read_script(path, *, guarded: bool = False) -> str:
    """The text of a `.txt` or `.md` script, under the text import's size and
    character limits. Markdown loses its markup and keeps its speaker tags.
    `guarded` (the MCP server, grill PG26) also wants the file inside the
    renders root or the current directory, links followed."""
    try:
        path = os.fspath(path)
    except TypeError as e:
        raise HeadlessError("bad_arguments", "the script path must be a path") from e
    if guarded:
        try:
            path = paths.check_out_path(path)
        except HeadlessError as e:
            raise HeadlessError(
                "bad_arguments", "a script file must be inside audio_output/headless or the current directory",
                valid=e.valid) from e
    suffix = os.path.splitext(path)[1].lower()
    if suffix not in SCRIPT_SUFFIXES:
        raise HeadlessError("bad_arguments", f"a script is a {' or '.join(SCRIPT_SUFFIXES)} file",
                            valid=SCRIPT_SUFFIXES)
    if not os.path.isfile(path):
        raise HeadlessError("bad_arguments", f"there is no script file {path}")
    try:
        text = text_extraction.extract_text_from_file(path)
    except UnicodeDecodeError as e:
        raise HeadlessError("bad_arguments", "the script file isn't UTF-8 text") from e
    except (OSError, ValueError) as e:  # ExtractionLimitError is a ValueError
        raise HeadlessError("bad_arguments", f"the script can't be read: {e}") from e
    return clean_markdown(text) if suffix == ".md" else text


def load_settings(path: str) -> dict:
    """A `config_qt.json` as the app loads it, with each engine's values
    checked against its schema's types."""
    if not os.path.isfile(path):
        raise HeadlessError("bad_arguments", f"there is no settings file {path}")
    settings = qt_settings.migrate_engine_settings(qt_settings.load_settings(path))
    engines = {}
    for engine_id in registry.list_engines():
        bucket = settings["engines"].get(engine_id)
        if not isinstance(bucket, dict):
            continue
        fields = {f.key: f.default for f in per_engine_fields(registry.get_config_schema(engine_id))}
        engines[engine_id] = {key: value for key, value in bucket.items()
                              if key in fields and qt_settings._has_default_type(fields[key], value)}
    settings["engines"] = engines
    return settings


def default_engine(settings: Optional[dict]) -> str:
    """The settings' default engine, when it's one this install knows."""
    chosen = (settings or {}).get("default_engine")
    return chosen if isinstance(chosen, str) and chosen in registry.list_all_engines() else registry.DEFAULT_ENGINE_ID
