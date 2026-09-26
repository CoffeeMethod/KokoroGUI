"""Load/save `config_qt.json` (the Qt frontend's app-settings file), plus
the base64 helpers `kokoro_gui.qt.workspace` uses for `QMainWindow`
dock-layout bytes.

Pure functions (no `QMainWindow`/app-instance state held here) so they're
easy to unit test in isolation - `app.py` calls these and owns the debounce
timer (`QTimer.singleShot`).
"""
from __future__ import annotations

import base64
import copy
import json
import os

from kokoro_gui.qt import spec


def load_settings(config_file: str) -> dict:
    # deepcopy, not dict(...): SETTINGS_DEFAULTS["lexicon"] is a mutable {}
    # shared across every call - a shallow copy would let one instance's
    # in-place `settings["lexicon"][k] = v` (lexicon_dock.py's add_rule)
    # leak into every other instance/test that reads the same defaults.
    defaults = copy.deepcopy(spec.SETTINGS_DEFAULTS)
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                return {**defaults, **json.load(f)}
        except Exception:
            pass
    return defaults


# Settings that used to be one flat value for every engine and now live per
# engine in `settings["engines"][<id>]` (grill EN5).
FLAT_ENGINE_KEYS = ("lang_code", "num_threads", "voice")


def migrate_engine_settings(settings: dict, engine_ids=None) -> dict:
    """Moves the flat per-engine keys of an older `config_qt.json` into
    `settings["engines"]`, in place, and returns `settings`. A flat
    `lang_code` goes to every engine whose `get_languages()` lists it (so
    Kokoro's "a" lands on Kokoro and Dummy, never on Audio8); the flat
    `voice` to the default engine only (it was that engine's voice); any
    other flat key to every engine whose schema has the field. An engine
    that already has a value keeps it. The flat keys are dropped
    afterwards."""
    from kokoro_gui.engines import registry

    flat = {key: settings.pop(key) for key in FLAT_ENGINE_KEYS if key in settings}
    engines = settings.get("engines")
    if not isinstance(engines, dict):
        engines = settings["engines"] = {}
    if not flat:
        return settings
    voice_engine = settings.get("default_engine") or registry.DEFAULT_ENGINE_ID
    for engine_id in (engine_ids if engine_ids is not None else registry.list_engines()):
        fields = {f.key: f for f in registry.get_config_schema(engine_id)}
        bucket = engines.get(engine_id) if isinstance(engines.get(engine_id), dict) else {}
        for key, value in flat.items():
            field = fields.get(key)
            if field is None or key in bucket:
                continue
            if key == "voice" and engine_id != voice_engine:
                continue
            if key == "lang_code" and value not in {code for _label, code in (field.choices or [])}:
                continue
            bucket[key] = value
        if bucket:
            engines[engine_id] = bucket
    return settings


def save_settings(config_file: str, settings: dict) -> None:
    try:
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
    except Exception as e:
        print(f"Failed to save Qt settings: {e}")


def encode_bytes(qbytearray) -> str:
    return base64.b64encode(bytes(qbytearray)).decode("ascii")


def decode_bytes(b64_str: str):
    from PySide6.QtCore import QByteArray
    return QByteArray(base64.b64decode(b64_str.encode("ascii")))
