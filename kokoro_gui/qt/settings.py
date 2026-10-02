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
import math
import os

from kokoro_gui.qt import spec


def _has_default_type(default, value) -> bool:
    """True when `value` is the kind of thing `default` is. A bool is not an
    int here, an int is fine for a float default, a float is not for an int
    default (a spin box raises on it), and a float must be finite. A `None`
    default (a path that may be unset) takes `None` or a str."""
    if default is None:
        return value is None or isinstance(value, str)
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, int):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            return math.isfinite(value)
        except OverflowError:
            return False
    return isinstance(value, type(default))


def clean_loaded_settings(loaded, defaults: dict) -> dict:
    """`defaults` overlaid with `loaded`, a parsed `config_qt.json`. The file
    is untrusted: one that isn't a JSON object gives the defaults, and a key
    whose value has another type than its default's goes back to the
    default, so a hand-edited or damaged value ("speed": "fast") can't reach
    a Qt setter and stop the app from starting. Keys the defaults don't
    list pass through. The reset keys are printed (plan 07 logs them)."""
    if not isinstance(loaded, dict):
        print(f"config_qt.json holds {type(loaded).__name__}, not an object; using the defaults.")
        return defaults
    merged = {**defaults, **loaded}
    reset = [key for key, default in defaults.items()
             if key in loaded and not _has_default_type(default, loaded[key])]
    for key in reset:
        merged[key] = defaults[key]
    if reset:
        print(f"config_qt.json: reset to the default (wrong type): {', '.join(sorted(reset))}")
    return merged


def load_settings(config_file: str) -> dict:
    # deepcopy, not dict(...): SETTINGS_DEFAULTS["lexicon"] is a mutable {}
    # shared across every call - a shallow copy would let one instance's
    # in-place `settings["lexicon"][k] = v` (lexicon_dock.py's add_rule)
    # leak into every other instance/test that reads the same defaults.
    defaults = copy.deepcopy(spec.SETTINGS_DEFAULTS)
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                return clean_loaded_settings(json.load(f), defaults)
        except Exception as e:
            print(f"Couldn't read {config_file}, using the defaults: {e}")
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
    """Writes `config_file` through `config_file + ".tmp"` and `os.replace`,
    so a crash or a full disk mid-write leaves the old file whole instead of
    a truncated one that the next launch would read as "no settings"."""
    tmp = config_file + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
        os.replace(tmp, config_file)
    except Exception as e:
        print(f"Failed to save Qt settings: {e}")
        try:
            os.remove(tmp)
        except OSError:
            pass


def encode_bytes(qbytearray) -> str:
    return base64.b64encode(bytes(qbytearray)).decode("ascii")


def decode_bytes(b64_str: str):
    from PySide6.QtCore import QByteArray
    return QByteArray(base64.b64decode(b64_str.encode("ascii")))
