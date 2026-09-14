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
