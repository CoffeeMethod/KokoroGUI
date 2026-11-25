"""What there is to choose from: engines, voices, devices. Loads no model."""
from __future__ import annotations

from typing import Optional

from kokoro_gui.core import engines as core_engines
from kokoro_gui.engines import registry

from .player import playback_reason


def list_engines(default_engine_id: str = registry.DEFAULT_ENGINE_ID) -> list:
    """`{id, name, installed, reason, default}` per engine. `reason` is why
    one that isn't installed isn't."""
    return [{
        "id": engine_id,
        "name": registry.get_display_name(engine_id),
        "installed": registry.is_registered(engine_id),
        "reason": registry.unavailable_reason(engine_id),
        "default": engine_id == default_engine_id,
    } for engine_id in registry.list_all_engines()]


def list_voices(host, engine: Optional[str] = None, lang_code: Optional[str] = None) -> list:
    """`{id, name, lang_code, custom, default}` per voice of an engine, of one
    language when `lang_code` is given."""
    backend = host.backend(engine)
    settings = core_engines.engine_settings(host.settings, backend.id)
    default = core_engines.default_voice(backend, settings)
    seen: dict = {}
    for voice in backend.get_voices(lang_code):
        seen.setdefault(voice.id, {"id": voice.id, "name": voice.display_name, "lang_code": voice.lang_code,
                                   "custom": bool(voice.is_custom), "default": voice.id == default})
    return list(seen.values())


def list_devices() -> dict:
    """The values `--device` takes here, and why audio can't play (None when
    it can)."""
    devices = ["auto", "cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            devices.append("cuda")
    except Exception:  # noqa: BLE001 - no torch, or a broken driver
        pass
    return {"devices": devices, "playback": playback_reason()}
