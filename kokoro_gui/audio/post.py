"""Read-time post-processing for clip segments.

Generation writes a clip's segments as raw model output (`Segment.raw`).
Everything the Audio FX tab and the Settings tab's volume / pitch / normalize
/ trim controls describe is applied here, when the transport, the exporter
or the timeline waveform reads the file, and never written back. Changing an
FX setting therefore never dirties a clip: `daw/dirty.py` only looks at the
generation keys, and this module only looks at `POST_KEYS`.

`render()` memoizes by `(path, mtime, post_key, target_rate)`, so a slider
move re-renders only the clips whose resolved post config changed, and a
transport rebuild with nothing changed is a dict lookup per segment.

Qt-free. `process_audio` is the same function `process_chunk_task` uses on
the whole-document path, called later.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import numpy as np

from kokoro_gui.engine.presets import ALLOWED_FX_PRESET_KEYS

# Every config key the post stage reads. `pitch` is here (the resample) and
# also in the generation cache key (the speed compensation), which is why a
# pitch change both dirties the clip and re-renders it.
POST_KEYS = frozenset(ALLOWED_FX_PRESET_KEYS | {"apply_fx", "volume", "pitch", "normalize", "trim_silence"})

_RENDER_CACHE: dict = {}


def extract_post_config(config: dict) -> dict:
    """The `POST_KEYS` subset of a full clip config."""
    return {k: config[k] for k in POST_KEYS if k in config}


def post_key(config: dict) -> str:
    """A stable fingerprint of `config`'s post-processing keys. Key order and
    non-post keys don't affect it."""
    subset = extract_post_config(config)
    payload = json.dumps(subset, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_identity(config: dict) -> bool:
    """True when `process_audio` would return its input unchanged, so
    callers can skip the FX import entirely."""
    if config.get("trim_silence", False) or config.get("normalize", False):
        return False
    if config.get("volume", 1.0) != 1.0 or float(config.get("pitch", 0.0) or 0.0) != 0.0:
        return False
    if not config.get("apply_fx", True):
        return True
    for key in ALLOWED_FX_PRESET_KEYS:
        if key.endswith("_enabled") and config.get(key, False):
            return False
    return not (config.get("eq_bass", 0.0) or config.get("eq_treble", 0.0))


def _read_mono(path: str):
    import soundfile as sf

    data, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1).astype(np.float32) if data.shape[1] > 1 else data[:, 0]
    return mono, int(rate)


def render(path: str, post_config: Optional[dict], target_rate: int) -> np.ndarray:
    """`path` read, post-processed at its native rate per `post_config`, then
    resampled to `target_rate`. Mono float32. Raises whatever soundfile
    raises for an unreadable file. `post_config=None` means no processing."""
    from kokoro_gui.audio.mixer import resample

    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None
    key = (os.path.abspath(path), mtime, post_key(post_config or {}), int(target_rate))
    cached = _RENDER_CACHE.get(key)
    if cached is not None:
        return cached

    mono, rate = _read_mono(path)
    if post_config and not is_identity(post_config):
        from kokoro_gui.engine.audio_fx import process_audio

        mono = np.asarray(process_audio(mono, rate, post_config), dtype=np.float32).reshape(-1)
    out = resample(mono, rate, int(target_rate))
    _RENDER_CACHE[key] = out
    return out


def rendered_duration_s(path: str, post_config: Optional[dict], target_rate: int) -> float:
    return len(render(path, post_config, target_rate)) / float(target_rate)


def clear_render_cache() -> None:
    _RENDER_CACHE.clear()
