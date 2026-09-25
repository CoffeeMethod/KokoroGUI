"""Read-time post-processing for clip segments.

Generation writes a clip's segments as raw model output (`Segment.raw`).
Everything the Audio FX tab and the Settings tab's volume / pitch / normalize
/ trim controls describe is applied here, when the transport, the exporter
or the timeline waveform reads the file, and never written back. Changing an
FX setting therefore never dirties a clip: `daw/dirty.py` only looks at the
generation keys, and this module only looks at `POST_KEYS`.

`render()` memoizes by `(path, mtime, post_key, target_rate, range_s)`, so
a slider move re-renders only the clips whose resolved post config changed,
and a transport rebuild with nothing changed is a dict lookup per segment.
A config naming a convolution impulse response also carries `project_dir`
(where the IR resolves first), and `post_key` folds in the resolved IR's
path and mtime, so replacing the IR file or adding a project-local copy
re-renders.

`range_s` (or `render_slice`) plays a time range of a file instead of the
whole file: a `Segment.range` (an imported recording's words), a source
track sliced per clip, a trimmed music bed. Only those frames are read.

Qt-free. `process_audio` is the same function `process_chunk_task` uses on
the whole-document path, called later.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import numpy as np

from kokoro_gui.engine.presets import ALLOWED_FX_PRESET_KEYS, resolve_ir

# Every config key the post stage reads. `pitch` is here (the resample) and
# also in the generation cache key (the speed compensation), which is why a
# pitch change both dirties the clip and re-renders it.
POST_KEYS = frozenset(ALLOWED_FX_PRESET_KEYS | {"apply_fx", "volume", "pitch", "normalize", "trim_silence"})

_RENDER_CACHE: dict = {}


def extract_post_config(config: dict) -> dict:
    """The `POST_KEYS` subset of a full clip config, plus `project_dir` when
    it names a convolution impulse response (the IR resolves there first)."""
    subset = {k: config[k] for k in POST_KEYS if k in config}
    if subset.get("convolution_ir") and config.get("project_dir"):
        subset["project_dir"] = config["project_dir"]
    return subset


def _ir_stamp(config: dict):
    """`[path, mtime]` of the impulse response `config` names, `None` when it
    names none or it resolves nowhere."""
    name = config.get("convolution_ir")
    if not name:
        return None
    path = resolve_ir(name, config.get("project_dir"))
    if path is None:
        return None
    try:
        return [path, os.path.getmtime(path)]
    except OSError:
        return None


def post_key(config: dict) -> str:
    """A stable fingerprint of `config`'s post-processing keys and, for a
    convolution reverb, the file its IR name resolves to. Key order and
    non-post keys don't affect it."""
    subset = extract_post_config(config)
    if subset.get("convolution_ir"):
        subset["convolution_ir_file"] = _ir_stamp(subset)
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
    if config.get("convolution_ir"):
        return False
    return not (config.get("eq_bass", 0.0) or config.get("eq_treble", 0.0))


def segment_range(segment) -> Optional[tuple]:
    """`segment.range` as a `(start_s, end_s)` float pair, or None when the
    segment plays its whole file (no range, or one that isn't two numbers,
    as a hand-edited `document.json` might hold)."""
    return _clean_range(getattr(segment, "range", None))


def _clean_range(range_s) -> Optional[tuple]:
    if not isinstance(range_s, (list, tuple)):
        return None
    try:
        start, end = float(range_s[0]), float(range_s[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (np.isfinite(start) and np.isfinite(end)):
        return None
    return start, end


def _read_mono(path: str, range_s: Optional[tuple] = None):
    """`(mono, rate)` for the whole file, or with `range_s` only the frames
    from `start_s` to `end_s`: a seek and a read, so a slice of a long
    recording never decodes the rest. The range is clamped to the file; an
    empty or inverted one gives an empty array."""
    import soundfile as sf

    if range_s is None:
        data, rate = sf.read(path, dtype="float32", always_2d=True)
    else:
        with sf.SoundFile(path) as f:
            rate = f.samplerate
            first = min(max(0, int(round(range_s[0] * rate))), f.frames)
            last = min(max(0, int(round(range_s[1] * rate))), f.frames)
            if last <= first:
                return np.zeros(0, dtype=np.float32), int(rate)
            f.seek(first)
            data = f.read(last - first, dtype="float32", always_2d=True)
    mono = data.mean(axis=1).astype(np.float32) if data.shape[1] > 1 else data[:, 0]
    return mono, int(rate)


def _slice_config(post_config: Optional[dict], range_s: Optional[tuple]) -> Optional[dict]:
    """The config a render applies. A slice ignores `trim_silence`: its
    range already says where the audio starts and ends (a word boundary, a
    cue, a trimmed bed), and trimming inside it would move that."""
    if range_s is None or not post_config or not post_config.get("trim_silence", False):
        return post_config
    return dict(post_config, trim_silence=False)


def render(path: str, post_config: Optional[dict], target_rate: int,
           range_s: Optional[tuple] = None) -> np.ndarray:
    """`path` read, post-processed at its native rate per `post_config`, then
    resampled to `target_rate`. Mono float32. Raises whatever soundfile
    raises for an unreadable file. `post_config=None` means no processing.
    `range_s` (`(start_s, end_s)` seconds into the file) reads and processes
    only that slice; None reads the whole file."""
    from kokoro_gui.audio.mixer import resample

    range_s = _clean_range(range_s)
    post_config = _slice_config(post_config, range_s)
    key = _cache_key(path, post_config, target_rate, range_s)
    cached = _RENDER_CACHE.get(key)
    if cached is not None:
        return cached

    mono, rate = _read_mono(path, range_s)
    if len(mono) and post_config and not is_identity(post_config):
        from kokoro_gui.engine.audio_fx import process_audio

        mono = np.asarray(process_audio(mono, rate, post_config), dtype=np.float32).reshape(-1)
    out = resample(mono, rate, int(target_rate))
    _RENDER_CACHE[key] = out
    return out


def render_slice(path: str, start_s: float, end_s: float, post_config: Optional[dict],
                 target_rate: int) -> np.ndarray:
    """`render` of the `[start_s, end_s)` seconds of `path`."""
    return render(path, post_config, target_rate, range_s=(start_s, end_s))


def _cache_key(path: str, post_config: Optional[dict], target_rate: int,
               range_s: Optional[tuple] = None) -> tuple:
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None
    return (os.path.abspath(path), mtime, post_key(post_config or {}), int(target_rate), range_s)


def duration_hint(segment, post_config: Optional[dict]) -> Optional[float]:
    """The segment's rendered length computed from what generation stored
    (`duration`, `onset_s`, `tail_s`) without reading audio, or None for a
    segment that predates those fields. Trim removes the onset and tail;
    pitch resamples by `2 ** (semitones / 12)`; nothing else in
    `process_audio` changes the length. A segment with a `range` is
    `end - start` long before pitch, and trim doesn't apply to it (see
    `_slice_config`)."""
    config = post_config or {}
    range_s = segment_range(segment)
    if range_s is not None:
        length = max(0.0, range_s[1] - range_s[0])
    else:
        duration = getattr(segment, "duration", None)
        onset, tail = getattr(segment, "onset_s", None), getattr(segment, "tail_s", None)
        if duration is None or onset is None or tail is None:
            return None
        length = float(duration)
        if config.get("trim_silence", False):
            length = max(0.0, length - float(onset) - float(tail))
    from kokoro_gui.engine.audio_fx import clamp_pitch_semitones

    semitones = clamp_pitch_semitones(config.get("pitch", 0.0))
    if semitones:
        length /= 2 ** (semitones / 12.0)
    return length


def rendered_duration_s(path: str, post_config: Optional[dict], target_rate: int,
                        hint: Optional[float] = None, range_s: Optional[tuple] = None) -> float:
    """The rendered length in seconds. A render already in the memo answers
    exactly; otherwise `hint` (from `duration_hint`) answers without reading
    the file, which is what lets a long project place every clip at open.
    `range_s` measures that slice of the file, as `render` does."""
    range_s = _clean_range(range_s)
    cached = _RENDER_CACHE.get(_cache_key(path, _slice_config(post_config, range_s), target_rate, range_s))
    if cached is not None:
        return len(cached) / float(target_rate)
    if hint is not None:
        return float(hint)
    return len(render(path, post_config, target_rate, range_s)) / float(target_rate)


def clear_render_cache() -> None:
    _RENDER_CACHE.clear()
