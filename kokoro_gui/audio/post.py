"""Read-time post-processing for clip segments.

Generation writes a clip's segments as raw model output (`Segment.raw`).
Everything the Audio FX tab and the Settings tab's volume / pitch / normalize
/ trim controls describe, and fit to slot's `time_stretch`, is applied here,
when the transport, the exporter or the timeline waveform reads the file,
and never written back. Changing an FX setting therefore never dirties a
clip: `daw/dirty.py` only looks at the generation keys, and this module
only looks at `POST_KEYS`.

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
import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from kokoro_gui.daw import revision
from kokoro_gui.engine.presets import ALLOWED_FX_PRESET_KEYS, resolve_ir

logger = logging.getLogger(__name__)

# Every config key the post stage reads. `pitch` is here (the resample) and
# also in the generation cache key (the speed compensation), which is why a
# pitch change both dirties the clip and re-renders it. `time_stretch` is a
# factor (1.0 = none, above 1 = shorter) that fit to slot sets from
# `Clip.overrides` on a clip whose engine has no speed control.
POST_KEYS = frozenset(ALLOWED_FX_PRESET_KEYS | {"apply_fx", "volume", "pitch", "normalize", "trim_silence",
                                                "time_stretch"})

_RENDER_CACHE: dict = {}
# How many renders went into `_RENDER_CACHE`, in all and per file. A render
# changes what `rendered_duration_s` answers (exact instead of the hint),
# so the app's duration and arrangement memos key on these.
RENDERS = 0
_RENDERS_BY_PATH: dict = {}


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
    if _stretches(config):
        return False
    if not config.get("apply_fx", True):
        return True
    for key in ALLOWED_FX_PRESET_KEYS:
        if key.endswith("_enabled") and config.get(key, False):
            return False
    if config.get("convolution_ir"):
        return False
    return not (config.get("eq_bass", 0.0) or config.get("eq_treble", 0.0))


def _stretches(config: dict) -> bool:
    """True when `config` carries a `time_stretch` other than 1.0. Checked
    without importing audio_fx, which pulls in Pedalboard."""
    try:
        return float(config.get("time_stretch", 1.0)) != 1.0
    except (TypeError, ValueError):
        return False


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
    range_s = _clean_range(range_s)
    post_config = _slice_config(post_config, range_s)
    key = _cache_key(path, post_config, target_rate, range_s)
    cached = _RENDER_CACHE.get(key)
    if cached is not None:
        return cached
    return _store(key, _render_uncached(path, post_config, target_rate, range_s))


def _render_uncached(path: str, post_config: Optional[dict], target_rate: int,
                     range_s: Optional[tuple]) -> np.ndarray:
    """The read, the FX and the resample, with no memo and no counter: safe
    on a worker thread. `post_config` is already `_slice_config`ed."""
    from kokoro_gui.audio.mixer import resample

    mono, rate = _read_mono(path, range_s)
    if len(mono) and post_config and not is_identity(post_config):
        from kokoro_gui.engine.audio_fx import process_audio

        mono = np.asarray(process_audio(mono, rate, post_config), dtype=np.float32).reshape(-1)
    return resample(mono, rate, int(target_rate))


def _store(key: tuple, out: np.ndarray) -> np.ndarray:
    """Puts a finished render in the memo and counts it, once per key: the
    copy already there wins, so two renders of one key (a synchronous one
    racing a pool one) leave one entry and one count."""
    with _LOCK:
        held = _RENDER_CACHE.get(key)
        if held is not None:
            return held
        _RENDER_CACHE[key] = out
        _count_render(key[0])
        return out


def _count_render(abs_path: str) -> None:
    global RENDERS
    RENDERS += 1
    _RENDERS_BY_PATH[abs_path] = _RENDERS_BY_PATH.get(abs_path, 0) + 1


def render_count(path) -> int:
    """How many renders of `path` the memo has taken (0 for None)."""
    with _LOCK:
        return _RENDERS_BY_PATH.get(os.path.abspath(path), 0) if path else 0


def cached_render(path: str, post_config: Optional[dict], target_rate: int,
                  range_s: Optional[tuple] = None) -> Optional[np.ndarray]:
    """What `render` would return, or None when it would have to render."""
    range_s = _clean_range(range_s)
    return _RENDER_CACHE.get(_cache_key(path, _slice_config(post_config, range_s), target_rate, range_s))


# --- renders off the calling thread ---------------------------------------
#
# A worker only reads, processes and resamples (`_render_uncached`). It never
# writes `_RENDER_CACHE` or the counters: the arrangement's duration memo keys
# on `RENDERS` and `render_count`, so a render landing in the middle of a
# refresh would make verify mode compare two different answers. A finished
# render waits in `_FINISHED` until `drain()` runs on the GUI thread (the
# app's queued `_rendersReady` signal calls it), which stores it, counts it and
# calls the waiting callbacks there.

RENDER_WORKERS = 2

_LOCK = threading.Lock()
_POOL: Optional[ThreadPoolExecutor] = None
_PENDING: dict = {}      # key -> callbacks waiting for that render, until `drain` delivers it
_FINISHED: list = []     # (key, array or None) from workers, not yet drained
_BACKLOG: deque = deque()  # prewarm requests not yet started
_notify = None           # zero-arg hook a worker calls when `_FINISHED` gets its first item
_notified = False


def set_notifier(notify) -> None:
    """`notify()` is called from a worker thread when a render finishes and
    nothing is waiting to be drained; it must hand off to the thread that
    calls `drain()` (a queued Qt signal). None removes it."""
    global _notify
    _notify = notify


def render_async(path: str, post_config: Optional[dict], target_rate: int,
                 range_s: Optional[tuple], done) -> None:
    """`render` on the pool. `done(array)` is called with the render, or
    with None when the file can't be read: at once when the memo already holds
    it, otherwise from `drain()`. Calls for one key share one render."""
    range_s = _clean_range(range_s)
    post_config = _slice_config(post_config, range_s)
    key = _cache_key(path, post_config, target_rate, range_s)
    cached = _RENDER_CACHE.get(key)
    if cached is not None:
        done(cached)
        return
    with _LOCK:
        waiting = _PENDING.get(key)
        if waiting is not None:
            waiting.append(done)
            return
        _PENDING[key] = [done]
        _pool().submit(_work, key, path, post_config, target_rate, range_s)


def prewarm(requests) -> None:
    """Fills the memo ahead of a reader that will want these renders
    (`Transport.load` after an FX change). `requests` are `(path,
    post_config, target_rate, range_s)`. Replaces whatever backlog an earlier
    call left, and keeps at most `RENDER_WORKERS` of them in the pool at a
    time, so a `render_async` for something on screen never waits behind a
    whole book."""
    requests = list(requests)  # a generator may call back into this module
    with _LOCK:
        _BACKLOG.clear()
        _BACKLOG.extend(requests)
        for _ in range(RENDER_WORKERS):
            _pool().submit(_background_step)


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=RENDER_WORKERS, thread_name_prefix="post-render")
    return _POOL


def _work(key, path, post_config, target_rate, range_s) -> None:
    """Renders `key` on a worker and queues the result for `drain`. The
    caller has put `key` in `_PENDING`."""
    global _notified
    try:
        out = _render_uncached(path, post_config, target_rate, range_s)
    except Exception as e:  # noqa: BLE001 - an unreadable file is the caller's None
        logger.debug("render of %s failed: %s", path, e)
        out = None
    with _LOCK:
        _FINISHED.append((key, out))
        notify = _notify if not _notified else None
        _notified = True
    if notify is not None:
        try:
            notify()
        except Exception:  # noqa: BLE001 - a closed window must not kill the worker
            pass


def _background_step() -> None:
    """One prewarm render, then a request for the next one. Resubmitting
    puts the next step behind any `render_async` already queued."""
    while True:
        # One lock span from the pop to `_PENDING`, so `wait_idle` never sees
        # a request that has left the backlog and isn't pending yet.
        with _LOCK:
            if not _BACKLOG:
                return
            path, post_config, target_rate, range_s = _BACKLOG.popleft()
            range_s = _clean_range(range_s)
            post_config = _slice_config(post_config, range_s)
            try:
                key = _cache_key(path, post_config, target_rate, range_s)
            except Exception:  # noqa: BLE001 - a bad request skips itself, not the rest
                continue
            if key in _RENDER_CACHE or key in _PENDING:
                continue
            _PENDING[key] = []
            break
    _work(key, path, post_config, target_rate, range_s)
    with _LOCK:
        more = bool(_BACKLOG)
    if more:
        _pool().submit(_background_step)


def drain() -> int:
    """Stores and counts every render the workers have finished and calls the
    callbacks waiting on them. Call it on the thread that reads the memo (the
    GUI thread); returns how many it delivered."""
    global _notified
    with _LOCK:
        finished = list(_FINISHED)
        _FINISHED.clear()
        _notified = False
    for key, out in finished:
        with _LOCK:
            callbacks = _PENDING.pop(key, [])
        result = _store(key, out) if out is not None else None
        for callback in callbacks:
            try:
                callback(result)
            except Exception:  # noqa: BLE001 - one reader's failure must not drop the others
                logger.exception("a render callback failed")
    return len(finished)


def busy() -> bool:
    """True while a `render_async` or prewarm render hasn't been delivered."""
    with _LOCK:
        return bool(_PENDING or _BACKLOG or _FINISHED)


def wait_idle(timeout_s: float = 30.0) -> None:
    """Test hook: blocks until the pool and the backlog are empty, draining as
    it goes."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        drain()
        with _LOCK:
            idle = not _PENDING and not _BACKLOG and not _FINISHED
        if idle:
            return
        time.sleep(0.005)
    drain()


def render_slice(path: str, start_s: float, end_s: float, post_config: Optional[dict],
                 target_rate: int) -> np.ndarray:
    """`render` of the `[start_s, end_s)` seconds of `path`."""
    return render(path, post_config, target_rate, range_s=(start_s, end_s))


def _cache_key(path: str, post_config: Optional[dict], target_rate: int,
               range_s: Optional[tuple] = None) -> tuple:
    # Remembered until `revision.FILES` moves: a regenerated segment gets a
    # new file name (its key), and an import or Generate moves FILES.
    return (os.path.abspath(path), revision.file_mtime(path), post_key(post_config or {}), int(target_rate), range_s)


def duration_hint(segment, post_config: Optional[dict]) -> Optional[float]:
    """The segment's rendered length computed from what generation stored
    (`duration`, `onset_s`, `tail_s`) without reading audio, or None for a
    segment that predates those fields. Trim removes the onset and tail;
    pitch resamples by `2 ** (semitones / 12)`; `time_stretch` divides by
    its factor; nothing else in `process_audio` changes the length. A segment with a `range` is
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
    from kokoro_gui.engine.audio_fx import clamp_pitch_semitones, clamp_time_stretch

    semitones = clamp_pitch_semitones(config.get("pitch", 0.0))
    if semitones:
        length /= 2 ** (semitones / 12.0)
    return length / clamp_time_stretch(config.get("time_stretch", 1.0))


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
    global RENDERS
    with _LOCK:
        _BACKLOG.clear()
        _RENDER_CACHE.clear()
        _RENDERS_BY_PATH.clear()
        RENDERS += 1
