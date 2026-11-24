"""The segment key function a document's dirty check uses.

`install_key_fn` sets `Document.segment_key_fn`, `generation_config_fn`
and `inputs_fn` (all runtime-only, never saved). `QtTTSApp._install_segment_key_fn`
calls it with its own bound methods for each open project; a headless
caller passes its `EngineSet.for_clip` and `core.config.generation_config`.
"""
from __future__ import annotations

import json
from typing import Callable, Optional

from kokoro_gui.daw import revision
from kokoro_gui.engine import caching
from kokoro_gui.engines.missing import MissingBackend


def make_key_fn(backend_for_clip: Callable, config_for_clip: Callable) -> Callable:
    """`key_fn(text, clip, engine_version=None) -> caching.segment_key`
    over `config_for_clip(clip)`, with the backend `backend_for_clip(clip)`
    returns (Claude/old/PLAN_tbaw_bundle.md section 2.3).

    Memoized on the text, the config and the version. A hit re-checks the
    voice file's mtime and the backend's `cache_key_extra` (Audio8's
    transcript, itself cached by mtime), which is the only way a key
    changes without its inputs changing (a re-saved mix, a re-recorded
    reference). So a rehighlight of a book costs about two stats per clip
    and no hashing or reads. A clip whose engine is missing
    (`MissingBackend`, grill EN6) keeps the key its first segment stored."""
    memo: dict = {}

    def key_fn(text, clip, engine_version=None):
        backend = backend_for_clip(clip)
        if isinstance(backend, MissingBackend):
            # No engine to compute a key with: the stored one stands while
            # the files are there. A clip with no segments or a missing file
            # is stale anyway (`is_clip_dirty`).
            return clip.segments[0].cache_key if clip.segments else None
        config = config_for_clip(clip)
        memo_key = (text, json.dumps(config, sort_keys=True, default=str), engine_version)
        name, _fp = caching.normalize_voice(config.get("voice"), backend, config.get("project_dir"))
        voice_file = backend.resolve_voice_file(name, config.get("project_dir")) if name else None
        # Remembered until `revision.FILES` moves (a voice save, window
        # activation, Force refresh).
        stamp = revision.file_mtime(voice_file) if voice_file else None
        extra = backend.cache_key_extra(config)
        hit = memo.get(memo_key)
        if hit is not None and hit[0] == stamp and hit[1] == extra:
            return hit[2]
        key = caching.segment_key(text, config, backend, engine_version)
        memo[memo_key] = (stamp, extra, key)
        return key

    return key_fn


def install_key_fn(document, backend_for_clip: Callable, config_for_clip: Callable,
                   dirty_config_for_clip: Callable, inputs_fn: Optional[Callable] = None) -> None:
    """Sets `document.segment_key_fn` to a fresh `make_key_fn` (an empty
    memo), `document.generation_config_fn` to `dirty_config_for_clip` (the
    dirty check reads the lexicon and segmentation keys from it) and
    `document.inputs_fn` to `inputs_fn`, the fingerprint of the outside
    state the config reads (`DirtyTracker` rechecks every clip when it
    changes; None means only the document's own changes count).

    Each callback is called per clip, at call time, so a caller that
    patches its own `backend_for` still reaches the key."""
    document.segment_key_fn = make_key_fn(backend_for_clip, config_for_clip)
    document.generation_config_fn = dirty_config_for_clip
    document.inputs_fn = inputs_fn
