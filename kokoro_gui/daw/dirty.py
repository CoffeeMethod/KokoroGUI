"""Dirty/stale detection for `Clip`s (Q2/Q3: editing already-generated text
marks it dirty rather than auto-regenerating; the app waits for an explicit
scoped Generate action).

There is no stored "is_dirty" flag on `Clip` - it's computed on demand by
re-deriving what `CachingMixin.process_chunk_task`
(kokoro_gui/engine/caching.py) would compute for the clip's *current* text
and effective config, and comparing that against what its `Segment`s
actually recorded from the last successful generation. This keeps dirtiness
a pure function of current state instead of a flag that some code path could
forget to set or clear.

Deliberately reuses `compute_cache_key` and `clamp_pitch_semitones`
unmodified rather than inventing a parallel hashing scheme - see
`compute_expected_cache_hash` below, which mirrors
`CachingMixin.process_chunk_task`'s own cache-hash computation line for
line, and `predict_segment_texts`, which mirrors that same method's
"predict segments to verify cache integrity" block.
"""
import re

from kokoro_gui.daw.models import Segment
from kokoro_gui.engine.audio_fx import clamp_pitch_semitones
from kokoro_gui.engine.caching import compute_cache_key


def _effective_speed(config: dict) -> float:
    """Speed adjusted for pitch compensation - exactly the
    `eff_speed`/`pitch_semitones` computation in
    `CachingMixin.process_chunk_task`, duplicated here rather than imported
    since that logic lives inline in a method, not a standalone function."""
    eff_speed = config.get("speed", 1.0)
    pitch_semitones = clamp_pitch_semitones(config.get("pitch", 0.0))
    if pitch_semitones != 0.0:
        factor = 2 ** (pitch_semitones / 12.0)
        eff_speed = eff_speed / factor
    return eff_speed


def compute_expected_cache_hash(text: str, config: dict) -> str:
    """The cache hash a generation run over `text`/`config` would produce
    right now - mirrors `process_chunk_task`'s
    `compute_cache_key(text, config['voice'], eff_speed, lang_code,
    engine_id)` call exactly, including its use of the *whole* clip text
    (caching.py hashes once per chunk; per-sub-segment file names are that
    one hash plus a `_{sub_idx}` suffix, not independently hashed - see
    `Segment`'s docstring in models.py)."""
    lang_code = config.get("lang_code", "a")
    engine_id = config.get("engine_id", "kokoro")
    return compute_cache_key(text, config.get("voice"), _effective_speed(config), lang_code, engine_id)


def predict_segment_texts(text: str, config: dict) -> list:
    """The list of sub-segment texts `process_chunk_task` would expect to
    find cached (or generate) for `text`/`config` - mirrors that method's
    own heuristic split-and-strip-and-filter-empty logic exactly, so this
    stays a read of the same prediction rather than a second, divergent
    implementation of it."""
    split_pat = config.get("split_pattern", r"\n+")
    return [t.strip() for t in re.split(split_pat, text) if t.strip()]


def is_clip_dirty(clip, text: str, config: dict) -> bool:
    """True if `clip` needs (re)generation: it has never been generated, or
    its current text/effective-config no longer matches what its stored
    `Segment`s were generated from (Q16: an in-place edit keeps the same
    `Clip` object, now dirty; models.Document.apply_text_change is what
    guarantees a fully-deleted-then-retyped clip never reaches this function
    as the *same* object in the first place - see Q18)."""
    if not clip.segments:
        return True

    expected_hash = compute_expected_cache_hash(text, config)
    expected_count = len(predict_segment_texts(text, config))

    if len(clip.segments) != expected_count:
        return True
    return any(segment.cache_key != expected_hash for segment in clip.segments)


def build_segments_from_results(expected_hash: str, results: list) -> list:
    """Builds the `Segment` list for a clip from a `generate_clip_audio`/
    `process_chunk_task` result list - factors out the exact construction
    that was inlined once in `TimelineDock._on_clip_generation_finished`
    (kokoro_gui/qt/docks/timeline_dock.py) so both the single-clip and
    batch (item 3) Generate paths share one implementation.

    `order_index` comes from `enumerate(results)`, NOT each result dict's
    `seg_idx` field: every sub-segment of one `process_chunk_task` call
    shares the same `seg_idx` (the chunk's outer index), so using it
    directly would give every `Segment` in a multi-segment clip
    `order_index=0`, breaking `is_clip_dirty`'s segment-count comparison.
    """
    return [
        Segment(order_index=i, text=result["text"], cache_key=expected_hash,
                audio_path=result["path"], duration=result["duration"])
        for i, result in enumerate(results)
    ]
