"""Dirty/stale detection for `Clip`s (Q2/Q3: editing already-generated text
marks it dirty rather than auto-regenerating; the app waits for an explicit
scoped Generate action).

There is no stored "is_dirty" flag on `Clip` - it's computed on demand by
re-deriving the segment key `process_chunk_task`
(kokoro_gui/engine/caching.py) would compute for the clip's *current* text
and generation inputs, and comparing that against what its `Segment`s
recorded from the last successful generation. This keeps dirtiness a pure
function of current state instead of a flag some code path could forget to
set or clear.

Only generation inputs count. FX, volume, pitch's resample, normalize and
trim are read-time post-processing (kokoro_gui/audio/post.py) and changing
them never dirties a clip; `Segment.raw` is the one post-related check here,
and it only catches segments generated before that was true.

The key itself comes from `caching.segment_key`, which needs the backend
(to resolve and fingerprint a voice file, and for Audio8 to read the
transcript). This daw layer has no backend, so the app injects a key
function: `Document.segment_key_fn`, a memoized closure
`(text, clip, engine_version=None) -> key` (kokoro_gui/qt/app.py). Without
one (tests, headless use) `compute_expected_cache_hash` falls back to the
name-only `compute_cache_key`, which is the same value for a built-in voice
on Kokoro.

Two rules from the `.tbaw` plan (Claude/old/PLAN_tbaw_bundle.md section 2.3):
a segment is compared under the engine version it was generated with while
its file exists (TB9: a bundle from another machine's model version opens
clean), and a segment whose `audio_path` is set but whose file is missing is
dirty (TB11: close-time GC and an undo across it can't leave a clip playing
silence).
"""
import os

from kokoro_gui.daw.models import Segment
from kokoro_gui.engine import caching
from kokoro_gui.engine.caching import compute_cache_key, effective_speed
from kokoro_gui.engine.lexicon import apply_lexicon

# Compiled lexicon patterns, shared across dirty checks (the engine keeps
# its own per instance).
_lexicon_patterns = {}


def spoken_text(text: str, config: dict) -> str:
    """`text` after `config["lexicon"]`: what generation hands the engine,
    and so what the key and the predicted segment texts are over."""
    return apply_lexicon(text, config.get("lexicon") or {}, _lexicon_patterns)


def compute_expected_cache_hash(text: str, config: dict, engine_version=None, key_fn=None, clip=None) -> str:
    """The segment key a generation over `text`/`config` would produce right
    now. With `key_fn` (the app's closure) that is `caching.segment_key`
    over the clip's real generation inputs; without one it's the name-only
    `compute_cache_key` over `config["voice"]`, defaulting `lang_code` to
    "a" the way the pre-bundle check did. `engine_version=None` means the
    installed one. Applies the lexicon to `text` first."""
    return _key_for_spoken(spoken_text(text, config), config, engine_version, key_fn, clip)


def _key_for_spoken(text, config, engine_version, key_fn, clip):
    """`compute_expected_cache_hash` over text the lexicon has already
    rewritten (a lexicon isn't idempotent, so it must run once)."""
    if key_fn is not None:
        return key_fn(text, clip, engine_version)
    lang_code = config.get("lang_code", "a")
    engine_id = config.get("engine_id", "kokoro")
    return compute_cache_key(text, config.get("voice"), effective_speed(config), lang_code, engine_id,
                             engine_version=engine_version)


def predict_segment_texts(text: str, config: dict) -> list:
    """The pieces `process_chunk_task` would generate (or find cached) for
    `text`/`config`: `caching.split_segments`, read from there rather than
    duplicated."""
    return caching.split_segments(text, config)


def _collapse_ws(text) -> str:
    return " ".join((text or "").split())


def segment_file_missing(segment) -> bool:
    """True when the segment names a file that isn't there. A segment with
    no `audio_path` at all (a test fixture, a pre-audio record) isn't
    "missing", it's simply not backed by a file."""
    return bool(segment.audio_path) and not os.path.isfile(segment.audio_path)


def is_clip_dirty(clip, text: str, config: dict, key_fn=None) -> bool:
    """True if `clip` needs (re)generation: it has never been generated, or
    its current text/generation inputs no longer match what its stored
    `Segment`s were generated from (Q16: an in-place edit keeps the same
    `Clip` object, now dirty; models.Document.replace_text is what
    guarantees a fully-deleted-then-retyped clip never reaches this function
    as the *same* object in the first place - see Q18), or a segment's file
    is gone. A nested clip (a subproject) has no segments to compare;
    `Document.dirty_clips` asks the app about it, and this headless answer
    is "stale"."""
    if getattr(clip, "source", None) == "nested":
        return True
    if not clip.segments:
        return True
    # A segment baked with its FX at generation time (pre non-destructive
    # FX) can't be post-processed again without doubling the effect.
    if any(not segment.raw for segment in clip.segments):
        return True

    text = spoken_text(text, config)
    predicted = predict_segment_texts(text, config)
    if len(clip.segments) != len(predicted):
        return True
    # Same count, different boundaries (a splitter or budget change) is
    # dirty too. Whitespace is collapsed on both sides: segments generated
    # before per-piece generation stored KPipeline's rebuilt graphemes,
    # which can differ from the piece in whitespace alone.
    stored = sorted(clip.segments, key=lambda s: s.order_index)
    if any(_collapse_ws(s.text) != _collapse_ws(p) for s, p in zip(stored, predicted)):
        return True

    expected_by_version = {}
    for segment in clip.segments:
        if segment_file_missing(segment):
            return True
        # Stored keys win while the file is there (TB9). A missing file has
        # to regenerate with what's installed, and that's the version the
        # `None` key looks up.
        version = segment.engine_version if segment.audio_path else None
        if version not in expected_by_version:
            expected_by_version[version] = _key_for_spoken(text, config, version, key_fn, clip)
        if segment.cache_key != expected_by_version[version]:
            return True
    return False


def build_segments_from_results(expected_hash, results: list) -> list:
    """Builds the `Segment` list for a clip from a `generate_clip_audio`/
    `process_chunk_task` result list - one implementation shared by the
    single-clip and batch Generate paths (kokoro_gui/qt/docks/timeline_dock.py).

    `cache_key` and `engine_version` come from each result dict: the engine
    reports what it generated under, including any take bump (TB8), so
    nothing predicts the key before dispatch. `expected_hash` is the
    fallback for a result built by hand without a `cache_key`.

    `order_index` comes from `enumerate(results)`, NOT each result dict's
    `seg_idx` field: every sub-segment of one `process_chunk_task` call
    shares the same `seg_idx` (the chunk's outer index), so using it
    directly would give every `Segment` in a multi-segment clip
    `order_index=0`, breaking `is_clip_dirty`'s segment-count comparison.

    `raw` comes from each result's "raw" field (every backend's
    `process_chunk_task` sets it; `generate_clip_audio` always requests raw
    output) and defaults True for a result dict built by hand.
    """
    return [
        Segment(order_index=i, text=result["text"], cache_key=result.get("cache_key") or expected_hash,
                audio_path=result["path"], duration=result["duration"],
                raw=bool(result.get("raw", True)), engine_version=result.get("engine_version"),
                words=[list(w) for w in result.get("words") or []],
                onset_s=result.get("onset_s"), tail_s=result.get("tail_s"))
        for i, result in enumerate(results)
    ]


def carry_segment_timing(new_segments: list, previous_lists) -> None:
    """Copies `words`, `onset_s` and `tail_s` onto freshly built segments
    from an earlier segment for the same file (same `cache_key`,
    `order_index` and `audio_path`) when the new one lacks them. A cache
    hit hands back the file without re-running the model, so it has no
    token timings of its own; the segment it was first generated as does."""
    known = {}
    for segments in previous_lists:
        for segment in segments or []:
            known.setdefault((segment.cache_key, segment.order_index, segment.audio_path), segment)
    for segment in new_segments:
        old = known.get((segment.cache_key, segment.order_index, segment.audio_path))
        if old is None:
            continue
        if not segment.words and old.words:
            segment.words = [list(w) for w in old.words]
        if segment.onset_s is None:
            segment.onset_s = old.onset_s
        if segment.tail_s is None:
            segment.tail_s = old.tail_s


def take_from_results(results: list, default: int = 0) -> int:
    """The take index the engine landed on for a clip, read off its results
    (they all carry the same one); `default` when the results don't say."""
    for result in results:
        if "take" in result:
            return int(result["take"] or 0)
    return default
