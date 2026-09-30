"""State derived from a `Document`, cached until the document changes
(Claude/PLAN_performance.md).

`DocumentIndex` answers the offset lookups (`clip_covering`,
`clip_extent`, `clip_text`, `get_clip`, `get_character`) with a bisect or a
dict read instead of a walk over every run. `DirtyTracker` holds each
clip's stale flag and recomputes only the clips whose inputs changed.

Both key on `kokoro_gui/daw/revision.py`'s counters. In verify mode
(`revision.VERIFY`, on for the whole test suite) each also computes its
answer from scratch and raises `StaleCacheError` on a mismatch.
"""
from __future__ import annotations

from bisect import bisect_right
from typing import Optional

from kokoro_gui.daw import revision


class StaleCacheError(AssertionError):
    """A cached answer differs from the one computed from scratch: some
    change didn't move a `revision` counter."""


def index_key(document) -> tuple:
    return (revision.TEXT, id(document.runs), len(document.runs), id(document.clips), len(document.clips),
            id(document.characters), len(document.characters))


class DocumentIndex:
    """Offsets and lookups for one state of a document's run list.

    `starts[i]` is where `runs[i]` begins. A clip's extent runs from the
    start of its first run to the end of its last, and its text joins its
    runs in document order, the same answers the run walks gave. `get_clip`
    and `get_character` return the first object with the id, as the linear
    `next(...)` lookups did."""

    __slots__ = ("key", "runs", "starts", "extents", "texts", "clips", "characters", "_text")

    def __init__(self, document, key: tuple):
        self.key = key
        runs = list(document.runs)
        starts = []
        extents = {}
        parts = {}
        pos = 0
        for run in runs:
            starts.append(pos)
            end = pos + len(run.text)
            clip_id = run.clip_id
            if clip_id is not None:
                extent = extents.get(clip_id)
                extents[clip_id] = (pos, end) if extent is None else (extent[0], end)
                parts.setdefault(clip_id, []).append(run.text)
            pos = end
        self.runs = runs
        self.starts = starts
        self.extents = extents
        self.texts = {clip_id: "".join(texts) for clip_id, texts in parts.items()}
        clips = {}
        for clip in document.clips:
            clips.setdefault(clip.id, clip)
        self.clips = clips
        characters = {}
        for character in document.characters:
            characters.setdefault(character.id, character)
        self.characters = characters
        self._text = None

    @property
    def text(self) -> str:
        if self._text is None:
            self._text = "".join(run.text for run in self.runs)
        return self._text

    def run_index_at(self, position: int) -> Optional[int]:
        """Index of the run covering `position` (start inclusive, end
        exclusive), or None. Runs are contiguous, so only the last run
        starting at or before `position` can cover it."""
        i = bisect_right(self.starts, position) - 1
        if i >= 0 and position < self.starts[i] + len(self.runs[i].text):
            return i
        return None

    def run_at(self, position: int):
        i = self.run_index_at(position)
        return self.runs[i] if i is not None else None

    def runs_in(self, start: int, end: int):
        """`(run, run_start, run_end)` for every run overlapping
        `[start, end)`, in order."""
        i = max(0, bisect_right(self.starts, start) - 1)
        runs, starts = self.runs, self.starts
        while i < len(runs):
            r_start = starts[i]
            if r_start >= end:
                break
            r_end = r_start + len(runs[i].text)
            if r_end > start:
                yield runs[i], r_start, r_end
            i += 1

    def clip_text(self, clip_id: str) -> str:
        return self.texts.get(clip_id, "")

    def extent(self, clip_id: str) -> Optional[tuple]:
        return self.extents.get(clip_id)

    def comparable(self) -> tuple:
        """What verify mode compares against a fresh build."""
        return (tuple(id(r) for r in self.runs), self.starts, self.extents, self.texts,
                {k: id(v) for k, v in self.clips.items()}, {k: id(v) for k, v in self.characters.items()})


def build_index(document) -> DocumentIndex:
    """The document's `DocumentIndex`, rebuilt when `index_key` moved."""
    key = index_key(document)
    cached = document.__dict__.get("_derived_index")
    if cached is not None and cached.key == key:
        if revision.VERIFY:
            fresh = DocumentIndex(document, key)
            if fresh.comparable() != cached.comparable():
                raise StaleCacheError("DocumentIndex is stale: the run list or clip list changed "
                                      "without moving revision.TEXT")
        return cached
    index = DocumentIndex(document, key)
    object.__setattr__(document, "_derived_index", index)
    return index


def cached(document, name: str, compute):
    """`compute(document)`, remembered on the document until `revision.TEXT`
    or `revision.MODEL` moves. For whole-document scans a paint needs (the
    untimed gaps of imported text) that depend on nothing but the model.
    Verify mode recomputes and compares."""
    key = (revision.TEXT, revision.MODEL)
    store = document.__dict__.get("_derived_memo")
    if store is None:
        store = {}
        object.__setattr__(document, "_derived_memo", store)
    hit = store.get(name)
    if hit is not None and hit[0] == key:
        if revision.VERIFY and compute(document) != hit[1]:
            raise StaleCacheError(f"derived.cached({name!r}) is stale: the document changed without moving "
                                  "a revision counter")
        return hit[1]
    value = compute(document)
    store[name] = (key, value)
    return value


def _segments_signature(segments) -> tuple:
    return tuple((s.order_index, s.text, s.cache_key, s.engine_version, s.raw, s.audio_path) for s in segments)


class DirtyTracker:
    """Each clip's stale flag, recomputed only when that clip's inputs
    changed.

    A pass runs when any `revision` counter, the app's input fingerprint
    (`Document.inputs_fn`: the generation settings, the lexicon, the engine
    settings) or one of the injected functions changed since the last one.
    It builds a token per clip from its text, character, overrides, segments
    and the character's preset, and runs `is_clip_dirty` only for a clip
    whose token differs from what it had last time. A keystroke inside one
    clip reruns one check; a settings change reruns every check, each of
    which is cheap with the key closure's own memo.

    A nested clip (a subproject) asks `Document.nested_state_fn` on every
    read, outside the memo: its staleness lives in another project, and the
    app keeps its own cache of it (`QtTTSApp.nested_state`)."""

    def __init__(self):
        self._state = None
        self._generated: frozenset = frozenset()
        self._nested: tuple = ()
        self._memo: dict = {}

    def clear(self) -> None:
        self._state = None
        self._generated = frozenset()
        self._nested = ()
        self._memo = {}

    def _pass_state(self, document) -> tuple:
        inputs = document.inputs_fn() if document.inputs_fn is not None else None
        return (revision.TEXT, revision.MODEL, revision.FILES, inputs, document.segment_key_fn,
                document.generation_config_fn, id(document.clips), len(document.clips))

    def dirty_ids(self, document) -> frozenset:
        state = self._pass_state(document)
        if state != self._state:
            self._pass(document, state)
        elif revision.VERIFY:
            self._verify(document)
        stale_nested = {clip.id for clip in self._nested if _nested_stale(document, clip)}
        return self._generated | stale_nested if stale_nested else self._generated

    def _pass(self, document, state) -> None:
        from kokoro_gui.daw.dirty import is_clip_dirty

        index = document.index()
        config_for = document.generation_config_fn or document.effective_config_for_clip
        key_fn = document.segment_key_fn
        character_tokens = {c.id: repr((c.preset_data, c.variants, c.backend_id))
                            for c in document.characters}
        inputs = state[3:6]
        memo = {}
        ids = set()
        nested = []
        for clip in document.clips:
            if clip.is_nested:
                nested.append(clip)
                continue
            text = index.clip_text(clip.id)
            token = (text, clip.source, clip.character_id, repr(clip.overrides),
                     character_tokens.get(clip.character_id), _segments_signature(clip.segments),
                     revision.FILES, inputs)
            hit = self._memo.get(clip.id)
            if hit is not None and hit[0] == token:
                stale = hit[1]
            else:
                stale = is_clip_dirty(clip, text, config_for(clip), key_fn=key_fn, file_exists=revision.file_exists)
            memo[clip.id] = (token, stale)
            if stale:
                ids.add(clip.id)
        self._memo = memo
        self._generated = frozenset(ids)
        self._nested = tuple(nested)
        self._state = state

    def _verify(self, document) -> None:
        from kokoro_gui.daw.dirty import is_clip_dirty

        config_for = document.generation_config_fn or document.effective_config_for_clip
        fresh = set()
        for clip in document.clips:
            if clip.is_nested:
                continue
            if is_clip_dirty(clip, document._clip_text_walk(clip), config_for(clip),
                             key_fn=document.segment_key_fn, file_exists=revision.file_exists):
                fresh.add(clip.id)
        if fresh != set(self._generated):
            raise StaleCacheError(f"DirtyTracker is stale: cached {sorted(self._generated)}, fresh {sorted(fresh)}; "
                                  "a clip's inputs changed without moving a revision counter")


def _nested_stale(document, clip) -> bool:
    return document.nested_state_fn(clip) if document.nested_state_fn is not None else True
