"""Core dataclasses: Document, Run, Clip, Segment, Track, Character.

Resolves the "grill chat" architecture (Claude/Kokorogui grill chat.md, Q1-Q29
plus the Text Editor Grill TE1-TE6) into concrete types. As of
Claude/PLAN_text_editor_redesign.md, the offset-based `Clip` (`start_offset`/
`end_offset` into a flat `Document.text`) has been replaced by a **tagged
run list**: `Document.runs` is the authoritative structure, each `Run` a
contiguous stretch of text carrying (at most) one `Clip.id`. A clip's extent
is wherever its id is applied across the run list, found by walking runs -
not stored as numbers that need shifting on every edit. `Document.text`
remains available as a *derived* property (the join of every run's text),
used only for TTS generation, cache-key input, word count, and export - it
is never the authoritative field.

- A `Clip` is a user-facing unit that may map to multiple engine-level
  `Segment`s (Q3) - the pieces a clip's text is generated as
  (kokoro_gui/engine/segmenting.py) stay internal to the clip, not surfaced
  as the primary structure.
- `Clip.id` is a UUID, never derived from content - this is what keeps cache
  identity (a content hash, see kokoro_gui/engine/caching.py) and clip
  identity (Q18) genuinely independent: deleting a clip removes the object
  entirely (and untags its runs), and a later, coincidentally-identical clip
  gets a fresh id and default metadata even though its audio may still
  cache-hit.
- `Track` is a lane keyed by `character_id` for auto-placement (Q8) - kept as
  a separate object from `Character` on purpose, since a clip can be dragged
  onto a track whose `character_id` differs from the clip's own (Q9).
- `Character` wraps the existing `presets/*.json` shape (see
  kokoro_gui/engine/presets.py's `ALLOWED_PRESET_KEYS`) rather than
  reinventing preset storage.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

from kokoro_gui.daw.undo import UndoStack
from kokoro_gui.engine.presets import ALLOWED_PRESET_KEYS, filter_allowed_keys

# Small fixed palette cycled by migration.py when assigning default
# highlight colors to characters created from existing presets - not a
# proposed app-wide color scheme, just distinguishable defaults a user can
# change later from the (future) Characters menu, per the design doc's note
# that per-character highlight colors are one of the few places this redesign
# does make an actual color decision.
# Eight hues at similar lightness so any of them takes the same label
# color on a clip and tints the transcript evenly in both themes; the
# Characters dialog offers the same eight as the color picker's presets.
DEFAULT_HIGHLIGHT_PALETTE = (
    "#e3a72f",  # amber
    "#4f8fe6",  # blue
    "#e0655c",  # coral
    "#3fae7a",  # green
    "#a76fd6",  # purple
    "#3bb3c4",  # teal
    "#e78a3e",  # orange
    "#d75c9a",  # pink
)


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Character:
    """A named, reusable voice/settings preset - Q7's "character = a preset
    of settings". Wraps a `presets/*.json`-shaped dict (`preset_data`)
    instead of reinventing preset storage, so
    `PresetsMixin.load_preset`/`load_fx_preset`
    (kokoro_gui/engine/presets.py) keep working unchanged."""

    name: str
    preset_data: dict = field(default_factory=dict)
    highlight_color: str = DEFAULT_HIGHLIGHT_PALETTE[0]
    backend_id: str = "kokoro"
    # Variant name ("angry", "whisper") -> voice name. For a cloning backend
    # (Audio8) the voice is a reference name, so a clip's
    # `overrides["variant"]` swaps the reference and the segment key follows
    # through the normal voice fingerprint. Ignored for other backends.
    variants: dict = field(default_factory=dict)
    id: str = field(default_factory=_new_id)
    # Fields this version doesn't know, carried through a load/save so an
    # older KokoroGUI doesn't strip what a newer one wrote (see
    # serialization.py). Nothing in the app reads it. For a Character it
    # also holds `extra["preset_data"]`: the preset keys ALLOWED_PRESET_KEYS
    # strips from what reaches a config dict.
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_preset_dict(cls, name, preset_data, highlight_color=None, backend_id="kokoro", id=None):
        """Wraps a preset dict already loaded via `PresetsMixin.load_preset`
        (or an equivalent plain `json.load`) into a `Character`. Only keys in
        `ALLOWED_PRESET_KEYS` are kept, mirroring the same untrusted-preset
        whitelist `filter_allowed_keys` enforces elsewhere - a `Character`
        should never carry more trust than a raw preset file already has."""
        kwargs = {"name": name, "preset_data": filter_allowed_keys(preset_data or {}, ALLOWED_PRESET_KEYS)}
        if highlight_color is not None:
            kwargs["highlight_color"] = highlight_color
        if backend_id is not None:
            kwargs["backend_id"] = backend_id
        if id is not None:
            kwargs["id"] = id
        return cls(**kwargs)

    def to_preset_dict(self) -> dict:
        """The plain dict shape `PresetsMixin`'s preset JSON files use - the
        inverse of `from_preset_dict`, for saving a `Character`'s settings
        back out as a `presets/<name>.json` file."""
        return dict(self.preset_data)


@dataclass
class Track:
    """An organizational timeline lane. Not identical to a `Character` (Q8) -
    kept separate so a clip can be dragged onto a track belonging to a
    different character (Q9's reassign-vs-move prompt)."""

    name: str
    character_id: Optional[str] = None
    order_index: int = 0
    # Mixer controls, applied when the transport and the exporter build
    # their schedule: fader gain, mute, solo (any soloed track silences the
    # rest), constant-power pan in [-1, 1], and a volume automation lane of
    # `[seconds, gain]` breakpoints (gain in [0, 2], linear between points;
    # empty means none).
    gain: float = 1.0
    mute: bool = False
    solo: bool = False
    pan: float = 0.0
    automation: list = field(default_factory=list)
    id: str = field(default_factory=_new_id)
    extra: dict = field(default_factory=dict)  # unknown fields, see Character


@dataclass
class Segment:
    """One engine-level generation unit inside a `Clip` - a persisted record
    of a `(index, text, config)` tuple already handled by
    `CachingMixin.process_chunk_task` (kokoro_gui/engine/caching.py). Not a
    new execution concept: `cache_key` is the chunk-level hash
    `compute_cache_key` returns for the clip's full text (see
    kokoro_gui/daw/dirty.py), and `order_index` is that hash's `_{i}` file
    suffix (`caching.py`'s `sub_idx`) - multiple segments of one clip share
    the same `cache_key` and differ only by `order_index`.

    `raw` is True when `audio_path` holds unprocessed model output
    (`process_chunk_task` with `config['raw_output']`), which is what every
    clip generated since non-destructive FX landed is, so it defaults True;
    FX, volume, pitch, normalize and trim are applied on read by
    `kokoro_gui.audio.post`. `raw=False` marks a segment baked with its FX at
    generation time: `serialization.document_from_dict` assigns it to a
    saved segment that predates the flag, and `dirty.is_clip_dirty` reports
    such a clip dirty so it regenerates once (a cache hit when caching is
    on) instead of getting FX applied twice. `duration` is the raw length;
    the arrangement measures the rendered length itself.

    `cache_key` is `caching.segment_key` for the clip's text and generation
    inputs, and it is also the stem of the file `audio_path` names in a
    project dir (`<cache_key>_<order_index>.<ext>`). `engine_version` is the
    version string the backend reported when this segment was generated;
    the dirty check keys with it while the file is present, so a project
    made with one model version opens clean on a machine with another
    (grill TB9). `None` means "written before the field", which the dirty
    check treats as the installed version."""

    order_index: int = 0
    text: str = ""
    cache_key: Optional[str] = None
    audio_path: Optional[str] = None
    duration: Optional[float] = None
    raw: bool = True
    engine_version: Optional[str] = None
    # `[text, start_s, end_s]` per spoken word, relative to this segment's
    # start: Kokoro's token timestamps, or a Whisper alignment for engines
    # without them (kokoro_gui/daw/wordalign.py). Not a generation input.
    words: list = field(default_factory=list)
    # Seconds of leading and trailing audio under the trim threshold
    # (`audio_fx.TRIM_THRESHOLD`), measured from the raw output, so the
    # trimmed length is known without reading the file. None: not measured
    # (a segment generated before these fields).
    onset_s: Optional[float] = None
    tail_s: Optional[float] = None
    id: str = field(default_factory=_new_id)
    extra: dict = field(default_factory=dict)  # unknown fields, see Character


CLIP_STATUSES = ("todo", "generated", "approved", "needs_rewrite")


@dataclass
class Clip:
    """A user-facing unit of text-anchored audio (Q19: every clip, imported
    or generated, is text-anchored). Unlike the retired offset-based model,
    a `Clip` no longer stores its own extent - that's wherever `Document.runs`
    tags a run with this clip's `id` (see `Document.clip_extent`/
    `Document.clip_text`)."""

    character_id: Optional[str] = None
    track_id: Optional[str] = None
    overrides: dict = field(default_factory=dict)
    fx_override: Optional[dict] = None
    timeline_timestamp: Optional[float] = None
    segments: list = field(default_factory=list)
    source: str = "generated"  # "generated" | "imported"
    original_audio_path: Optional[str] = None
    # Silence before this clip when it's placed after its text-order
    # predecessor; None uses the document's `gap_s`/`paragraph_gap_s`.
    gap_before_s: Optional[float] = None
    # Parked takes: take index -> segment list. `segments` is the active
    # take and `overrides["take"]` its index, so readers of `segments` never
    # see a parked one. JSON stores the index as a string.
    takes: dict = field(default_factory=dict)
    fade_in_s: float = 0.0
    fade_out_s: float = 0.0
    # Review state (`CLIP_STATUSES`) and a free-text note, for the cue sheet.
    status: str = "todo"
    note: str = ""
    # The original-language line a dub is written against, when there is one.
    source_text: Optional[str] = None
    id: str = field(default_factory=_new_id)
    extra: dict = field(default_factory=dict)  # unknown fields, see Character

    def __post_init__(self):
        if self.source not in ("generated", "imported"):
            raise ValueError(f"Clip.source must be 'generated' or 'imported', got {self.source!r}")


@dataclass
class Run:
    """One contiguous stretch of `Document` text carrying (at most) one
    `Clip.id` - the tagged-run-list replacement for offset-shifted `Clip`
    ranges (Claude/PLAN_text_editor_redesign.md, TE5). Mirrors, at the
    Qt-free data-model level, the `QTextCharFormat` custom property the real
    `TranscriptEditor` widget applies to the equivalent stretch of its
    `QTextDocument` - this class is what lets `kokoro_gui/daw/` stay
    Qt-free (per this package's `__init__.py`) while still round-tripping
    through `serialization.py`'s JSON shape and being usable headlessly (a
    future CLI, or this module's own test suite).

    `clip_id=None` means "untagged" - ordinary narration nobody has assigned
    a character to yet, exactly like today's "some text has no clip" state.
    `kind` mirrors the owning `Clip.source` ("generated"/"imported") for a
    tagged run, or is `None` for an untagged one; `"placeholder"` is reserved
    for a future ASR-anchored import awaiting transcription (deliberately
    unused for now - see the plan doc's "Open items", this pass only
    reserves the marker, it doesn't build the import UX behind it).
    """

    text: str = ""
    clip_id: Optional[str] = None
    kind: Optional[str] = None
    extra: dict = field(default_factory=dict)  # unknown fields, see Character


@dataclass
class Document:
    """The whole project's source of truth (Q15's closing principle). Owns
    the canonical run list plus the clip/track/character metadata layered on
    top of it. `text` is a computed property (the join of every run's text),
    not a stored field - see the module docstring."""

    runs: list = field(default_factory=list)
    clips: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    characters: list = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    # Runtime/session-only (item 4, "Undo/redo") - a `UndoStack` needs a
    # reference to its owning `Document`, which a `field(default_factory=...)`
    # can't capture (no access to `self` there), hence the `__post_init__`
    # construction below instead. NEVER include this in
    # `kokoro_gui/daw/serialization.py`'s `document_to_dict` (or any other
    # persistence path) - undo history is not part of a saved project, it's
    # this session's editing history only.
    undo_stack: Optional[UndoStack] = field(default=None, init=False, repr=False)
    # Runtime-only like `undo_stack`: `(text, clip, engine_version=None) ->
    # segment key`, set by the app (kokoro_gui/qt/app.py's
    # `_switch_document`) as a closure over the active backend and the
    # project dir, since this daw layer has neither. `dirty_clips` hands it
    # to `dirty.is_clip_dirty`; unset (tests, headless use) the check falls
    # back to the name-only `compute_cache_key`. Never serialized.
    segment_key_fn: Optional[Callable] = field(default=None, init=False, repr=False)
    # Runtime-only, set beside `segment_key_fn`: `clip -> config`, the app's
    # `_assemble_generation_config`. The dirty check reads the lexicon and
    # the segmentation keys from it, which live in app settings, not in the
    # clip's character or overrides. Unset, `dirty_clips` falls back to
    # `effective_config_for_clip`. Never serialized.
    generation_config_fn: Optional[Callable] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.undo_stack = UndoStack(self)

    @classmethod
    def from_plain_text(cls, text: str = "", **kwargs) -> "Document":
        """Convenience constructor for a fresh `Document` whose entire text
        is one untagged run - the common case for a brand-new project, a
        freshly-loaded plain-text import, or a test fixture that doesn't
        care about tagging."""
        return cls(runs=[Run(text=text)] if text else [], **kwargs)

    def set_plain_text(self, text: str) -> None:
        """Replaces the whole document with one untagged run, discarding
        every existing run tag - a full reload/reset, NOT an ordinary edit
        primitive (see `replace_text` for that)."""
        self.runs = [Run(text=text)] if text else []

    # -- text (derived) ------------------------------------------------------

    @property
    def text(self) -> str:
        return "".join(run.text for run in self.runs)

    @text.setter
    def text(self, value: str) -> None:
        """Convenience alias for `set_plain_text` - a full reload/reset, NOT
        an ordinary edit primitive (use `replace_text` for that). Kept as a
        settable property (rather than getter-only) since a lot of call
        sites - tests especially - reasonably expect `doc.text = "..."` to
        keep working the way it always did before `text` became derived."""
        self.set_plain_text(value)

    def _iter_runs_with_offsets(self):
        """Yields `(run, start, end)` for every run, in document order -
        the shared walk every offset-deriving lookup below builds on."""
        pos = 0
        for run in self.runs:
            end = pos + len(run.text)
            yield run, pos, end
            pos = end

    def _run_covering(self, position: int) -> Optional[Run]:
        for run, start, end in self._iter_runs_with_offsets():
            if start <= position < end:
                return run
        return None

    # -- lookups -----------------------------------------------------------

    def get_character(self, character_id: Optional[str]) -> Optional[Character]:
        if character_id is None:
            return None
        return next((c for c in self.characters if c.id == character_id), None)

    def get_character_by_name(self, name: str) -> Optional[Character]:
        """Case-insensitive, whitespace-stripped lookup by `Character.name`
        - unlike `get_character`/`get_track`/`get_clip` above, this is a
        name-based (not id-based) lookup, since a `[Speaker:FX]:` tag
        (auto-split, item 7 of the DAW-for-text remaining-work roadmap)
        names a character by its display name, not its id. Returns `None`
        if no character's name matches, case-insensitively, ignoring
        leading/trailing whitespace on both sides."""
        if name is None:
            return None
        target = name.strip().lower()
        return next((c for c in self.characters if c.name.strip().lower() == target), None)

    def get_track(self, track_id: Optional[str]) -> Optional[Track]:
        if track_id is None:
            return None
        return next((t for t in self.tracks if t.id == track_id), None)

    def get_clip(self, clip_id: str) -> Optional[Clip]:
        return next((c for c in self.clips if c.id == clip_id), None)

    def clip_covering(self, position: int) -> Optional[Clip]:
        """The `Clip` covering text offset `position`, if any (inclusive
        start, exclusive end) - reads whichever run's tag covers that
        position, rather than scanning a stored offset-range list."""
        run = self._run_covering(position)
        return self.get_clip(run.clip_id) if run is not None else None

    def clip_extent(self, clip_id: str) -> Optional[tuple]:
        """The `(start, end)` character-offset span a clip's tagged run(s)
        currently occupy in `self.text`, or `None` if no run carries that
        id. Computed on demand by walking `self.runs` - this is the
        run-based replacement for reading `clip.start_offset`/`end_offset`
        directly (timeline positioning, sub-range TTS replacement, etc. all
        go through this now)."""
        start = end = None
        for run, r_start, r_end in self._iter_runs_with_offsets():
            if run.clip_id == clip_id:
                if start is None:
                    start = r_start
                end = r_end
        return None if start is None else (start, end)

    # -- text/config -------------------------------------------------------

    def clip_text(self, clip: Clip) -> str:
        """The clip's current text - every run tagged with `clip.id`,
        concatenated in document order."""
        return "".join(run.text for run in self.runs if run.clip_id == clip.id)

    def effective_config_for_clip(self, clip: Clip) -> dict:
        """The clip's character preset merged with its own overrides (Q7:
        editing a character retroactively affects every clip using it,
        unless that clip has an explicit override) - reuses the same
        `filter_allowed_keys` whitelist `presets.py` already applies to a
        loaded preset, so a clip override can't smuggle in a disallowed key
        either."""
        character = self.get_character(clip.character_id)
        config = dict(character.preset_data) if character else {}
        config.update(filter_allowed_keys(clip.overrides, ALLOWED_PRESET_KEYS))
        return config

    def dirty_clips(self) -> list:
        """Every `Clip` that needs (re)generation - see `dirty.is_clip_dirty`
        for what "dirty" means. Imported lazily to avoid a module-level
        import cycle (dirty.py has no need to import models.py, but keeping
        the dependency one-directional and local here is simplest)."""
        from kokoro_gui.daw.dirty import is_clip_dirty

        config_for = self.generation_config_fn or self.effective_config_for_clip
        return [
            clip
            for clip in self.clips
            if is_clip_dirty(clip, self.clip_text(clip), config_for(clip), key_fn=self.segment_key_fn)
        ]

    # -- run-list maintenance (private) -------------------------------------

    def _split_at(self, offset: int) -> None:
        """Splits whichever run straddles `offset` into two runs at that
        boundary, so later code can retag/replace an exact `[start, end)`
        span without disturbing text on either side of it. A no-op if
        `offset` already falls on a run boundary (including the document's
        own start/end)."""
        if offset <= 0 or offset >= len(self.text):
            return
        pos = 0
        for i, run in enumerate(self.runs):
            end = pos + len(run.text)
            if pos < offset < end:
                cut = offset - pos
                self.runs[i:i + 1] = [
                    Run(text=run.text[:cut], clip_id=run.clip_id, kind=run.kind),
                    Run(text=run.text[cut:], clip_id=run.clip_id, kind=run.kind),
                ]
                return
            pos = end

    def _normalize_runs(self) -> None:
        """Drops zero-length runs and merges adjacent runs sharing the same
        `clip_id`/`kind` - the run-list equivalent of Qt's own "typing
        inside a run just extends it" merge behavior, kept true here too so
        two operations that happen to retag neighboring spans identically
        don't leave a meaningless split between them."""
        merged: list = []
        for run in self.runs:
            if not run.text:
                continue
            if merged and merged[-1].clip_id == run.clip_id and merged[-1].kind == run.kind:
                merged[-1] = Run(text=merged[-1].text + run.text, clip_id=run.clip_id, kind=run.kind)
            else:
                merged.append(run)
        self.runs = merged

    def _retag_range(self, start: int, end: int, clip_id: Optional[str], kind: Optional[str]) -> None:
        """Replaces whatever runs currently occupy `[start, end)` with a
        single run of that same text, tagged `clip_id`/`kind` - the shared
        "retag an exact span" primitive `assign_character_to_range` below
        applies once per clip it touches (the new clip's span, plus one per
        leftover fragment). Never changes `len(self.text)`."""
        self._split_at(start)
        self._split_at(end)

        new_runs: list = []
        merged_text_parts: list = []
        inserted_at: Optional[int] = None
        pos = 0
        for run in self.runs:
            run_end = pos + len(run.text)
            if run_end <= start or pos >= end:
                new_runs.append(run)
            else:
                merged_text_parts.append(run.text)
                if inserted_at is None:
                    inserted_at = len(new_runs)
                    new_runs.append(None)
            pos = run_end

        new_runs[inserted_at] = Run(text="".join(merged_text_parts), clip_id=clip_id, kind=kind)
        self.runs = new_runs
        self._normalize_runs()

    # -- UI-driven authoring (Q20): Characters menu / paste-splitting ------

    def assign_character_to_range(self, start: int, end: int, character_id: Optional[str]) -> Clip:
        """Assigns `character_id` to `self.text[start:end]`, creating a new
        `Clip` for that exact range and splitting off "leftover" clips for
        whatever the range partially overlapped - the shared split-or-create
        primitive behind the transcript panel's Characters menu and
        paste-splitting, and (later) the timeline's sub-range TTS replacement
        and auto-split features.

        A clip fully inside `[start, end)` is simply removed (no leftover).
        A clip only partially overlapping keeps a leftover fragment for the
        portion outside `[start, end)`, carrying its original character/
        track/overrides/fx_override - but as a brand-new `Clip` (fresh id,
        no segments), since a split invalidates whatever was cached for the
        now-different range. Even an exact range-for-range reassignment goes
        through remove-then-recreate: identity is independent of content,
        the same rule `replace_text`'s fully-consumed-clip removal already
        establishes.

        No manual dirty-marking is needed - every clip this method touches
        ends up with no `segments`, which `dirty.is_clip_dirty` already
        treats as dirty.
        """
        if end <= start:
            raise ValueError(f"assign_character_to_range requires end > start, got start={start}, end={end}")
        text_len = len(self.text)
        if start < 0 or end > text_len:
            raise ValueError(
                f"assign_character_to_range requires [start, end) within [0, {text_len}), "
                f"got start={start}, end={end}"
            )

        track_id = next((t.id for t in self.tracks if t.character_id == character_id), None)

        overlapping_ids = set()
        for run, r_start, r_end in self._iter_runs_with_offsets():
            if run.clip_id is not None and r_start < end and r_end > start:
                overlapping_ids.add(run.clip_id)

        leftover_ranges = []  # (start, end, old_clip)
        for clip_id in overlapping_ids:
            old_clip = self.get_clip(clip_id)
            if old_clip is None:
                continue
            o_start, o_end = self.clip_extent(clip_id)
            if o_start < start:
                leftover_ranges.append((o_start, start, old_clip))
            if o_end > end:
                leftover_ranges.append((end, o_end, old_clip))

        leftover_clips = [
            Clip(
                character_id=old_clip.character_id, track_id=old_clip.track_id,
                overrides=dict(old_clip.overrides), fx_override=old_clip.fx_override,
                source=old_clip.source, original_audio_path=old_clip.original_audio_path,
            )
            for (_l_start, _l_end, old_clip) in leftover_ranges
        ]

        new_clip = Clip(character_id=character_id, track_id=track_id)

        self.clips = [c for c in self.clips if c.id not in overlapping_ids]
        self.clips.extend(leftover_clips)
        self.clips.append(new_clip)

        # None of these _retag_range calls change len(self.text), so it's
        # safe to apply them in any order using offsets all computed above,
        # against the pre-edit run layout.
        self._retag_range(start, end, new_clip.id, new_clip.source)
        for (l_start, l_end, _old_clip), leftover_clip in zip(leftover_ranges, leftover_clips):
            self._retag_range(l_start, l_end, leftover_clip.id, leftover_clip.source)

        return new_clip

    # -- plain text edits (typing, paste, programmatic replace) -------------

    def replace_text(self, position: int, chars_removed: int, chars_added: int, new_text: str) -> list:
        """Applies one `QTextDocument.contentsChange`-shaped edit
        (position/charsRemoved/charsAdded, plus the resulting full text -
        Qt's signal doesn't carry the inserted characters themselves, so the
        caller passes `editor.toPlainText()` after the change) directly
        against the run list - the run-based replacement for the retired
        `apply_text_change`/offset-shift mechanism (see the module
        docstring's "core inversion").

        A clip whose entire extent falls inside `[position, position +
        chars_removed)` is fully consumed and removed outright (Q18: a
        later, textually-identical retype creates a brand-new `Clip` with a
        fresh id even though its audio may still cache-hit). A clip that
        only partially overlaps the edited range keeps its identity - the
        portion of its run(s) outside the edited range is untouched by this
        splice, so it simply survives, shrunk or extended in place.

        The newly inserted text inherits the tag of whatever run ends
        exactly at `position` (i.e. the text immediately to the edit's
        left) - ordinary "typing extends the current run" behavior, same as
        a real rich-text editor's cursor format inheritance. Typing at the
        very start of the document, or right after a clip that this same
        edit fully consumed, leaves the inserted text untagged.

        Returns the list of `Clip`s removed by this edit, for callers that
        need to react (e.g. dropping them from a track view).
        """
        removed_end = position + chars_removed

        overlapping_ids = set()
        for run, r_start, r_end in self._iter_runs_with_offsets():
            if run.clip_id is not None and r_start < removed_end and r_end > position:
                overlapping_ids.add(run.clip_id)

        fully_consumed_ids = {
            clip_id for clip_id in overlapping_ids
            if chars_removed > 0
            for (o_start, o_end) in [self.clip_extent(clip_id)]
            if position <= o_start and o_end <= removed_end
        }

        removed_clips = [self.get_clip(clip_id) for clip_id in fully_consumed_ids]
        self.clips = [c for c in self.clips if c.id not in fully_consumed_ids]

        inherited = self._run_covering(position - 1) if position > 0 else None
        inherited_clip_id = inherited.clip_id if inherited is not None else None
        inherited_kind = inherited.kind if inherited is not None else None
        if inherited_clip_id in fully_consumed_ids:
            inherited_clip_id = None
            inherited_kind = None

        inserted_text = new_text[position:position + chars_added]

        self._split_at(position)
        self._split_at(removed_end)

        new_run = Run(text=inserted_text, clip_id=inherited_clip_id, kind=inherited_kind)
        new_runs: list = []
        inserted = False
        pos = 0
        for run in self.runs:
            run_end = pos + len(run.text)
            if pos >= position and run_end <= removed_end and pos < removed_end:
                if not inserted:
                    new_runs.append(new_run)
                    inserted = True
                pos = run_end
                continue
            if not inserted and pos >= position:
                new_runs.append(new_run)
                inserted = True
            new_runs.append(run)
            pos = run_end
        if not inserted:
            new_runs.append(new_run)

        self.runs = new_runs
        self._normalize_runs()
        return removed_clips
