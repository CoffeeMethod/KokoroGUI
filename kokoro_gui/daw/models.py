"""Core dataclasses: Document, Clip, Segment, Track, Character.

Resolves the "grill chat" architecture (Claude/Kokorogui grill chat.md, Q1-Q29)
into concrete types:

- `Document.text` is the canonical source of truth (Q15's closing principle:
  "the document is the source of truth; generated audio is a render of that
  document state").
- A `Clip` is a user-facing unit that may map to multiple engine-level
  `Segment`s (Q3) - the engine's own text splitting (KPipeline's
  `split_pattern`) stays internal to a clip, not surfaced as the primary
  structure.
- `Clip.id` is a UUID, never derived from content - this is what keeps cache
  identity (a content hash, see kokoro_gui/engine/caching.py) and clip
  identity (Q18) genuinely independent: deleting a clip removes the object
  entirely, and a later, coincidentally-identical clip gets a fresh id and
  default metadata even though its audio may still cache-hit.
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
from typing import Optional

from kokoro_gui.engine.presets import ALLOWED_PRESET_KEYS, filter_allowed_keys

# Small fixed palette cycled by migration.py when assigning default
# highlight colors to characters created from existing presets - not a
# proposed app-wide color scheme, just distinguishable defaults a user can
# change later from the (future) Characters menu, per the design doc's note
# that per-character highlight colors are one of the few places this redesign
# does make an actual color decision.
DEFAULT_HIGHLIGHT_PALETTE = (
    "#f4b400",  # amber
    "#4285f4",  # blue
    "#db4437",  # red
    "#0f9d58",  # green
    "#ab47bc",  # purple
    "#00acc1",  # teal
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
    id: str = field(default_factory=_new_id)

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
    id: str = field(default_factory=_new_id)


@dataclass
class Segment:
    """One engine-level generation unit inside a `Clip` - a persisted record
    of a `(index, text, config)` tuple already handled by
    `CachingMixin.process_chunk_task` (kokoro_gui/engine/caching.py). Not a
    new execution concept: `cache_key` is the chunk-level hash
    `compute_cache_key` returns for the clip's full text (see
    kokoro_gui/daw/dirty.py), and `order_index` is that hash's `_{i}` file
    suffix (`caching.py`'s `sub_idx`) - multiple segments of one clip share
    the same `cache_key` and differ only by `order_index`."""

    order_index: int = 0
    text: str = ""
    cache_key: Optional[str] = None
    audio_path: Optional[str] = None
    duration: Optional[float] = None
    id: str = field(default_factory=_new_id)


@dataclass
class Clip:
    """A user-facing unit of text-anchored audio (Q19: every clip, imported
    or generated, is text-anchored). `start_offset`/`end_offset` index into
    the owning `Document.text`."""

    start_offset: int = 0
    end_offset: int = 0
    character_id: Optional[str] = None
    track_id: Optional[str] = None
    overrides: dict = field(default_factory=dict)
    fx_override: Optional[dict] = None
    timeline_timestamp: Optional[float] = None
    segments: list = field(default_factory=list)
    source: str = "generated"  # "generated" | "imported"
    original_audio_path: Optional[str] = None
    id: str = field(default_factory=_new_id)

    def __post_init__(self):
        if self.source not in ("generated", "imported"):
            raise ValueError(f"Clip.source must be 'generated' or 'imported', got {self.source!r}")


@dataclass
class Document:
    """The whole project's source of truth (Q15's closing principle). Owns
    the canonical text plus the clip/track/character metadata layered on top
    of it."""

    text: str = ""
    clips: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    characters: list = field(default_factory=list)
    settings: dict = field(default_factory=dict)

    # -- lookups -----------------------------------------------------------

    def get_character(self, character_id: Optional[str]) -> Optional[Character]:
        if character_id is None:
            return None
        return next((c for c in self.characters if c.id == character_id), None)

    def get_track(self, track_id: Optional[str]) -> Optional[Track]:
        if track_id is None:
            return None
        return next((t for t in self.tracks if t.id == track_id), None)

    def get_clip(self, clip_id: str) -> Optional[Clip]:
        return next((c for c in self.clips if c.id == clip_id), None)

    def clip_covering(self, position: int) -> Optional[Clip]:
        """The `Clip` containing text offset `position`, if any (inclusive
        start, exclusive end - consistent with `start_offset`/`end_offset`
        slicing elsewhere in this class)."""
        return next((c for c in self.clips if c.start_offset <= position < c.end_offset), None)

    # -- text/config -------------------------------------------------------

    def clip_text(self, clip: Clip) -> str:
        """The clip's current slice of the canonical document text."""
        return self.text[clip.start_offset:clip.end_offset]

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

        return [
            clip
            for clip in self.clips
            if is_clip_dirty(clip, self.clip_text(clip), self.effective_config_for_clip(clip))
        ]

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
        the same rule `apply_text_change`'s fully-consumed-clip removal
        already establishes.

        No manual dirty-marking is needed - every clip this method touches
        ends up with no `segments`, which `dirty.is_clip_dirty` already
        treats as dirty.
        """
        if end <= start:
            raise ValueError(f"assign_character_to_range requires end > start, got start={start}, end={end}")

        track_id = next((t.id for t in self.tracks if t.character_id == character_id), None)

        overlapping = [c for c in self.clips if c.start_offset < end and c.end_offset > start]
        leftovers = []
        for clip in overlapping:
            if clip.start_offset < start:
                leftovers.append(Clip(
                    start_offset=clip.start_offset, end_offset=start,
                    character_id=clip.character_id, track_id=clip.track_id,
                    overrides=dict(clip.overrides), fx_override=clip.fx_override,
                    source=clip.source, original_audio_path=clip.original_audio_path,
                ))
            if clip.end_offset > end:
                leftovers.append(Clip(
                    start_offset=end, end_offset=clip.end_offset,
                    character_id=clip.character_id, track_id=clip.track_id,
                    overrides=dict(clip.overrides), fx_override=clip.fx_override,
                    source=clip.source, original_audio_path=clip.original_audio_path,
                ))

        for clip in overlapping:
            self.clips.remove(clip)
        self.clips.extend(leftovers)

        new_clip = Clip(start_offset=start, end_offset=end, character_id=character_id, track_id=track_id)
        self.clips.append(new_clip)
        return new_clip

    # -- incremental offset maintenance (Q16/Q18 dirty-tracking mechanism) --

    def apply_text_change(self, position: int, chars_removed: int, chars_added: int, new_text: str) -> list:
        """Applies one `QTextDocument.contentsChange`-shaped edit
        (position/charsRemoved/charsAdded, plus the resulting full text -
        Qt's signal doesn't carry the inserted characters themselves, so the
        caller passes `editor.toPlainText()` after the change) and
        incrementally shifts/extends every `Clip`'s offsets - rather than
        reconstructing clip boundaries after the fact via text diffing.

        An edit strictly inside a clip's range extends/shrinks that same
        `Clip` object (now stale/dirty by construction, since its text no
        longer matches what its segments were generated from - see
        `dirty.is_clip_dirty`). An edit whose removed range fully contains a
        clip's range removes the `Clip` object outright, so a later,
        textually-identical retype creates a brand-new `Clip` with a fresh id
        (Q18) even though its audio may still cache-hit. This one incremental
        rule resolves both Q16 ("propagate metadata across edits, kinda like
        git") and Q18 without an actual content-diffing algorithm.

        Returns the list of `Clip`s removed by this edit, for callers that
        need to react (e.g. dropping them from a track view).
        """
        removed_end = position + chars_removed
        delta = chars_added - chars_removed

        def map_offset(x: int) -> int:
            if x <= position:
                return x
            if x >= removed_end:
                return x + delta
            # An offset that fell strictly inside the replaced range has no
            # single well-defined mapping - collapse it to the edit's start,
            # which is what makes a fully-consumed clip's start >= its
            # (mapped) end below.
            return position

        removed_clips = []
        for clip in list(self.clips):
            old_start, old_end = clip.start_offset, clip.end_offset
            fully_consumed = chars_removed > 0 and position <= old_start and removed_end >= old_end
            if fully_consumed:
                self.clips.remove(clip)
                removed_clips.append(clip)
                continue
            clip.start_offset = map_offset(old_start)
            clip.end_offset = max(map_offset(old_end), clip.start_offset)

        self.text = new_text
        return removed_clips
