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

import copy
import math
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
    # The global character library entry this record is a live-linked view
    # of (grill WF12, kokoro_gui/daw/library.py), or None for a character
    # local to this project. The record stays inlined either way, so a
    # project opens where the library entry doesn't exist.
    library_id: Optional[str] = None
    id: str = field(default_factory=_new_id)
    # Fields this version doesn't know, carried through a load/save so an
    # older KokoroGUI doesn't strip what a newer one wrote (see
    # serialization.py). Nothing in the app reads it. For a Character it
    # also holds `extra["preset_data"]`: the preset keys ALLOWED_PRESET_KEYS
    # strips from what reaches a config dict.
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_preset_dict(cls, name, preset_data, highlight_color=None, backend_id="kokoro", id=None,
                         library_id=None):
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
        if library_id is not None:
            kwargs["library_id"] = library_id
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
    # 1-based lane number when this is one of the unified layout's "Lane N"
    # tracks (kokoro_gui/daw/lanes.py), else None. Such a track has no
    # character; clips of any character land on it by the lane rule.
    lane: Optional[int] = None
    # "subprojects" for the track nested clips land on by default (phase 4;
    # `Document.subprojects_track`), "music" for the one music beds land on
    # (phase 5 P2; `Document.music_track`), else None.
    role: Optional[str] = None
    # Ducked under speech: every clip on this track is turned down by the
    # mixer's sidechain while the other clips play (kokoro_gui/audio/mixer.py,
    # `DuckState`; depth from `Document.settings["duck_db"]`).
    duck: bool = False
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
    # `[start_s, end_s]` seconds into `audio_path`: the segment is that
    # slice of the file (an imported recording's words, a trimmed bed).
    # None is the whole file. Saved only when set.
    range: Optional[list] = None
    id: str = field(default_factory=_new_id)
    extra: dict = field(default_factory=dict)  # unknown fields, see Character


CLIP_STATUSES = ("todo", "generated", "approved", "needs_rewrite")
# "nested": a subproject placed as a clip (phase 4, grill NP1-NP8). Its
# audio is the child project's mixdown; it has no segments or takes.
# "imported": audio that came from a file, never TTS. A music bed (phase 5
# P2, grill Q30) is one with `original_audio_path` set.
CLIP_SOURCES = ("generated", "imported", "nested")
# `Run.kind` of the one read-only run a nested clip or a music bed owns: the
# child's title, or the bed's file name.
PLACEHOLDER = "placeholder"


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
    source: str = "generated"  # one of CLIP_SOURCES
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
    # Locked in time: ripple on regenerate never moves it (subtitle cues set
    # it). A drag still does. Not the same as having a `timeline_timestamp`.
    pinned: bool = False
    # A nested clip's subproject: `{"kind": "embedded", "id": project_id}`
    # (the child bundle rides inside the parent at `projects/<id>.tbaw`) or
    # `{"kind": "linked", "id": project_id, "path": relative_or_absolute}`
    # (a `.tbaw` on disk). `id` is the child's manifest `project_id`, so a
    # relink can tell it found the right file. None for any other clip.
    child: Optional[dict] = None
    id: str = field(default_factory=_new_id)
    extra: dict = field(default_factory=dict)  # unknown fields, see Character

    def __post_init__(self):
        if self.source not in CLIP_SOURCES:
            raise ValueError(f"Clip.source must be one of {CLIP_SOURCES}, got {self.source!r}")

    @property
    def is_nested(self) -> bool:
        return self.source == "nested"

    @property
    def is_bed(self) -> bool:
        """A music bed (grill Q30): an imported clip that plays
        `original_audio_path` and owns one placeholder run holding the file
        name. An imported recording (phase 5 P3) keeps its timing on its
        runs instead and has no `original_audio_path`, so it is not a bed."""
        return self.source == "imported" and bool(self.original_audio_path)

    @property
    def has_placeholder(self) -> bool:
        """True for the clips whose one run is a read-only placeholder: a
        nested clip or a music bed."""
        return self.is_nested or self.is_bed

    @property
    def run_kind(self) -> str:
        """The `Run.kind` of this clip's runs: its source, or
        `PLACEHOLDER` for a nested clip or a music bed."""
        return PLACEHOLDER if self.has_placeholder else self.source


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
    tagged run, or is `None` for an untagged one. `"placeholder"`
    (`PLACEHOLDER`) is a nested clip's one read-only run, whose text is the
    subproject's title (phase 4), or a music bed's, whose text is its file
    name (phase 5 P2); the editor refuses edits inside it.

    `words` is the timing of an imported recording's text (phase 5 P3,
    grill Q32): one `[char_start, char_end, source, start_s, end_s]` entry
    per word, char offsets relative to this run's text, `source` a key of
    `Document.sources`, times in seconds into that file. Only a run of
    `kind == "imported"` carries any; whitespace and punctuation between
    words carry nothing. Characters no word covers are untimed text.
    """

    text: str = ""
    clip_id: Optional[str] = None
    kind: Optional[str] = None
    words: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)  # unknown fields, see Character


# `Run.kind` (and `Clip.source`) of an imported recording's text.
IMPORTED = "imported"
# The `Document.settings` key holding `Document.sources`.
SOURCES_KEY = "sources"


def clean_words(words, length: int) -> list:
    """`words` as `[char_start, char_end, source, start_s, end_s]` lists
    sorted by `char_start`, keeping only entries whose span lies inside
    `[0, length)` and whose times are numbers with `end_s > start_s`. What
    `apply_words` accepts from a paste (untrusted mime data)."""
    out = []
    for word in words or []:
        try:
            char_start, char_end = int(word[0]), int(word[1])
            source = str(word[2])
            start_s, end_s = float(word[3]), float(word[4])
        except (TypeError, ValueError, IndexError):
            continue
        if not (0 <= char_start < char_end <= length) or not source:
            continue
        if not (math.isfinite(start_s) and math.isfinite(end_s)) or end_s <= start_s or start_s < 0:
            continue
        out.append([char_start, char_end, source, start_s, end_s])
    out.sort(key=lambda w: w[0])
    return out


def _shift_words(words, delta: int) -> list:
    return [[w[0] + delta, w[1] + delta, *w[2:]] for w in words]


def _split_words(words, cut: int) -> tuple:
    """`(left, right)` for a run split at `cut`: a word goes with the side
    holding its first character (clipped to it), so a split never
    duplicates or loses a word's audio."""
    left, right = [], []
    for word in words:
        if word[0] < cut:
            left.append([word[0], min(word[1], cut), *word[2:]])
        else:
            right.append([word[0] - cut, word[1] - cut, *word[2:]])
    return left, right


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
    # `_switch_document`) as a closure over each clip's backend and the
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
    # Runtime-only, set beside `segment_key_fn`: `clip -> bool`, True when a
    # nested clip's subproject is stale (a stale clip inside it, or its
    # mixdown missing or older than its document). Unset, a nested clip
    # counts as stale. Never serialized.
    nested_state_fn: Optional[Callable] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.undo_stack = UndoStack(self)
        # In the unified track layout, an edit that changes clip order or a
        # clip's character re-runs the lane rule in the same undo step.
        from kokoro_gui.daw.lanes import relane_follow_up

        self.undo_stack.follow_up = relane_follow_up

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

    # -- imported recordings (phase 5 P3) -----------------------------------

    @property
    def sources(self) -> dict:
        """`source -> {"path", "sample_rate", "duration_s"}` for every
        imported recording a run's words name, stored as
        `settings["sources"]` so it rides the settings round trip. `path`
        is absolute in memory (bundle-relative in a `.tbaw`, like a
        segment's `audio_path`) and None when the file is missing. An empty
        dict, not stored, when the document has none; add entries with
        `add_sources`."""
        value = (self.settings or {}).get(SOURCES_KEY)
        return value if isinstance(value, dict) else {}

    def add_sources(self, entries: dict) -> list:
        """Adds each `source -> entry` the document doesn't already have
        (a source's name is its content hash, so a known one is the same
        file). A known source with no file (missing on open) takes the new
        entry's, which relinks it. An entry without a `path` string is
        skipped. Returns the names added or relinked."""
        added = []
        for source, entry in (entries or {}).items():
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not entry["path"]:
                continue
            if source in self.sources and self.source_path(source) is not None:
                continue
            self.settings.setdefault(SOURCES_KEY, {})[str(source)] = {
                "path": entry["path"],
                "sample_rate": entry.get("sample_rate"),
                "duration_s": entry.get("duration_s"),
            }
            added.append(str(source))
        return added

    def source_path(self, source: str) -> Optional[str]:
        """The file `source` names, or None (unknown, or missing on open)."""
        entry = self.sources.get(source)
        path = entry.get("path") if isinstance(entry, dict) else None
        return path if isinstance(path, str) and path else None

    def refresh_imported_segments(self, clip_ids=None) -> None:
        """Rebuilds `clip.segments` of each imported recording clip (all of
        them, or those in `clip_ids`) from its runs' words
        (`imported.segments_for`). The segments are a cache every reader of
        `clip.segments` uses unchanged; a clip whose derived segments equal
        what it has keeps its list. Every edit that touches such a clip's
        runs calls this."""
        from kokoro_gui.daw.imported import is_recording_clip, same_segments, segments_by_clip

        targets = [c for c in self.clips if (clip_ids is None or c.id in clip_ids) and is_recording_clip(c)]
        if not targets:
            return
        derived = segments_by_clip(self, [c.id for c in targets])
        for clip in targets:
            if not same_segments(clip.segments, derived[clip.id]):
                clip.segments = derived[clip.id]

    def _recording_clip_of(self, run: Optional[Run]) -> Optional[Clip]:
        """`run`'s clip when it is an imported recording clip, else None."""
        from kokoro_gui.daw.imported import is_recording_clip

        if run is None or run.clip_id is None or run.kind != IMPORTED:
            return None
        clip = self.get_clip(run.clip_id)
        return clip if clip is not None and is_recording_clip(clip) else None

    def _split_clip_at(self, clip: Clip, offset: int) -> Clip:
        """Moves `clip`'s runs at or after text offset `offset` to a new
        imported clip (fresh id, same character, track, overrides and FX),
        inserted after `clip` in `clips`, and returns it. The new clip
        follows the old one with no added silence (`gap_before_s = 0`) and
        takes its fade-out, so the recording plays as it did. Segments are
        left for `refresh_imported_segments`."""
        self._split_at(offset)
        new_clip = Clip(
            character_id=clip.character_id, track_id=clip.track_id,
            overrides=copy.deepcopy(clip.overrides), fx_override=copy.deepcopy(clip.fx_override),
            source=clip.source, gap_before_s=0.0, fade_out_s=clip.fade_out_s, status=clip.status,
        )
        clip.fade_out_s = 0.0
        for run, start, _end in self._iter_runs_with_offsets():
            if run.clip_id == clip.id and start >= offset:
                run.clip_id = new_clip.id
        self.clips.insert(self.clips.index(clip) + 1, new_clip)
        return new_clip

    @staticmethod
    def _continues(left: Clip, right: Clip) -> bool:
        """True when imported clip `right` reads as the rest of `left`: it
        follows with no added silence and isn't placed on its own, with the
        same character and track (what `_split_clip_at` leaves)."""
        return (right.gap_before_s == 0.0 and right.timeline_timestamp is None
                and right.character_id == left.character_id and right.track_id == left.track_id)

    def edit_touches_imported(self, position: int, chars_removed: int) -> bool:
        """True when a `replace_text` of `[position, position +
        chars_removed)` changes imported recording text: a delete overlapping
        a recording clip's runs, a delete of everything between two halves
        of a split clip (`replace_text` joins them), or an insert strictly
        inside one (which splits it). Qt's native undo replays only
        characters, so it can't give dropped words back or split a joined
        clip again; the editor can send such an edit through
        `TextEditCommand` instead, whose undo restores the runs."""
        removed_end = position + chars_removed
        if chars_removed > 0:
            if any(self._recording_clip_of(run) is not None
                   for run, r_start, r_end in self._iter_runs_with_offsets()
                   if r_start < removed_end and r_end > position):
                return True
            if position <= 0 or removed_end >= len(self.text):
                return False
            left = self._recording_clip_of(self._run_covering(position - 1))
            right = self._recording_clip_of(self._run_covering(removed_end))
            return left is not None and right is not None and left is not right and self._continues(left, right)
        left = self._recording_clip_of(self._run_covering(position - 1)) if position > 0 else None
        right = self._run_covering(position)
        return left is not None and right is not None and right.clip_id == left.id

    def _merge_clip_into(self, target: Clip, other: Clip) -> None:
        """Retags `other`'s runs with `target` and drops `other`, which
        takes over its fade-out. For two imported halves that are adjacent
        again."""
        for run in self.runs:
            if run.clip_id == other.id:
                run.clip_id = target.id
        target.fade_out_s = other.fade_out_s
        self.clips = [c for c in self.clips if c.id != other.id]

    def _last_word_source(self, clip_id: str, last: bool = True) -> Optional[str]:
        """The source of the clip's last (or, `last=False`, first) word in
        text order, or None when it has no words."""
        found = None
        for run in self.runs:
            if run.clip_id == clip_id and run.words:
                if not last:
                    return run.words[0][2]
                found = run.words[-1][2]
        return found

    def track_layout(self) -> dict:
        """`settings["track_layout"]` normalised: `{"mode": "character"}`
        (one track per used character, the default) or `{"mode":
        "unified", "lanes": N}` (grill PR4, kokoro_gui/daw/lanes.py)."""
        raw = (self.settings or {}).get("track_layout")
        if isinstance(raw, dict) and raw.get("mode") == "unified":
            try:
                lanes = int(raw.get("lanes", 3))
            except (TypeError, ValueError):
                lanes = 3
            return {"mode": "unified", "lanes": max(1, min(lanes, 16))}
        return {"mode": "character"}

    def track_for_character(self, character_id: Optional[str], create: bool = False) -> Optional[str]:
        """The id of `character_id`'s track. With `create`, a character
        that has none gets one, appended and named after it (grill PR4: a
        track exists only once its character is used). Never creates in
        the unified layout, where lanes are assigned after the edit, or
        for an unknown character."""
        track = next((t for t in self.tracks if t.character_id == character_id), None) if character_id else None
        if track is not None:
            return track.id
        if not create or self.track_layout()["mode"] == "unified":
            return None
        character = self.get_character(character_id)
        if character is None:
            return None
        order = max((t.order_index for t in self.tracks), default=-1) + 1
        track = Track(name=character.name, character_id=character.id, order_index=order)
        self.tracks.append(track)
        return track.id

    def subprojects_track(self, create: bool = False) -> Optional[str]:
        """The id of the "Subprojects" track nested clips go on, made on
        first use like a character's track."""
        track = next((t for t in self.tracks if t.role == "subprojects"), None)
        if track is None and create:
            order = max((t.order_index for t in self.tracks), default=-1) + 1
            track = Track(name="Subprojects", order_index=order, role="subprojects")
            self.tracks.append(track)
        return track.id if track is not None else None

    def music_track(self) -> Optional[str]:
        """The id of the "Music" track music beds go on (phase 5 P2), or
        None before the first bed (`undo.ImportBedCommand` makes it)."""
        track = next((t for t in self.tracks if t.role == "music"), None)
        return track.id if track is not None else None

    def used_tracks(self) -> list:
        """Tracks at least one clip sits on, in `order_index` order: the
        lanes the timeline draws. A track whose clips are all gone stays in
        `tracks` with its mixer settings and comes back when used again."""
        used = {clip.track_id for clip in self.clips if clip.track_id is not None}
        return sorted((t for t in self.tracks if t.id in used), key=lambda t: t.order_index)

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
        out = []
        for clip in self.clips:
            if clip.is_nested:
                stale = self.nested_state_fn(clip) if self.nested_state_fn is not None else True
            else:
                stale = is_clip_dirty(clip, self.clip_text(clip), config_for(clip), key_fn=self.segment_key_fn)
            if stale:
                out.append(clip)
        return out

    def overlaps_nested(self, start: int, end: int) -> bool:
        """True when `[start, end)` touches a placeholder run (a nested
        clip's or a music bed's), which no character assignment or split
        may retag."""
        for run, r_start, r_end in self._iter_runs_with_offsets():
            if run.kind == PLACEHOLDER and r_start < end and r_end > start:
                return True
        return False

    def nested_clips(self) -> list:
        return [clip for clip in self.clips if clip.is_nested]

    def placeholder_extent(self, clip_id: str) -> Optional[tuple]:
        """`clip_extent` of a nested clip's or a music bed's placeholder
        run, or None."""
        clip = self.get_clip(clip_id)
        return self.clip_extent(clip_id) if clip is not None and clip.has_placeholder else None

    def insert_nested_clip(self, position: int, child: dict, title: str, character_id=None) -> Clip:
        """Inserts a nested clip at text offset `position`: one placeholder
        run holding `title`. An insert inside another clip's run lands at
        that run's end instead, so no clip is split. Returns the clip; no
        track is assigned (the caller puts it on one)."""
        position = max(0, min(int(position), len(self.text)))
        run = self._run_covering(position - 1) if position > 0 else None
        if run is not None and run.clip_id is not None:
            extent = self.clip_extent(run.clip_id)
            if extent is not None and extent[0] < position < extent[1]:
                position = extent[1]
        clip = Clip(character_id=character_id, source="nested", child=dict(child))
        self._split_at(position)
        new_runs = []
        inserted = False
        pos = 0
        for existing in self.runs:
            if not inserted and pos >= position:
                new_runs.append(Run(text=title or "Subproject", clip_id=clip.id, kind=PLACEHOLDER))
                inserted = True
            new_runs.append(existing)
            pos += len(existing.text)
        if not inserted:
            new_runs.append(Run(text=title or "Subproject", clip_id=clip.id, kind=PLACEHOLDER))
        self.runs = new_runs
        self.clips.append(clip)
        self._normalize_runs()
        return clip

    def set_placeholder_text(self, clip_id: str, text: str) -> None:
        """Renames a nested clip's placeholder run (the subproject's title
        changed)."""
        for run in self.runs:
            if run.clip_id == clip_id and run.kind == PLACEHOLDER:
                run.text = text or "Subproject"
                return

    # -- run-list maintenance (private) -------------------------------------

    def _split_at(self, offset: int) -> None:
        """Splits whichever run straddles `offset` into two runs at that
        boundary, so later code can retag/replace an exact `[start, end)`
        span without disturbing text on either side of it. A no-op if
        `offset` already falls on a run boundary (including the document's
        own start/end). An imported run's words split with it
        (`_split_words`)."""
        if offset <= 0 or offset >= len(self.text):
            return
        pos = 0
        for i, run in enumerate(self.runs):
            end = pos + len(run.text)
            if pos < offset < end:
                cut = offset - pos
                left_words, right_words = _split_words(run.words, cut)
                self.runs[i:i + 1] = [
                    Run(text=run.text[:cut], clip_id=run.clip_id, kind=run.kind, words=left_words),
                    Run(text=run.text[cut:], clip_id=run.clip_id, kind=run.kind, words=right_words),
                ]
                return
            pos = end

    def _normalize_runs(self) -> None:
        """Drops zero-length runs and merges adjacent runs sharing the same
        `clip_id`/`kind` - the run-list equivalent of Qt's own "typing
        inside a run just extends it" merge behavior, kept true here too so
        two operations that happen to retag neighboring spans identically
        don't leave a meaningless split between them. Merged runs keep
        their words, shifted; a run that isn't imported keeps none."""
        merged: list = []
        for run in self.runs:
            if not run.text:
                continue
            if run.words and run.kind != IMPORTED:
                run.words = []
            if merged and merged[-1].clip_id == run.clip_id and merged[-1].kind == run.kind:
                previous = merged[-1]
                merged[-1] = Run(text=previous.text + run.text, clip_id=run.clip_id, kind=run.kind,
                                 words=previous.words + _shift_words(run.words, len(previous.text)))
            else:
                merged.append(run)
        self.runs = merged

    def _retag_range(self, start: int, end: int, clip_id: Optional[str], kind: Optional[str],
                     words: Optional[list] = None) -> None:
        """Replaces whatever runs currently occupy `[start, end)` with a
        single run of that same text, tagged `clip_id`/`kind` - the shared
        "retag an exact span" primitive `assign_character_to_range` below
        applies once per clip it touches (the new clip's span, plus one per
        leftover fragment). Never changes `len(self.text)`. The new run
        carries `words` (relative to `start`) when given, else the words
        the replaced runs had, when `kind` is imported."""
        self._split_at(start)
        self._split_at(end)

        new_runs: list = []
        merged_text_parts: list = []
        merged_words: list = []
        inserted_at: Optional[int] = None
        pos = 0
        for run in self.runs:
            run_end = pos + len(run.text)
            if run_end <= start or pos >= end:
                new_runs.append(run)
            else:
                merged_words.extend(_shift_words(run.words, pos - start))
                merged_text_parts.append(run.text)
                if inserted_at is None:
                    inserted_at = len(new_runs)
                    new_runs.append(None)
            pos = run_end

        if words is not None:
            merged_words = [list(w) for w in words]
        elif kind != IMPORTED:
            merged_words = []
        new_runs[inserted_at] = Run(text="".join(merged_text_parts), clip_id=clip_id, kind=kind,
                                    words=merged_words)
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

        Imported recording text in the range (phase 5 P3, grill Q32) is
        different: it keeps its clip, words and audio, and only its label
        changes. An imported clip wholly inside the range gets the new
        character (and its track); one partly inside is split so the part
        inside becomes its own imported clip with the new character. The
        rest of the range goes through the rule above, one new clip per
        stretch between imported text (a whitespace-only stretch is left
        alone). Returns the first new generated clip, or the first
        relabeled imported clip when the range held nothing else.
        """
        if end <= start:
            raise ValueError(f"assign_character_to_range requires end > start, got start={start}, end={end}")
        text_len = len(self.text)
        if start < 0 or end > text_len:
            raise ValueError(
                f"assign_character_to_range requires [start, end) within [0, {text_len}), "
                f"got start={start}, end={end}"
            )
        if self.overlaps_nested(start, end):
            raise ValueError("assign_character_to_range can't retag a placeholder run")

        imported = self._imported_spans(start, end)
        if not imported:
            return self._assign_generated(start, end, character_id)

        relabeled = self._relabel_imported(start, end, character_id)
        text = self.text
        first_new = None
        cursor = start
        for span_start, span_end in imported + [(end, end)]:
            if span_start > cursor and text[cursor:span_start].strip():
                clip = self._assign_generated(cursor, span_start, character_id)
                first_new = first_new or clip
            cursor = max(cursor, span_end)
        return first_new or relabeled[0]

    def _imported_spans(self, start: int, end: int) -> list:
        """The maximal `(start, end)` stretches of `[start, end)` covered by
        imported recording clips' runs, in order."""
        spans: list = []
        for run, r_start, r_end in self._iter_runs_with_offsets():
            lo, hi = max(start, r_start), min(end, r_end)
            if hi <= lo or self._recording_clip_of(run) is None:
                continue
            if spans and spans[-1][1] == lo:
                spans[-1] = (spans[-1][0], hi)
            else:
                spans.append((lo, hi))
        return spans

    def _relabel_imported(self, start: int, end: int, character_id: Optional[str]) -> list:
        """Gives the imported recording text inside `[start, end)` the
        character `character_id` without touching its runs' text, words or
        kind (see `assign_character_to_range`). Returns the relabeled clips
        in text order."""
        track_id = self.track_for_character(character_id, create=True)
        ids: list = []
        for run, r_start, r_end in self._iter_runs_with_offsets():
            if r_start < end and r_end > start and self._recording_clip_of(run) is not None:
                if run.clip_id not in ids:
                    ids.append(run.clip_id)
        relabeled = []
        for clip_id in ids:
            clip = self.get_clip(clip_id)
            c_start, c_end = self.clip_extent(clip_id)
            lo, hi = max(start, c_start), min(end, c_end)
            if hi < c_end:
                self._split_clip_at(clip, hi)
            target = self._split_clip_at(clip, lo) if lo > c_start else clip
            target.character_id = character_id
            target.track_id = track_id
            relabeled.append(target)
        self._normalize_runs()
        self.refresh_imported_segments()
        return relabeled

    def _assign_generated(self, start: int, end: int, character_id: Optional[str], create: bool = True):
        """The split-or-create rule of `assign_character_to_range` for
        `[start, end)`. With `create=False` no new clip is made and the
        span is left untagged (what `apply_words` needs before it tags a
        paste); returns None then."""
        track_id = self.track_for_character(character_id, create=True) if create else None

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

        new_clip = Clip(character_id=character_id, track_id=track_id) if create else None

        self.clips = [c for c in self.clips if c.id not in overlapping_ids]
        self.clips.extend(leftover_clips)
        if new_clip is not None:
            self.clips.append(new_clip)

        # None of these _retag_range calls change len(self.text), so it's
        # safe to apply them in any order using offsets all computed above,
        # against the pre-edit run layout.
        if new_clip is not None:
            self._retag_range(start, end, new_clip.id, new_clip.run_kind)
        else:
            self._retag_range(start, end, None, None)
        for (l_start, l_end, _old_clip), leftover_clip in zip(leftover_ranges, leftover_clips):
            self._retag_range(l_start, l_end, leftover_clip.id, leftover_clip.run_kind)
        if any(c.source == IMPORTED for c in leftover_clips):
            self.refresh_imported_segments({c.id for c in leftover_clips})

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

        Imported recording text (phase 5 P3, grill Q32) follows two more
        rules. Delete: a word whose span the removed range touches loses its
        timing entry whole (its audio goes with it; any of its characters
        left behind stay as untimed text), and the rest keep theirs. Type:
        inserted text never joins an imported run; it becomes an untagged
        run, and when it lands strictly inside an imported clip (the text on
        both sides belongs to it) the clip splits there, the part after the
        insertion becoming a new imported clip (`_split_clip_at`). A word
        the insertion point falls inside loses its timing too. Typing at
        either edge of an imported clip doesn't split it. The touched
        clips' segments are rebuilt (`refresh_imported_segments`).

        Returns the list of `Clip`s removed by this edit, for callers that
        need to react (e.g. dropping them from a track view).
        """
        removed_end = position + chars_removed
        self._drop_touched_words(position, removed_end, chars_added > 0)

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
        # Text typed after a subproject's placeholder line is never part of
        # it: the placeholder holds the child's title and nothing else.
        if inherited_kind == PLACEHOLDER or inherited_clip_id in fully_consumed_ids:
            inherited_clip_id = None
            inherited_kind = None
        # Typed text has no timing, so it never joins an imported recording
        # clip; inside one, the clip splits around it.
        split_clip = None
        recording = self._recording_clip_of(inherited) if inherited_clip_id is not None else None
        if recording is not None:
            if chars_added > 0 and removed_end < len(self.text):
                right = self._run_covering(removed_end)
                if right is not None and right.clip_id == recording.id:
                    split_clip = recording
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
        touched = overlapping_ids - fully_consumed_ids
        if split_clip is not None:
            touched.add(split_clip.id)
            touched.add(self._split_clip_at(split_clip, position + chars_added).id)
        elif chars_removed > 0 and chars_added == 0 and 0 < position < len(self.text):
            # Deleting what was typed between two halves of a split clip
            # (Qt's native undo of the typing does exactly this) joins them.
            left = self._recording_clip_of(self._run_covering(position - 1))
            right = self._recording_clip_of(self._run_covering(position))
            if left is not None and right is not None and left is not right and self._continues(left, right):
                self._merge_clip_into(left, right)
                touched.add(left.id)
        self._normalize_runs()
        self.refresh_imported_segments(touched)
        return removed_clips

    def apply_words(self, position: int, length: int, words, sources=None,
                    character_id: Optional[str] = None) -> Optional[str]:
        """Tags `[position, position + length)`, text already inserted (a
        paste or drop of timed text), as imported recording text carrying
        `words` (`Run.words` entries, char offsets relative to `position`).
        `sources` entries the document lacks, or has no file for, are added
        first (`add_sources`); a word whose source still has no file is
        dropped.
        With no word left, nothing changes and None is returned: the span
        stays as it is, untimed (grill Q32).

        The span is cut out of any clip it sits in (the `assign` rule with
        no new clip), then joins the imported clip it touches when the
        sources meet: the clip ending at `position` whose last word has the
        first pasted word's source, else the clip starting right after the
        span whose first word has the last pasted word's source. When both
        sides are such clips and the right one reads as the rest of the
        left (`_continues`: two halves of one split clip), it merges into
        the left, so pasting a cut word back where it was leaves one clip. Otherwise a new imported clip is made with `character_id`, on
        that character's track. Returns the id of the clip the span belongs
        to."""
        end = position + length
        if length <= 0 or position < 0 or end > len(self.text):
            raise ValueError(f"apply_words requires a span inside the text, got {position}+{length}")
        if self.overlaps_nested(position, end):
            raise ValueError("apply_words can't retag a subproject's placeholder")
        added = self.add_sources(sources or {})
        cleaned = [w for w in clean_words(words, length) if self.source_path(w[2])]
        if not cleaned:
            return None

        if any(run.clip_id is not None for run, r_start, r_end in self._iter_runs_with_offsets()
               if r_start < end and r_end > position):
            self._assign_generated(position, end, None, create=False)

        left = self._recording_clip_of(self._run_covering(position - 1)) if position > 0 else None
        right = self._recording_clip_of(self._run_covering(end))
        if left is not None and self._last_word_source(left.id) != cleaned[0][2]:
            left = None
        if right is not None and self._last_word_source(right.id, last=False) != cleaned[-1][2]:
            right = None

        if left is not None:
            target = left
            if right is not None and right.id != left.id and self._continues(left, right):
                self._merge_clip_into(left, right)
        elif right is not None:
            target = right
        else:
            target = Clip(character_id=character_id, track_id=self.track_for_character(character_id, create=True),
                          source=IMPORTED)
            self.clips.append(target)
        self._retag_range(position, end, target.id, IMPORTED, words=cleaned)
        # A relinked source gives other clips' words their file back too.
        self.refresh_imported_segments(None if added else {target.id})
        return target.id

    def _drop_touched_words(self, position: int, removed_end: int, inserting: bool) -> None:
        """The whole-word rule of `replace_text`: drops every word entry
        whose span overlaps `[position, removed_end)`, or, for a pure
        insertion, whose span has `position` strictly inside it."""
        for run, r_start, _r_end in self._iter_runs_with_offsets():
            if not run.words:
                continue
            if removed_end > position:
                kept = [w for w in run.words if not (r_start + w[0] < removed_end and r_start + w[1] > position)]
            elif inserting:
                kept = [w for w in run.words if not (r_start + w[0] < position < r_start + w[1])]
            else:
                continue
            if len(kept) != len(run.words):
                run.words = kept
