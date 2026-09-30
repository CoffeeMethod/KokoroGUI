"""Where a clip's segments and lexicon rewrites sit in the transcript, for
the transcript's details overlays (Options > Transcript details).

A clip is generated as the pieces `segmenting.split_text` cuts from its
text after the lexicon (`dirty.spoken_text`). For a clean clip those pieces
are the stored `Segment.text`s (that is what clean means), so the same
prediction is right for every clip, stale or not. `clip_pieces` gives each
piece as document offsets: `segmenting.split_spans` on the spoken text, each
span mapped back through the lexicon (`lexicon.original_span`), plus the
clip's start. Like `QtTTSApp._word_offsets` it takes the clip's runs to be
contiguous.

Pure, no Qt. Results are memoized on the clip's text, the segmentation keys
and the lexicon, so a keystroke re-splits one clip.
"""
from collections import OrderedDict
from dataclasses import dataclass

from kokoro_gui.daw import arrangement
from kokoro_gui.engine import segmenting
from kokoro_gui.engine.lexicon import apply_lexicon, lexicon_signature, original_span

_MEMO_SIZE = 4096
_memo: "OrderedDict[tuple, tuple]" = OrderedDict()
_lexicon_patterns: dict = {}


@dataclass(frozen=True)
class PieceSpan:
    """One segment of a clip: `index` matches `Segment.order_index`,
    `start`/`end` are offsets into the clip's text (add the clip's start for
    document offsets), `level` is the `segmenting` boundary level the piece
    ends at and `words` its word count."""
    index: int
    start: int
    end: int
    level: int
    words: int


def has_pieces(clip) -> bool:
    """Whether the clip is generated from its text: not a subproject, a
    music bed or an imported recording."""
    return clip is not None and not clip.has_placeholder and clip.source != "imported"


def _config_key(config: dict) -> tuple:
    return (tuple(config.get(key, True) for key in segmenting.BOUNDARY_KEYS.values()),
            segmenting.target_words(config), lexicon_signature(config.get("lexicon")))


def _analyse(text: str, config: dict) -> tuple:
    """`(pieces, rewrites)` for a clip's `text`, relative to it: the
    `PieceSpan`s and `(start, end, spoken)` per lexicon rewrite. Memoized."""
    key = (text, _config_key(config))
    hit = _memo.get(key)
    if hit is not None:
        _memo.move_to_end(key)
        return hit
    spoken, spans = apply_lexicon(text, config.get("lexicon") or {}, _lexicon_patterns, with_spans=True)
    pieces = []
    for index, (start, end, level) in enumerate(segmenting.split_spans(spoken, config)):
        o_start, o_end = original_span(spans, start, end)
        pieces.append(PieceSpan(index, o_start, o_end, level, len(spoken[start:end].split())))
    rewrites = [(o_start, o_end, spoken[n_start:n_end]) for o_start, o_end, n_start, n_end in spans
                if text[o_start:o_end] != spoken[n_start:n_end]]
    result = (tuple(pieces), tuple(rewrites))
    _memo[key] = result
    if len(_memo) > _MEMO_SIZE:
        _memo.popitem(last=False)
    return result


def _config_for(document, clip, config):
    if config is not None:
        return config
    config_fn = getattr(document, "generation_config_fn", None) or document.effective_config_for_clip
    return config_fn(clip)


def clip_pieces(document, clip, config=None) -> list:
    """The clip's `PieceSpan`s with document offsets, `[]` for a clip that
    isn't generated from its text or isn't in the text. `config` defaults
    to the document's per-clip generation config (what the dirty check
    reads)."""
    if not has_pieces(clip):
        return []
    extent = document.clip_extent(clip.id)
    if extent is None:
        return []
    pieces, _rewrites = _analyse(document.clip_text(clip), _config_for(document, clip, config))
    base = extent[0]
    return [PieceSpan(p.index, p.start + base, p.end + base, p.level, p.words) for p in pieces]


def lexicon_rewrites(document, clip, config=None) -> list:
    """`(start, end, spoken)` document offsets of each stretch of the
    clip's text the lexicon rewrites before synthesis, and what it becomes."""
    if not has_pieces(clip):
        return []
    extent = document.clip_extent(clip.id)
    if extent is None:
        return []
    _pieces, rewrites = _analyse(document.clip_text(clip), _config_for(document, clip, config))
    base = extent[0]
    return [(start + base, end + base, spoken) for start, end, spoken in rewrites]


def gaps_before(document) -> dict:
    """`{clip_id: (kind, seconds)}` for the silence `compute_arrangement`
    places before each clip in the text, in one pass in text order: kind
    "time" when the clip is placed by `timeline_timestamp` (seconds is the
    timestamp), else "override" (its own `gap_before_s`), "paragraph" or
    "clip" (the document's gaps), or "first" for the first clip. Like the
    arrangement, a bed placed in time doesn't count as the clip before the
    next one."""
    # Every clip's extent in one walk over the runs (`clip_extent` is a scan
    # of its own per clip).
    spans: dict = {}
    for run, start, end in document._iter_runs_with_offsets():
        if run.clip_id is None:
            continue
        first, last = spans.get(run.clip_id, (start, end))
        spans[run.clip_id] = (min(first, start), max(last, end))
    extents = sorted(((spans[c.id][0], spans[c.id][1], c) for c in document.clips if c.id in spans),
                     key=lambda item: item[0])
    text = document.text
    out = {}
    previous_end = None
    for start, end, clip in extents:
        if clip.timeline_timestamp is not None:
            out[clip.id] = ("time", float(clip.timeline_timestamp))
        else:
            seconds = arrangement.boundary_gap_s(document, text, previous_end, clip, start)
            if clip.gap_before_s is not None:
                kind = "override"
            elif previous_end is None:
                kind = "first"
            else:
                kind = "paragraph" if arrangement.is_paragraph_break(text[previous_end:start]) else "clip"
            out[clip.id] = (kind, seconds)
        if not (clip.is_bed and clip.timeline_timestamp is not None):
            previous_end = end
    return out


def gap_before(document, clip):
    """`gaps_before` for one clip, None when it isn't in the text."""
    return gaps_before(document).get(clip.id)


def estimated_length_s(document, clip, rates: dict, text=None) -> float:
    """What a stale clip (or `text`, one of its pieces) will run to at its
    speed and its character's learned pace (`fit.speaking_rates`), the way
    the timeline estimates an ungenerated clip."""
    speed = document.effective_config_for_clip(clip).get("speed", 1.0)
    try:
        speed = float(speed)
    except (TypeError, ValueError):
        speed = 1.0
    rate = rates.get(clip.character_id) or rates.get(None)
    return arrangement.estimate_duration_s(document.clip_text(clip) if text is None else text, speed, rate)


def piece_at(pieces: list, offset: int):
    """The piece covering document `offset`, else None."""
    for piece in pieces:
        if piece.start <= offset < piece.end:
            return piece
    return None


# What a piece ends at, for its tooltip ("ends at a sentence").
LEVEL_ENDINGS = {
    segmenting.PARAGRAPH: "a paragraph",
    segmenting.SENTENCE: "a sentence",
    segmenting.PAUSE: "a pause",
    segmenting.WORD: "a word break (forced)",
}


def format_length(seconds, estimate: bool = False) -> str:
    """`11.4 s`, `1:02` or `1:02:03`; an estimate is rounded and marked,
    `~11 s`."""
    seconds = max(0.0, float(seconds or 0.0))
    prefix = "~" if estimate else ""
    if seconds < 60:
        return f"{prefix}{seconds:.0f} s" if estimate else f"{seconds:.1f} s"
    whole = int(round(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{prefix}{hours}:{minutes:02d}:{secs:02d}"
    return f"{prefix}{minutes}:{secs:02d}"
