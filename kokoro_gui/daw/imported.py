"""Imported recordings edited as text (phase 5 P3, grill Q32/Q33).

Timing belongs to the text. Every word of an imported recording's
transcript carries where it sits in the original file, as a `Run.words`
entry `[char_start, char_end, source, start_s, end_s]` on the run holding
it; `source` is a key of `Document.sources` (the file's hash stem under
`audio/imported/`). An imported recording clip (`is_recording_clip`) has no
stored cut list: `segments_for` derives its segments from its words, one
`Segment` per stretch of contiguous audio, and `Document` keeps
`clip.segments` as a cache of that (`Document.refresh_imported_segments`),
so the transport, mixdown, waveform and SRT read it unchanged. Delete, cut,
paste and move edit the words with the text, and the audio follows.

The helpers here build words for an import: `run_from_asr_words` from a
Whisper pass (`asr.transcribe_wav_words`), `words_from_cue` for a caption
cue (proportional times), and `run_words_for_text` for text realigned to
ASR words (`wordalign.align`). `words_payload` is the clipboard shape a
cut or copy carries (`WORDS_MIME_TYPE`), which `Document.apply_words`
reads back on paste. `segment_plays` is what playback and export read:
consecutive ranges of one imported clip join with a short crossfade, so a
deleted word leaves no click.

Qt-free, like the rest of `kokoro_gui/daw/`.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

from kokoro_gui.daw.models import IMPORTED, Segment

# Consecutive words of one source whose times are closer than this (either
# way) play as one range.
JOIN_GAP_S = 0.06
# Crossfade at each join between two ranges of one imported clip.
JOIN_CROSSFADE_S = 0.005
# The clipboard format for timed text (JSON, see `words_payload`).
WORDS_MIME_TYPE = "application/x-kokorogui-words+json"

# Punctuation trimmed off a word's character span, so deleting a comma or a
# quote next to a word doesn't take the word's audio with it.
_EDGE_PUNCTUATION = "\"'.,;:!?()[]{}<>-_*/\\|~`‘’“”«»¿¡…–—"
_TOKEN = re.compile(r"\S+")


def is_recording_clip(clip) -> bool:
    """True for an imported clip whose audio comes from its text's words:
    `source == "imported"` with no `original_audio_path` (a music bed,
    phase 5 P2, has one and plays that file instead)."""
    return getattr(clip, "source", None) == IMPORTED and not getattr(clip, "original_audio_path", None)


# -- deriving segments ---------------------------------------------------------------


def clip_words(document, clip) -> list:
    """`(doc_start, doc_end, source, start_s, end_s)` for every word of
    `clip`, in text order, with document character offsets."""
    out = []
    for run, r_start, _r_end in document._iter_runs_with_offsets():
        if run.clip_id != clip.id or not run.words:
            continue
        for word in sorted(run.words, key=lambda w: w[0]):
            try:
                out.append((r_start + int(word[0]), r_start + int(word[1]), str(word[2]),
                            float(word[3]), float(word[4])))
            except (TypeError, ValueError, IndexError):
                continue
    return out


def word_spans(document) -> list:
    """`(doc_start, doc_end)` of every timed word in the document, in text
    order: what the editor underlines as carrying audio."""
    out = []
    for run, r_start, _r_end in document._iter_runs_with_offsets():
        for word in run.words or ():
            try:
                out.append((r_start + int(word[0]), r_start + int(word[1])))
            except (TypeError, ValueError, IndexError):
                continue
    out.sort()
    return out


def _round(value: float) -> float:
    return round(float(value), 6)


def segments_for(document, clip) -> list:
    """The segments an imported recording clip plays: its words walked in
    text order, consecutive words of one source whose times meet (a gap
    under `JOIN_GAP_S` either way) merged into one range, one `Segment` per
    range. `audio_path` is the source's file (None when the file is
    missing, which playback skips), `range` `[start_s, end_s]` into it,
    `words` the words inside as `[text, start_s, end_s]` relative to the
    range start (the `Segment.words` shape the playhead highlight reads),
    `duration` the range's length, `text` the transcript it covers, `raw`
    True so read-time FX apply. A clip with no words has none."""
    words = clip_words(document, clip)
    if not words:
        return []
    text = document.text
    groups: list = []
    for word in words:
        _s, _e, source, start_s, end_s = word
        if end_s <= start_s:
            continue
        if groups:
            group = groups[-1]
            if group["source"] == source and abs(start_s - group["end"]) < JOIN_GAP_S and end_s > group["end"]:
                group["end"] = end_s
                group["words"].append(word)
                continue
        groups.append({"source": source, "start": start_s, "end": end_s, "words": [word]})

    segments = []
    for index, group in enumerate(groups):
        start_s, end_s = group["start"], group["end"]
        segments.append(Segment(
            order_index=index,
            text=text[group["words"][0][0]:group["words"][-1][1]],
            audio_path=document.source_path(group["source"]),
            duration=_round(end_s - start_s),
            raw=True,
            words=[[text[w[0]:w[1]], _round(w[3] - start_s), _round(w[4] - start_s)] for w in group["words"]],
            range=[start_s, end_s],
        ))
    return segments


def same_segments(a: list, b: list) -> bool:
    """True when two segment lists play the same audio with the same words
    and text (ids aside), so a refresh can keep the old list."""
    def shape(segments):
        return [(s.order_index, s.text, s.audio_path, s.duration, s.raw, s.words, s.range) for s in segments]

    return shape(a) == shape(b)


def missing_sources(document) -> list:
    """Names of the sources some run's words use whose file is unknown or
    missing (not found on open, or deleted since). Their words play
    nothing; the GUI says so."""
    used = {w[2] for run in document.runs for w in run.words or () if len(w) > 2}
    return sorted(s for s in used
                  if document.source_path(s) is None or not os.path.isfile(document.source_path(s)))


def clip_missing_source(document, clip) -> bool:
    """True when one of the clip's words names a source without a file."""
    missing = set(missing_sources(document))
    return any(word[2] in missing for word in clip_words(document, clip))


# -- building words for an import ----------------------------------------------------


def _core_span(token: str) -> tuple:
    """`(start, end)` of `token` with edge punctuation trimmed, or the whole
    token when it is nothing but punctuation."""
    stripped = token.strip(_EDGE_PUNCTUATION)
    if not stripped:
        return 0, len(token)
    start = token.index(stripped)
    return start, start + len(stripped)


def _tile(times: list) -> list:
    """`times` (`(start_s, end_s)` per word, in order) with each pause
    between two words split at its midpoint, so consecutive words meet and
    a deleted word takes its share of the pauses around it. The first
    start and the last end stay as given."""
    out = [list(t) for t in times]
    for i in range(len(out) - 1):
        end, start = out[i][1], out[i + 1][0]
        if start > end:
            middle = (end + start) / 2.0
            out[i][1] = middle
            out[i + 1][0] = middle
    return [(_round(s), _round(max(s, e))) for s, e in out]


def run_words_for_text(text: str, timed: list, source: str) -> list:
    """`Run.words` for `text` from `timed`, one `[word, start_s, end_s]`
    row per whitespace token of `text` in order (what `wordalign.align`
    returns, so text edited in the review dialog keeps its times). Each
    word's characters exclude edge punctuation; pauses between words are
    split between them (`_tile`). A token without a row, or a row with
    `end_s <= start_s` after tiling, gets no entry."""
    tokens = list(_TOKEN.finditer(text or ""))
    pairs = []
    for match, row in zip(tokens, timed or []):
        try:
            start_s, end_s = float(row[1]), float(row[2])
        except (TypeError, ValueError, IndexError):
            continue
        pairs.append((match, (max(0.0, start_s), max(0.0, end_s))))
    tiled = _tile([t for _m, t in pairs])
    words = []
    for (match, _raw), (start_s, end_s) in zip(pairs, tiled):
        if end_s <= start_s:
            continue
        lo, hi = _core_span(match.group())
        words.append([match.start() + lo, match.start() + hi, source, start_s, end_s])
    return words


def run_from_asr_words(words: list, source: str) -> tuple:
    """`(text, run_words)` for a list of ASR words `[(word, start_s,
    end_s)]` (`asr.transcribe_wav_words`, or a slice of it for one clip):
    the words joined by single spaces, and their `Run.words`
    (`run_words_for_text`)."""
    rows = []
    for item in words or []:
        try:
            word, start_s, end_s = str(item[0]).strip(), float(item[1]), float(item[2])
        except (TypeError, ValueError, IndexError):
            continue
        if word:
            rows.append([word, start_s, end_s])
    text = " ".join(row[0] for row in rows)
    return text, run_words_for_text(text, rows, source)


def words_from_cue(text: str, start_s: float, end_s: float, source: str) -> list:
    """`Run.words` for a caption cue's `text` spoken over `[start_s,
    end_s]` of `source` (grill Q33): the cue span shared among the
    whitespace tokens in proportion to their character counts, which is all
    a caption knows. The words tile the span, so an untouched cue plays as
    one range."""
    tokens = list(_TOKEN.finditer(text or ""))
    total = sum(len(t.group()) for t in tokens)
    start_s, end_s = float(start_s), float(end_s)
    if not tokens or total <= 0 or end_s <= start_s:
        return []
    span = end_s - start_s
    words = []
    done = 0
    for match in tokens:
        w_start = start_s + span * done / total
        done += len(match.group())
        w_end = end_s if done == total else start_s + span * done / total
        lo, hi = _core_span(match.group())
        words.append([match.start() + lo, match.start() + hi, source, _round(w_start), _round(w_end)])
    return words


def source_entry(path: str) -> tuple:
    """`(source, entry)` for a file `import_audio_file` wrote: its hash stem
    and the `Document.sources` entry `{"path", "sample_rate",
    "duration_s"}`, rate and length read from the header (None when
    soundfile can't read it)."""
    path = os.path.abspath(path)
    source = os.path.splitext(os.path.basename(path))[0]
    sample_rate = duration_s = None
    try:
        import soundfile as sf

        info = sf.info(path)
        sample_rate = int(info.samplerate)
        duration_s = _round(info.frames / float(info.samplerate)) if info.samplerate else None
    except Exception:
        pass
    return source, {"path": path, "sample_rate": sample_rate, "duration_s": duration_s}


# -- clipboard -----------------------------------------------------------------------


def words_payload(document, start: int, end: int) -> dict:
    """The `WORDS_MIME_TYPE` payload for a cut or copy of `[start, end)`:
    `{"words": [...], "sources": {...}}`, the words wholly inside the
    selection with offsets relative to `start`, and the `Document.sources`
    entries they use (absolute paths, so a paste into another project can
    import the file). Empty `words` when the selection holds no timed
    text. JSON-serializable."""
    words = []
    used = set()
    for run, r_start, r_end in document._iter_runs_with_offsets():
        if r_end <= start or r_start >= end or not run.words:
            continue
        for word in run.words:
            w_start, w_end = r_start + int(word[0]), r_start + int(word[1])
            if w_start >= start and w_end <= end:
                words.append([w_start - start, w_end - start, word[2], float(word[3]), float(word[4])])
                used.add(word[2])
    sources = {s: dict(document.sources[s]) for s in sorted(used) if s in document.sources}
    return {"words": words, "sources": sources}


# -- playback ------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentPlay:
    """How one segment of a clip plays: `range_s` is its nominal slice
    (what the timeline measures and the next segment starts after),
    `play_range_s` the slice actually read (the nominal one plus the
    crossfade tail), and the fades in seconds."""
    segment: object
    range_s: Optional[tuple]
    play_range_s: Optional[tuple]
    fade_in_s: float = 0.0
    fade_out_s: float = 0.0


def segment_plays(clip, fade_in_s: float = 0.0, fade_out_s: float = 0.0) -> list:
    """`SegmentPlay`s for the clip's segments that have audio, in order.
    The clip's fade-in goes on the first and its fade-out on the last. For
    an imported clip, each join between two sliced segments crossfades over
    `JOIN_CROSSFADE_S`: the earlier segment reads that much past its range
    and fades out over it while the next one fades in from its start, so
    the next still starts where the earlier one's range ends and the clip's
    length is unchanged."""
    from kokoro_gui.audio.post import segment_range

    segments = [s for s in sorted(clip.segments, key=lambda s: s.order_index) if s.audio_path]
    joins = getattr(clip, "source", None) == IMPORTED
    plays = []
    last = len(segments) - 1
    for index, segment in enumerate(segments):
        range_s = segment_range(segment)
        play_range = range_s
        fade_in = fade_in_s if index == 0 else 0.0
        fade_out = fade_out_s if index == last else 0.0
        if joins and range_s is not None:
            if index > 0 and segment_range(segments[index - 1]) is not None:
                fade_in = max(fade_in, JOIN_CROSSFADE_S)
            if index < last and segment_range(segments[index + 1]) is not None:
                play_range = (range_s[0], range_s[1] + JOIN_CROSSFADE_S)
                fade_out = max(fade_out, JOIN_CROSSFADE_S)
        plays.append(SegmentPlay(segment=segment, range_s=range_s, play_range_s=play_range,
                                 fade_in_s=fade_in, fade_out_s=fade_out))
    return plays
