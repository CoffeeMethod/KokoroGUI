"""Offline export of a clip document (section 6 of
Claude/PLAN_ui_shell_redesign.md, WF8).

`render_mix()` walks `compute_arrangement` (the same placement the timeline
and transport use), reads every clip's segments through the same read-time
post-processing the transport plays (`kokoro_gui.audio.post`, via
`post_config_for_clip`), sums them at their start times with the same
plain-gain-sum rule the live `Transport` applies (grill Q21), and returns the
mix as a `MixResult` without writing anything (File > Measure Loudness uses
it that way). `mixdown()` is `render_mix` plus an optional loudness
normalize (`kokoro_gui.audio.loudness`) and the writes: one file. SRT rows come
straight from each `PlacedClip`'s start/duration and the clip's text, replacing `SrtMixin`'s segment-timing
walk for clip documents (that mixin stays for the no-clips whole-document
path).

The mix is stereo with the track controls (`kokoro_gui.daw.mixplan`):
muted and soloed-out tracks are left out, and the fader, pan, fades,
automation and ducking apply, so export is what the transport plays. The
mix runs `mixer.mix_block` block by block (`EXPORT_BLOCK_FRAMES`) with one
`mixer.DuckState` carried through, as the transport does per device block;
the sidechain's hops don't depend on block size, so the samples match. A
music bed (`Clip.is_bed`) is read from its file through
`beds.playable_segments` and gets no SRT row. `channels=1`
averages the two channels. `range_s` renders only a region (between two
markers): clips outside it are skipped, and the output, SRT and cue sheet
start at the region's start.

`render_mix(stems="track" | "character", dialogue_stem=True)` also returns one
`MixResult.stems` entry per group of clips: the same mix with every other
group's clips left out, the same length, so the stems line up at 0. The
speech outside a stem still feeds the ducking sidechain
(`LoadedClip.sidechain_only`), so a ducked bed stem is ducked as in the full
mix. `mixdown` gives the stems the full mix's normalize gain (and, in RMS
mode, its limiter gain), pads them alike and writes
`<base>_<stem>[_<timecode>].<ext>` beside the mixdown.

`mixdown(out_rate=...)` resamples the whole mix once (`resample_mix`) before
the loudness step, and `bitrate_kbps` sets the mp3 bitrate (`write_audio`).
`expand_name` turns the dialog's file-name template (`{project}`, `{date}`,
`{time}`, `{range}`) into a safe base name and `unused_path` finds the
`name (2).ext` the dialog offers instead of overwriting.

"Keep per-clip files" writes one mono file per clip next to the mixdown,
named `<base>_<index:03>_<character>.<ext>` (UI14), the character name
basename-sanitized like every other name-to-path site in this codebase.

`mixdown(tags=...)` writes the file tags (`kokoro_gui.daw.tagging`) into each
mixdown once it is written: the fields, the cover and, in an mp3, the markers
(or subprojects) as chapter frames from `transcripts.chapter_rows`. A split
export titles each file with its chapter and numbers it `n/total`. Stems and
per-clip files are working files and get no tags.

`mixdown(fmt="m4b")` writes the main file through `kokoro_gui.daw.m4b` (an
ffmpeg subprocess): AAC at `bitrate_kbps`, a chapter per subproject (else per
marker), and the tags and cover from `tags` when given. Stems and per-clip
files in M4B come out without chapters.

`write_srt(granularity="word")` writes one subtitle per stored word
(`Segment.words`) instead of one per clip. `write_cue_sheet` writes a CSV
row per clip for review and dubbing: timecode (when the document has it
enabled) or seconds, character, source text, text, status, note.

No Qt here, and the audio math is the shared `kokoro_gui.audio.mixer`, so
the whole thing is testable headlessly with a stub document.
"""
from __future__ import annotations

import csv
import datetime
import math
import os
import re
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np
from scipy.signal import resample_poly

from kokoro_gui.audio import limiter, loudness as loudness_mod, mixer, post
from kokoro_gui.daw.arrangement import Arrangement, compute_arrangement, segment_timeline
from kokoro_gui.daw import m4b, markers as marker_ops, tagging, transcripts
from kokoro_gui.daw.imported import segment_plays
from kokoro_gui.daw.mixplan import ClipMix, clip_mixes
from kokoro_gui.daw.timecode import format_position
from kokoro_gui.engine.text_extraction import strip_markup

SOUNDFILE_FORMATS = {"wav", "flac", "ogg"}
CUE_SHEET_COLUMNS = ("start", "end", "character", "source_text", "text", "status", "note")
EXPORT_BLOCK_FRAMES = 1 << 16


@dataclass
class ExportFile:
    """One written mixdown: the report from measuring it (None when nothing
    measured it), the `describe` of every preset check it failed, and the
    name its chapter has (the file's base name for a single export)."""
    path: str
    title: str = ""
    duration_s: float = 0.0
    report: Optional[object] = None
    failed: list = field(default_factory=list)
    tagged: bool = False


@dataclass
class ExportResult:
    audio_path: str
    srt_path: Optional[str] = None
    clip_files: list = field(default_factory=list)
    duration_s: float = 0.0
    skipped_clip_ids: list = field(default_factory=list)
    cue_sheet_path: Optional[str] = None
    # Set when `mixdown(loudness=...)` ran: the mix measured before and after
    # the gain (`audio.loudness.LoudnessReport`), and whether the true-peak
    # ceiling kept it short of the target.
    loudness_before: Optional[object] = None
    loudness_after: Optional[object] = None
    loudness_limited: bool = False
    # One entry per mixdown written (a split export has several, and
    # `audio_path` is the first); `warnings` are sentences for the status line.
    files: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    # The stem files written, in the order of `render_mix`'s stems.
    stem_files: list = field(default_factory=list)
    # The `transcripts.TEXT_EXTRAS` files written, in that order.
    text_files: list = field(default_factory=list)


def _format_srt_time(seconds: float) -> str:
    return transcripts.format_timestamp(seconds)


def word_rows(arrangement: Arrangement) -> list:
    """`(start_s, end_s, word)` for every stored word, in timeline seconds."""
    rows = []
    for placed in arrangement.placed:
        if placed.estimated:
            continue
        for segment, seg_start, scale in segment_timeline(placed):
            for word in segment.words or []:
                try:
                    text, w_start, w_end = str(word[0]).strip(), float(word[1]), float(word[2])
                except (TypeError, ValueError, IndexError):
                    continue
                if text:
                    rows.append((seg_start + w_start * scale, seg_start + w_end * scale, text))
    return rows


def write_srt(document, arrangement: Arrangement, path: str, granularity: str = "clip") -> str:
    if granularity == "word":
        spans = word_rows(arrangement)
    else:
        spans = []
        for placed in arrangement.placed:
            if placed.estimated or getattr(placed.clip, "is_bed", False):
                continue
            text = strip_markup(document.clip_text(placed.clip)).strip()
            if text:
                spans.append((placed.start_s, placed.end_s, text))
    rows = [
        f"{index}\n{_format_srt_time(start)} --> {_format_srt_time(end)}\n{text}\n"
        for index, (start, end, text) in enumerate(spans, start=1)
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
        if rows:
            f.write("\n")
    return path


def _cue_time(settings, seconds: float) -> str:
    return format_position(settings, seconds) or f"{seconds:.3f}"


def write_cue_sheet(document, arrangement: Arrangement, path: str) -> str:
    """One CSV row per placed clip, in timeline order."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CUE_SHEET_COLUMNS)
        for placed in sorted(arrangement.placed, key=lambda p: p.start_s):
            clip = placed.clip
            character = document.get_character(clip.character_id)
            writer.writerow([
                _cue_time(document.settings, placed.start_s),
                _cue_time(document.settings, placed.end_s),
                character.name if character is not None else "",
                clip.source_text or "",
                strip_markup(document.clip_text(clip)).strip(),
                clip.status,
                clip.note,
            ])
    return path


_UNSAFE_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
NAME_TOKENS = ("project", "date", "time", "range")


def name_context(project: str = "Untitled", range_label: Optional[str] = None,
                 now: Optional[datetime.datetime] = None, **extra) -> dict:
    """The values `expand_name` fills tokens from: `project`, `date`
    (YYYY-MM-DD), `time` (HHMMSS) and `range` ("full" for the whole project).
    Later export features pass their own tokens through `extra`."""
    now = now or datetime.datetime.now()
    return {"project": project or "Untitled", "date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H%M%S"),
            "range": range_label or "full", **extra}


def expand_name(template: str, context: dict) -> str:
    """A base filename from `template`: `{token}` is replaced by
    `context[token]`, an unknown token stays as typed. The template goes
    through `os.path.basename()` first (the guard every name-to-path site
    uses), then every substituted value loses the characters a filename
    can't hold, so a project called "a/b" can't add a folder. The result is
    never empty and never ends in a dot or a space."""
    template = os.path.basename((template or "").strip())

    def _fill(match):
        value = context.get(match.group(1))
        return match.group(0) if value is None else _UNSAFE_NAME_CHARS.sub("", str(value))

    name = _UNSAFE_NAME_CHARS.sub("", re.sub(r"\{(\w+)\}", _fill, template)).strip(" .")
    return name or "output"


def unused_path(path: str) -> str:
    """`path`, or `<base> (2).<ext>`, `<base> (3).<ext>`... for the first
    name that doesn't exist yet."""
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    number = 2
    while os.path.exists(f"{root} ({number}){ext}"):
        number += 1
    return f"{root} ({number}){ext}"


def resample_mix(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """`samples` (`(frames,)` or `(frames, channels)`) at `to_rate`, one
    polyphase pass over the whole mix."""
    from_rate, to_rate = int(from_rate), int(to_rate)
    if from_rate == to_rate or len(samples) == 0:
        return samples
    divisor = math.gcd(from_rate, to_rate)
    return resample_poly(samples, to_rate // divisor, from_rate // divisor, axis=0).astype(np.float32)


def _safe_component(name: str) -> str:
    """A character name as a filename piece: separators and other unsafe
    characters become "_" first (so "Bo b/ok" keeps both halves), then the
    same `os.path.basename()` guard every other name-to-path site uses."""
    name = re.sub(r'[<>:"/\|?*\\|?*\s]+', "_", (name or "").strip())
    name = os.path.basename(name)
    return name or "clip"


def _clip_samples(clip, sample_rate: int, post_config: Optional[dict] = None,
                  nested_audio_path: Optional[Callable] = None) -> Optional[np.ndarray]:
    """All of a clip's segments concatenated at `sample_rate`, post-processed
    per `post_config`, or None if none of them can be read. A segment with a
    `range` contributes only that slice of its file; a music bed is its
    file's trim range, repeated when it loops; an imported recording's
    ranges join with a short crossfade (`_crossfaded_samples`). The
    segments come from `imported.segment_plays`, which reads
    `beds.playable_segments`. A nested clip (a subproject) is its child's
    mixdown file, `nested_audio_path(clip)`."""
    if getattr(clip, "source", None) == "nested":
        path = nested_audio_path(clip) if nested_audio_path is not None else None
        if not path:
            return None
        try:
            return mixer.load_clip_samples(path, sample_rate, post_config).astype(np.float32)
        except Exception:
            return None
    plays = segment_plays(clip)
    if any(p.play_range_s != p.range_s for p in plays):
        return _crossfaded_samples(plays, sample_rate, post_config)
    parts = []
    for play in plays:
        try:
            parts.append(mixer.load_clip_samples(play.segment.audio_path, sample_rate, post_config, play.range_s))
        except Exception:
            continue
    if not parts:
        return None
    return np.concatenate(parts).astype(np.float32)


def _crossfaded_samples(plays: list, sample_rate: int, post_config: Optional[dict]) -> Optional[np.ndarray]:
    """An imported clip's ranges laid end to end at their nominal lengths,
    each join crossfaded the way the transport plays it
    (`imported.segment_plays`): the earlier range's read runs on past its
    end, fading out, over the start of the next one, fading in."""
    parts = []  # (samples, nominal_frames, fade_in_frames, fade_out_frames)
    for play in plays:
        path = play.segment.audio_path
        try:
            samples = mixer.load_clip_samples(path, sample_rate, post_config, play.play_range_s)
            nominal = samples if play.play_range_s == play.range_s else \
                mixer.load_clip_samples(path, sample_rate, post_config, play.range_s)
        except Exception:
            continue
        parts.append((samples, len(nominal), int(round(play.fade_in_s * sample_rate)),
                      int(round(play.fade_out_s * sample_rate))))
    if not parts:
        return None
    total = sum(nominal for _s, nominal, _i, _o in parts)
    out = np.zeros(max(total, 1), dtype=np.float32)
    cursor = 0
    for samples, nominal, fade_in, fade_out in parts:
        piece = np.array(samples, dtype=np.float32)
        n = len(piece)
        fade_in, fade_out = min(fade_in, n), min(fade_out, n)
        if fade_in:
            piece[:fade_in] *= np.arange(fade_in, dtype=np.float32) / float(fade_in)
        if fade_out:
            piece[n - fade_out:] *= (fade_out - np.arange(fade_out, dtype=np.float32)) / float(fade_out)
        end = min(len(out), cursor + n)
        out[cursor:end] += piece[:end - cursor]
        cursor += nominal
    return out[:total]


def write_audio(path: str, samples: np.ndarray, sample_rate: int, fmt: str,
                bitrate_kbps: Optional[int] = None) -> None:
    """`samples` is `(frames,)` mono or `(frames, channels)`. `bitrate_kbps`
    applies to mp3 and m4b (an M4B through here has no chapters or tags;
    `mixdown` writes the main file with them)."""
    fmt = (fmt or "wav").lower()
    if fmt == "m4b":
        m4b.write_m4b(samples, sample_rate, path, bitrate_kbps)
        return
    if fmt in SOUNDFILE_FORMATS:
        import soundfile as sf

        sf.write(path, samples, sample_rate, format=fmt.upper())
        return
    # mp3 and anything else soundfile can't encode: the same pedalboard
    # AudioFile path ConversionMixin.smart_combine uses.
    from pedalboard.io import AudioFile

    channels = 1 if samples.ndim == 1 else samples.shape[1]
    # pedalboard 0.9.23: `quality` takes an int of kilobits per second (a
    # string like "320" or "192k" works too) and encodes constant bitrate. With
    # none given the encoder writes 320. Below 32 kHz (MPEG-2) LAME stops at
    # 160 kbps, so 192 and up come out at 160 there.
    with AudioFile(path, "w", samplerate=sample_rate, num_channels=channels,
                   quality=bitrate_kbps) as out_f:
        out_f.write(samples.reshape(1, -1) if samples.ndim == 1 else samples.T)


def _start_timecode(document, range_s: Optional[tuple]) -> str:
    """The timecode of where the export starts (the project start, or the
    range's), as digits only (`01000000`), or "" when the project's timecode
    is off."""
    start = min(max(0.0, float(range_s[0])), max(0.0, float(range_s[1]))) if range_s else 0.0
    return re.sub(r"\D", "", format_position(document.settings, start) or "")


def stem_path(out_dir: str, base: str, stem_name: str, timecode: str = "", ext: str = "wav",
              used: Optional[set] = None) -> str:
    """`<out_dir>/<base>_<stem>[_<timecode>].<ext>`, the stem name made safe
    (`_safe_component`). A name already in `used` (lower case, because
    Windows doesn't tell the case apart) gets `_2`, `_3`... and the name is
    added to `used`."""
    stem = _safe_component(stem_name)
    tail = f"_{timecode}" if timecode else ""
    name = f"{base}_{stem}{tail}.{ext}"
    if used is not None:
        number = 2
        while name.lower() in used:
            name = f"{base}_{stem}_{number}{tail}.{ext}"
            number += 1
        used.add(name.lower())
    return os.path.join(out_dir, name)


def _in_range(arrangement: Arrangement, range_s: tuple) -> Arrangement:
    """The placed clips overlapping `range_s`, shifted so the range starts
    at 0; the total is the range's length."""
    lo, hi = sorted((max(0.0, float(range_s[0])), max(0.0, float(range_s[1]))))
    placed = [replace(p, start_s=p.start_s - lo) for p in arrangement.placed if p.end_s > lo and p.start_s < hi]
    return Arrangement(placed=placed, total_duration_s=hi - lo)


STEM_MODES = (None, "track", "character")
DIALOGUE_STEM = "Dialogue"
UNASSIGNED_STEM = "Unassigned"
MUSIC_STEM = "Music"


@dataclass
class Stem:
    """One stem of `render_mix`: its name (a track, a character, "Dialogue"),
    and the `MixResult.samples` shape of the mix with only its clips."""
    name: str
    samples: np.ndarray


@dataclass
class MixResult:
    """What `render_mix` hands back: the mix before any write.

    `samples` is `(frames, 2)` float32, or `(frames,)` for mono. `arrangement`
    is the one the mix was laid out on: shifted to start at 0 when `range_s`
    was given, which is what the SRT and cue sheet need. `per_clip` holds
    `(index, PlacedClip, samples)` for each clip that was read, before the
    track mix."""
    samples: np.ndarray
    arrangement: Arrangement
    per_clip: list
    skipped: list
    sample_rate: int
    stems: list = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return len(self.samples) / float(self.sample_rate)


def render_mix(document, sample_rate: int = 24000, arrangement: Optional[Arrangement] = None,
               engine_id: Optional[str] = None, progress: Optional[Callable[[float, str], None]] = None,
               post_config_for_clip: Optional[Callable] = None, channels: int = 2,
               range_s: Optional[tuple] = None, nested_audio_path: Optional[Callable] = None,
               stems: Optional[str] = None, dialogue_stem: bool = False) -> MixResult:
    """Reads every audible clip and sums them at their start times, writing
    nothing. `progress` runs over 0.0 to 0.8; the rest is `mixdown`'s.

    `stems` is "track" or "character": the result's `stems` then holds one
    entry for each track (or character) that has an audible clip. A track
    stem is that track's clips; a character stem is the clips of that
    character, with music beds together in "Music" and clips of no character
    in "Unassigned". `dialogue_stem` adds a "Dialogue" stem, every audible
    clip except the music beds. Muted and soloed-out tracks have no stem, as
    they have no clips in the mix. Each stem has the mix's length and the
    same ducking.

    `post_config_for_clip(clip)` returns the clip's resolved read-time
    post-processing config (the app passes `QtTTSApp.post_config_for_clip`);
    None uses the raw segment files as they are. `nested_audio_path(clip)`
    names a subproject's mixdown file (phase 4)."""
    if arrangement is None:
        arrangement = compute_arrangement(document, engine_id=engine_id)
    mixes = clip_mixes(document, arrangement)
    if range_s is not None:
        arrangement = _in_range(arrangement, range_s)
    sample_rate = int(sample_rate)

    if stems not in STEM_MODES:
        raise ValueError(f"unknown stem mode {stems!r}")
    loaded: list = []
    loaded_clips: list = []
    skipped: list = []
    per_clip: list = []
    total = max(1, len(arrangement.placed))
    for index, placed in enumerate(arrangement.placed):
        if progress:
            progress(index / total * 0.8, f"Reading clip {index + 1}/{total}")
        mix = mixes.get(placed.clip.id)
        if mix is None:
            continue  # muted, or soloed out
        post_config = post_config_for_clip(placed.clip) if post_config_for_clip else None
        samples = _clip_samples(placed.clip, sample_rate, post_config, nested_audio_path)
        if samples is None:
            skipped.append(placed.clip.id)
            continue
        loaded.append(_loaded(placed, samples, mix, sample_rate, range_s))
        loaded_clips.append(placed.clip)
        per_clip.append((index, placed, samples))

    total_frames = int(round(arrangement.total_duration_s * sample_rate))
    if range_s is None:
        total_frames = max(mixer.total_frames(loaded), total_frames)
    duck_db = duck_db_setting(document)
    mixed = _mix_clips(loaded, total_frames, sample_rate, duck_db, channels)
    stem_list = []
    groups = _stem_groups(document, loaded_clips, stems) if stems else []
    if dialogue_stem:
        groups.append((DIALOGUE_STEM, [i for i, clip in enumerate(loaded_clips) if not clip.is_bed]))
    for number, (name, indices) in enumerate(groups):
        if not indices:
            continue
        if progress:
            progress(0.8 + 0.01 * number / len(groups), f"Mixing stem {name}")
        stem_list.append(Stem(name, _mix_clips(_stem_clips(loaded, set(indices)), total_frames, sample_rate,
                                               duck_db, channels)))
    return MixResult(samples=mixed, arrangement=arrangement, per_clip=per_clip, skipped=skipped,
                     sample_rate=sample_rate, stems=stem_list)


def _mix_clips(loaded: list, total_frames: int, sample_rate: int, duck_db: float, channels: int) -> np.ndarray:
    """`loaded` summed block by block into `(frames, 2)`, or `(frames,)` for
    mono; one `DuckState` runs through when any clip is ducked."""
    mixed = np.zeros((max(0, total_frames), mixer.CHANNELS), dtype=np.float32)
    duck = mixer.DuckState(sample_rate, duck_db) if any(c.duck for c in loaded) else None
    for start in range(0, max(0, total_frames), EXPORT_BLOCK_FRAMES):
        frames = min(EXPORT_BLOCK_FRAMES, total_frames - start)
        mixer.mix_block(loaded, start, frames, out=mixed[start:start + frames], duck=duck)
    if int(channels) == 1:
        mixed = mixed.mean(axis=1).astype(np.float32)
    return mixed


def _stem_groups(document, clips: list, mode: str) -> list:
    """`[(name, [indices into clips])]` for a stem mode, in track order or
    character order, the leftovers last."""
    if mode == "track":
        by_key: dict = {}
        for index, clip in enumerate(clips):
            by_key.setdefault(clip.track_id, []).append(index)
        groups = []
        for track in sorted(document.tracks, key=lambda t: t.order_index):
            if track.id in by_key:
                groups.append((track.name or "Track", by_key.pop(track.id)))
        rest = sorted(i for indices in by_key.values() for i in indices)
        return groups + ([(UNASSIGNED_STEM, rest)] if rest else [])
    if mode == "character":
        by_key = {}
        music = []
        for index, clip in enumerate(clips):
            if clip.is_bed:
                music.append(index)
            else:
                by_key.setdefault(clip.character_id, []).append(index)
        groups = []
        for character in document.characters:
            if character.id in by_key:
                groups.append((character.name or "Character", by_key.pop(character.id)))
        rest = sorted(i for indices in by_key.values() for i in indices)
        return (groups + ([(UNASSIGNED_STEM, rest)] if rest else []) + ([(MUSIC_STEM, music)] if music else []))
    raise ValueError(f"unknown stem mode {mode!r}")


def _stem_clips(loaded: list, chosen: set) -> list:
    """The clips one stem mixes: those at the `chosen` indices, plus, when
    one of them is ducked, the rest of the speech as sidechain-only clips so
    the stem ducks exactly as the full mix does."""
    members = [c for i, c in enumerate(loaded) if i in chosen]
    if not any(c.duck for c in members):
        return members
    return [c if i in chosen else replace(c, sidechain_only=True)
            for i, c in enumerate(loaded) if i in chosen or (c.sidechain and not c.duck)]


def mixdown(document, out_path: str, fmt: str = "wav", sample_rate: int = 24000,
            include_srt: bool = False, keep_clip_files: bool = False,
            arrangement: Optional[Arrangement] = None, engine_id: Optional[str] = None,
            progress: Optional[Callable[[float, str], None]] = None,
            post_config_for_clip: Optional[Callable] = None, channels: int = 2,
            range_s: Optional[tuple] = None, srt_granularity: str = "clip",
            include_cue_sheet: bool = False, nested_audio_path: Optional[Callable] = None,
            loudness: Optional[dict] = None, bitrate_kbps: Optional[int] = None,
            out_rate: Optional[int] = None, head_s: float = 0.0, tail_s: float = 0.0,
            checks: tuple = (), stems: Optional[str] = None, dialogue_stem: bool = False,
            extras: tuple = (), transcript_speakers: bool = True, tags: Optional[dict] = None) -> ExportResult:
    """`render_mix`, then an optional resample, an optional loudness
    normalize, optional head and tail silence, then the writes.

    `stems` ("track" or "character") and `dialogue_stem` write the stems
    `render_mix` makes as `<base>_<stem>[_<timecode>].<ext>` next to the
    mixdown (`stem_path`; the paths land in `ExportResult.stem_files`). A stem
    is resampled, given the full mix's normalize gain, padded and written like
    the mixdown, so every stem starts at the same sample and they sum to it.
    In RMS mode the peak limiter's gain is applied to the stems too. A stem is
    not measured and the preset checks don't run on it.

    `extras` (keys of `transcripts.TEXT_EXTRAS`: "vtt", "srt_speakers",
    "transcript_json", "txt", "chapters_json", "show_notes") write those text
    files as `<base><suffix>` beside the mixdown, from the same placed clips
    as the SRT, so a range or head padding moves their times the same way.
    `transcript_speakers` puts the character names in the transcripts. The
    paths land in `ExportResult.text_files`.

    `tags` writes file tags into the mixdown (mp3, flac and ogg; wav has none):
    any of `tagging.META_KEYS`, plus `"cover"` (a path) and `"chapters"` (true
    writes the markers, else the subprojects, as ID3 chapters in an mp3). A
    cover that can't be used, a missing `mutagen` or a write that fails adds a
    warning; the export goes on. An M4B takes the same fields through ffmpeg
    (no `mutagen` needed) and always carries its chapters, `tags["chapters"]`
    or not.

    `out_rate` writes the mixdown at that rate instead of `sample_rate`: the
    whole mix is resampled once before the loudness step, so what gets
    measured is what gets written. Per-clip files stay at `sample_rate`, and
    SRT and cue sheet times don't depend on the rate. `bitrate_kbps` is the
    mp3 bitrate (`write_audio`).

    `loudness` is one of
    `{"mode": "lufs", "target_lufs": float, "ceiling_dbtp": float}` (no mode
    means this one): the mix gets the gain that reaches the target without its
    true peak passing the ceiling (`audio.loudness.gain_to_target`; there is
    no limiter, so a peaky mix can end short of the target, which
    `ExportResult.loudness_limited` reports); or
    `{"mode": "rms", "target_rms_dbfs": float, "limiter_dbfs": float}`: the
    gain that reaches the RMS target, then `audio.limiter.limit_peaks` holds
    every sample under `limiter_dbfs`. The report before and after lands on
    the result. None writes the mix as rendered and measures nothing. Per-clip
    files are never normalized.

    `head_s` and `tail_s` pad the mixdown with silence after the loudness
    step, and the SRT and cue sheet move later by `head_s`. `checks`
    (`export_presets.Check`) run on the file as written; each file's report
    and failed checks are in `ExportResult.files`."""
    sample_rate = int(sample_rate)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(out_path))[0]
    ext = (fmt or "wav").lower()

    mix = render_mix(document, sample_rate, arrangement=arrangement, engine_id=engine_id, progress=progress,
                     post_config_for_clip=post_config_for_clip, channels=channels, range_s=range_s,
                     nested_audio_path=nested_audio_path, stems=stems, dialogue_stem=dialogue_stem)
    mixed = mix.samples
    stem_samples = [stem.samples for stem in mix.stems]
    mix_rate = int(out_rate) if out_rate else sample_rate
    if mix_rate != sample_rate:
        if progress:
            progress(0.81, f"Resampling to {mix_rate} Hz")
        mixed = resample_mix(mixed, sample_rate, mix_rate)
        stem_samples = [resample_mix(s, sample_rate, mix_rate) for s in stem_samples]
    result = ExportResult(audio_path=out_path, skipped_clip_ids=mix.skipped)
    if loudness is not None:
        if progress:
            progress(0.82, "Measuring loudness")
        mixed, result.loudness_before, result.loudness_after, result.loudness_limited, stem_samples = _normalized(
            mixed, mix_rate, loudness, stem_samples)
    head_s, tail_s = max(0.0, float(head_s or 0.0)), max(0.0, float(tail_s or 0.0))
    arrangement_out = mix.arrangement
    report = result.loudness_after
    if head_s or tail_s:
        mixed = _padded(mixed, mix_rate, head_s, tail_s)
        stem_samples = [_padded(s, mix_rate, head_s, tail_s) for s in stem_samples]
        arrangement_out = replace(arrangement_out, placed=[
            replace(p, start_s=p.start_s + head_s) for p in arrangement_out.placed])
        if loudness is not None or checks:
            report = result.loudness_after = loudness_mod.measure(mixed, mix_rate)
    elif checks and report is None:
        report = loudness_mod.measure(mixed, mix_rate)
    result.duration_s = len(mixed) / float(mix_rate)
    if progress:
        progress(0.85, "Writing mixdown")
    tagged = False
    if ext == "m4b":
        tagged = _write_m4b(out_path, mixed, mix_rate, bitrate_kbps, tags, document, arrangement_out, range_s,
                            head_s, result.duration_s, result.warnings)
    else:
        write_audio(out_path, mixed, mix_rate, ext, bitrate_kbps=bitrate_kbps)
    failed = [c.describe(report) for c in checks if not c.passed(report)] if checks else []
    if tags and tagging.supports(ext):
        tagged = _tag_file(out_path, ext, tags, document, arrangement_out, range_s, head_s, result.duration_s,
                           result.warnings)
    result.files.append(ExportFile(out_path, base, result.duration_s, report, failed, tagged))

    if stem_samples:
        used = {os.path.basename(out_path).lower()}
        timecode = _start_timecode(document, range_s)
        for number, (stem, samples) in enumerate(zip(mix.stems, stem_samples)):
            if progress:
                progress(0.85 + 0.1 * number / len(stem_samples), f"Writing stem {stem.name}")
            stem_file = stem_path(out_dir, base, stem.name, timecode, ext, used)
            write_audio(stem_file, np.clip(samples, -1.0, 1.0), mix_rate, ext, bitrate_kbps=bitrate_kbps)
            result.stem_files.append(stem_file)

    if keep_clip_files:
        for index, placed, samples in mix.per_clip:
            character = document.get_character(placed.clip.character_id)
            who = _safe_component(character.name if character is not None else "clip")
            clip_path = os.path.join(out_dir, f"{base}_{index + 1:03d}_{who}.{ext}")
            write_audio(clip_path, samples, sample_rate, ext, bitrate_kbps=bitrate_kbps)
            result.clip_files.append(clip_path)

    if include_srt:
        srt_path = os.path.join(out_dir, f"{base}.srt")
        result.srt_path = write_srt(document, arrangement_out, srt_path, granularity=srt_granularity)

    if include_cue_sheet:
        result.cue_sheet_path = write_cue_sheet(document, arrangement_out, os.path.join(out_dir, f"{base}.csv"))

    if extras:
        result.text_files, text_warnings = transcripts.write_extras(
            document, arrangement_out, out_dir, base, extras, speakers=transcript_speakers,
            range_s=range_s, head_s=head_s)
        result.warnings.extend(text_warnings)

    if progress:
        progress(1.0, "Export finished")
    return result


def _write_m4b(path: str, samples: np.ndarray, rate: int, bitrate_kbps: Optional[int], tags: Optional[dict],
               document, arrangement: Arrangement, range_s: Optional[tuple], head_s: float, duration_s: float,
               warnings: list) -> bool:
    """Writes the main mixdown as an M4B: the chapters (subprojects, else
    markers) always, the tags and cover when `tags` is given. ffmpeg writes
    them, so this doesn't need `mutagen`. True when tags went in; a cover
    that can't be used lands in `warnings`."""
    chapters = m4b.m4b_chapters(document, arrangement, range_s, head_s)
    meta = {key: tags[key] for key in tagging.META_KEYS if tags.get(key)} if tags else {}
    problems = m4b.write_m4b(samples, rate, path, bitrate_kbps, chapters, meta, (tags or {}).get("cover"))
    warnings.extend(problem for problem in problems if problem not in warnings)
    return bool(tags)


def _tag_file(path: str, ext: str, tags: dict, document, arrangement: Arrangement, range_s: Optional[tuple],
              head_s: float, duration_s: float, warnings: list) -> bool:
    """Writes `mixdown`'s `tags` into the file at `path`; True when it did.
    Anything that goes wrong lands in `warnings`, since the audio is already
    on disk."""
    name = os.path.basename(path)
    if not tagging.available():
        warning = "mutagen isn't installed, so the files have no tags (pip install mutagen)"
        if warning not in warnings:
            warnings.append(warning)
        return False
    chapters = None
    if tags.get("chapters") and ext == "mp3":
        chapters = transcripts.chapter_rows(document, arrangement, range_s, head_s) or None
    try:
        problems = tagging.write_tags(path, ext, tags, chapters, tags.get("cover"), duration_s=duration_s)
    except Exception as e:  # noqa: BLE001 - the audio is written; tagging is a courtesy
        warnings.append(f"Couldn't write the tags into {name}: {e}")
        return False
    warnings.extend(problem for problem in problems if problem not in warnings)
    return True


@dataclass(frozen=True)
class Chapter:
    """One file of a split export: its base name, the `(start_s, end_s)` of
    the project it covers and the title its tags carry (the base name when
    empty)."""
    name: str
    range_s: tuple
    title: str = ""


@dataclass
class ChapterPlan:
    chapters: list
    warnings: list = field(default_factory=list)


SPLIT_NOUNS = {"subprojects": "subproject", "markers": "marker range"}
MAX_TITLE_CHARS = 100


def _chapter_title(text: str) -> str:
    """A chapter's name as a filename piece: reserved characters out, spaces
    collapsed, "Untitled" when nothing is left."""
    title = " ".join(_UNSAFE_NAME_CHARS.sub("", text or "").split()).strip(" .")[:MAX_TITLE_CHARS].strip(" .")
    return title or "Untitled"


def _cut_long(lo: float, hi: float, boundaries: list, max_file_s: Optional[float]) -> list:
    """`(start, end)` pieces of `[lo, hi]` no longer than `max_file_s`. A
    piece ends at the clip boundary nearest before `max_file_s` less a minute
    (119 of 120 minutes), or at that time when no clip boundary falls in
    reach."""
    if not max_file_s or hi - lo <= max_file_s:
        return [(lo, hi)]
    limit = max_file_s - min(60.0, max_file_s / 120.0)
    pieces, start = [], lo
    while hi - start > max_file_s:
        cut = max((b for b in boundaries if start < b <= start + limit), default=start + limit)
        pieces.append((start, cut))
        start = cut
    return pieces + [(start, hi)]


def plan_chapters(document, arrangement: Arrangement, split: str, max_file_s: Optional[float] = None) -> ChapterPlan:
    """The files a split export writes. `split` is "subprojects" (one per
    placed nested clip, over its own span, named by its placeholder text) or
    "markers" (one per consecutive marker pair, named by the first marker).
    Names are `NN - <title>`, numbered in time order; a chapter longer than
    `max_file_s` becomes `NN - <title> part 1`, `part 2`... cut at clip
    boundaries. A warning counts the clips that fall inside no chapter."""
    if split == "subprojects":
        spans = [(document.clip_text(p.clip), p.start_s, p.end_s)
                 for p in sorted(arrangement.placed, key=lambda p: p.start_s) if p.clip.is_nested]
    elif split == "markers":
        found = marker_ops.list_markers(document.settings)
        spans = [(a["name"], a["seconds"], b["seconds"]) for a, b in zip(found, found[1:])]
    else:
        raise ValueError(f"unknown split mode {split!r}")
    spans = [span for span in spans if span[2] > span[1]]
    boundaries = sorted({t for p in arrangement.placed for t in (p.start_s, p.end_s)})
    width = max(2, len(str(len(spans))))
    chapters = []
    for number, (title, lo, hi) in enumerate(spans, start=1):
        base = f"{number:0{width}d} - {_chapter_title(title)}"
        tag_title = " ".join(str(title or "").split()) or "Untitled"
        pieces = _cut_long(lo, hi, boundaries, max_file_s)
        for part, piece in enumerate(pieces, start=1):
            if len(pieces) == 1:
                chapters.append(Chapter(base, piece, tag_title))
            else:
                chapters.append(Chapter(f"{base} part {part}", piece, f"{tag_title} part {part}"))
    outside = sum(1 for p in arrangement.placed
                  if not any(p.end_s > lo and p.start_s < hi for _t, lo, hi in spans))
    warnings = []
    if outside:
        warnings.append(f"{outside} clip{'s' if outside != 1 else ''} outside any "
                        f"{SPLIT_NOUNS[split]} weren't exported")
    return ChapterPlan(chapters, warnings)


def mixdown_chapters(document, out_dir: str, plan: ChapterPlan, fmt: str = "wav",
                     numbered: bool = False, progress: Optional[Callable[[float, str], None]] = None,
                     **options) -> ExportResult:
    """`mixdown` once per chapter of `plan`, each file `<chapter name>.<fmt>`
    in `out_dir` (`numbered` gives each the first free `name (2)`), with its
    own SRT, cue sheet and per-clip files when `options` ask for them. With
    `tags` in `options`, each file's title is its chapter's and its track is
    `n/total`; the other fields are shared.
    `options` are `mixdown`'s, minus `out_path` and `range_s`. The result's
    `files` lists every chapter and `audio_path` is the first."""
    ext = (fmt or "wav").lower()
    total = max(1, len(plan.chapters))
    result = ExportResult(audio_path="", warnings=list(plan.warnings))
    for index, chapter in enumerate(plan.chapters):
        path = os.path.join(out_dir, f"{chapter.name}.{ext}")
        if numbered:
            path = unused_path(path)

        def _chapter_progress(fraction: float, detail: str, _index=index) -> None:
            if progress:
                progress((_index + fraction) / total, f"Chapter {_index + 1}/{total}: {detail}")

        file_options = options
        if options.get("tags"):
            file_options = {**options, "tags": {**options["tags"], "title": chapter.title or chapter.name,
                                                 "track": f"{index + 1}/{total}"}}
        one = mixdown(document, path, fmt=fmt, range_s=chapter.range_s, progress=_chapter_progress, **file_options)
        result.files.extend(one.files)
        result.clip_files.extend(one.clip_files)
        result.stem_files.extend(one.stem_files)
        result.text_files.extend(one.text_files)
        result.warnings.extend(w for w in one.warnings if w not in result.warnings)
        result.duration_s += one.duration_s
        result.skipped_clip_ids.extend(c for c in one.skipped_clip_ids if c not in result.skipped_clip_ids)
        result.audio_path = result.audio_path or one.audio_path
    return result


def _padded(samples: np.ndarray, rate: int, head_s: float, tail_s: float) -> np.ndarray:
    """`samples` with `head_s` of silence before and `tail_s` after."""
    tail_shape = samples.shape[1:]
    head = np.zeros((int(round(head_s * rate)),) + tail_shape, dtype=samples.dtype)
    tail = np.zeros((int(round(tail_s * rate)),) + tail_shape, dtype=samples.dtype)
    return np.concatenate([head, samples, tail])


# RMS mode calls the target missed when the limiter took this much off the level.
RMS_TOLERANCE_DB = 0.5


def _normalized(samples: np.ndarray, sample_rate: int, options: dict, companions: list = ()) -> tuple:
    """`(samples, before, after, limited, companions)` for `mixdown`'s
    `loudness` option. `companions` (the stems) get the gain, and the limiter's
    gain in RMS mode, that the mix got; they aren't measured."""
    if options.get("mode", "lufs") == "rms":
        return _normalized_rms(samples, sample_rate, options, companions)
    before = loudness_mod.measure(samples, sample_rate)
    gain_db, limited = loudness_mod.gain_to_target(
        before, float(options["target_lufs"]), float(options.get("ceiling_dbtp", 0.0)))
    if gain_db == 0.0:
        return samples, before, before, limited, list(companions)
    samples = loudness_mod.apply_gain(samples, gain_db)
    return (samples, before, loudness_mod.measure(samples, sample_rate), limited,
            [loudness_mod.apply_gain(c, gain_db) for c in companions])


def _normalized_rms(samples: np.ndarray, sample_rate: int, options: dict, companions: list = ()) -> tuple:
    """RMS mode: the gain that brings the whole-file RMS to `target_rms_dbfs`,
    then the peak limiter at `limiter_dbfs` (none when it's absent). `limited`
    is True when the limiter left the RMS more than `RMS_TOLERANCE_DB` short."""
    before = loudness_mod.measure(samples, sample_rate)
    if not math.isfinite(before.rms_dbfs):
        return samples, before, before, False, list(companions)
    target = float(options["target_rms_dbfs"])
    gain_db = target - before.rms_dbfs
    samples = loudness_mod.apply_gain(samples, gain_db)
    companions = [loudness_mod.apply_gain(c, gain_db) for c in companions]
    if options.get("limiter_dbfs") is not None:
        samples, *companions = limiter.limit_peaks_together(
            samples, companions, sample_rate, float(options["limiter_dbfs"]))
    after = loudness_mod.measure(samples, sample_rate)
    return samples, before, after, after.rms_dbfs < target - RMS_TOLERANCE_DB, companions


def _loaded(placed, samples: np.ndarray, mix: ClipMix, sample_rate: int, range_s) -> "mixer.LoadedClip":
    gain_l, gain_r = mixer.pan_gains(mix.pan)
    automation = mixer.automation_arrays(mix.automation, sample_rate)
    if automation is not None and range_s is not None:
        # Breakpoints are on the unshifted timeline.
        offset = min(range_s) * sample_rate
        automation = (automation[0] - offset, automation[1])
    return mixer.LoadedClip(
        clip_id=placed.clip.id,
        start_frame=int(round(placed.start_s * sample_rate)),
        samples=samples,
        gain=mix.gain,
        gain_l=gain_l,
        gain_r=gain_r,
        fade_in_frames=int(round(mix.fade_in_s * sample_rate)),
        fade_out_frames=int(round(mix.fade_out_s * sample_rate)),
        automation=automation,
        duck=mix.duck,
        sidechain=mix.sidechain,
    )


def duck_db_setting(document) -> float:
    """`document.settings["duck_db"]`, how far a ducked track goes down
    under speech, or `mixer.DEFAULT_DUCK_DB`."""
    try:
        return min(0.0, float((document.settings or {}).get("duck_db", mixer.DEFAULT_DUCK_DB)))
    except (TypeError, ValueError):
        return mixer.DEFAULT_DUCK_DB
