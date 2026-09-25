"""Offline export of a clip document (section 6 of
Claude/PLAN_ui_shell_redesign.md, WF8).

`mixdown()` walks `compute_arrangement` (the same placement the timeline
and transport use), reads every clip's segments through the same read-time
post-processing the transport plays (`kokoro_gui.audio.post`, via
`post_config_for_clip`), sums them at their start times with the same
plain-gain-sum rule the live `Transport` applies (grill Q21), and writes one
file. SRT rows come straight from each `PlacedClip`'s
start/duration and the clip's text, replacing `SrtMixin`'s segment-timing
walk for clip documents (that mixin stays for the no-clips whole-document
path).

The mix is stereo with the track controls (`kokoro_gui.daw.mixplan`):
muted and soloed-out tracks are left out, and the fader, pan, fades and
automation apply, so export is what the transport plays. `channels=1`
averages the two channels. `range_s` renders only a region (between two
markers): clips outside it are skipped, and the output, SRT and cue sheet
start at the region's start.

"Keep per-clip files" writes one mono file per clip next to the mixdown,
named `<base>_<index:03>_<character>.<ext>` (UI14), the character name
basename-sanitized like every other name-to-path site in this codebase.

`write_srt(granularity="word")` writes one subtitle per stored word
(`Segment.words`) instead of one per clip. `write_cue_sheet` writes a CSV
row per clip for review and dubbing: timecode (when the document has it
enabled) or seconds, character, source text, text, status, note.

No Qt here, and the audio math is the shared `kokoro_gui.audio.mixer`, so
the whole thing is testable headlessly with a stub document.
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np

from kokoro_gui.audio import mixer, post
from kokoro_gui.daw.arrangement import Arrangement, compute_arrangement, segment_timeline
from kokoro_gui.daw.mixplan import ClipMix, clip_mixes
from kokoro_gui.daw.timecode import format_position

SOUNDFILE_FORMATS = {"wav", "flac", "ogg"}
CUE_SHEET_COLUMNS = ("start", "end", "character", "source_text", "text", "status", "note")


@dataclass
class ExportResult:
    audio_path: str
    srt_path: Optional[str] = None
    clip_files: list = field(default_factory=list)
    duration_s: float = 0.0
    skipped_clip_ids: list = field(default_factory=list)
    cue_sheet_path: Optional[str] = None


def _format_srt_time(seconds: float) -> str:
    millis = int(round((seconds - int(seconds)) * 1000))
    whole = int(seconds)
    if millis == 1000:
        whole += 1
        millis = 0
    minutes, secs = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


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
            if placed.estimated:
                continue
            text = document.clip_text(placed.clip).strip()
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
                document.clip_text(clip).strip(),
                clip.status,
                clip.note,
            ])
    return path


def _safe_component(name: str) -> str:
    """A character name as a filename piece: separators and other unsafe
    characters become "_" first (so "Bo b/ok" keeps both halves), then the
    same `os.path.basename()` guard every other name-to-path site uses."""
    name = re.sub(r'[<>:"/\\|?*\s]+', "_", (name or "").strip())
    name = os.path.basename(name)
    return name or "clip"


def _clip_samples(clip, sample_rate: int, post_config: Optional[dict] = None,
                  nested_audio_path: Optional[Callable] = None) -> Optional[np.ndarray]:
    """All of a clip's segments concatenated at `sample_rate`, post-processed
    per `post_config`, or None if none of them can be read. A segment with a
    `range` contributes only that slice of its file. A nested clip
    (a subproject) is its child's mixdown file, `nested_audio_path(clip)`."""
    if getattr(clip, "source", None) == "nested":
        path = nested_audio_path(clip) if nested_audio_path is not None else None
        if not path:
            return None
        try:
            return mixer.load_clip_samples(path, sample_rate, post_config).astype(np.float32)
        except Exception:
            return None
    parts = []
    for segment in sorted(clip.segments, key=lambda s: s.order_index):
        if not segment.audio_path:
            continue
        try:
            parts.append(mixer.load_clip_samples(segment.audio_path, sample_rate, post_config,
                                                 post.segment_range(segment)))
        except Exception:
            continue
    if not parts:
        return None
    return np.concatenate(parts).astype(np.float32)


def write_audio(path: str, samples: np.ndarray, sample_rate: int, fmt: str) -> None:
    """`samples` is `(frames,)` mono or `(frames, channels)`."""
    fmt = (fmt or "wav").lower()
    if fmt in SOUNDFILE_FORMATS:
        import soundfile as sf

        sf.write(path, samples, sample_rate, format=fmt.upper())
        return
    # mp3 and anything else soundfile can't encode: the same pedalboard
    # AudioFile path ConversionMixin.smart_combine uses.
    from pedalboard.io import AudioFile

    channels = 1 if samples.ndim == 1 else samples.shape[1]
    with AudioFile(path, "w", samplerate=sample_rate, num_channels=channels) as out_f:
        out_f.write(samples.reshape(1, -1) if samples.ndim == 1 else samples.T)


def _in_range(arrangement: Arrangement, range_s: tuple) -> Arrangement:
    """The placed clips overlapping `range_s`, shifted so the range starts
    at 0; the total is the range's length."""
    lo, hi = sorted((max(0.0, float(range_s[0])), max(0.0, float(range_s[1]))))
    placed = [replace(p, start_s=p.start_s - lo) for p in arrangement.placed if p.end_s > lo and p.start_s < hi]
    return Arrangement(placed=placed, total_duration_s=hi - lo)


def mixdown(document, out_path: str, fmt: str = "wav", sample_rate: int = 24000,
            include_srt: bool = False, keep_clip_files: bool = False,
            arrangement: Optional[Arrangement] = None, engine_id: Optional[str] = None,
            progress: Optional[Callable[[float, str], None]] = None,
            post_config_for_clip: Optional[Callable] = None, channels: int = 2,
            range_s: Optional[tuple] = None, srt_granularity: str = "clip",
            include_cue_sheet: bool = False, nested_audio_path: Optional[Callable] = None) -> ExportResult:
    """`post_config_for_clip(clip)` returns the clip's resolved read-time
    post-processing config (the app passes `QtTTSApp.post_config_for_clip`);
    None exports the raw segment files as they are. `nested_audio_path(clip)`
    names a subproject's mixdown file (phase 4)."""
    if arrangement is None:
        arrangement = compute_arrangement(document, engine_id=engine_id)
    mixes = clip_mixes(document, arrangement)
    if range_s is not None:
        arrangement = _in_range(arrangement, range_s)
    sample_rate = int(sample_rate)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(out_path))[0]
    ext = (fmt or "wav").lower()

    loaded: list = []
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
        per_clip.append((index, placed, samples))

    total_frames = int(round(arrangement.total_duration_s * sample_rate))
    if range_s is None:
        total_frames = max(mixer.total_frames(loaded), total_frames)
    if total_frames > 0:
        mixed = mixer.mix_block(loaded, 0, total_frames)
    else:
        mixed = np.zeros((0, mixer.CHANNELS), dtype=np.float32)
    if int(channels) == 1:
        mixed = mixed.mean(axis=1).astype(np.float32)
    if progress:
        progress(0.85, "Writing mixdown")
    write_audio(out_path, mixed, sample_rate, ext)

    result = ExportResult(audio_path=out_path, duration_s=total_frames / float(sample_rate), skipped_clip_ids=skipped)

    if keep_clip_files:
        for index, placed, samples in per_clip:
            character = document.get_character(placed.clip.character_id)
            who = _safe_component(character.name if character is not None else "clip")
            clip_path = os.path.join(out_dir, f"{base}_{index + 1:03d}_{who}.{ext}")
            write_audio(clip_path, samples, sample_rate, ext)
            result.clip_files.append(clip_path)

    if include_srt:
        srt_path = os.path.join(out_dir, f"{base}.srt")
        result.srt_path = write_srt(document, arrangement, srt_path, granularity=srt_granularity)

    if include_cue_sheet:
        result.cue_sheet_path = write_cue_sheet(document, arrangement, os.path.join(out_dir, f"{base}.csv"))

    if progress:
        progress(1.0, "Export finished")
    return result


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
    )
