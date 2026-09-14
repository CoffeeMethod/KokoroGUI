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

"Keep per-clip files" writes one file per clip next to the mixdown, named
`<base>_<index:03>_<character>.<ext>` (UI14), the character name
basename-sanitized like every other name-to-path site in this codebase.

No Qt here, and the audio math is the shared `kokoro_gui.audio.mixer`, so
the whole thing is testable headlessly with a stub document.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from kokoro_gui.audio import mixer
from kokoro_gui.daw.arrangement import Arrangement, compute_arrangement

SOUNDFILE_FORMATS = {"wav", "flac", "ogg"}


@dataclass
class ExportResult:
    audio_path: str
    srt_path: Optional[str] = None
    clip_files: list = field(default_factory=list)
    duration_s: float = 0.0
    skipped_clip_ids: list = field(default_factory=list)


def _format_srt_time(seconds: float) -> str:
    millis = int(round((seconds - int(seconds)) * 1000))
    whole = int(seconds)
    if millis == 1000:
        whole += 1
        millis = 0
    minutes, secs = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def write_srt(document, arrangement: Arrangement, path: str) -> str:
    rows = []
    index = 1
    for placed in arrangement.placed:
        if placed.estimated:
            continue
        text = document.clip_text(placed.clip).strip()
        if not text:
            continue
        rows.append(f"{index}\n{_format_srt_time(placed.start_s)} --> {_format_srt_time(placed.end_s)}\n{text}\n")
        index += 1
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
        if rows:
            f.write("\n")
    return path


def _safe_component(name: str) -> str:
    """A character name as a filename piece: separators and other unsafe
    characters become "_" first (so "Bo b/ok" keeps both halves), then the
    same `os.path.basename()` guard every other name-to-path site uses."""
    name = re.sub(r'[<>:"/\\|?*\s]+', "_", (name or "").strip())
    name = os.path.basename(name)
    return name or "clip"


def _clip_samples(clip, sample_rate: int, post_config: Optional[dict] = None) -> Optional[np.ndarray]:
    """All of a clip's segments concatenated at `sample_rate`, post-processed
    per `post_config`, or None if none of them can be read."""
    parts = []
    for segment in sorted(clip.segments, key=lambda s: s.order_index):
        if not segment.audio_path:
            continue
        try:
            parts.append(mixer.load_clip_samples(segment.audio_path, sample_rate, post_config))
        except Exception:
            continue
    if not parts:
        return None
    return np.concatenate(parts).astype(np.float32)


def write_audio(path: str, samples: np.ndarray, sample_rate: int, fmt: str) -> None:
    fmt = (fmt or "wav").lower()
    if fmt in SOUNDFILE_FORMATS:
        import soundfile as sf

        sf.write(path, samples, sample_rate, format=fmt.upper())
        return
    # mp3 and anything else soundfile can't encode: the same pedalboard
    # AudioFile path ConversionMixin.smart_combine uses.
    from pedalboard.io import AudioFile

    with AudioFile(path, "w", samplerate=sample_rate, num_channels=1) as out_f:
        out_f.write(samples.reshape(1, -1))


def mixdown(document, out_path: str, fmt: str = "wav", sample_rate: int = 24000,
            include_srt: bool = False, keep_clip_files: bool = False,
            arrangement: Optional[Arrangement] = None, engine_id: Optional[str] = None,
            progress: Optional[Callable[[float, str], None]] = None,
            post_config_for_clip: Optional[Callable] = None) -> ExportResult:
    """`post_config_for_clip(clip)` returns the clip's resolved read-time
    post-processing config (the app passes `QtTTSApp.post_config_for_clip`);
    None exports the raw segment files as they are."""
    if arrangement is None:
        arrangement = compute_arrangement(document, engine_id=engine_id)
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
        post_config = post_config_for_clip(placed.clip) if post_config_for_clip else None
        samples = _clip_samples(placed.clip, sample_rate, post_config)
        if samples is None:
            skipped.append(placed.clip.id)
            continue
        loaded.append(mixer.LoadedClip(
            clip_id=placed.clip.id,
            start_frame=int(round(placed.start_s * sample_rate)),
            samples=samples,
        ))
        per_clip.append((index, placed, samples))

    total_frames = max(mixer.total_frames(loaded), int(round(arrangement.total_duration_s * sample_rate)))
    mixed = mixer.mix_block(loaded, 0, total_frames) if total_frames > 0 else np.zeros(0, dtype=np.float32)
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
        result.srt_path = write_srt(document, arrangement, srt_path)

    if progress:
        progress(1.0, "Export finished")
    return result
