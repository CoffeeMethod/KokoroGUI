"""M4B export: one AAC audiobook file with chapter markers, tags and a cover.

pedalboard and soundfile can't write M4B, so `write_m4b` hands the finished
mix to an `ffmpeg` it finds on PATH (`find_ffmpeg`; the app doesn't ship one,
grill PG5). It writes the mix to a temporary WAV, the tags and chapters to an
FFMETADATA1 file (`metadata_text`) and runs

    ffmpeg -i mix.wav -i metadata.txt [-i cover] ... -f ipod out.m4b

with an argument list, never a shell. Everything temporary sits in the output
folder and is removed afterwards, whether ffmpeg succeeded or not, and the
M4B itself is written under a temporary name and moved into place so a failed
run never leaves a half file where the export was meant to go.

`m4b_chapters` picks the chapters: the placed subprojects (the book's
chapters), else the markers. A file with neither gets none. Tags and the
cover come from ffmpeg too, so an M4B needs no `mutagen`. The cover goes
through `tagging.check_cover`, the same gate every other format uses.

No Qt. Tests mock `subprocess.run`; one optional test runs a real ffmpeg.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np

from kokoro_gui.daw import tagging, transcripts
from kokoro_gui.daw.arrangement import Arrangement

BITRATES_KBPS = (64, 96, 128)
DEFAULT_BITRATE_KBPS = 96
MIN_TIMEOUT_S = 120.0
STDERR_LINES = 6
# Tag key to the FFMETADATA key ffmpeg's ipod muxer writes as the matching atom.
META_FIELDS = {"title": "title", "artist": "artist", "album": "album", "track": "track", "year": "date",
               "description": "description", "genre": "genre"}
_ESCAPED = re.compile(r"([=;#\\\n])")


class M4bError(RuntimeError):
    """ffmpeg is missing or failed; the message says what it reported."""


def find_ffmpeg() -> Optional[str]:
    """The `ffmpeg` on PATH, or None. The Export dialog hides M4B on None."""
    return shutil.which("ffmpeg")


def escape(value) -> str:
    """`value` as FFMETADATA text: `=`, `;`, `#`, `\\` and a newline each get
    a backslash in front."""
    return _ESCAPED.sub(r"\\\1", str(value if value is not None else ""))


def metadata_text(meta: Optional[dict], chapters: Optional[list], duration_s: Optional[float]) -> str:
    """The FFMETADATA1 file for the tags in `meta` (any of `tagging.META_KEYS`)
    and `chapters` (`[(start_s, title, ...)]`, each ending where the next
    starts and the last at `duration_s`, in milliseconds)."""
    lines = [";FFMETADATA1"]
    clean = tagging._clean(meta)
    for key, name in META_FIELDS.items():
        if key in clean:
            lines.append(f"{name}={escape(clean[key])}")
    for start, end, title in tagging.chapter_spans(chapters, duration_s):
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={start}", f"END={end}",
                  f"title={escape(' '.join(title.split()))}"]
    return "\n".join(lines) + "\n"


def m4b_chapters(document, arrangement: Arrangement, range_s: Optional[tuple] = None,
                 head_s: float = 0.0) -> list:
    """`[(start_s, title, note)]` for an M4B written from `arrangement` (the
    shifted one `mixdown` uses): the placed subprojects, each titled by its
    text; with none, the markers (`transcripts.chapter_rows`)."""
    rows = []
    for placed in sorted(arrangement.placed, key=lambda p: p.start_s):
        if placed.clip.is_nested:
            title = " ".join(document.clip_text(placed.clip).split())
            rows.append((placed.start_s, title or f"Chapter {len(rows) + 1}", ""))
    return rows or transcripts.chapter_rows(document, arrangement, range_s, head_s)


def ffmpeg_args(ffmpeg: str, wav: str, meta_file: str, out_path: str, bitrate_kbps: int,
                cover: Optional[str] = None) -> list:
    """The ffmpeg command line: AAC audio from the WAV, tags and chapters from
    the metadata file and, with `cover`, that image as the attached picture."""
    args = [ffmpeg, "-hide_banner", "-nostdin", "-y", "-i", wav, "-i", meta_file]
    if cover:
        args += ["-i", cover]
    args += ["-map", "0:a"]
    if cover:
        args += ["-map", "2:v", "-c:v", "copy", "-disposition:v:0", "attached_pic"]
    args += ["-map_metadata", "1", "-map_chapters", "1", "-c:a", "aac", "-b:a", f"{int(bitrate_kbps)}k",
             "-movflags", "+faststart", "-f", "ipod", out_path]
    return args


def _tail(text: bytes | str | None) -> str:
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return " | ".join(lines[-STDERR_LINES:])


def write_m4b(samples: np.ndarray, rate: int, path: str, bitrate_kbps: Optional[int] = None,
              chapters: Optional[list] = None, meta: Optional[dict] = None,
              cover_path: Optional[str] = None) -> list:
    """Encodes `samples` (`(frames,)` or `(frames, channels)` floats) as an
    M4B at `path` and returns warning sentences (an unusable cover).

    Raises `M4bError` when ffmpeg isn't on PATH or exits with an error; the
    message ends with its last lines of output."""
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise M4bError("ffmpeg isn't on PATH, so the M4B couldn't be written")
    import soundfile as sf

    warnings = []
    cover = None
    if cover_path and str(cover_path).strip():
        try:
            cover = tagging.check_cover(cover_path)[0]
        except tagging.CoverError as e:
            reason = str(e)
            warnings.append(f"{reason[0].upper()}{reason[1:]}, so it was left out of the M4B")
    kbps = int(bitrate_kbps) if bitrate_kbps else DEFAULT_BITRATE_KBPS
    duration_s = len(samples) / float(rate)
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    temps = []

    def _temp(suffix: str) -> str:
        handle, name = tempfile.mkstemp(prefix=".m4b-", suffix=suffix, dir=out_dir)
        os.close(handle)
        temps.append(name)
        return name

    try:
        wav, meta_file, partial = _temp(".wav"), _temp(".txt"), _temp(".m4b")
        sf.write(wav, np.clip(samples, -1.0, 1.0), int(rate), format="WAV")
        with open(meta_file, "w", encoding="utf-8", newline="\n") as f:
            f.write(metadata_text(meta, chapters, duration_s))
        args = ffmpeg_args(ffmpeg, wav, meta_file, partial, kbps, cover)
        timeout = max(MIN_TIMEOUT_S, 60.0 + duration_s)  # AAC encodes far faster than real time
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if hasattr(subprocess, "CREATE_NO_WINDOW") else {}
        try:
            subprocess.run(args, check=True, capture_output=True, timeout=timeout, **kwargs)
        except subprocess.CalledProcessError as e:
            raise M4bError(f"ffmpeg failed (exit {e.returncode}): {_tail(e.stderr)}") from e
        except subprocess.TimeoutExpired as e:
            raise M4bError(f"ffmpeg took longer than {int(timeout)} s and was stopped: {_tail(e.stderr)}") from e
        except OSError as e:
            raise M4bError(f"ffmpeg couldn't be started: {e}") from e
        os.replace(partial, path)
        temps.remove(partial)
    finally:
        for name in temps:
            try:
                os.remove(name)
            except OSError:
                pass
    return warnings
