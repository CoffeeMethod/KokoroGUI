"""Subtitle files as a list of timed cues (phase 5 D2).

`parse(path)` reads an `.srt`, `.vtt`, `.ass` or `.ssa` file and returns its
cues in start order; `parse_text(text, fmt)` does the same for a string.
Qt-free like the rest of `kokoro_gui/daw/`: File > Import Subtitles (dubbing)
and the caption-file path of Import Recording (podcast) both read cues
through here.

What every format comes back as:

- `Cue.text` is the cue's lines with markup removed, each line stripped,
  empty lines dropped, joined with a newline (the same thing ASS writes as
  `\\N`). A caller that wants one line (the transcript does) joins them
  itself. A cue whose text is empty after that is dropped: it has nothing to
  speak or align.
- `Cue.speaker` is the ASS `Name` column or the first WebVTT `<v Name>`
  voice tag, stripped, or None. SRT has no speaker field.
- Cues are sorted by start time (stable, so equal starts keep file order).
  Overlapping cues are kept as they are. An end before its start is raised
  to the start.

Encoding: UTF-8 with or without a BOM, UTF-16 with a BOM, else latin-1
(which decodes any byte, so a file never fails on encoding). Newlines may
be LF, CRLF or CR.
"""
from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass
from typing import Optional

SUBTITLE_FORMATS = ("srt", "vtt", "ass", "ssa")
# The file dialog filter, and what `format_for_path` accepts.
SUBTITLE_EXTENSIONS = tuple("." + fmt for fmt in SUBTITLE_FORMATS)
# A subtitle file is text; anything this big is not one.
MAX_SUBTITLE_BYTES = 32 * 1024 * 1024


class SubtitleError(ValueError):
    """A file that can't be read as subtitles: unknown extension, too big,
    not a file."""


@dataclass(frozen=True)
class Cue:
    start_s: float
    end_s: float
    text: str
    speaker: Optional[str] = None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


# -- reading -------------------------------------------------------------------


def format_for_path(path: str) -> str:
    """The format name ("srt", "vtt", "ass", "ssa") from the extension."""
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext not in SUBTITLE_FORMATS:
        raise SubtitleError(f"Not a subtitle file (.srt, .vtt, .ass, .ssa): {os.path.basename(path)}")
    return ext


def decode_bytes(data: bytes) -> str:
    """UTF-8 (a BOM is dropped), UTF-16 when it starts with a UTF-16 BOM,
    else latin-1."""
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="replace")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def read_text(path: str) -> str:
    """The decoded contents of the subtitle file at `path`. The path comes
    from a file dialog; it is normalised with realpath before it is opened."""
    source = os.path.realpath(os.path.abspath(path))
    drive = os.path.splitdrive(source)[0]
    if not source.startswith(drive + os.sep) or not os.path.isfile(source):
        raise SubtitleError(f"Not a file: {path}")
    size = os.path.getsize(source)
    if size > MAX_SUBTITLE_BYTES:
        raise SubtitleError(f"{os.path.basename(source)} is {size / 1024 ** 2:.0f} MB, too big for a subtitle file.")
    with open(source, "rb") as f:
        return decode_bytes(f.read())


def parse(path: str) -> list:
    """The cues of the `.srt`/`.vtt`/`.ass`/`.ssa` file at `path`, in start
    order. Raises `SubtitleError` for an unknown extension or a file that
    can't be read as one, `OSError` when the read fails."""
    fmt = format_for_path(path)
    return parse_text(read_text(path), fmt)


def parse_text(text: str, fmt: str) -> list:
    """The cues in `text`, a whole subtitle file as a string; `fmt` is
    "srt", "vtt", "ass" or "ssa" (a leading dot and case are ignored)."""
    fmt = (fmt or "").lower().lstrip(".")
    if fmt not in SUBTITLE_FORMATS:
        raise SubtitleError(f"Unknown subtitle format: {fmt!r}")
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    if fmt == "srt":
        cues = _parse_srt(text)
    elif fmt == "vtt":
        cues = _parse_vtt(text)
    else:
        cues = _parse_ass(text)
    return sorted(cues, key=lambda cue: cue.start_s)


# -- shared helpers ----------------------------------------------------------------

# H:MM:SS with a comma or dot fraction; hours optional (WebVTT allows MM:SS.mmm).
_TIME = r"(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:[,.](\d+))?"
_TIME_RE = re.compile(_TIME)
_TIMING_RE = re.compile(r"^\s*(" + _TIME + r")\s*-->\s*(" + _TIME + r")")
_BLOCK_SPLIT = re.compile(r"\n[ \t]*\n")
_TAG_RE = re.compile(r"<[^>]*>")
_ASS_OVERRIDE_RE = re.compile(r"\{[^}]*\}")


def _seconds(stamp: str) -> float:
    """`HH:MM:SS,mmm`, `MM:SS.mmm` or ASS's `H:MM:SS.cc` as seconds. The
    fraction is read as a decimal fraction, so `.5`, `.50` and `.500` agree."""
    match = _TIME_RE.fullmatch(stamp.strip())
    if match is None:
        raise SubtitleError(f"Bad timestamp: {stamp!r}")
    hours, minutes, seconds, fraction = match.groups()
    total = int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)
    if fraction:
        total += int(fraction) / (10 ** len(fraction))
    return float(total)


def _cue(start_s: float, end_s: float, lines, speaker: Optional[str]) -> Optional[Cue]:
    text = "\n".join(line.strip() for line in lines if line.strip())
    if not text:
        return None
    speaker = (speaker or "").strip() or None
    return Cue(start_s=start_s, end_s=max(end_s, start_s), text=text, speaker=speaker)


def _timed_blocks(text: str):
    """Yields `(start_s, end_s, text_lines, lines_before_timing)` for every
    blank-line separated block that has a `-->` timing line. The timing
    line's tail (WebVTT cue settings, SRT position hints) is dropped."""
    for block in _BLOCK_SPLIT.split(text.strip("\n")):
        lines = block.split("\n")
        index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if index is None:
            continue
        match = _TIMING_RE.match(lines[index])
        if match is None:
            continue
        start = _seconds(match.group(1))
        end = _seconds(match.group(6))
        yield start, end, lines[index + 1:], lines[:index]


# -- SubRip ---------------------------------------------------------------------------


def _parse_srt(text: str) -> list:
    """An optional index line, `HH:MM:SS,mmm --> HH:MM:SS,mmm`, the text
    lines, a blank line. HTML-style tags (`<i>`, `<font ...>`) and the ASS
    override blocks some files carry (`{\\an8}`) are removed."""
    cues = []
    for start, end, lines, _before in _timed_blocks(text):
        cleaned = [_ASS_OVERRIDE_RE.sub("", _TAG_RE.sub("", line)) for line in lines]
        cue = _cue(start, end, cleaned, None)
        if cue is not None:
            cues.append(cue)
    return cues


# -- WebVTT ---------------------------------------------------------------------------

_VOICE_RE = re.compile(r"<v(?:\.[^\s>]*)?\s+([^>]*)>")
_VTT_SKIPPED_BLOCKS = ("NOTE", "STYLE", "REGION")


def _parse_vtt(text: str) -> list:
    """SRT's grammar plus a `WEBVTT` header block, optional cue identifiers,
    cue settings after the end time, and NOTE/STYLE/REGION blocks (skipped).
    The first `<v Name>` voice tag gives the speaker; every tag is removed
    and entities (`&amp;`) are decoded."""
    cues = []
    for start, end, lines, before in _timed_blocks(text):
        first = before[0].strip() if before else ""
        if first.startswith("WEBVTT") or first.split(" ", 1)[0] in _VTT_SKIPPED_BLOCKS:
            # A timing line inside the header or a NOTE block isn't a cue.
            continue
        speaker = None
        cleaned = []
        for line in lines:
            if speaker is None:
                voice = _VOICE_RE.search(line)
                if voice is not None:
                    speaker = html.unescape(voice.group(1))
            cleaned.append(html.unescape(_TAG_RE.sub("", line)))
        cue = _cue(start, end, cleaned, speaker)
        if cue is not None:
            cues.append(cue)
    return cues


# -- Advanced SubStation Alpha ----------------------------------------------------

_ASS_DEFAULT_FORMAT = ("layer", "start", "end", "style", "name", "marginl", "marginr", "marginv", "effect", "text")


def _ass_text(raw: str) -> list:
    """An ASS `Text` column as lines: `{...}` override blocks removed, `\\N`
    and `\\n` as line breaks, `\\h` as a space."""
    raw = _ASS_OVERRIDE_RE.sub("", raw)
    raw = raw.replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " ")
    return raw.split("\n")


def _parse_ass(text: str) -> list:
    """The `Dialogue:` lines of the `[Events]` section. Its `Format:` line
    names the columns (the ASS default when it has none); `Text` is the last
    column and may hold commas; `Name` is the speaker. `Comment:` lines and
    every other section are ignored."""
    cues = []
    in_events = False
    columns = _ASS_DEFAULT_FORMAT
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_events = stripped.lower() == "[events]"
            continue
        if not in_events or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip().lower()
        if key == "format":
            names = tuple(name.strip().lower() for name in value.split(","))
            if "start" in names and "end" in names and "text" in names:
                columns = names
            continue
        if key != "dialogue":
            continue
        fields = value.lstrip().split(",", len(columns) - 1)
        if len(fields) < len(columns):
            continue
        row = dict(zip(columns, fields))
        try:
            start = _seconds(row["start"])
            end = _seconds(row["end"])
        except SubtitleError:
            continue
        cue = _cue(start, end, _ass_text(row["text"]), row.get("name"))
        if cue is not None:
            cues.append(cue)
    return cues
