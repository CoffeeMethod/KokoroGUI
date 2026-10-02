"""Text exports of a clip document: transcripts and chapter lists.

Every function reads the placed clips (`Arrangement`) the audio export was laid
out on, so the times match the file `mixdown` writes. A clip's text goes
through `text_extraction.strip_markup` first, so a `[Name:FX]:` tag or a
`[pause:x]` marker never reaches a transcript, and whitespace collapses to
single spaces. Music beds, subprojects (their text is a placeholder, not
speech) and clips with no audio yet (estimated) get no cue.

Transcripts: `write_vtt` (WebVTT, speakers as `<v Name>`), `write_srt_speakers`
(SRT, `Name: text`), `write_podcast_transcript_json` (the Podcasting 2.0
transcript JSON) and `write_plain_text`. `speakers=False` leaves the names out.

Chapters: `chapter_rows` takes the document's markers, or its subprojects when
no marker falls inside the export; `write_chapters_json` is the Podcasting 2.0
chapters file and `write_show_notes` is a Markdown list with a timestamp per
chapter and the marker's note under it.

`TEXT_EXTRAS` names the six outputs `mixdown(extras=...)` writes, and
`extra_path` the file each gets next to the audio. No Qt here.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

from kokoro_gui.daw import markers as marker_ops
from kokoro_gui.daw.arrangement import Arrangement
from kokoro_gui.engine.text_extraction import strip_markup

TEXT_EXTRAS = ("vtt", "srt_speakers", "transcript_json", "txt", "chapters_json", "show_notes")
# What follows the base name. The speaker SRT has its own, so it never
# replaces the plain `.srt` of "Also write .srt".
EXTRA_SUFFIXES = {
    "vtt": ".vtt",
    "srt_speakers": ".speakers.srt",
    "transcript_json": ".transcript.json",
    "txt": ".txt",
    "chapters_json": ".chapters.json",
    "show_notes": ".show-notes.md",
}
TRANSCRIPT_JSON_VERSION = "1.0.0"
CHAPTERS_JSON_VERSION = "1.2.0"


def clean_extras(values) -> list:
    """`values` as a list of known extras in `TEXT_EXTRAS` order; anything
    else (a hand-edited project.json) is dropped."""
    wanted = set(values) if isinstance(values, (list, tuple, set)) else set()
    return [key for key in TEXT_EXTRAS if key in wanted]


def extra_path(out_dir: str, base: str, key: str) -> str:
    return os.path.join(out_dir, base + EXTRA_SUFFIXES[key])


def format_timestamp(seconds: float, separator: str = ",") -> str:
    """`HH:MM:SS,mmm` (SRT) or, with `separator="."`, `HH:MM:SS.mmm` (WebVTT)."""
    millis = max(0, int(round(float(seconds) * 1000)))
    whole, millis = divmod(millis, 1000)
    minutes, secs = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}"


def _collapse(text: str) -> str:
    return " ".join((text or "").split())


def cue_rows(document, arrangement: Arrangement) -> list:
    """`[(start_s, end_s, speaker, text)]` in timeline order. `speaker` is the
    clip's character name, "" when it has none."""
    rows = []
    for placed in sorted(arrangement.placed, key=lambda p: p.start_s):
        clip = placed.clip
        if placed.estimated or clip.is_bed or clip.is_nested:
            continue
        text = _collapse(strip_markup(document.clip_text(clip)))
        if not text:
            continue
        character = document.get_character(clip.character_id)
        speaker = _collapse(character.name) if character is not None else ""
        rows.append((placed.start_s, placed.end_s, speaker, text))
    return rows


def _write(path: str, text: str) -> str:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def _escape_vtt(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_vtt(document, arrangement: Arrangement, path: str, *, speakers: bool = True) -> str:
    """WebVTT. With `speakers` a named clip's text sits in a voice span,
    `<v Name>text`, which players show as a speaker label."""
    blocks = ["WEBVTT\n"]
    for start, end, speaker, text in cue_rows(document, arrangement):
        body = _escape_vtt(text)
        if speakers and speaker:
            body = f"<v {_escape_vtt(speaker)}>{body}"
        blocks.append(f"{format_timestamp(start, '.')} --> {format_timestamp(end, '.')}\n{body}\n")
    return _write(path, "\n".join(blocks))


def write_srt_speakers(document, arrangement: Arrangement, path: str, *, speakers: bool = True) -> str:
    """SRT with `Name: text` for a named clip when `speakers` is on."""
    blocks = []
    for index, (start, end, speaker, text) in enumerate(cue_rows(document, arrangement), start=1):
        body = f"{speaker}: {text}" if speakers and speaker else text
        blocks.append(f"{index}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{body}\n")
    return _write(path, "\n".join(blocks))


def _seconds(value: float):
    """A time for JSON: milliseconds, a whole number when it is one."""
    value = round(float(value), 3)
    return int(value) if value == int(value) else value


def write_podcast_transcript_json(document, arrangement: Arrangement, path: str, *, speakers: bool = True) -> str:
    """The Podcasting 2.0 transcript JSON: `{"version", "segments": [{"speaker",
    "startTime", "endTime", "body"}]}`. `speaker` is left out of a segment
    with no name, or for every segment when `speakers` is off."""
    segments = []
    for start, end, speaker, text in cue_rows(document, arrangement):
        segment = {}
        if speakers and speaker:
            segment["speaker"] = speaker
        segment.update(startTime=_seconds(start), endTime=_seconds(end), body=text)
        segments.append(segment)
    payload = {"version": TRANSCRIPT_JSON_VERSION, "segments": segments}
    return _write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_plain_text(document, arrangement: Arrangement, path: str, *, speakers: bool = True) -> str:
    """One line per clip, `Name: text`, with a blank line where the speaker
    changes (also when `speakers` is off, so the paragraphs stay)."""
    lines, last = [], None
    for _start, _end, speaker, text in cue_rows(document, arrangement):
        if lines and speaker != last:
            lines.append("")
        lines.append(f"{speaker}: {text}" if speakers and speaker else text)
        last = speaker
    return _write(path, "\n".join(lines) + ("\n" if lines else ""))


def chapter_rows(document, arrangement: Arrangement, range_s: Optional[tuple] = None, head_s: float = 0.0) -> list:
    """`[(start_s, title, note)]` for the file the arrangement was laid out
    for. Markers come first (name as title, their note kept); when none falls
    inside the export, the placed subprojects (their title, no note).

    `arrangement` is the one `mixdown` writes the audio from, already shifted
    to start at 0 when `range_s` was set and later by `head_s`. Markers sit on
    the project's timeline, so they move the same way: a marker before the
    range is dropped, the rest shift by `head_s` less the range's start."""
    rows = []
    found = marker_ops.list_markers(document.settings)
    if range_s is not None:
        lo, hi = sorted((max(0.0, float(range_s[0])), max(0.0, float(range_s[1]))))
        found = [m for m in found if lo <= m["seconds"] < hi]
    else:
        lo = 0.0
    for marker in found:
        rows.append((marker["seconds"] - lo + head_s, _collapse(marker["name"]), marker["note"].strip()))
    if not rows:
        for placed in sorted(arrangement.placed, key=lambda p: p.start_s):
            if placed.clip.is_nested:
                rows.append((placed.start_s, _collapse(document.clip_text(placed.clip)), ""))
    return [(start, title or f"Chapter {number}", note) for number, (start, title, note) in enumerate(rows, start=1)]


def write_chapters_json(document, arrangement: Arrangement, path: str, *, range_s: Optional[tuple] = None,
                        head_s: float = 0.0) -> str:
    """The Podcasting 2.0 chapters file: `{"version", "chapters": [{"startTime",
    "title"}]}`. The list is empty when the export has no marker or subproject."""
    chapters = [{"startTime": _seconds(start), "title": title}
                for start, title, _note in chapter_rows(document, arrangement, range_s, head_s)]
    payload = {"version": CHAPTERS_JSON_VERSION, "chapters": chapters}
    return _write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _clock(seconds: float) -> str:
    """`M:SS`, or `H:MM:SS` from an hour on."""
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02}:{secs:02}" if hours else f"{minutes:02}:{secs:02}"


def write_show_notes(document, arrangement: Arrangement, path: str, *, range_s: Optional[tuple] = None,
                     head_s: float = 0.0) -> str:
    """Markdown, one line per chapter, `- (MM:SS) Title`, with its note
    indented under it."""
    lines = []
    for start, title, note in chapter_rows(document, arrangement, range_s, head_s):
        lines.append(f"- ({_clock(start)}) {title}")
        lines.extend(f"  {line.strip()}" for line in re.split(r"\r?\n", note) if line.strip())
    return _write(path, "\n".join(lines) + ("\n" if lines else ""))


def write_extras(document, arrangement: Arrangement, out_dir: str, base: str, extras, *, speakers: bool = True,
                 range_s: Optional[tuple] = None, head_s: float = 0.0) -> tuple:
    """Writes each of `extras` (keys of `TEXT_EXTRAS`) as `<base><suffix>` in
    `out_dir`. `(paths, warnings)`; a warning says the chapters file or the show
    notes came out empty."""
    paths, empty = [], []
    has_chapters = None
    for key in clean_extras(extras):
        path = extra_path(out_dir, base, key)
        if key == "vtt":
            write_vtt(document, arrangement, path, speakers=speakers)
        elif key == "srt_speakers":
            write_srt_speakers(document, arrangement, path, speakers=speakers)
        elif key == "transcript_json":
            write_podcast_transcript_json(document, arrangement, path, speakers=speakers)
        elif key == "txt":
            write_plain_text(document, arrangement, path, speakers=speakers)
        else:
            if has_chapters is None:
                has_chapters = bool(chapter_rows(document, arrangement, range_s, head_s))
            if not has_chapters:
                empty.append("the chapters file" if key == "chapters_json" else "the show notes")
            if key == "chapters_json":
                write_chapters_json(document, arrangement, path, range_s=range_s, head_s=head_s)
            else:
                write_show_notes(document, arrangement, path, range_s=range_s, head_s=head_s)
        paths.append(path)
    warnings = [f"no markers or subprojects, so {' and '.join(empty)} came out empty"] if empty else []
    return paths, warnings
