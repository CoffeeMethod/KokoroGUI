"""File tags for an exported mixdown: title, artist, show, track, year,
description, genre, a cover image and, in an mp3, chapter frames.

`write_tags(path, fmt, meta, chapters, cover_path)` runs after the audio is
written. mp3 gets ID3v2.4 (TIT2, TPE1, TALB, TRCK, TDRC, COMM, TCON, an APIC
front cover, and a CHAP frame per chapter under a top-level ordered CTOC),
flac and ogg get Vorbis comments with the cover as a FLAC picture block
(`METADATA_BLOCK_PICTURE` in ogg), wav gets nothing because it has no standard
tag block. Chapters go into mp3 only.

The cover is read through `check_cover`: a JPEG or PNG by its first bytes (not
its extension), at most `MAX_COVER_BYTES`, never on a network path (a UNC path
makes Windows connect to that host just to look at the file). A cover that
fails the check is left out and the reason comes back as a warning; the tags
are still written.

`mutagen` (pure Python) is imported inside the functions, so the app starts
without it and the Export dialog disables the Tags tab (`available`). No Qt.
"""
from __future__ import annotations

import base64
import os
import struct
from typing import Optional

MAX_COVER_BYTES = 5 * 1024 * 1024
TAG_FORMATS = ("mp3", "flac", "ogg")
META_KEYS = ("title", "artist", "album", "track", "year", "description", "genre")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
FRONT_COVER = 3
# Pictures are stored as Picture frames, which carry their pixel size and depth.
_PNG_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


class CoverError(ValueError):
    """The cover image can't be used; the message says why."""


def available() -> bool:
    """True when `mutagen` imports."""
    try:
        import mutagen  # noqa: F401
    except ImportError:
        return False
    return True


def supports(fmt: str) -> bool:
    """True when a file of this format can carry tags."""
    return (fmt or "").lower() in TAG_FORMATS


def _is_network_path(path: str) -> bool:
    r"""A UNC path (`\\server\share\x.png` or `//server/x.png`). The extended
    form `\\?\C:\...` that `realpath` can give a local drive is local."""
    if path[:2] not in ("\\\\", "//"):
        return False
    return path[2:4] not in ("?\\", "?/") or path[5:6] != ":"


def _image_info(data: bytes) -> tuple:
    """`(mime, width, height, depth)` of a JPEG or PNG from its header, or
    `("", 0, 0, 0)` for anything else. Width, height and depth are 0 when the
    header is cut short."""
    if data[:8] == PNG_MAGIC:
        if len(data) >= 26:
            width, height = struct.unpack(">II", data[16:24])
            return "image/png", width, height, data[24] * _PNG_CHANNELS.get(data[25], 1)
        return "image/png", 0, 0, 0
    if data[:3] == b"\xff\xd8\xff":
        position = 2
        while position + 9 < len(data):
            if data[position] != 0xFF:
                break
            marker = data[position + 1]
            if marker == 0xFF:  # padding
                position += 1
                continue
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:  # no length
                position += 2
                continue
            length = struct.unpack(">H", data[position + 2:position + 4])[0]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):  # a frame header
                precision, height, width, components = struct.unpack(">BHHB", data[position + 4:position + 10])
                return "image/jpeg", width, height, precision * components
            position += 2 + length
        return "image/jpeg", 0, 0, 0
    return "", 0, 0, 0


def check_cover(path: str) -> tuple:
    """`(real_path, data, mime, width, height, depth)` for the cover at
    `path`; raises `CoverError` with a sentence when it can't be used."""
    path = (path or "").strip()
    if not path:
        raise CoverError("no cover image chosen")
    if _is_network_path(path):
        raise CoverError("network paths aren't supported; copy the cover locally first")
    real = os.path.realpath(path)
    if _is_network_path(real):
        raise CoverError("network paths aren't supported; copy the cover locally first")
    if not os.path.isfile(real):
        raise CoverError(f"the cover image {os.path.basename(path)} doesn't exist")
    if os.path.getsize(real) > MAX_COVER_BYTES:
        raise CoverError(f"the cover image is over {MAX_COVER_BYTES // (1024 * 1024)} MB")
    with open(real, "rb") as f:
        data = f.read(MAX_COVER_BYTES + 1)
    mime, width, height, depth = _image_info(data)
    if not mime:
        raise CoverError("the cover image must be a JPEG or PNG")
    return real, data, mime, width, height, depth


def cover_problem(path: str) -> Optional[str]:
    """The sentence `check_cover` raises for `path`, or None when it's fine."""
    try:
        check_cover(path)
    except CoverError as e:
        return str(e)
    return None


def _clean(meta: Optional[dict]) -> dict:
    """`meta` with only the known keys, as stripped strings, empty ones out."""
    clean = {}
    for key in META_KEYS:
        value = (meta or {}).get(key)
        text = str(value).strip() if value is not None else ""
        if text:
            clean[key] = text
    return clean


def chapter_spans(chapters, duration_s: Optional[float]) -> list:
    """`[(start_ms, end_ms, title)]` from `[(start_s, title, ...)]`, in time
    order. A chapter ends where the next one starts and the last at
    `duration_s` (when that isn't known, it ends where it starts). A chapter
    that starts at or after the end of the file is dropped."""
    rows = sorted(((max(0.0, float(row[0])), str(row[1] or "").strip()) for row in chapters or ()),
                  key=lambda row: row[0])
    limit = float(duration_s) if duration_s is not None else None
    if limit is not None:
        rows = [row for row in rows if row[0] < limit]
    spans = []
    for index, (start, title) in enumerate(rows):
        if index + 1 < len(rows):
            end = rows[index + 1][0]
        else:
            end = limit if limit is not None else start
        spans.append((int(round(start * 1000)), int(round(max(start, end) * 1000)), title or f"Chapter {index + 1}"))
    return spans


def _write_mp3(path: str, meta: dict, chapters: list, cover: Optional[tuple]) -> None:
    from mutagen import id3
    from mutagen.id3 import ID3

    tags = ID3()  # a fresh tag replaces whatever the file had, so a re-export doesn't stack frames
    text = {"title": id3.TIT2, "artist": id3.TPE1, "album": id3.TALB, "track": id3.TRCK, "year": id3.TDRC,
            "genre": id3.TCON}
    for key, frame in text.items():
        if key in meta:
            tags.add(frame(encoding=id3.Encoding.UTF8, text=[meta[key]]))
    if "description" in meta:
        tags.add(id3.COMM(encoding=id3.Encoding.UTF8, lang="eng", desc="", text=[meta["description"]]))
    if cover is not None:
        _real, data, mime, _w, _h, _d = cover
        tags.add(id3.APIC(encoding=id3.Encoding.UTF8, mime=mime, type=FRONT_COVER, desc="Cover", data=data))
    if chapters:
        ids = [f"chp{number}" for number in range(len(chapters))]
        for element_id, (start, end, title) in zip(ids, chapters):
            tags.add(id3.CHAP(element_id=element_id, start_time=start, end_time=end,
                              start_offset=0xFFFFFFFF, end_offset=0xFFFFFFFF,
                              sub_frames=[id3.TIT2(encoding=id3.Encoding.UTF8, text=[title])]))
        tags.add(id3.CTOC(element_id="toc", flags=id3.CTOCFlags.TOP_LEVEL | id3.CTOCFlags.ORDERED,
                          child_element_ids=ids,
                          sub_frames=[id3.TIT2(encoding=id3.Encoding.UTF8, text=["Chapters"])]))
    tags.save(path, v2_version=4)


def _vorbis_fields(meta: dict) -> dict:
    fields = {"title": "TITLE", "artist": "ARTIST", "album": "ALBUM", "year": "DATE",
              "description": "DESCRIPTION", "genre": "GENRE"}
    comments = {name: [meta[key]] for key, name in fields.items() if key in meta}
    if "track" in meta:
        number, _slash, total = meta["track"].partition("/")
        comments["TRACKNUMBER"] = [number.strip()]
        if total.strip():
            comments["TRACKTOTAL"] = [total.strip()]
    return comments


def _picture(cover: tuple):
    from mutagen.flac import Picture

    _real, data, mime, width, height, depth = cover
    picture = Picture()
    picture.type = FRONT_COVER
    picture.mime = mime
    picture.desc = "Cover"
    picture.data = data
    picture.width, picture.height, picture.depth = width, height, depth
    return picture


def _write_flac(path: str, meta: dict, cover: Optional[tuple]) -> None:
    from mutagen.flac import FLAC

    audio = FLAC(path)
    audio.clear_pictures()
    if audio.tags is None:
        audio.add_tags()
    audio.tags.clear()
    for name, values in _vorbis_fields(meta).items():
        audio[name] = values
    if cover is not None:
        audio.add_picture(_picture(cover))
    audio.save()


def _write_ogg(path: str, meta: dict, cover: Optional[tuple]) -> None:
    from mutagen.oggvorbis import OggVorbis

    audio = OggVorbis(path)
    if audio.tags is None:
        audio.add_tags()
    audio.tags.clear()
    for name, values in _vorbis_fields(meta).items():
        audio[name] = values
    if cover is not None:
        audio["METADATA_BLOCK_PICTURE"] = [base64.b64encode(_picture(cover).write()).decode("ascii")]
    audio.save()


def write_tags(path: str, fmt: str, meta: Optional[dict] = None, chapters: Optional[list] = None,
               cover_path: Optional[str] = None, duration_s: Optional[float] = None) -> list:
    """Writes the tags into the finished audio file at `path` and returns a
    list of warning sentences (an unusable cover, an ignored chapter list).

    `meta` holds any of `META_KEYS` (empty values are skipped; `track` may be
    "3" or "3/12"). `chapters` is `[(start_s, title, ...)]`, written into an
    mp3 only, each ending at the next one's start and the last at
    `duration_s`. `cover_path` goes through `check_cover`. A format with no
    tags (wav) is left alone and returns no warning."""
    fmt = (fmt or "").lower()
    if not supports(fmt):
        return []
    warnings = []
    cover = None
    if cover_path and str(cover_path).strip():
        try:
            cover = check_cover(cover_path)
        except CoverError as e:
            reason = str(e)
            warnings.append(f"{reason[0].upper()}{reason[1:]}, so it was left out of the tags")
    meta = _clean(meta)
    if fmt == "mp3":
        _write_mp3(path, meta, chapter_spans(chapters, duration_s), cover)
    elif fmt == "flac":
        _write_flac(path, meta, cover)
    else:
        _write_ogg(path, meta, cover)
    return warnings
