"""Tests for kokoro_gui/daw/tagging.py: the tags, cover and ID3 chapters written
into a finished mp3, flac or ogg. Real files in tmp_path through `write_audio`,
read back with mutagen. No Qt, no engine."""
import base64
import struct
import zlib

import numpy as np
import pytest

mutagen = pytest.importorskip("mutagen")  # a pinned requirement; CI installs it

from mutagen.flac import FLAC, Picture  # noqa: E402
from mutagen.id3 import ID3  # noqa: E402
from mutagen.oggvorbis import OggVorbis  # noqa: E402

from kokoro_gui.daw import tagging  # noqa: E402
from kokoro_gui.daw.mixdown import write_audio  # noqa: E402

RATE = 24000
META = {"title": "Episode One", "artist": "The Cast", "album": "The Show", "track": "3", "year": "2026",
        "description": "First line.\nSecond line.", "genre": "Podcast"}


def _png(width=4, height=3) -> bytes:
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    rows = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    return (tagging.PNG_MAGIC + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def _jpeg(width=640, height=480) -> bytes:
    """A JPEG header (SOI, APP0, SOF0) is all the size reader needs."""
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 3) + b"\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    return b"\xff\xd8" + app0 + sof + b"\xff\xd9"


def _audio(tmp_path, fmt, seconds=2.0):
    path = str(tmp_path / f"mix.{fmt}")
    samples = (np.sin(np.arange(int(RATE * seconds)) * 0.05) * 0.3).astype(np.float32)
    write_audio(path, samples, RATE, fmt)
    return path


def _cover(tmp_path, name="cover.png", data=None):
    path = tmp_path / name
    path.write_bytes(_png() if data is None else data)
    return str(path)


# -- mp3 -----------------------------------------------------------------------------------------


def test_an_mp3_gets_every_field_and_the_cover(tmp_path):
    path = _audio(tmp_path, "mp3")
    cover = _cover(tmp_path)

    assert tagging.write_tags(path, "mp3", META, None, cover) == []

    tags = ID3(path)
    assert str(tags["TIT2"]) == "Episode One"
    assert str(tags["TPE1"]) == "The Cast"
    assert str(tags["TALB"]) == "The Show"
    assert str(tags["TRCK"]) == "3"
    assert str(tags["TDRC"]) == "2026"
    assert str(tags["TCON"]) == "Podcast"
    assert tags.getall("COMM")[0].text == ["First line.\nSecond line."]
    picture = tags.getall("APIC")[0]
    assert (picture.mime, picture.type, picture.data) == ("image/png", 3, _png())
    assert not tags.getall("CHAP") and not tags.getall("CTOC")


def test_the_tagged_mp3_still_decodes_with_its_length(tmp_path):
    path = _audio(tmp_path, "mp3")

    tagging.write_tags(path, "mp3", META, None, _cover(tmp_path))

    info = mutagen.File(path).info
    assert info.length == pytest.approx(2.0, abs=0.1)


def test_chapters_become_chap_frames_under_an_ordered_ctoc(tmp_path):
    path = _audio(tmp_path, "mp3")

    tagging.write_tags(path, "mp3", META, [(0.0, "Intro", ""), (65.5 / 100, "Middle", "note"), (1.25, "End", "")],
                       None, duration_s=2.0)

    tags = ID3(path)
    chapters = {f.element_id: (f.start_time, f.end_time, str(f.sub_frames["TIT2"])) for f in tags.getall("CHAP")}
    assert chapters == {"chp0": (0, 655, "Intro"), "chp1": (655, 1250, "Middle"), "chp2": (1250, 2000, "End")}
    toc = tags.getall("CTOC")[0]
    assert toc.child_element_ids == ["chp0", "chp1", "chp2"]
    assert toc.flags & 1 and toc.flags & 2  # top-level and ordered


def test_chapters_are_sorted_and_one_past_the_end_is_dropped(tmp_path):
    spans = tagging.chapter_spans([(5.0, "Late"), (1.0, "Early"), (12.0, "Past the end")], duration_s=10.0)

    assert spans == [(1000, 5000, "Early"), (5000, 10000, "Late")]


def test_an_untitled_chapter_is_numbered():
    assert tagging.chapter_spans([(0.0, ""), (1.0, "  ")], duration_s=2.0) == [
        (0, 1000, "Chapter 1"), (1000, 2000, "Chapter 2")]


def test_tagging_twice_replaces_the_first_tags(tmp_path):
    path = _audio(tmp_path, "mp3")
    tagging.write_tags(path, "mp3", META, [(0.0, "A"), (1.0, "B")], _cover(tmp_path), duration_s=2.0)

    tagging.write_tags(path, "mp3", {"title": "Second"}, [(0.0, "Only")], None, duration_s=2.0)

    tags = ID3(path)
    assert str(tags["TIT2"]) == "Second"
    assert not tags.getall("TPE1") and not tags.getall("APIC")
    assert [f.element_id for f in tags.getall("CHAP")] == ["chp0"]


def test_empty_fields_are_left_out(tmp_path):
    path = _audio(tmp_path, "mp3")

    tagging.write_tags(path, "mp3", {"title": "Only a title", "artist": "  ", "album": None, "bogus": "x"})

    tags = ID3(path)
    assert sorted(tags.keys()) == ["TIT2"]


# -- flac and ogg --------------------------------------------------------------------------------


def test_a_flac_gets_vorbis_comments_and_a_picture_block(tmp_path):
    path = _audio(tmp_path, "flac")

    assert tagging.write_tags(path, "flac", {**META, "track": "3/12"}, [(0.0, "ignored")], _cover(tmp_path)) == []

    audio = FLAC(path)
    assert audio["TITLE"] == ["Episode One"] and audio["ARTIST"] == ["The Cast"] and audio["ALBUM"] == ["The Show"]
    assert audio["DATE"] == ["2026"] and audio["GENRE"] == ["Podcast"]
    assert audio["DESCRIPTION"] == ["First line.\nSecond line."]
    assert audio["TRACKNUMBER"] == ["3"] and audio["TRACKTOTAL"] == ["12"]
    assert len(audio.pictures) == 1
    picture = audio.pictures[0]
    assert (picture.mime, picture.type, picture.width, picture.height, picture.data) == (
        "image/png", 3, 4, 3, _png())
    assert not [k for k in audio if k.upper().startswith("CHAP")]  # chapters are mp3 only


def test_an_ogg_gets_comments_and_a_base64_picture(tmp_path):
    path = _audio(tmp_path, "ogg")

    assert tagging.write_tags(path, "ogg", META, None, _cover(tmp_path, "c.jpg", _jpeg())) == []

    audio = OggVorbis(path)
    assert audio["TITLE"] == ["Episode One"] and audio["TRACKNUMBER"] == ["3"] and "TRACKTOTAL" not in audio
    picture = Picture(base64.b64decode(audio["METADATA_BLOCK_PICTURE"][0]))
    assert (picture.mime, picture.type, picture.width, picture.height) == ("image/jpeg", 3, 640, 480)
    assert picture.data == _jpeg()


def test_a_wav_is_left_alone_with_no_warning(tmp_path):
    path = _audio(tmp_path, "wav")
    before = open(path, "rb").read()

    assert tagging.write_tags(path, "wav", META, [(0.0, "A")], _cover(tmp_path)) == []

    assert open(path, "rb").read() == before
    assert not tagging.supports("wav") and tagging.supports("MP3")


# -- the cover -----------------------------------------------------------------------------------


def test_a_gif_cover_is_refused_with_a_warning_and_the_tags_are_still_written(tmp_path):
    path = _audio(tmp_path, "mp3")
    gif = _cover(tmp_path, "cover.gif", b"GIF89a\x01\x00\x01\x00\x00\x00\x00;")

    warnings = tagging.write_tags(path, "mp3", META, None, gif)

    assert len(warnings) == 1 and "JPEG or PNG" in warnings[0] and warnings[0][0].isupper()
    tags = ID3(path)
    assert str(tags["TIT2"]) == "Episode One" and not tags.getall("APIC")


def test_the_type_comes_from_the_bytes_not_the_extension(tmp_path):
    path = _audio(tmp_path, "mp3")

    assert tagging.write_tags(path, "mp3", META, None, _cover(tmp_path, "really-a-png.jpg")) == []
    assert ID3(path).getall("APIC")[0].mime == "image/png"

    assert tagging.write_tags(path, "mp3", META, None, _cover(tmp_path, "text.png", b"not an image")) != []
    assert not ID3(path).getall("APIC")


def test_a_missing_cover_is_ignored_with_a_warning(tmp_path):
    path = _audio(tmp_path, "flac")

    warnings = tagging.write_tags(path, "flac", META, None, str(tmp_path / "gone.png"))

    assert len(warnings) == 1 and "gone.png" in warnings[0]
    audio = FLAC(path)
    assert audio["TITLE"] == ["Episode One"] and not audio.pictures


def test_a_cover_over_the_size_limit_is_refused(tmp_path):
    big = _cover(tmp_path, "big.png", tagging.PNG_MAGIC + b"\x00" * tagging.MAX_COVER_BYTES)

    assert "over 5 MB" in tagging.cover_problem(big)


def test_a_network_cover_path_is_refused_before_it_is_touched(tmp_path, monkeypatch):
    import os

    def _boom(*_args, **_kwargs):
        raise AssertionError("a UNC path must not reach the filesystem")

    monkeypatch.setattr(os.path, "realpath", _boom)
    monkeypatch.setattr(os.path, "isfile", _boom)

    for path in ("\\\\server\\share\\cover.png", "//server/share/cover.png"):
        assert "network" in tagging.cover_problem(path)


def test_a_fine_cover_has_no_problem_and_a_blank_path_is_not_one_to_embed(tmp_path):
    assert tagging.cover_problem(_cover(tmp_path)) is None
    assert tagging.cover_problem(_cover(tmp_path, "c.jpg", _jpeg())) is None
    assert tagging.cover_problem("") is not None


def test_image_sizes_come_from_the_headers():
    assert tagging._image_info(_png(7, 5)) == ("image/png", 7, 5, 24)
    assert tagging._image_info(_jpeg(100, 50)) == ("image/jpeg", 100, 50, 24)
    assert tagging._image_info(tagging.PNG_MAGIC) == ("image/png", 0, 0, 0)
    assert tagging._image_info(b"\xff\xd8\xff\xe0") == ("image/jpeg", 0, 0, 0)
    assert tagging._image_info(b"") == ("", 0, 0, 0)
