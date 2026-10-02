"""Tests for kokoro_gui/daw/m4b.py and `mixdown(fmt="m4b")`: the ffmpeg metadata
file, the argument list, the temporary files and the chapters. The subprocess is
mocked, so none of it needs a real ffmpeg; one round trip at the end runs when
ffmpeg is on PATH. No Qt, no engine."""
import os
import struct
import subprocess
import zlib

import numpy as np
import pytest
import soundfile as sf

from kokoro_gui.daw import m4b, markers as marker_ops, tagging
from kokoro_gui.daw.arrangement import compute_arrangement
from kokoro_gui.daw.mixdown import mixdown, mixdown_chapters, plan_chapters, write_audio
from kokoro_gui.daw.models import Character, Document, Segment

RATE = 8000
FAKE = "C:/tools/ffmpeg.exe"


def _png(width=4, height=3) -> bytes:
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    rows = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    return (tagging.PNG_MAGIC + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def _tone(seconds=2.0, rate=RATE):
    return (np.sin(np.arange(int(rate * seconds)) * 0.05) * 0.3).astype(np.float32)


class FakeFfmpeg:
    """Stands in for `subprocess.run`: records the command and the temporary
    files as ffmpeg would see them, then writes the output file."""

    def __init__(self, fail=None):
        self.calls, self.kwargs, self.seen, self.fail = [], [], {}, fail

    def __call__(self, args, **kwargs):
        self.calls.append(list(args))
        self.kwargs.append(kwargs)
        wav, meta = args[args.index("-i") + 1], args[args.index("-i", args.index("-i") + 1) + 1]
        with open(meta, encoding="utf-8") as f:
            self.seen = {"meta": f.read(), "wav_exists": os.path.exists(wav), "wav_info": sf.info(wav)}
        if self.fail is not None:
            raise self.fail
        with open(args[-1], "wb") as f:
            f.write(b"M4B")
        return subprocess.CompletedProcess(args, 0, b"", b"")


@pytest.fixture
def ffmpeg(monkeypatch):
    fake = FakeFfmpeg()
    monkeypatch.setattr(m4b, "find_ffmpeg", lambda: FAKE)
    monkeypatch.setattr(m4b.subprocess, "run", fake)
    return fake


def _left_over(folder):
    return sorted(os.listdir(folder))


# -- the metadata file ---------------------------------------------------------------------------


def test_the_metadata_file_lists_the_tags_and_two_chapters():
    text = m4b.metadata_text({"title": "The Book", "artist": "A. Writer", "album": "Series", "track": "1/2",
                              "year": "2026", "description": "About it.", "genre": "Audiobook"},
                             [(0.0, "One"), (65.25, "Two", "a note")], 130.5)

    assert text == (
        ";FFMETADATA1\ntitle=The Book\nartist=A. Writer\nalbum=Series\ntrack=1/2\ndate=2026\n"
        "description=About it.\ngenre=Audiobook\n"
        "[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=65250\ntitle=One\n"
        "[CHAPTER]\nTIMEBASE=1/1000\nSTART=65250\nEND=130500\ntitle=Two\n")


def test_empty_fields_and_no_chapters_leave_only_the_header():
    assert m4b.metadata_text({"title": "", "artist": "  "}, None, 10.0) == ";FFMETADATA1\n"
    assert m4b.metadata_text(None, [], None) == ";FFMETADATA1\n"


def test_values_escape_the_characters_ffmpeg_reads_as_syntax():
    assert m4b.escape("a=b;c#d\\e") == "a\\=b\\;c\\#d\\\\e"
    assert m4b.escape("line one\nline two") == "line one\\\nline two"

    text = m4b.metadata_text({"title": "x=1; y#2", "description": "one\ntwo"}, [(0.0, "Q=A; #1\\2")], 5.0)

    assert "title=x\\=1\\; y\\#2\n" in text
    assert "description=one\\\ntwo\n" in text
    assert "title=Q\\=A\\; \\#1\\\\2\n" in text


def test_a_chapter_title_stays_on_one_line():
    text = m4b.metadata_text(None, [(0.0, "Part\none\r\n two")], 5.0)

    assert text.splitlines()[-1] == "title=Part one two"


def test_a_chapter_past_the_end_of_the_file_is_dropped():
    text = m4b.metadata_text(None, [(0.0, "In"), (9.0, "Out")], 5.0)

    assert text.count("[CHAPTER]") == 1 and "Out" not in text


# -- the command ---------------------------------------------------------------------------------


def test_the_argument_list_without_a_cover():
    args = m4b.ffmpeg_args(FAKE, "mix.wav", "meta.txt", "out.m4b", 96)

    assert args == [FAKE, "-hide_banner", "-nostdin", "-y", "-i", "mix.wav", "-i", "meta.txt", "-map", "0:a",
                    "-map_metadata", "1", "-map_chapters", "1", "-c:a", "aac", "-b:a", "96k",
                    "-movflags", "+faststart", "-f", "ipod", "out.m4b"]


def test_the_cover_is_a_third_input_kept_as_the_attached_picture():
    args = m4b.ffmpeg_args(FAKE, "mix.wav", "meta.txt", "out.m4b", 64, cover="cover.png")

    assert args[args.index("-i", 8) + 1] == "cover.png"
    assert args[args.index("-map", args.index("-map") + 1) + 1] == "2:v"
    assert args[args.index("-c:v") + 1] == "copy"
    assert args[args.index("-disposition:v:0") + 1] == "attached_pic"
    assert args.index("-map_metadata") > args.index("-i", 8)  # the inputs come before the options that use them
    assert args[-1] == "out.m4b" and args[args.index("-b:a") + 1] == "64k"


# -- write_m4b -----------------------------------------------------------------------------------


def test_write_m4b_runs_ffmpeg_and_leaves_only_the_m4b(tmp_path, ffmpeg):
    out = tmp_path / "out" / "book.m4b"

    warnings = m4b.write_m4b(_tone(2.0), RATE, str(out), 96, [(0.0, "One"), (1.0, "Two")], {"title": "Book"})

    assert warnings == []
    assert _left_over(tmp_path / "out") == ["book.m4b"] and out.read_bytes() == b"M4B"
    (args,) = ffmpeg.calls
    assert args[0] == FAKE and args[args.index("-b:a") + 1] == "96k" and args[args.index("-f") + 1] == "ipod"
    assert os.path.dirname(args[-1]) == str(tmp_path / "out") and args[-1] != str(out)  # a part file, then moved
    assert ffmpeg.seen["meta"].startswith(";FFMETADATA1\ntitle=Book\n") and ffmpeg.seen["meta"].count("[CHAPTER]") == 2
    assert ffmpeg.seen["wav_exists"] and ffmpeg.seen["wav_info"].samplerate == RATE
    assert ffmpeg.seen["wav_info"].frames == 2 * RATE


def test_the_command_is_a_list_with_a_timeout_that_grows_with_the_length(tmp_path, ffmpeg):
    m4b.write_m4b(_tone(1.0), RATE, str(tmp_path / "short.m4b"))
    m4b.write_m4b(np.zeros(RATE * 600, dtype=np.float32), RATE, str(tmp_path / "long.m4b"))

    short, long = ffmpeg.kwargs
    for kwargs in (short, long):
        assert not kwargs.get("shell") and kwargs["check"] is True and kwargs["capture_output"] is True
    assert short["timeout"] == m4b.MIN_TIMEOUT_S and long["timeout"] == pytest.approx(660.0)
    assert ffmpeg.calls[0][ffmpeg.calls[0].index("-b:a") + 1] == f"{m4b.DEFAULT_BITRATE_KBPS}k"


def test_samples_past_full_scale_are_clipped_not_wrapped(tmp_path, ffmpeg, monkeypatch):
    seen = {}
    real_write = sf.write

    def spy(path, data, *a, **k):
        seen["peak"] = float(np.max(np.abs(data)))
        return real_write(path, data, *a, **k)

    monkeypatch.setattr(sf, "write", spy)
    m4b.write_m4b(np.full(RATE, 1.7, dtype=np.float32), RATE, str(tmp_path / "loud.m4b"))

    assert seen["peak"] == 1.0


def test_a_failed_ffmpeg_raises_with_its_last_lines_and_removes_everything(tmp_path, monkeypatch):
    stderr = "\n".join(f"noise {n}" for n in range(20)).encode() + b"\nInvalid argument\n"
    fake = FakeFfmpeg(fail=subprocess.CalledProcessError(1, ["ffmpeg"], b"", stderr))
    monkeypatch.setattr(m4b, "find_ffmpeg", lambda: FAKE)
    monkeypatch.setattr(m4b.subprocess, "run", fake)

    with pytest.raises(m4b.M4bError) as raised:
        m4b.write_m4b(_tone(), RATE, str(tmp_path / "book.m4b"))

    assert "exit 1" in str(raised.value) and "Invalid argument" in str(raised.value)
    assert "noise 0" not in str(raised.value)  # only the tail
    assert _left_over(tmp_path) == []


def test_a_failure_keeps_the_file_that_was_already_there(tmp_path, monkeypatch):
    out = tmp_path / "book.m4b"
    out.write_bytes(b"old export")
    monkeypatch.setattr(m4b, "find_ffmpeg", lambda: FAKE)
    monkeypatch.setattr(m4b.subprocess, "run", FakeFfmpeg(fail=subprocess.CalledProcessError(1, ["ffmpeg"], b"", b"x")))

    with pytest.raises(m4b.M4bError):
        m4b.write_m4b(_tone(), RATE, str(out))

    assert out.read_bytes() == b"old export" and _left_over(tmp_path) == ["book.m4b"]


def test_a_timeout_is_an_error_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setattr(m4b, "find_ffmpeg", lambda: FAKE)
    monkeypatch.setattr(m4b.subprocess, "run", FakeFfmpeg(fail=subprocess.TimeoutExpired(["ffmpeg"], 120, None, b"stuck")))

    with pytest.raises(m4b.M4bError, match="longer than 120 s"):
        m4b.write_m4b(_tone(), RATE, str(tmp_path / "book.m4b"))

    assert _left_over(tmp_path) == []


def test_ffmpeg_that_cannot_start_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(m4b, "find_ffmpeg", lambda: FAKE)
    monkeypatch.setattr(m4b.subprocess, "run", FakeFfmpeg(fail=FileNotFoundError("gone")))

    with pytest.raises(m4b.M4bError, match="couldn't be started"):
        m4b.write_m4b(_tone(), RATE, str(tmp_path / "book.m4b"))

    assert _left_over(tmp_path) == []


def test_without_ffmpeg_it_says_so_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(m4b, "find_ffmpeg", lambda: None)

    with pytest.raises(m4b.M4bError, match="ffmpeg isn't on PATH"):
        m4b.write_m4b(_tone(), RATE, str(tmp_path / "book.m4b"))

    assert _left_over(tmp_path) == []


def test_find_ffmpeg_asks_the_path(monkeypatch):
    monkeypatch.setattr(m4b.shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
    assert m4b.find_ffmpeg() == "/usr/bin/ffmpeg"
    monkeypatch.setattr(m4b.shutil, "which", lambda name: None)
    assert m4b.find_ffmpeg() is None


def test_the_cover_goes_to_ffmpeg_by_its_real_path(tmp_path, ffmpeg):
    cover = tmp_path / "cover.png"
    cover.write_bytes(_png())
    out_dir = tmp_path / "out"

    warnings = m4b.write_m4b(_tone(), RATE, str(out_dir / "book.m4b"), cover_path=str(cover))

    assert warnings == []
    args = ffmpeg.calls[0]
    assert args[args.index("-i", 8) + 1] == os.path.realpath(cover) and "attached_pic" in args


def test_a_bad_cover_is_a_warning_and_the_export_goes_on(tmp_path, ffmpeg):
    (tmp_path / "cover.gif").write_bytes(b"GIF89a....")

    warnings = m4b.write_m4b(_tone(), RATE, str(tmp_path / "book.m4b"), cover_path=str(tmp_path / "cover.gif"))

    assert warnings == ["The cover image must be a JPEG or PNG, so it was left out of the M4B"]
    assert "attached_pic" not in ffmpeg.calls[0] and (tmp_path / "book.m4b").exists()


def test_write_audio_sends_m4b_to_ffmpeg_without_chapters(tmp_path, ffmpeg):
    write_audio(str(tmp_path / "clip.m4b"), _tone(), RATE, "m4b", bitrate_kbps=64)

    assert ffmpeg.calls[0][ffmpeg.calls[0].index("-b:a") + 1] == "64k"
    assert ffmpeg.seen["meta"] == ";FFMETADATA1\n"


# -- the chapters of a document ------------------------------------------------------------------


def _wav(path, value, seconds):
    sf.write(str(path), np.full(int(RATE * seconds), value, dtype=np.float32), RATE)
    return str(path)


def _book(tmp_path, titles=("Chapter 1", "Chapter 2")):
    """Intro (1 s), a nested clip per title (2 s, then 1 s) and an Outro (0.5 s)."""
    narrator = Character.from_preset_dict("Narrator", {})
    doc = Document.from_plain_text("Intro.\n\nOutro.", characters=[narrator])
    doc.settings.update({"gap_s": 0.0, "paragraph_gap_s": 0.0})
    intro = doc.assign_character_to_range(0, 6, narrator.id)
    outro = doc.assign_character_to_range(8, 14, narrator.id)
    intro.segments = [Segment(order_index=0, duration=1.0, audio_path=_wav(tmp_path / "intro.wav", 0.1, 1.0))]
    outro.segments = [Segment(order_index=0, duration=0.5, audio_path=_wav(tmp_path / "outro.wav", 0.4, 0.5))]
    durations, paths, position = {intro.id: 1.0, outro.id: 0.5}, {}, 8
    for number, title in enumerate(titles, start=1):
        chapter = doc.insert_nested_clip(position, {"kind": "embedded", "id": f"c{number}"}, title)
        position = doc.clip_extent(chapter.id)[1]
        durations[chapter.id] = 3.0 - number
        paths[chapter.id] = _wav(tmp_path / f"c{number}.wav", 0.1 * (number + 1), 3.0 - number)
    arrangement = compute_arrangement(doc, clip_duration=lambda c: durations.get(c.id), chars_per_second=15.0)
    return doc, arrangement, lambda clip: paths.get(clip.id)


def _marker(doc, *named):
    for seconds, name in named:
        doc.settings["markers"], _m = marker_ops.add_marker(doc.settings, seconds, name)


def test_the_chapters_are_the_subprojects_in_time_order(tmp_path):
    doc, arrangement, _nested = _book(tmp_path)

    rows = m4b.m4b_chapters(doc, arrangement)
    assert [title for _start, title, _note in rows] == ["Chapter 1", "Chapter 2"]
    assert rows[0][0] < rows[1][0]


def test_subprojects_win_over_markers_for_an_audiobook(tmp_path):
    doc, arrangement, _nested = _book(tmp_path)
    _marker(doc, (0.0, "A marker"), (2.0, "Another"))

    assert [title for _s, title, _n in m4b.m4b_chapters(doc, arrangement)] == ["Chapter 1", "Chapter 2"]


def test_without_subprojects_the_markers_are_the_chapters(tmp_path):
    doc, arrangement, _nested = _book(tmp_path, titles=())
    _marker(doc, (0.0, "Opening"), (1.0, "Closing"))

    assert [(s, t) for s, t, _n in m4b.m4b_chapters(doc, arrangement)] == [(0.0, "Opening"), (1.0, "Closing")]


def test_a_document_with_neither_has_no_chapters(tmp_path):
    doc, arrangement, _nested = _book(tmp_path, titles=())

    assert m4b.m4b_chapters(doc, arrangement) == []


# -- mixdown -------------------------------------------------------------------------------------


def test_mixdown_writes_an_m4b_with_a_chapter_per_subproject_and_the_tags(tmp_path, ffmpeg):
    doc, arrangement, nested = _book(tmp_path)

    result = mixdown(doc, str(tmp_path / "out" / "book.m4b"), fmt="m4b", sample_rate=RATE, arrangement=arrangement,
                     nested_audio_path=nested, bitrate_kbps=96,
                     tags={"title": "The Book", "artist": "A. Writer", "genre": "Audiobook", "chapters": False})

    seen = ffmpeg.seen["meta"]
    assert (tmp_path / "out" / "book.m4b").exists() and _left_over(tmp_path / "out") == ["book.m4b"]
    assert "title=The Book\nartist=A. Writer\ngenre=Audiobook\n" in seen
    chapters = seen.split("[CHAPTER]")[1:]
    assert [c.splitlines()[-1] for c in chapters] == ["title=Chapter 1", "title=Chapter 2"]
    assert "START=1000\nEND=3000" in chapters[0] and "START=3000\nEND=4500" in chapters[1]  # the last one runs to the end
    assert result.duration_s == pytest.approx(4.5) and ffmpeg.seen["wav_info"].frames == int(4.5 * RATE)
    assert result.files[0].tagged is True and result.warnings == []
    assert ffmpeg.calls[0][ffmpeg.calls[0].index("-b:a") + 1] == "96k"


def test_mixdown_keeps_the_m4b_chapters_when_the_tags_are_off(tmp_path, ffmpeg):
    doc, arrangement, nested = _book(tmp_path)

    result = mixdown(doc, str(tmp_path / "book.m4b"), fmt="m4b", sample_rate=RATE, arrangement=arrangement,
                     nested_audio_path=nested, tags=None)

    assert ffmpeg.seen["meta"].startswith(";FFMETADATA1\n[CHAPTER]") and ffmpeg.seen["meta"].count("[CHAPTER]") == 2
    assert result.files[0].tagged is False


def test_a_bad_cover_in_an_m4b_export_is_a_warning(tmp_path, ffmpeg):
    doc, arrangement, nested = _book(tmp_path)

    result = mixdown(doc, str(tmp_path / "book.m4b"), fmt="m4b", sample_rate=RATE, arrangement=arrangement,
                     nested_audio_path=nested, tags={"title": "T", "cover": str(tmp_path / "missing.png")})

    assert result.warnings == ["The cover image missing.png doesn't exist, so it was left out of the M4B"]
    assert (tmp_path / "book.m4b").exists()


def test_a_range_export_starts_its_chapters_at_the_range(tmp_path, ffmpeg):
    doc, arrangement, nested = _book(tmp_path, titles=())
    _marker(doc, (0.0, "Intro"), (1.0, "Rest"), (4.5, "End"))

    mixdown(doc, str(tmp_path / "book.m4b"), fmt="m4b", sample_rate=RATE, arrangement=arrangement,
            nested_audio_path=nested, range_s=(1.0, 4.5))

    chapters = ffmpeg.seen["meta"].split("[CHAPTER]")[1:]
    assert len(chapters) == 1 and "START=0\nEND=3500" in chapters[0] and chapters[0].endswith("title=Rest\n")


def test_head_silence_moves_the_m4b_chapters(tmp_path, ffmpeg):
    doc, arrangement, nested = _book(tmp_path)

    mixdown(doc, str(tmp_path / "book.m4b"), fmt="m4b", sample_rate=RATE, arrangement=arrangement,
            nested_audio_path=nested, head_s=2.0)

    assert "START=3000\nEND=5000" in ffmpeg.seen["meta"]


def test_a_split_m4b_export_gives_each_file_its_own_chapter_at_zero(tmp_path, ffmpeg):
    doc, arrangement, nested = _book(tmp_path)
    plan = plan_chapters(doc, arrangement, "subprojects")

    result = mixdown_chapters(doc, str(tmp_path / "out"), plan, fmt="m4b", sample_rate=RATE, arrangement=arrangement,
                              nested_audio_path=nested)

    assert [os.path.basename(f.path) for f in result.files] == ["01 - Chapter 1.m4b", "02 - Chapter 2.m4b"]
    assert _left_over(tmp_path / "out") == ["01 - Chapter 1.m4b", "02 - Chapter 2.m4b"]
    assert len(ffmpeg.calls) == 2


def test_stems_and_clip_files_in_m4b_come_out_without_chapters(tmp_path, ffmpeg):
    doc, arrangement, nested = _book(tmp_path)

    result = mixdown(doc, str(tmp_path / "book.m4b"), fmt="m4b", sample_rate=RATE, arrangement=arrangement,
                     nested_audio_path=nested, keep_clip_files=True)

    assert result.clip_files and all(path.endswith(".m4b") for path in result.clip_files)
    assert len(ffmpeg.calls) == 1 + len(result.clip_files)  # the clip files went through ffmpeg too


# -- a real ffmpeg -------------------------------------------------------------------------------


@pytest.mark.skipif(m4b.find_ffmpeg() is None, reason="ffmpeg isn't on PATH")
def test_a_real_round_trip_has_two_chapters_the_tags_and_the_cover(tmp_path):
    cover = tmp_path / "cover.png"
    cover.write_bytes(_png(64, 64))
    out = tmp_path / "book.m4b"

    warnings = m4b.write_m4b(_tone(4.0, 24000), 24000, str(out), 64,
                             [(0.0, "One; a=b #x"), (2.0, "Two")],
                             {"title": "The Book", "artist": "A. Writer", "year": "2026", "genre": "Audiobook",
                              "description": "Line one.\nLine two."}, str(cover))

    assert warnings == [] and out.exists() and _left_over(tmp_path) == ["book.m4b", "cover.png"]
    dump = subprocess.run([m4b.find_ffmpeg(), "-v", "error", "-i", str(out), "-f", "ffmetadata", "-"],
                          capture_output=True, text=True, check=True).stdout
    assert dump.count("[CHAPTER]") == 2
    assert "title=One\\; a\\=b \\#x" in dump and "title=Two" in dump and "START=2000" in dump
    assert "title=The Book" in dump and "artist=A. Writer" in dump
    mutagen_mp4 = pytest.importorskip("mutagen.mp4")
    audio = mutagen_mp4.MP4(str(out))
    assert [(c.start, c.title) for c in audio.chapters] == [(0.0, "One; a=b #x"), (2.0, "Two")]
    assert audio.tags["desc"] == ["Line one.\nLine two."] and len(audio.tags["covr"]) == 1
    assert audio.info.length == pytest.approx(4.0, abs=0.2)
