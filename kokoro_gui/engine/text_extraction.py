"""Text extraction from source files (.txt/.pdf/.epub), multi-speaker script
parsing, and long-text splitting into synthesis-sized chunks.

`extract_text_from_file` reads `pypdf.PdfReader`/`epub.read_epub` qualified,
at call time, so tests can monkeypatch them on those modules (e.g.
`monkeypatch.setattr(text_extraction.pypdf, "PdfReader", FakeReader)`).
"""
import math
import os
import re
import zipfile
from typing import Callable, NamedTuple, Optional

from bs4 import BeautifulSoup

import warnings

import ebooklib
import pypdf
from ebooklib import epub

# ebooklib warns on every EPUB it opens.
warnings.filterwarnings("ignore", category=UserWarning, module="ebooklib")
warnings.filterwarnings("ignore", category=FutureWarning, module="ebooklib")

# A book file is untrusted input: an EPUB is a zip that can inflate a
# thousandfold, and a PDF can claim any number of pages. These caps refuse
# such a file with `ExtractionLimitError` before it fills memory.
MAX_BOOK_FILE_BYTES = 500 * 1024 * 1024        # the file on disk, any type
MAX_EPUB_ENTRY_BYTES = 50 * 1024 * 1024        # one zip entry, uncompressed
MAX_EPUB_TOTAL_BYTES = 500 * 1024 * 1024       # all entries, uncompressed
MAX_EPUB_RATIO = 100                           # uncompressed / compressed, per entry
MAX_PDF_PAGES = 5000
MAX_TEXT_CHARS = 20_000_000                    # the extracted text


class ExtractionLimitError(ValueError):
    """A book file is over one of the limits above; the message names it."""


class ExtractionCancelled(Exception):
    """`should_stop` returned True while a file was being read."""


def _mb(n: int) -> str:
    return f"{n / (1024 * 1024):.0f} MB"


def _check_stop(should_stop) -> None:
    if should_stop is not None and should_stop():
        raise ExtractionCancelled("Import cancelled.")


def _check_file_size(fpath: str) -> None:
    size = os.path.getsize(fpath)
    if size > MAX_BOOK_FILE_BYTES:
        raise ExtractionLimitError(
            f"The file is {_mb(size)}, over the {_mb(MAX_BOOK_FILE_BYTES)} limit for a book file.")


def _check_chars(count: int) -> None:
    if count > MAX_TEXT_CHARS:
        raise ExtractionLimitError(
            f"The book holds more than {MAX_TEXT_CHARS:,} characters of text, over the limit.")


def _check_epub(fpath: str) -> None:
    """Refuses an EPUB whose zip would inflate past the limits, reading only
    the directory (`zipfile` never decompresses here). Call before
    `epub.read_epub`, which decompresses every entry into memory."""
    _check_file_size(fpath)
    total = 0
    with zipfile.ZipFile(fpath) as archive:
        for info in archive.infolist():
            if info.file_size > MAX_EPUB_ENTRY_BYTES:
                raise ExtractionLimitError(
                    f"'{info.filename}' inflates to {_mb(info.file_size)}, over the "
                    f"{_mb(MAX_EPUB_ENTRY_BYTES)} limit for one EPUB entry.")
            if info.compress_size and info.file_size / info.compress_size > MAX_EPUB_RATIO:
                raise ExtractionLimitError(
                    f"'{info.filename}' is compressed more than {MAX_EPUB_RATIO} to 1, "
                    "over the limit for an EPUB entry.")
            total += info.file_size
            if total > MAX_EPUB_TOTAL_BYTES:
                raise ExtractionLimitError(
                    f"The EPUB inflates to more than {_mb(MAX_EPUB_TOTAL_BYTES)}, over the limit.")


# Same tag syntax `TextExtractionMixin.parse_multispeaker_text` matches -
# duplicated here deliberately rather than shared/refactored out of that
# method, so `find_character_fx_spans` below can never accidentally change
# what conversion.py/jit.py (parse_multispeaker_text's only callers) see.
_SPEAKER_FX_TAG_PATTERN = r"\[([^\]\n]{1,100})\]:\s*"
# `[pause:1.5]`: silence before the next clip. Auto-split leaves the marker
# untagged and gives the next clip `gap_before_s`; the whole-document path
# strips it so it's never spoken. Not a speaker tag (no trailing colon).
PAUSE_MARKER_PATTERN = r"\[pause:(\d+(?:\.\d+)?)\]"
_MARKUP = re.compile(f"{_SPEAKER_FX_TAG_PATTERN}|{PAUSE_MARKER_PATTERN}")

# A tag's bracket reads `name[:fx](, key:value)*`: `[Sam, overlap:0.3]:`,
# `[Sam:Radio, overlap:0.3]:`. `TAG_OPTION_KEYS` are the keys that do
# something; a lowercase key that isn't one is still an option (and gets a
# warning), so a typo doesn't turn into part of the speaker's name.
TAG_OPTION_KEYS = ("overlap",)
# `Clip.overrides` key `overlap` becomes, and its upper limit in seconds.
OVERLAP_OVERRIDE_KEY = "overlap_s"
OVERLAP_MAX_S = 5.0
_TAG_OPTION = re.compile(r"^\s*([A-Za-z][A-Za-z_]*)\s*:\s*(.*?)\s*$")


def parse_tag_content(raw: str) -> tuple:
    """`(speaker_name, fx_name, options)` from the text between a tag's
    brackets. Split on `,` first: the first part is `name` or `name:FX` (on
    its first `:`), each later part a `key:value` option. A name or FX with a
    comma in it ("Smith, John") still parses as a name: the options count
    only when every part after the first is a `key:value` whose key is a
    known option or is all lowercase ("John:Radio" is not an option)."""
    options: dict = {}
    parts = raw.split(",")
    if len(parts) > 1:
        found = {}
        for part in parts[1:]:
            match = _TAG_OPTION.match(part)
            key = match.group(1) if match else ""
            if not match or not (key.lower() in TAG_OPTION_KEYS or key == key.lower()):
                found = None
                break
            found[key.lower()] = match.group(2)
        if found is not None:
            options = found
            raw = parts[0].strip()
    speaker_name, fx_name = raw, None
    if ":" in raw:
        head, _colon, tail = raw.partition(":")
        speaker_name, fx_name = head.strip(), tail.strip()
    return speaker_name, fx_name, options


def tag_overrides(options: dict) -> tuple:
    """`(overrides, warnings)` for a tag's `options`: `overlap` becomes
    `{"overlap_s": seconds}` clamped to `0..OVERLAP_MAX_S`; an unknown key
    or a value that isn't a number is dropped with a warning line."""
    overrides: dict = {}
    problems: list = []
    for key, value in (options or {}).items():
        if key.lower() == "overlap":
            try:
                seconds = float(value)
            except (TypeError, ValueError):
                seconds = None
            if seconds is None or not math.isfinite(seconds):
                problems.append(f"Tag option overlap:{value} isn't a number of seconds, so it was ignored.")
            else:
                overrides[OVERLAP_OVERRIDE_KEY] = round(min(OVERLAP_MAX_S, max(0.0, seconds)), 3)
        else:
            problems.append(f"Unknown tag option '{key}', so it was ignored.")
    return overrides, problems


def strip_markup(text: str, with_origin: bool = False):
    """`text` without its `[Name]:`/`[Name:FX]:` tags and `[pause:x]`
    markers: what a clip speaks (grill TE11). A clip made from a tagged line
    covers its tag, so every clip path (generation, the dirty check, the
    segmenter, subtitles) reads its text through this first. A marker goes
    away entirely when whitespace already sits on either side of it, else it
    becomes one space so the words around it stay apart.

    `with_origin=True` returns `(text, origin)`: per character of the result,
    `(orig_start, orig_end, rid)`, the shape `lexicon.apply_lexicon` takes as
    `origin` to chain its spans onto these (`lexicon.spoken`)."""
    if "[" not in text:
        return (text, [(i, i + 1, None) for i in range(len(text))]) if with_origin else text
    parts, origin, last = [], [], 0
    for index, match in enumerate(_MARKUP.finditer(text)):
        start, end = match.span()
        parts.append(text[last:start])
        if with_origin:
            origin.extend((i, i + 1, None) for i in range(last, start))
        before = text[start - 1] if start else " "
        after = text[end] if end < len(text) else " "
        if not (before.isspace() or after.isspace()):
            parts.append(" ")
            if with_origin:
                origin.append((start, end, ("markup", index)))
        last = end
    parts.append(text[last:])
    stripped = "".join(parts)
    if not with_origin:
        return stripped
    origin.extend((i, i + 1, None) for i in range(last, len(text)))
    return stripped, origin


class InlineTagSpan(NamedTuple):
    """One `[Name]:`/`[Name:FX]:`-tagged run, with real (unstripped) offsets
    into the original text - unlike `parse_multispeaker_text`'s tuples,
    which discard offsets and strip/filter the segment text. Used by the
    Qt transcript editor's syntax highlighter (kokoro_gui/qt/transcript_editor.py),
    which needs exact `QTextDocument` character positions, not cleaned-up text.
    """

    start: int
    end: int
    speaker_name: str
    fx_name: Optional[str]
    # `key:value` pairs after the name, as the typed strings (`{"overlap": "0.3"}`).
    options: dict = {}


def find_character_fx_spans(text: str) -> list:
    """Offset-preserving sibling of `TextExtractionMixin.parse_multispeaker_text`
    for `[Name]:`/`[Name:FX]:` tags. Returns `[]` for tagless text (not
    `parse_multispeaker_text`'s `[(None, None, text)]` sentinel - a
    highlighter has nothing to paint when there's no tag at all). Each
    `InlineTagSpan` covers from its tag's own start through the character
    just before the next tag (or end of text) - the whole `[Name]: spoken
    text` run, unstripped, so it maps 1:1 onto document character positions.

    Module-level rather than a `TextExtractionMixin` method: this is a pure
    text-in/data-out utility with no need for a live engine instance.
    """
    matches = list(re.finditer(_SPEAKER_FX_TAG_PATTERN, text))
    spans = []
    for i, match in enumerate(matches):
        speaker_name, fx_name, options = parse_tag_content(match.group(1))

        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        spans.append(InlineTagSpan(start=match.start(), end=end, speaker_name=speaker_name, fx_name=fx_name,
                                   options=options))
    return spans


def _epub_sections(fpath: str, should_stop: Optional[Callable[[], bool]] = None) -> list:
    """One `(title, text)` per EPUB spine document with text, in reading
    order; the title is the document's first `<h1>`/`<h2>`, else
    "Chapter N"."""
    _check_epub(fpath)
    book = epub.read_epub(fpath, options={'ignore_ncx': True})
    documents = [item for item in book.get_items() if item.get_type() == ebooklib.ITEM_DOCUMENT]
    spine = [entry[0] if isinstance(entry, (tuple, list)) else entry for entry in (getattr(book, "spine", None) or [])]
    if spine:
        by_id = {getattr(item, "id", None) or item.get_id(): item for item in documents}
        ordered = [by_id[i] for i in spine if i in by_id]
        ordered += [item for item in documents if item not in ordered]
        documents = ordered
    sections = []
    chars = 0
    for item in documents:
        _check_stop(should_stop)
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        heading = soup.find(["h1", "h2"])
        text = soup.get_text(separator='\n\n').strip()
        if not text:
            continue
        chars += len(text)
        _check_chars(chars)
        title = heading.get_text(" ", strip=True) if heading is not None else ""
        sections.append((title or f"Chapter {len(sections) + 1}", text))
    return sections


def _pdf_pages(fpath: str, should_stop: Optional[Callable[[], bool]] = None) -> tuple:
    """`(reader, [page text, ...])` under the page and text limits;
    `should_stop` is asked after each page."""
    _check_file_size(fpath)
    reader = pypdf.PdfReader(fpath)
    count = len(reader.pages)
    if count > MAX_PDF_PAGES:
        raise ExtractionLimitError(f"The PDF has {count:,} pages, over the {MAX_PDF_PAGES:,} page limit.")
    pages, chars = [], 0
    for page in reader.pages:
        text = page.extract_text() or ""
        chars += len(text)
        _check_chars(chars)
        pages.append(text)
        _check_stop(should_stop)
    return reader, pages


def _pdf_sections(fpath: str, should_stop: Optional[Callable[[], bool]] = None) -> list:
    """One `(title, text)` per top-level outline entry, from its page to the
    next entry's; the whole document as one section when there's no
    outline."""
    reader, pages = _pdf_pages(fpath, should_stop)
    starts = []
    try:
        outline = reader.outline or []
    except Exception:
        outline = []
    for entry in outline:
        if isinstance(entry, list):
            continue  # nested entries belong to the chapter above
        try:
            starts.append((str(entry.title).strip(), reader.get_destination_page_number(entry)))
        except Exception:
            continue
    starts = sorted(((t, p) for t, p in starts if 0 <= p < len(pages)), key=lambda item: item[1])
    if not starts:
        text = "\n\n".join(p for p in pages if p).strip()
        return [(os.path.splitext(os.path.basename(fpath))[0], text)] if text else []
    sections = []
    for index, (title, first) in enumerate(starts):
        last = starts[index + 1][1] if index + 1 < len(starts) else len(pages)
        text = "\n\n".join(p for p in pages[first:max(last, first + 1)] if p).strip()
        if text:
            sections.append((title or f"Chapter {index + 1}", text))
    return sections


def extract_sections(fpath: str, should_stop: Optional[Callable[[], bool]] = None) -> list:
    """`[(title, text), ...]`: a book split at its chapters (grill NP8, the
    New-from-eBook path that makes one subproject each). EPUB: spine
    documents, titled by their first heading. PDF: the outline's top-level
    page ranges. Anything else, or a PDF without an outline: one section.

    A book over the module's limits raises `ExtractionLimitError`.
    `should_stop`, when given, is asked between spine documents and pages;
    True raises `ExtractionCancelled`."""
    if not os.path.exists(fpath):
        raise FileNotFoundError("File does not exist.")
    lower = fpath.lower()
    if lower.endswith(".epub"):
        return _epub_sections(fpath, should_stop)
    if lower.endswith(".pdf"):
        return _pdf_sections(fpath, should_stop)
    text = _read_text_file(fpath).strip()
    return [(os.path.splitext(os.path.basename(fpath))[0], text)] if text else []


def _read_text_file(fpath: str) -> str:
    """A plain-text file under the size and character limits."""
    _check_file_size(fpath)
    with open(fpath, "r", encoding="utf-8") as f:
        text = f.read(MAX_TEXT_CHARS + 1)
    _check_chars(len(text))
    return text


def extract_text_from_file(fpath, should_stop: Optional[Callable[[], bool]] = None):
    """The text of a .txt, .pdf or .epub file. A module function: it never
    needed an engine (the GUI calls it without one). A file over the
    module's limits raises `ExtractionLimitError`; `should_stop`, when
    given, is asked between pages and spine documents and True raises
    `ExtractionCancelled`."""
    if not os.path.exists(fpath):
        raise FileNotFoundError("File does not exist.")

    lower_path = fpath.lower()

    if lower_path.endswith(".pdf"):
        _reader, pages = _pdf_pages(fpath, should_stop)
        return "".join(page + "\n\n" for page in pages if page)

    if lower_path.endswith(".epub"):
        _check_epub(fpath)
        book = epub.read_epub(fpath, options={'ignore_ncx': True})
        parts, chars = [], 0
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                _check_stop(should_stop)
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                part = soup.get_text(separator='\n\n') + "\n\n"
                chars += len(part)
                _check_chars(chars)
                parts.append(part)
        return "".join(parts)

    # Assume text based
    return _read_text_file(fpath)


class TextExtractionMixin:
    def extract_sections(self, fpath):
        return extract_sections(fpath)

    def extract_text_from_file(self, fpath):
        return extract_text_from_file(fpath)

    def parse_multispeaker_text(self, text):
        """
        Parses text for [PresetName]: or [PresetName:FXPresetName]: syntax.
        Returns a list of (speaker_name, fx_name, text_segment)
        """
        # Regex to find [Name]: or [Name:FX]:

        # A `[pause:x]` marker has no clip to carry a gap on this path;
        # drop it so it's never read aloud.
        text = re.sub(PAUSE_MARKER_PATTERN, " ", text)
        pattern = r"\[([^\]\n]{1,100})\]:\s*"
        matches = list(re.finditer(pattern, text))

        if not matches:
            return [(None, None, text)]

        segments = []
        for i in range(len(matches)):
            # `[Name, overlap:0.3]:` options mean nothing here: this path has
            # no timeline. They are parsed only so the name is right.
            speaker_name, fx_name, _options = parse_tag_content(matches[i].group(1))

            start = matches[i].end()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            segment_text = text[start:end].strip()
            if segment_text:
                segments.append((speaker_name, fx_name, segment_text))

        return segments

    def smart_split(self, text, chunk_size=3000):
        chunks = []
        current_chunk = []
        current_len = 0
        paragraphs = text.split('\n\n')

        for para in paragraphs:
            if len(para) > chunk_size:
                lines = para.split('\n')
                for line in lines:
                    if current_len + len(line) > chunk_size and current_chunk:
                        chunks.append("\n".join(current_chunk))
                        current_chunk = []
                        current_len = 0
                    current_chunk.append(line)
                    current_len += len(line)
            else:
                if current_len + len(para) > chunk_size and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                current_chunk.append(para)
                current_len += len(para)

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        return [c for c in chunks if c.strip()]
