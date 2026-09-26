"""Text extraction from source files (.txt/.pdf/.epub), multi-speaker script
parsing, and long-text splitting into synthesis-sized chunks.

`extract_text_from_file` reads `pypdf.PdfReader`/`epub.read_epub` qualified,
at call time, so tests can monkeypatch them on those modules (e.g.
`monkeypatch.setattr(text_extraction.pypdf, "PdfReader", FakeReader)`).
"""
import os
import re
from typing import NamedTuple, Optional

from bs4 import BeautifulSoup

import warnings

import ebooklib
import pypdf
from ebooklib import epub

# ebooklib warns on every EPUB it opens.
warnings.filterwarnings("ignore", category=UserWarning, module="ebooklib")
warnings.filterwarnings("ignore", category=FutureWarning, module="ebooklib")

# Same tag syntax `TextExtractionMixin.parse_multispeaker_text` matches -
# duplicated here deliberately rather than shared/refactored out of that
# method, so `find_character_fx_spans` below can never accidentally change
# what conversion.py/jit.py (parse_multispeaker_text's only callers) see.
_SPEAKER_FX_TAG_PATTERN = r"\[([^\]\n]{1,100})\]:\s*"
# `[pause:1.5]`: silence before the next clip. Auto-split leaves the marker
# untagged and gives the next clip `gap_before_s`; the whole-document path
# strips it so it's never spoken. Not a speaker tag (no trailing colon).
PAUSE_MARKER_PATTERN = r"\[pause:(\d+(?:\.\d+)?)\]"


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
        raw_name = match.group(1)
        speaker_name, fx_name = raw_name, None
        if ":" in raw_name:
            parts = raw_name.split(":", 1)
            speaker_name = parts[0].strip()
            fx_name = parts[1].strip()

        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        spans.append(InlineTagSpan(start=match.start(), end=end, speaker_name=speaker_name, fx_name=fx_name))
    return spans


def _epub_sections(fpath: str) -> list:
    """One `(title, text)` per EPUB spine document with text, in reading
    order; the title is the document's first `<h1>`/`<h2>`, else
    "Chapter N"."""
    book = epub.read_epub(fpath, options={'ignore_ncx': True})
    documents = [item for item in book.get_items() if item.get_type() == ebooklib.ITEM_DOCUMENT]
    spine = [entry[0] if isinstance(entry, (tuple, list)) else entry for entry in (getattr(book, "spine", None) or [])]
    if spine:
        by_id = {getattr(item, "id", None) or item.get_id(): item for item in documents}
        ordered = [by_id[i] for i in spine if i in by_id]
        ordered += [item for item in documents if item not in ordered]
        documents = ordered
    sections = []
    for item in documents:
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        heading = soup.find(["h1", "h2"])
        text = soup.get_text(separator='\n\n').strip()
        if not text:
            continue
        title = heading.get_text(" ", strip=True) if heading is not None else ""
        sections.append((title or f"Chapter {len(sections) + 1}", text))
    return sections


def _pdf_sections(fpath: str) -> list:
    """One `(title, text)` per top-level outline entry, from its page to the
    next entry's; the whole document as one section when there's no
    outline."""
    reader = pypdf.PdfReader(fpath)
    pages = [(page.extract_text() or "") for page in reader.pages]
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


def extract_sections(fpath: str) -> list:
    """`[(title, text), ...]`: a book split at its chapters (grill NP8, the
    New-from-eBook path that makes one subproject each). EPUB: spine
    documents, titled by their first heading. PDF: the outline's top-level
    page ranges. Anything else, or a PDF without an outline: one section."""
    if not os.path.exists(fpath):
        raise FileNotFoundError("File does not exist.")
    lower = fpath.lower()
    if lower.endswith(".epub"):
        return _epub_sections(fpath)
    if lower.endswith(".pdf"):
        return _pdf_sections(fpath)
    with open(fpath, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return [(os.path.splitext(os.path.basename(fpath))[0], text)] if text else []


def extract_text_from_file(fpath):
    """The text of a .txt, .pdf or .epub file. A module function: it never
    needed an engine (the GUI calls it without one)."""
    if not os.path.exists(fpath):
        raise FileNotFoundError("File does not exist.")

    text_data = ""
    lower_path = fpath.lower()

    if lower_path.endswith(".pdf"):
        reader = pypdf.PdfReader(fpath)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text_data += extracted + "\n\n"

    elif lower_path.endswith(".epub"):
        book = epub.read_epub(fpath, options={'ignore_ncx': True})
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text_data += soup.get_text(separator='\n\n') + "\n\n"
    else:
        # Assume text based
        with open(fpath, "r", encoding="utf-8") as f:
            text_data = f.read()

    return text_data


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
            raw_name = matches[i].group(1)
            speaker_name = raw_name
            fx_name = None

            if ":" in raw_name:
                parts = raw_name.split(":", 1)
                speaker_name = parts[0].strip()
                fx_name = parts[1].strip()

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
