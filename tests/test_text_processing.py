"""Tests for parse_multispeaker_text, smart_split, extract_text_from_file
(kokoro_engine.py:459-489, 517-543, 432-457), and find_character_fx_spans
(kokoro_gui/engine/text_extraction.py's offset-preserving sibling of
parse_multispeaker_text, used by the Qt transcript editor's highlighter)."""
import pytest

import kokoro_engine
from kokoro_gui.engine.text_extraction import find_character_fx_spans


# --- parse_multispeaker_text ---

def test_parse_multispeaker_no_markers_returns_single_none_tuple(engine):
    result = engine.parse_multispeaker_text("Just plain text.")
    assert result == [(None, None, "Just plain text.")]


def test_parse_multispeaker_single_speaker_marker(engine):
    result = engine.parse_multispeaker_text("[Narrator]: Hello there.")
    assert result == [("Narrator", None, "Hello there.")]


def test_parse_multispeaker_speaker_and_fx_marker(engine):
    result = engine.parse_multispeaker_text("[Narrator:Radio]: Hello there.")
    assert result == [("Narrator", "Radio", "Hello there.")]


def test_parse_multispeaker_multiple_segments(engine):
    result = engine.parse_multispeaker_text("[A]: first\n\n[B]: second")
    assert result == [("A", None, "first"), ("B", None, "second")]


def test_parse_multispeaker_marker_regex_length_limit(engine):
    # The marker regex caps bracket contents at 100 chars; longer bracket
    # contents should not match as a marker at all (kokoro_engine.py:466).
    long_name = "A" * 150
    text = f"[{long_name}]: hello"
    result = engine.parse_multispeaker_text(text)
    assert result == [(None, None, text)]


def test_parse_multispeaker_empty_segment_is_skipped(engine):
    result = engine.parse_multispeaker_text("[A]: \n\n[B]: real text")
    assert result == [("B", None, "real text")]


# --- find_character_fx_spans (no `engine` fixture needed - module-level) ---

def test_find_character_fx_spans_no_tags_returns_empty_list():
    # Unlike parse_multispeaker_text's [(None, None, text)] sentinel - a
    # highlighter has nothing to paint when there's no tag at all.
    assert find_character_fx_spans("Just plain text.") == []


def test_find_character_fx_spans_single_tag_covers_tag_through_end():
    text = "[Narrator]: Hello there."
    spans = find_character_fx_spans(text)
    assert len(spans) == 1
    span = spans[0]
    assert span.speaker_name == "Narrator"
    assert span.fx_name is None
    assert span.start == 0
    assert span.end == len(text)
    # Unstripped: the span's slice is the literal tag plus its trailing
    # space, exactly as it appears in the source text.
    assert text[span.start:span.end] == text


def test_find_character_fx_spans_speaker_and_fx():
    span = find_character_fx_spans("[Narrator:Radio]: Hi.")[0]
    assert span.speaker_name == "Narrator"
    assert span.fx_name == "Radio"


def test_find_character_fx_spans_multiple_tags_boundaries_at_next_tag_start():
    text = "[A]: first\n\n[B]: second"
    spans = find_character_fx_spans(text)
    assert len(spans) == 2
    a_span, b_span = spans
    assert a_span.start == 0
    assert a_span.end == text.index("[B]")
    assert b_span.start == text.index("[B]")
    assert b_span.end == len(text)


def test_find_character_fx_spans_does_not_strip_or_filter_empty_segments():
    # parse_multispeaker_text would drop the empty "[A]: " segment entirely;
    # find_character_fx_spans keeps every tag's span since offset fidelity,
    # not clean text, is the point.
    text = "[A]: \n\n[B]: real text"
    spans = find_character_fx_spans(text)
    assert [s.speaker_name for s in spans] == ["A", "B"]


def test_find_character_fx_spans_marker_regex_length_limit():
    long_name = "A" * 150
    text = f"[{long_name}]: hello"
    assert find_character_fx_spans(text) == []


# --- smart_split ---

def test_smart_split_splits_on_paragraph_boundaries(engine):
    text = "para one" + "\n\n" + ("x" * 20)
    chunks = engine.smart_split(text, chunk_size=15)
    assert len(chunks) == 2


def test_smart_split_respects_chunk_size_budget(engine):
    para = "y" * 50
    text = "\n\n".join([para] * 5)
    chunks = engine.smart_split(text, chunk_size=60)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 60


def test_smart_split_single_short_text_returns_one_chunk(engine):
    assert engine.smart_split("short text", chunk_size=3000) == ["short text"]


def test_smart_split_filters_whitespace_only_chunks(engine):
    assert engine.smart_split("   ", chunk_size=3000) == []


# --- extract_text_from_file ---

def test_extract_text_from_file_txt(engine, tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("Hello file.", encoding="utf-8")
    assert engine.extract_text_from_file(str(p)) == "Hello file."


def test_extract_text_from_file_missing_raises(engine, tmp_path):
    with pytest.raises(FileNotFoundError):
        engine.extract_text_from_file(str(tmp_path / "nope.txt"))


def test_extract_text_from_file_pdf(engine, tmp_path, monkeypatch):
    class FakePage:
        def extract_text(self):
            return "Page text."

    class FakeReader:
        def __init__(self, path):
            self.pages = [FakePage(), FakePage()]

    monkeypatch.setattr(kokoro_engine.pypdf, "PdfReader", FakeReader)
    p = tmp_path / "sample.pdf"
    p.write_bytes(b"%PDF-fake")

    text = engine.extract_text_from_file(str(p))
    assert text.count("Page text.") == 2


def test_extract_text_from_file_epub(engine, tmp_path, monkeypatch):
    class FakeItem:
        def get_type(self):
            return kokoro_engine.ebooklib.ITEM_DOCUMENT

        def get_content(self):
            return b"<html><body><p>Chapter text.</p></body></html>"

    class FakeBook:
        def get_items(self):
            return [FakeItem()]

    monkeypatch.setattr(kokoro_engine.epub, "read_epub", lambda path, options=None: FakeBook())
    p = tmp_path / "sample.epub"
    p.write_bytes(b"fake-epub")

    text = engine.extract_text_from_file(str(p))
    assert "Chapter text." in text


def test_parse_multispeaker_drops_pause_markers(engine):
    result = engine.parse_multispeaker_text("[Narrator]: Hello. [pause:1.5]\n[Bob]: Hi.")
    assert result == [("Narrator", None, "Hello."), ("Bob", None, "Hi.")]


# --- extract_sections (phase 4, New from eBook: one subproject per chapter) ---

def _fake_epub(monkeypatch, documents, spine=None):
    class FakeItem:
        def __init__(self, item_id, html):
            self.id = item_id
            self._html = html

        def get_type(self):
            return kokoro_engine.ebooklib.ITEM_DOCUMENT

        def get_content(self):
            return self._html.encode("utf-8")

        def get_id(self):
            return self.id

    class FakeBook:
        def __init__(self):
            self.spine = spine or []

        def get_items(self):
            return [FakeItem(i, h) for i, h in documents]

    monkeypatch.setattr(kokoro_engine.epub, "read_epub", lambda path, options=None: FakeBook())


def test_extract_sections_epub_titles_from_headings_in_spine_order(tmp_path, monkeypatch):
    from kokoro_gui.engine.text_extraction import extract_sections

    _fake_epub(monkeypatch, [
        ("c2", "<html><body><h2>The Storm</h2><p>Rain fell.</p></body></html>"),
        ("c1", "<html><body><h1>Arrival</h1><p>She came home.</p></body></html>"),
        ("blank", "<html><body></body></html>"),
        ("c3", "<html><body><p>No heading here.</p></body></html>"),
    ], spine=[("c1", "yes"), ("c2", "yes"), ("blank", "yes"), ("c3", "yes")])
    p = tmp_path / "book.epub"
    p.write_bytes(b"fake-epub")

    sections = extract_sections(str(p))

    assert [t for t, _ in sections] == ["Arrival", "The Storm", "Chapter 3"]
    assert "She came home." in sections[0][1]
    assert "Rain fell." in sections[1][1]


def test_extract_sections_pdf_by_outline_page_ranges(tmp_path, monkeypatch):
    from kokoro_gui.engine.text_extraction import extract_sections

    class Page:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class Entry:
        def __init__(self, title, page):
            self.title = title
            self.page = page

    class Reader:
        def __init__(self, path):
            self.pages = [Page("One a."), Page("One b."), Page("Two a.")]
            self.outline = [Entry("One", 0), [Entry("One sub", 1)], Entry("Two", 2)]

        def get_destination_page_number(self, entry):
            return entry.page

    monkeypatch.setattr(kokoro_engine.pypdf, "PdfReader", Reader)
    p = tmp_path / "book.pdf"
    p.write_bytes(b"%PDF-fake")

    sections = extract_sections(str(p))

    assert [t for t, _ in sections] == ["One", "Two"]
    assert "One a." in sections[0][1] and "One b." in sections[0][1]
    assert sections[1][1] == "Two a."


def test_extract_sections_pdf_without_outline_and_txt_are_one_section(tmp_path, monkeypatch):
    from kokoro_gui.engine.text_extraction import extract_sections

    class Reader:
        def __init__(self, path):
            self.pages = [type("P", (), {"extract_text": lambda self: "All of it."})()]
            self.outline = []

    monkeypatch.setattr(kokoro_engine.pypdf, "PdfReader", Reader)
    pdf = tmp_path / "flat.pdf"
    pdf.write_bytes(b"%PDF-fake")
    assert extract_sections(str(pdf)) == [("flat", "All of it.")]

    txt = tmp_path / "notes.txt"
    txt.write_text("Plain words.", encoding="utf-8")
    assert extract_sections(str(txt)) == [("notes", "Plain words.")]
