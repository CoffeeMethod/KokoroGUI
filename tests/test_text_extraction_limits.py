"""Book files are untrusted input: `text_extraction`'s size, ratio, page and
character limits, and the `should_stop` hook that cancels a read."""
import zipfile

import pytest

from kokoro_gui.engine import text_extraction
from kokoro_gui.engine.text_extraction import (
    ExtractionCancelled, ExtractionLimitError, extract_sections, extract_text_from_file)

MB = 1024 * 1024


def _zip_with_entry(path, size, name="OEBPS/big.xhtml"):
    """An EPUB-shaped zip whose one entry inflates to `size` zero bytes:
    deflate keeps the file on disk tiny."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(name, b"\0" * size)
    return str(path)


@pytest.fixture
def read_epub_spy(monkeypatch):
    calls = []
    monkeypatch.setattr(text_extraction.epub, "read_epub", lambda *a, **k: calls.append(a))
    return calls


@pytest.mark.parametrize("func", [extract_text_from_file, extract_sections])
def test_an_epub_entry_over_the_size_limit_is_refused_before_it_is_read(tmp_path, read_epub_spy, func):
    path = _zip_with_entry(tmp_path / "bomb.epub", 60 * MB)
    with pytest.raises(ExtractionLimitError, match="OEBPS/big.xhtml"):
        func(path)
    assert read_epub_spy == []


def test_an_epub_entry_over_the_ratio_is_refused(tmp_path, read_epub_spy):
    path = _zip_with_entry(tmp_path / "ratio.epub", 5 * MB)
    with pytest.raises(ExtractionLimitError, match="to 1"):
        extract_text_from_file(path)
    assert read_epub_spy == []


def test_an_epub_over_the_total_limit_is_refused(tmp_path, read_epub_spy, monkeypatch):
    monkeypatch.setattr(text_extraction, "MAX_EPUB_TOTAL_BYTES", 150)
    monkeypatch.setattr(text_extraction, "MAX_EPUB_RATIO", 10**9)
    path = tmp_path / "many.epub"
    with zipfile.ZipFile(path, "w") as z:
        for i in range(4):
            z.writestr(f"c{i}.xhtml", "x" * 50)
    with pytest.raises(ExtractionLimitError, match="inflates to more than"):
        extract_text_from_file(str(path))
    assert read_epub_spy == []


@pytest.mark.parametrize("name", ["big.txt", "big.pdf", "big.epub"])
def test_a_file_over_the_byte_limit_is_refused_by_type(tmp_path, monkeypatch, name):
    monkeypatch.setattr(text_extraction, "MAX_BOOK_FILE_BYTES", 10)
    path = tmp_path / name
    path.write_bytes(b"x" * 11)
    with pytest.raises(ExtractionLimitError, match="limit for a book file"):
        extract_text_from_file(str(path))
    with pytest.raises(ExtractionLimitError):
        extract_sections(str(path))


def test_a_text_file_over_the_character_limit_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(text_extraction, "MAX_TEXT_CHARS", 100)
    path = tmp_path / "long.txt"
    path.write_text("a" * 101, encoding="utf-8")
    with pytest.raises(ExtractionLimitError, match="characters"):
        extract_text_from_file(str(path))
    path.write_text("a" * 100, encoding="utf-8")
    assert extract_text_from_file(str(path)) == "a" * 100


def _fake_pdf(monkeypatch, tmp_path, pages, seen=None):
    class FakePage:
        def extract_text(self):
            if seen is not None:
                seen.append(1)
            return "words"

    class FakeReader:
        outline = []

        def __init__(self, path):
            self.pages = [FakePage() for _ in range(pages)]

    monkeypatch.setattr(text_extraction.pypdf, "PdfReader", FakeReader)
    path = tmp_path / "book.pdf"
    path.write_bytes(b"%PDF-fake")
    return str(path)


@pytest.mark.parametrize("func", [extract_text_from_file, extract_sections])
def test_a_pdf_over_the_page_limit_is_refused_before_any_page_is_read(tmp_path, monkeypatch, func):
    seen = []
    path = _fake_pdf(monkeypatch, tmp_path, 6000, seen)
    with pytest.raises(ExtractionLimitError, match="6,000 pages"):
        func(path)
    assert seen == []


def test_a_pdf_at_the_page_limit_reads(tmp_path, monkeypatch):
    path = _fake_pdf(monkeypatch, tmp_path, text_extraction.MAX_PDF_PAGES)
    assert extract_text_from_file(path).count("words") == text_extraction.MAX_PDF_PAGES


def test_pdf_text_over_the_character_limit_stops_the_read(tmp_path, monkeypatch):
    monkeypatch.setattr(text_extraction, "MAX_TEXT_CHARS", 12)
    seen = []
    path = _fake_pdf(monkeypatch, tmp_path, 50, seen)
    with pytest.raises(ExtractionLimitError, match="characters"):
        extract_text_from_file(path)
    assert len(seen) == 3  # 5 characters a page: the third page crosses 12


@pytest.mark.parametrize("func", [extract_text_from_file, extract_sections])
def test_should_stop_ends_a_pdf_read_after_the_first_page(tmp_path, monkeypatch, func):
    seen = []
    path = _fake_pdf(monkeypatch, tmp_path, 10, seen)
    with pytest.raises(ExtractionCancelled):
        func(path, should_stop=lambda: True)
    assert len(seen) == 1


def test_should_stop_ends_an_epub_read_between_documents(tmp_path, monkeypatch):
    class FakeItem:
        def get_type(self):
            return text_extraction.ebooklib.ITEM_DOCUMENT

        def get_content(self):
            return b"<html><body><p>Chapter text.</p></body></html>"

    class FakeBook:
        spine = []

        def get_items(self):
            return [FakeItem(), FakeItem()]

    monkeypatch.setattr(text_extraction.epub, "read_epub", lambda path, options=None: FakeBook())
    path = _zip_with_entry(tmp_path / "ok.epub", 10)
    asked = []

    def stop():
        asked.append(1)
        return True

    with pytest.raises(ExtractionCancelled):
        extract_text_from_file(path, should_stop=stop)
    assert len(asked) == 1
    with pytest.raises(ExtractionCancelled):
        extract_sections(path, should_stop=lambda: True)


def test_a_book_under_every_limit_still_reads(tmp_path, monkeypatch):
    path = tmp_path / "plain.txt"
    path.write_text("Hello there.", encoding="utf-8")
    assert extract_text_from_file(str(path), should_stop=lambda: False) == "Hello there."
    assert extract_sections(str(path)) == [("plain", "Hello there.")]
