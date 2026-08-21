"""Tests for parse_multispeaker_text, smart_split, extract_text_from_file
(kokoro_engine.py:459-489, 517-543, 432-457)."""
import pytest

import kokoro_engine


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
