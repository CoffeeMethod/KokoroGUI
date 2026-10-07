"""Tests for parse_multispeaker_text, smart_split, extract_text_from_file
(kokoro_engine.py:459-489, 517-543, 432-457), and find_character_fx_spans
(kokoro_gui/engine/text_extraction.py's offset-preserving sibling of
parse_multispeaker_text, used by the Qt transcript editor's highlighter)."""
import pytest

from kokoro_gui.engine import text_extraction
from kokoro_gui.engine.text_extraction import find_character_fx_spans


def _write_zip(path):
    """A real (tiny) zip: the EPUB reader is faked, but the size check opens
    the file as a zip first."""
    import zipfile

    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")


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

    monkeypatch.setattr(text_extraction.pypdf, "PdfReader", FakeReader)
    p = tmp_path / "sample.pdf"
    p.write_bytes(b"%PDF-fake")

    text = engine.extract_text_from_file(str(p))
    assert text.count("Page text.") == 2


def test_extract_text_from_file_epub(engine, tmp_path, monkeypatch):
    class FakeItem:
        def get_type(self):
            return text_extraction.ebooklib.ITEM_DOCUMENT

        def get_content(self):
            return b"<html><body><p>Chapter text.</p></body></html>"

    class FakeBook:
        def get_items(self):
            return [FakeItem()]

    monkeypatch.setattr(text_extraction.epub, "read_epub", lambda path, options=None: FakeBook())
    p = tmp_path / "sample.epub"
    _write_zip(p)

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
            return text_extraction.ebooklib.ITEM_DOCUMENT

        def get_content(self):
            return self._html.encode("utf-8")

        def get_id(self):
            return self.id

    class FakeBook:
        def __init__(self):
            self.spine = spine or []

        def get_items(self):
            return [FakeItem(i, h) for i, h in documents]

    monkeypatch.setattr(text_extraction.epub, "read_epub", lambda path, options=None: FakeBook())


def test_extract_sections_epub_titles_from_headings_in_spine_order(tmp_path, monkeypatch):
    from kokoro_gui.engine.text_extraction import extract_sections

    _fake_epub(monkeypatch, [
        ("c2", "<html><body><h2>The Storm</h2><p>Rain fell.</p></body></html>"),
        ("c1", "<html><body><h1>Arrival</h1><p>She came home.</p></body></html>"),
        ("blank", "<html><body></body></html>"),
        ("c3", "<html><body><p>No heading here.</p></body></html>"),
    ], spine=[("c1", "yes"), ("c2", "yes"), ("blank", "yes"), ("c3", "yes")])
    p = tmp_path / "book.epub"
    _write_zip(p)

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

    monkeypatch.setattr(text_extraction.pypdf, "PdfReader", Reader)
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

    monkeypatch.setattr(text_extraction.pypdf, "PdfReader", Reader)
    pdf = tmp_path / "flat.pdf"
    pdf.write_bytes(b"%PDF-fake")
    assert extract_sections(str(pdf)) == [("flat", "All of it.")]

    txt = tmp_path / "notes.txt"
    txt.write_text("Plain words.", encoding="utf-8")
    assert extract_sections(str(txt)) == [("notes", "Plain words.")]


# --- strip_markup (grill TE11) ---


def test_strip_markup_drops_a_leading_tag_and_its_space():
    assert text_extraction.strip_markup("[Alice:Radio]: Hello there.") == "Hello there."


def test_strip_markup_handles_a_name_with_spaces_and_a_mid_line_tag():
    text = "[Old Man:Big Hall]: Come in. [Alice]: Thanks."
    assert text_extraction.strip_markup(text) == "Come in. Thanks."


def test_strip_markup_keeps_words_apart_around_a_glued_pause_marker():
    assert text_extraction.strip_markup("Hello.[pause:1.5]Next.") == "Hello. Next."
    assert text_extraction.strip_markup("Hello. [pause:1.5] Next.") == "Hello.  Next."


def test_strip_markup_leaves_other_brackets_alone():
    text = "He said [sic] it twice."
    assert text_extraction.strip_markup(text) == text


def test_strip_markup_origin_maps_each_character_back():
    text = "[Bob]: Hi.[pause:1]Go."
    stripped, origin = text_extraction.strip_markup(text, with_origin=True)
    assert stripped == "Hi. Go."
    assert len(origin) == len(stripped)
    assert origin[0][:2] == (7, 8)  # the "H" after the tag
    assert origin[3][:2] == (10, 19)  # the space that stands in for the marker
    assert text[origin[4][0]] == "G"


def test_lexicon_spoken_chains_markup_and_lexicon_spans():
    from kokoro_gui.engine.lexicon import original_span, spoken

    text = "[Alice:Radio]: Mr Nguyen arrived."
    out, spans = spoken(text, {"Nguyen": "Win"}, with_spans=True)
    assert out == "Mr Win arrived."
    assert spoken(text, {"Nguyen": "Win"}) == out
    win = out.index("Win")
    assert text[slice(*original_span(spans, win, win + 3))] == "Nguyen"
    arrived = out.index("arrived")
    assert text[slice(*original_span(spans, arrived, arrived + 7))] == "arrived"


# --- tag options: `[Name, overlap:0.3]:` ---

def test_find_character_fx_spans_reads_tag_options():
    span = find_character_fx_spans("[Sam, overlap:0.3]: right, exactly")[0]
    assert (span.speaker_name, span.fx_name, span.options) == ("Sam", None, {"overlap": "0.3"})
    span = find_character_fx_spans("[Sam:Radio, overlap:0.3]: right")[0]
    assert (span.speaker_name, span.fx_name, span.options) == ("Sam", "Radio", {"overlap": "0.3"})


def test_a_tag_without_options_has_an_empty_options_dict():
    assert find_character_fx_spans("[Sam]: hi")[0].options == {}
    assert find_character_fx_spans("[Sam:Radio]: hi")[0].options == {}


def test_tag_options_accept_spacing_case_and_several_keys():
    span = find_character_fx_spans("[Sam ,Overlap : 0.5 , mood:calm]: hi")[0]
    assert span.speaker_name == "Sam"
    assert span.options == {"overlap": "0.5", "mood": "calm"}


def test_a_comma_in_a_name_or_fx_is_not_an_option():
    span = find_character_fx_spans("[Smith, John]: hi")[0]
    assert (span.speaker_name, span.fx_name, span.options) == ("Smith, John", None, {})
    span = find_character_fx_spans("[Smith, John:Radio]: hi")[0]
    assert (span.speaker_name, span.fx_name, span.options) == ("Smith, John", "Radio", {})
    span = find_character_fx_spans("[Sam:Old, tinny]: hi")[0]
    assert (span.speaker_name, span.fx_name, span.options) == ("Sam", "Old, tinny", {})


def test_parse_multispeaker_text_strips_tag_options_from_the_speaker(engine):
    result = engine.parse_multispeaker_text("[Sam, overlap:0.3]: right, exactly")
    assert result == [("Sam", None, "right, exactly")]
    result = engine.parse_multispeaker_text("[Sam:Radio, overlap:0.3]: right")
    assert result == [("Sam", "Radio", "right")]


def test_strip_markup_removes_a_tag_with_options():
    from kokoro_gui.engine.text_extraction import strip_markup

    assert strip_markup("[Sam, overlap:0.3]: right, exactly") == "right, exactly"


def test_tag_overrides_turns_overlap_into_the_clip_override_and_warns_about_the_rest():
    from kokoro_gui.engine.text_extraction import tag_overrides

    assert tag_overrides({"overlap": "0.3"}) == ({"overlap_s": 0.3}, [])
    assert tag_overrides({"overlap": "9"}) == ({"overlap_s": 5.0}, [])
    assert tag_overrides({"overlap": "-2"}) == ({"overlap_s": 0.0}, [])
    overrides, problems = tag_overrides({"overlap": "soon", "mood": "calm"})
    assert overrides == {}
    assert len(problems) == 2 and "overlap" in problems[0] and "mood" in problems[1]
    assert tag_overrides({"overlap": "nan"})[0] == {}
    assert tag_overrides({}) == ({}, [])
