"""Import Text reads the file on a worker thread: the window stays live, Cancel
stops the read, and a refused book is reported with its reason."""
import threading

from PySide6.QtWidgets import QMessageBox

from kokoro_gui.engine import text_extraction


def _slow_extractor(monkeypatch, text="late words"):
    """Patches the extractor to block on an event; returns `(started, release)`."""
    started, release = threading.Event(), threading.Event()

    def _extract(path, should_stop=None):
        started.set()
        while not release.wait(0.01):
            if should_stop is not None and should_stop():
                raise text_extraction.ExtractionCancelled("Import cancelled.")
        return text

    monkeypatch.setattr(text_extraction, "extract_text_from_file", _extract)
    return started, release


def test_import_text_returns_before_the_read_finishes_then_lands_the_text(qt_app, tmp_path, monkeypatch):
    started, release = _slow_extractor(monkeypatch)
    src = tmp_path / "book.txt"
    src.write_text("x", encoding="utf-8")

    qt_app.import_text(str(src), target="add")

    assert started.wait(5)
    assert qt_app.is_busy()
    assert qt_app.document.text == ""
    release.set()
    qt_app.wait_for_text_import()
    assert not qt_app.is_busy()
    assert qt_app.document.text == "late words"


def test_a_second_import_is_refused_while_one_reads(qt_app, tmp_path, monkeypatch):
    started, release = _slow_extractor(monkeypatch)
    src = tmp_path / "book.txt"
    src.write_text("x", encoding="utf-8")
    qt_app.import_text(str(src), target="add")
    assert started.wait(5)
    try:
        assert qt_app._read_book("again", lambda stop: "never", lambda r: None) is False
    finally:
        release.set()
        qt_app.wait_for_text_import()
    assert qt_app.document.text == "late words"


def test_cancel_stops_the_read_and_imports_nothing(qt_app, tmp_path, monkeypatch):
    started, _release = _slow_extractor(monkeypatch)
    src = tmp_path / "book.txt"
    src.write_text("x", encoding="utf-8")
    qt_app.import_text(str(src), target="add")
    assert started.wait(5)

    qt_app.cancel_conversion()
    qt_app.wait_for_text_import()

    assert not qt_app.is_busy()
    assert qt_app.document.text == ""


def test_a_refused_book_is_reported_with_its_reason(qt_app, tmp_path, monkeypatch):
    def _refuse(path, should_stop=None):
        raise text_extraction.ExtractionLimitError("The PDF has 6,000 pages, over the 5,000 page limit.")

    monkeypatch.setattr(text_extraction, "extract_text_from_file", _refuse)
    shown = []
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda parent, title, text: shown.append(text)))
    src = tmp_path / "book.pdf"
    src.write_bytes(b"%PDF-fake")

    qt_app.import_text(str(src), target="add")
    qt_app.wait_for_text_import()

    assert len(shown) == 1 and "6,000 pages" in shown[0]
    assert not qt_app.is_busy()
    assert qt_app.document.text == ""


def test_new_from_ebook_reads_off_the_gui_thread(qt_app, tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()

    def _sections(path, should_stop=None):
        started.set()
        release.wait(5)
        return [("One", "1."), ("Two", "2.")]

    monkeypatch.setattr(text_extraction, "extract_sections", _sections)
    book = tmp_path / "novel.epub"
    book.write_bytes(b"fake")

    assert qt_app.new_from_ebook(str(book)) == []

    assert started.wait(5)
    assert qt_app.is_busy()
    assert qt_app.children == {}
    release.set()
    qt_app.wait_for_text_import()
    assert len(qt_app.children) == 2
