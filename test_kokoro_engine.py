import unittest
from unittest.mock import MagicMock
import sys
import tempfile
import os
import asyncio

# Mock necessary modules
sys.modules['torch'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['soundfile'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.signal'] = MagicMock()
sys.modules['pedalboard'] = MagicMock()
sys.modules['pedalboard.io'] = MagicMock()
sys.modules['pypdf'] = MagicMock()
sys.modules['ebooklib'] = MagicMock()
sys.modules['ebooklib.epub'] = MagicMock()
sys.modules['bs4'] = MagicMock()
sys.modules['winsound'] = MagicMock()

# Mock KPipeline
sys.modules['kokoro'] = MagicMock()
KPipelineMock = MagicMock()
sys.modules['kokoro'].KPipeline = KPipelineMock

from kokoro_engine import KokoroEngine

class TestKokoroEnginePerformanceFix(unittest.TestCase):
    def setUp(self):
        self.engine = KokoroEngine()

    def tearDown(self):
        self.engine.worker.stop()

    @unittest.mock.patch('os.path.exists', return_value=True)
    def test_extract_text_from_file_pdf(self, mock_exists):
        # Mock pypdf behavior
        mock_reader = MagicMock()
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 Text"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 Text"
        mock_reader.pages = [mock_page1, mock_page2]

        sys.modules['pypdf'].PdfReader.return_value = mock_reader

        # Test pdf extraction
        result = self.engine.extract_text_from_file("dummy.pdf")

        self.assertEqual(result, "Page 1 Text\n\nPage 2 Text\n\n")

    @unittest.mock.patch('os.path.exists', return_value=True)
    def test_extract_text_from_file_epub(self, mock_exists):
        # Mock ebooklib and bs4 behavior
        mock_book = MagicMock()
        mock_item1 = MagicMock()
        mock_item1.get_type.return_value = sys.modules['ebooklib'].ITEM_DOCUMENT
        mock_item1.get_content.return_value = "html content 1"
        mock_item2 = MagicMock()
        mock_item2.get_type.return_value = sys.modules['ebooklib'].ITEM_DOCUMENT
        mock_item2.get_content.return_value = "html content 2"
        mock_book.get_items.return_value = [mock_item1, mock_item2]
        sys.modules['ebooklib'].epub.read_epub.return_value = mock_book

        mock_soup = MagicMock()
        # Ensure successive calls return the right text
        mock_soup.get_text.side_effect = ["Chapter 1 Text", "Chapter 2 Text"]
        sys.modules['bs4'].BeautifulSoup.return_value = mock_soup

        result = self.engine.extract_text_from_file("dummy.epub")
        self.assertEqual(result, "Chapter 1 Text\n\nChapter 2 Text\n\n")

    def test_extract_text_from_file_txt(self):
        # Use temp file for pure text read
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Line 1\nLine 2")
            temp_path = f.name

        try:
            result = self.engine.extract_text_from_file(temp_path)
            self.assertEqual(result, "Line 1\nLine 2")
        finally:
            os.remove(temp_path)

if __name__ == "__main__":
    unittest.main()
